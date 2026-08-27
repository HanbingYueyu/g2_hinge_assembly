#!/usr/bin/env python3
"""Live right-D405 inference for the reviewed-mask shaft segmentation baseline."""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from train_right_d405_shaft_seg_torch import ModelConfig, TinyUNet


WINDOW_NAME = "Right D405 Slotted-Shaft Segmentation (Trained Baseline)"


def configure_auto_exposure(device, exposure_limit: int) -> None:
    sensor = device.first_depth_sensor()
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 1)
    if sensor.supports(rs.option.auto_exposure_limit_toggle) and sensor.supports(rs.option.auto_exposure_limit):
        sensor.set_option(rs.option.auto_exposure_limit_toggle, 1)
        limits = sensor.get_option_range(rs.option.auto_exposure_limit)
        sensor.set_option(rs.option.auto_exposure_limit, min(max(float(exposure_limit), limits.min), limits.max))


def find_device(context: rs.context, serial: str):
    for device in context.query_devices():
        if device.get_info(rs.camera_info.serial_number) == serial:
            return device
    available = [device.get_info(rs.camera_info.serial_number) for device in context.query_devices()]
    raise RuntimeError(f"RealSense {serial} not found; available={available}")


class ShaftSegmenter:
    def __init__(self, checkpoint_path: Path, requested_device: str) -> None:
        self.device = torch.device("cuda" if requested_device == "cuda" or (requested_device == "auto" and torch.cuda.is_available()) else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.config = ModelConfig(**checkpoint["model_config"])
        self.model = TinyUNet(self.config.width).to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.training_metrics = checkpoint.get("metrics", {})

    def probabilities(self, bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.config.image_width, self.config.image_height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probability = self.model(tensor).sigmoid()[0, 0].cpu().numpy()
        return cv2.resize(probability, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def select_component(probability: np.ndarray, args):
    binary = (probability >= args.threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[label])
        if not args.min_area_px <= area <= args.max_area_px:
            continue
        component = labels == label
        mean_probability = float(probability[component].mean())
        # Optional object-specific gates.  The white-hole model does not pass
        # these fields, so its established detector remains unchanged.
        if mean_probability < float(getattr(args, "min_mean_probability", 0.0)):
            continue
        short_side = min(width, height)
        if short_side < int(getattr(args, "min_short_side_px", 0)):
            continue
        aspect_ratio = max(width, height) / max(short_side, 1)
        if aspect_ratio > float(getattr(args, "max_aspect_ratio", float("inf"))):
            continue
        fill_ratio = area / max(width * height, 1)
        if fill_ratio < float(getattr(args, "min_fill_ratio", 0.0)):
            continue
        candidates.append(
            (
                mean_probability * np.sqrt(area),
                component,
                (x, y, width, height),
                area,
                mean_probability,
                aspect_ratio,
                fill_ratio,
            )
        )
    if not candidates:
        return None
    _, mask, (x, y, width, height), area, mean_probability, aspect_ratio, fill_ratio = max(
        candidates, key=lambda item: item[0]
    )
    return {
        "mask": mask,
        "bbox_xyxy": [x, y, x + width - 1, y + height - 1],
        "area_px": area,
        "mean_probability": mean_probability,
        "max_probability": float(probability.max()),
        "bbox_width_px": width,
        "bbox_height_px": height,
        "aspect_ratio": aspect_ratio,
        "fill_ratio": fill_ratio,
    }


def add_3d(component: dict, depth_u16: np.ndarray, depth_scale: float, intrinsics: dict, target_name: str) -> dict | None:
    rows, cols = np.where(component["mask"])
    depths_m = depth_u16[rows, cols].astype(np.float32) * depth_scale
    valid = depths_m > 0
    if int(valid.sum()) < 20:
        return None
    rows, cols, depths_m = rows[valid], cols[valid], depths_m[valid]
    depth_m = float(np.median(depths_m))
    center_u, center_v = float(np.mean(cols)), float(np.mean(rows))
    return {
        "label": target_name,
        "bbox_xyxy": component["bbox_xyxy"],
        "mask_area_px": component["area_px"],
        "mean_probability": component["mean_probability"],
        "max_probability": component["max_probability"],
        "center_px": [center_u, center_v],
        "center_xyz_m": [float((center_u - intrinsics["ppx"]) / intrinsics["fx"] * depth_m), float((center_v - intrinsics["ppy"]) / intrinsics["fy"] * depth_m), depth_m],
        "depth_median_m": depth_m,
        "feature_type": "trained_positive_only_segmentation_baseline",
    }


def draw(frame: np.ndarray, component, detection, args, inference_ms: float) -> np.ndarray:
    view = frame.copy()
    if component is not None:
        overlay = np.zeros_like(view)
        overlay[component["mask"]] = (0, 0, 255)
        view = cv2.addWeighted(view, 1.0, overlay, 0.38, 0.0)
    if detection is not None:
        x0, y0, x1, y1 = detection["bbox_xyxy"]
        cv2.rectangle(view, (x0, y0), (x1, y1), (0, 255, 0), 2, cv2.LINE_AA)
        u, v = detection["center_px"]
        cv2.circle(view, (round(u), round(v)), 4, (0, 255, 0), -1)
        status = f"DETECTED p={detection['mean_probability']:.2f} area={detection['mask_area_px']} z={detection['depth_median_m']:.3f}m"
    else:
        status = "NO ACCEPTED MASK"
    lines = [
        f"Right D405 trained {args.target_name} segmentation — positive-only baseline",
        f"{status}  inference={inference_ms:.0f}ms",
        f"threshold={args.threshold:.2f} accepted_area={args.min_area_px}-{args.max_area_px}px",
        "s: save debug   q/esc: quit   (red=predicted mask; green=accepted component)",
    ]
    for row, text in enumerate(lines):
        y = 25 + row * 22
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="runs/right_d405_shaft_seg_torch/positive_only_v3_low_lr/best.pt")
    parser.add_argument("--serial", default="260322275349")
    parser.add_argument("--target-name", default="slotted_hinge_shaft")
    parser.add_argument("--out-dir", default="outputs/right_d405_shaft_seg")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--auto-exposure-limit", type=int, default=4000)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--min-area-px", type=int, default=500)
    parser.add_argument("--max-area-px", type=int, default=5500)
    parser.add_argument("--min-mean-probability", type=float, default=0.70)
    parser.add_argument("--max-aspect-ratio", type=float, default=2.05)
    parser.add_argument("--min-short-side-px", type=int, default=18)
    parser.add_argument("--min-fill-ratio", type=float, default=0.45)
    parser.add_argument("--inference-hz", type=float, default=8.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"trained checkpoint not found: {checkpoint_path}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    segmenter = ShaftSegmenter(checkpoint_path, args.device)
    print(f"loaded trained segmentation checkpoint={checkpoint_path}; device={segmenter.device}; training_metrics={segmenter.training_metrics}", flush=True)

    context = rs.context()
    device = find_device(context, args.serial)
    configure_auto_exposure(device, args.auto_exposure_limit)
    pipeline, config, align = rs.pipeline(context), rs.config(), rs.align(rs.stream.color)
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    try:
        configure_auto_exposure(profile.get_device(), args.auto_exposure_limit)
        depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        intrinsics_raw = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        intrinsics = {"fx": intrinsics_raw.fx, "fy": intrinsics_raw.fy, "ppx": intrinsics_raw.ppx, "ppy": intrinsics_raw.ppy}
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        component = detection = None
        inference_ms, next_inference = 0.0, 0.0
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color, depth = np.asanyarray(color_frame.get_data()).copy(), np.asanyarray(depth_frame.get_data()).copy()
            now = time.monotonic()
            if now >= next_inference:
                start = time.monotonic()
                component = select_component(segmenter.probabilities(color), args)
                detection = None if component is None else add_3d(component, depth, depth_scale, intrinsics, args.target_name)
                inference_ms = (time.monotonic() - start) * 1000.0
                next_inference = now + 1.0 / max(args.inference_hz, 0.1)
            view = draw(color, component, detection, args, inference_ms)
            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(out_dir / f"trained_seg_{stamp}.jpg"), view)
                cv2.imwrite(str(out_dir / f"trained_seg_depth_{stamp}.png"), depth)
                print(f"saved debug frame {stamp}", flush=True)
            payload = {"timestamp_unix": time.time(), "camera_serial": args.serial, "checkpoint": str(checkpoint_path), "positive_only_warning": True, "detected": detection is not None, "detection": detection}
            (out_dir / "right_d405_slotted_shaft_seg_latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
