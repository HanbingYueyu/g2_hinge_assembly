#!/usr/bin/env python3
"""Create reviewed pixel masks for the right-D405 slotted-shaft dataset.

Workflow: draw a tight box around the slotted shaft, let SAM3.1 propose an
initial mask, then repair it with brushes before saving.  The output contains
RGB, aligned depth, a binary PNG mask, and a YOLO-seg polygon label.  The SAM
proposal is never saved without the operator explicitly pressing ``s``.
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from PIL import Image

# The SAM3.1 checkout is vendored in this project rather than installed as a
# site package.  Make the launcher independent of any inherited PYTHONPATH.
SAM3_SOURCE = Path(__file__).resolve().parent / "sam3"
if str(SAM3_SOURCE) not in sys.path:
    sys.path.insert(0, str(SAM3_SOURCE))
from sam3.model_builder import build_sam3_predictor


WINDOW_NAME = "Right D405 Slotted-Shaft Segmentation Capture"
DEFAULT_TARGET_NAME = "slotted_hinge_shaft"


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


def box_from_roi(roi, width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, box_width, box_height = (int(value) for value in roi)
    if box_width < 6 or box_height < 6:
        return None
    x0, y0 = max(0, min(width - 1, x)), max(0, min(height - 1, y))
    x1, y1 = min(width, x0 + box_width), min(height, y0 + box_height)
    return (x0, y0, x1, y1) if x1 - x0 >= 6 and y1 - y0 >= 6 else None


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def sam_box_mask(predictor, bgr: np.ndarray, box: tuple[int, int, int, int], threshold: float) -> np.ndarray:
    """Return a conservative SAM proposal, clipped to the hand-drawn box."""
    height, width = bgr.shape[:2]
    x0, y0, x1, y1 = box
    normalized_box = [[x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]]
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    state = predictor.model.init_state(resource_path=[image], offload_video_to_cpu=True, async_loading_frames=False)
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, output = predictor.model.add_prompt(
                inference_state=state,
                frame_idx=0,
                text_str="visual",
                boxes_xywh=normalized_box,
                box_labels=[1],
                output_prob_thresh=threshold,
            )
        raw_masks = to_numpy(output.get("out_binary_masks"))
        if raw_masks is None:
            return np.zeros((height, width), dtype=bool)
        if raw_masks.ndim == 4:
            raw_masks = raw_masks[:, 0]
        if raw_masks.ndim == 2:
            raw_masks = raw_masks[None]
        guard = np.zeros((height, width), dtype=bool)
        guard[y0:y1, x0:x1] = True
        candidates = [mask.astype(bool) & guard for mask in raw_masks]
        return max(candidates, key=lambda mask: int(mask.sum()), default=np.zeros_like(guard))
    finally:
        state.clear()


def next_index(image_dir: Path) -> int:
    numbers = []
    for path in image_dir.glob("frame_*.jpg"):
        try:
            numbers.append(int(path.stem.rsplit("_", 1)[1]))
        except ValueError:
            pass
    return max(numbers, default=0) + 1


def sample_counts(image_dir: Path, label_dir: Path) -> tuple[int, int]:
    positive = negative = 0
    for image_path in image_dir.glob("frame_*.jpg"):
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        if label_path.read_text(encoding="utf-8").strip():
            positive += 1
        else:
            negative += 1
    return positive, negative


def mask_to_yolo_segmentation(mask: np.ndarray) -> str:
    height, width = mask.shape
    found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 8.0:
            continue
        simplified = cv2.approxPolyDP(contour, epsilon=0.8, closed=True).reshape(-1, 2)
        if len(simplified) < 3:
            continue
        points = " ".join(f"{x / width:.8f} {y / height:.8f}" for x, y in simplified)
        lines.append(f"0 {points}")
    return "\n".join(lines) + ("\n" if lines else "")


def write_intrinsics(path: Path, serial: str, depth_scale: float, intrinsics, target_name: str) -> None:
    if path.exists():
        return
    path.write_text(json.dumps({
        "camera_serial": serial,
        "depth_scale_m_per_unit": depth_scale,
        "depth_is_aligned_to": "color",
        "class_names": [target_name],
        "color": {"width": intrinsics.width, "height": intrinsics.height, "fx": intrinsics.fx, "fy": intrinsics.fy, "ppx": intrinsics.ppx, "ppy": intrinsics.ppy},
    }, indent=2), encoding="utf-8")


class MaskEditor:
    def __init__(self, brush_radius: int) -> None:
        self.mask: np.ndarray | None = None
        self.brush_radius = brush_radius

    def set_mask(self, mask: np.ndarray | None) -> None:
        self.mask = None if mask is None else mask.astype(bool).copy()

    def _paint(self, x: int, y: int, value: bool) -> None:
        if self.mask is None:
            return
        brush = np.zeros(self.mask.shape, dtype=np.uint8)
        cv2.circle(brush, (x, y), self.brush_radius, 255, -1, cv2.LINE_AA)
        if value:
            self.mask |= brush.astype(bool)
        else:
            self.mask &= ~brush.astype(bool)

    def mouse(self, event, x, y, flags, _userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            self._paint(x, y, True)
        if event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON):
            self._paint(x, y, False)


def draw(color: np.ndarray, editor: MaskEditor, prompt_box, positive_count: int, negative_count: int, status: str, target_name: str) -> np.ndarray:
    view = color.copy()
    if editor.mask is not None:
        overlay = np.zeros_like(view)
        overlay[editor.mask] = (0, 255, 0)
        view = cv2.addWeighted(view, 1.0, overlay, 0.38, 0.0)
    if prompt_box is not None:
        x0, y0, x1, y1 = prompt_box
        cv2.rectangle(view, (x0, y0), (x1, y1), (255, 0, 255), 2, cv2.LINE_AA)
    mask_area = 0 if editor.mask is None else int(editor.mask.sum())
    lines = [
        f"Right D405 {target_name} segmentation capture — review every mask",
        f"positive={positive_count}  negative={negative_count}  mask_pixels={mask_area}",
        "b: tight target box + SAM proposal   left: add mask   right: erase mask",
        "n: mark negative/no target   [ / ]: brush size   c: clear   s: save   q/esc: quit",
        status,
    ]
    for row, text in enumerate(lines):
        y = 25 + 22 * row
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (70, 255, 70), 1, cv2.LINE_AA)
    return view


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sam3.1_multiplex.pt")
    parser.add_argument("--serial", default="260322275349")
    parser.add_argument("--out-dir", default="datasets/right_d405_slotted_shaft_seg_raw")
    parser.add_argument("--target-name", default=DEFAULT_TARGET_NAME, help="human-readable name stored with this isolated dataset")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--auto-exposure-limit", type=int, default=4000)
    parser.add_argument("--sam-threshold", type=float, default=0.35)
    parser.add_argument("--brush-radius", type=int, default=5)
    parser.add_argument("--min-mask-pixels", type=int, default=30)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3.1 checkpoint not found: {checkpoint}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    image_dir, depth_dir, mask_dir, label_dir = out_dir / "images_all", out_dir / "depth_u16", out_dir / "masks_all", out_dir / "labels_all"
    for path in (image_dir, depth_dir, mask_dir, label_dir):
        path.mkdir(parents=True, exist_ok=True)
    index = next_index(image_dir)
    positive_count, negative_count = sample_counts(image_dir, label_dir)
    manifest_path = out_dir / "manifest.jsonl"

    print(f"loading SAM3.1 mask assistant: {checkpoint}", flush=True)
    muted = io.StringIO()
    with contextlib.redirect_stdout(muted):
        predictor = build_sam3_predictor(version="sam3.1", checkpoint_path=str(checkpoint), use_fa3=False, use_rope_real=False, async_loading_frames=False, default_output_prob_thresh=args.sam_threshold)
    print("SAM3.1 loaded. The proposed green mask must be reviewed before saving.", flush=True)

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
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        write_intrinsics(out_dir / "intrinsics.json", args.serial, depth_scale, intrinsics, args.target_name)
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        editor = MaskEditor(args.brush_radius)
        prompt_box = None
        pending_negative = False
        status = "Press b, draw only the target, then inspect the green mask."
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(WINDOW_NAME, editor.mouse)
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color, depth = np.asanyarray(color_frame.get_data()).copy(), np.asanyarray(depth_frame.get_data()).copy()
            cv2.imshow(WINDOW_NAME, draw(color, editor, prompt_box, positive_count, negative_count, status, args.target_name))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                editor.set_mask(None)
                prompt_box, pending_negative = None, False
                status = "Cleared."
            elif key == ord("["):
                editor.brush_radius = max(1, editor.brush_radius - 1)
                status = f"Brush radius={editor.brush_radius}px"
            elif key == ord("]"):
                editor.brush_radius = min(50, editor.brush_radius + 1)
                status = f"Brush radius={editor.brush_radius}px"
            elif key == ord("n"):
                editor.set_mask(None)
                prompt_box, pending_negative = None, True
                status = "Negative selected; press s to save this no-shaft/distractor frame."
            elif key == ord("b"):
                roi = cv2.selectROI(WINDOW_NAME, color, showCrosshair=True, fromCenter=False)
                prompt_box = box_from_roi(roi, color.shape[1], color.shape[0])
                pending_negative = False
                if prompt_box is None:
                    status = "Box cancelled/too small."
                else:
                    start = time.monotonic()
                    editor.set_mask(sam_box_mask(predictor, color, prompt_box, args.sam_threshold))
                    status = f"SAM proposal ready in {(time.monotonic() - start) * 1000:.0f} ms; left add/right erase, then s saves."
            elif key == ord("s"):
                if pending_negative:
                    saved_mask = np.zeros(depth.shape, dtype=bool)
                elif editor.mask is not None and int(editor.mask.sum()) >= args.min_mask_pixels:
                    saved_mask = editor.mask.copy()
                else:
                    status = "Refusing empty positive mask. Draw b then repair it, or press n for a negative."
                    continue
                stem = f"frame_{index:06d}"
                color_path, depth_path = image_dir / f"{stem}.jpg", depth_dir / f"{stem}.png"
                mask_path, label_path = mask_dir / f"{stem}.png", label_dir / f"{stem}.txt"
                if not cv2.imwrite(str(color_path), color, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                    raise RuntimeError(f"failed to write {color_path}")
                if not cv2.imwrite(str(depth_path), depth) or not cv2.imwrite(str(mask_path), saved_mask.astype(np.uint8) * 255):
                    raise RuntimeError(f"failed to write RGB-D/mask for {stem}")
                label_path.write_text("" if pending_negative else mask_to_yolo_segmentation(saved_mask), encoding="utf-8")
                annotation = "negative" if pending_negative else "positive"
                record = {"index": index, "timestamp_unix": time.time(), "camera_serial": args.serial, "annotation": annotation, "prompt_box_xyxy": prompt_box, "mask_pixels": int(saved_mask.sum()), "color_path": str(color_path), "depth_u16_path": str(depth_path), "mask_path": str(mask_path), "yolo_seg_label_path": str(label_path)}
                with manifest_path.open("a", encoding="utf-8") as manifest:
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                if pending_negative:
                    negative_count += 1
                else:
                    positive_count += 1
                status = f"Saved {stem} ({annotation}). Reposition target before the next sample."
                print(status, flush=True)
                index += 1
                editor.set_mask(None)
                prompt_box, pending_negative = None, False
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        del predictor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
