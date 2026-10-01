#!/usr/bin/env python3
"""Read-only two-hole 3D localization from the head RGB-D camera and GDK TF."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .build_head_rgbd_two_hole_dataset import CLASS_NAMES
from .build_head_rgbd_two_hole_v2 import ROI
from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
from .two_hole_rgbd import base_from_head_at, localize_box


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/two_hole_head_rgbd_yolo_v2/best.pt"))
    parser.add_argument("--count", type=int, default=1, help="Successful samples; 0 runs until Ctrl-C")
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--max-rgb-depth-delta-ms", type=float, default=50.0)
    parser.add_argument("--max-tf-age-ms", type=float, default=100.0)
    parser.add_argument("--sample-timeout", type=float, default=30.0)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/two_hole_head_rgbd"))
    args = parser.parse_args()
    if args.count < 0 or args.interval < 0 or not 0 < args.confidence < 1:
        parser.error("Invalid count, interval, or confidence")
    if min(args.max_rgb_depth_delta_ms, args.max_tf_age_ms, args.sample_timeout) <= 0:
        parser.error("Time limits must be positive")
    if not args.model.is_file():
        parser.error(f"Model does not exist: {args.model}")
    return args


def select_two_boxes(result, *, crop_origin: tuple[int, int]) -> dict[int, dict]:
    candidates: dict[int, list[dict]] = {0: [], 1: []}
    for box in result.boxes:
        class_id = int(box.cls.item())
        if class_id not in candidates:
            continue
        x0, y0, x1, y1 = (float(value) for value in box.xyxy[0].tolist())
        candidates[class_id].append({
            "class_id": class_id,
            "confidence": float(box.conf.item()),
            "box_full_xyxy_px": [x0 + crop_origin[0], y0 + crop_origin[1],
                                 x1 + crop_origin[0], y1 + crop_origin[1]],
            "box_crop_xyxy_px": [x0, y0, x1, y1],
        })
    missing = [CLASS_NAMES[class_id] for class_id, items in candidates.items() if not items]
    if missing:
        raise ValueError(f"Detector missed: {', '.join(missing)}")
    return {class_id: max(items, key=lambda item: item["confidence"])
            for class_id, items in candidates.items()}


def main() -> int:
    args = parse_args()
    import agibot_gdk as gdk
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    if [model.names.get(i) for i in range(2)] != list(CLASS_NAMES):
        raise RuntimeError(f"Unexpected model classes: {model.names}")
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK initialization failed")
    camera = None
    try:
        color_type, depth_type = gdk.CameraType.kHeadColor, gdk.CameraType.kHeadDepth
        camera = gdk.Camera([color_type, depth_type])
        tf = gdk.TF()
        color_from_depth = transform_to_dict(tf.get_tf_from_sensor(
            gdk.SensorExtrinsicType.kHeadDepthToHeadColor))
        head_from_color = transform_to_dict(tf.get_tf_from_sensor(
            gdk.SensorExtrinsicType.kHeadRGBDToHeadLink3))
        session = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        session.mkdir(parents=True, exist_ok=False)
        previous_stamp = None
        color_intrinsic = depth_intrinsic = None
        successful = 0
        deadline = time.monotonic() + args.sample_timeout
        while args.count == 0 or successful < args.count:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"No valid two-hole RGB-D sample in {args.sample_timeout:g} seconds")
            color = camera.get_latest_image(color_type, 1000.0)
            color_stamp = int(color.timestamp_ns)
            if color_stamp == previous_stamp:
                time.sleep(0.01)
                continue
            previous_stamp = color_stamp
            depth = camera.get_nearest_image(depth_type, color_stamp, 1000.0)
            depth_stamp = int(depth.timestamp_ns)
            delta_ms = abs(color_stamp - depth_stamp) / 1e6
            if delta_ms > args.max_rgb_depth_delta_ms:
                print(f"Skip unsynchronized RGB-D pair: {delta_ms:.1f} ms", flush=True)
                continue
            rgb = decode_color_rgb(color, gdk)
            depth_u16 = decode_depth_u16(depth, gdk)
            if rgb.shape[:2] != (400, 640) or depth_u16.shape != (400, 640):
                raise RuntimeError(f"Unexpected head RGB-D dimensions: {rgb.shape}, {depth_u16.shape}")
            if color_intrinsic is None:
                color_intrinsic = intrinsic_to_dict(
                    camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height))
                depth_intrinsic = intrinsic_to_dict(
                    camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height))
                (session / "calibration.json").write_text(json.dumps({
                    "camera": "GDK kHeadColor + kHeadDepth",
                    "color_intrinsic": color_intrinsic,
                    "depth_intrinsic": depth_intrinsic,
                    "color_from_depth": color_from_depth,
                    "head_from_color": head_from_color,
                    "depth_scale_m_per_unit": 0.001,
                    "fixed_color_crop_xyxy": list(ROI),
                }, indent=2) + "\n", encoding="utf-8")
            color_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            x0, y0, x1, y1 = ROI
            crop_bgr = color_bgr[y0:y1, x0:x1].copy()
            result = model.predict(crop_bgr, imgsz=960, conf=args.confidence,
                                   iou=0.5, device=0, verbose=False)[0]
            try:
                boxes = select_two_boxes(result, crop_origin=(x0, y0))
                aligned = align_depth_to_color(
                    depth_u16, depth_intrinsic=depth_intrinsic,
                    color_intrinsic=color_intrinsic, depth_to_color=color_from_depth,
                    depth_scale_m=0.001,
                )
                base_from_head, tf_status = base_from_head_at(
                    tf, color_stamp, max_fallback_age_ms=args.max_tf_age_ms)
                targets = {}
                for class_id, detection in boxes.items():
                    target = localize_box(
                        detection["box_full_xyxy_px"], aligned, color_intrinsic,
                        head_from_color, base_from_head,
                    )
                    target["class_id"] = class_id
                    target["confidence"] = detection["confidence"]
                    targets[CLASS_NAMES[class_id]] = target
            except (ValueError, RuntimeError) as error:
                print(f"Skip incomplete localization: {error}", flush=True)
                continue
            table = targets[CLASS_NAMES[0]]
            tip = targets[CLASS_NAMES[1]]
            difference = {
                "point_head_color_optical_m": (np.asarray(tip["point_head_color_optical_m"])
                                               - np.asarray(table["point_head_color_optical_m"])).tolist(),
                "point_head_link3_m": (np.asarray(tip["point_head_link3_m"])
                                       - np.asarray(table["point_head_link3_m"])).tolist(),
                "point_base_link_m": (np.asarray(tip["point_base_link_m"])
                                       - np.asarray(table["point_base_link_m"])).tolist(),
            }
            payload = {
                "camera": "GDK kHeadColor + kHeadDepth RGB-D",
                "color_timestamp_ns": color_stamp,
                "depth_timestamp_ns": depth_stamp,
                "rgb_depth_delta_ms": delta_ms,
                "tf": tf_status,
                "base_from_head_link3": base_from_head,
                "targets": targets,
                "hinge_tip_minus_table_hole_m": difference,
            }
            stem = f"sample_{successful:04d}"
            cv2.imwrite(str(session / f"{stem}_color.png"), color_bgr)
            cv2.imwrite(str(session / f"{stem}_depth_u16.png"), depth_u16)
            cv2.imwrite(str(session / f"{stem}_aligned_depth_u16.png"), aligned)
            overlay = crop_bgr.copy()
            for class_id, detection in boxes.items():
                bx0, by0, bx1, by1 = map(round, detection["box_crop_xyxy_px"])
                color = (0, 255, 0) if class_id == 0 else (255, 180, 0)
                cv2.rectangle(overlay, (bx0, by0), (bx1, by1), color, 1)
                cv2.putText(overlay, f"{class_id} {detection['confidence']:.2f}",
                            (bx0, by0-3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            cv2.imwrite(str(session / f"{stem}_crop_overlay.png"), overlay)
            (session / f"{stem}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            print(json.dumps({"sample": successful,
                              "table_hole_base_link_m": table["point_base_link_m"],
                              "hinge_tip_base_link_m": tip["point_base_link_m"],
                              "tip_minus_table_base_link_m": difference["point_base_link_m"],
                              "rgb_depth_delta_ms": delta_ms, "tf": tf_status},
                             ensure_ascii=False), flush=True)
            successful += 1
            deadline = time.monotonic() + args.sample_timeout
            if args.count == 0 or successful < args.count:
                time.sleep(args.interval)
        print(f"Saved {successful} localized samples to {session}")
        return 0
    finally:
        if camera is not None:
            camera.close_camera()
        gdk.gdk_release()


if __name__ == "__main__":
    raise SystemExit(main())
