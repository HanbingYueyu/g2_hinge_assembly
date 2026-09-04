#!/usr/bin/env python3
"""Read-only right-hand RGB-D / hole alignment diagnostic.

This tool never creates ``Robot`` and never sends arm or gripper commands.  It
captures synchronized right color/depth frames, applies the same GDK depth to
color projection used by the bilateral direct path, and reports whether valid
depth actually reaches the detected hole neighborhood.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from .bilateral_rgbd_geometry import estimate_hole_depth_adaptive
    from .camera import decode_color_rgb, decode_depth_u16, depth_on_color_grid, intrinsic_to_dict, transform_to_dict
    from .hole_target_detector import center_crop, infer_detection, load_detection_model, select_locked_hole, translate_detection
except ImportError:
    from bilateral_rgbd_geometry import estimate_hole_depth_adaptive
    from camera import decode_color_rgb, decode_depth_u16, depth_on_color_grid, intrinsic_to_dict, transform_to_dict
    from hole_target_detector import center_crop, infer_detection, load_detection_model, select_locked_hole, translate_detection


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hole_target_det_v2" / "best.pt"
# GDK returns intrinsics in the pixel domain used by the published depth
# frame.  Do not infer a 640x480 source mode from fx/cx alone: the current
# hand-depth packet is 1280x1056 and an invented scale produces visible global
# RGB/depth mis-registration.  An explicit CLI value remains available for a
# separately verified stream.
DEFAULT_DEPTH_CALIBRATION_SIZE = None
DEPTH_DISPLAY_MIN_M = 0.03
DEPTH_DISPLAY_MAX_M = 2.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读检查右手 hole 周围的 RGB-D 对齐和深度覆盖")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "right_rgbd_hole_diagnostic")
    parser.add_argument("--frames", type=int, default=5, help="采集帧数，默认 5")
    parser.add_argument("--warmup", type=int, default=3, help="丢弃初始彩色帧数，默认 3")
    parser.add_argument("--right-crop-percent", type=float, default=60.0)
    parser.add_argument("--right-hole-track-max-jump-px", type=float, default=120.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--depth-registration",
        choices=("reproject", "registered_same_pixel"),
        default="reproject",
        help="深度坐标模式；reproject=按官方外参与深度内参重投影（默认），registered_same_pixel=仅用于对比诊断",
    )
    parser.add_argument(
        "--depth-calibration-size",
        type=_parse_depth_calibration_size,
        default=DEFAULT_DEPTH_CALIBRATION_SIZE,
        metavar="WIDTHxHEIGHT",
        help="仅在已确认内参属于另一分辨率时指定原生标定尺寸；默认直接使用 GDK 内参",
    )
    parser.add_argument(
        "--depth-region",
        choices=("ring", "box"),
        default="ring",
        help="孔深度采样区域；box 使用检测框内全部有效像素",
    )
    return parser.parse_args()


def _parse_depth_calibration_size(value: str) -> tuple[int, int]:
    import re

    match = re.fullmatch(r"\s*(\d+)\s*[xX,; ]\s*(\d+)\s*", value or "")
    if match is None:
        raise argparse.ArgumentTypeError(
            "--depth-calibration-size 必须是 WIDTHxHEIGHT，例如 640x480"
        )
    width, height = (int(item) for item in match.groups())
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("--depth-calibration-size 必须为正整数")
    return width, height


def _depth_summary(depth_u16: np.ndarray) -> dict:
    depth = np.asarray(depth_u16)
    valid = depth[depth > 0]
    rows, cols = np.nonzero(depth > 0)
    return {
        "shape": [int(value) for value in depth.shape],
        "valid_pixels": int(valid.size),
        "valid_fraction": round(float(valid.size / max(1, depth.size)), 6),
        "min_m": None if valid.size == 0 else round(float(valid.min()) * 0.001, 4),
        "median_m": None if valid.size == 0 else round(float(np.median(valid)) * 0.001, 4),
        "max_m": None if valid.size == 0 else round(float(valid.max()) * 0.001, 4),
        "nonzero_bbox_xyxy": None if rows.size == 0 else [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())],
    }


def _local_depth_stats(depth_u16: np.ndarray, center_px: tuple[float, float] | None) -> dict:
    depth = np.asarray(depth_u16)
    if center_px is None:
        return {"center_px": None, "nearest_nonzero": None, "nearest_in_range": None, "radius_counts": {}}
    cx, cy = (float(center_px[0]), float(center_px[1]))
    yy, xx = np.mgrid[: depth.shape[0], : depth.shape[1]]
    distance_sq = (xx - cx) ** 2 + (yy - cy) ** 2

    def nearest_depth(mask: np.ndarray) -> dict | None:
        rows, cols = np.nonzero(mask)
        if not rows.size:
            return None
        distances = (cols.astype(np.float64) - cx) ** 2 + (rows.astype(np.float64) - cy) ** 2
        index = int(np.argmin(distances))
        return {
            "px": [int(cols[index]), int(rows[index])],
            "distance_px": round(float(math.sqrt(float(distances[index]))), 2),
            "depth_m": round(float(depth[rows[index], cols[index]]) * 0.001, 4),
        }

    nearest = nearest_depth(depth > 0)
    nearest_in_range = nearest_depth(
        (depth > 0)
        & (depth >= DEPTH_DISPLAY_MIN_M * 1000.0)
        & (depth <= DEPTH_DISPLAY_MAX_M * 1000.0)
    )
    radius_counts = {
        str(radius): int(np.count_nonzero((distance_sq <= float(radius * radius)) & (depth > 0)))
        for radius in (16, 32, 64, 128, 256, 400)
    }
    return {
        "center_px": [round(cx, 1), round(cy, 1)],
        "nearest_nonzero": nearest,
        "nearest_in_range": nearest_in_range,
        "radius_counts": radius_counts,
    }


def _tile_valid_counts(depth_u16: np.ndarray, tiles: int = 8) -> list[list[int]]:
    depth = np.asarray(depth_u16)
    result: list[list[int]] = []
    for row_slice in np.array_split(np.arange(depth.shape[0]), tiles):
        row_counts: list[int] = []
        for col_slice in np.array_split(np.arange(depth.shape[1]), tiles):
            if row_slice.size == 0 or col_slice.size == 0:
                row_counts.append(0)
            else:
                row_counts.append(int(np.count_nonzero(depth[np.ix_(row_slice, col_slice)] > 0)))
        result.append(row_counts)
    return result


def _overlay(frame_bgr: np.ndarray, raw_depth: np.ndarray, hole, diagnostic: dict) -> np.ndarray:
    """Show RGB and the raw depth frame directly overlaid, pixel-for-pixel.

    This image is intentionally independent of the geometry path: no
    reproject, intrinsic scaling, resize, or fixed offset is applied. The
    saved overlay answers only whether the two GDK arrays are already in the
    same pixel grid.
    """
    image = frame_bgr.copy()
    depth = np.asarray(raw_depth)
    if depth.ndim != 2 or depth.shape[:2] != image.shape[:2]:
        cv2.putText(image, "RAW DEPTH SIZE MISMATCH (NO RESIZE)", (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 255), 2, cv2.LINE_AA)
        return image
    depth_float = depth.astype(np.float32)
    valid = (
        (depth > 0)
        & (depth_float >= DEPTH_DISPLAY_MIN_M * 1000.0)
        & (depth_float <= DEPTH_DISPLAY_MAX_M * 1000.0)
    )
    if np.any(valid):
        depth_view = np.zeros_like(depth, dtype=np.uint8)
        valid_values = depth[valid].astype(np.float32)
        depth_view[valid] = np.clip(
            (valid_values - DEPTH_DISPLAY_MIN_M * 1000.0)
            * 255.0
            / ((DEPTH_DISPLAY_MAX_M - DEPTH_DISPLAY_MIN_M) * 1000.0),
            0.0,
            255.0,
        ).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_view, cv2.COLORMAP_TURBO)
        image[valid] = cv2.addWeighted(image[valid], 0.45, depth_color[valid], 0.55, 0.0)
    if hole is not None:
        x0, y0, x1, y1 = (int(round(value)) for value in hole.xyxy)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 3, cv2.LINE_AA)
        cx, cy = (int(round(value)) for value in hole.center_px)
        cv2.drawMarker(image, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
        cv2.putText(image, "LOCKED HOLE", (max(8, x0), max(24, y0 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, "RGB + RAW DEPTH (SAME PIXEL)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    nearest = diagnostic["local_depth"].get("nearest_in_range")
    if nearest is not None:
        distance = nearest["distance_px"]
        cv2.putText(image, f"nearest valid depth={distance:.1f}px", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return image


def main() -> int:
    args = parse_args()
    if args.frames < 1 or args.warmup < 0 or not (0.0 < args.right_crop_percent <= 100.0):
        print("参数错误：frames/warmup/crop 无效。", file=sys.stderr)
        return 2
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"找不到 hole 模型：{model_path}", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败。", file=sys.stderr)
        return 2
    camera = None
    try:
        color_type = agibot_gdk.CameraType.kHandRightColor
        depth_type = agibot_gdk.CameraType.kHandRightDepth
        camera = agibot_gdk.Camera([color_type, depth_type])
        model = load_detection_model(str(model_path), device=args.device)
        tf = agibot_gdk.TF()
        depth_to_color = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kRightHandDepthToRightHandColor))
        color_intrinsic = None
        depth_intrinsic = None
        previous_hole_center = None
        reports: list[dict] = []
        output_dir = args.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(args.warmup):
            camera.get_latest_image(color_type, 1000.0)
        for frame_index in range(1, args.frames + 1):
            color = camera.get_latest_image(color_type, 1000.0)
            color_timestamp = int(color.timestamp_ns)
            depth_image = camera.get_nearest_image(depth_type, color_timestamp, 1000.0)
            depth_timestamp = int(depth_image.timestamp_ns)
            color_rgb = decode_color_rgb(color, agibot_gdk)
            frame_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            depth_u16 = decode_depth_u16(depth_image, agibot_gdk)
            if color_intrinsic is None:
                color_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height))
                depth_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(depth_type), width=int(depth_image.width), height=int(depth_image.height))
            assert color_intrinsic is not None and depth_intrinsic is not None
            aligned_depth = depth_on_color_grid(
                depth_u16,
                registration_mode=args.depth_registration,
                depth_intrinsic=depth_intrinsic,
                color_intrinsic=color_intrinsic,
                depth_to_color=depth_to_color,
                depth_calibration_size=args.depth_calibration_size,
            )
            crop, offset = center_crop(frame_bgr, args.right_crop_percent)
            detections = [translate_detection(item, offset) for item in infer_detection(model, crop, confidence=args.confidence, iou=args.iou, imgsz=args.imgsz, device=args.device)]
            holes = [item for item in detections if item.class_name == "hole"]
            hole, lock_reason = select_locked_hole(holes, previous_hole_center, max_jump_px=args.right_hole_track_max_jump_px)
            if hole is not None:
                previous_hole_center = hole.center_px
            center = None if hole is None else hole.center_px
            diagnostic = {
                "frame": frame_index,
                "depth_registration": args.depth_registration,
                "depth_region": args.depth_region,
                "color_timestamp_ns": color_timestamp,
                "depth_timestamp_ns": depth_timestamp,
                "timestamp_skew_ms": round(abs(depth_timestamp - color_timestamp) / 1_000_000.0, 3),
                "color_shape": [int(color.height), int(color.width)],
                "depth_shape": [int(depth_image.height), int(depth_image.width)],
                "hole_lock": {
                    "reason": lock_reason,
                    "candidate_count": len(holes),
                    "locked": hole is not None,
                    "xyxy": None if hole is None else [round(float(value), 1) for value in hole.xyxy],
                    "center_px": None if center is None else [round(float(value), 1) for value in center],
                },
                "raw_depth": _depth_summary(depth_u16),
                "aligned_depth": _depth_summary(aligned_depth),
                "local_depth": _local_depth_stats(aligned_depth, center),
                "aligned_tile_valid_counts_8x8": _tile_valid_counts(aligned_depth),
            }
            if hole is not None:
                try:
                    _, adaptive = estimate_hole_depth_adaptive(
                        aligned_depth,
                        hole.xyxy,
                        intrinsic=color_intrinsic,
                        scales=(1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
                        max_padding_px=300.0,
                        min_valid_pixels=1,
                        min_support_fraction=0.0,
                        min_covered_sectors=1,
                        max_mad_m=0.2,
                        sampling_region=args.depth_region,
                    )
                    diagnostic["adaptive_probe"] = adaptive
                except Exception as exc:
                    diagnostic["adaptive_probe"] = {
                        "error": str(exc),
                        "diagnostics": getattr(exc, "diagnostics", None),
                    }
            prefix = output_dir / f"frame_{frame_index:03d}"
            cv2.imwrite(str(prefix.with_name(prefix.name + "_color.jpg")), frame_bgr)
            cv2.imwrite(str(prefix.with_name(prefix.name + "_depth_raw_u16.png")), depth_u16)
            cv2.imwrite(str(prefix.with_name(prefix.name + "_depth_aligned_u16.png")), aligned_depth)
            cv2.imwrite(str(prefix.with_name(prefix.name + "_overlay.jpg")), _overlay(frame_bgr, depth_u16, hole, diagnostic))
            reports.append(diagnostic)
        report = {
            "event": "right_rgbd_hole_diagnostic",
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": str(model_path),
            "depth_registration": args.depth_registration,
            "depth_calibration_size": None if args.depth_calibration_size is None else list(args.depth_calibration_size),
            "depth_region": args.depth_region,
            "right_crop_percent": args.right_crop_percent,
            "color_intrinsic": color_intrinsic,
            "depth_intrinsic": depth_intrinsic,
            "depth_to_color": depth_to_color,
            "frames": reports,
            "safety": "read-only camera/TF/detector access; no Robot, arm, or gripper command",
        }
        report_path = output_dir / "report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"output_dir": str(output_dir), "report": str(report_path), "frames": len(reports)}, ensure_ascii=False, indent=2))
        print("只读诊断完成：未发送机器人或夹爪命令。", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"右手 RGB-D 诊断失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        try:
            agibot_gdk.gdk_release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
