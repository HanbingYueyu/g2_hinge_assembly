#!/usr/bin/env python3
"""Capture a calibrated left-hand G2 RGB-D sample without any motion call."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    from .camera import align_depth_to_color, decode_depth_u16, intrinsic_to_dict, transform_to_dict
except ImportError:
    from camera import align_depth_to_color, decode_depth_u16, intrinsic_to_dict, transform_to_dict


DEPTH_SCALE_M = 0.001  # GDK RS2_FORMAT_Z16 is documented in millimetres.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读采集一帧左手 GDK RGB-D，并用 GDK 外参投影深度到彩色坐标系。"
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="当前最终检测场景名，例如 left_final_detection；不是预移动点 P2/P3。",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "left_d405_gdk",
        help="采样输出目录；默认位于项目 outputs/ 下。",
    )
    parser.add_argument("--warmup", type=int, default=5, help="丢弃的初始帧数，默认 5。")
    return parser.parse_args()


def _write_json(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _color_filename(image) -> str:
    encoding = str(image.encoding)
    if "JPEG" in encoding:
        return "color.jpg"
    if "PNG" in encoding:
        return "color.png"
    raise RuntimeError(f"left color stream is not JPEG/PNG: {encoding}")


def _depth_summary(depth_u16: np.ndarray) -> dict:
    valid = depth_u16[depth_u16 > 0]
    return {
        "valid_pixels": int(valid.size),
        "valid_fraction": float(valid.size / depth_u16.size),
        "median_m": None if valid.size == 0 else float(np.median(valid) * DEPTH_SCALE_M),
    }


def main() -> int:
    args = parse_args()
    if args.warmup < 0 or not args.scene.strip() or "/" in args.scene or "\\" in args.scene:
        print("参数错误：--warmup 不能为负数，--scene 必须是非空目录名。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_python.sh 运行。", file=sys.stderr)
        return 2
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败。", file=sys.stderr)
        return 2
    camera = None
    try:
        color_type = agibot_gdk.CameraType.kHandLeftColor
        depth_type = agibot_gdk.CameraType.kHandLeftDepth
        camera = agibot_gdk.Camera([color_type, depth_type])
        for _ in range(args.warmup):
            camera.get_latest_image(color_type, 1000.0)
        color = camera.get_latest_image(color_type, 1000.0)
        depth = camera.get_nearest_image(depth_type, int(color.timestamp_ns), 1000.0)
        depth_u16 = decode_depth_u16(depth, agibot_gdk)
        color_intrinsic = intrinsic_to_dict(
            camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height)
        )
        depth_intrinsic = intrinsic_to_dict(
            camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height)
        )
        tf = agibot_gdk.TF()
        depth_to_color = transform_to_dict(tf.get_tf_from_sensor(
            agibot_gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor
        ))
        rgbd_to_left_end_link = transform_to_dict(tf.get_tf_from_sensor(
            agibot_gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink
        ))
        depth_aligned = align_depth_to_color(
            depth_u16,
            depth_intrinsic=depth_intrinsic,
            color_intrinsic=color_intrinsic,
            depth_to_color=depth_to_color,
            depth_scale_m=DEPTH_SCALE_M,
        )
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        output = args.output_dir.expanduser().resolve() / args.scene / timestamp
        output.mkdir(parents=True, exist_ok=False)
        color_name = _color_filename(color)
        (output / color_name).write_bytes(bytes(color.data))
        np.save(output / "depth_raw_u16.npy", depth_u16)
        np.save(output / "depth_aligned_to_color_u16.npy", depth_aligned)
        payload = {
            "schema_version": 1,
            "scene": args.scene,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "camera_source": "agibot_gdk.Camera(kHandLeftColor, kHandLeftDepth)",
            "color": {
                "file": color_name,
                "timestamp_ns": int(color.timestamp_ns),
                "encoding": str(color.encoding),
                "color_format": str(color.color_format),
                "intrinsic": color_intrinsic,
            },
            "depth": {
                "raw_file": "depth_raw_u16.npy",
                "aligned_to_color_file": "depth_aligned_to_color_u16.npy",
                "timestamp_ns": int(depth.timestamp_ns),
                "timestamp_delta_ms": abs(int(depth.timestamp_ns) - int(color.timestamp_ns)) / 1_000_000.0,
                "encoding": str(depth.encoding),
                "color_format": str(depth.color_format),
                "scale_m_per_unit": DEPTH_SCALE_M,
                "intrinsic": depth_intrinsic,
                "raw_summary": _depth_summary(depth_u16),
                "aligned_summary": _depth_summary(depth_aligned),
            },
            "gdk_extrinsics": {
                "depth_to_color": depth_to_color,
                "rgbd_to_left_arm_end_link": rgbd_to_left_end_link,
            },
            "alignment": "raw depth was reprojected into the left color optical frame using GDK intrinsics and kLeftHandDepthToLeftHandColor",
            "safety": "read-only camera and TF access; no robot or gripper command was sent",
        }
        _write_json(output / "metadata.json", payload)
        print(json.dumps({
            "output": str(output),
            "scene": args.scene,
            "color_shape": [int(color.height), int(color.width)],
            "depth_shape": [int(depth.height), int(depth.width)],
            "depth_timestamp_delta_ms": payload["depth"]["timestamp_delta_ms"],
            "raw_depth": payload["depth"]["raw_summary"],
            "aligned_depth": payload["depth"]["aligned_summary"],
        }, indent=2, ensure_ascii=False))
        print("只读采集完成：已保存相机数据；未发送任何机器人或夹爪命令。", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"左手 GDK RGB-D 采集失败：{exc}", file=sys.stderr)
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
