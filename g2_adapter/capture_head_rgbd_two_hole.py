#!/usr/bin/env python3
"""Read-only paired capture from the head RGB-D color and depth streams."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from .camera import decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--interval", type=float, default=3.0)
    parser.add_argument("--max-rgb-depth-delta-ms", type=float, default=50.0)
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/two_hole_head_rgbd_raw"))
    args = parser.parse_args()
    if args.count <= 0 or args.interval < 0 or args.max_rgb_depth_delta_ms <= 0:
        parser.error("--count and max RGB/depth delta must be positive; interval must be nonnegative")

    import agibot_gdk as gdk

    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK initialization failed")
    camera = None
    try:
        color_type, depth_type = gdk.CameraType.kHeadColor, gdk.CameraType.kHeadDepth
        camera = gdk.Camera([color_type, depth_type])
        session = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        (session / "color").mkdir(parents=True, exist_ok=False)
        (session / "depth_u16").mkdir(parents=True, exist_ok=False)
        tf = gdk.TF()
        metadata = {
            "camera": "GDK kHeadColor + kHeadDepth",
            "depth_scale_m_per_unit": 0.001,
            "color_from_depth": transform_to_dict(tf.get_tf_from_sensor(
                gdk.SensorExtrinsicType.kHeadDepthToHeadColor)),
            "head_from_color": transform_to_dict(tf.get_tf_from_sensor(
                gdk.SensorExtrinsicType.kHeadRGBDToHeadLink3)),
        }
        previous_timestamp = None
        index = 0
        deadline = time.monotonic() + 10.0
        while index < args.count:
            if time.monotonic() >= deadline:
                raise TimeoutError("No synchronized head RGB-D pair for 10 seconds")
            while True:
                color = camera.get_latest_image(color_type, 1000.0)
                color_stamp = int(color.timestamp_ns)
                if color_stamp != previous_timestamp:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("No new head color frame for 10 seconds")
            depth = camera.get_nearest_image(depth_type, color_stamp, 1000.0)
            depth_stamp = int(depth.timestamp_ns)
            delta_ms = abs(color_stamp - depth_stamp) / 1e6
            previous_timestamp = color_stamp
            if delta_ms > args.max_rgb_depth_delta_ms:
                print(f"Skip unsynchronized RGB-D pair: {delta_ms:.1f} ms", flush=True)
                continue
            rgb = decode_color_rgb(color, gdk)
            depth_u16 = decode_depth_u16(depth, gdk)
            if index == 0:
                metadata["color_intrinsic"] = intrinsic_to_dict(
                    camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height))
                metadata["depth_intrinsic"] = intrinsic_to_dict(
                    camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height))
                (session / "calibration.json").write_text(json.dumps(metadata, indent=2) + "\n")
            stem = f"frame_{index:04d}.png"
            if not cv2.imwrite(str(session / "color" / stem), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Cannot save head color frame {stem}")
            if not cv2.imwrite(str(session / "depth_u16" / stem), depth_u16):
                raise RuntimeError(f"Cannot save head depth frame {stem}")
            with (session / "frames.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"file": stem, "color_timestamp_ns": color_stamp,
                                         "depth_timestamp_ns": depth_stamp,
                                         "delta_ms": abs(color_stamp - depth_stamp) / 1e6}) + "\n")
            index += 1
            deadline = time.monotonic() + 10.0
            print(f"{index}/{args.count}: {stem}, delta={delta_ms:.1f} ms", flush=True)
            if index < args.count:
                time.sleep(args.interval)
        return 0
    finally:
        if camera is not None:
            camera.close_camera()
        gdk.gdk_release()


if __name__ == "__main__":
    raise SystemExit(main())
