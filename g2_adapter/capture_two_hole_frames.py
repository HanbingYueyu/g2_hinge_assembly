#!/usr/bin/env python3
"""Save lossless frames from the GDK head stereo right camera; camera only."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import cv2

from g2_adapter.camera import decode_color_rgb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/two_hole_stereo_right_raw"))
    args = parser.parse_args()
    if args.count <= 0 or args.interval < 0:
        parser.error("--count must be positive and --interval nonnegative")

    import agibot_gdk

    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK initialization failed")
    camera = None
    try:
        camera_type = agibot_gdk.CameraType.kHeadStereoRight
        camera = agibot_gdk.Camera([camera_type])
        session = args.out_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
        session.mkdir(parents=True, exist_ok=False)
        previous_stamp = None
        for index in range(args.count):
            deadline = time.monotonic() + 10
            while True:
                image = camera.get_latest_image(camera_type, 1000.0)
                stamp = int(image.timestamp_ns)
                if stamp != previous_stamp:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("No new camera frame for 10 seconds")
            rgb = decode_color_rgb(image, agibot_gdk)
            path = session / f"frame_{index:04d}.png"
            if not cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Failed to save {path}")
            with (session / "frames.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps({
                    "file": path.name,
                    "timestamp_ns": stamp,
                    "width": int(rgb.shape[1]),
                    "height": int(rgb.shape[0]),
                    "camera": "kHeadStereoRight",
                }) + "\n")
            previous_stamp = stamp
            print(f"{index + 1}/{args.count}: {path} {rgb.shape[1]}x{rgb.shape[0]}", flush=True)
            if index + 1 < args.count:
                time.sleep(args.interval)
        return 0
    finally:
        if camera is not None:
            camera.close_camera()
        agibot_gdk.gdk_release()


if __name__ == "__main__":
    raise SystemExit(main())
