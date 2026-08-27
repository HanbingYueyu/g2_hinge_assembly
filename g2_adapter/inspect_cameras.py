#!/usr/bin/env python3
"""Read-only inspection of the GDK camera API and one frame per G2 camera."""

from __future__ import annotations

import json
import sys


def _public_attributes(value) -> list[str]:
    return sorted(name for name in dir(value) if not name.startswith("_"))


def main() -> int:
    try:
        import agibot_gdk
    except ImportError as exc:
        print("agibot_gdk 不可用；请先 source GDK env.sh。", file=sys.stderr)
        return 2
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败。", file=sys.stderr)
        return 2
    try:
        camera = agibot_gdk.Camera()
        types = {
            "head_color": agibot_gdk.CameraType.kHeadColor,
            "head_depth": agibot_gdk.CameraType.kHeadDepth,
            "left_color": agibot_gdk.CameraType.kHandLeftColor,
            "left_depth": agibot_gdk.CameraType.kHandLeftDepth,
            "right_color": agibot_gdk.CameraType.kHandRightColor,
            "right_depth": agibot_gdk.CameraType.kHandRightDepth,
        }
        report = {
            "camera_api": _public_attributes(camera),
            "camera_type_api": _public_attributes(agibot_gdk.CameraType),
            "streams": {},
        }
        for name, camera_type in types.items():
            try:
                image = camera.get_latest_image(camera_type, 1000.0)
                report["streams"][name] = {
                    "available": True,
                    "shape_api": list(camera.get_image_shape(camera_type)),
                    "fps": float(camera.get_image_fps(camera_type)),
                    "image_attributes": _public_attributes(image),
                    "width": int(image.width),
                    "height": int(image.height),
                    "timestamp_ns": int(image.timestamp_ns),
                    "encoding": str(image.encoding),
                    "color_format": str(image.color_format),
                    "bit_depth": int(image.bit_depth),
                }
            except Exception as exc:
                report["streams"][name] = {"available": False, "error": str(exc)}
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("只读完成：未保存图像，未发送任何机器人控制命令。", file=sys.stderr)
        return 0
    finally:
        try:
            agibot_gdk.gdk_release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
