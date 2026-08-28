#!/usr/bin/env python3
"""Read-only inspection of the GDK camera API and one frame per G2 camera."""

from __future__ import annotations

import json
import sys
import argparse


STREAM_TYPES = {
    "head_color": "kHeadColor",
    "head_depth": "kHeadDepth",
    "left_color": "kHandLeftColor",
    "left_depth": "kHandLeftDepth",
    "right_color": "kHandRightColor",
    "right_depth": "kHandRightDepth",
}
DEFAULT_STREAMS = tuple(STREAM_TYPES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读检查指定的 G2 相机流；默认检查六路 RGB-D。"
    )
    parser.add_argument(
        "--streams",
        default=",".join(DEFAULT_STREAMS),
        help="以逗号分隔的流名，例如 head_color,head_depth。",
    )
    return parser.parse_args()


def selected_stream_names(value: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise ValueError("--streams 至少要指定一路相机")
    unknown = set(names) - STREAM_TYPES.keys()
    if unknown:
        raise ValueError(f"未知相机流：{', '.join(sorted(unknown))}")
    if len(set(names)) != len(names):
        raise ValueError("--streams 中不能重复相机流")
    return names


def _public_attributes(value) -> list[str]:
    return sorted(name for name in dir(value) if not name.startswith("_"))


def _intrinsic_report(camera, camera_type) -> dict:
    """Return numeric GDK intrinsics when that stream supports them."""
    try:
        intrinsic = camera.get_camera_intrinsic(camera_type)
        return {
            "available": True,
            "fx_fy_cx_cy": [float(value) for value in intrinsic.intrinsic],
            "distortion": [float(value) for value in intrinsic.distortion],
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def main() -> int:
    args = parse_args()
    try:
        stream_names = selected_stream_names(args.streams)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError as exc:
        print("agibot_gdk 不可用；请先 source GDK env.sh。", file=sys.stderr)
        return 2
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败。", file=sys.stderr)
        return 2
    types = {
        name: getattr(agibot_gdk.CameraType, STREAM_TYPES[name])
        for name in stream_names
    }
    camera = None
    try:
        # Avoid Camera()'s default "start all cameras" behaviour.
        camera = agibot_gdk.Camera(list(types.values()))
        report = {
            "requested_streams": list(types),
            "camera_api": _public_attributes(camera),
            "camera_type_api": _public_attributes(agibot_gdk.CameraType),
            "streams": {},
        }
        for name, camera_type in types.items():
            stream = {"intrinsic": _intrinsic_report(camera, camera_type)}
            try:
                image = camera.get_latest_image(camera_type, 1000.0)
                stream.update({
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
                })
            except Exception as exc:
                stream.update({"available": False, "error": str(exc)})
            report["streams"][name] = stream
        print(json.dumps(report, indent=2, ensure_ascii=False))
        print("只读完成：未保存图像，未发送任何机器人控制命令。", file=sys.stderr)
        return 0
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
