#!/usr/bin/env python3
"""Record a live hand-camera GDK color stream to an MP4 file.

This utility is camera-only: it does not create a Robot and sends no motion or
gripper command.  The resulting video can be passed to
``annotate_hole_target_video.py`` for offline annotation.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

try:
    from .camera import decode_color_rgb
except ImportError:
    from camera import decode_color_rgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读录制左右手 GDK 彩色视频")
    parser.add_argument("--side", choices=("left", "right"), default="right", help="录制左手或右手彩色相机；默认 right")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "datasets" / "hole_target_videos" / "right_scene.mp4")
    parser.add_argument("--fps", type=float, default=30.0, help="输出视频帧率")
    parser.add_argument("--seconds", type=float, default=0.0, help="录制秒数；0 表示按 Ctrl-C 停止")
    parser.add_argument("--warmup", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.seconds < 0 or args.warmup < 0:
        print("参数错误：fps、seconds 或 warmup 无效。", file=sys.stderr)
        return 2
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2
    initialized = False
    camera = None
    writer = None
    frames = 0
    started = time.monotonic()
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        color_type = getattr(agibot_gdk.CameraType, f"kHand{args.side.title()}Color")
        camera = agibot_gdk.Camera([color_type])
        for _ in range(args.warmup):
            camera.get_latest_image(color_type, 1000.0)
        print(f"开始录制{args.side}手彩色视频：{output}")
        print("按 Ctrl-C 停止；录制过程不发送任何机器人或夹爪命令。")
        while args.seconds == 0.0 or time.monotonic() - started < args.seconds:
            image = camera.get_latest_image(color_type, 1000.0)
            rgb = decode_color_rgb(image, agibot_gdk)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if writer is None:
                height, width = bgr.shape[:2]
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"无法创建视频文件：{output}")
            writer.write(bgr)
            frames += 1
            if frames % max(1, int(args.fps)) == 0:
                print(f"frames={frames} elapsed_s={time.monotonic() - started:.1f}", flush=True)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"{args.side}手视频录制失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if writer is not None:
            writer.release()
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass
        if frames:
            print(f"录制完成：{output}，frames={frames}")


if __name__ == "__main__":
    raise SystemExit(main())
