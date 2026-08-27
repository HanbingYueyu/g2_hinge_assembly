#!/usr/bin/env python3
"""Read or execute one recorded G2 preset directly through GDK."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from .control import G2Controller
    from .record_pose import DEFAULT_OUTPUT, POSE_NAMES
except ImportError:
    from control import G2Controller
    from record_pose import DEFAULT_OUTPUT, POSE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2 GDK preset runner")
    parser.add_argument("--preset", required=True, choices=POSE_NAMES)
    parser.add_argument("--poses", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--velocity-rad-s", type=float, default=0.08)
    parser.add_argument(
        "--allow-motion",
        action="store_true",
        help="唯一的运动确认参数；不指定时只打印目标和当前状态。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = json.loads(args.poses.expanduser().read_text(encoding="utf-8"))
    try:
        preset = document["poses"][args.preset]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"姿态文件中没有 {args.preset}") from exc
    try:
        import agibot_gdk
    except ImportError as exc:
        raise SystemExit("agibot_gdk 不可用；请通过 agi 的 uv 环境运行") from exc
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise SystemExit("GDK 初始化失败")
    try:
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(
            agibot_gdk, robot, allow_motion=args.allow_motion
        )
        current = controller.checked_arm_state()
        print(json.dumps({
            "preset": args.preset,
            "side": preset["side"],
            "target_rad": preset["joint_positions_rad"],
            "current_both_arms_rad": current,
            "allow_motion": args.allow_motion,
        }, indent=2, ensure_ascii=False))
        if not args.allow_motion:
            print("只读预检完成；未发送运动命令。")
            return 0
        controller.move_preset(
            preset["side"],
            preset["joint_positions_rad"],
            velocity_rad_s=args.velocity_rad_s,
            detail=f"traditional {args.preset}",
        )
        print("GDK 已接受规划式关节请求；请现场观察状态和机器人反馈。")
        return 0
    except Exception as exc:
        print(f"预设请求失败：{exc}", file=sys.stderr)
        return 1
    finally:
        try:
            agibot_gdk.gdk_release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
