#!/usr/bin/env python3
"""Read or set one G2 Omnipicker position through the project safety gate.

The default invocation is read-only.  A gripper command requires the sole
explicit confirmation flag, ``--allow-motion``.  This tool never moves an arm,
switches a GDK application/control mode, or changes collision/gravity settings.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

try:
    from .control import G2Controller
except ImportError:
    from control import G2Controller


OPEN_RAD = -0.785
CLOSE_RAD = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="读取或设置 G2 Omnipicker")
    parser.add_argument("--side", choices=("left", "right"), default="left")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--target", choices=("open", "close"), default=None)
    selection.add_argument(
        "--open-percent",
        type=float,
        default=None,
        metavar="0..100",
        help="按全闭合 0%%、最大张开 100%% 设置张开程度。",
    )
    parser.add_argument("--allow-motion", action="store_true", help="唯一动作确认参数；未指定时严格只读。")
    parser.add_argument("--settle-s", type=float, default=0.8, help="以约 100 ms 周期重复发命令并等待反馈的时间。")
    return parser.parse_args()


def target_position_rad(*, target: str | None, open_percent: float | None) -> tuple[float, str, float | None]:
    """Resolve a named endpoint or an operator-friendly opening percentage."""
    if open_percent is not None:
        if not 0.0 <= open_percent <= 100.0:
            raise ValueError("--open-percent 必须在 0..100。")
        # The measured range is 0 rad (closed) through -0.785 rad (fully
        # open), so percentage increases in the physical opening direction.
        return OPEN_RAD * (open_percent / 100.0), "open_percent", open_percent
    resolved_target = target or "open"  # Preserve the old default invocation.
    return (OPEN_RAD if resolved_target == "open" else CLOSE_RAD), resolved_target, None


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.settle_s <= 5.0:
        print("参数错误：--settle-s 必须在 0..5 秒。", file=sys.stderr)
        return 2
    try:
        desired, target_label, open_percent = target_position_rad(
            target=args.target, open_percent=args.open_percent,
        )
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_python.sh 运行。", file=sys.stderr)
        return 2

    initialized = False
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(0.5)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        controller.checked_arm_state()
        before = controller.read_gripper_position(args.side)
        result = {
            "side": args.side,
            "target": target_label,
            "open_percent": open_percent,
            "requested_position_rad": desired,
            "before_position_rad": before,
            "motion_enabled": args.allow_motion,
        }
        if not args.allow_motion:
            result["command_sent"] = False
            result["safety"] = "只读预览；追加 --allow-motion 才会发送夹爪命令。"
            print(json.dumps(result, ensure_ascii=False))
            return 0

        attempts = 0
        deadline = time.monotonic() + args.settle_s
        while True:
            controller.move_gripper(args.side, desired)
            attempts += 1
            if not args.settle_s:
                break
            time.sleep(min(0.10, max(0.0, deadline - time.monotonic())))
            if time.monotonic() >= deadline:
                break
        result["command_sent"] = True
        result["command_attempts"] = attempts
        result["after_position_rad"] = controller.read_gripper_position(args.side)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"Omnipicker 操作失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
