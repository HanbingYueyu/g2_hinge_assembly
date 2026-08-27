#!/usr/bin/env python3
"""Read-only G2/GDK preflight.  This program contains no motion call."""

from __future__ import annotations

import json
import sys
import time

try:
    from .safety import ARM_JOINT_NAMES, reorder_arm_positions, validate_robot_status
except ImportError:
    from safety import ARM_JOINT_NAMES, reorder_arm_positions, validate_robot_status


def main() -> int:
    try:
        import agibot_gdk
    except ImportError as exc:
        print("agibot_gdk 不可用；请在干净终端中 source GDK env.sh。", file=sys.stderr)
        return 2
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败。", file=sys.stderr)
        return 2
    try:
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        whole = dict(robot.get_whole_body_status())
        motion = robot.get_motion_control_status()
        positions = reorder_arm_positions(robot.get_joint_states())
        validate_robot_status(whole, motion)
        print("G2 只读预检通过；未发送任何运动或夹爪命令。")
        print(json.dumps(dict(zip(ARM_JOINT_NAMES, positions)), indent=2))
        print(f"motion_mode={int(motion.mode)}, error_code={int(motion.error_code)}")
        return 0
    except Exception as exc:
        print(f"G2 只读预检失败：{exc}", file=sys.stderr)
        return 1
    finally:
        try:
            agibot_gdk.gdk_release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
