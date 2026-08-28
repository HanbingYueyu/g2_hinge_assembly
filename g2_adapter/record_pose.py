#!/usr/bin/env python3
"""Read and optionally save the current G2 pose; never sends a command."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .safety import (
        ARM_JOINT_NAMES,
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        reorder_arm_positions,
        validate_robot_status,
    )
except ImportError:
    from safety import (
        ARM_JOINT_NAMES,
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        reorder_arm_positions,
        validate_robot_status,
    )

POSE_NAMES = ("left_p1", "left_p2", "left_p3", "right_post_grasp")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "recorded_poses.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读当前 G2 姿态；不会调用任何运动或夹爪控制 API。"
    )
    parser.add_argument("--name", required=True, choices=POSE_NAMES)
    parser.add_argument("--save", action="store_true", help="写入 recorded_poses.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _gripper(end_state: dict, side: str) -> float | None:
    group = end_state.get(f"{side}_end_state")
    if not isinstance(group, dict):
        return None
    states = group.get("end_states", ())
    if len(states) != 1:
        return None
    value = float(states[0].get("position", math.nan))
    return value if math.isfinite(value) else None


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    try:
        import agibot_gdk
    except ImportError as exc:
        print("agibot_gdk 不可用；请先 source GDK env.sh。", file=sys.stderr)
        return 2
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败。", file=sys.stderr)
        return 2
    try:
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        whole = dict(robot.get_whole_body_status())
        motion = robot.get_motion_control_status()
        validate_robot_status(whole, motion)
        raw_states = robot.get_joint_states()
        arms = reorder_arm_positions(raw_states)
        all_joints = {}
        for item in raw_states.get("states", ()):
            name = str(item.get("name", ""))
            value = float(item.get("motor_position", math.nan))
            if name and math.isfinite(value):
                all_joints[name] = value
        end_state = dict(robot.get_end_state())
        side = "left" if args.name.startswith("left_") else "right"
        selected_names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
        offset = 0 if side == "left" else 7
        record = {
            "name": args.name,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "side": side,
            "joint_names": list(selected_names),
            "joint_positions_rad": list(arms[offset : offset + 7]),
            "both_arms_rad": dict(zip(ARM_JOINT_NAMES, arms)),
            "all_gdk_motor_positions_rad": all_joints,
            "left_omnipicker_rad": _gripper(end_state, "left"),
            "right_omnipicker_rad": _gripper(end_state, "right"),
            "motion_mode": int(motion.mode),
            "motion_error_code": int(motion.error_code),
        }
        print(json.dumps(record, indent=2, ensure_ascii=False))
        print("只读完成：未发送任何运动或夹爪命令。", file=sys.stderr)
        if args.save:
            output = args.output.expanduser().resolve()
            document = {"schema_version": 1, "poses": {}}
            if output.exists():
                document = json.loads(output.read_text(encoding="utf-8"))
                if not isinstance(document.get("poses"), dict):
                    raise RuntimeError(f"无效姿态文件：{output}")
            document["poses"][args.name] = record
            _atomic_json(output, document)
            print(f"已保存：{output}", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"读取姿态失败：{exc}", file=sys.stderr)
        return 1
    finally:
        try:
            agibot_gdk.gdk_release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
