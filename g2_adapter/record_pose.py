#!/usr/bin/env python3
"""Read and optionally save the current G2 pose; never sends a command.

The recorded ``right_pr1`` and ``right_pr2`` poses are consumed by the right
AprilTag pre-move and post-grasp transport sequence.  Each record also stores
the measured end-link position and orientation when GDK exposes them for
diagnostics and later calibration.
"""

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
    from .workflow_home import WORKFLOW_HOME_POSE_NAME
except ImportError:
    from safety import (
        ARM_JOINT_NAMES,
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        reorder_arm_positions,
        validate_robot_status,
    )
    from workflow_home import WORKFLOW_HOME_POSE_NAME

POSE_NAMES = (
    "left_p1",
    "left_p2",
    "left_p3",
    "left_p4",
    "left_handoff_1",
    "left_handoff_2",
    "right_post_grasp",
    "right_pr1",
    "right_pr2",
    # Staged bilateral flow presets, named "<stage><arm><index>" and recorded
    # for one arm only.
    "LR1",
    "LL1",
    "LL2",
    "RR1",
    "RR2",
)

# Presets whose arm cannot be inferred from a left_/right_ prefix.  The staged
# flow records one arm per pose, so the resolved side is also what
# ``--side`` must agree with.
POSE_SIDES = {
    "LR1": "right",
    "RR1": "right",
    "RR2": "right",
    "LL1": "left",
    "LL2": "left",
}

DEFAULT_OUTPUT = Path(__file__).resolve().parent / "recorded_poses.json"


def pose_side(name: str) -> str:
    """Resolve the arm a preset name belongs to."""
    if name == WORKFLOW_HOME_POSE_NAME:
        return "both"
    if name in POSE_SIDES:
        return POSE_SIDES[name]
    return "left" if name.startswith("left_") else "right"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="只读当前 G2 姿态；不会调用任何运动或夹爪控制 API。"
    )
    parser.add_argument("--name", required=True, choices=(*POSE_NAMES, WORKFLOW_HOME_POSE_NAME))
    parser.add_argument(
        "--side",
        choices=("left", "right"),
        default=None,
        help="单臂姿态所属手臂；默认按姿态名推断，显式指定时必须与名字一致。",
    )
    parser.add_argument(
        "--both-arms",
        action="store_true",
        help=f"记录左右两臂 14 个关节；必须与 --name {WORKFLOW_HOME_POSE_NAME} 一起使用。",
    )
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


def _gripper_names(end_state: dict, side: str) -> list[str]:
    group = end_state.get(f"{side}_end_state")
    if not isinstance(group, dict):
        return []
    return [str(value) for value in group.get("names", ())]


def _end_link_pose(motion, side: str) -> tuple[list[float], list[float]] | None:
    """Read the GDK end-link pose so a preset can be orientation-only later."""
    link = "arm_l_end_link" if side == "left" else "arm_r_end_link"
    names = [str(value) for value in getattr(motion, "frame_names", ())]
    poses = getattr(motion, "frame_poses", ())
    try:
        pose = poses[names.index(link)]
    except (ValueError, IndexError, TypeError):
        return None
    position = [float(pose.position.x), float(pose.position.y), float(pose.position.z)]
    orientation = [float(pose.orientation.x), float(pose.orientation.y), float(pose.orientation.z), float(pose.orientation.w)]
    if not all(math.isfinite(value) for value in (*position, *orientation)):
        return None
    return position, orientation


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
    if (args.name == WORKFLOW_HOME_POSE_NAME) != bool(args.both_arms):
        print(
            f"--both-arms 只能与 --name {WORKFLOW_HOME_POSE_NAME} 一起使用。",
            file=sys.stderr,
        )
        return 2
    expected_side = pose_side(args.name)
    if args.side is not None and args.side != expected_side:
        print(
            f"--side {args.side} 与姿态名 {args.name} 对应的手臂 {expected_side} 不一致。",
            file=sys.stderr,
        )
        return 2
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
        is_home = args.name == WORKFLOW_HOME_POSE_NAME
        side = expected_side
        selected_names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
        offset = 0 if side == "left" else 7
        end_link_pose = None if is_home else _end_link_pose(motion, side)
        record = {
            "name": args.name,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "side": side,
            "joint_names": list(ARM_JOINT_NAMES if is_home else selected_names),
            "joint_positions_rad": list(arms if is_home else arms[offset : offset + 7]),
            "both_arms_rad": dict(zip(ARM_JOINT_NAMES, arms)),
            "all_gdk_motor_positions_rad": all_joints,
            "left_omnipicker_names": _gripper_names(end_state, "left"),
            "left_omnipicker_rad": _gripper(end_state, "left"),
            "right_omnipicker_names": _gripper_names(end_state, "right"),
            "right_omnipicker_rad": _gripper(end_state, "right"),
            "motion_mode": int(motion.mode),
            "motion_error_code": int(motion.error_code),
        }
        if end_link_pose is not None:
            record["end_link_position_m"] = end_link_pose[0]
            record["end_link_orientation_xyzw"] = end_link_pose[1]
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
