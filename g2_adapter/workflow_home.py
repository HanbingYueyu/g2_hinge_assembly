"""Load and execute the recorded bilateral workflow home pose."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from collections.abc import Sequence

try:
    from .safety import ARM_JOINT_NAMES, SafetyError, validate_arm_target
except ImportError:
    from safety import ARM_JOINT_NAMES, SafetyError, validate_arm_target


WORKFLOW_HOME_POSE_NAME = "workflow_home"
ARM_COUNT = len(ARM_JOINT_NAMES)


def load_workflow_home(
    path: Path,
    pose_name: str = WORKFLOW_HOME_POSE_NAME,
) -> list[float]:
    """Load one recorded pose containing all 14 arm joints."""
    path = Path(path).expanduser()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RuntimeError(f"无法读取双臂归位姿态文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"双臂归位姿态文件不是有效 JSON：{path}") from exc
    poses = document.get("poses")
    item = poses.get(pose_name) if isinstance(poses, dict) else None
    if not isinstance(item, dict) or item.get("side") != "both":
        raise RuntimeError(
            f"姿态文件缺少双臂 {pose_name}；请先记录："
            f"record_pose.py --name {pose_name} --both-arms --save"
        )
    names = item.get("joint_names")
    if names is not None and tuple(str(name) for name in names) != ARM_JOINT_NAMES:
        raise RuntimeError(f"{pose_name} 的关节顺序必须与 G2 双臂关节顺序一致")
    raw = item.get("joint_positions_rad")
    if not isinstance(raw, list) or len(raw) != ARM_COUNT:
        both = item.get("both_arms_rad")
        if not isinstance(both, dict):
            raise RuntimeError(f"{pose_name} 必须包含 14 个双臂关节值")
        raw = [both.get(name) for name in ARM_JOINT_NAMES]
    try:
        values = [float(value) for value in raw]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{pose_name} 包含无效关节值") from exc
    try:
        return list(validate_arm_target(values))
    except SafetyError as exc:
        raise RuntimeError(f"{pose_name} 关节值无效或超出范围：{exc}") from exc


def move_workflow_home(
    controller,
    target: Sequence[float],
    *,
    velocity_rad_s: float,
    arrival_tolerance_rad: float,
    timeout_s: float,
    clock=time.monotonic,
    sleep=time.sleep,
) -> dict:
    """Move both arms to the recorded home pose and confirm joint feedback."""
    values = list(validate_arm_target(target))
    if not math.isfinite(float(velocity_rad_s)) or float(velocity_rad_s) <= 0.0:
        raise ValueError("双臂归位速度必须为正的有限值")
    if not 0.001 <= float(arrival_tolerance_rad) <= 0.10:
        raise ValueError("双臂归位到位容差必须在 [0.001, 0.10] rad")
    if not 1.0 <= float(timeout_s) <= 30.0:
        raise ValueError("双臂归位超时必须在 [1, 30] s")
    controller.move_bilateral_preset(
        values,
        velocity_rad_s=float(velocity_rad_s),
        life_time_s=float(timeout_s),
        detail=f"traditional workflow {WORKFLOW_HOME_POSE_NAME}",
    )
    deadline = clock() + float(timeout_s)
    while True:
        measured = list(controller.checked_arm_state())
        max_error = max(abs(actual - expected) for actual, expected in zip(measured, values))
        if max_error <= float(arrival_tolerance_rad):
            return {
                "event": "workflow_home_arrived",
                "preset": WORKFLOW_HOME_POSE_NAME,
                "target_joint_positions_rad": values,
                "actual_joint_positions_rad": measured,
                "max_joint_error_rad": float(max_error),
                "arrival_tolerance_rad": float(arrival_tolerance_rad),
            }
        if clock() >= deadline:
            raise SafetyError(
                f"双臂归位在 {float(timeout_s):.1f} s 内未到位；"
                f"max_joint_error_rad={max_error:.6f}"
            )
        sleep(0.10)
