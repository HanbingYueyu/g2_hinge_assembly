#!/usr/bin/env python3
"""Right-hand PR1/PR2 transport around the AprilTag grasp.

PR1 and PR2 are operator-recorded right-arm joint presets.  The held gripper
is not commanded by this module; it remains closed after the successful grasp
while the arm first makes a measured base-link +Z lift and then uses GDK's
planned joint request to reach PR2.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

try:
    from .control import CartesianPoseTarget, G2Controller
    from .record_pose import DEFAULT_OUTPUT
    from .safety import ARM_POSITION_LIMITS
except ImportError:
    from control import CartesianPoseTarget, G2Controller
    from record_pose import DEFAULT_OUTPUT
    from safety import ARM_POSITION_LIMITS


PR1_POSE_NAME = "right_pr1"
PR2_POSE_NAME = "right_pr2"
RIGHT_ARM_OFFSET = 7
POST_GRASP_LIFT_M = 0.10
POST_GRASP_LIFT_SPEED_MPS = 0.01
POST_GRASP_LIFT_TOLERANCE_M = 0.01
POST_GRASP_LIFT_ORIENTATION_TOLERANCE_DEG = 0.5
POST_GRASP_LIFT_STABLE_SAMPLES = 3
PRESET_ARRIVAL_TOLERANCE_RAD = 0.03
PRESET_VELOCITY_RAD_S = 0.08
PRESET_TIMEOUT_S = 25.0
PRESET_JOINT_LIMIT_MARGIN = 1e-4
# Keep the old names as source-compatible aliases for callers that imported
# the PR1-specific constants before PR2 was added.
PR1_ARRIVAL_TOLERANCE_RAD = PRESET_ARRIVAL_TOLERANCE_RAD
PR1_VELOCITY_RAD_S = PRESET_VELOCITY_RAD_S
PR1_TIMEOUT_S = PRESET_TIMEOUT_S
PR1_JOINT_LIMIT_MARGIN = PRESET_JOINT_LIMIT_MARGIN


def _validate_right_preset_target(values: list[float], preset_name: str) -> list[float]:
    """Reject malformed or out-of-range right-preset joints before motion."""
    if len(values) != 7 or not all(math.isfinite(float(value)) for value in values):
        raise RuntimeError(f"{preset_name} 必须包含 7 个有限右臂关节值")
    limits = ARM_POSITION_LIMITS[RIGHT_ARM_OFFSET : RIGHT_ARM_OFFSET + 7]
    violations = [
        f"joint{index + 1}={value:.6f} not in [{lower + PRESET_JOINT_LIMIT_MARGIN:.6f}, {upper - PRESET_JOINT_LIMIT_MARGIN:.6f}]"
        for index, (value, (lower, upper)) in enumerate(zip(values, limits))
        if value < lower + PRESET_JOINT_LIMIT_MARGIN or value > upper - PRESET_JOINT_LIMIT_MARGIN
    ]
    if violations:
        raise RuntimeError(f"{preset_name} 超出右臂关节范围：" + "; ".join(violations))
    return values


def _validate_right_pr1_target(values: list[float]) -> list[float]:
    """Backward-compatible PR1 validator alias."""
    return _validate_right_preset_target(values, PR1_POSE_NAME)


def load_right_pose(path: Path = DEFAULT_OUTPUT, pose_name: str = PR1_POSE_NAME) -> list[float]:
    """Load and validate one named seven-joint right-arm preset.

    Any pose whose recorded ``side`` is ``right`` is accepted, so the staged
    bilateral flow can use RR1/RR2 without touching this module.
    """
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    poses = document.get("poses")
    item = poses.get(pose_name) if isinstance(poses, dict) else None
    if not isinstance(item, dict) or item.get("side") != "right":
        raise RuntimeError(
            f"姿态文件缺少右手 {pose_name}；请先记录："
            f"record_pose.py --name {pose_name} --save"
        )
    values = [float(value) for value in item.get("joint_positions_rad", ())]
    return _validate_right_preset_target(values, pose_name)


def load_pr1_pose(path: Path = DEFAULT_OUTPUT) -> list[float]:
    """Load the seven right-arm joints recorded for the pre-grasp PR1."""
    return load_right_pose(path, PR1_POSE_NAME)


def load_pr2_pose(path: Path = DEFAULT_OUTPUT) -> list[float]:
    """Load the seven right-arm joints recorded for the post-grasp PR2."""
    return load_right_pose(path, PR2_POSE_NAME)


def lift_target_from_pose(before: CartesianPoseTarget, lift_m: float = POST_GRASP_LIFT_M) -> CartesianPoseTarget:
    """Build a base-link +Z lift target while preserving the wrist orientation."""
    if not math.isfinite(lift_m) or not 0.005 <= lift_m <= 0.30:
        raise ValueError("post-grasp lift must be in [0.005, 0.30] m")
    return CartesianPoseTarget(
        (before.position_m[0], before.position_m[1], before.position_m[2] + lift_m),
        before.orientation_xyzw,
    )


def pose_snapshot(controller: G2Controller) -> dict:
    """Read and return the right end-link pose plus right-arm joints."""
    pose = controller.read_end_effector_pose("arm_r_end_link")
    joints = controller.checked_arm_state()[RIGHT_ARM_OFFSET : RIGHT_ARM_OFFSET + 7]
    return {
        "frame": "arm_r_end_link",
        "position_m": [float(value) for value in pose.position_m],
        "orientation_xyzw": [float(value) for value in pose.orientation_xyzw],
        "joint_positions_rad": [float(value) for value in joints],
    }


def pose_from_snapshot(snapshot: dict) -> CartesianPoseTarget:
    """Convert a saved snapshot back to the Cartesian pose used for lifting."""
    return CartesianPoseTarget(
        tuple(float(value) for value in snapshot["position_m"]),
        tuple(float(value) for value in snapshot["orientation_xyzw"]),
    )


def lift_after_grasp(
    controller: G2Controller,
    before: CartesianPoseTarget,
    *,
    speed_mps: float | None = None,
    control_mode: int = 3,
) -> dict:
    """Lift the held right end-link 100 mm in base +Z and verify feedback."""
    target = lift_target_from_pose(before)
    lift_speed_mps = POST_GRASP_LIFT_SPEED_MPS if speed_mps is None else float(speed_mps)
    if not 0.001 <= lift_speed_mps <= 0.03:
        raise ValueError("post-grasp lift speed must be in [0.001, 0.03] m/s")
    stream = controller.start_right_end_effector_position_stream(
        before,
        rate_hz=50.0,
        life_time_s=0.10,
        max_translation_velocity_mps=lift_speed_mps,
        control_mode=control_mode,
    )
    try:
        stream.update_target(target)
        timeout_s = POST_GRASP_LIFT_M / lift_speed_mps + 1.0
        actual = stream.wait_until_actual_target(
            target,
            timeout_s,
            position_tolerance_m=0.005,
            orientation_tolerance_deg=POST_GRASP_LIFT_ORIENTATION_TOLERANCE_DEG,
            stable_samples=POST_GRASP_LIFT_STABLE_SAMPLES,
        )
        if actual is None:
            raise RuntimeError("抓取后右手上抬 100 mm 未在限定时间内实际到位；停止前往 PR2")
        after = controller.read_end_effector_pose("arm_r_end_link")
        delta = tuple(
            float(actual_value - before_value)
            for actual_value, before_value in zip(after.position_m, before.position_m)
        )
        lateral_m = math.hypot(delta[0], delta[1])
        if (
            abs(delta[2] - POST_GRASP_LIFT_M) > POST_GRASP_LIFT_TOLERANCE_M
            or lateral_m > POST_GRASP_LIFT_TOLERANCE_M
        ):
            raise RuntimeError(
                "抓取后右手上抬实测位移不满足要求；"
                f"actual_delta_m={list(delta)}, expected_z_m={POST_GRASP_LIFT_M:.3f}, "
                f"tolerance_m={POST_GRASP_LIFT_TOLERANCE_M:.3f}；停止前往 PR2"
            )
        return {
            "event": "right_post_grasp_lift_arrived",
            "before_end_link_base_link": list(before.position_m),
            "target_end_link_base_link": list(target.position_m),
            "actual_end_link_base_link": list(after.position_m),
            "actual_delta_base_m": list(delta),
            "lift_m": POST_GRASP_LIFT_M,
            "speed_mps": lift_speed_mps,
            "tolerance_m": POST_GRASP_LIFT_TOLERANCE_M,
            "gripper": "preserved_closed; no gripper request is issued",
        }
    finally:
        stream.stop()


def move_right_after_current_lift(controller, target, *, preset_name, velocity_rad_s, emit=None):
    """Reach current TCP +100 mm in base Z, then the named preset joints."""
    try:
        from .right_hole_joint_move import move_right_hole_joint_target
    except ImportError:
        from right_hole_joint_move import move_right_hole_joint_target
    current = controller.read_end_effector_pose('arm_r_end_link')
    above = CartesianPoseTarget(
        (current.position_m[0], current.position_m[1], current.position_m[2] + .10),
        current.orientation_xyzw,
    )
    if emit is not None:
        emit({'event': 'right_preset_clearance_start', 'control_method': 'endpoint_ik_joint_plan',
              'control_mode': 3, 'velocity_rad_s': velocity_rad_s, 'lift_mm': 100.0,
              'start_position_m': list(current.position_m), 'target_position_m': list(above.position_m)})
    actual = move_right_hole_joint_target(
        controller, above, velocity_rad_s=velocity_rad_s, control_mode=3,
    )
    if emit is not None:
        emit({'event': 'right_preset_clearance_arrived', 'control_mode': 3,
              'actual_position_m': list(actual.position_m), 'target_position_m': list(above.position_m)})
    return move_right_preset(
        controller, target, preset_name=preset_name, velocity_rad_s=velocity_rad_s,
        control_mode=3,
    )


def move_right_preset(
    controller: G2Controller,
    target: list[float],
    *,
    preset_name: str,
    velocity_rad_s: float = PRESET_VELOCITY_RAD_S,
    arrival_tolerance_rad: float = PRESET_ARRIVAL_TOLERANCE_RAD,
    timeout_s: float = PRESET_TIMEOUT_S,
    control_mode: int = 3,
) -> dict:
    """Move the right arm to a named preset with feedback confirmation."""
    if not math.isfinite(float(velocity_rad_s)) or not float(velocity_rad_s) > 0.0:
        raise ValueError(f"{preset_name} velocity must be a positive finite value in rad/s")
    if not 0.001 <= arrival_tolerance_rad <= 0.10:
        raise ValueError(f"{preset_name} arrival tolerance must be in [0.001, 0.10] rad")
    if not 1.0 <= timeout_s <= 30.0:
        raise ValueError(f"{preset_name} timeout must be in [1, 30] s")
    _validate_right_preset_target(target, preset_name)
    controller.move_preset(
        "right",
        target,
        velocity_rad_s=velocity_rad_s,
        life_time_s=timeout_s,
        detail=f"traditional right preset {preset_name}",
        control_mode=control_mode,
    )
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        measured = controller.checked_arm_state()[RIGHT_ARM_OFFSET : RIGHT_ARM_OFFSET + 7]
        max_error = max(abs(actual - expected) for actual, expected in zip(measured, target))
        if max_error <= arrival_tolerance_rad:
            return {
                "event": f"{preset_name}_arrived",
                "preset": preset_name,
                "target_joint_positions_rad": [float(value) for value in target],
                "actual_joint_positions_rad": [float(value) for value in measured],
                "max_joint_error_rad": float(max_error),
                "arrival_tolerance_rad": float(arrival_tolerance_rad),
                "gripper": "unchanged; no gripper request is issued",
            }
        time.sleep(0.10)
    raise RuntimeError(f"右手 {preset_name} 在 {timeout_s:.1f} s 内未到位；停止后续动作")


def move_right_pr1(
    controller: G2Controller,
    target: list[float],
    *,
    velocity_rad_s: float = PR1_VELOCITY_RAD_S,
    arrival_tolerance_rad: float = PR1_ARRIVAL_TOLERANCE_RAD,
    timeout_s: float = PR1_TIMEOUT_S,
) -> dict:
    """Move the right arm to the pre-grasp PR1 preset."""
    return move_right_preset(
        controller,
        target,
        preset_name=PR1_POSE_NAME,
        velocity_rad_s=velocity_rad_s,
        arrival_tolerance_rad=arrival_tolerance_rad,
        timeout_s=timeout_s,
    )


def move_right_pr2(
    controller: G2Controller,
    target: list[float],
    *,
    preset_name: str = PR2_POSE_NAME,
    velocity_rad_s: float = PRESET_VELOCITY_RAD_S,
    arrival_tolerance_rad: float = PRESET_ARRIVAL_TOLERANCE_RAD,
    timeout_s: float = PRESET_TIMEOUT_S,
    control_mode: int = 3,
) -> dict:
    """Move the right arm to the post-grasp transport preset."""
    result = move_right_preset(
        controller,
        target,
        preset_name=preset_name,
        velocity_rad_s=velocity_rad_s,
        arrival_tolerance_rad=arrival_tolerance_rad,
        timeout_s=timeout_s,
        control_mode=control_mode,
    )
    result["gripper"] = "preserved_closed; no gripper request is issued"
    return result


def execute_post_grasp_transport(
    controller: G2Controller,
    pr2_target: list[float],
    *,
    pr2_pose_name: str = PR2_POSE_NAME,
    lift_speed_mps: float | None = None,
    preset_velocity_rad_s: float = PRESET_VELOCITY_RAD_S,
    control_mode: int = 3,
) -> tuple[dict, dict, dict]:
    """Save the grasp pose, lift, then move to the transport preset in order."""
    saved = pose_snapshot(controller)
    saved.update({
        "event": "right_post_grasp_pose_saved",
        "reason": "save_before_post_grasp_lift_and_pr2",
    })
    lift_args = {} if lift_speed_mps is None else {"speed_mps": lift_speed_mps}
    if control_mode != 3:
        lift_args["control_mode"] = control_mode
    lift = lift_after_grasp(controller, pose_from_snapshot(saved), **lift_args)
    pr2_args: dict = {}
    if control_mode != 3:
        pr2_args["control_mode"] = control_mode
    if pr2_pose_name != PR2_POSE_NAME:
        pr2_args["preset_name"] = pr2_pose_name
    if preset_velocity_rad_s != PRESET_VELOCITY_RAD_S:
        pr2_args["velocity_rad_s"] = preset_velocity_rad_s
    pr2 = move_right_pr2(controller, pr2_target, **pr2_args)
    return saved, lift, pr2
