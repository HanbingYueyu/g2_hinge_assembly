"""Move the grasped electric screwdriver's CAD centre axis above head hole."""

from __future__ import annotations

import math

import numpy as np

try:
    from .arm_orientation import fixed_forward_down_orientation
    from .control import CartesianPoseTarget, quaternion_angle_deg
    from .right_post_grasp_sequence import lift_after_grasp
except ImportError:
    from arm_orientation import fixed_forward_down_orientation
    from control import CartesianPoseTarget, quaternion_angle_deg
    from right_post_grasp_sequence import lift_after_grasp


def _rotation_from_quaternion(quaternion_xyzw) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion_xyzw)
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if not math.isfinite(norm) or norm < 1e-12:
        raise ValueError("orientation quaternion must be finite and non-zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array((
        (1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)),
        (2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)),
        (2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)),
    ), dtype=np.float64)


def right_hole_orientation(*, yaw_deg: float, pitch_deg: float, roll_deg: float):
    """Use the same yaw/pitch/roll convention as the left full workflow."""
    return fixed_forward_down_orientation(
        forward_yaw_deg=yaw_deg,
        gripper_pitch_deg=pitch_deg,
        gripper_roll_deg=roll_deg,
    )


def end_target_for_tool_axis(
    saved_grasp_end: CartesianPoseTarget,
    base_from_workpiece: np.ndarray,
    *,
    tool_axis_target_base_m,
    target_orientation_xyzw,
) -> CartesianPoseTarget:
    """Place STEP workpiece (0, 0, 0) at the requested bit-axis point.

    The STEP/tag workpiece frame is rigidly attached to the screwdriver.  Its
    central circular axis has X=Y=0; choosing Z=0 only establishes a reference
    point along that axis.  The operator's right-hole Z offset absorbs the
    remaining axial calibration.
    """
    transform = np.asarray(base_from_workpiece, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("base_from_workpiece must be a finite 4x4 transform")
    saved_rotation = _rotation_from_quaternion(saved_grasp_end.orientation_xyzw)
    axis_at_grasp = transform[:3, 3]
    end_at_grasp = np.asarray(saved_grasp_end.position_m, dtype=np.float64)
    end_from_axis = saved_rotation.T @ (axis_at_grasp - end_at_grasp)
    target_rotation = _rotation_from_quaternion(target_orientation_xyzw)
    requested_axis = np.asarray(tool_axis_target_base_m, dtype=np.float64)
    target_end = requested_axis - target_rotation @ end_from_axis
    return CartesianPoseTarget(
        tuple(float(value) for value in target_end),
        tuple(float(value) for value in target_orientation_xyzw),
    )


def lift_right_before_return(controller, *, speed_mm_s: float, distance_mm: float = 20.0) -> dict:
    """Clear the taught endpoint along base +Z, without changing orientation."""
    if not math.isfinite(distance_mm) or distance_mm <= 0.0:
        raise ValueError("distance_mm must be positive and finite")
    current = controller.read_end_effector_pose("arm_r_end_link")
    target = CartesianPoseTarget(
        (current.position_m[0], current.position_m[1], current.position_m[2] + distance_mm / 1000.0),
        current.orientation_xyzw,
    )
    speed_mps = speed_mm_s / 1000.0
    stream = controller.start_right_end_effector_position_stream(
        current, rate_hz=50.0, life_time_s=0.10,
        max_translation_velocity_mps=speed_mps,
    )
    try:
        stream.update_target(target)
        actual = stream.wait_until_actual_target(
            target, distance_mm / 1000.0 / speed_mps + 5.0,
            position_tolerance_m=0.002,
            orientation_tolerance_deg=0.5, stable_samples=3,
        )
        if actual is None:
            raise RuntimeError(f"右手归位前上抬 {distance_mm:g} mm 未到位；停止且不松爪")
    finally:
        stream.stop()
    return {"event": "right_return_lift_complete", "distance_mm": float(distance_mm)}


def move_right_tool_above_hole(
    controller,
    saved_grasp_end: CartesianPoseTarget,
    base_from_workpiece: np.ndarray,
    hole_position_base_m,
    *,
    x_mm: float,
    y_mm: float,
    z_mm: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    speed_mm_s: float,
) -> dict:
    """Lift 100 mm, then move the CAD bit axis to the configured hole offset."""
    lift = lift_after_grasp(controller, saved_grasp_end, speed_mps=speed_mm_s / 1000.0)
    current = controller.read_end_effector_pose("arm_r_end_link")
    orientation = right_hole_orientation(yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg)
    hole = np.asarray(hole_position_base_m, dtype=np.float64)
    axis_target = hole + np.asarray((x_mm, y_mm, z_mm), dtype=np.float64) / 1000.0
    target = end_target_for_tool_axis(
        saved_grasp_end, base_from_workpiece,
        tool_axis_target_base_m=axis_target,
        target_orientation_xyzw=orientation,
    )
    speed_mps = float(speed_mm_s) / 1000.0
    stream = controller.start_right_end_effector_position_stream(
        current, rate_hz=50.0, life_time_s=0.10,
        max_translation_velocity_mps=speed_mps,
        allow_orientation_change=True,
        max_angular_velocity_rad_s=math.radians(20.0),
    )
    try:
        stream.update_target(target)
        distance = float(np.linalg.norm(np.asarray(target.position_m) - np.asarray(current.position_m)))
        angle = quaternion_angle_deg(current.orientation_xyzw, target.orientation_xyzw)
        timeout_s = distance / speed_mps + angle / 20.0 + 5.0
        actual = stream.wait_until_actual_target(
            target, timeout_s, position_tolerance_m=0.005,
            orientation_tolerance_deg=0.5, stable_samples=3,
        )
        if actual is None:
            raise RuntimeError("右手批头未在限定时间内到达 hole 偏移点")
    finally:
        stream.stop()
    return {
        "event": "right_tool_above_hole_arrived",
        "tool_axis_target_base_m": [float(value) for value in axis_target],
        "end_target_base_m": list(target.position_m),
        "orientation_yaw_pitch_roll_deg": [float(yaw_deg), float(pitch_deg), float(roll_deg)],
        "lift": lift,
    }
