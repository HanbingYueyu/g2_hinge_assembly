"""Small Cartesian motions used to seat and release the left-hand hinge."""

from __future__ import annotations

import math
import time

import numpy as np

try:
    from .control import CartesianPoseTarget
    from .left_grasp_gdk import measured_left_end_link_pose, read_wrench_force
    from .move_left_above_head_hinge_gdk import wait_actual_left_target
    from .safety import SafetyError
except ImportError:
    from control import CartesianPoseTarget
    from left_grasp_gdk import measured_left_end_link_pose, read_wrench_force
    from move_left_above_head_hinge_gdk import wait_actual_left_target
    from safety import SafetyError


LEFT_FORCE_FRAME = "arm_l_end_link"


def read_left_force(robot) -> np.ndarray:
    _status, force = read_wrench_force(robot, LEFT_FORCE_FRAME)
    return np.asarray(force, dtype=np.float64)


def capture_left_force_baseline(robot, *, samples: int = 20, sample_period_s: float = 0.02,
                                sleep=time.sleep) -> np.ndarray:
    readings = []
    for index in range(samples):
        readings.append(read_left_force(robot))
        if index + 1 < samples:
            sleep(sample_period_s)
    return np.median(np.asarray(readings, dtype=np.float64), axis=0)


def force_guarded_left_descent(controller, robot, *, max_distance_mm: float,
                               force_threshold_n: float, speed_mm_s: float,
                               completion_tolerance_mm: float = 0.5,
                               timeout_s: float | None = None,
                               sleep=time.sleep, clock=time.monotonic) -> dict:
    """Descend in base_link -Z until distance or relative resultant force wins."""
    if not math.isfinite(max_distance_mm) or not 0.0 < max_distance_mm <= 100.0:
        raise ValueError("max_distance_mm must be in (0, 100]")
    if not math.isfinite(force_threshold_n) or not 0.0 < force_threshold_n <= 50.0:
        raise ValueError("force_threshold_n must be in (0, 50]")
    if not math.isfinite(speed_mm_s) or not 0.0 < speed_mm_s <= 50.0:
        raise ValueError("speed_mm_s must be in (0, 50]")
    if not math.isfinite(completion_tolerance_mm) or not 0.0 < completion_tolerance_mm <= 10.0:
        raise ValueError("completion_tolerance_mm must be in (0, 10]")
    if timeout_s is not None and (not math.isfinite(timeout_s) or not 1.0 <= timeout_s <= 120.0):
        raise ValueError("timeout_s must be in [1, 120]")

    origin = measured_left_end_link_pose(controller)
    baseline = capture_left_force_baseline(robot, sleep=sleep)
    target = CartesianPoseTarget(
        (origin.position_m[0], origin.position_m[1], origin.position_m[2] - max_distance_mm / 1000.0),
        origin.orientation_xyzw,
    )
    stream = controller.start_left_end_effector_position_stream(
        origin, rate_hz=50.0, life_time_s=0.10,
        max_translation_velocity_mps=speed_mm_s / 1000.0,
    )
    allowed_s = max_distance_mm / speed_mm_s + 3.0 if timeout_s is None else float(timeout_s)
    deadline = clock() + allowed_s
    actual = origin
    stop_reason = None
    delta_force_n = 0.0
    try:
        stream.update_target(target)
        while stop_reason is None:
            stream.raise_if_failed()
            actual = measured_left_end_link_pose(controller)
            descent_mm = max(0.0, (origin.position_m[2] - actual.position_m[2]) * 1000.0)
            delta_force_n = float(np.linalg.norm(read_left_force(robot) - baseline))
            if delta_force_n >= force_threshold_n:
                stop_reason = "force_threshold"
            elif descent_mm >= max_distance_mm - completion_tolerance_mm:
                stop_reason = "max_distance"
            elif clock() >= deadline:
                raise SafetyError(
                    f"左臂力保护下探在 {allowed_s:.1f} s 内未到原位或力阈值："
                    f"目标 {max_distance_mm:.1f} mm，实测 {descent_mm:.1f} mm，"
                    f"力差 {delta_force_n:.3f} N，允许剩余 {completion_tolerance_mm:.1f} mm；"
                    "已停止并保持夹持"
                )
            else:
                sleep(0.02)
        hold = CartesianPoseTarget(actual.position_m, origin.orientation_xyzw)
        stream.update_target(hold)
        hold_timeout_s = max(0.5, max_distance_mm / speed_mm_s + 0.5)
        if not stream.wait_until_commanded_target(hold, hold_timeout_s):
            raise SafetyError("左臂力保护下探停止后，保持位姿未确认发送")
    finally:
        stream.stop()
    return {
        "event": "left_seat_descent_complete",
        "stop_reason": stop_reason,
        "descent_mm": round(max(0.0, (origin.position_m[2] - actual.position_m[2]) * 1000.0), 3),
        "delta_force_n": round(delta_force_n, 3),
        "force_threshold_n": float(force_threshold_n),
        "max_distance_mm": float(max_distance_mm),
        "completion_tolerance_mm": float(completion_tolerance_mm),
        "timeout_s": float(allowed_s),
    }


def move_left_vertical(controller, *, distance_mm: float, speed_mm_s: float) -> dict:
    if not math.isfinite(distance_mm) or not 0.0 < distance_mm <= 200.0:
        raise ValueError("distance_mm must be in (0, 200]")
    origin = measured_left_end_link_pose(controller)
    target = CartesianPoseTarget(
        (origin.position_m[0], origin.position_m[1], origin.position_m[2] + distance_mm / 1000.0),
        origin.orientation_xyzw,
    )
    stream = controller.start_left_end_effector_position_stream(
        origin, rate_hz=50.0, life_time_s=0.10,
        max_translation_velocity_mps=speed_mm_s / 1000.0,
    )
    try:
        stream.update_target(target)
        wait_actual_left_target(
            controller, stream, target,
            timeout_s=distance_mm / speed_mm_s + 5.0,
            tolerance_m=0.005, orientation_tolerance_deg=2.0,
        )
    finally:
        stream.stop()
    return {"event": "left_release_lift_complete", "distance_mm": float(distance_mm)}
