"""Pure validation helpers shared by the G2 bridge and its tests.

This module deliberately has no ROS or GDK imports, so its safety rules can be
tested without connecting to a robot.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

LEFT_ARM_JOINT_NAMES = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
RIGHT_ARM_JOINT_NAMES = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES

# G2_t2v2_crsB_omnipicker.urdf installed with the current GDK 4.1.5 image.
_ONE_ARM_LIMITS = (
    (-3.0718, 3.0718),
    (-2.0595, 2.0595),
    (-3.0718, 3.0718),
    (-2.4959, 1.0123),
    (-3.0718, 3.0718),
    (-1.0123, 1.0123),
    (-1.5359, 1.5359),
)
ARM_POSITION_LIMITS = _ONE_ARM_LIMITS + _ONE_ARM_LIMITS
OMNIPICKER_MIN = -0.7853981633974483
OMNIPICKER_MAX = 0.0

STATUS_ERROR_KEYS = (
    "right_arm_error",
    "left_arm_error",
    "right_end_error",
    "left_end_error",
    "waist_error",
    "lift_error",
    "neck_error",
    "chassis_error",
)


class SafetyError(RuntimeError):
    """A command or current robot state failed a mandatory safety check."""


def finite_vector(values: Iterable[float], size: int, label: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != size:
        raise SafetyError(f"{label} must contain {size} values, got {len(result)}")
    if not all(math.isfinite(value) for value in result):
        raise SafetyError(f"{label} contains NaN or Inf")
    return result


def reorder_arm_positions(joint_states: Mapping) -> tuple[float, ...]:
    """Read motor positions by name; GDK array order is never trusted."""
    by_name: dict[str, float] = {}
    for item in joint_states.get("states", ()):
        name = str(item.get("name", ""))
        if name in by_name:
            raise SafetyError(f"duplicate GDK joint state: {name}")
        by_name[name] = float(item.get("motor_position", math.nan))
    missing = [name for name in ARM_JOINT_NAMES if name not in by_name]
    if missing:
        raise SafetyError(f"missing G2 arm joint states: {missing}")
    return finite_vector((by_name[name] for name in ARM_JOINT_NAMES), 14, "arm state")


def validate_arm_target(target: Sequence[float], margin: float = 1e-4) -> tuple[float, ...]:
    values = finite_vector(target, 14, "arm target")
    violations = []
    for name, value, (lower, upper) in zip(ARM_JOINT_NAMES, values, ARM_POSITION_LIMITS):
        if value < lower + margin or value > upper - margin:
            violations.append(f"{name}={value:.6f} not in [{lower + margin:.6f}, {upper - margin:.6f}]")
    if violations:
        raise SafetyError("joint limit violation: " + "; ".join(violations))
    return values


def validate_segment(
    start: Sequence[float],
    end: Sequence[float],
    duration_s: float,
    max_velocity_rad_s: float,
) -> None:
    before = finite_vector(start, 14, "segment start")
    after = validate_arm_target(end)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise SafetyError(f"trajectory segment duration must be positive, got {duration_s}")
    if not math.isfinite(max_velocity_rad_s) or max_velocity_rad_s <= 0.0:
        raise SafetyError("max velocity must be positive and finite")
    worst = max(abs(b - a) / duration_s for a, b in zip(before, after))
    if worst > max_velocity_rad_s + 1e-9:
        raise SafetyError(
            f"trajectory requires {worst:.4f} rad/s, software limit is {max_velocity_rad_s:.4f} rad/s"
        )


def validate_robot_status(whole_body: Mapping, motion_status: object) -> None:
    errors = {
        key: int(whole_body.get(key, -1))
        for key in STATUS_ERROR_KEYS
        if int(whole_body.get(key, -1)) != 0
    }
    if errors:
        raise SafetyError(f"whole-body errors: {errors}")
    estops = [
        key for key in ("left_arm_estop", "right_arm_estop")
        if bool(whole_body.get(key, False))
    ]
    if estops:
        raise SafetyError(f"emergency stop active: {estops}")
    error_code = int(getattr(motion_status, "error_code", -1))
    if error_code != 0:
        raise SafetyError(
            f"motion-control error {error_code}: {getattr(motion_status, 'error_msg', '')}"
        )


def clamp_omnipicker(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise SafetyError("Omnipicker target contains NaN or Inf")
    return min(OMNIPICKER_MAX, max(OMNIPICKER_MIN, value))

