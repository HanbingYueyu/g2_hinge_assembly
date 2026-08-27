"""Direct, safety-gated GDK control for the G2 traditional workflow."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

try:
    from .safety import (
        ARM_JOINT_NAMES,
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        SafetyError,
        clamp_omnipicker,
        reorder_arm_positions,
        validate_arm_target,
        validate_robot_status,
    )
except ImportError:  # Direct script/module use from this directory.
    from safety import (
        ARM_JOINT_NAMES,
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        SafetyError,
        clamp_omnipicker,
        reorder_arm_positions,
        validate_arm_target,
        validate_robot_status,
    )


class MotionDisabledError(SafetyError):
    pass


class G2Controller:
    """Own all GDK motion calls; construction and state reads never move G2."""

    def __init__(
        self,
        gdk,
        robot,
        *,
        allow_motion: bool = False,
        control_period_s: float = 0.01,
        max_servo_velocity_rad_s: float = 0.15,
    ) -> None:
        if not 0.005 <= control_period_s <= 0.02:
            raise ValueError("control_period_s must be in [0.005, 0.02]")
        if not 0.0 < max_servo_velocity_rad_s <= 0.3:
            raise ValueError("max_servo_velocity_rad_s must be in (0, 0.3]")
        self.gdk = gdk
        self.robot = robot
        self.allow_motion = bool(allow_motion)
        self.control_period_s = float(control_period_s)
        self.max_servo_velocity_rad_s = float(max_servo_velocity_rad_s)

    def checked_arm_state(self) -> tuple[float, ...]:
        whole = dict(self.robot.get_whole_body_status())
        motion = self.robot.get_motion_control_status()
        validate_robot_status(whole, motion)
        return reorder_arm_positions(self.robot.get_joint_states())

    def _require_motion(self) -> None:
        if not self.allow_motion:
            raise MotionDisabledError("allow_motion is false; no GDK command was sent")

    def move_preset(
        self,
        side: str,
        positions_rad: Sequence[float],
        *,
        velocity_rad_s: float = 0.08,
        life_time_s: float = 10.0,
        detail: str = "traditional preset",
    ) -> None:
        """Submit one planning-mode request so GDK performs its collision checks."""
        self._require_motion()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        if not math.isfinite(velocity_rad_s) or not 0.0 < velocity_rad_s <= 0.15:
            raise SafetyError("preset velocity must be in (0, 0.15] rad/s")
        if not math.isfinite(life_time_s) or not 0.1 <= life_time_s <= 30.0:
            raise SafetyError("life_time_s must be in [0.1, 30]")
        current = list(self.checked_arm_state())
        target = tuple(float(value) for value in positions_rad)
        if len(target) != 7 or not all(math.isfinite(value) for value in target):
            raise SafetyError("preset must contain seven finite joint positions")
        names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
        offset = 0 if side == "left" else 7
        current[offset : offset + 7] = target
        validate_arm_target(current)
        request = self.gdk.JointControlReq()
        request.detail = detail
        request.life_time = float(life_time_s)
        request.joint_names = list(names)
        request.joint_positions = list(target)
        request.joint_velocities = [float(velocity_rad_s)] * 7
        self.robot.joint_control_request(request)

    def servo_to(
        self,
        target_rad: Sequence[float],
        *,
        stop_requested: Callable[[], bool] | None = None,
    ) -> int:
        """Interpolate from measured state and stream absolute 14-D targets."""
        self._require_motion()
        target = validate_arm_target(target_rad)
        start = self.checked_arm_state()
        max_delta = max(abs(end - begin) for begin, end in zip(start, target))
        duration = max_delta / self.max_servo_velocity_rad_s
        steps = max(1, math.ceil(duration / self.control_period_s))
        deadline = time.monotonic()
        last_status_check = deadline
        sent = 0
        for step in range(1, steps + 1):
            if stop_requested is not None and stop_requested():
                return sent
            fraction = step / steps
            command = [
                begin + fraction * (end - begin)
                for begin, end in zip(start, target)
            ]
            now = time.monotonic()
            if now - last_status_check >= 0.1:
                self.checked_arm_state()
                last_status_check = now
            self.robot.move_arm_joint_servo(
                command, self.control_period_s, 2, False
            )
            sent += 1
            deadline += self.control_period_s
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
        return sent

    def move_gripper(self, side: str, position_rad: float) -> None:
        self._require_motion()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        self.checked_arm_state()
        raw = float(position_rad)
        clipped = clamp_omnipicker(raw)
        if clipped != raw:
            raise SafetyError("Omnipicker target is outside [-pi/4, 0] rad")
        self.robot.move_end_effector_joint(
            [clipped], [0.0], f"{side}_tool"
        )
