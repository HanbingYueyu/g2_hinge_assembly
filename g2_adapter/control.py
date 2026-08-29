"""Direct, safety-gated GDK control for the G2 traditional workflow."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

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


@dataclass(frozen=True)
class CartesianPoseTarget:
    """A finite TCP pose in GDK's ``base_link`` frame."""

    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        if len(self.position_m) != 3 or not all(math.isfinite(value) for value in self.position_m):
            raise SafetyError("cartesian position must contain three finite values")
        if len(self.orientation_xyzw) != 4 or not all(math.isfinite(value) for value in self.orientation_xyzw):
            raise SafetyError("cartesian orientation must contain four finite values")
        norm = math.sqrt(sum(value * value for value in self.orientation_xyzw))
        if norm < 1e-6:
            raise SafetyError("cartesian orientation quaternion has zero length")


class LeftCartesianImpedanceStream:
    """Publish one left-TCP target at a fixed rate while it remains valid.

    GDK requires ``end_effector_pose_control`` commands at 50 Hz.  SAM3 is
    deliberately slower than that, so perception only updates this object's
    target; this class performs the required interpolation-free setpoint
    publication.  Calling ``stop`` prevents all subsequent GDK pose calls.
    """

    def __init__(self, controller: "G2Controller", *, rate_hz: float, life_time_s: float) -> None:
        if not 45.0 <= rate_hz <= 55.0:
            raise ValueError("Cartesian impedance publication rate must be in [45, 55] Hz")
        if not 0.04 <= life_time_s <= 0.20:
            raise ValueError("Cartesian impedance life_time must be in [0.04, 0.20] seconds")
        self._controller = controller
        self._period_s = 1.0 / float(rate_hz)
        self._life_time_s = float(life_time_s)
        self._lock = threading.Lock()
        self._target: CartesianPoseTarget | None = None
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, initial_target: CartesianPoseTarget) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("Cartesian impedance stream is already started")
            self._target = initial_target
            self._thread = threading.Thread(target=self._run, name="g2-left-cartesian-impedance", daemon=True)
            self._thread.start()

    def update_target(self, target: CartesianPoseTarget) -> None:
        self.raise_if_failed()
        with self._lock:
            if self._stop.is_set() or self._thread is None:
                raise RuntimeError("Cartesian impedance stream is not running")
            self._target = target

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise SafetyError(f"Cartesian impedance stream stopped: {error}") from error

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, self._period_s * 4.0))
        self._thread = None

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                target = self._target
            if target is None:
                return
            try:
                self._controller._send_left_cartesian_pose(target, self._life_time_s)
            except Exception as exc:  # Surface failure to the perception loop and stop immediately.
                with self._lock:
                    self._error = exc
                self._stop.set()
                return
            deadline += self._period_s
            self._stop.wait(max(0.0, deadline - time.monotonic()))


class LeftEndEffectorPositionStream:
    """Keep a mode-1 left end-link target alive at 50 Hz without pose steps.

    ``end_effector_pose_control`` is a whole-body-IK interface in position
    mode.  Per GDK's contract, commands must be published at 50 Hz and may
    not jump.  This stream therefore holds the last target continuously and
    slews only translation; the end-link orientation is fixed at startup.
    """

    def __init__(
        self,
        controller: "G2Controller",
        *,
        rate_hz: float,
        life_time_s: float,
        max_translation_velocity_mps: float,
    ) -> None:
        if not 45.0 <= rate_hz <= 55.0:
            raise ValueError("end-effector position publication rate must be in [45, 55] Hz")
        if not 0.04 <= life_time_s <= 0.20:
            raise ValueError("end-effector position life_time must be in [0.04, 0.20] seconds")
        if not 0.0 < max_translation_velocity_mps <= 0.05:
            raise ValueError("end-effector translation velocity must be in (0, 0.05] m/s")
        self._controller = controller
        self._period_s = 1.0 / float(rate_hz)
        self._life_time_s = float(life_time_s)
        self._max_translation_velocity_mps = float(max_translation_velocity_mps)
        self._lock = threading.Lock()
        self._commanded: CartesianPoseTarget | None = None
        self._desired: CartesianPoseTarget | None = None
        self._fixed_orientation: tuple[float, float, float, float] | None = None
        self._gripper_joint_names: tuple[str, ...] = ()
        self._gripper_joint_positions: tuple[float, ...] = ()
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._at_target = threading.Event()
        self._target_generation = 0
        self._arrived_generation = 0
        self._thread: threading.Thread | None = None

    def start(self, initial_target: CartesianPoseTarget) -> None:
        with self._lock:
            if self._thread is not None:
                raise RuntimeError("end-effector position stream is already started")
            self._commanded = initial_target
            self._desired = initial_target
            self._fixed_orientation = initial_target.orientation_xyzw
            self._target_generation = 0
            self._arrived_generation = 0
            self._at_target.set()
            self._thread = threading.Thread(target=self._run, name="g2-left-end-effector-position", daemon=True)
            self._thread.start()

    def update_target(self, target: CartesianPoseTarget) -> None:
        self.raise_if_failed()
        with self._lock:
            if self._stop.is_set() or self._thread is None:
                raise RuntimeError("end-effector position stream is not running")
            if target.orientation_xyzw != self._fixed_orientation:
                raise SafetyError("mode-1 visual servo must keep the initial end-link orientation")
            self._desired = target
            self._target_generation += 1
            self._at_target.clear()

    def wait_until_target(self, timeout_s: float) -> bool:
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("target wait timeout must be positive and finite")
        deadline = time.monotonic() + timeout_s
        with self._lock:
            requested_generation = self._target_generation
        while True:
            self.raise_if_failed()
            with self._lock:
                if self._arrived_generation >= requested_generation:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._at_target.wait(remaining)

    def set_left_gripper_target(self, position_rad: float) -> None:
        """Synchronize the left Omnipicker with this 50-Hz pose stream.

        GDK documents ``EndEffectorPose.end_effector_joint_*`` as the
        supported way to drive an end effector together with an active pose
        stream.  The installed end-effector name is queried at run time rather
        than being guessed from the URDF.
        """
        self.raise_if_failed()
        raw = float(position_rad)
        clipped = clamp_omnipicker(raw)
        if clipped != raw:
            raise SafetyError("Omnipicker target is outside [-pi/4, 0] rad")
        names, current_positions = self._controller.read_gripper_state("left")
        if len(names) != 1 or len(current_positions) != 1:
            raise SafetyError(
                "left_tool 的 GDK end_state 未提供恰好一个可同步控制的夹爪关节"
            )
        with self._lock:
            if self._stop.is_set() or self._thread is None:
                raise RuntimeError("end-effector position stream is not running")
            self._gripper_joint_names = names
            self._gripper_joint_positions = (clipped,)

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise SafetyError(f"end-effector position stream stopped: {error}") from error

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.2, self._period_s * 4.0))
        self._thread = None

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                commanded = self._commanded
                desired = self._desired
                generation = self._target_generation
                gripper_joint_names = self._gripper_joint_names
                gripper_joint_positions = self._gripper_joint_positions
            if commanded is None or desired is None:
                return
            current_position = np.asarray(commanded.position_m, dtype=np.float64)
            desired_position = np.asarray(desired.position_m, dtype=np.float64)
            delta = desired_position - current_position
            distance = float(np.linalg.norm(delta))
            max_step_m = self._max_translation_velocity_mps * self._period_s
            if distance <= max_step_m:
                next_target = desired
                self._at_target.set()
            else:
                next_position = current_position + delta * (max_step_m / distance)
                next_target = CartesianPoseTarget(
                    tuple(float(value) for value in next_position),
                    commanded.orientation_xyzw,
                )
            try:
                self._controller._send_left_cartesian_pose(
                    next_target,
                    self._life_time_s,
                    end_effector_joint_names=gripper_joint_names,
                    end_effector_joint_positions=gripper_joint_positions,
                )
            except Exception as exc:
                with self._lock:
                    self._error = exc
                self._stop.set()
                return
            with self._lock:
                self._commanded = next_target
                if next_target == desired:
                    self._arrived_generation = max(self._arrived_generation, generation)
                    if generation == self._target_generation:
                        self._at_target.set()
            deadline += self._period_s
            self._stop.wait(max(0.0, deadline - time.monotonic()))


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

    def motion_status_summary(self) -> dict:
        """Return the live mode/error fields for diagnostics without motion."""
        status = self.robot.get_motion_control_status()
        return {
            "mode": int(getattr(status, "mode", -1)),
            "control_mode": int(getattr(status, "control_mode", -1)),
            "error_code": int(getattr(status, "error_code", -1)),
            "error_msg": str(getattr(status, "error_msg", "")),
        }

    def read_end_effector_pose(self, frame_name: str) -> CartesianPoseTarget:
        """Read a named end-link pose from GDK's motion-status source of truth."""
        status = self.robot.get_motion_control_status()
        if int(status.error_code) != 0:
            raise SafetyError(f"cannot read end-effector pose: motion error {int(status.error_code)}: {status.error_msg}")
        names = [str(value) for value in status.frame_names]
        try:
            pose = status.frame_poses[names.index(str(frame_name))]
        except (AttributeError, ValueError, IndexError) as exc:
            raise SafetyError(f"GDK motion status does not contain frame {frame_name}") from exc
        return CartesianPoseTarget(
            (float(pose.position.x), float(pose.position.y), float(pose.position.z)),
            (
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ),
        )

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
        duration_s: float | None = None,
    ) -> int:
        """Interpolate from measured state and stream absolute 14-D targets.

        An explicit duration is accepted only when it does not exceed the
        configured joint velocity limit.  This lets a supervised final grasp
        descent hold a physical duration even when camera inference is slow.
        """
        self._require_motion()
        target = validate_arm_target(target_rad)
        start = self.checked_arm_state()
        max_delta = max(abs(end - begin) for begin, end in zip(start, target))
        minimum_duration = max_delta / self.max_servo_velocity_rad_s
        if duration_s is None:
            duration = minimum_duration
        else:
            duration = float(duration_s)
            if not math.isfinite(duration) or duration <= 0.0:
                raise SafetyError("servo duration must be a positive finite value")
            if duration + 1e-9 < minimum_duration:
                raise SafetyError(
                    f"requested servo duration {duration:.3f}s exceeds max joint velocity "
                    f"{self.max_servo_velocity_rad_s:.3f} rad/s"
                )
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

    def servo_waypoints(
        self,
        waypoint_targets_rad: Sequence[Sequence[float]],
        *,
        duration_s: float,
        stop_requested: Callable[[], bool] | None = None,
    ) -> int:
        """Stream bounded joint waypoints over one explicit total duration."""
        self._require_motion()
        targets = tuple(validate_arm_target(target) for target in waypoint_targets_rad)
        if not targets:
            raise SafetyError("servo waypoints must not be empty")
        duration = float(duration_s)
        if not math.isfinite(duration) or duration <= 0.0:
            raise SafetyError("servo waypoint duration must be a positive finite value")
        start = self.checked_arm_state()
        path = (start, *targets)
        minimums = tuple(
            max(abs(end - begin) for begin, end in zip(previous, following)) / self.max_servo_velocity_rad_s
            for previous, following in zip(path[:-1], path[1:])
        )
        minimum_total = sum(minimums)
        if duration + 1e-9 < minimum_total:
            raise SafetyError(
                f"requested waypoint duration {duration:.3f}s exceeds max joint velocity "
                f"{self.max_servo_velocity_rad_s:.3f} rad/s"
            )
        surplus_per_segment = (duration - minimum_total) / len(targets)
        segment_durations = tuple(minimum + surplus_per_segment for minimum in minimums)
        boundaries = []
        elapsed = 0.0
        for segment_duration in segment_durations:
            elapsed += segment_duration
            boundaries.append(elapsed)
        steps = max(1, math.ceil(duration / self.control_period_s))
        deadline = time.monotonic()
        last_status_check = deadline
        sent = 0
        for step in range(1, steps + 1):
            if stop_requested is not None and stop_requested():
                return sent
            now_elapsed = min(duration, step * duration / steps)
            segment_index = next(index for index, boundary in enumerate(boundaries) if now_elapsed <= boundary + 1e-12)
            segment_start = 0.0 if segment_index == 0 else boundaries[segment_index - 1]
            fraction = (now_elapsed - segment_start) / segment_durations[segment_index]
            begin = path[segment_index]
            end = path[segment_index + 1]
            command = [begin_value + fraction * (end_value - begin_value) for begin_value, end_value in zip(begin, end)]
            now = time.monotonic()
            if now - last_status_check >= 0.1:
                self.checked_arm_state()
                last_status_check = now
            self.robot.move_arm_joint_servo(command, self.control_period_s, 2, False)
            sent += 1
            deadline += self.control_period_s
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
        return sent

    def servo_step(self, target_rad: Sequence[float]) -> None:
        """Send one bounded, absolute arm-servo setpoint.

        This is deliberately a single setpoint rather than a trajectory.  A
        visual-servo owner must recalculate its next target from fresh camera
        data and current joint feedback before calling this method again.
        When it stops calling, this class does not send any further motion
        request.
        """
        self._require_motion()
        target = validate_arm_target(target_rad)
        self.checked_arm_state()
        self.robot.move_arm_joint_servo(
            list(target), self.control_period_s, 2, False
        )

    def enable_left_cartesian_impedance(self, kp: Sequence[float]) -> None:
        """Enter GDK mode 2 and configure only the left arm's six ``kp`` values.

        This makes no end-pose request by itself.  The caller must immediately
        start a :class:`LeftCartesianImpedanceStream` using the measured TCP
        pose, so a mode switch cannot be followed by an unknown target.
        """
        self._require_motion()
        values = tuple(float(value) for value in kp)
        if len(values) != 6 or not all(math.isfinite(value) for value in values):
            raise SafetyError("left Cartesian impedance must contain six finite kp values")
        lower = (600.0, 600.0, 600.0, 30.0, 30.0, 30.0)
        upper = (6000.0, 6000.0, 6000.0, 300.0, 300.0, 300.0)
        if any(value < lo or value > hi for value, lo, hi in zip(values, lower, upper)):
            raise SafetyError("left Cartesian impedance kp is outside GDK's documented safe range")
        status = self.robot.get_motion_control_status()
        if int(status.error_code) != 0:
            raise SafetyError(f"cannot enter Cartesian impedance: motion error {int(status.error_code)}: {status.error_msg}")
        if int(status.control_mode) != 1:
            raise SafetyError(f"left grasp must begin in position mode 1, current control_mode={int(status.control_mode)}")
        self.set_control_mode(2)
        impedances = [self.gdk.Impedance(kp=value) for value in values]
        self.robot.set_cartesian_impedance(impedances, 0)

    def set_control_mode(self, control_mode: int) -> None:
        """Set and verify a GDK control mode through the sole command boundary."""
        self._require_motion()
        mode = int(control_mode)
        if mode not in {1, 2, 3}:
            raise SafetyError("GDK control mode must be 1, 2, or 3")
        before = self.robot.get_motion_control_status()
        if int(before.error_code) != 0:
            raise SafetyError(f"cannot switch control mode: motion error {int(before.error_code)}: {before.error_msg}")
        if int(before.control_mode) == mode:
            return
        self.robot.set_control_mode(mode)
        # GDK documents control_mode as a persistent robot state.  Its status
        # update can lag the setter by a short DDS cycle, so do not mistake an
        # immediate stale read for a failed mode transition.
        deadline = time.monotonic() + 2.0
        after = before
        while True:
            after = self.robot.get_motion_control_status()
            if int(after.control_mode) == mode and int(after.error_code) == 0:
                return
            if time.monotonic() >= deadline:
                raise SafetyError(
                    f"failed to set control_mode={mode}: actual={int(after.control_mode)}, "
                    f"error={int(after.error_code)}: {after.error_msg}"
                )
            time.sleep(0.05)

    def start_left_cartesian_impedance_stream(
        self,
        initial_target: CartesianPoseTarget,
        *,
        rate_hz: float = 50.0,
        life_time_s: float = 0.10,
    ) -> LeftCartesianImpedanceStream:
        self._require_motion()
        status = self.motion_status_summary()
        if status["control_mode"] != 2 or status["error_code"] != 0:
            raise SafetyError(f"Cartesian impedance stream requires healthy GDK control_mode=2; actual={status}")
        stream = LeftCartesianImpedanceStream(self, rate_hz=rate_hz, life_time_s=life_time_s)
        stream.start(initial_target)
        return stream

    def start_left_end_effector_position_stream(
        self,
        initial_target: CartesianPoseTarget,
        *,
        rate_hz: float = 50.0,
        life_time_s: float = 0.10,
        max_translation_velocity_mps: float = 0.01,
    ) -> LeftEndEffectorPositionStream:
        """Start GDK's direct mode-1 end-link pose control without a mode change."""
        self._require_motion()
        status = self.motion_status_summary()
        if status["control_mode"] != 1 or status["error_code"] != 0:
            raise SafetyError(
                "end-effector position stream requires healthy GDK control_mode=1 (position control); "
                f"actual={status}. control_mode=3 is joint impedance and is deliberately refused by this position-only workflow"
            )
        stream = LeftEndEffectorPositionStream(
            self,
            rate_hz=rate_hz,
            life_time_s=life_time_s,
            max_translation_velocity_mps=max_translation_velocity_mps,
        )
        stream.start(initial_target)
        return stream

    def send_left_end_effector_position_setpoint(
        self, target: CartesianPoseTarget, *, life_time_s: float = 0.10
    ) -> None:
        """Send one mode-1 left end-link setpoint from the caller's thread.

        This is intentionally a low-level, motion-gated primitive for the
        supervised diagnostic probe.  Production visual servo uses the 50 Hz
        stream; callers must interpolate before repeatedly using this method.
        """
        self._require_motion()
        if not math.isfinite(life_time_s) or not 0.04 <= life_time_s <= 0.20:
            raise SafetyError("end-effector position life_time must be in [0.04, 0.20] seconds")
        status = self.motion_status_summary()
        if status["control_mode"] != 1 or status["error_code"] != 0:
            raise SafetyError(
                "end-effector position setpoint requires healthy GDK control_mode=1 (position control); "
                f"actual={status}. control_mode=3 is joint impedance and is deliberately refused by this position-only workflow"
            )
        self._send_left_cartesian_pose(target, life_time_s)

    def _send_left_cartesian_pose(
        self,
        target: CartesianPoseTarget,
        life_time_s: float,
        *,
        end_effector_joint_names: Sequence[str] = (),
        end_effector_joint_positions: Sequence[float] = (),
    ) -> None:
        """Internal 50-Hz sender; public callers use the gated stream above."""
        self._require_motion()
        if len(end_effector_joint_names) != len(end_effector_joint_positions):
            raise ValueError("end-effector joint names and positions must have the same length")
        request = self.gdk.EndEffectorPose()
        request.life_time = float(life_time_s)
        request.group = self.gdk.EndEffectorControlGroup.kLeftArm
        pose = request.left_end_effector_pose
        pose.position.x, pose.position.y, pose.position.z = target.position_m
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = target.orientation_xyzw
        if end_effector_joint_names:
            request.end_effector_joint_names = list(end_effector_joint_names)
            request.end_effector_joint_positions = [float(value) for value in end_effector_joint_positions]
        result = self.robot.end_effector_pose_control(request)
        if result not in (None, 0):
            raise SafetyError(f"GDK end_effector_pose_control failed: {result}")

    def read_gripper_state(self, side: str) -> tuple[tuple[str, ...], tuple[float, ...]]:
        """Read GDK's installed end-effector names and finite positions."""
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        end_state = dict(self.robot.get_end_state())
        group = end_state.get(f"{side}_end_state")
        if not isinstance(group, dict):
            return (), ()
        names = tuple(str(value) for value in group.get("names", ()))
        states = tuple(group.get("end_states", ()))
        if len(names) != len(states):
            return (), ()
        positions = tuple(float(state.get("position", math.nan)) for state in states)
        if not all(math.isfinite(value) for value in positions):
            return (), ()
        return names, positions

    def read_gripper_position(self, side: str) -> float | None:
        """Read a single Omnipicker position without issuing a command."""
        names, positions = self.read_gripper_state(side)
        return positions[0] if len(names) == 1 and len(positions) == 1 else None

    def move_gripper(self, side: str, position_rad: float) -> None:
        self._require_motion()
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        self.checked_arm_state()
        raw = float(position_rad)
        clipped = clamp_omnipicker(raw)
        if clipped != raw:
            raise SafetyError("Omnipicker target is outside [-pi/4, 0] rad")
        result = self.robot.move_end_effector_joint(
            [clipped], [0.0], f"{side}_tool"
        )
        if result not in (None, 0):
            raise SafetyError(f"GDK move_end_effector_joint failed: {result}")
