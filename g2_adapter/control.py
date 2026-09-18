"""Direct, safety-gated GDK control for the G2 traditional workflow."""

from __future__ import annotations

import math
import threading
import time
from contextlib import nullcontext
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

try:
    from .safety import (
        AdvanceRightRequested,
        ARM_JOINT_NAMES,
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        MotionWriteCancelled,
        SafetyError,
        clamp_omnipicker,
        reorder_arm_positions,
        validate_arm_target,
        validate_robot_status,
    )
except ImportError:  # Direct script/module use from this directory.
    from safety import (
        AdvanceRightRequested,
        ARM_JOINT_NAMES,
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        MotionWriteCancelled,
        SafetyError,
        clamp_omnipicker,
        reorder_arm_positions,
        validate_arm_target,
        validate_robot_status,
    )


class MotionDisabledError(SafetyError):
    pass


class MotionRebasePending(Exception):
    """A resumed stream is waiting for post-pause measured feedback."""


_FIXED_ORIENTATION_TOLERANCE_DEG = 0.01
_DEFAULT_FEEDBACK_RATE_HZ = 10.0
_DEFAULT_SOFT_FOLLOW_ERROR_M = 0.005
_DEFAULT_FEEDBACK_STALE_TIMEOUT_S = 0.30
_DEFAULT_ACTUAL_ARRIVAL_POSITION_TOLERANCE_M = 0.005
_DEFAULT_ACTUAL_ARRIVAL_ORIENTATION_TOLERANCE_DEG = 2.0
_DEFAULT_ACTUAL_ARRIVAL_STABLE_SAMPLES = 3


def quaternion_slerp(
    first: Sequence[float], second: Sequence[float], fraction: float,
) -> tuple[float, float, float, float]:
    """Interpolate two XYZW quaternions along the shortest unit arc."""
    if not math.isfinite(float(fraction)) or not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("quaternion interpolation fraction must be in [0, 1]")
    q0 = np.asarray(tuple(float(value) for value in first), dtype=np.float64)
    q1 = np.asarray(tuple(float(value) for value in second), dtype=np.float64)
    if q0.shape != (4,) or q1.shape != (4,):
        raise ValueError("quaternion interpolation requires two XYZW quaternions")
    n0 = float(np.linalg.norm(q0))
    n1 = float(np.linalg.norm(q1))
    if n0 <= 1e-12 or n1 <= 1e-12:
        raise ValueError("quaternion interpolation cannot use a zero quaternion")
    q0 /= n0
    q1 /= n1
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(1.0, max(-1.0, dot))
    t = float(fraction)
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        result /= np.linalg.norm(result)
        return tuple(float(value) for value in result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    result = (math.sin((1.0 - t) * theta) * q0 + math.sin(t * theta) * q1) / sin_theta
    return tuple(float(value) for value in result)


def quaternion_angle_deg(first: Sequence[float], second: Sequence[float]) -> float:
    """Return the shortest rotational distance between two XYZW quaternions."""
    q0 = np.asarray(tuple(float(value) for value in first), dtype=np.float64)
    q1 = np.asarray(tuple(float(value) for value in second), dtype=np.float64)
    if q0.shape != (4,) or q1.shape != (4,):
        raise ValueError("quaternion angle requires two XYZW quaternions")
    n0 = float(np.linalg.norm(q0))
    n1 = float(np.linalg.norm(q1))
    if n0 <= 1e-12 or n1 <= 1e-12:
        raise ValueError("quaternion angle cannot use a zero quaternion")
    dot = abs(float(np.dot(q0 / n0, q1 / n1)))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


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


def _pose_errors(actual: CartesianPoseTarget, target: CartesianPoseTarget) -> tuple[float, float]:
    """Return translational metres and rotational degrees from target to actual."""
    position_error_m = float(
        np.linalg.norm(
            np.asarray(actual.position_m, dtype=np.float64)
            - np.asarray(target.position_m, dtype=np.float64)
        )
    )
    orientation_error_deg = quaternion_angle_deg(actual.orientation_xyzw, target.orientation_xyzw)
    return position_error_m, orientation_error_deg


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
        guard = getattr(self._controller, 'check_automatic_guard', None)
        if guard is not None:
            guard()
        with self._lock:
            error = self._error
        if error is not None:
            if isinstance(error, AdvanceRightRequested):
                raise error
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
    Target progression is paced by a lower-rate actual end-link feedback
    cache, so a slow robot can catch up without accumulating open-loop error.
    """

    def __init__(
        self,
        controller: "G2Controller",
        *,
        rate_hz: float,
        life_time_s: float,
        max_translation_velocity_mps: float,
        allow_orientation_change: bool = False,
        max_angular_velocity_rad_s: float = 0.5,
        feedback_rate_hz: float = _DEFAULT_FEEDBACK_RATE_HZ,
        soft_follow_error_m: float = _DEFAULT_SOFT_FOLLOW_ERROR_M,
        feedback_stale_timeout_s: float = _DEFAULT_FEEDBACK_STALE_TIMEOUT_S,
    ) -> None:
        if not 45.0 <= rate_hz <= 55.0:
            raise ValueError("end-effector position publication rate must be in [45, 55] Hz")
        if not 0.04 <= life_time_s <= 0.20:
            raise ValueError("end-effector position life_time must be in [0.04, 0.20] seconds")
        if not math.isfinite(max_translation_velocity_mps) or max_translation_velocity_mps <= 0.0:
            raise ValueError("end-effector translation velocity must be a positive finite value")
        if not 0.01 <= max_angular_velocity_rad_s <= 1.0:
            raise ValueError("end-effector angular velocity must be in [0.01, 1.0] rad/s")
        if not 10.0 <= feedback_rate_hz <= 20.0:
            raise ValueError("end-effector feedback rate must be in [10, 20] Hz")
        if not 0.0 < soft_follow_error_m <= 0.05:
            raise ValueError("soft follow error must be in (0, 0.05] m")
        if not 0.10 <= feedback_stale_timeout_s <= 2.0:
            raise ValueError("feedback stale timeout must be in [0.10, 2.0] seconds")
        self._controller = controller
        self._period_s = 1.0 / float(rate_hz)
        self._feedback_period_s = 1.0 / float(feedback_rate_hz)
        self._life_time_s = float(life_time_s)
        self._max_translation_velocity_mps = float(max_translation_velocity_mps)
        self._allow_orientation_change = bool(allow_orientation_change)
        self._max_angular_velocity_rad_s = float(max_angular_velocity_rad_s)
        self._soft_follow_error_m = float(soft_follow_error_m)
        self._feedback_stale_timeout_s = float(feedback_stale_timeout_s)
        self._frame_name = "arm_l_end_link"
        self._lock = threading.Lock()
        self._commanded: CartesianPoseTarget | None = None
        self._desired: CartesianPoseTarget | None = None
        self._actual: CartesianPoseTarget | None = None
        self._actual_at = 0.0
        self._feedback_sequence = 0
        self._fixed_orientation: tuple[float, float, float, float] | None = None
        self._gripper_joint_names: tuple[str, ...] = ()
        self._gripper_joint_positions: tuple[float, ...] = ()
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._at_target = threading.Event()
        self._target_generation = 0
        self._arrived_generation = 0
        self._actual_arrival_stable_samples = 0
        self._thread: threading.Thread | None = None
        self._feedback_thread: threading.Thread | None = None
        self._gate_generation = 0
        self._pending_rebase_generation: int | None = None
        self._pending_rebase_feedback_sequence = 0

    def start(self, initial_target: CartesianPoseTarget) -> None:
        with self._lock:
            if (
                self._thread is not None and self._thread.is_alive()
            ) or (
                self._feedback_thread is not None and self._feedback_thread.is_alive()
            ):
                raise RuntimeError("end-effector position stream is already started")
            self._thread = None
            self._feedback_thread = None
            self._commanded = initial_target
            self._desired = initial_target
            self._actual = None
            self._actual_at = 0.0
            self._feedback_sequence = 0
            self._fixed_orientation = initial_target.orientation_xyzw
            self._target_generation = 0
            self._arrived_generation = -1
            self._actual_arrival_stable_samples = 0
            self._at_target.clear()
            self._stop.clear()
            self._error = None
            gate = self._controller.motion_gate
            self._gate_generation = 0 if gate is None else gate.pause_generation
            self._pending_rebase_generation = None
            self._pending_rebase_feedback_sequence = 0
            self._feedback_thread = threading.Thread(
                target=self._feedback_loop, name="g2-left-end-effector-feedback", daemon=True,
            )
            self._thread = threading.Thread(target=self._run, name="g2-left-end-effector-position", daemon=True)
            self._feedback_thread.start()
            self._thread.start()

    def update_target(self, target: CartesianPoseTarget) -> None:
        self.raise_if_failed()
        with self._lock:
            if self._stop.is_set() or self._thread is None:
                raise RuntimeError("end-effector position stream is not running")
            if not self._allow_orientation_change and quaternion_angle_deg(target.orientation_xyzw, self._fixed_orientation) > _FIXED_ORIENTATION_TOLERANCE_DEG:
                raise SafetyError("mode-1 visual servo must keep the initial end-link orientation")
            self._desired = target
            self._target_generation += 1
            self._actual_arrival_stable_samples = 0
            self._at_target.clear()

    def wait_until_target(self, timeout_s: float) -> bool:
        """Wait until actual end-link feedback reaches the current target."""
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("target wait timeout must be positive and finite")
        deadline = time.monotonic() + timeout_s
        with self._lock:
            requested_generation = self._target_generation
        while True:
            self.raise_if_failed()
            with self._lock:
                arrived = self._arrived_generation >= requested_generation
                actual_at = self._actual_at
                feedback_sequence = self._feedback_sequence
            if arrived:
                age_s = time.monotonic() - actual_at
                if feedback_sequence > 0 and age_s <= self._feedback_stale_timeout_s:
                    return True
                if age_s > self._feedback_stale_timeout_s:
                    raise SafetyError("left end-effector feedback is stale while waiting for actual arrival")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._at_target.wait(remaining)

    def wait_until_commanded_target(self, target: CartesianPoseTarget, timeout_s: float,
                                    *, tolerance_m: float = 0.0001) -> bool:
        """Confirm the stream writer has transmitted a requested hold target."""
        if not math.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("commanded-target timeout must be positive and finite")
        if not math.isfinite(tolerance_m) or not 0.0 < tolerance_m <= 0.005:
            raise ValueError("commanded-target tolerance must be in (0, 0.005]")
        deadline = time.monotonic() + timeout_s
        while True:
            self.raise_if_failed()
            with self._lock:
                commanded = self._commanded
            if commanded is not None:
                position_error_m, orientation_error_deg = _pose_errors(commanded, target)
                if position_error_m <= tolerance_m and orientation_error_deg <= 0.1:
                    return True
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            time.sleep(min(self._period_s, remaining))

    def is_target_reached(self) -> bool:
        """Return whether actual end-link feedback has reached the target.

        State-machine callers use this to keep perception and the web UI
        responsive while a 50-Hz position stream is moving.
        """
        self.raise_if_failed()
        with self._lock:
            return (
                self._feedback_sequence > 0
                and self._arrived_generation >= self._target_generation
                and time.monotonic() - self._actual_at <= self._feedback_stale_timeout_s
            )

    def set_translation_velocity(self, velocity_mps: float) -> None:
        """Change stream rate limit without replacing the sole command writer."""
        if not math.isfinite(velocity_mps) or velocity_mps <= 0.0:
            raise ValueError("translation velocity must be a positive finite value")
        with self._lock:
            self._max_translation_velocity_mps = float(velocity_mps)

    def wait_until_actual_target(
        self,
        target: CartesianPoseTarget,
        timeout_s: float,
        *,
        position_tolerance_m: float = _DEFAULT_ACTUAL_ARRIVAL_POSITION_TOLERANCE_M,
        orientation_tolerance_deg: float = _DEFAULT_ACTUAL_ARRIVAL_ORIENTATION_TOLERANCE_DEG,
        stable_samples: int = _DEFAULT_ACTUAL_ARRIVAL_STABLE_SAMPLES,
        poll_period_s: float = 0.02,
    ) -> CartesianPoseTarget | None:
        """Wait for fresh cached feedback to stay at ``target``."""
        if not 0.0 < position_tolerance_m <= 0.05:
            raise ValueError("actual position tolerance must be in (0, 0.05] m")
        if not 0.0 < orientation_tolerance_deg <= 10.0:
            raise ValueError("actual orientation tolerance must be in (0, 10] deg")
        if not 1 <= int(stable_samples) <= 50 or not 0.01 <= poll_period_s <= 0.10:
            raise ValueError("actual-pose stability parameters are invalid")
        deadline = time.monotonic() + float(timeout_s)
        last_sequence = -1
        consecutive = 0
        latest: CartesianPoseTarget | None = None
        while time.monotonic() < deadline:
            self.raise_if_failed()
            with self._lock:
                actual = self._actual
                actual_at = self._actual_at
                sequence = self._feedback_sequence
            if actual is not None:
                latest = actual
                if time.monotonic() - actual_at > self._feedback_stale_timeout_s:
                    raise SafetyError("left end-effector feedback is stale while waiting for actual arrival")
                if sequence != last_sequence:
                    last_sequence = sequence
                    position_error_m, orientation_error_deg = _pose_errors(actual, target)
                    if position_error_m <= position_tolerance_m and orientation_error_deg <= orientation_tolerance_deg:
                        consecutive += 1
                        if consecutive >= int(stable_samples):
                            return actual
                    else:
                        consecutive = 0
            self._stop.wait(min(float(poll_period_s), self._feedback_period_s))
        return None

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
        guard = getattr(self._controller, 'check_automatic_guard', None)
        if guard is not None:
            guard()
        with self._lock:
            error = self._error
        if error is not None:
            if isinstance(error, AdvanceRightRequested):
                raise error
            raise SafetyError(f"end-effector position stream stopped: {error}") from error

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        gate = self._controller.motion_gate
        if gate is not None:
            gate.cancel_automatic_thread(thread)
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        feedback_thread = self._feedback_thread
        if feedback_thread is not None and feedback_thread is not threading.current_thread():
            feedback_thread.join(timeout=max(0.2, self._feedback_period_s * 4.0))
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = None
            if self._feedback_thread is None or not self._feedback_thread.is_alive():
                self._feedback_thread = None
        if gate is not None:
            gate.clear_cancelled_thread(thread)

    def _set_error(self, error: Exception) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
        self._stop.set()

    def _refresh_target_before_send(self, proposed: CartesianPoseTarget) -> CartesianPoseTarget:
        """Replace a pre-pause proposal with a measured hold after manual motion."""
        gate = self._controller.motion_gate
        if gate is None:
            return proposed
        generation = gate.pause_generation
        with self._lock:
            if generation == self._gate_generation:
                return proposed
            if self._pending_rebase_generation != generation:
                # Establish the feedback baseline only after this pause has
                # completed and automatic motion has regained the writer lock.
                # A sample received before takeover therefore cannot be used
                # as the post-manual measured hold.
                self._pending_rebase_generation = generation
                self._pending_rebase_feedback_sequence = self._feedback_sequence
                raise MotionRebasePending("waiting for fresh post-pause left feedback")
            actual = self._actual
            actual_at = self._actual_at
            sequence = self._feedback_sequence
            baseline_sequence = self._pending_rebase_feedback_sequence
        if (
            actual is None
            or sequence <= baseline_sequence
            or time.monotonic() - actual_at > self._feedback_stale_timeout_s
        ):
            raise MotionRebasePending("waiting for fresh post-pause left feedback")
        with self._lock:
            # Commit only the generation captured above. A concurrent pause
            # increments the gate again and remains pending for the next send.
            self._gate_generation = generation
            self._pending_rebase_generation = None
        return actual

    def _feedback_loop(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                actual = self._controller.read_end_effector_pose(self._frame_name)
            except Exception as exc:
                self._set_error(exc)
                return
            now = time.monotonic()
            with self._lock:
                self._actual = actual
                self._actual_at = now
                self._feedback_sequence += 1
                desired = self._desired
                generation = self._target_generation
                if desired is not None:
                    position_error_m, orientation_error_deg = _pose_errors(actual, desired)
                    within_tolerance = (
                        position_error_m <= _DEFAULT_ACTUAL_ARRIVAL_POSITION_TOLERANCE_M
                        and orientation_error_deg <= _DEFAULT_ACTUAL_ARRIVAL_ORIENTATION_TOLERANCE_DEG
                    )
                    if within_tolerance:
                        self._actual_arrival_stable_samples += 1
                    else:
                        self._actual_arrival_stable_samples = 0
                        if generation == self._target_generation:
                            self._arrived_generation = min(self._arrived_generation, generation - 1)
                            self._at_target.clear()
                    if (
                        self._actual_arrival_stable_samples >= _DEFAULT_ACTUAL_ARRIVAL_STABLE_SAMPLES
                    ):
                        self._arrived_generation = max(self._arrived_generation, generation)
                        if generation == self._target_generation:
                            self._at_target.set()
            deadline += self._feedback_period_s
            now = time.monotonic()
            if deadline <= now:
                deadline = now + self._feedback_period_s
            self._stop.wait(max(0.0, deadline - now))

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                commanded = self._commanded
                desired = self._desired
                actual = self._actual
                actual_at = self._actual_at
                gripper_joint_names = self._gripper_joint_names
                gripper_joint_positions = self._gripper_joint_positions
            if commanded is None or desired is None:
                return
            if actual is None:
                self._stop.wait(min(self._period_s, self._feedback_period_s))
                deadline = time.monotonic()
                continue
            if time.monotonic() - actual_at > self._feedback_stale_timeout_s:
                self._set_error(SafetyError("left end-effector feedback is stale; holding motion stopped"))
                return
            actual_follow_error_m, _ = _pose_errors(actual, commanded)
            current_position = np.asarray(commanded.position_m, dtype=np.float64)
            desired_position = np.asarray(desired.position_m, dtype=np.float64)
            delta = desired_position - current_position
            distance = float(np.linalg.norm(delta))
            max_step_m = self._max_translation_velocity_mps * self._period_s
            current_orientation = np.asarray(commanded.orientation_xyzw, dtype=np.float64)
            desired_orientation = np.asarray(desired.orientation_xyzw, dtype=np.float64)
            orientation_dot = abs(float(np.dot(
                current_orientation / np.linalg.norm(current_orientation),
                desired_orientation / np.linalg.norm(desired_orientation),
            )))
            orientation_angle = 2.0 * math.acos(min(1.0, max(-1.0, orientation_dot)))
            max_angle_step = self._max_angular_velocity_rad_s * self._period_s
            can_advance = actual_follow_error_m <= self._soft_follow_error_m
            if not can_advance:
                next_target = commanded
            else:
                if distance <= max_step_m:
                    next_position = desired_position
                else:
                    next_position = current_position + delta * (max_step_m / distance)
                if orientation_angle <= max_angle_step:
                    next_orientation = desired.orientation_xyzw
                else:
                    next_orientation = quaternion_slerp(
                        commanded.orientation_xyzw,
                        desired.orientation_xyzw,
                        max_angle_step / orientation_angle,
                    )
                next_target = CartesianPoseTarget(tuple(float(value) for value in next_position), next_orientation)
            try:
                next_target = self._controller._send_left_cartesian_pose(
                    next_target,
                    self._life_time_s,
                    end_effector_joint_names=gripper_joint_names,
                    end_effector_joint_positions=gripper_joint_positions,
                    refresh_target=self._refresh_target_before_send,
                )
            except MotionWriteCancelled:
                return
            except MotionRebasePending:
                self._stop.wait(self._period_s)
                continue
            except Exception as exc:
                self._set_error(exc)
                return
            with self._lock:
                self._commanded = next_target
            deadline += self._period_s
            now = time.monotonic()
            if deadline <= now:
                deadline = now + self._period_s
            self._stop.wait(max(0.0, deadline - now))


class RightEndEffectorPositionStream:
    """50-Hz, rate-limited mode-1 stream for ``arm_r_end_link``.

    The stream can optionally synchronize the right Omnipicker through GDK's
    ``EndEffectorPose.end_effector_joint_*`` fields.  This is important while
    a pose stream is active: sending a separate ``right_tool`` command after
    stopping the stream can be ignored by the pose controller until its
    command lease has fully expired.

    Its use is limited to calibrated, local right-hand motions because GDK's
    pose-control interface does not plan or collision-check a trajectory.
    Target progression is paced by a lower-rate actual end-link feedback
    cache, so a slow robot can catch up without accumulating open-loop error.
    """

    def __init__(
        self, controller: "G2Controller", *, rate_hz: float, life_time_s: float,
        max_translation_velocity_mps: float,
        allow_orientation_change: bool = False,
        max_angular_velocity_rad_s: float = 0.5,
        feedback_rate_hz: float = _DEFAULT_FEEDBACK_RATE_HZ,
        soft_follow_error_m: float = _DEFAULT_SOFT_FOLLOW_ERROR_M,
        feedback_stale_timeout_s: float = _DEFAULT_FEEDBACK_STALE_TIMEOUT_S,
    ) -> None:
        if not 45.0 <= rate_hz <= 55.0:
            raise ValueError("Cartesian position publication rate must be in [45, 55] Hz")
        if not 0.04 <= life_time_s <= 0.20:
            raise ValueError("Cartesian position life_time must be in [0.04, 0.20] seconds")
        if not math.isfinite(max_translation_velocity_mps) or max_translation_velocity_mps <= 0.0:
            raise ValueError("Cartesian translation velocity must be a positive finite value")
        if not 0.01 <= max_angular_velocity_rad_s <= 1.0:
            raise ValueError("Cartesian angular velocity must be in [0.01, 1.0] rad/s")
        if not 10.0 <= feedback_rate_hz <= 20.0:
            raise ValueError("end-effector feedback rate must be in [10, 20] Hz")
        if not 0.0 < soft_follow_error_m <= 0.05:
            raise ValueError("soft follow error must be in (0, 0.05] m")
        if not 0.10 <= feedback_stale_timeout_s <= 2.0:
            raise ValueError("feedback stale timeout must be in [0.10, 2.0] seconds")
        self._controller = controller
        self._period_s = 1.0 / float(rate_hz)
        self._feedback_period_s = 1.0 / float(feedback_rate_hz)
        self._life_time_s = float(life_time_s)
        self._max_translation_velocity_mps = float(max_translation_velocity_mps)
        self._allow_orientation_change = bool(allow_orientation_change)
        self._max_angular_velocity_rad_s = float(max_angular_velocity_rad_s)
        self._soft_follow_error_m = float(soft_follow_error_m)
        self._feedback_stale_timeout_s = float(feedback_stale_timeout_s)
        self._frame_name = "arm_r_end_link"
        self._lock = threading.Lock()
        self._commanded: CartesianPoseTarget | None = None
        self._desired: CartesianPoseTarget | None = None
        self._actual: CartesianPoseTarget | None = None
        self._actual_at = 0.0
        self._feedback_sequence = 0
        self._fixed_orientation: tuple[float, float, float, float] | None = None
        self._gripper_joint_names: tuple[str, ...] = ()
        self._gripper_joint_positions: tuple[float, ...] = ()
        self._error: Exception | None = None
        self._stop = threading.Event()
        self._at_target = threading.Event()
        self._target_generation = 0
        self._arrived_generation = 0
        self._actual_arrival_stable_samples = 0
        self._thread: threading.Thread | None = None
        self._feedback_thread: threading.Thread | None = None
        self._gate_generation = 0
        self._pending_rebase_generation: int | None = None
        self._pending_rebase_feedback_sequence = 0

    def start(self, initial_target: CartesianPoseTarget) -> None:
        with self._lock:
            if (
                self._thread is not None and self._thread.is_alive()
            ) or (
                self._feedback_thread is not None and self._feedback_thread.is_alive()
            ):
                raise RuntimeError("right end-effector position stream is already started")
            self._thread = None
            self._feedback_thread = None
            self._commanded = initial_target
            self._desired = initial_target
            self._actual = None
            self._actual_at = 0.0
            self._feedback_sequence = 0
            self._fixed_orientation = initial_target.orientation_xyzw
            self._target_generation = 0
            self._arrived_generation = -1
            self._actual_arrival_stable_samples = 0
            self._at_target.clear()
            self._stop.clear()
            self._error = None
            gate = self._controller.motion_gate
            self._gate_generation = 0 if gate is None else gate.pause_generation
            self._pending_rebase_generation = None
            self._pending_rebase_feedback_sequence = 0
            self._feedback_thread = threading.Thread(
                target=self._feedback_loop, name="g2-right-end-effector-feedback", daemon=True,
            )
            self._thread = threading.Thread(target=self._run, name="g2-right-end-effector-position", daemon=True)
            self._feedback_thread.start()
            self._thread.start()

    def update_target(self, target: CartesianPoseTarget) -> None:
        self.raise_if_failed()
        with self._lock:
            if self._stop.is_set() or self._thread is None:
                raise RuntimeError("right end-effector position stream is not running")
            if (
                not self._allow_orientation_change
                and quaternion_angle_deg(target.orientation_xyzw, self._fixed_orientation) > _FIXED_ORIENTATION_TOLERANCE_DEG
            ):
                raise SafetyError("right mode-1 local grasp motion must keep the calibrated end-link orientation")
            self._desired = target
            self._target_generation += 1
            self._actual_arrival_stable_samples = 0
            self._at_target.clear()

    def set_right_gripper_target(self, position_rad: float) -> None:
        """Synchronize the right Omnipicker with this 50-Hz pose stream.

        GDK obtains the installed end-effector joint name from ``get_end_state``;
        do not guess it from a URDF or hard-code a model-specific name.
        """
        self.raise_if_failed()
        raw = float(position_rad)
        clipped = clamp_omnipicker(raw)
        if clipped != raw:
            raise SafetyError("Omnipicker target is outside [-pi/4, 0] rad")
        names, current_positions = self._controller.read_gripper_state("right")
        if len(names) != 1 or len(current_positions) != 1:
            raise SafetyError(
                "right_tool 的 GDK end_state 未提供恰好一个可同步控制的夹爪关节"
            )
        with self._lock:
            if self._stop.is_set() or self._thread is None:
                raise RuntimeError("end-effector position stream is not running")
            self._gripper_joint_names = names
            self._gripper_joint_positions = (clipped,)

    def wait_until_target(self, timeout_s: float) -> bool:
        """Wait until actual end-link feedback reaches the current target."""
        deadline = time.monotonic() + float(timeout_s)
        with self._lock:
            requested_generation = self._target_generation
        while True:
            self.raise_if_failed()
            with self._lock:
                arrived = self._arrived_generation >= requested_generation
                actual_at = self._actual_at
                feedback_sequence = self._feedback_sequence
            if arrived:
                age_s = time.monotonic() - actual_at
                if feedback_sequence > 0 and age_s <= self._feedback_stale_timeout_s:
                    return True
                if age_s > self._feedback_stale_timeout_s:
                    raise SafetyError("right end-effector feedback is stale while waiting for actual arrival")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return False
            self._at_target.wait(remaining)

    def is_target_reached(self) -> bool:
        """Return whether actual end-link feedback has reached the target."""
        self.raise_if_failed()
        with self._lock:
            return (
                self._feedback_sequence > 0
                and self._arrived_generation >= self._target_generation
                and time.monotonic() - self._actual_at <= self._feedback_stale_timeout_s
            )

    def wait_until_actual_target(
        self,
        target: CartesianPoseTarget,
        timeout_s: float,
        *,
        position_tolerance_m: float,
        orientation_tolerance_deg: float,
        stable_samples: int = 10,
        poll_period_s: float = 0.02,
    ) -> CartesianPoseTarget | None:
        """Keep streaming until cached GDK feedback is continuously at ``target``."""
        if not 0.0 < position_tolerance_m <= 0.005:
            raise ValueError("actual position tolerance must be in (0, 0.005] m")
        if not 0.0 < orientation_tolerance_deg <= 2.0:
            raise ValueError("actual orientation tolerance must be in (0, 2] deg")
        if not 1 <= int(stable_samples) <= 50 or not 0.01 <= poll_period_s <= 0.10:
            raise ValueError("actual-pose stability parameters are invalid")
        with self._lock:
            requested_generation = self._target_generation
        deadline = time.monotonic() + float(timeout_s)
        last_sequence = -1
        consecutive = 0
        latest: CartesianPoseTarget | None = None
        while time.monotonic() < deadline:
            self.raise_if_failed()
            with self._lock:
                if self._target_generation != requested_generation:
                    raise SafetyError("right end-effector target changed while waiting for actual arrival")
                actual = self._actual
                actual_at = self._actual_at
                sequence = self._feedback_sequence
            status = self._controller.robot.get_motion_control_status()
            # ``mode`` is the instantaneous motion state (0=stop, 1=servo,
            # 2=planning).  It is legitimately 0 immediately after switching
            # control_mode and before the first pose command, so the position
            # stream gates only on the persistent control mode and errors.
            if int(status.mode) not in {0, 1} or int(status.control_mode) != 1 or int(status.error_code) != 0:
                raise SafetyError(
                    "right end-effector actual-arrival wait requires motion mode 0/1 and healthy control_mode=1; "
                    f"actual={{'mode': {int(status.mode)}, 'control_mode': {int(status.control_mode)}, "
                    f"'error_code': {int(status.error_code)}, 'error_msg': '{status.error_msg}'}}"
                )
            if actual is not None:
                latest = actual
                if time.monotonic() - actual_at > self._feedback_stale_timeout_s:
                    raise SafetyError("right end-effector feedback is stale while waiting for actual arrival")
                if sequence != last_sequence:
                    last_sequence = sequence
                    position_error_m, orientation_error_deg = _pose_errors(actual, target)
                    if position_error_m <= position_tolerance_m and orientation_error_deg <= orientation_tolerance_deg:
                        consecutive += 1
                        if consecutive >= stable_samples:
                            return actual
                    else:
                        consecutive = 0
            self._stop.wait(min(float(poll_period_s), self._feedback_period_s))
        return None

    def raise_if_failed(self) -> None:
        guard = getattr(self._controller, 'check_automatic_guard', None)
        if guard is not None:
            guard()
        with self._lock:
            error = self._error
        if error is not None:
            if isinstance(error, AdvanceRightRequested):
                raise error
            raise SafetyError(f"right end-effector position stream stopped: {error}") from error

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        gate = self._controller.motion_gate
        if gate is not None:
            gate.cancel_automatic_thread(thread)
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        feedback_thread = self._feedback_thread
        if feedback_thread is not None and feedback_thread is not threading.current_thread():
            feedback_thread.join(timeout=max(0.2, self._feedback_period_s * 4.0))
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = None
            if self._feedback_thread is None or not self._feedback_thread.is_alive():
                self._feedback_thread = None
        if gate is not None:
            gate.clear_cancelled_thread(thread)

    def _set_error(self, error: Exception) -> None:
        with self._lock:
            if self._error is None:
                self._error = error
        self._stop.set()

    def _refresh_target_before_send(self, proposed: CartesianPoseTarget) -> CartesianPoseTarget:
        """Replace a pre-pause proposal with a measured hold after manual motion."""
        gate = self._controller.motion_gate
        if gate is None:
            return proposed
        generation = gate.pause_generation
        with self._lock:
            if generation == self._gate_generation:
                return proposed
            if self._pending_rebase_generation != generation:
                self._pending_rebase_generation = generation
                self._pending_rebase_feedback_sequence = self._feedback_sequence
                raise MotionRebasePending("waiting for fresh post-pause right feedback")
            actual = self._actual
            actual_at = self._actual_at
            sequence = self._feedback_sequence
            baseline_sequence = self._pending_rebase_feedback_sequence
        if (
            actual is None
            or sequence <= baseline_sequence
            or time.monotonic() - actual_at > self._feedback_stale_timeout_s
        ):
            raise MotionRebasePending("waiting for fresh post-pause right feedback")
        with self._lock:
            self._gate_generation = generation
            self._pending_rebase_generation = None
        return actual

    def _feedback_loop(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            try:
                actual = self._controller.read_end_effector_pose(self._frame_name)
            except Exception as exc:
                self._set_error(exc)
                return
            now = time.monotonic()
            with self._lock:
                self._actual = actual
                self._actual_at = now
                self._feedback_sequence += 1
                desired = self._desired
                generation = self._target_generation
                if desired is not None:
                    position_error_m, orientation_error_deg = _pose_errors(actual, desired)
                    within_tolerance = (
                        position_error_m <= _DEFAULT_ACTUAL_ARRIVAL_POSITION_TOLERANCE_M
                        and orientation_error_deg <= _DEFAULT_ACTUAL_ARRIVAL_ORIENTATION_TOLERANCE_DEG
                    )
                    if within_tolerance:
                        self._actual_arrival_stable_samples += 1
                    else:
                        self._actual_arrival_stable_samples = 0
                        if generation == self._target_generation:
                            self._arrived_generation = min(self._arrived_generation, generation - 1)
                            self._at_target.clear()
                    if (
                        self._actual_arrival_stable_samples >= _DEFAULT_ACTUAL_ARRIVAL_STABLE_SAMPLES
                    ):
                        self._arrived_generation = max(self._arrived_generation, generation)
                        if generation == self._target_generation:
                            self._at_target.set()
            deadline += self._feedback_period_s
            now = time.monotonic()
            if deadline <= now:
                deadline = now + self._feedback_period_s
            self._stop.wait(max(0.0, deadline - now))

    def _run(self) -> None:
        deadline = time.monotonic()
        while not self._stop.is_set():
            with self._lock:
                commanded, desired = self._commanded, self._desired
                actual = self._actual
                actual_at = self._actual_at
                gripper_joint_names = self._gripper_joint_names
                gripper_joint_positions = self._gripper_joint_positions
            if commanded is None or desired is None:
                return
            if actual is None:
                self._stop.wait(min(self._period_s, self._feedback_period_s))
                deadline = time.monotonic()
                continue
            if time.monotonic() - actual_at > self._feedback_stale_timeout_s:
                self._set_error(SafetyError("right end-effector feedback is stale; holding motion stopped"))
                return
            actual_follow_error_m, _ = _pose_errors(actual, commanded)
            current = np.asarray(commanded.position_m, dtype=np.float64)
            wanted = np.asarray(desired.position_m, dtype=np.float64)
            delta = wanted - current
            distance = float(np.linalg.norm(delta))
            max_step_m = self._max_translation_velocity_mps * self._period_s
            current_orientation = np.asarray(commanded.orientation_xyzw, dtype=np.float64)
            desired_orientation = np.asarray(desired.orientation_xyzw, dtype=np.float64)
            orientation_dot = abs(float(np.dot(
                current_orientation / np.linalg.norm(current_orientation),
                desired_orientation / np.linalg.norm(desired_orientation),
            )))
            orientation_angle_rad = 2.0 * math.acos(min(1.0, max(-1.0, orientation_dot)))
            max_angle_step_rad = self._max_angular_velocity_rad_s * self._period_s
            can_advance = actual_follow_error_m <= self._soft_follow_error_m
            if not can_advance:
                next_target = commanded
            else:
                if distance <= max_step_m:
                    next_position = wanted
                else:
                    next_position = current + delta * (max_step_m / distance)
                if not self._allow_orientation_change or orientation_angle_rad <= max_angle_step_rad:
                    next_orientation = desired.orientation_xyzw
                else:
                    next_orientation = quaternion_slerp(
                        commanded.orientation_xyzw, desired.orientation_xyzw,
                        max_angle_step_rad / orientation_angle_rad,
                    )
                if distance <= max_step_m and orientation_angle_rad <= max_angle_step_rad:
                    next_target = desired
                else:
                    next_target = CartesianPoseTarget(
                        tuple(float(value) for value in next_position), next_orientation,
                    )
            try:
                status = self._controller.motion_status_summary()
                if status["mode"] not in {0, 1} or status["control_mode"] != 1 or status["error_code"] != 0:
                    raise SafetyError(
                        "right end-effector position stream requires motion mode 0/1 and healthy control_mode=1; "
                        f"actual={status}"
                    )
                next_target = self._controller._send_right_cartesian_pose(
                    next_target,
                    self._life_time_s,
                    end_effector_joint_names=gripper_joint_names,
                    end_effector_joint_positions=gripper_joint_positions,
                    refresh_target=self._refresh_target_before_send,
                )
            except MotionWriteCancelled:
                return
            except MotionRebasePending:
                self._stop.wait(self._period_s)
                continue
            except Exception as exc:
                self._set_error(exc)
                return
            with self._lock:
                self._commanded = next_target
            deadline += self._period_s
            now = time.monotonic()
            if deadline <= now:
                deadline = now + self._period_s
            self._stop.wait(max(0.0, deadline - now))


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
        motion_gate=None,
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
        self.motion_gate = motion_gate

    def checked_arm_state(self) -> tuple[float, ...]:
        self.check_automatic_guard()
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
        if self.motion_gate is not None:
            self.motion_gate.wait_for_auto()

    def manual_motion(self):
        """Return a context that bypasses an active automatic-motion pause."""
        if self.motion_gate is None:
            return nullcontext()
        return self.motion_gate.manual_motion()

    def _motion_write(self):
        """Reserve the single GDK motion writer for one robot request."""
        self.check_automatic_guard()
        if self.motion_gate is None:
            return nullcontext()
        return self.motion_gate.automatic_motion()

    def check_automatic_guard(self):
        # Background servo/feedback threads must not consume a recovery
        # request. The owning main thread unwinds and stops their streams.
        guard = getattr(self, 'automatic_guard', None)
        if guard is not None and threading.current_thread() is threading.main_thread():
            if self.motion_gate is None or not self.motion_gate.is_paused():
                guard()

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
        if not math.isfinite(velocity_rad_s) or not velocity_rad_s > 0.0:
            raise SafetyError("preset velocity must be a positive finite value")
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
        with self._motion_write():
            self.robot.joint_control_request(request)

    def move_bilateral_preset(
        self,
        positions_rad: Sequence[float],
        *,
        velocity_rad_s: float = 0.08,
        life_time_s: float = 10.0,
        detail: str = "traditional bilateral preset",
    ) -> None:
        """Submit one planned request for all 14 arm joints."""
        self._require_motion()
        if not math.isfinite(velocity_rad_s) or not velocity_rad_s > 0.0:
            raise SafetyError("bilateral preset velocity must be a positive finite value")
        if not math.isfinite(life_time_s) or not 0.1 <= life_time_s <= 30.0:
            raise SafetyError("bilateral preset life_time_s must be in [0.1, 30]")
        target = validate_arm_target(positions_rad)
        request = self.gdk.JointControlReq()
        request.detail = detail
        request.life_time = float(life_time_s)
        request.joint_names = list(ARM_JOINT_NAMES)
        request.joint_positions = list(target)
        request.joint_velocities = [float(velocity_rad_s)] * len(ARM_JOINT_NAMES)
        with self._motion_write():
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
            with self._motion_write():
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
        with self._motion_write():
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
        allow_orientation_change: bool = False,
        max_angular_velocity_rad_s: float = 0.5,
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
            allow_orientation_change=allow_orientation_change,
            max_angular_velocity_rad_s=max_angular_velocity_rad_s,
        )
        stream.start(initial_target)
        return stream

    def start_left_gripper_position_stream(
        self,
        position_rad: float,
        *,
        rate_hz: float = 50.0,
        life_time_s: float = 0.10,
        max_translation_velocity_mps: float = 0.005,
    ) -> LeftEndEffectorPositionStream:
        """Hold the measured left end-link pose while driving its Omnipicker.

        GDK mode-1 pose control accepts the end-effector joint target in the
        same 50-Hz request as the pose.  A short-lived stream avoids relying
        on a standalone ``move_end_effector_joint`` request while a preceding
        arm preset is still settling.
        """
        initial_target = self.read_end_effector_pose("arm_l_end_link")
        stream = self.start_left_end_effector_position_stream(
            initial_target,
            rate_hz=rate_hz,
            life_time_s=life_time_s,
            max_translation_velocity_mps=max_translation_velocity_mps,
        )
        try:
            stream.set_left_gripper_target(position_rad)
        except Exception:
            stream.stop()
            raise
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

    def start_right_end_effector_position_stream(
        self,
        initial_target: CartesianPoseTarget,
        *,
        rate_hz: float = 50.0,
        life_time_s: float = 0.10,
        max_translation_velocity_mps: float = 0.01,
        allow_orientation_change: bool = False,
        max_angular_velocity_rad_s: float = 0.5,
    ) -> RightEndEffectorPositionStream:
        """Start a mode-1 right end-link stream.

        The default keeps the calibrated local-motion invariant of a fixed
        orientation.  Autonomous coarse approach must opt in explicitly and
        supplies a separately bounded angular velocity.
        """
        self._require_motion()
        status = self.motion_status_summary()
        if status["mode"] not in {0, 1} or status["control_mode"] != 1 or status["error_code"] != 0:
            raise SafetyError(
                "right end-effector position stream requires motion mode 0/1 and healthy GDK control_mode=1; "
                f"actual={status}"
            )
        stream = RightEndEffectorPositionStream(
            self, rate_hz=rate_hz, life_time_s=life_time_s,
            max_translation_velocity_mps=max_translation_velocity_mps,
            allow_orientation_change=allow_orientation_change,
            max_angular_velocity_rad_s=max_angular_velocity_rad_s,
        )
        stream.start(initial_target)
        return stream

    def _send_left_cartesian_pose(
        self,
        target: CartesianPoseTarget,
        life_time_s: float,
        *,
        end_effector_joint_names: Sequence[str] = (),
        end_effector_joint_positions: Sequence[float] = (),
        refresh_target: Callable[[CartesianPoseTarget], CartesianPoseTarget] | None = None,
    ) -> CartesianPoseTarget:
        """Internal 50-Hz sender; public callers use the gated stream above."""
        self._require_motion()
        if len(end_effector_joint_names) != len(end_effector_joint_positions):
            raise ValueError("end-effector joint names and positions must have the same length")
        with self._motion_write():
            if refresh_target is not None:
                target = refresh_target(target)
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
        return target

    def _send_right_cartesian_pose(
        self,
        target: CartesianPoseTarget,
        life_time_s: float,
        *,
        end_effector_joint_names: Sequence[str] = (),
        end_effector_joint_positions: Sequence[float] = (),
        refresh_target: Callable[[CartesianPoseTarget], CartesianPoseTarget] | None = None,
    ) -> CartesianPoseTarget:
        """Internal sender for the right mode-1 stream; callers never bypass it."""
        self._require_motion()
        if len(end_effector_joint_names) != len(end_effector_joint_positions):
            raise ValueError("end-effector joint names and positions must have the same length")
        with self._motion_write():
            if refresh_target is not None:
                target = refresh_target(target)
            request = self.gdk.EndEffectorPose()
            request.life_time = float(life_time_s)
            request.group = self.gdk.EndEffectorControlGroup.kRightArm
            pose = request.right_end_effector_pose
            pose.position.x, pose.position.y, pose.position.z = target.position_m
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = target.orientation_xyzw
            if end_effector_joint_names:
                request.end_effector_joint_names = list(end_effector_joint_names)
                request.end_effector_joint_positions = [float(value) for value in end_effector_joint_positions]
            result = self.robot.end_effector_pose_control(request)
        if result not in (None, 0):
            raise SafetyError(f"GDK right end_effector_pose_control failed: {result}")
        return target

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
        with self._motion_write():
            result = self.robot.move_end_effector_joint(
                [clipped], [0.0], f"{side}_tool"
            )
        if result not in (None, 0):
            raise SafetyError(f"GDK move_end_effector_joint failed: {result}")
