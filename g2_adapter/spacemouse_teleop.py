"""Single-writer 50-Hz SpaceMouse control; no robot/input worker threads.

This deliberately does not change the fixed-orientation visual-servo streams.
Only the selected arm receives commands. It does not plan collision-free paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

try:
    from .control import CartesianPoseTarget, G2Controller, quaternion_angle_deg
    from .safety import SafetyError
    from .spacemouse_input import InputFrame, parse_axis_map
except ImportError:
    from control import CartesianPoseTarget, G2Controller, quaternion_angle_deg
    from safety import SafetyError
    from spacemouse_input import InputFrame, parse_axis_map


PERIOD_S = .02
LEASE_S = .10
SWITCH_DELAY_S = .12
# HID button reports are sampled in the 50-Hz loop.  A 350-ms window made a
# normal human double click easy to miss once report batching and release
# timing were included.
DOUBLE_CLICK_WINDOW_S = .55
LONG_PRESS_S = .40
OPEN_RAD = -.785
CLOSE_RAD = 0.0


@dataclass(frozen=True)
class TeleopConfig:
    side: str = 'right'
    speed_m_s: float = .01
    angular_speed_rad_s: float = .2
    deadzone: float = .12
    # HID -> ROS convention (forward/left/up). Apply the same proper
    # rotation to translation and axial rotation vectors.
    axis_map: tuple[int, ...] = (-2, -1, -3, -5, -4, -6)
    input_timeout_s: float = .15
    max_position_error_m: float = .01
    max_orientation_error_rad: float = .10

    def __post_init__(self):
        if self.side not in {'left', 'right'}:
            raise ValueError('side 必须是 left 或 right')
        bounds = {
            'speed_m_s': (0, .03), 'angular_speed_rad_s': (0, .5),
            'deadzone': (0, .5), 'input_timeout_s': (.04, .3),
            'max_position_error_m': (0, .02), 'max_orientation_error_rad': (0, .2),
        }
        for name, (low, high) in bounds.items():
            value = getattr(self, name)
            if not math.isfinite(value) or not low < value <= high:
                raise ValueError(f'{name} 必须在 ({low}, {high}]')
        parse_axis_map(','.join(str(value) for value in self.axis_map))


def integrate_pose(pose: CartesianPoseTarget, axes, dt: float, speed: float, angular_speed: float) -> CartesianPoseTarget:
    """Integrate a base-frame twist, with vector-norm (not per-axis) caps."""
    values = np.asarray(axes, dtype=float)
    if values.shape != (6,) or not np.all(np.isfinite(values)) or not 0 <= dt <= .08:
        raise SafetyError('无效六轴输入或积分周期')
    linear, angular = values[:3].copy(), values[3:].copy()
    linear /= max(1.0, float(np.linalg.norm(linear)))
    angular /= max(1.0, float(np.linalg.norm(angular)))
    position = np.asarray(pose.position_m) + linear * speed * dt
    q = np.asarray(pose.orientation_xyzw, dtype=float)
    q /= np.linalg.norm(q)
    rotvec = angular * angular_speed * dt
    angle = float(np.linalg.norm(rotvec))
    if angle > 0:
        vector = rotvec * (math.sin(angle / 2) / angle)
        scalar = math.cos(angle / 2)
        q = np.r_[scalar * q[:3] + q[3] * vector + np.cross(vector, q[:3]),
                  scalar * q[3] - np.dot(vector, q[:3])]
        q /= np.linalg.norm(q)
    return CartesianPoseTarget(tuple(float(v) for v in position), tuple(float(v) for v in q))


class TeleopSession:
    def __init__(
        self,
        controller: G2Controller,
        config: TeleopConfig,
        *,
        clock: Callable[[], float] | None = None,
        allow_side_switch: bool = True,
    ):
        self.controller, self.config, self.clock = controller, config, clock
        self.allow_side_switch = bool(allow_side_switch)
        self.side = config.side
        self.target: CartesianPoseTarget | None = None
        self._gripper_open: dict[str, bool] = {}
        self._gripper_goal: float | None = None
        self._last_time: float | None = None
        self._tick_time: float | None = None
        self._last_click = [-math.inf, -math.inf]
        self._right_down_since: float | None = None
        self._right_was_down = False
        self._left_down_since: float | None = None
        self._left_was_down = False
        self._left_long = False
        self._left_second_click = False
        self._left_pending_click: float | None = None
        self._neutral_since: float | None = None
        self._switch_at: float | None = None
        self._stopped = False
        self._failed = False
        self.event = 'waiting_for_neutral'

    def _healthy(self):
        self.controller.checked_arm_state()
        status = self.controller.motion_status_summary()
        if status['mode'] not in {0, 1} or status['control_mode'] != 1 or status['error_code'] != 0:
            raise SafetyError(f'遥操作需要 mode=0/1、control_mode=1 且无故障: {status}')

    def _actual(self):
        return self.controller.read_end_effector_pose('arm_l_end_link' if self.side == 'left' else 'arm_r_end_link')

    def _gripper(self):
        names, positions = self.controller.read_gripper_state(self.side)
        if len(names) != 1 or len(positions) != 1 or not -.786 <= positions[0] <= .001:
            raise SafetyError(f'{self.side} 夹爪反馈无效；需要一个 Omnipicker 关节')
        return names, positions

    def _reset_left_gesture(self) -> None:
        self._left_down_since = None
        self._left_was_down = False
        self._left_long = False
        self._left_second_click = False
        self._left_pending_click = None

    def _update_left_gesture(self, frame: InputFrame, now: float) -> bool:
        """Track short releases and return true on a completed double click."""
        down = bool(frame.buttons[0])
        double_click = False
        if down and not self._left_was_down:
            self._left_down_since = now
            self._left_long = False
            self._left_second_click = (
                self._left_pending_click is not None
                and now - self._left_pending_click <= DOUBLE_CLICK_WINDOW_S
            )
        if down and self._left_down_since is not None:
            if now - self._left_down_since >= LONG_PRESS_S:
                self._left_long = True
                self._left_pending_click = None
                self._left_second_click = False
        if not down and self._left_was_down:
            duration = now - (self._left_down_since or now)
            if not self._left_long and duration < LONG_PRESS_S:
                if self._left_second_click and self._left_pending_click is not None and now - self._left_pending_click <= DOUBLE_CLICK_WINDOW_S:
                    double_click = True
                    self._left_pending_click = None
                else:
                    self._left_pending_click = now
            else:
                self._left_pending_click = None
            self._left_down_since = None
            self._left_long = False
            self._left_second_click = False

        # A complete press/release can be drained between two 50-Hz polls.
        # ``pressed`` preserves that rising edge even when ``buttons`` is
        # already false in the snapshot.
        instant_presses = frame.pressed.count(0) if not down and not self._left_was_down else 0
        if instant_presses:
            if instant_presses >= 2:
                double_click = True
                self._left_pending_click = None
            elif self._left_pending_click is not None and now - self._left_pending_click <= DOUBLE_CLICK_WINDOW_S:
                double_click = True
                self._left_pending_click = None
            else:
                self._left_pending_click = now
        self._left_was_down = down
        return double_click

    def _send(self, target, *, cancel_gripper=False):
        names, positions = (), ()
        if self._gripper_goal is not None:
            names, actual = self._gripper()
            positions = (min(CLOSE_RAD, max(OPEN_RAD, actual[0])),) if cancel_gripper else (self._gripper_goal,)
        # SDK reads must not cause old HID input to be sent after a long stall.
        if self.clock is not None and self._tick_time is not None and self.clock() - self._tick_time > .06:
            raise SafetyError('GDK 状态读取超时；不发送过期运动目标')
        sender = self.controller._send_left_cartesian_pose if self.side == 'left' else self.controller._send_right_cartesian_pose
        sender(target, LEASE_S, end_effector_joint_names=names, end_effector_joint_positions=positions)

    def _activate(self):
        self.target = self._actual()
        _, positions = self._gripper()
        self._gripper_open[self.side] = positions[0] < OPEN_RAD / 2
        self._gripper_goal = None

    def tick(self, frame: InputFrame, now: float) -> str:
        if self._stopped or self._failed:
            raise SafetyError('遥操作已停止；需要重新启动')
        try:
            return self._tick(frame, now)
        except Exception:
            self._failed = True
            raise

    def _tick(self, frame, now):
        if not math.isfinite(now) or (self._last_time is not None and not 0 < now - self._last_time <= .08):
            raise SafetyError('遥操作循环超时或时钟异常')
        dt = PERIOD_S if self._last_time is None else min(PERIOD_S, now - self._last_time)
        self._last_time = self._tick_time = now
        raw = np.asarray(frame.axes, dtype=float)
        if raw.shape != (6,) or not np.all(np.isfinite(raw)):
            raise SafetyError('六轴输入必须是六个有限数')
        if not frame.ready:
            if self.target is not None:
                raise SafetyError('SpaceMouse 完整输入丢失')
            return self.event
        for group in range(2):
            stamp = frame.axis_times[group]
            if stamp is None or not math.isfinite(stamp) or stamp > now + .001:
                raise SafetyError('SpaceMouse 轴时间戳无效')
            if np.any(np.abs(raw[group * 3:group * 3 + 3]) > self.config.deadzone) and now - stamp > self.config.input_timeout_s:
                raise SafetyError('SpaceMouse 非零轴输入过期')
        mapped = np.array([raw[abs(index) - 1] * (1 if index > 0 else -1) for index in self.config.axis_map])
        # With the left button released, keep the default control to XYZ
        # translation. Holding it opens the three rotational axes for full 6D.
        if not frame.buttons[0]:
            mapped[3:] = 0.0
        axes = np.sign(mapped) * np.maximum(0, np.minimum(1, np.abs(mapped)) - self.config.deadzone) / (1 - self.config.deadzone)
        moving = (bool(np.any(axes[:3])), bool(np.any(axes[3:])))
        neutral = not any(moving)
        # A press followed by a zero report in the same poll must not erase
        # evidence that the operator was deflecting when the button was used.
        batch_centered = neutral and all(value <= self.config.deadzone for value in frame.peak_axes)
        self._neutral_since = (now if self._neutral_since is None else self._neutral_since) if batch_centered else None
        left_double_click = self._update_left_gesture(frame, now) if self.allow_side_switch else False
        if self.target is None:
            if not batch_centered or any(frame.buttons):
                return self.event
            self._healthy()
            self._activate()
            self._reset_left_gesture()
            self._send(self.target)
            self.event = 'ready'
            return self.event

        self._healthy()
        if self._switch_at is not None:
            if now < self._switch_at or not batch_centered or any(frame.buttons):
                self.event = 'switch_wait_neutral'
                return self.event
            self.side = 'left' if self.side == 'right' else 'right'
            self._activate()
            self._switch_at = None
            self._send(self.target)
            self.event = 'switched'
            return self.event

        actual = self._actual()
        position_error_m = float(np.linalg.norm(np.asarray(self.target.position_m) - actual.position_m))
        orientation_error_deg = quaternion_angle_deg(self.target.orientation_xyzw, actual.orientation_xyzw)
        if (position_error_m > self.config.max_position_error_m or
                math.radians(orientation_error_deg) > self.config.max_orientation_error_rad):
            raise SafetyError(
                '末端跟随误差超限；撤销未到达的目标：'
                f'side={self.side}, position_error_mm={position_error_m * 1000:.3f}, '
                f'orientation_error_deg={orientation_error_deg:.3f}, '
                f'position_limit_mm={self.config.max_position_error_m * 1000:.3f}, '
                f'orientation_limit_deg={math.degrees(self.config.max_orientation_error_rad):.3f}, '
                f'target_position_m={self.target.position_m}, actual_position_m={actual.position_m}, '
                f'target_orientation_xyzw={self.target.orientation_xyzw}, '
                f'actual_orientation_xyzw={actual.orientation_xyzw}'
            )
        clicks = set(frame.pressed)
        both = clicks == {0, 1} or all(frame.buttons)
        for button in clicks:
            if button not in (0, 1):
                raise SafetyError('无效按钮编号')
        self.event = 'moving' if not neutral else 'holding'
        if left_double_click:
            if batch_centered and self._neutral_since is not None and now - self._neutral_since >= .15:
                self._send(actual, cancel_gripper=True)
                self.target = actual
                self._switch_at = (self.clock() if self.clock is not None else now) + SWITCH_DELAY_S
                self.event = 'switching'
                return self.event
            self.event = 'switch_rejected_not_centered'
        right_down = bool(frame.buttons[1])
        right_click = False
        if right_down and not self._right_was_down:
            self._right_down_since = now if 1 in clicks and not both else None
        if both:
            self._right_down_since = None
        if not right_down:
            if self._right_was_down:
                right_click = (
                    self._right_down_since is not None
                    and now - self._right_down_since < LONG_PRESS_S
                )
            else:
                # A short press/release may be drained in one HID poll.
                right_click = 1 in clicks and not both
            self._right_down_since = None
        self._right_was_down = right_down
        right_click = right_click and now - self._last_click[1] >= .25
        if right_click:
            self._last_click[1] = now
            self._gripper_open[self.side] = not self._gripper_open[self.side]
            self._gripper_goal = OPEN_RAD if self._gripper_open[self.side] else CLOSE_RAD
            self.event = 'gripper_open' if self._gripper_open[self.side] else 'gripper_close'

        # Integrate one velocity step from the latest measured pose. Using
        # the previous command as the base lets normal servo latency compound
        # into a growing target/feedback error and eventually trips the guard.
        # A measured-pose base keeps this interface velocity-like: a stalled
        # arm receives a bounded one-step target instead of an old backlog.
        base = actual
        target = integrate_pose(base, axes, dt, self.config.speed_m_s, self.config.angular_speed_rad_s)
        if (np.linalg.norm(np.asarray(target.position_m) - actual.position_m) > self.config.max_position_error_m or
                math.radians(quaternion_angle_deg(target.orientation_xyzw, actual.orientation_xyzw)) > self.config.max_orientation_error_rad):
            raise SafetyError('新目标跟随误差超限')
        self._send(target)
        self.target = target
        return self.event

    def stop(self):
        """One measured hold if healthy, then no further calls from this session.

        This is lease cancellation, not a hardware emergency stop. If GDK
        reads/sends fail, the last command is left to expire after LEASE_S.
        """
        if self._stopped:
            return
        self._stopped = True
        if self.target is not None and self._switch_at is None:
            self._tick_time = self.clock() if self.clock is not None else None
            self._healthy()
            self._send(self._actual(), cancel_gripper=True)


class PreviewController:
    """In-memory robot for the default CLI; never imports/initializes GDK."""

    def __init__(self):
        self.poses = {side: CartesianPoseTarget((0, 0, 0), (0, 0, 0, 1)) for side in ('left', 'right')}
        self.grippers = {'left': OPEN_RAD, 'right': OPEN_RAD}

    def checked_arm_state(self):
        return ()

    def motion_status_summary(self):
        return {'mode': 1, 'control_mode': 1, 'error_code': 0}

    def read_end_effector_pose(self, frame):
        return self.poses['left' if frame == 'arm_l_end_link' else 'right']

    def read_gripper_state(self, side):
        return (side + '_preview_gripper',), (self.grippers[side],)

    def _send(self, side, target, positions):
        self.poses[side] = target
        if positions:
            self.grippers[side] = positions[0]

    def _send_left_cartesian_pose(self, target, life_time_s, *, end_effector_joint_names=(), end_effector_joint_positions=()):
        self._send('left', target, end_effector_joint_positions)

    def _send_right_cartesian_pose(self, target, life_time_s, *, end_effector_joint_names=(), end_effector_joint_positions=()):
        self._send('right', target, end_effector_joint_positions)
