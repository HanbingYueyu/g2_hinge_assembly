import math
import struct
import unittest

import numpy as np

from control import CartesianPoseTarget, G2Controller, MotionDisabledError, quaternion_angle_deg
from safety import SafetyError
from spacemouse_input import CompactReports, InputFrame
from spacemouse_teleop import TeleopConfig, TeleopSession, integrate_pose
from test_control import FakeGdk, FakePose, FakeRobot


class BilateralRobot(FakeRobot):
    def __init__(self):
        super().__init__()
        self.left_pose = FakePose()
        self.left_pose.position.y = .2

    def get_motion_control_status(self):
        status = super().get_motion_control_status()
        status.frame_names = ['arm_l_end_link', 'arm_r_end_link']
        status.frame_poses = [self.left_pose, self.right_end_pose]
        return status


def frame(now, axes=(0,) * 6, pressed=(), buttons=(False, False), times=None, ready=True):
    return InputFrame(axes, buttons, pressed, times or (now, now), ready)


class TeleopTests(unittest.TestCase):
    def setUp(self):
        self.robot = BilateralRobot()
        self.controller = G2Controller(FakeGdk, self.robot, allow_motion=True)
        self.session = TeleopSession(self.controller, TeleopConfig(side='right', axis_map=(1, 2, 3, 4, 5, 6)))
        self.now = 1.0
        self.tick()

    def tick(self, axes=(0,) * 6, pressed=(), buttons=(False, False), **kwargs):
        self.now += .02
        return self.session.tick(frame(self.now, axes, pressed, buttons, **kwargs), self.now)

    def left_click(self, axes=(0,) * 6):
        self.tick(axes, buttons=(True, False))
        return self.tick(axes, buttons=(False, False))

    def left_double_click(self, axes=(0,) * 6):
        self.left_click(axes)
        return self.left_click(axes)

    def test_start_waits_for_complete_neutral_input_without_commands(self):
        robot = BilateralRobot()
        session = TeleopSession(G2Controller(FakeGdk, robot, allow_motion=True), TeleopConfig())
        session.tick(frame(1, ready=False), 1)
        session.tick(frame(1.02, axes=(1, 0, 0, 0, 0, 0)), 1.02)
        self.assertEqual(robot.requests, [])
        session.tick(frame(1.04), 1.04)
        self.assertEqual(len(robot.requests), 1)

    def test_motion_disabled_gate_is_preserved(self):
        self.controller.allow_motion = False
        count = len(self.robot.requests)
        with self.assertRaises(MotionDisabledError):
            self.tick()
        self.assertEqual(len(self.robot.requests), count)

    def test_diagonal_translation_and_rotation_are_norm_limited(self):
        start = CartesianPoseTarget((0, 0, 0), (0, 0, 0, 1))
        target = integrate_pose(start, (1,) * 6, .02, .01, .2)
        self.assertAlmostEqual(np.linalg.norm(target.position_m), .0002)
        self.assertAlmostEqual(quaternion_angle_deg(start.orientation_xyzw, target.orientation_xyzw), math.degrees(.004))
        self.assertAlmostEqual(np.linalg.norm(target.orientation_xyzw), 1)

    def test_rotation_uses_base_frame_and_normalizes_quaternion(self):
        start = CartesianPoseTarget((0, 0, 0), (0, 0, math.sqrt(2), math.sqrt(2)))
        target = integrate_pose(start, (0, 0, 0, 1, 0, 0), .02, .01, .2)
        # Rx(delta) * Rz(pi/2): quaternion y component is negative.
        self.assertGreater(target.orientation_xyzw[0], 0)
        self.assertLess(target.orientation_xyzw[1], 0)
        self.assertAlmostEqual(np.linalg.norm(target.orientation_xyzw), 1)

    def test_center_cancels_unreached_position_and_rotation_target(self):
        self.robot.follow_right_pose = False
        self.tick((1, 0, 0, 0, 0, 1))
        self.assertGreater(self.robot.requests[-1].right_end_effector_pose.position.x, 0)
        self.tick()
        pose = self.robot.requests[-1].right_end_effector_pose
        self.assertEqual(pose.position.x, 0)
        self.assertEqual(pose.orientation.z, 0)
        self.assertEqual(pose.orientation.w, 1)

    def test_switch_while_deflected_is_rejected_and_not_queued(self):
        self.left_double_click((1, 0, 0, 0, 0, 0))
        self.assertEqual(self.session.side, 'right')

    def test_switch_waits_for_old_lease_and_starts_from_actual_left_pose(self):
        for _ in range(12):
            self.tick()
        self.left_double_click()
        count = len(self.robot.requests)
        for _ in range(4):
            self.tick()
        self.assertEqual(len(self.robot.requests), count)
        for _ in range(5):
            self.tick()
        self.assertEqual(self.session.side, 'left')
        self.assertEqual(self.robot.requests[-1].group, FakeGdk.EndEffectorControlGroup.kLeftArm)
        self.assertEqual(self.robot.requests[-1].left_end_effector_pose.position.y, .2)

    def test_gripper_is_synchronized_and_long_hold_does_not_retoggle(self):
        self.tick(pressed=(1,), buttons=(False, True))
        self.assertEqual(self.robot.requests[-1].end_effector_joint_names, [])
        self.tick()
        request = self.robot.requests[-1]
        self.assertEqual(request.end_effector_joint_names, ['right_omnipicker'])
        self.assertEqual(request.end_effector_joint_positions, [0.0])
        self.tick(buttons=(False, True))
        self.assertEqual(self.robot.requests[-1].end_effector_joint_positions, [0.0])
        self.tick()
        # Advance beyond debounce before the second click.
        for _ in range(15):
            self.tick()
        self.tick(pressed=(1,))
        self.assertEqual(self.robot.requests[-1].end_effector_joint_positions, [-.785])
        self.assertEqual(self.robot.gripper_requests, [])

    def test_right_long_press_never_toggles_gripper_even_on_release(self):
        self.tick(pressed=(1,), buttons=(False, True))
        for _ in range(120):
            self.tick(buttons=(False, True))
            self.assertEqual(self.robot.requests[-1].end_effector_joint_names, [])
        self.tick()
        self.assertEqual(self.robot.requests[-1].end_effector_joint_names, [])

    def test_simultaneous_buttons_do_neither_action(self):
        self.tick(pressed=(0, 1), buttons=(True, True))
        self.assertEqual(self.session.side, 'right')
        self.assertEqual(self.robot.requests[-1].end_effector_joint_names, [])

    def test_single_left_click_does_not_switch(self):
        for _ in range(12):
            self.tick()
        self.left_click()
        for _ in range(10):
            self.tick()
        self.assertEqual(self.session.side, 'right')

    def test_left_double_click_switches_only_after_second_release(self):
        for _ in range(12):
            self.tick()
        self.tick(buttons=(True, False))
        self.tick(buttons=(False, False))
        self.assertEqual(self.session.side, 'right')
        event = self.left_click()
        self.assertEqual(event, 'switching')
        self.assertEqual(self.session.side, 'right')
        for _ in range(8):
            self.tick()
        self.assertEqual(self.session.side, 'left')

    def test_left_double_click_accepts_a_slightly_slower_click(self):
        for _ in range(12):
            self.tick()
        self.left_click()
        for _ in range(20):
            self.tick()
        event = self.left_click()
        self.assertEqual(event, 'switching')
        self.assertEqual(self.session.side, 'right')

    def test_external_gesture_mode_keeps_side_fixed(self):
        self.session = TeleopSession(
            self.controller,
            TeleopConfig(side='right', axis_map=(1, 2, 3, 4, 5, 6)),
            allow_side_switch=False,
        )
        self.tick()
        self.left_double_click()
        for _ in range(8):
            self.tick()
        self.assertEqual(self.session.side, 'right')

    def test_left_button_hold_allows_complete_six_dof(self):
        self.tick((1, 0, 0, 1, 0, 0), buttons=(True, False))
        pose = self.robot.requests[-1].right_end_effector_pose
        self.assertGreater(pose.position.x, 0)
        self.assertNotEqual(pose.orientation.x, 0)

    def test_left_button_release_allows_translation_but_suppresses_rotation(self):
        self.tick((1, 0, 0, 1, 0, 0), buttons=(False, False))
        pose = self.robot.requests[-1].right_end_effector_pose
        self.assertGreater(pose.position.x, 0)
        self.assertEqual(pose.orientation.x, 0)
        self.assertEqual(pose.orientation.w, 1)

    def test_long_left_hold_cannot_complete_a_double_click(self):
        for _ in range(12):
            self.tick()
        self.tick(buttons=(True, False))
        for _ in range(25):
            self.tick(buttons=(True, False))
        self.tick(buttons=(False, False))
        self.left_click()
        for _ in range(10):
            self.tick()
        self.assertEqual(self.session.side, 'right')

    def test_old_nonzero_axes_are_not_kept_alive_by_buttons(self):
        count = len(self.robot.requests)
        with self.assertRaises(SafetyError):
            self.tick((1, 0, 0, 0, 0, 0), pressed=(1,), times=(0.0, self.now))
        self.assertEqual(len(self.robot.requests), count)

    def test_idle_zero_report_can_remain_idle_without_false_timeout(self):
        self.tick(times=(0.0, 0.0))
        self.assertEqual(self.session.side, 'right')

    def test_stalled_feedback_does_not_accumulate_a_far_target(self):
        self.robot.follow_right_pose = False
        for _ in range(100):
            self.tick((1, 0, 0, 0, 0, 0))
        self.assertEqual(len(self.robot.requests), 101)
        self.assertLessEqual(self.robot.requests[-1].right_end_effector_pose.position.x, .00021)

    def test_motion_can_continue_beyond_previous_cumulative_workspace_limit(self):
        for _ in range(800):
            self.tick((1, 0, 0, 0, 0, 0))
        self.assertGreater(self.robot.requests[-1].right_end_effector_pose.position.x, .15)

    def test_bad_mode_and_loop_stall_prevent_further_commands(self):
        count = len(self.robot.requests)
        self.robot.control_mode = 3
        with self.assertRaises(SafetyError):
            self.tick()
        self.assertEqual(len(self.robot.requests), count)
        self.robot.control_mode = 1
        with self.assertRaises(SafetyError):
            self.session.tick(frame(self.now + 1), self.now + 1)
        self.assertEqual(len(self.robot.requests), count)

    def test_stop_replaces_arm_and_gripper_targets_with_actual_and_is_idempotent(self):
        self.robot.follow_right_pose = False
        self.robot.follow_right_gripper = False
        self.tick((1, 0, 0, 0, 0, 0), pressed=(1,))
        self.session.stop()
        request = self.robot.requests[-1]
        self.assertEqual(request.right_end_effector_pose.position.x, 0)
        self.assertEqual(request.end_effector_joint_positions, [-.785])
        count = len(self.robot.requests)
        self.session.stop()
        with self.assertRaises(SafetyError):
            self.tick()
        self.assertEqual(len(self.robot.requests), count)

    def test_invalid_limits_are_rejected(self):
        for values in ({'speed_m_s': float('nan')}, {'angular_speed_rad_s': -1}, {'deadzone': 1}, {'side': 'both'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TeleopConfig(**values)

    def test_default_mapping_push_forward_and_lift_are_positive_base_xz(self):
        self.session = TeleopSession(self.controller, TeleopConfig())
        self.tick()
        self.tick((0, -1, -1, 0, 0, 0))
        pose = self.robot.requests[-1].right_end_effector_pose
        self.assertGreater(pose.position.x, 0)
        self.assertGreater(pose.position.z, 0)

    def test_slow_feedback_read_cannot_publish_stale_input(self):
        self.session.clock = lambda: self.now + .07
        count = len(self.robot.requests)
        with self.assertRaises(SafetyError):
            self.tick((1, 0, 0, 0, 0, 0))
        self.assertEqual(len(self.robot.requests), count)

    def test_nonfinite_axis_timestamp_is_rejected(self):
        count = len(self.robot.requests)
        with self.assertRaises(SafetyError):
            self.tick((1, 0, 0, 0, 0, 0), times=(float('nan'), self.now))
        self.assertEqual(len(self.robot.requests), count)

    def test_switch_lease_begins_after_slow_send_completes(self):
        for _ in range(12):
            self.tick()
        self.session.clock = lambda: self.now + .05
        self.left_double_click()
        count = len(self.robot.requests)
        for _ in range(7):
            self.tick()
        self.assertEqual(len(self.robot.requests), count)
        for _ in range(3):
            self.tick()
        self.assertEqual(self.session.side, 'left')

    def test_switch_click_during_deflection_in_same_hid_batch_is_rejected(self):
        for _ in range(12):
            self.tick()
        reports = CompactReports()
        for channel in (1, 2):
            reports.feed(bytes([channel]) + struct.pack('<hhh', 0, 0, 0), self.now)
        reports.feed(bytes([3, 0]), self.now)
        reports.snapshot()
        self.now += .02
        reports.feed(bytes([1]) + struct.pack('<hhh', 350, 0, 0), self.now)
        reports.feed(bytes([3, 1]), self.now)
        reports.feed(bytes([1]) + struct.pack('<hhh', 0, 0, 0), self.now)
        reports.feed(bytes([3, 0]), self.now)
        reports.feed(bytes([3, 1]), self.now)
        reports.feed(bytes([3, 0]), self.now)
        event = self.session.tick(reports.snapshot(), self.now)
        self.assertEqual(event, 'switch_rejected_not_centered')
        for _ in range(15):
            self.tick()
        self.assertEqual(self.session.side, 'right')

    def test_cancelled_gripper_is_resynced_when_switching_back(self):
        self.robot.follow_right_gripper = False
        self.tick(pressed=(1,))  # Closing target, but physical gripper remains open.
        for _ in range(15):
            self.tick()
        self.left_double_click()
        for _ in range(15):
            self.tick()
        self.assertEqual(self.session.side, 'left')
        self.left_double_click()
        for _ in range(15):
            self.tick()
        self.assertEqual(self.session.side, 'right')
        self.tick(pressed=(1,))
        self.assertEqual(self.robot.requests[-1].end_effector_joint_positions, [0.0])


if __name__ == '__main__':
    unittest.main()
