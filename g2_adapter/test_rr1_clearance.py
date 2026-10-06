import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from control import CartesianPoseTarget
import right_post_grasp_sequence as sequence


class RR1ClearanceTests(unittest.TestCase):
    def test_lift_is_based_on_current_pose_and_precedes_rr1(self):
        pose = CartesianPoseTarget((.4, -.2, .7), (0., 0., 0., 1.))
        controller = Mock()
        controller.read_end_effector_pose.return_value = pose
        events = []
        def lift(_controller, target, **kwargs):
            self.assertAlmostEqual(target.position_m[2], .8)
            self.assertEqual(target.position_m[:2], pose.position_m[:2])
            self.assertEqual(target.orientation_xyzw, pose.orientation_xyzw)
            self.assertEqual(kwargs['velocity_rad_s'], .2)
            self.assertEqual(kwargs['control_mode'], 3)
            events.append('lift')
        def rr1(*_args, **kwargs):
            events.append('RR1')
            return {'preset': 'RR1'}
        with patch('right_hole_joint_move.move_right_hole_joint_target', side_effect=lift), \
             patch('right_post_grasp_sequence.move_right_preset', side_effect=rr1):
            sequence.move_right_after_current_lift(controller, [0.] * 7, preset_name="LR2", velocity_rad_s=.2)
        self.assertEqual(events, ['lift', 'RR1'])

    def test_failed_lift_never_sends_rr1_target(self):
        controller = Mock()
        controller.read_end_effector_pose.return_value = CartesianPoseTarget((.4, -.2, .7), (0., 0., 0., 1.))
        with patch('right_hole_joint_move.move_right_hole_joint_target', side_effect=RuntimeError('not arrived')), \
             patch('right_post_grasp_sequence.move_right_preset') as rr1:
            with self.assertRaises(RuntimeError):
                sequence.move_right_after_current_lift(controller, [0.] * 7, preset_name="LR2", velocity_rad_s=.2)
        rr1.assert_not_called()
