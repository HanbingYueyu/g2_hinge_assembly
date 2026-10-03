import unittest
from unittest.mock import Mock, patch

import numpy as np

from control import CartesianPoseTarget


class RightHoleJointMoveTests(unittest.TestCase):
    def test_endpoint_ik_plans_right_joints_in_position_mode_at_requested_speed(self):
        import right_hole_joint_move as motion

        controller = Mock()
        target = CartesianPoseTarget((.4, -.2, .6), (0., 0., 0., 1.))
        controller.read_end_effector_pose.return_value = target
        model = Mock()
        model.solve_pose.return_value = (np.zeros(7), None)
        joints = dict.fromkeys(motion.RIGHT_ARM_JOINT_NAMES, 0.)
        with patch.object(motion, 'UrdfKinematics', return_value=model), \
             patch.object(motion, 'verify_live_model'), \
             patch.object(motion, '_joint_map', return_value=joints):
            actual = motion.move_right_hole_joint_target(controller, target, velocity_rad_s=.1)
        self.assertEqual(actual, target)
        self.assertEqual(controller.move_preset.call_args.args[0], 'right')
        np.testing.assert_allclose(controller.move_preset.call_args.args[1], np.zeros(7))
        self.assertEqual(controller.move_preset.call_args.kwargs['velocity_rad_s'], .1)
        self.assertEqual(controller.move_preset.call_args.kwargs['control_mode'], 1)
        controller.start_right_end_effector_position_stream.assert_not_called()

    def test_failed_ik_never_sends_joint_motion(self):
        import right_hole_joint_move as motion

        controller = Mock()
        model = Mock()
        model.solve_pose.side_effect = ValueError('unreachable')
        with patch.object(motion, 'UrdfKinematics', return_value=model), \
             patch.object(motion, 'verify_live_model'), \
             patch.object(motion, '_joint_map', return_value=dict.fromkeys(motion.RIGHT_ARM_JOINT_NAMES, 0.)):
            with self.assertRaisesRegex(ValueError, 'unreachable'):
                motion.move_right_hole_joint_target(
                    controller, CartesianPoseTarget((.4, -.2, .6), (0., 0., 0., 1.)),
                    velocity_rad_s=.1,
                )
        controller.move_preset.assert_not_called()


if __name__ == '__main__':
    unittest.main()
