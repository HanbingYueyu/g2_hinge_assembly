import unittest
from unittest.mock import Mock, patch
import numpy as np
from control import CartesianPoseTarget


class LeftHoleJointTests(unittest.TestCase):
    def test_endpoint_ik_uses_preset_speed_and_checks_actual_pose(self):
        import left_hole_joint_move as motion
        controller = Mock()
        target = CartesianPoseTarget((.4, .2, .5), (0., 0., 0., 1.))
        controller.read_end_effector_pose.return_value = target
        model = Mock()
        model.solve_pose.return_value = (np.zeros(7), None)
        joints = dict.fromkeys(motion.LEFT_ARM_JOINT_NAMES, 0.)
        with patch.object(motion, 'UrdfKinematics', return_value=model), \
             patch.object(motion, 'verify_live_model'), patch.object(motion, '_joint_map', return_value=joints):
            actual = motion.move_left_hole_joint_target(controller, target, velocity_rad_s=.23)
        self.assertEqual(actual, target)
        self.assertEqual(controller.move_preset.call_args.kwargs['velocity_rad_s'], .23)
        self.assertEqual(controller.move_preset.call_args.kwargs['control_mode'], 3)
        self.assertEqual(controller.move_preset.call_args.args[0], 'left')
        model.solve_pose.assert_called_once()
        controller.start_left_end_effector_position_stream.assert_not_called()

    def test_failed_ik_does_not_move(self):
        import left_hole_joint_move as motion
        controller = Mock()
        model = Mock()
        model.solve_pose.side_effect = ValueError('unreachable')
        with patch.object(motion, 'UrdfKinematics', return_value=model), \
             patch.object(motion, 'verify_live_model'), \
             patch.object(motion, '_joint_map', return_value=dict.fromkeys(motion.LEFT_ARM_JOINT_NAMES, 0.)):
            with self.assertRaisesRegex(ValueError, 'unreachable'):
                motion.move_left_hole_joint_target(controller,
                    CartesianPoseTarget((.4,.2,.5), (0.,0.,0.,1.)), velocity_rad_s=.2)
        controller.move_preset.assert_not_called()

    def test_wrong_actual_orientation_is_not_success(self):
        import left_hole_joint_move as motion
        controller = Mock()
        target = CartesianPoseTarget((.4,.2,.5), (0.,0.,0.,1.))
        controller.read_end_effector_pose.return_value = CartesianPoseTarget(target.position_m, (1.,0.,0.,0.))
        model = Mock()
        model.solve_pose.return_value = (np.zeros(7), None)
        with patch.object(motion, 'UrdfKinematics', return_value=model), \
             patch.object(motion, 'verify_live_model'), \
             patch.object(motion, '_joint_map', return_value=dict.fromkeys(motion.LEFT_ARM_JOINT_NAMES, 0.)):
            with self.assertRaisesRegex(motion.SafetyError, '未到位'):
                motion.move_left_hole_joint_target(controller, target, velocity_rad_s=.2,
                    timeout_s=0., clock=lambda: 0.)
