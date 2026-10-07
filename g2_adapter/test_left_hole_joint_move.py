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

class RobustEndpointTests(unittest.TestCase):
    def test_retry_changes_seed_and_keeps_accuracy_requirements(self):
        import left_hole_joint_move as motion
        from g2_kinematics import KinematicsError
        model = Mock()
        model.solve_pose.side_effect = [KinematicsError('local minimum'), (np.zeros(7), None)]
        with patch.object(motion, '_LAST_SOLUTION', None):
            result = motion.solve_endpoint(model, Mock(), {}, np.eye(4), np.zeros(7))
        self.assertEqual(model.solve_pose.call_count, 2)
        self.assertFalse(np.array_equal(model.solve_pose.call_args_list[0].args[-1],
                                       model.solve_pose.call_args_list[1].args[-1]))
        self.assertEqual(model.solve_pose.call_args.kwargs['position_tolerance_m'], .001)
        self.assertTrue(model.solve_pose.call_args.kwargs['adaptive_steps'])

    def test_adaptive_solver_reaches_known_reachable_urdf_target(self):
        import json
        from pathlib import Path
        import left_hole_joint_move as motion
        model = motion.UrdfKinematics(motion.DEFAULT_URDF.resolve())
        root = Path(__file__).resolve().parents[1]
        q = np.asarray(json.loads((root/'g2_adapter/recorded_poses.json').read_text())['poses']['left_p4']['joint_positions_rad'])
        state = dict(zip(motion.LEFT_ARM_JOINT_NAMES, q))
        goal = model.forward('base_link', 'arm_l_end_link', state)
        solution, diagnostics = model.solve_pose('base_link', 'arm_l_end_link', state,
            motion.LEFT_ARM_JOINT_NAMES, goal, q + .08,
            joint_limits=motion.ARM_POSITION_LIMITS[:7], reference_weight=0.,
            adaptive_steps=True, position_tolerance_m=.001,
            orientation_tolerance_rad=np.radians(1.))
        actual = model.forward('base_link', 'arm_l_end_link', dict(zip(motion.LEFT_ARM_JOINT_NAMES, solution)))
        self.assertLessEqual(np.linalg.norm(actual[:3,3]-goal[:3,3]), .001)
        self.assertLessEqual(np.linalg.norm(motion.rotation_error(goal[:3,:3], actual[:3,:3])), np.radians(1.))
