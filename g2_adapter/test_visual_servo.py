import unittest

import numpy as np

from visual_servo import (
    base_position_after_delta,
    empirical_horizontal_base_delta_from_pixel_error,
    horizontal_base_delta_from_camera_displacement,
    relief_translation_speed_mps,
    anchored_descent_xy_target,
    ReverseXYMotionGuard,
)


class HorizontalVisualServoTests(unittest.TestCase):
    def test_descent_goal_does_not_follow_opposite_measured_drift(self):
        goal = (0.5, 0.399, 0.7)
        for actual_y in (0.402, 0.406, 0.410):
            goal = anchored_descent_xy_target(goal, (0.5, actual_y, 0.9), (0, -.001, 0))
            self.assertLessEqual(goal[1], .399)
        np.testing.assert_allclose(goal, (0.5, .398, .7))

    def test_descent_goal_advances_with_impedance_tracking_residual(self):
        goal = anchored_descent_xy_target((.5, .398, .7), (.5, .4, .9), (0, -.001, 0))
        np.testing.assert_allclose(goal, (.5, .397, .7))
        held = anchored_descent_xy_target(goal, (.5, .41, .9), (0, -.001, 0))
        np.testing.assert_allclose(held, goal)

    def test_descent_goal_advances_after_robot_catches_up(self):
        goal = anchored_descent_xy_target((.5, .399, .7), (.5, .399, .9), (0, -.001, 0))
        np.testing.assert_allclose(goal, (.5, .398, .7))

    def test_reverse_guard_detects_persistent_opposite_motion(self):
        guard = ReverseXYMotionGuard()
        goal = (.5, .395, .7)
        self.assertFalse(guard.update(0.0, (.5, .4, .9), goal))
        self.assertFalse(guard.update(.4, (.5, .402, .9), goal))
        self.assertTrue(guard.update(.9, (.5, .404, .9), goal))

    def test_reverse_guard_accepts_forward_motion_and_direction_changes(self):
        guard = ReverseXYMotionGuard()
        self.assertFalse(guard.update(0, (.5, .4, .9), (.5, .395, .7)))
        self.assertFalse(guard.update(.9, (.5, .397, .9), (.5, .395, .7)))
        self.assertFalse(guard.update(1, (.5, .397, .9), (.5, .405, .7)))
        self.assertFalse(guard.update(1.9, (.5, .4, .9), (.5, .405, .7)))

    def test_identity_camera_preserves_expected_horizontal_direction(self):
        result = horizontal_base_delta_from_camera_displacement(
            (0.002, -0.001, 0.0), (0.01, -0.02, 0.20),
            fx=400.0, fy=400.0, base_from_camera_rotation=np.eye(3), max_step_m=0.003,
        )
        np.testing.assert_allclose(result, (0.002, -0.001, 0.0), atol=1e-10)

    def test_result_is_strictly_horizontal_for_rotated_camera(self):
        angle = np.deg2rad(35.0)
        rotation = np.asarray(((1.0, 0.0, 0.0), (0.0, np.cos(angle), -np.sin(angle)), (0.0, np.sin(angle), np.cos(angle))))
        result = horizontal_base_delta_from_camera_displacement(
            (0.003, 0.001, 0.0), (0.02, 0.01, 0.25),
            fx=400.0, fy=400.0, base_from_camera_rotation=rotation, max_step_m=0.002,
        )
        self.assertEqual(float(result[2]), 0.0)
        self.assertLessEqual(float(np.linalg.norm(result[:2])), 0.0020000001)


class EmpiricalHorizontalVisualServoTests(unittest.TestCase):
    # Left D405 mapping measured at P3 on 2026-08-28.  Units are px/m.
    J_PX_PER_M = np.asarray(
        ((-334.7421230552792, -2016.4544644817508), (1905.076335706773, -152.0325590824015)),
        dtype=np.float64,
    )

    def test_pixel_error_uses_measured_inverse_mapping_and_opposes_error(self):
        result = empirical_horizontal_base_delta_from_pixel_error(
            (1.0, 0.0), self.J_PX_PER_M, max_step_m=0.002,
        )
        np.testing.assert_allclose(result, (0.0000390589085, 0.0004894359657, 0.0), atol=1e-12)
        np.testing.assert_allclose(self.J_PX_PER_M @ result[:2], (-1.0, 0.0), atol=1e-9)

    def test_empirical_correction_keeps_horizontal_step_bound(self):
        result = empirical_horizontal_base_delta_from_pixel_error(
            (100.0, 100.0), self.J_PX_PER_M, max_step_m=0.002,
        )
        self.assertEqual(float(result[2]), 0.0)
        self.assertLessEqual(float(np.linalg.norm(result[:2])), 0.0020000001)


class ReliefSpeedTests(unittest.TestCase):
    def test_relief_uses_half_of_automatic_translation_speed(self):
        self.assertAlmostEqual(relief_translation_speed_mps(0.030), 0.015)

    def test_relief_rejects_non_positive_speed(self):
        with self.assertRaises(ValueError):
            relief_translation_speed_mps(0.0)


class EndLinkTargetTests(unittest.TestCase):
    def test_base_target_is_current_end_link_position_plus_base_delta(self):
        result = base_position_after_delta((0.61, 0.22, 1.03), (0.002, -0.001, -0.005))
        np.testing.assert_allclose(result, (0.612, 0.219, 1.025), atol=1e-12)
