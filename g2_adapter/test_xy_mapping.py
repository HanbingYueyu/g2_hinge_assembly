import unittest

import numpy as np

from xy_mapping import (
    choose_candidate_by_left_rank,
    choose_locked_candidate,
    pixel_jacobian_from_end_pose_trials,
    planned_base_displacement_is_safe,
    single_axis_base_displacement,
)


class SingleAxisBaseDisplacementTests(unittest.TestCase):
    def test_x_positive_is_one_millimetre_along_base_x_only(self):
        result = single_axis_base_displacement("x", "positive", 0.001)
        np.testing.assert_allclose(result, (0.001, 0.0, 0.0), atol=1e-12)

    def test_y_negative_is_one_millimetre_along_base_y_only(self):
        result = single_axis_base_displacement("y", "negative", 0.001)
        np.testing.assert_allclose(result, (0.0, -0.001, 0.0), atol=1e-12)

    def test_invalid_step_is_rejected_before_any_servo_target_can_be_built(self):
        with self.assertRaises(ValueError):
            single_axis_base_displacement("x", "positive", 0.0)


class LockedCandidateTests(unittest.TestCase):
    def test_nearest_candidate_inside_lock_radius_is_retained(self):
        statistics = [
            {"center_px": [120.0, 80.0]},
            {"center_px": [105.0, 102.0]},
            {"center_px": [400.0, 600.0]},
        ]
        result = choose_locked_candidate(
            statistics, [0, 1, 2], reference_center_px=(100.0, 100.0), max_distance_px=30.0
        )
        self.assertEqual(result, 1)

    def test_candidate_outside_lock_radius_is_not_reacquired(self):
        statistics = [{"center_px": [145.0, 100.0]}]
        result = choose_locked_candidate(
            statistics, [0], reference_center_px=(100.0, 100.0), max_distance_px=40.0
        )
        self.assertIsNone(result)


class CandidateLeftRankTests(unittest.TestCase):
    def test_second_from_left_uses_pixel_order_not_sam_enumeration_order(self):
        statistics = [
            {"center_px": [830.0, 500.0]},
            {"center_px": [230.0, 500.0]},
            {"center_px": [520.0, 500.0]},
        ]
        self.assertEqual(choose_candidate_by_left_rank(statistics, [0, 1, 2], rank=2), 2)

    def test_missing_second_candidate_returns_none(self):
        statistics = [{"center_px": [230.0, 500.0]}]
        self.assertIsNone(choose_candidate_by_left_rank(statistics, [0], rank=2))


class EndPosePixelJacobianTests(unittest.TestCase):
    def test_uses_actual_cross_axis_motion_not_nominal_requests(self):
        result = pixel_jacobian_from_end_pose_trials(
            x_actual_base_delta_m=(0.001, 0.0002, 0.0),
            y_actual_base_delta_m=(-0.0001, 0.001, 0.0),
            x_pixel_delta_px=(2.0, 4.0),
            y_pixel_delta_px=(-3.0, 1.0),
        )
        actual_xy = np.asarray(((0.001, -0.0001), (0.0002, 0.001)))
        pixel_delta = np.asarray(((2.0, -3.0), (4.0, 1.0)))
        np.testing.assert_allclose(result @ actual_xy, pixel_delta, atol=1e-12)

    def test_rejects_parallel_actual_trials(self):
        with self.assertRaises(ValueError):
            pixel_jacobian_from_end_pose_trials(
                x_actual_base_delta_m=(0.001, 0.0, 0.0),
                y_actual_base_delta_m=(0.002, 0.0, 0.0),
                x_pixel_delta_px=(2.0, 1.0),
                y_pixel_delta_px=(4.0, 2.0),
            )


class PlannedDisplacementSafetyTests(unittest.TestCase):
    def test_planned_displacement_must_remain_in_requested_horizontal_direction(self):
        self.assertTrue(
            planned_base_displacement_is_safe(
                requested_base_delta_m=(0.001, 0.0, 0.0),
                planned_base_delta_m=(0.0008, 0.0001, 0.0),
                min_fraction=0.25,
            )
        )

    def test_reversed_planned_displacement_is_rejected(self):
        self.assertFalse(
            planned_base_displacement_is_safe(
                requested_base_delta_m=(0.001, 0.0, 0.0),
                planned_base_delta_m=(-0.0002, 0.0, 0.0),
                min_fraction=0.25,
            )
        )

    def test_ten_micrometre_vertical_ik_residual_is_accepted(self):
        self.assertTrue(
            planned_base_displacement_is_safe(
                requested_base_delta_m=(0.001, 0.0, 0.0),
                planned_base_delta_m=(0.000955, -0.000008, 0.000010),
                min_fraction=0.25,
            )
        )

    def test_vertical_residual_above_fifty_micrometres_is_rejected(self):
        self.assertFalse(
            planned_base_displacement_is_safe(
                requested_base_delta_m=(0.001, 0.0, 0.0),
                planned_base_delta_m=(0.000955, -0.000008, 0.000051),
                min_fraction=0.25,
            )
        )

if __name__ == "__main__":
    unittest.main()
