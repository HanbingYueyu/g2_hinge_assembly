import unittest

import numpy as np

from bilateral_servo import (
    AlignmentPolicy,
    BilateralAlignmentGate,
    BilateralMetricAlignmentGate,
    MetricAlignmentPolicy,
    metric_and_pixel_corrections_agree,
)


class BilateralAlignmentGateTests(unittest.TestCase):
    def setUp(self):
        self.jacobian = np.eye(2) * 1000.0
        self.policy = AlignmentPolicy(
            pixel_tolerance_px=5.0,
            stable_frames=3,
            max_pair_jump_px=50.0,
            max_error_px=200.0,
            max_step_m=0.001,
            max_total_displacement_m=0.010,
            max_total_travel_m=0.002,
            max_steps=10,
            min_confidence=0.25,
        )

    def _update(self, *, error=(20.0, 0.0), can_move=False, **kwargs):
        self.sequence = getattr(self, "sequence", 0) + 1
        hole = (100.0, 100.0)
        target = (hole[0] + error[0], hole[1] + error[1])
        return self.gate.update(
            sequence=self.sequence,
            hole_center_px=hole,
            target_center_px=target,
            hole_confidence=0.9,
            target_confidence=0.9,
            can_move=can_move,
            **kwargs,
        )

    def test_requires_three_stable_frames_before_motion(self):
        self.gate = BilateralAlignmentGate(self.jacobian, self.policy)
        self.assertEqual(self._update().state, "waiting_stability")
        self.assertEqual(self._update().state, "waiting_stability")
        decision = self._update(can_move=True)
        self.assertEqual(decision.state, "move")
        np.testing.assert_allclose(decision.accepted_delta_m, (-0.001, 0.0, 0.0))

    def test_missing_or_multiple_candidates_holds_without_old_error(self):
        self.gate = BilateralAlignmentGate(self.jacobian, self.policy)
        self._update()
        self.sequence += 1
        decision = self.gate.update(sequence=self.sequence, hole_center_px=None, target_center_px=None, hole_count=0, target_count=0)
        self.assertEqual(decision.state, "hold")
        self.assertEqual(decision.reason, "require_exactly_one_hole_and_target")
        self.assertEqual(decision.accepted_delta_m, (0.0, 0.0, 0.0))

    def test_in_flight_frame_does_not_accumulate_motion(self):
        self.gate = BilateralAlignmentGate(self.jacobian, self.policy)
        for _ in range(3):
            self._update()
        first = self._update(can_move=True)
        second = self._update(can_move=False)
        self.assertEqual(first.state, "move")
        self.assertEqual(second.state, "ready")
        self.assertAlmostEqual(self.gate.cumulative_distance_m, 0.001)

    def test_cumulative_limit_is_path_length_not_net_displacement(self):
        self.gate = BilateralAlignmentGate(self.jacobian, self.policy)
        for _ in range(3):
            self._update(error=(20.0, 0.0))
        first = self._update(error=(20.0, 0.0), can_move=True)
        self.assertEqual(first.state, "move")
        self._update(error=(-20.0, 0.0))
        second = self._update(error=(-20.0, 0.0), can_move=True)
        self.assertEqual(second.state, "move")
        self.assertAlmostEqual(self.gate.cumulative_distance_m, 0.002)
        self._update(error=(20.0, 0.0))
        limited = self._update(error=(20.0, 0.0), can_move=True)
        self.assertEqual(limited.state, "limit_hold")

    def test_aligned_requires_stability_and_stops(self):
        self.gate = BilateralAlignmentGate(self.jacobian, self.policy)
        for _ in range(4):
            decision = self._update(error=(2.0, 1.0))
            self.assertIn(decision.state, {"waiting_stability", "waiting_alignment"})
        decision = self._update(error=(2.0, 1.0), can_move=True)
        self.assertEqual(decision.state, "aligned")
        self.assertEqual(decision.accepted_delta_m, (0.0, 0.0, 0.0))

    def test_jump_restarts_stability(self):
        self.gate = BilateralAlignmentGate(self.jacobian, self.policy)
        self._update()
        self._update()
        decision = self.gate.update(
            sequence=self.sequence + 1,
            hole_center_px=(200.0, 100.0), target_center_px=(220.0, 100.0),
            hole_confidence=0.9, target_confidence=0.9,
        )
        self.assertEqual(decision.state, "waiting_stability")
        self.assertEqual(decision.stable_count, 1)

    def test_duplicate_sequence_does_not_advance_stability(self):
        self.gate = BilateralAlignmentGate(self.jacobian, self.policy)
        decision = self.gate.update(
            sequence=1, hole_center_px=(100.0, 100.0), target_center_px=(120.0, 100.0),
        )
        self.assertEqual(decision.stable_count, 1)
        duplicate = self.gate.update(
            sequence=1, hole_center_px=(100.0, 100.0), target_center_px=(120.0, 100.0),
        )
        self.assertEqual(duplicate.reason, "duplicate_or_old_observation")
        self.assertEqual(duplicate.stable_count, 1)

    def test_bad_jacobian_is_rejected(self):
        with self.assertRaises(ValueError):
            BilateralAlignmentGate(((1.0, 0.0), (0.0, 0.001)), self.policy)


class BilateralMetricAlignmentGateTests(unittest.TestCase):
    def setUp(self):
        self.policy = MetricAlignmentPolicy(
            position_tolerance_m=0.002,
            stable_frames=3,
            aligned_frames=3,
            max_pair_jump_m=0.02,
            max_error_m=0.10,
            max_step_m=0.001,
            max_total_displacement_m=0.003,
            max_total_travel_m=0.006,
        )
        self.gate = BilateralMetricAlignmentGate(self.policy)
        self.sequence = 0

    def update(self, hole=(0.0, 0.0, 0.5), target=(0.004, 0.0, 0.5), **kwargs):
        self.sequence += 1
        return self.gate.update(
            sequence=self.sequence,
            hole_position_m=hole,
            target_position_m=target,
            **kwargs,
        )

    def test_metric_gate_moves_hole_minus_target_without_z(self):
        for _ in range(3):
            self.assertIn(self.update().state, {"waiting_stability", "ready"})
        decision = self.update(can_move=True)
        self.assertEqual(decision.state, "move")
        self.assertAlmostEqual(decision.accepted_delta_m[0], -0.001)
        self.assertAlmostEqual(decision.accepted_delta_m[2], 0.0)

    def test_metric_gate_requires_exact_pair_and_finite_points(self):
        decision = self.update(hole=None, target=None, hole_count=0, target_count=0)
        self.assertEqual(decision.state, "hold")
        self.assertEqual(decision.reason, "require_exactly_one_hole_and_target")

    def test_metric_gate_aligned_after_stable_frames(self):
        for _ in range(3):
            decision = self.update(target=(0.001, 0.001, 0.5))
        self.assertEqual(decision.state, "aligned")

    def test_pixel_metric_veto_rejects_opposite_direction(self):
        self.assertFalse(metric_and_pixel_corrections_agree((0.001, 0.0, 0.0), (-0.001, 0.0, 0.0)))
        self.assertTrue(metric_and_pixel_corrections_agree((0.001, 0.0, 0.0), (0.0012, 0.0, 0.0)))


if __name__ == "__main__":
    unittest.main()
