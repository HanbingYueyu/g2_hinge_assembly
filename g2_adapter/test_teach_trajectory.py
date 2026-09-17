import unittest
import math

from safety import LEFT_ARM_JOINT_NAMES, SafetyError
from teach_trajectory import (
    TrajectoryError,
    build_trajectory,
    iter_smooth_targets,
    load_trajectory,
    trajectory_document,
)


class TrajectoryModelTests(unittest.TestCase):
    def test_trim_leading_stationary_keeps_preceding_frame_and_later_pauses(self):
        from teach_trajectory import trim_leading_stationary
        trajectory = build_trajectory("left", [(0, [0]*7), (10, [.001]*7), (20, [.002]*7), (30, [.01]*7), (40, [.01]*7), (50, [.01]*7)])
        trimmed, removed = trim_leading_stationary(trajectory)
        self.assertEqual(removed, 2)
        self.assertEqual([s.t_ns for s in trimmed.samples], [0, 10, 20, 30])
        self.assertEqual([s.joint_positions_rad for s in trimmed.samples], [(0.002,)*7, (.01,)*7, (.01,)*7, (.01,)*7])

    def test_trim_uses_any_joint_and_strict_half_degree_threshold(self):
        from teach_trajectory import trim_leading_stationary
        threshold = math.radians(.5)
        trajectory = build_trajectory("right", [(0, [0]*7), (10, [0]*6 + [threshold]), (20, [0]*6 + [-threshold*1.1])])
        trimmed, removed = trim_leading_stationary(trajectory)
        self.assertEqual(removed, 1)
        self.assertEqual(trimmed.samples[0].joint_positions_rad[-1], threshold)

    def test_trim_all_stationary_or_single_sample_is_unchanged(self):
        from teach_trajectory import trim_leading_stationary
        for data in ([(0, [0]*7)], [(0, [0]*7), (10, [.001]*7)]):
            trajectory = build_trajectory("left", data)
            self.assertEqual(trim_leading_stationary(trajectory), (trajectory, 0))

    def test_build_trajectory_preserves_selected_arm_and_relative_times(self):
        trajectory = build_trajectory(
            "left",
            [(0, [0.0] * 7), (10_000_000, [0.1] * 7)],
        )

        self.assertEqual(trajectory.side, "left")
        self.assertEqual(trajectory.joint_names, LEFT_ARM_JOINT_NAMES)
        self.assertEqual(trajectory.samples[1].t_ns, 10_000_000)
        self.assertEqual(trajectory.samples[1].joint_positions_rad, (0.1,) * 7)

    def test_load_trajectory_rejects_non_monotonic_sample_time(self):
        document = {
            "schema_version": 1,
            "side": "left",
            "joint_names": list(LEFT_ARM_JOINT_NAMES),
            "samples": [
                {"t_ns": 0, "joint_positions_rad": [0.0] * 7},
                {"t_ns": 0, "joint_positions_rad": [0.1] * 7},
            ],
        }

        with self.assertRaisesRegex(TrajectoryError, "strictly increasing"):
            load_trajectory(document)

    def test_load_trajectory_rejects_out_of_limit_selected_joint(self):
        document = {
            "schema_version": 1,
            "side": "left",
            "joint_names": list(LEFT_ARM_JOINT_NAMES),
            "samples": [{"t_ns": 0, "joint_positions_rad": [4.0] + [0.0] * 6}],
        }

        with self.assertRaises(SafetyError):
            load_trajectory(document)

    def test_document_round_trips_complete_trajectory(self):
        original = build_trajectory("left", [(0, [0.0] * 7), (5, [0.2] * 7)])

        restored = load_trajectory(trajectory_document(original))

        self.assertEqual(restored, original)

    def test_smooth_targets_start_end_and_scale_relative_time(self):
        trajectory = build_trajectory(
            "right",
            [(0, [0.0] * 7), (1_000_000_000, [1.0] * 7)],
        )

        targets = list(iter_smooth_targets(trajectory, rate_hz=50.0, speed_scale=0.2))

        self.assertEqual(targets[0].t_ns, 0)
        self.assertEqual(targets[0].joint_positions_rad, (0.0,) * 7)
        self.assertEqual(targets[-1].t_ns, 5_000_000_000)
        self.assertEqual(targets[-1].joint_positions_rad, (1.0,) * 7)
        self.assertEqual(len(targets), 251)

    def test_smooth_targets_follow_quintic_curve_not_linear_midpoint(self):
        trajectory = build_trajectory(
            "right",
            [(0, [0.0] * 7), (1_000_000_000, [1.0] * 7)],
        )

        targets = list(iter_smooth_targets(trajectory, rate_hz=4.0, speed_scale=1.0))

        self.assertAlmostEqual(targets[1].joint_positions_rad[0], 0.103515625)


if __name__ == "__main__":
    unittest.main()
