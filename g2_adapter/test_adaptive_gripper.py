import unittest

from left_grasp_gdk import stalled_gripper_position


class AdaptiveGripperTests(unittest.TestCase):
    def test_reports_stall_only_after_full_stable_window(self):
        samples = [(0.0, -0.31), (0.25, -0.308), (0.50, -0.307)]
        self.assertAlmostEqual(
            stalled_gripper_position(
                samples,
                now=0.50,
                window_s=0.50,
                position_delta_rad=0.005,
            ),
            -0.307,
        )

    def test_reports_stall_with_sparse_feedback_after_window_boundary(self):
        # This mirrors the live SAM loop: samples arrive about every 0.23 s.
        # The sample at t=0.00 is the baseline immediately before the 0.50 s
        # stability window ending at t=0.69.
        samples = [(0.00, -0.273), (0.23, -0.273), (0.46, -0.273), (0.69, -0.273)]
        self.assertAlmostEqual(
            stalled_gripper_position(
                samples,
                now=0.69,
                window_s=0.50,
                position_delta_rad=0.005,
            ),
            -0.273,
        )

    def test_rejects_moving_or_too_short_feedback_history(self):
        self.assertIsNone(
            stalled_gripper_position(
                [(0.20, -0.5), (0.50, -0.4)],
                now=0.50,
                window_s=0.50,
                position_delta_rad=0.005,
            )
        )
        self.assertIsNone(
            stalled_gripper_position(
                [(0.0, -0.31), (0.25, -0.25), (0.50, -0.20)],
                now=0.50,
                window_s=0.50,
                position_delta_rad=0.005,
            )
        )


if __name__ == "__main__":
    unittest.main()
