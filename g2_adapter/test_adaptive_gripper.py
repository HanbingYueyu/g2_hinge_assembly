import io
import json
import unittest

from left_grasp_gdk import (
    AdaptiveGripperCloseParameters,
    SnapshotVisualMailbox,
    compact_record,
    decide_gripper_close,
    held_part_monitor_failure,
    json_record,
    stalled_gripper_position,
)


class AdaptiveGripperTests(unittest.TestCase):
    def test_frame_jsonl_uses_the_compact_runtime_summary(self):
        record = {
            "event": "frame",
            "frame": 3,
            "state": "align_xy",
            "reason": "xy_correction",
            "gripper_position_rad": 0.0,
            "candidate_diagnostics": [{"large": "discarded"}],
            "selected": {"center_px": [10.0, 20.0], "area_px": 999, "depth_median_m": 0.2},
        }
        stream = io.StringIO()
        json_record(stream, record)
        stored = json.loads(stream.getvalue())
        self.assertEqual(stored, compact_record(record))
        self.assertNotIn("candidate_diagnostics", stored)
        self.assertNotIn("area_px", stored["target"])
        self.assertEqual(stored["gripper_position_rad"], 0.0)

    def test_held_part_monitor_rejects_loss_or_geometry_change(self):
        common = {
            "reference_center_px": [320.0, 240.0],
            "reference_area_px": 1000,
            "max_center_drift_px": 12.0,
            "max_area_change_fraction": 0.10,
        }
        self.assertIsNone(held_part_monitor_failure(
            {"valid": True, "center_px": [325.0, 242.0], "area_px": 1050}, **common,
        ))
        self.assertEqual(
            held_part_monitor_failure({"valid": False, "reason": "locked_target_lost"}, **common)["reason"],
            "locked_target_lost",
        )
        self.assertEqual(
            held_part_monitor_failure(
                {"valid": True, "center_px": [340.0, 240.0], "area_px": 1000}, **common,
            )["reason"],
            "held_target_geometry_changed",
        )
        self.assertIsNone(held_part_monitor_failure({"valid": None}, **common))

    def test_held_part_monitor_can_wait_for_loss_timeout_without_geometry_reject(self):
        common = {
            "reference_center_px": [320.0, 240.0],
            "reference_area_px": 1000,
            "max_center_drift_px": 12.0,
            "max_area_change_fraction": 0.10,
            "check_geometry": False,
        }
        self.assertIsNone(held_part_monitor_failure(
            {"valid": True, "center_px": [360.0, 240.0], "area_px": 1500}, **common,
        ))

    def test_missing_frame_is_counted_once_and_then_confirmed(self):
        class FakePerception:
            def __init__(self):
                self.snapshot = {
                    "sequence": 1,
                    "candidates": [],
                    "statistics": [],
                }

            def newer_than(self, sequence, timeout_s=0.0):
                return None if sequence >= self.snapshot["sequence"] else self.snapshot

        mailbox = SnapshotVisualMailbox(
            FakePerception(), initial_center_px=[320.0, 240.0],
            track_max_distance_px=120.0, target_lost_frames=2,
        )
        pending = mailbox.newer_than(0)
        self.assertEqual(pending["reason"], "locked_target_missing_pending")
        self.assertIsNone(mailbox.newer_than(pending["sequence"]))

        mailbox.perception.snapshot = {"sequence": 2, "candidates": [], "statistics": []}
        confirmed = mailbox.newer_than(pending["sequence"])
        self.assertEqual(confirmed["reason"], "locked_target_lost")

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

    def test_full_close_is_held_for_visual_verification(self):
        params = AdaptiveGripperCloseParameters(
            seek_target_rad=0.0,
            minimum_close_travel_rad=0.1,
            stall_window_s=0.5,
            stall_position_delta_rad=0.005,
            additional_close_rad=0.03,
        )
        decision = decide_gripper_close(
            position_rad=0.0,
            gripper_open_rad=-0.51025,
            tolerance_rad=0.035,
            parameters=params,
            samples=[(0.0, -0.02), (0.25, 0.0), (0.50, 0.0)],
            now=0.50,
        )
        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.reason, "full_close_hold_pending_visual_verification")
        self.assertEqual(decision.hold_target_rad, 0.0)


if __name__ == "__main__":
    unittest.main()
