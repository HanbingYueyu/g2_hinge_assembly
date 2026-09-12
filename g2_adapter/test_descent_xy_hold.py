import unittest
from types import SimpleNamespace
import numpy as np
from left_grasp_gdk import SnapshotVisualMailbox


class DescentXYHoldTests(unittest.TestCase):
    def test_monitor_only_rejects_loss_but_never_corrects(self):
        item = {"center_px": [100, 100], "area_px": 1000,
                "step_pose": {"accepted": True, "grasp_pixel_px": [150, 150], "base_xy_error_m": [.02, .02]}}
        snapshot = {"sequence": 1, "candidates": [0], "statistics": [item]}
        perception = SimpleNamespace(newer_than=lambda *a, **kw: snapshot)
        mailbox = SnapshotVisualMailbox(perception, initial_center_px=[100, 100],
            track_max_distance_px=30, target_lost_frames=1,
            target_pixel_px=[100, 100], pixel_jacobian=np.eye(2),
            pixel_tolerance_px=1, max_xy_step_m=.002, allow_xy_correction=False)
        result = mailbox.newer_than(0)
        self.assertTrue(result["valid"])
        self.assertNotIn("delta_base_m", result)
        item["step_pose"]["accepted"] = False
        self.assertTrue(mailbox.newer_than(0)["valid"])
        snapshot["candidates"] = []
        self.assertEqual(mailbox.newer_than(0)["reason"], "locked_target_lost")

    def test_visual_servo_corrects_only_beyond_five_mm(self):
        item = {"center_px": [104, 100], "area_px": 1000}
        snapshot = {"sequence": 1, "candidates": [0], "statistics": [item]}
        perception = SimpleNamespace(newer_than=lambda *a, **kw: snapshot)
        mailbox = SnapshotVisualMailbox(
            perception,
            initial_center_px=[100, 100],
            track_max_distance_px=30,
            target_lost_frames=1,
            target_pixel_px=[100, 100],
            pixel_jacobian=np.diag([1000.0, 1000.0]),
            pixel_tolerance_px=1,
            max_xy_step_m=.002,
            visual_tolerance_m=.005,
            allow_xy_correction=True,
        )
        within = mailbox.newer_than(0)
        self.assertNotIn("delta_base_m", within)
        self.assertAlmostEqual(within["visual_error_m"], .004)

        item["center_px"] = [106, 100]
        beyond = mailbox.newer_than(0)
        self.assertIn("delta_base_m", beyond)
        self.assertAlmostEqual(np.linalg.norm(beyond["delta_base_m"][:2]), .002)

    def test_visual_servo_refreshes_target_from_current_step_pose(self):
        item = {
            "center_px": [100, 100], "area_px": 1000,
            "step_pose": {"accepted": True, "grasp_pixel_px": [112, 100]},
        }
        snapshot = {"sequence": 1, "candidates": [0], "statistics": [item]}
        perception = SimpleNamespace(newer_than=lambda *a, **kw: snapshot)
        mailbox = SnapshotVisualMailbox(
            perception, initial_center_px=[100, 100], track_max_distance_px=30,
            target_lost_frames=1, target_pixel_px=[100, 100],
            pixel_jacobian=np.diag([1000.0, 1000.0]), pixel_tolerance_px=1,
            max_xy_step_m=.002, visual_tolerance_m=.005,
            target_pixel_provider=lambda _item: [112, 100],
        )
        result = mailbox.newer_than(0)
        self.assertEqual(result["target_pixel_px"], [112.0, 100.0])
        self.assertNotIn("delta_base_m", result)
