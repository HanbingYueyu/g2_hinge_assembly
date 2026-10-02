import unittest
from types import SimpleNamespace
import numpy as np
import left_grasp_gdk
from left_grasp_gdk import SnapshotVisualMailbox


class DescentXYHoldTests(unittest.TestCase):
    def test_three_mm_visual_error_holds_z_with_two_mm_tolerance(self):
        priorities = []
        stream = SimpleNamespace(set_xy_priority=priorities.append)
        request_hold = getattr(left_grasp_gdk, 'request_descent_xy_hold', None)
        self.assertIsNotNone(request_hold)
        self.assertTrue(request_hold(stream, visual_error_m=.003,
                                     tolerance_m=.002, already_pending=False))
        self.assertEqual(priorities, [True])
        self.assertTrue(request_hold(stream, visual_error_m=.001,
                                     tolerance_m=.002, already_pending=True))
        self.assertEqual(priorities, [True, True])

    def test_filtered_detection_jitter_does_not_reverse_correction_each_frame(self):
        item = {"center_px": [104, 100], "area_px": 1000}
        snapshot = {"sequence": 1, "candidates": [0], "statistics": [item]}
        perception = SimpleNamespace(newer_than=lambda *a, **kw: snapshot)
        mailbox = SnapshotVisualMailbox(
            perception, initial_center_px=[100, 100], track_max_distance_px=30,
            target_lost_frames=1, target_pixel_px=[100, 100],
            pixel_jacobian=np.diag([1000.0, 1000.0]), max_xy_step_m=.001,
            visual_tolerance_m=.002, error_filter_alpha=.4, correction_gain=.5,
        )
        first = mailbox.newer_than(0)
        self.assertLess(first["delta_base_m"][0], 0)
        snapshot["sequence"] = 2
        item["center_px"] = [96, 100]
        jitter = mailbox.newer_than(1)
        self.assertNotIn("delta_base_m", jitter)
        self.assertAlmostEqual(jitter["filtered_pixel_error_px"][0], .8)

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

    def test_visual_servo_corrects_only_beyond_two_mm(self):
        item = {"center_px": [101, 100], "area_px": 1000}
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
            visual_tolerance_m=.002,
            allow_xy_correction=True,
        )
        within = mailbox.newer_than(0)
        self.assertNotIn("delta_base_m", within)
        self.assertAlmostEqual(within["visual_error_m"], .001)

        item["center_px"] = [103, 100]
        beyond = mailbox.newer_than(0)
        self.assertIn("delta_base_m", beyond)
        self.assertAlmostEqual(np.linalg.norm(beyond["delta_base_m"][:2]), .002)

    def test_visual_servo_refreshes_target_from_current_step_pose(self):
        item = {
            "center_px": [100, 100], "area_px": 1000,
            "yolo_box_grasp_pixel_px": [112, 100],
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
