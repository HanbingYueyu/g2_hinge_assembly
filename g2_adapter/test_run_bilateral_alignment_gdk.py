import io
import unittest

import numpy as np
from unittest.mock import patch

from control import CartesianPoseTarget
from run_bilateral_alignment_gdk import (
    compact_frame,
    can_accumulate_hole_depth,
    emit_terminal_record,
    infer_left_target_boxes,
    left_target_from_offset,
    parse_depth_scales,
    parse_jacobian,
    terminal_summary,
)


class BilateralRunnerPureTests(unittest.TestCase):
    def test_right_only_veto_accumulates_hole_without_target(self):
        self.assertTrue(
            can_accumulate_hole_depth(
                selected_hole=True,
                aligned_depth=True,
                target_count=0,
                geometry_mode="veto",
            )
        )

    def test_direct_requires_exactly_one_target_for_hole_batch(self):
        self.assertFalse(
            can_accumulate_hole_depth(
                selected_hole=True,
                aligned_depth=True,
                target_count=0,
                geometry_mode="direct",
            )
        )
        self.assertTrue(
            can_accumulate_hole_depth(
                selected_hole=True,
                aligned_depth=True,
                target_count=1,
                geometry_mode="direct",
            )
        )

    def test_hole_batch_needs_selected_hole_and_aligned_depth(self):
        for selected_hole, aligned_depth in ((False, True), (True, False), (False, False)):
            self.assertFalse(
                can_accumulate_hole_depth(
                    selected_hole=selected_hole,
                    aligned_depth=aligned_depth,
                    target_count=1,
                    geometry_mode="veto",
                )
            )

    def test_parse_jacobian_accepts_comma_and_space_forms(self):
        self.assertEqual(parse_jacobian("1, 2, 3, 4"), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(parse_jacobian("1 2 3 4"), [[1.0, 2.0], [3.0, 4.0]])

    def test_parse_jacobian_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            parse_jacobian("1,2,3")

    def test_parse_depth_scales_requires_one_and_sorted(self):
        self.assertEqual(parse_depth_scales("1,1.25,1.5"), (1.0, 1.25, 1.5))
        with self.assertRaises(ValueError):
            parse_depth_scales("1.25,1.5")
        with self.assertRaises(ValueError):
            parse_depth_scales("1.5,1.25")

    def test_left_target_preserves_z_and_orientation(self):
        initial = CartesianPoseTarget((0.4, -0.2, 0.8), (0.1, 0.2, 0.3, 0.9))
        target = left_target_from_offset(initial, (0.001, -0.002, 0.0))
        self.assertEqual(target.position_m, (0.401, -0.202, 0.8))
        self.assertEqual(target.orientation_xyzw, initial.orientation_xyzw)
        with self.assertRaises(ValueError):
            left_target_from_offset(initial, (0.0, 0.0, 0.001))

    def test_compact_frame_drops_images_and_masks(self):
        compact = compact_frame({
            "event": "frame", "frame": 1, "state": "hold",
            "detections": [], "image": b"jpeg", "masks": [1],
        })
        self.assertNotIn("image", compact)
        self.assertNotIn("masks", compact)
        self.assertEqual(compact["state"], "hold")

    def test_terminal_summary_prioritizes_right_hole_camera_xyz(self):
        summary = terminal_summary({
            "event": "frame",
            "frame": 4,
            "state": "geometry_hold",
            "reason": "hole_depth_collecting",
            "right_hole_lock": {"locked": True},
            "hole_depth": {"state": "collecting", "window_frames": 4, "temporal_min_hits": 3},
            "right_hole_xyz_color_camera_m": [0.0123, -0.0456, 0.2090],
            "right_hole_xyz_base_link_m": [0.4, -0.6, 0.2],
        })
        self.assertIn("frame=4", summary)
        self.assertIn("hole=LOCK", summary)
        self.assertIn("hole_cam=(12.3,-45.6,209.0)mm", summary)
        self.assertIn("r_cam=214.3mm", summary)
        self.assertIn("hole_base=(400.0,-600.0,200.0)mm", summary)
        self.assertNotIn("win=", summary)
        self.assertNotIn("depth=", summary)
        self.assertNotIn("valid=", summary)
        self.assertNotIn("mean=", summary)
        self.assertNotIn("median=", summary)
        self.assertNotIn("mad=", summary)
        self.assertNotIn("\n", summary)

    def test_terminal_summary_shows_raw_xyz_without_unlocking_motion(self):
        summary = terminal_summary({
            "event": "frame",
            "frame": 327,
            "state": "geometry_hold",
            "reason": "locked_hole_jump_exceeds_limit",
            "right_hole_lock": {"locked": False, "reason": "locked_hole_jump_exceeds_limit"},
            "right_hole_xyz_color_camera_m": [0.0123, -0.0456, 0.2090],
            "right_hole_xyz_base_link_m": [0.4, -0.6, 0.2],
        })
        self.assertIn("hole=RAW", summary)
        self.assertIn("hole_cam=(12.3,-45.6,209.0)mm", summary)
        self.assertIn("hole_base=(400.0,-600.0,200.0)mm", summary)

    def test_terminal_summary_marks_disabled_lock_as_live(self):
        summary = terminal_summary({
            "event": "frame",
            "frame": 1,
            "state": "geometry_hold",
            "reason": "lock_disabled_current_frame",
            "right_hole_lock": {"locked": False, "mode": "disabled"},
            "right_hole_xyz_color_camera_m": [0.0123, -0.0456, 0.2090],
        })
        self.assertIn("hole=LIVE", summary)

    def test_terminal_summary_refreshes_in_tty(self):
        stream = io.StringIO()
        stream.isatty = lambda: True
        with patch("run_bilateral_alignment_gdk.sys.stdout", stream):
            live = emit_terminal_record(
                {"event": "frame", "frame": 1, "state": "geometry_hold", "reason": "collecting"},
                "summary",
            )
        self.assertTrue(live)
        self.assertIn("\rframe=1", stream.getvalue())
        self.assertIn("\033[K", stream.getvalue())
        self.assertNotIn('"event"', stream.getvalue())

    def test_left_shaft_is_exposed_as_target_tip_away_from_hinge(self):
        hinge = np.zeros((100, 100), dtype=bool)
        hinge[45:55, 25:35] = True
        shaft = np.zeros((100, 100), dtype=bool)
        shaft[48:52, 38:64] = True
        with patch(
            "run_bilateral_alignment_gdk.infer_yolo_masks",
            return_value=([hinge, shaft], np.asarray([0.9, 0.8]), np.asarray([0, 1])),
        ):
            targets = infer_left_target_boxes(object(), np.zeros((100, 100, 3), dtype=np.uint8), confidence=0.25, imgsz=640, device="cpu")
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0].class_name, "target")
        self.assertGreater(targets[0].tip_px[0], 55.0)

    def test_multiple_hinges_use_the_nearest_hinge_for_each_shaft(self):
        hinge_left = np.zeros((100, 120), dtype=bool)
        hinge_left[45:55, 10:20] = True
        hinge_right = np.zeros((100, 120), dtype=bool)
        hinge_right[45:55, 80:90] = True
        shaft = np.zeros((100, 120), dtype=bool)
        shaft[48:52, 62:78] = True
        overlay = []
        with patch(
            "run_bilateral_alignment_gdk.infer_yolo_masks",
            return_value=(
                [hinge_left, hinge_right, shaft],
                np.asarray([0.9, 0.8, 0.85]),
                np.asarray([0, 0, 1]),
            ),
        ):
            targets = infer_left_target_boxes(
                object(),
                np.zeros((100, 120, 3), dtype=np.uint8),
                confidence=0.25,
                imgsz=640,
                device="cpu",
                selection="rightmost",
                overlay_detections=overlay,
            )
        self.assertEqual(len(targets), 1)
        self.assertIsNotNone(targets[0].hinge_center_px)
        self.assertGreater(targets[0].hinge_center_px[0], 60.0)
        self.assertEqual([item.class_name for item in overlay], ["hinge", "hinge", "shaft"])

    def test_initial_selection_prefers_nearest_depth(self):
        hinge_far = np.zeros((40, 100), dtype=bool)
        hinge_far[15:25, 10:20] = True
        hinge_near = np.zeros((40, 100), dtype=bool)
        hinge_near[15:25, 70:80] = True
        shaft = np.zeros((40, 100), dtype=bool)
        shaft[18:22, 58:68] = True
        aligned_depth = np.zeros((40, 100), dtype=np.uint16)
        aligned_depth[hinge_far] = 900
        aligned_depth[hinge_near] = 500
        with patch(
            "run_bilateral_alignment_gdk.infer_yolo_masks",
            return_value=(
                [hinge_far, hinge_near, shaft],
                np.asarray([0.9, 0.8, 0.85]),
                np.asarray([0, 0, 1]),
            ),
        ):
            targets = infer_left_target_boxes(
                object(),
                np.zeros((40, 100, 3), dtype=np.uint8),
                confidence=0.25,
                imgsz=640,
                device="cpu",
                aligned_depth=aligned_depth,
            )
        self.assertEqual(len(targets), 1)
        self.assertGreater(targets[0].hinge_center_px[0], 60.0)

    def test_previous_tip_keeps_the_same_shaft_when_another_pair_is_closer(self):
        hinge_left = np.zeros((100, 120), dtype=bool)
        hinge_left[45:55, 20:30] = True
        hinge_right = np.zeros((100, 120), dtype=bool)
        hinge_right[45:55, 80:90] = True
        shaft_left = np.zeros((100, 120), dtype=bool)
        shaft_left[48:52, 35:45] = True
        shaft_right = np.zeros((100, 120), dtype=bool)
        shaft_right[48:52, 58:68] = True
        with patch(
            "run_bilateral_alignment_gdk.infer_yolo_masks",
            return_value=(
                [hinge_left, hinge_right, shaft_left, shaft_right],
                np.asarray([0.9, 0.9, 0.75, 0.99]),
                np.asarray([0, 0, 1, 1]),
            ),
        ):
            targets = infer_left_target_boxes(
                object(),
                np.zeros((100, 120, 3), dtype=np.uint8),
                confidence=0.25,
                imgsz=640,
                device="cpu",
                previous_tip_px=(58.0, 50.0),
                previous_hinge_px=(84.5, 50.0),
                max_track_jump_px=30.0,
            )
        self.assertEqual(len(targets), 1)
        self.assertGreater(targets[0].tip_px[0], 50.0)

    def test_previous_track_does_not_switch_to_a_far_neighbor(self):
        hinge = np.zeros((100, 120), dtype=bool)
        hinge[45:55, 80:90] = True
        shaft = np.zeros((100, 120), dtype=bool)
        shaft[48:52, 62:72] = True
        with patch(
            "run_bilateral_alignment_gdk.infer_yolo_masks",
            return_value=(
                [hinge, shaft],
                np.asarray([0.9, 0.9]),
                np.asarray([0, 1]),
            ),
        ):
            targets = infer_left_target_boxes(
                object(),
                np.zeros((100, 120, 3), dtype=np.uint8),
                confidence=0.25,
                imgsz=640,
                device="cpu",
                previous_tip_px=(20.0, 50.0),
                previous_hinge_px=(20.0, 50.0),
                max_track_jump_px=30.0,
            )
        self.assertEqual(targets, [])


if __name__ == "__main__":
    unittest.main()
