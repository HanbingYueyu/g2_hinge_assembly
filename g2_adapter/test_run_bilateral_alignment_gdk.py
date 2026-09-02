import unittest

import numpy as np
from unittest.mock import patch

from control import CartesianPoseTarget
from run_bilateral_alignment_gdk import compact_frame, infer_left_target_boxes, left_target_from_offset, parse_jacobian


class BilateralRunnerPureTests(unittest.TestCase):
    def test_parse_jacobian_accepts_comma_and_space_forms(self):
        self.assertEqual(parse_jacobian("1, 2, 3, 4"), [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(parse_jacobian("1 2 3 4"), [[1.0, 2.0], [3.0, 4.0]])

    def test_parse_jacobian_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            parse_jacobian("1,2,3")

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


if __name__ == "__main__":
    unittest.main()
