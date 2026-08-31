import unittest

import numpy as np

try:
    from .yolo_hinge import resize_masks_to_full_image, review_matches
except ImportError:
    from yolo_hinge import resize_masks_to_full_image, review_matches


class YoloReviewTests(unittest.TestCase):
    def test_matching_target_is_accepted(self):
        accepted, detail = review_matches(
            {"center_px": [500.0, 400.0], "depth_median_m": 0.200},
            {"center_px": [510.0, 405.0], "depth_median_m": 0.215},
            max_pixel_error=20.0,
            max_depth_error_m=0.02,
        )
        self.assertTrue(accepted)
        self.assertEqual(detail["reason"], "matched")

    def test_disagreement_is_rejected(self):
        accepted, detail = review_matches(
            {"center_px": [500.0, 400.0], "depth_median_m": 0.200},
            {"center_px": [650.0, 400.0], "depth_median_m": 0.260},
            max_pixel_error=20.0,
            max_depth_error_m=0.02,
        )
        self.assertFalse(accepted)
        self.assertEqual(detail["reason"], "selected_target_disagreement")

    def test_resize_mask_to_full_image_uses_nearest_neighbor(self):
        mask = np.array([[True, False], [False, True]])
        expanded = resize_masks_to_full_image([mask], full_shape=(4, 6))
        self.assertEqual(expanded[0].shape, (4, 6))
        self.assertTrue(expanded[0][0, 0])
        self.assertFalse(expanded[0][0, 5])
        self.assertTrue(expanded[0][3, 5])


if __name__ == "__main__":
    unittest.main()
