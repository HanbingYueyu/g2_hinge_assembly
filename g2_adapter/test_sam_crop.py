import unittest

import numpy as np

try:
    from .sam_crop import center_crop_bounds, center_crop_bounds_aspect, center_crop_bounds_size, center_crop_image, expand_crop_masks
except ImportError:
    from sam_crop import center_crop_bounds, center_crop_bounds_aspect, center_crop_bounds_size, center_crop_image, expand_crop_masks


class SamCropTests(unittest.TestCase):
    def test_center_70_percent_crop_and_mask_expansion(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        crop, bounds = center_crop_image(image)
        self.assertEqual(bounds, (30, 15, 170, 85))
        self.assertEqual(crop.shape, (70, 140, 3))
        masks = expand_crop_masks([np.ones((70, 140), dtype=bool)], full_shape=image.shape[:2], bounds=bounds)
        self.assertEqual(masks[0].shape, (100, 200))
        self.assertEqual(int(masks[0].sum()), 70 * 140)
        self.assertFalse(masks[0][0, 0])
        self.assertTrue(masks[0][15, 30])

    def test_bounds_reject_invalid_fraction(self):
        with self.assertRaises(ValueError):
            center_crop_bounds((10, 10), fraction=0.0)

    def test_fixed_size_center_crop_bounds(self):
        self.assertEqual(
            center_crop_bounds_size((1056, 1280, 3), crop_width=680, crop_height=420),
            (300, 318, 980, 738),
        )

    def test_largest_center_crop_with_training_aspect(self):
        self.assertEqual(
            center_crop_bounds_aspect((1056, 1280, 3), aspect_width=680, aspect_height=420),
            (0, 132, 1280, 923),
        )


if __name__ == "__main__":
    unittest.main()
