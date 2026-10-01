from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from g2_adapter.two_hole_rgbd import base_from_head_at, box_depth_m, localize_box, project_color_pixel_to_base_z


IDENTITY = {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}


class TwoHoleRgbdTest(unittest.TestCase):
    def test_depth_median_rejects_far_background(self):
        depth = np.zeros((20, 20), dtype=np.uint16)
        depth[4:12, 4:12] = 800
        depth[5:7, 5:7] = 1200
        result = box_depth_m(depth, (4, 4, 12, 12), min_valid_pixels=12)
        self.assertAlmostEqual(result["depth_m"], 0.8)
        self.assertEqual(result["valid_pixels"], 64)
        self.assertEqual(result["inlier_pixels"], 60)

    def test_localize_box_applies_color_to_head_to_base(self):
        depth = np.full((20, 20), 1000, dtype=np.uint16)
        intrinsic = {"fx": 100.0, "fy": 100.0, "ppx": 0.0, "ppy": 0.0, "distortion": []}
        head_from_color = {"translation_m": [1.0, 0.0, 0.0], "rotation_xyzw": [0, 0, 0, 1]}
        base_from_head = {"translation_m": [0.0, 2.0, 0.0], "rotation_xyzw": [0, 0, 0, 1]}
        result = localize_box((8, 8, 12, 12), depth, intrinsic, head_from_color, base_from_head)
        self.assertTrue(np.allclose(result["point_head_color_optical_m"], [0.1, 0.1, 1.0]))
        self.assertTrue(np.allclose(result["point_head_link3_m"], [1.1, 0.1, 1.0]))
        self.assertTrue(np.allclose(result["point_base_link_m"], [1.1, 2.1, 1.0]))

    def test_pixel_ray_intersects_known_horizontal_plane(self):
        intrinsic = {"fx": 100.0, "fy": 100.0, "ppx": 0.0, "ppy": 0.0, "distortion": []}
        head_from_color = {"translation_m": [0.0, 0.0, 1.0], "rotation_xyzw": [1.0, 0.0, 0.0, 0.0]}
        result = project_color_pixel_to_base_z((10.0, 5.0), 0.7, intrinsic, head_from_color, IDENTITY)
        self.assertTrue(np.allclose(result, [0.03, -0.015, 0.7]))

    def test_gdk_reverse_tf_is_inverted_at_boundary(self):
        raw_head_from_base = {"translation_m": [-1.0, 0.0, 0.0], "rotation_xyzw": [0, 0, 0, 1]}

        class FakeTF:
            def lookup_transform_latest(self, target, source, return_timestamp):
                self._check(target, source)
                return raw_head_from_base, 1_000_000_000

            def lookup_transform(self, target, source, timestamp):
                self._check(target, source)
                return raw_head_from_base

            @staticmethod
            def _check(target, source):
                assert (target, source) == ("base_link", "head_link3")

        with patch("g2_adapter.two_hole_rgbd.transform_to_dict", side_effect=lambda value: value):
            transform, status = base_from_head_at(FakeTF(), 900_000_000)
        self.assertEqual(status["mode"], "exact_image_time")
        self.assertTrue(np.allclose(transform["translation_m"], [1.0, 0.0, 0.0]))


if __name__ == "__main__":
    unittest.main()
