import math
import unittest

import numpy as np

from bilateral_rgbd_geometry import (
    AdaptiveDepthError,
    apply_transform,
    backproject_pixel,
    compose_transform,
    estimate_box_depth,
    estimate_hole_depth_adaptive,
    estimate_pair_geometry,
    estimate_point_depth,
    inverse_transform,
    right_color_point_to_base,
)


def identity_transform():
    return {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}


class BilateralRgbdGeometryTests(unittest.TestCase):
    def test_compose_and_inverse_use_explicit_frame_order(self):
        quarter_turn = [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
        t_a_b = {"translation_m": [1.0, 0.0, 0.0], "rotation_xyzw": quarter_turn}
        t_b_c = {"translation_m": [0.0, 2.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        composed = compose_transform(t_a_b, t_b_c)
        point = apply_transform(composed, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(point[0], -1.0, places=6)
        self.assertAlmostEqual(point[1], 0.0, places=6)
        recovered = apply_transform(inverse_transform(t_a_b), apply_transform(t_a_b, [0.2, -0.3, 1.0]))
        np.testing.assert_allclose(recovered, [0.2, -0.3, 1.0], atol=1e-6)

    def test_backproject_pixel(self):
        point = backproject_pixel((330.0, 250.0), 0.5, {"fx": 100.0, "fy": 100.0, "ppx": 320.0, "ppy": 240.0})
        np.testing.assert_allclose(point, [0.05, 0.05, 0.5], atol=1e-9)

    def test_backproject_pixel_undoes_brown_conrady_distortion(self):
        # An undistorted normalized point (0.1, 0.2) projected with k1=0.1.
        distorted_x, distorted_y = 0.1005, 0.201
        intrinsic = {
            "fx": 100.0,
            "fy": 100.0,
            "ppx": 320.0,
            "ppy": 240.0,
            "distortion": [0.1, 0.0, 0.0, 0.0, 0.0],
        }
        point = backproject_pixel(
            (320.0 + 100.0 * distorted_x, 240.0 + 100.0 * distorted_y),
            0.5,
            intrinsic,
        )
        np.testing.assert_allclose(point, [0.05, 0.1, 0.5], atol=1e-6)

    def test_right_color_point_to_base_uses_camera_then_end_to_base_order(self):
        quarter_turn = [0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)]
        right_end_from_color = {
            "translation_m": [0.1, 0.0, 0.0],
            "rotation_xyzw": quarter_turn,
        }
        base_from_right_end = {
            "translation_m": [1.0, 2.0, 3.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        }
        point = right_color_point_to_base(
            [0.0, 1.0, 0.0], right_end_from_color, base_from_right_end
        )
        np.testing.assert_allclose(point, [0.1, 2.0, 3.0], atol=1e-6)

    def test_hole_uses_rim_and_target_uses_inner_depth(self):
        depth = np.zeros((100, 120), dtype=np.uint16)
        # The hole center is background (1 m), while its rim is the fixture
        # plane (0.5 m). Target has a front layer with a few outliers.
        yy, xx = np.mgrid[20:60, 20:60]
        ring = ((xx - 40.0) / 18.0) ** 2 + ((yy - 40.0) / 14.0) ** 2
        patch = depth[20:60, 20:60]
        patch[ring <= 1.0] = 500
        patch[ring < 0.16] = 1000
        depth[60:80, 70:90] = 600
        depth[62, 72] = 1200
        intrinsic = {"fx": 100.0, "fy": 100.0, "ppx": 60.0, "ppy": 50.0}
        hole = estimate_box_depth(depth, (22, 24, 58, 56), intrinsic=intrinsic, role="hole")
        target = estimate_box_depth(depth, (70, 60, 90, 80), intrinsic=intrinsic, role="target")
        self.assertAlmostEqual(hole.depth_m, 0.5, places=6)
        self.assertAlmostEqual(target.depth_m, 0.6, places=6)

    def test_pair_translation_cancels_in_relative_error(self):
        depth = np.full((80, 100), 500, dtype=np.uint16)
        depth[20:40, 20:40] = 500
        depth[40:60, 60:80] = 500
        intrinsic = {"fx": 100.0, "fy": 100.0, "ppx": 50.0, "ppy": 40.0}
        t_end_color = {"translation_m": [0.1, -0.2, 0.3], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        t_base_end = {"translation_m": [1.0, 2.0, 3.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        pair = estimate_pair_geometry(
            depth,
            hole_xyxy=(20, 20, 40, 40),
            target_xyxy=(60, 40, 80, 60),
            color_intrinsic=intrinsic,
            rgbd_to_right_end=t_end_color,
            right_end_to_base=t_base_end,
        )
        expected = np.asarray(pair.target.point_color_m) - np.asarray(pair.hole.point_color_m)
        np.testing.assert_allclose(pair.error_target_minus_hole_m, expected, atol=1e-9)

    def test_point_depth_uses_tip_neighborhood(self):
        depth = np.zeros((40, 40), dtype=np.uint16)
        depth[18:23, 18:23] = 700
        estimate = estimate_point_depth(
            depth,
            (20.0, 20.0),
            intrinsic={"fx": 100.0, "fy": 100.0, "ppx": 20.0, "ppy": 20.0},
            radius_px=3,
            min_valid_pixels=8,
        )
        self.assertAlmostEqual(estimate.depth_m, 0.7, places=6)

    def test_adaptive_hole_depth_expands_to_next_scale(self):
        depth = np.zeros((120, 120), dtype=np.uint16)
        yy, xx = np.mgrid[:120, :120]
        normalized = ((xx - 60.0) / 5.0) ** 2 + ((yy - 60.0) / 5.0) ** 2
        depth[(normalized >= 1.0) & (normalized <= 2.0)] = 500
        estimate, diagnostics = estimate_hole_depth_adaptive(
            depth,
            (55.0, 55.0, 65.0, 65.0),
            intrinsic={"fx": 100.0, "fy": 100.0, "ppx": 60.0, "ppy": 60.0},
            scales=(1.0, 1.5),
            min_valid_pixels=8,
            min_support_fraction=0.20,
            min_covered_sectors=3,
        )
        self.assertAlmostEqual(estimate.depth_m, 0.5, places=6)
        self.assertEqual(diagnostics["selected"]["scale"], 1.5)

    def test_depth_statistics_include_mean_median_and_mad(self):
        depth = np.zeros((40, 40), dtype=np.uint16)
        depth[14:26, 14:26] = 500
        depth[14, 14] = 900  # rejected by the robust inlier pass
        estimate, diagnostics = estimate_hole_depth_adaptive(
            depth,
            (14.0, 14.0, 26.0, 26.0),
            intrinsic={"fx": 100.0, "fy": 100.0, "ppx": 20.0, "ppy": 20.0},
            scales=(1.0,),
            min_valid_pixels=8,
            min_support_fraction=0.20,
            min_covered_sectors=3,
        )
        selected = diagnostics["selected"]
        self.assertEqual(selected["valid_pixels"], estimate.valid_pixels)
        self.assertIn("mean_m", selected)
        self.assertIn("median_m", selected)
        self.assertIn("mad_mm", selected)
        self.assertAlmostEqual(selected["median_m"], estimate.depth_m, places=6)
        self.assertAlmostEqual(selected["mean_m"], estimate.mean_m, places=6)
        self.assertAlmostEqual(selected["mad_mm"], estimate.mad_m * 1000.0, places=6)

    def test_box_sampling_uses_all_valid_pixels_inside_detector_box(self):
        depth = np.zeros((30, 30), dtype=np.uint16)
        depth[10:20, 10:20] = 700
        depth[12, 12] = 900
        estimate, diagnostics = estimate_hole_depth_adaptive(
            depth,
            (10.0, 10.0, 20.0, 20.0),
            intrinsic={"fx": 100.0, "fy": 100.0, "ppx": 15.0, "ppy": 15.0},
            scales=(1.0,),
            sampling_region="box",
            min_valid_pixels=8,
            min_support_fraction=0.20,
            min_covered_sectors=3,
        )
        selected = diagnostics["selected"]
        self.assertEqual(selected["valid_pixels"], 100)
        self.assertAlmostEqual(selected["mean_m"], 0.702, places=6)
        self.assertAlmostEqual(selected["median_m"], 0.7, places=6)
        self.assertEqual(estimate.valid_pixels, 100)

    def test_adaptive_hole_depth_fails_when_all_scales_are_empty(self):
        with self.assertRaises(AdaptiveDepthError):
            estimate_hole_depth_adaptive(
                np.zeros((40, 40), dtype=np.uint16),
                (15.0, 15.0, 25.0, 25.0),
                intrinsic={"fx": 100.0, "fy": 100.0, "ppx": 20.0, "ppy": 20.0},
                scales=(1.0, 1.5),
            )


if __name__ == "__main__":
    unittest.main()
