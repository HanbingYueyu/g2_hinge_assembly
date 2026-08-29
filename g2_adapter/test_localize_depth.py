from __future__ import annotations

import unittest

import numpy as np

from localize_left_d405_sam3 import depth_points


class ForegroundDepthLayerTests(unittest.TestCase):
    def test_nearest_supported_layer_wins_over_distant_background_majority(self):
        mask = np.ones((40, 50), dtype=bool)
        depth = np.full(mask.shape, 760, dtype=np.uint16)
        depth[:16, :] = 200  # foreground metal, 40% of valid mask pixels
        _inliers, points = depth_points(
            mask,
            depth,
            0.001,
            {"fx": 400.0, "fy": 400.0, "ppx": 25.0, "ppy": 20.0},
        )
        self.assertGreater(points.shape[0], 500)
        self.assertAlmostEqual(float(np.median(points[:, 2])), 0.2, places=3)

    def test_does_not_promote_an_isolated_near_depth_noise_pixel(self):
        mask = np.ones((30, 30), dtype=bool)
        depth = np.full(mask.shape, 760, dtype=np.uint16)
        depth[0, 0] = 120
        _inliers, points = depth_points(
            mask,
            depth,
            0.001,
            {"fx": 400.0, "fy": 400.0, "ppx": 15.0, "ppy": 15.0},
        )
        self.assertAlmostEqual(float(np.median(points[:, 2])), 0.76, places=3)


if __name__ == "__main__":
    unittest.main()
