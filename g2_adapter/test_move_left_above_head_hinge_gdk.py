import unittest
import numpy as np

from move_left_above_head_hinge_gdk import (
    base_from_head_at,
    build_head_targets,
    cluster_head_target,
    dominant_sample_cluster,
    draw_head_coarse_overlay,
    fixed_forward_down_orientation,
    gripper_end_target,
    sample_head_target_points,
    select_highest_confidence_class,
    select_leftmost_hinge,
)
from control import quaternion_angle_deg


class HeadHingeMoveTests(unittest.TestCase):
    def test_head_overlay_marks_selected_hinge_and_sampling_progress(self):
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        mask = np.zeros((80, 120), dtype=bool)
        mask[25:55, 40:70] = True
        overlay = draw_head_coarse_overlay(image, [mask], [0.95], [0], 0, 3, 15)
        self.assertEqual(overlay.shape, image.shape)
        self.assertGreater(int(overlay.sum()), 0)
    def test_dominant_sample_cluster_discards_distant_samples(self):
        stable = np.array([
            [0.500, 0.200, 0.800], [0.510, 0.198, 0.802],
            [0.495, 0.205, 0.799], [0.503, 0.201, 0.805],
            [0.507, 0.203, 0.797],
        ])
        samples = np.vstack((stable, [[0.80, -0.10, 0.90], [0.10, 0.60, 0.40]]))
        center, inliers, spread_m = dominant_sample_cluster(samples, 0.05)
        self.assertEqual(int(inliers.sum()), len(stable))
        self.assertTrue(np.allclose(center, np.median(stable, axis=0)))
        self.assertLessEqual(spread_m, 0.05)
    def test_selects_leftmost_hinge_and_ignores_other_class(self):
        masks = []
        for x in (8, 2, 0):
            mask = np.zeros((10, 12), dtype=bool)
            mask[3:6, x:x+2] = True
            masks.append(mask)
        self.assertEqual(select_leftmost_hinge(masks, [0.8]*3, [0, 0, 1]), 1)

    def test_selects_highest_confidence_hole_of_its_own_class(self):
        hinge = np.zeros((10, 12), dtype=bool)
        hinge[3:6, 1:3] = True
        weak_hole = np.zeros((10, 12), dtype=bool)
        weak_hole[3:6, 4:6] = True
        strong_hole = np.zeros((10, 12), dtype=bool)
        strong_hole[3:6, 8:10] = True
        empty = np.zeros((10, 12), dtype=bool)
        masks = [hinge, weak_hole, strong_hole, empty]
        scores = [0.99, 0.40, 0.71, 0.95]
        classes = [0, 1, 1, 1]
        self.assertEqual(select_highest_confidence_class(masks, scores, classes, 1), 2)
        self.assertEqual(select_highest_confidence_class(masks, scores, classes, 0), 0)
        self.assertIsNone(select_highest_confidence_class([empty], [0.9], [1], 1))

    def test_sample_head_target_points_collects_hinge_and_hole_from_one_stream(self):
        hinge = np.zeros((10, 12), dtype=bool)
        hinge[2:5, 0:2] = True
        hole = np.zeros((10, 12), dtype=bool)
        hole[2:5, 7:9] = True
        points = {"hinge": np.array([0.0, 0.0, 0.0]), "hole": np.array([0.1, 0.1, 0.0])}
        frames = {"n": 0}

        def next_observation():
            frames["n"] += 1
            if frames["n"] % 2:
                return [hinge], [0.9], [0], lambda mask: points["hinge"]
            return [hole], [0.8], [1], lambda mask: points["hole"]

        observations, seen = sample_head_target_points(
            next_observation=next_observation, samples=3, timeout_s=5.0,
        )
        self.assertEqual(len(observations["hinge"]), 3)
        self.assertEqual(len(observations["hole"]), 3)
        self.assertGreaterEqual(seen, 6)

    def test_sample_head_target_points_gives_up_when_a_class_is_missing(self):
        hinge = np.zeros((10, 12), dtype=bool)
        hinge[2:5, 0:2] = True
        ticks = {"now": 0.0}

        def clock():
            ticks["now"] += 0.5
            return ticks["now"]

        observations, _ = sample_head_target_points(
            next_observation=lambda: ([hinge], [0.9], [0], lambda mask: np.zeros(3)),
            samples=5, timeout_s=2.0, clock=clock,
        )
        # deadline = 0.5 + 2.0; the loop appends while the clock is below it.
        self.assertEqual(len(observations["hinge"]), 3)
        self.assertEqual(observations["hole"], [])

    def test_cluster_head_target_reports_dominant_cluster(self):
        stable = [np.array([0.5, 0.2, 0.8]) for _ in range(9)]
        outlier = [np.array([5.0, 5.0, 5.0])]
        report = cluster_head_target(
            "hole", stable + outlier, max_sample_spread_mm=50.0, min_inlier_samples=8,
        )
        self.assertEqual(report["inlier_samples"], 9)
        self.assertEqual(report["rejected_samples"], 1)
        self.assertAlmostEqual(report["position_base_link_m"][0], 0.5, places=6)
        with self.assertRaises(RuntimeError):
            cluster_head_target("hole", [], max_sample_spread_mm=50.0, min_inlier_samples=1)
        with self.assertRaises(RuntimeError):
            cluster_head_target("hole", [np.array([0.0, 0.0, 0.0])], max_sample_spread_mm=50.0, min_inlier_samples=8)

    def test_build_head_targets_reports_a_missing_class_instead_of_raising(self):
        hinge = [np.array([0.5, 0.2, 0.8]) for _ in range(9)]
        targets, rejected = build_head_targets(
            {"hinge": hinge, "hole": []}, max_sample_spread_mm=50.0, min_inlier_samples=8,
        )
        self.assertEqual(set(targets), {"hinge"})
        self.assertNotIn("hole", targets)
        self.assertEqual(rejected["hole"]["samples"], 0)
        self.assertIn("hole", rejected["hole"]["reason"])

    def test_build_head_targets_reports_a_scattered_class(self):
        scattered = [np.array([float(index), 0.0, 0.0]) for index in range(9)]
        targets, rejected = build_head_targets(
            {"hole": scattered}, max_sample_spread_mm=10.0, min_inlier_samples=8,
        )
        self.assertEqual(targets, {})
        self.assertIn("hole", rejected)
        self.assertIn("主簇", rejected["hole"]["reason"])

    def test_target_accounts_for_tcp_and_above_height(self):
        target = gripper_end_target(np.array([0.5, 0.2, 0.7]), (0, 0, 0, 1), np.array([0, 0, 0.145]), 0.2)
        self.assertTrue(np.allclose(target.position_m, [0.5, 0.2, 0.755]))

    def test_fixed_orientation_points_forward_and_down_thirty_degrees(self):
        # Keep the axis-direction assertion independent of the default tool
        # roll; the default left-hand orientation intentionally adds a 90°
        # clockwise roll around the forward axis.
        orientation = fixed_forward_down_orientation(gripper_roll_deg=0.0)
        x, y, z, w = orientation
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        self.assertTrue(np.allclose(rotation[:, 2], [np.sqrt(3) / 2, 0.0, -0.5], atol=1e-6))
        self.assertTrue(np.allclose(rotation[:, 1], [0.5, 0.0, np.sqrt(3) / 2], atol=1e-6))

    def test_default_orientation_adds_one_hundred_eighty_degree_roll(self):
        zero_roll = fixed_forward_down_orientation(gripper_roll_deg=0.0)
        default = fixed_forward_down_orientation()
        self.assertAlmostEqual(quaternion_angle_deg(zero_roll, default), 180.0, places=5)

    def test_fixed_orientation_supports_yaw_and_pitch_adjustment(self):
        orientation = fixed_forward_down_orientation(forward_yaw_deg=90.0, gripper_pitch_deg=0.0)
        x, y, z, w = orientation
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        self.assertTrue(np.allclose(rotation[:, 2], [0.0, 1.0, 0.0], atol=1e-6))

    def test_tf_uses_bounded_latest_fallback(self):
        class Vector:
            x = y = z = 0.0
        class Rotation:
            x = y = z = 0.0
            w = 1.0
        class Transform:
            translation = Vector()
            rotation = Rotation()
        class TF:
            def lookup_transform(self, *_args):
                raise RuntimeError("before oldest transform")
            def lookup_transform_latest(self, *_args, **_kwargs):
                return Transform(), 1_001_000_000
        transform, mode, gap_ms = base_from_head_at(TF(), 1_000_000_000)
        self.assertEqual(mode, "near_latest")
        self.assertAlmostEqual(gap_ms, 1.0)
        self.assertTrue(np.allclose(transform["translation_m"], [0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
