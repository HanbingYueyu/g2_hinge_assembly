import unittest
import json
import tempfile
from pathlib import Path

import numpy as np

try:
    from .left_step_height import (
        HEIGHT_REFERENCE_SCHEMA_VERSION,
        aggregate_height_observations,
        interpolate_target_pixel,
        validate_height_samples,
        load_height_samples,
        camera_distance_from_base_point,
    )
except ImportError:
    from left_step_height import (
        HEIGHT_REFERENCE_SCHEMA_VERSION,
        aggregate_height_observations,
        interpolate_target_pixel,
        validate_height_samples,
        load_height_samples,
        camera_distance_from_base_point,
    )


class LeftStepHeightReferenceTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            {"height_mm": 100.0, "camera_distance_m": 0.300, "target_pixel_px": [640.0, 460.0]},
            {"height_mm": 50.0, "camera_distance_m": 0.200, "target_pixel_px": [630.0, 500.0]},
            {"height_mm": 0.0, "camera_distance_m": 0.100, "target_pixel_px": [620.0, 540.0]},
        ]

    def test_interpolates_target_pixel_between_neighbouring_heights(self):
        pixel, detail = interpolate_target_pixel(self.samples, camera_distance_m=0.150)
        np.testing.assert_allclose(pixel, [625.0, 520.0])
        self.assertEqual(detail["lower_height_mm"], 0.0)
        self.assertEqual(detail["upper_height_mm"], 50.0)
        self.assertAlmostEqual(detail["alpha"], 0.5)

    def test_clamps_to_nearest_anchor_outside_calibrated_distance_range(self):
        pixel, detail = interpolate_target_pixel(self.samples, camera_distance_m=0.350)
        np.testing.assert_allclose(pixel, [640.0, 460.0])
        self.assertEqual(detail["mode"], "clamped_high")

        pixel, detail = interpolate_target_pixel(self.samples, camera_distance_m=0.050)
        np.testing.assert_allclose(pixel, [620.0, 540.0])
        self.assertEqual(detail["mode"], "clamped_low")

    def test_validation_requires_100_50_and_zero_height_anchors(self):
        validated = validate_height_samples(self.samples)
        self.assertEqual(validated[0]["height_mm"], 0.0)
        self.assertEqual(validated[-1]["height_mm"], 100.0)

        with self.assertRaises(ValueError):
            validate_height_samples(self.samples[:2])

    def test_validation_rejects_duplicate_or_invalid_depth(self):
        duplicate = [*self.samples, {**self.samples[0], "target_pixel_px": [1.0, 2.0]}]
        with self.assertRaises(ValueError):
            validate_height_samples(duplicate)

        invalid = [*self.samples]
        invalid[1] = {**invalid[1], "camera_distance_m": 0.0}
        with self.assertRaises(ValueError):
            validate_height_samples(invalid)

    def test_schema_version_is_explicit(self):
        self.assertEqual(HEIGHT_REFERENCE_SCHEMA_VERSION, 2)

    def test_loads_height_samples_without_retired_step_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "height.json"
            path.write_text(json.dumps({"height_samples": self.samples, "source_step": "missing.step"}), encoding="utf-8")
            loaded = load_height_samples(path)
        self.assertEqual(len(loaded), 3)
        self.assertEqual(loaded[-1]["height_mm"], 100.0)

    def test_camera_distance_can_be_estimated_without_hand_depth(self):
        distance = camera_distance_from_base_point(
            (0.0, 0.0, 0.20), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0),
            {"translation_m": [0.0, 0.0, 0.05], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
        )
        self.assertAlmostEqual(distance, 0.15, places=6)

    def test_camera_distance_handles_rotated_end_link_and_camera_extrinsic(self):
        distance = camera_distance_from_base_point(
            (0.7, -0.2, 0.9),
            (0.3, -0.1, 0.2),
            (0.2, 0.3, 0.1, 0.9),
            {"translation_m": [0.04, -0.03, 0.05], "rotation_xyzw": [-0.1, 0.2, -0.3, 0.9]},
        )
        self.assertAlmostEqual(distance, 0.7778174593, places=7)

    def test_aggregates_marked_frames_by_median(self):
        sample = aggregate_height_observations(
            50.0,
            pixels=[[630.0, 500.0], [632.0, 498.0], [629.0, 501.0]],
            grasp_xyz=[[0.0, 0.0, 0.20], [0.002, 0.0, 0.20], [0.0, 0.001, 0.20]],
        )
        self.assertEqual(sample["height_mm"], 50.0)
        np.testing.assert_allclose(sample["target_pixel_px"], [630.0, 500.0])
        np.testing.assert_allclose(sample["grasp_xyz_camera_m"], [0.0, 0.0, 0.20])
        self.assertAlmostEqual(sample["camera_distance_m"], 0.20, places=6)
        self.assertEqual(sample["sample_count"], 3)

        with self.assertRaises(ValueError):
            aggregate_height_observations(50.0, pixels=[], grasp_xyz=[])


if __name__ == "__main__":
    unittest.main()
