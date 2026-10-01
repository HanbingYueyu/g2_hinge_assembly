"""Checks the endpoint geometry and constant-translation rule."""

from __future__ import annotations

import math
import json
import tempfile
import unittest
from pathlib import Path

from .retarget_two_hole_cartesian import _dataset_dir, _reference_dir, endpoint_translation, read_target_pair, translate_document


class RetargetTwoHoleTests(unittest.TestCase):
    def test_latest_relocalization_can_be_nested_under_run_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "run_001" / "capture_001"
            latest = root / "run_002" / "capture_002"
            first.mkdir(parents=True)
            latest.mkdir(parents=True)
            (first / "sample_0011.json").write_text("{}")
            (latest / "sample_0011.json").write_text("{}")
            self.assertEqual(_reference_dir(root, latest=True), latest)

    def test_dataset_directory_matches_teaching_session_name(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            teaching = root / "trajectories" / "two_hole_right_teach" / "session_1"
            dataset = root / "datasets" / "two_hole_right_teach" / "session_1"
            teaching.mkdir(parents=True)
            dataset.mkdir(parents=True)
            self.assertEqual(_dataset_dir(teaching), dataset)

    def test_foreground_depth_does_not_shift_coplanar_tip(self):
        intrinsic = {"fx": 100.0, "fy": 100.0, "ppx": 0.0, "ppy": 0.0, "distortion": []}
        head_from_color = {"translation_m": [0.0, 0.0, 1.0], "rotation_xyzw": [1.0, 0.0, 0.0, 0.0]}
        identity = {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "calibration.json").write_text(json.dumps({
                "camera": "GDK kHeadColor + kHeadDepth", "color_intrinsic": intrinsic,
                "head_from_color": head_from_color,
            }))
            sample = {
                "camera": "GDK kHeadColor + kHeadDepth RGB-D",
                "tf": {"mode": "exact_image_time"}, "base_from_head_link3": identity,
                "targets": {
                    "irregular_table_hole": {"class_id": 0, "point_base_link_m": [0.0, 0.0, 0.7]},
                    "round_hinge_tip_hole": {"class_id": 1, "point_base_link_m": [0.0, 0.0, 0.9],
                                              "center_head_color_px": [20.0, 0.0]},
                },
            }
            for index in range(12):
                (root / f"sample_{index:04d}.json").write_text(json.dumps(sample))
            pair = read_target_pair(root)
        self.assertAlmostEqual(pair["tip_base_link_m"][0], 0.06)
        self.assertAlmostEqual(pair["tip_base_link_m"][2], 0.7)
        self.assertAlmostEqual(pair["tip_raw_depth_z_disagreement_m"], 0.2)

    def test_pair_yaw_changes_endpoint_but_path_only_translates(self):
        old = {
            "midpoint_xy_m": [0.0, 0.0], "yaw_rad": 0.0, "spacing_m": 0.1,
            "table_base_link_m": [-0.05, 0.0, 0.7], "tip_base_link_m": [0.05, 0.0, 0.7],
        }
        angle = math.radians(20.0)
        new = {
            "midpoint_xy_m": [0.02, -0.01], "yaw_rad": angle, "spacing_m": 0.1,
            "table_base_link_m": [0.02 - 0.05*math.cos(angle), -0.01 - 0.05*math.sin(angle), 0.7],
            "tip_base_link_m": [0.02 + 0.05*math.cos(angle), -0.01 + 0.05*math.sin(angle), 0.7],
        }
        endpoint = (0.08, 0.02, 0.8)
        relocated, offset, info = endpoint_translation(old, new, endpoint)
        expected_end = (
            0.02 + 0.08*math.cos(angle) - 0.02*math.sin(angle),
            -0.01 + 0.08*math.sin(angle) + 0.02*math.cos(angle),
            0.8,
        )
        for measured, expected in zip(relocated, expected_end):
            self.assertAlmostEqual(measured, expected)
        self.assertAlmostEqual(info["pair_yaw_change_deg"], 20.0)

        # Use an actual valid right-arm document shape, with two distinct poses.
        from .cartesian_teach import CartesianSample, CartesianTrajectory, cartesian_document
        from .safety import RIGHT_ARM_JOINT_NAMES
        fixed = {f"idx0{i}_body_joint{i}": 0.0 for i in range(1, 6)}
        poses = (
            CartesianSample(0, (0.01, -0.04, 0.9), (0.0, 0.0, 0.0, 1.0), (0.0,)*7),
            CartesianSample(1_000_000_000, endpoint, (0.0, 0.0, 0.2, math.sqrt(0.96)), (0.0,)*7),
        )
        source = cartesian_document(CartesianTrajectory(
            "right", "base_link", "arm_r_end_link", RIGHT_ARM_JOINT_NAMES, fixed, poses,
        ), urdf_sha256="a"*64)
        shifted = translate_document(source, offset)
        for before, after in zip(source["samples"], shifted["samples"]):
            self.assertAlmostEqual(after["position_m"][0] - before["position_m"][0], offset[0])
            self.assertAlmostEqual(after["position_m"][1] - before["position_m"][1], offset[1])
            self.assertEqual(after["position_m"][2], before["position_m"][2])
            self.assertEqual(after["orientation_xyzw"], before["orientation_xyzw"])
        self.assertEqual(source["samples"][0]["position_m"], [0.01, -0.04, 0.9])


if __name__ == "__main__":
    unittest.main()
