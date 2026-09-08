import unittest

import numpy as np

from apriltag_workpiece import (
    TagGeometry,
    RollingWorkpiecePoseEstimator,
    TAG_OBJECT_POINTS,
    grasp_tcp_targets,
    g2_grasp_targets,
    pose_distance,
    quaternion_from_rotation,
    right_end_from_grasp_tcp_candidate,
    constrained_pregrasp_end_candidate,
    workpiece_from_pregrasp_end_candidate,
    inverse_transform_dict,
    transform_from_dict,
)


class AprilTagWorkpieceTests(unittest.TestCase):
    def test_default_pnp_reprojection_gate_is_three_pixels(self):
        estimator = RollingWorkpiecePoseEstimator()
        self.assertAlmostEqual(estimator.max_reprojection_error_px, 3.0)

    def test_independent_tag_geometry_profile_builds_object_points(self):
        geometry = TagGeometry.from_mapping({
            "tag_family": "tagStandard41h12",
            "tag_size_m": 0.008333333333333334,
            "centers_m": {
                "0": [0.0, -0.0415, -0.0075],
                "1": [-0.03475, -0.05148543023, -0.01379941798],
                "2": [0.03475, -0.05148543023, -0.01379941798],
            },
            "u": [1.0, 0.0, 0.0],
            "v": {
                "0": [0.0, 0.0, 1.0],
                "1": [0.0, 0.57746472062681, 0.81641563950687],
                "2": [0.0, 0.57746472062681, 0.81641563950687],
            },
        })
        self.assertTrue(np.allclose(geometry.centers_m[0], [0.0, -0.0415, -0.0075]))
        self.assertAlmostEqual(
            np.linalg.norm(geometry.object_corners(1)[1] - geometry.object_corners(1)[0]),
            0.008333333333333334,
        )

    def test_rolling_pnp_uses_supplied_geometry_profile(self):
        import cv2

        geometry = TagGeometry.from_mapping({
            "tag_size_m": 0.008333333333333334,
            "centers_m": {
                "0": [0.0, -0.0415, -0.0075],
                "1": [-0.03475, -0.05148543023, -0.01379941798],
                "2": [0.03475, -0.05148543023, -0.01379941798],
            },
            "u": [1.0, 0.0, 0.0],
            "v": {
                "0": [0.0, 0.0, 1.0],
                "1": [0.0, 0.57746472062681, 0.81641563950687],
                "2": [0.0, 0.57746472062681, 0.81641563950687],
            },
        })
        camera = np.asarray(((700.0, 0.0, 640.0), (0.0, 700.0, 528.0), (0.0, 0.0, 1.0)))
        rvec = np.asarray((0.08, -0.12, 0.04))
        translation = np.asarray((0.02, -0.01, 0.55))
        estimator = RollingWorkpiecePoseEstimator(sample_count=2, geometry=geometry)
        for frame in range(2):
            detections = {}
            for tag_id, points in geometry.object_points.items():
                image, _ = cv2.projectPoints(points, rvec, translation, camera, np.zeros(5))
                detections[tag_id] = image.reshape(4, 2)
            estimator.add(detections, timestamp_s=1.0 + frame * 0.02)
        result = estimator.estimate(camera, np.zeros(5), now_s=1.04)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(np.allclose(result.camera_from_workpiece[:3, 3], translation, atol=1e-5))
        self.assertLess(result.mean_reprojection_error_px, 1e-5)

    def test_cad_tag_geometry_keeps_original_effective_size(self):
        self.assertAlmostEqual(np.linalg.norm(TAG_OBJECT_POINTS[0][1] - TAG_OBJECT_POINTS[0][0]), 0.015 * 5.0 / 9.0)

    def test_default_geometry_uses_electric_screwdriver_centers(self):
        self.assertTrue(np.allclose(TAG_OBJECT_POINTS[0].mean(axis=0), [0, -0.0415, -0.0075]))
        self.assertTrue(np.allclose(TAG_OBJECT_POINTS[1].mean(axis=0), [-0.03475, -0.05148543023, -0.01379941798]))

    def test_g2_targets_keep_the_30mm_pregrasp_distance(self):
        identity = np.eye(4)
        targets = g2_grasp_targets(identity, identity, identity, identity, grasp_tcp_z_offset_m=0.0)
        distance, angle = pose_distance(targets.base_from_pregrasp_end, targets.base_from_grasp_end)
        self.assertAlmostEqual(distance, 0.03)
        self.assertAlmostEqual(angle, 0.0)

    def test_virtual_tcp_targets_are_independent_of_g2_tool_mapping(self):
        pregrasp, grasp = grasp_tcp_targets(np.eye(4), grasp_tcp_z_offset_m=0.0)
        distance, angle = pose_distance(pregrasp, grasp)
        self.assertAlmostEqual(distance, 0.03)
        self.assertAlmostEqual(angle, 0.0)

    def test_virtual_tcp_default_compensation_keeps_the_30mm_pregrasp_distance(self):
        pregrasp, grasp = grasp_tcp_targets(np.eye(4))
        distance, angle = pose_distance(pregrasp, grasp)
        self.assertAlmostEqual(distance, 0.03)
        self.assertAlmostEqual(angle, 0.0)

    def test_tcp_calibration_candidate_reconstructs_the_virtual_grasp_pose(self):
        base_from_end = np.eye(4)
        base_from_workpiece = np.eye(4)
        end_from_tcp, _pre, base_from_grasp = right_end_from_grasp_tcp_candidate(
            base_from_end, base_from_workpiece, grasp_tcp_z_offset_m=0.0,
        )
        self.assertTrue(np.allclose(base_from_end @ end_from_tcp, base_from_grasp))

    def test_pregrasp_candidate_is_relative_to_the_workpiece(self):
        base_from_workpiece = np.eye(4)
        base_from_workpiece[:3, 3] = (0.4, -0.2, 0.8)
        base_from_end = np.eye(4)
        base_from_end[:3, 3] = (0.35, -0.25, 0.82)
        workpiece_from_end = workpiece_from_pregrasp_end_candidate(base_from_workpiece, base_from_end)
        self.assertTrue(np.allclose(base_from_workpiece @ workpiece_from_end, base_from_end))

    def test_constrained_pregrasp_uses_grasp_orientation_and_approach_axis(self):
        grasp = np.eye(4)
        grasp[:3, 3] = (0.6, 0.1, 0.9)
        raw = np.eye(4)
        raw[:3, 3] = (0.6, 0.1, 0.86)  # 40 mm behind grasp along -Z.
        candidate, distance_m, lateral_m, orientation_deg = constrained_pregrasp_end_candidate(raw, grasp)
        self.assertTrue(np.allclose(candidate[:3, :3], grasp[:3, :3]))
        self.assertTrue(np.allclose(candidate[:3, 3], raw[:3, 3]))
        self.assertAlmostEqual(distance_m, 0.04)
        self.assertAlmostEqual(lateral_m, 0.0)
        self.assertAlmostEqual(orientation_deg, 0.0)

    def test_gdk_transform_and_quaternion_conversion(self):
        transform = transform_from_dict({"translation_m": [1, 2, 3], "rotation_xyzw": [0, 0, 0, 2]})
        self.assertTrue(np.allclose(transform[:3, 3], [1, 2, 3]))
        self.assertTrue(np.allclose(quaternion_from_rotation(np.eye(3)), [0, 0, 0, 1]))

    def test_inverse_transform_dict_round_trips_a_nontrivial_pose(self):
        raw = {"translation_m": [0.48, -0.84, -0.13], "rotation_xyzw": [-0.24, -0.72, -0.60, 0.24]}
        self.assertTrue(np.allclose(
            transform_from_dict(raw) @ transform_from_dict(inverse_transform_dict(raw)), np.eye(4), atol=1e-8,
        ))

    def test_rolling_pnp_recovers_a_known_workpiece_pose(self):
        import cv2

        camera = np.asarray(((700.0, 0.0, 640.0), (0.0, 700.0, 528.0), (0.0, 0.0, 1.0)))
        rvec = np.asarray((0.08, -0.12, 0.04))
        translation = np.asarray((0.02, -0.01, 0.55))
        estimator = RollingWorkpiecePoseEstimator(sample_count=3)
        for frame in range(3):
            detections = {}
            for tag_id, points in TAG_OBJECT_POINTS.items():
                image, _ = cv2.projectPoints(points, rvec, translation, camera, np.zeros(5))
                detections[tag_id] = image.reshape(4, 2)
            estimator.add(detections, timestamp_s=1.0 + frame * 0.02)
        result = estimator.estimate(camera, np.zeros(5), now_s=1.04)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(np.allclose(result.camera_from_workpiece[:3, 3], translation, atol=1e-5))
        self.assertLess(result.mean_reprojection_error_px, 1e-5)

    def test_any_two_current_tags_recover_pose_but_one_tag_does_not(self):
        import cv2
        camera = np.array([[700., 0, 640], [0, 700., 528], [0, 0, 1.]])
        translation = np.array([0.02, -0.01, 0.55])
        detections = {tag: cv2.projectPoints(points, np.array([0.08, -0.12, 0.04]), translation, camera, np.zeros(5))[0].reshape(4, 2)
                      for tag, points in TAG_OBJECT_POINTS.items()}
        for pair in ((0, 1), (0, 2), (1, 2)):
            with self.subTest(pair=pair):
                estimator = RollingWorkpiecePoseEstimator(sample_count=1)
                estimator.add(detections, timestamp_s=1.0)
                estimator.add({tag: detections[tag] for tag in pair}, timestamp_s=1.02)
                pose = estimator.estimate(camera, now_s=1.02)
                self.assertIsNotNone(pose)
                self.assertEqual(pose.inlier_count, 8)
                self.assertTrue(np.allclose(pose.camera_from_workpiece[:3, 3], translation, atol=1e-5))
                estimator.add({pair[0]: detections[pair[0]]}, timestamp_s=1.04)
                self.assertIsNone(estimator.estimate(camera, now_s=1.04))


if __name__ == "__main__":
    unittest.main()
