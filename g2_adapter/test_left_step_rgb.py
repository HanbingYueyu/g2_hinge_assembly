import unittest
from pathlib import Path

import cv2
import numpy as np

from extract_left_step_profile import extract_marked
from left_step_rgb import detect_hole_centres, match_holes, solve_correspondences, OcclusionTracker, OcclusionTrackerPool


class LeftStepRgbTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = extract_marked(Path(__file__).resolve().parents[1] / "models/顶铰链.stp")
        cls.points = np.array(cls.profile["hole_centers_m"])
        # Synthetic target is for testing the transform, not a production grasp.
        cls.target = cls.points.mean(axis=0) + [0, .02, .001]
        cls.profile["grasp_center_m"] = cls.target.tolist()
        cls.intrinsic = dict(fx=800, fy=810, ppx=640, ppy=480, distortion=[.03, -.01, 0, 0, 0])
        cls.k = np.array([[800., 0, 640], [0, 810, 480], [0, 0, 1]])
        cls.rvec = np.array([np.pi - .12, -.08, .3])
        cls.rotation = cv2.Rodrigues(cls.rvec)[0]
        cls.tvec = np.array([0, 0, .35]) - cls.rotation @ cls.points.mean(axis=0)
        cls.pixels = cv2.projectPoints(cls.points, cls.rvec, cls.tvec, cls.k, np.array(cls.intrinsic["distortion"]))[0].reshape(-1, 2)

    def test_cad_holes_have_metric_dimensions(self):
        self.assertEqual(self.points.shape, (7, 3))
        self.assertTrue(.03 < np.ptp(self.points[:, 0]) < .05)
        self.assertTrue(.02 < np.ptp(self.points[:, 1]) < .04)
        self.assertFalse(self.profile["verified"])

    def test_occlusion_requires_initial_lock_but_does_not_expire(self):
        tracker = OcclusionTracker()
        partial = self.pixels[[0, 1, 4, 5]]
        self.assertFalse(tracker.update(self.profile, partial, self.intrinsic, 0)["accepted"])
        self.assertTrue(tracker.update(self.profile, self.pixels, self.intrinsic, .1)["accepted"])
        result = tracker.update(self.profile, partial, self.intrinsic, .2)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["reason"], "occlusion_tracked")
        self.assertFalse(tracker.update(self.profile, partial[:3], self.intrinsic, .3)["accepted"])
        self.assertTrue(tracker.update(self.profile, partial, self.intrinsic, 60)["accepted"])

    def test_pool_retains_lock_across_missing_detections(self):
        pool = OcclusionTrackerPool()
        pool.update(self.profile, [self.pixels], [[100, 100]], self.intrinsic, 0)
        self.assertEqual(pool.update(self.profile, [], [], self.intrinsic, 30), [])
        result = pool.update(self.profile, [self.pixels[[0, 1, 4, 5]]], [[100, 100]], self.intrinsic, 60)
        self.assertTrue(result[0]["accepted"], result)
        self.assertEqual(result[0]["reason"], "occlusion_tracked")

    def test_tracker_does_not_reuse_yolo_index(self):
        pool = OcclusionTrackerPool()
        pool.update(self.profile, [self.pixels], [[100, 100]], self.intrinsic, 0)
        result = pool.update(self.profile, [self.pixels[[0, 1, 4, 5]]], [[300, 100]], self.intrinsic, .1)
        self.assertFalse(result[0]["accepted"])

    def test_occluded_jump_is_not_a_new_lock(self):
        tracker = OcclusionTracker()
        tracker.update(self.profile, self.pixels, self.intrinsic, 0)
        self.assertFalse(tracker.update(self.profile, self.pixels[[0, 1, 4, 5]] + 25, self.intrinsic, .1)["accepted"])

    def test_red_faces_define_midpoint(self):
        profile = extract_marked(Path(__file__).resolve().parents[1] / "models/顶铰链.stp")
        self.assertEqual(len(profile["red_face_entities"]), 2)
        centres = np.asarray(profile["red_face_extent_centers_mm"])
        np.testing.assert_allclose(profile["grasp_center_m"], centres.mean(axis=0) / 1000)
        self.assertTrue(35 < np.linalg.norm(centres[0] - centres[1]) < 40)

    def test_pnp_recovers_metric_target_without_depth(self):
        result = solve_correspondences(self.points, self.pixels, self.intrinsic, self.target)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result["grasp_xyz_camera_m"], self.rotation @ self.target + self.tvec, atol=1e-5)

    def test_unordered_candidates_with_distractors(self):
        pixels = np.vstack([self.pixels[[3, 0, 6, 1, 5, 2, 4]], [[400, 300], [800, 650]]])
        result = match_holes(self.profile, pixels, self.intrinsic)
        self.assertTrue(result["accepted"], result)
        np.testing.assert_allclose(result["grasp_xyz_camera_m"], self.rotation @ self.target + self.tvec, atol=1e-5)
        self.assertFalse(result["motion_enabled"])

    def test_missing_holes_and_bad_correspondence_rejected(self):
        self.assertFalse(match_holes(self.profile, self.pixels[:5], self.intrinsic)["accepted"])
        wrong = self.pixels.copy()
        wrong[2] += [30, -20]
        self.assertIsNone(solve_correspondences(self.points, wrong, self.intrinsic, self.target))

    def test_each_single_missing_hole(self):
        for missing in range(7):
            with self.subTest(missing=missing):
                pixels = np.delete(self.pixels, missing, axis=0)[::-1]
                result = match_holes(self.profile, pixels, self.intrinsic)
                self.assertTrue(result["accepted"], result)
                self.assertEqual(result["matched_count"], 6)
                self.assertNotIn(missing, result["matched_model_indices"])
                np.testing.assert_allclose(result["grasp_xyz_camera_m"], self.rotation @ self.target + self.tvec, atol=1e-5)

    def test_six_holes_with_false_candidates(self):
        pixels = np.vstack([self.pixels[1:], [[400, 300], [800, 650]]])
        result = match_holes(self.profile, pixels, self.intrinsic)
        self.assertTrue(result["accepted"], result)
        self.assertEqual(result["matched_count"], 6)
        self.assertNotIn(0, result["matched_model_indices"])

    def test_two_valid_patterns_are_ambiguous(self):
        pixels = np.vstack([self.pixels[:6], self.pixels[:6] + [180, 0]])
        result = match_holes(self.profile, pixels, self.intrinsic)
        self.assertFalse(result["accepted"], result)
        self.assertEqual(result["reason"], "ambiguous_pose")

    def test_rgb_holes_retained_when_mask_has_holes(self):
        image = np.full((400, 400, 3), 200, np.uint8)
        mask = np.zeros((400, 400), np.uint8)
        mask[20:380, 20:380] = 1
        centres = [(100, 100), (280, 110), (120, 260)]
        for centre in centres:
            cv2.circle(image, centre, 14, (20, 20, 20), -1)
            cv2.circle(mask, centre, 14, 0, -1)
        detected = detect_hole_centres(image, mask)
        self.assertEqual(len(detected), 3)
        for centre in centres:
            self.assertLess(np.linalg.norm(detected - centre, axis=1).min(), 1)


if __name__ == "__main__":
    unittest.main()
