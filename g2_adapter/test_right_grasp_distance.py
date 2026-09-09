import unittest
import numpy as np
from right_grasp_distance import robust_pose_mean, workpiece_moved
from right_apriltag_grasp_gdk import distance_grasp_targets, RED_GRIP_CENTER_M
from apriltag_workpiece import GraspTargets


class DistanceGraspTests(unittest.TestCase):
    def test_movement_is_relative_to_locked_pose(self):
        reference = np.eye(4)
        current = np.eye(4)
        for x in [.001, .002, .004]:
            current[0, 3] = x
            self.assertFalse(workpiece_moved(reference, current, .005, 3)[0])
        current[0, 3] = .006
        self.assertTrue(workpiece_moved(reference, current, .005, 3)[0])
        current = np.eye(4)
        a = np.deg2rad(4)
        current[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
        self.assertTrue(workpiece_moved(reference, current, .005, 3)[0])

    def test_outliers_do_not_pull_target(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, :3, 3] = [.4, -.3, .8]
        mean = robust_pose_mean(poses)
        np.testing.assert_allclose(mean[:3, 3], [.4, -.3, .8])
        np.testing.assert_allclose(mean[:3, :3], np.eye(3))

    def test_unstable_window_rejected(self):
        poses = np.repeat(np.eye(4)[None], 3, axis=0)
        poses[:, 0, 3] = np.arange(3) * .01
        with self.assertRaises(ValueError):
            robust_pose_mean(poses)

    def test_152mm_is_grasp_not_extra_approach(self):
        end = np.eye(4)
        end[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        workpiece = np.eye(4)
        workpiece[:3, 3] = [.5, -.3, .8]
        result = distance_grasp_targets(GraspTargets(workpiece, end, end), 152)
        np.testing.assert_allclose(result.base_from_grasp_end, result.base_from_pregrasp_end)
        np.testing.assert_allclose(result.base_from_grasp_end[:3, 3] + [.152, 0, 0], workpiece[:3, 3] + RED_GRIP_CENTER_M)
