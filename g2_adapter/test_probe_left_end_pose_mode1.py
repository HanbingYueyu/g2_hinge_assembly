import unittest

from probe_left_end_pose_mode1 import pose_from_status


class Point:
    x = 0.4
    y = 0.2
    z = 1.1


class Quaternion(Point):
    w = 0.7


class Pose:
    position = Point()
    orientation = Quaternion()


class Status:
    frame_names = ["arm_r_end_link", "arm_l_end_link"]
    frame_poses = [Pose(), Pose()]


class ProbePoseTests(unittest.TestCase):
    def test_reads_named_end_link_pose_from_motion_status(self):
        pose = pose_from_status(Status(), "arm_l_end_link")
        self.assertEqual(pose.position_m, (0.4, 0.2, 1.1))
        self.assertEqual(pose.orientation_xyzw, (0.4, 0.2, 1.1, 0.7))


if __name__ == "__main__":
    unittest.main()
