import sys
import unittest
from unittest.mock import patch

import record_pose


class RecordPoseTests(unittest.TestCase):
    def test_workflow_home_requires_both_arms_flag(self):
        with patch.object(sys, "argv", ["record_pose.py", "--name", "workflow_home"]):
            args = record_pose.parse_args()
        self.assertEqual(args.name, "workflow_home")
        self.assertFalse(args.both_arms)

    def test_normal_arm_pose_does_not_enable_bilateral_recording(self):
        with patch.object(sys, "argv", ["record_pose.py", "--name", "left_p3", "--both-arms"]):
            args = record_pose.parse_args()
        self.assertEqual(args.name, "left_p3")
        self.assertTrue(args.both_arms)


if __name__ == "__main__":
    unittest.main()
