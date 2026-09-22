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

    def test_staged_flow_pose_names_accept_an_explicit_side(self):
        for name, side in (("LR1", "right"), ("LR2", "right"), ("RR1", "right"), ("RR2", "right"), ("LL1", "left"), ("LL2", "left")):
            with self.subTest(name=name):
                with patch.object(sys, "argv", ["record_pose.py", "--name", name, "--side", side]):
                    args = record_pose.parse_args()
                self.assertEqual(args.name, name)
                self.assertEqual(args.side, side)

    def test_pose_side_maps_staged_names_to_their_recorded_arm(self):
        self.assertEqual(record_pose.pose_side("LL1"), "left")
        self.assertEqual(record_pose.pose_side("LL2"), "left")
        self.assertEqual(record_pose.pose_side("LR1"), "right")
        self.assertEqual(record_pose.pose_side("LR2"), "right")
        self.assertEqual(record_pose.pose_side("RR1"), "right")
        self.assertEqual(record_pose.pose_side("RR2"), "right")
        self.assertEqual(record_pose.pose_side("left_p3"), "left")
        self.assertEqual(record_pose.pose_side("right_pr1"), "right")
        self.assertEqual(record_pose.pose_side("workflow_home"), "both")

    def test_side_mismatch_is_rejected_before_any_gdk_call(self):
        with patch.object(sys, "argv", ["record_pose.py", "--name", "RR1", "--side", "left"]):
            self.assertEqual(record_pose.main(), 2)
        with patch.object(sys, "argv", ["record_pose.py", "--name", "LL1", "--side", "right", "--save"]):
            self.assertEqual(record_pose.main(), 2)


if __name__ == "__main__":
    unittest.main()
