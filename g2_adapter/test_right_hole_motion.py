import unittest

import numpy as np

from control import CartesianPoseTarget
from arm_orientation import fixed_forward_down_orientation
from right_hole_motion import _rotation_from_quaternion, end_target_for_tool_axis, right_hole_orientation


class RightHoleTargetTests(unittest.TestCase):
    def test_right_hole_uses_the_shared_full_flow_orientation(self):
        actual = right_hole_orientation(yaw_deg=20.0, pitch_deg=0.0, roll_deg=180.0)
        expected = fixed_forward_down_orientation(
            forward_yaw_deg=20.0, gripper_pitch_deg=0.0, gripper_roll_deg=180.0,
        )
        self.assertEqual(actual, expected)

    def test_places_cad_axis_origin_at_requested_hole_point(self):
        saved_end = CartesianPoseTarget((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        base_from_workpiece = np.eye(4)
        base_from_workpiece[:3, 3] = (1.1, 2.2, 3.3)

        target = end_target_for_tool_axis(
            saved_end, base_from_workpiece,
            tool_axis_target_base_m=(0.5, 0.6, 0.7),
            target_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )

        self.assertTrue(np.allclose(target.position_m, (0.4, 0.4, 0.4)))
        self.assertEqual(target.orientation_xyzw, (0.0, 0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
