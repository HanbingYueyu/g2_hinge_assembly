import unittest
from unittest.mock import Mock

from control import CartesianPoseTarget
from arm_orientation import fixed_forward_down_orientation
from set_arm_orientation_gdk import orientation_target, rotate_arm_orientation


class SetArmOrientationTests(unittest.TestCase):
    def test_target_keeps_current_xyz(self):
        current = CartesianPoseTarget((0.41, -0.22, 0.73), (0.0, 0.0, 0.0, 1.0))
        target = orientation_target(current, side="right", yaw_deg=20.0, pitch_deg=0.0, roll_deg=180.0)
        self.assertEqual(target.position_m, current.position_m)

    def test_left_side_uses_the_same_orientation_convention_as_full_flow(self):
        current = CartesianPoseTarget((0.41, -0.22, 0.73), (0.0, 0.0, 0.0, 1.0))
        target = orientation_target(
            current, side="left", yaw_deg=0.0, pitch_deg=0.0, roll_deg=180.0,
        )
        expected = fixed_forward_down_orientation(
            forward_yaw_deg=0.0, gripper_pitch_deg=0.0, gripper_roll_deg=180.0,
        )
        self.assertEqual(target.orientation_xyzw, expected)

    def test_right_side_uses_the_same_orientation_convention_as_left(self):
        current = CartesianPoseTarget((0.41, -0.22, 0.73), (0.0, 0.0, 0.0, 1.0))
        target = orientation_target(
            current, side="right", yaw_deg=20.0, pitch_deg=0.0, roll_deg=180.0,
        )
        expected = fixed_forward_down_orientation(
            forward_yaw_deg=20.0, gripper_pitch_deg=0.0, gripper_roll_deg=180.0,
        )
        self.assertEqual(target.orientation_xyzw, expected)

    def test_right_side_uses_right_stream_and_waits_for_actual_arrival(self):
        current = CartesianPoseTarget((0.41, -0.22, 0.73), (0.0, 0.0, 0.0, 1.0))
        controller = Mock()
        controller.read_end_effector_pose.return_value = current
        stream = controller.start_right_end_effector_position_stream.return_value
        stream.wait_until_actual_target.return_value = current

        result = rotate_arm_orientation(
            controller, side="right", yaw_deg=20.0, pitch_deg=0.0,
            roll_deg=180.0, angular_speed_deg_s=10.0,
        )

        target = stream.update_target.call_args.args[0]
        self.assertEqual(target.position_m, current.position_m)
        self.assertEqual(result["side"], "right")
        stream.wait_until_actual_target.assert_called_once()
        stream.stop.assert_called_once()


if __name__ == "__main__":
    unittest.main()
