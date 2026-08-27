import unittest

from control import G2Controller, MotionDisabledError
from safety import ARM_JOINT_NAMES


class FakeMotionStatus:
    mode = 0
    error_code = 0
    error_msg = ""


class FakeRequest:
    pass


class FakeGdk:
    JointControlReq = FakeRequest


class FakeRobot:
    def __init__(self):
        self.requests = []
        self.gripper_requests = []

    def get_whole_body_status(self):
        return {
            "right_arm_error": 0,
            "left_arm_error": 0,
            "right_end_error": 0,
            "left_end_error": 0,
            "waist_error": 0,
            "lift_error": 0,
            "neck_error": 0,
            "chassis_error": 0,
            "left_arm_estop": False,
            "right_arm_estop": False,
        }

    def get_motion_control_status(self):
        return FakeMotionStatus()

    def get_joint_states(self):
        return {
            "states": [
                {"name": name, "motor_position": 0.0}
                for name in reversed(ARM_JOINT_NAMES)
            ]
        }

    def joint_control_request(self, request):
        self.requests.append(request)

    def move_end_effector_joint(self, positions, velocities, group):
        self.gripper_requests.append((positions, velocities, group))


class ControlTests(unittest.TestCase):
    def test_read_only_gate_blocks_preset(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot)
        with self.assertRaises(MotionDisabledError):
            controller.move_preset("left", [0.0] * 7)
        self.assertEqual(robot.requests, [])

    def test_planning_request_uses_g2_names(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.move_preset("right", [0.0] * 7, velocity_rad_s=0.05)
        request = robot.requests[0]
        self.assertEqual(request.joint_names[0], "idx61_arm_r_joint1")
        self.assertEqual(request.joint_names[-1], "idx67_arm_r_joint7")
        self.assertEqual(request.joint_velocities, [0.05] * 7)

    def test_gripper_uses_gdk_group(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.move_gripper("left", -0.5)
        self.assertEqual(robot.gripper_requests[0], ([-0.5], [0.0], "left_tool"))


if __name__ == "__main__":
    unittest.main()
