import time
import unittest

from control import CartesianPoseTarget, G2Controller, MotionDisabledError
from safety import ARM_JOINT_NAMES


class FakeMotionStatus:
    mode = 0
    control_mode = 1
    error_code = 0
    error_msg = ""


class FakeRequest:
    pass


class FakeImpedance:
    def __init__(self, *, kp):
        self.kp = kp


class FakePoint:
    x = 0.0
    y = 0.0
    z = 0.0


class FakeQuaternion(FakePoint):
    w = 1.0


class FakePose:
    def __init__(self):
        self.position = FakePoint()
        self.orientation = FakeQuaternion()


class FakeEndEffectorPose:
    def __init__(self):
        self.life_time = 0.0
        self.group = None
        self.left_end_effector_pose = FakePose()
        self.end_effector_joint_names = []
        self.end_effector_joint_positions = []


class FakeGdk:
    JointControlReq = FakeRequest
    Impedance = FakeImpedance
    EndEffectorPose = FakeEndEffectorPose

    class EndEffectorControlGroup:
        kLeftArm = 0


class FakeRobot:
    def __init__(self):
        self.requests = []
        self.gripper_requests = []
        self.cartesian_impedance = []
        self.control_mode = 1

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
        status = FakeMotionStatus()
        status.control_mode = self.control_mode
        return status

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

    def move_arm_joint_servo(self, positions, period, mode, flag):
        self.requests.append((positions, period, mode, flag))

    def get_end_state(self):
        return {
            "left_end_state": {
                "names": ["left_omnipicker"],
                "end_states": [{"position": -0.2}],
            }
        }

    def set_control_mode(self, mode):
        self.control_mode = mode

    def set_cartesian_impedance(self, impedances, group):
        self.cartesian_impedance.append((impedances, group))

    def end_effector_pose_control(self, request):
        self.requests.append(request)


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

    def test_servo_step_and_gripper_read(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.servo_step([0.0] * 14)
        self.assertEqual(robot.requests[0][0], [0.0] * 14)
        self.assertEqual(controller.read_gripper_position("left"), -0.2)

    def test_servo_to_honors_explicit_safe_duration(self):
        robot = FakeRobot()
        controller = G2Controller(
            FakeGdk,
            robot,
            allow_motion=True,
            control_period_s=0.02,
            max_servo_velocity_rad_s=0.15,
        )
        sent = controller.servo_to([0.005] * 14, duration_s=0.05)

        self.assertEqual(sent, 3)
        self.assertEqual(len(robot.requests), 3)
        self.assertEqual(robot.requests[-1][0], [0.005] * 14)

    def test_servo_waypoints_reaches_final_target_within_requested_duration(self):
        robot = FakeRobot()
        controller = G2Controller(
            FakeGdk,
            robot,
            allow_motion=True,
            control_period_s=0.02,
            max_servo_velocity_rad_s=0.15,
        )
        sent = controller.servo_waypoints(
            ([0.002] * 14, [0.004] * 14),
            duration_s=0.06,
        )

        self.assertEqual(sent, 3)
        self.assertEqual(robot.requests[-1][0], [0.004] * 14)

    def test_cartesian_impedance_uses_mode_2_and_50hz_stream(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.enable_left_cartesian_impedance([600.0, 600.0, 600.0, 30.0, 30.0, 30.0])
        self.assertEqual(robot.control_mode, 2)
        self.assertEqual([item.kp for item in robot.cartesian_impedance[0][0]], [600.0, 600.0, 600.0, 30.0, 30.0, 30.0])
        self.assertEqual(robot.cartesian_impedance[0][1], 0)
        stream = controller.start_left_cartesian_impedance_stream(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        )
        time.sleep(0.05)
        stream.stop()
        self.assertGreaterEqual(len(robot.requests), 1)
        request = robot.requests[-1]
        self.assertEqual(request.group, 0)
        self.assertEqual(request.left_end_effector_pose.position.z, 0.3)

    def test_end_effector_position_stream_uses_mode_1_without_switching_modes(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_left_end_effector_position_stream(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            rate_hz=50.0,
            life_time_s=0.10,
        )
        time.sleep(0.05)
        stream.stop()

        self.assertEqual(robot.control_mode, 1)
        self.assertGreaterEqual(len(robot.requests), 1)
        request = robot.requests[-1]
        self.assertEqual(request.group, 0)
        self.assertEqual(request.left_end_effector_pose.position.x, 0.1)
        self.assertEqual(request.left_end_effector_pose.position.z, 0.3)

    def test_end_effector_position_stream_can_synchronize_left_gripper(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_left_end_effector_position_stream(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            rate_hz=50.0,
            life_time_s=0.10,
        )
        stream.set_left_gripper_target(0.0)
        time.sleep(0.05)
        stream.stop()

        request = robot.requests[-1]
        self.assertEqual(request.end_effector_joint_names, ["left_omnipicker"])
        self.assertEqual(request.end_effector_joint_positions, [0.0])

    def test_single_end_effector_position_setpoint_requires_mode_1(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.send_left_end_effector_position_setpoint(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            life_time_s=0.10,
        )

        self.assertEqual(robot.control_mode, 1)
        request = robot.requests[-1]
        self.assertEqual(request.group, 0)
        self.assertEqual(request.left_end_effector_pose.position.y, 0.2)

    def test_reads_left_end_link_pose_from_motion_status_frames(self):
        robot = FakeRobot()
        status = FakeMotionStatus()
        status.frame_names = ["arm_r_end_link", "arm_l_end_link"]
        right_pose = FakePose()
        left_pose = FakePose()
        left_pose.position.x, left_pose.position.y, left_pose.position.z = (0.56, 0.39, 0.95)
        left_pose.orientation.x, left_pose.orientation.y, left_pose.orientation.z, left_pose.orientation.w = (0.6, -0.6, 0.2, -0.4)
        status.frame_poses = [right_pose, left_pose]
        robot.get_motion_control_status = lambda: status
        controller = G2Controller(FakeGdk, robot)

        result = controller.read_end_effector_pose("arm_l_end_link")

        self.assertEqual(result.position_m, (0.56, 0.39, 0.95))
        self.assertEqual(result.orientation_xyzw, (0.6, -0.6, 0.2, -0.4))

    def test_end_effector_position_stream_slews_translation_at_configured_rate(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_left_end_effector_position_stream(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            rate_hz=50.0,
            life_time_s=0.10,
            max_translation_velocity_mps=0.01,
        )
        stream.update_target(CartesianPoseTarget((0.101, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
        self.assertTrue(stream.wait_until_target(0.5))
        stream.stop()

        x_positions = [request.left_end_effector_pose.position.x for request in robot.requests]
        self.assertAlmostEqual(x_positions[-1], 0.101)
        self.assertTrue(all(abs(second - first) <= 0.0002001 for first, second in zip(x_positions, x_positions[1:])))

    def test_cartesian_impedance_is_motion_gated(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot)
        with self.assertRaises(MotionDisabledError):
            controller.enable_left_cartesian_impedance([600.0, 600.0, 600.0, 30.0, 30.0, 30.0])
        self.assertEqual(robot.control_mode, 1)


if __name__ == "__main__":
    unittest.main()
