import math
import threading
import time
import unittest

from control import CartesianPoseTarget, G2Controller, MotionDisabledError, quaternion_angle_deg, quaternion_slerp
from safety import ARM_JOINT_NAMES, SafetyError
from spacemouse_takeover import MotionPauseGate


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
        self.right_end_effector_pose = FakePose()
        self.end_effector_joint_names = []
        self.end_effector_joint_positions = []


class FakeGdk:
    JointControlReq = FakeRequest
    Impedance = FakeImpedance
    EndEffectorPose = FakeEndEffectorPose

    class EndEffectorControlGroup:
        kLeftArm = 0
        kRightArm = 1


class FakeRobot:
    def __init__(self):
        self.requests = []
        self.gripper_requests = []
        self.cartesian_impedance = []
        self.motion_mode = 1
        self.control_mode = 1
        self.left_end_pose = FakePose()
        self.right_end_pose = FakePose()
        self.left_gripper_position = -0.2
        self.right_gripper_position = -0.785
        self.follow_left_gripper = True
        self.follow_right_pose = True
        self.follow_right_gripper = True

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
        status.mode = self.motion_mode
        status.control_mode = self.control_mode
        status.frame_names = ["arm_r_end_link", "arm_l_end_link"]
        status.frame_poses = [self.right_end_pose, self.left_end_pose]
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
        if group == "left_tool" and self.follow_left_gripper:
            self.left_gripper_position = float(positions[0])
        elif group == "right_tool" and self.follow_right_gripper:
            self.right_gripper_position = float(positions[0])

    def move_arm_joint_servo(self, positions, period, mode, flag):
        self.requests.append((positions, period, mode, flag))

    def get_end_state(self):
        return {
            "left_end_state": {
                "names": ["left_omnipicker"],
                "end_states": [{"position": self.left_gripper_position}],
            },
            "right_end_state": {
                "names": ["right_omnipicker"],
                "end_states": [{"position": self.right_gripper_position}],
            },
        }

    def set_control_mode(self, mode):
        self.control_mode = mode

    def set_cartesian_impedance(self, impedances, group):
        self.cartesian_impedance.append((impedances, group))

    def end_effector_pose_control(self, request):
        self.requests.append(request)
        if request.group == FakeGdk.EndEffectorControlGroup.kLeftArm and request.end_effector_joint_names and self.follow_left_gripper:
            self.left_gripper_position = float(request.end_effector_joint_positions[0])
        if request.group == FakeGdk.EndEffectorControlGroup.kRightArm and self.follow_right_pose:
            source = request.right_end_effector_pose
            target = self.right_end_pose
            target.position.x, target.position.y, target.position.z = source.position.x, source.position.y, source.position.z
            target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = (
                source.orientation.x, source.orientation.y, source.orientation.z, source.orientation.w,
            )
            if request.end_effector_joint_names and self.follow_right_gripper:
                self.right_gripper_position = float(request.end_effector_joint_positions[0])


class ControlTests(unittest.TestCase):
    def test_quaternion_slerp_follows_shortest_unit_arc(self):
        quarter_turn = quaternion_slerp(
            (0.0, 0.0, 0.0, 1.0),
            (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
            0.5,
        )
        expected = math.sin(math.pi / 8.0)
        self.assertAlmostEqual(quarter_turn[2], expected, places=6)
        self.assertAlmostEqual(quarter_turn[3], math.cos(math.pi / 8.0), places=6)
        self.assertAlmostEqual(sum(value * value for value in quarter_turn), 1.0, places=6)

    def test_right_stream_can_opt_in_to_slerped_orientation(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        target = CartesianPoseTarget(
            (0.1, -0.2, 0.3),
            (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
        )
        stream = controller.start_right_end_effector_position_stream(
            initial, max_translation_velocity_mps=0.01,
            allow_orientation_change=True, max_angular_velocity_rad_s=1.0,
        )
        try:
            stream.update_target(target)
            self.assertTrue(stream.wait_until_target(2.0))
        finally:
            stream.stop()

        quaternions = [
            (
                request.right_end_effector_pose.orientation.x,
                request.right_end_effector_pose.orientation.y,
                request.right_end_effector_pose.orientation.z,
                request.right_end_effector_pose.orientation.w,
            )
            for request in robot.requests
        ]
        self.assertGreater(len(quaternions), 1)
        self.assertAlmostEqual(quaternions[-1][2], math.sin(math.pi / 4.0), places=6)
        self.assertAlmostEqual(quaternions[-1][3], math.cos(math.pi / 4.0), places=6)
        for first, second in zip(quaternions, quaternions[1:]):
            dot = abs(sum(a * b for a, b in zip(first, second)))
            self.assertLessEqual(2.0 * math.acos(min(1.0, dot)), 0.02 + 1e-6)

    def test_right_stream_still_rejects_orientation_without_opt_in(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_right_end_effector_position_stream(
            CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        )
        try:
            with self.assertRaises(SafetyError):
                stream.update_target(CartesianPoseTarget(
                    (0.1, -0.2, 0.3), (0.0, 0.0, 0.1, 0.9950371902),
                ))
        finally:
            stream.stop()

    def test_right_stream_accepts_float_roundoff_in_fixed_orientation(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_right_end_effector_position_stream(
            CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        )
        try:
            target_orientation = (0.0, 0.0, 1e-8, 1.0)
            self.assertLess(quaternion_angle_deg((0.0, 0.0, 0.0, 1.0), target_orientation), 0.01)
            stream.update_target(CartesianPoseTarget((0.1, -0.2, 0.3), target_orientation))
        finally:
            stream.stop()

    def test_right_stream_stops_when_control_mode_changes(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_right_end_effector_position_stream(
            CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        )
        try:
            robot.control_mode = 3
            stream.update_target(CartesianPoseTarget((0.101, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
            with self.assertRaises(SafetyError):
                stream.wait_until_target(0.5)
        finally:
            stream.stop()

    def test_read_only_gate_blocks_preset(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot)
        with self.assertRaises(MotionDisabledError):
            controller.move_preset("left", [0.0] * 7)
        self.assertEqual(robot.requests, [])

    def test_motion_pause_gate_blocks_automatic_send_but_manual_bypasses(self):
        robot = FakeRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        target = CartesianPoseTarget((0.01, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        gate.pause()
        finished = threading.Event()

        def automatic_send():
            controller._send_right_cartesian_pose(target, 0.10)
            finished.set()

        thread = threading.Thread(target=automatic_send)
        thread.start()
        self.assertFalse(finished.wait(0.03))
        self.assertEqual(robot.requests, [])
        with gate.manual_motion():
            controller._send_right_cartesian_pose(target, 0.10)
        self.assertEqual(len(robot.requests), 1)
        gate.resume()
        self.assertTrue(finished.wait(1.0))
        thread.join(1.0)

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

    def test_set_control_mode_switches_from_joint_impedance_to_position(self):
        robot = FakeRobot()
        robot.control_mode = 3
        controller = G2Controller(FakeGdk, robot, allow_motion=True)

        controller.set_control_mode(1)

        self.assertEqual(robot.control_mode, 1)
        self.assertEqual(robot.requests, [])

    def test_right_position_stream_allows_stopped_motion_state_before_first_command(self):
        robot = FakeRobot()
        robot.motion_mode = 0
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_right_end_effector_position_stream(initial)
        try:
            stream.update_target(CartesianPoseTarget((0.101, -0.2, 0.3), initial.orientation_xyzw))
            self.assertIsNotNone(stream.wait_until_actual_target(
                CartesianPoseTarget((0.101, -0.2, 0.3), initial.orientation_xyzw),
                0.6, position_tolerance_m=0.00001, orientation_tolerance_deg=0.1, stable_samples=2,
            ))
        finally:
            stream.stop()

    def test_right_position_stream_rejects_planning_motion_state(self):
        robot = FakeRobot()
        robot.motion_mode = 2
        controller = G2Controller(FakeGdk, robot, allow_motion=True)

        with self.assertRaises(SafetyError):
            controller.start_right_end_effector_position_stream(
                CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
            )

    def test_stage_advance_survives_background_stream_failure(self):
        from safety import AdvanceRightRequested
        for side in ("left", "right"):
            with self.subTest(side=side):
                gate = MotionPauseGate()
                controller = G2Controller(FakeGdk, FakeRobot(), allow_motion=True, motion_gate=gate)
                factory = getattr(controller, f"start_{side}_end_effector_position_stream")
                stream = factory(CartesianPoseTarget((0.1, 0.2, 0.3), (0, 0, 0, 1)))
                try:
                    gate.request_advance()
                    self.assertTrue(stream._stop.wait(1.0))
                    with self.assertRaises(AdvanceRightRequested):
                        stream.raise_if_failed()
                finally:
                    stream.stop()

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

    def test_left_gripper_pose_stream_keeps_command_alive_when_direct_command_is_ignored(self):
        robot = FakeRobot()
        robot.follow_left_gripper = False
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_left_gripper_position_stream(-0.5)
        try:
            time.sleep(0.05)
        finally:
            stream.stop()

        self.assertGreater(len(robot.requests), 0)
        request = robot.requests[-1]
        self.assertEqual(request.end_effector_joint_names, ["left_omnipicker"])
        self.assertEqual(request.end_effector_joint_positions, [-0.5])

    def test_right_end_effector_position_stream_can_synchronize_right_gripper(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_right_end_effector_position_stream(
            CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            rate_hz=50.0,
            life_time_s=0.10,
        )
        stream.set_right_gripper_target(0.0)
        time.sleep(0.05)
        stream.stop()

        request = robot.requests[-1]
        self.assertEqual(request.end_effector_joint_names, ["right_omnipicker"])
        self.assertEqual(request.end_effector_joint_positions, [0.0])
        self.assertAlmostEqual(robot.right_gripper_position, 0.0, places=6)

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

    def test_right_end_effector_position_stream_uses_right_group_and_slews(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_right_end_effector_position_stream(initial, max_translation_velocity_mps=0.01)
        stream.update_target(CartesianPoseTarget((0.101, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
        self.assertTrue(stream.wait_until_target(0.5))
        stream.stop()

        self.assertEqual(robot.requests[-1].group, 1)
        positions = [request.right_end_effector_pose.position.x for request in robot.requests]
        self.assertAlmostEqual(positions[-1], 0.101)
        self.assertTrue(all(abs(second - first) <= 0.0002001 for first, second in zip(positions, positions[1:])))

    def test_right_stream_requires_actual_feedback_to_settle(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_right_end_effector_position_stream(initial, max_translation_velocity_mps=0.01)
        try:
            stream.update_target(CartesianPoseTarget((0.101, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
            settled = stream.wait_until_actual_target(
                CartesianPoseTarget((0.101, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)), 0.6,
                position_tolerance_m=0.00001, orientation_tolerance_deg=0.1, stable_samples=3,
            )
        finally:
            stream.stop()
        self.assertIsNotNone(settled)

    def test_right_stream_does_not_treat_command_completion_as_actual_arrival(self):
        robot = FakeRobot()
        robot.follow_right_pose = False
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_right_end_effector_position_stream(initial, max_translation_velocity_mps=0.01)
        try:
            stream.update_target(CartesianPoseTarget((0.101, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
            self.assertTrue(stream.wait_until_target(0.5))
            settled = stream.wait_until_actual_target(
                CartesianPoseTarget((0.101, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)), 0.08,
                position_tolerance_m=0.00001, orientation_tolerance_deg=0.1, stable_samples=1,
            )
        finally:
            stream.stop()
        self.assertIsNone(settled)

    def test_cartesian_impedance_is_motion_gated(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot)
        with self.assertRaises(MotionDisabledError):
            controller.enable_left_cartesian_impedance([600.0, 600.0, 600.0, 30.0, 30.0, 30.0])
        self.assertEqual(robot.control_mode, 1)


if __name__ == "__main__":
    unittest.main()
