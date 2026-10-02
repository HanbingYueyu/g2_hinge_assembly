import math
import threading
import time
import unittest

from control import (
    CartesianPoseTarget,
    G2Controller,
    LeftEndEffectorPositionStream,
    MotionDisabledError,
    MotionRebasePending,
    RightEndEffectorPositionStream,
    quaternion_angle_deg,
    quaternion_slerp,
)
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
        self.joint_impedance_calls = []
        self.joint_impedance_state = [FakeImpedance(kp=0.0) for _ in range(14)]
        self.joint_impedance_result = 0
        self.joint_impedance_readback = None
        self.set_control_mode_result = 0
        self.motion_mode = 1
        self.control_mode = 1
        self.left_end_pose = FakePose()
        self.right_end_pose = FakePose()
        self.left_gripper_position = -0.2
        self.right_gripper_position = -0.785
        self.follow_left_pose = True
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
        return self.set_control_mode_result

    def set_cartesian_impedance(self, impedances, group):
        self.cartesian_impedance.append((impedances, group))
        return 0

    def set_joint_impedance(self, impedances, group):
        self.joint_impedance_calls.append((impedances, group))
        if self.joint_impedance_result != 0:
            return self.joint_impedance_result
        offset = 0 if group == 0 else 7
        for index, item in enumerate(impedances):
            self.joint_impedance_state[offset + index] = FakeImpedance(kp=item.kp)
        return 0

    def get_joint_impedance(self):
        if self.joint_impedance_readback is not None:
            return [FakeImpedance(kp=value) for value in self.joint_impedance_readback]
        return [FakeImpedance(kp=item.kp) for item in self.joint_impedance_state]

    def end_effector_pose_control(self, request):
        self.requests.append(request)
        if request.group == FakeGdk.EndEffectorControlGroup.kLeftArm:
            if self.follow_left_pose:
                source = request.left_end_effector_pose
                target = self.left_end_pose
                target.position.x, target.position.y, target.position.z = source.position.x, source.position.y, source.position.z
                target.orientation.x, target.orientation.y, target.orientation.z, target.orientation.w = (
                    source.orientation.x, source.orientation.y, source.orientation.z, source.orientation.w,
                )
            if request.end_effector_joint_names and self.follow_left_gripper:
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
    def test_preset_custom_joint_impedance_is_active_before_planner_submission(self):
        custom_kp = (50, 50, 50, 60, 30, 30, 30)
        robot = FakeRobot()
        observed = []
        def submit(request):
            observed.append((robot.control_mode, [
                (group, tuple(value.kp for value in values))
                for values, group in robot.joint_impedance_calls
            ]))
            robot.requests.append(request)
        robot.joint_control_request = submit
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.move_preset("right", [0.0] * 7, joint_impedance_kp=custom_kp)
        self.assertEqual(observed, [(3, [(1, custom_kp)])])
        controller.move_preset("right", [0.0] * 7)
        self.assertEqual(observed[-1], (3, [
            (1, custom_kp), (1, (400, 400, 400, 500, 300, 200, 200)),
        ]))

    def test_prepare_joint_impedance_enters_mode_3_and_configures_both_arms(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)

        controller.prepare_joint_impedance_control()

        self.assertEqual(robot.control_mode, 3)
        self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 1])
        self.assertEqual(
            [[item.kp for item in values] for values, _ in robot.joint_impedance_calls],
            [[400, 400, 400, 500, 300, 200, 200]] * 2,
        )

    def test_prepare_joint_impedance_is_idempotent_while_mode_3_stays_healthy(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.prepare_joint_impedance_control()
        controller.prepare_joint_impedance_control()
        self.assertEqual(len(robot.joint_impedance_calls), 2)

    def test_formal_motion_interfaces_prepare_joint_impedance(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)

        controller.move_preset("left", [0.0] * 7)
        self.assertEqual(robot.control_mode, 3)

        controller.servo_step([0.0] * 14)
        stream = controller.start_left_end_effector_position_stream(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        )
        stream.stop()

        self.assertEqual(robot.control_mode, 3)
        self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 1])

    def test_read_only_controller_construction_does_not_change_mode(self):
        robot = FakeRobot()
        G2Controller(FakeGdk, robot, allow_motion=False)
        self.assertEqual(robot.control_mode, 1)
        self.assertEqual(robot.joint_impedance_calls, [])

    def test_set_joint_impedance_rejects_failed_write_without_caching(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        robot.joint_impedance_result = 7

        with self.assertRaisesRegex(SafetyError, "set_joint_impedance failed"):
            controller.set_joint_impedance("left", (400.0,) * 7)

        self.assertNotIn("left", controller._configured_joint_impedance)

    def test_set_joint_impedance_detects_readback_mismatch(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        robot.joint_impedance_readback = [0.0] * 14

        with self.assertRaisesRegex(SafetyError, "readback mismatch"):
            controller.set_joint_impedance("left", (400.0,) * 7)

        self.assertNotIn("left", controller._configured_joint_impedance)

    def test_set_control_mode_rejects_failed_switch(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        robot.set_control_mode_result = 5

        with self.assertRaisesRegex(SafetyError, "set_control_mode"):
            controller.set_control_mode(3)

    def test_prepare_rejects_invalid_gains_and_side(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)

        with self.assertRaisesRegex(SafetyError, "seven positive finite"):
            controller.prepare_joint_impedance_control(("left",), (0.0,) * 7)
        with self.assertRaisesRegex(SafetyError, "seven positive finite"):
            controller.prepare_joint_impedance_control(("left",), (400.0,) * 6)
        with self.assertRaises(ValueError):
            controller.prepare_joint_impedance_control(("middle",))

        self.assertEqual(robot.joint_impedance_calls, [])

    def test_prepare_reconfigures_the_arm_after_leaving_mode_three(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.prepare_joint_impedance_control(("left",))
        self.assertEqual([group for _, group in robot.joint_impedance_calls], [0])

        controller.set_control_mode(1)
        controller.prepare_joint_impedance_control(("left",))

        self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 0])

    def test_prepare_configures_only_the_requested_stale_group(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)

        controller.prepare_joint_impedance_control(("left",))
        controller.prepare_joint_impedance_control(("left", "right"))

        self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 1])

    def test_hold_joint_positions_requires_mode_three_and_checks_arguments(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)

        with self.assertRaisesRegex(SafetyError, "control_mode=3"):
            controller.hold_joint_positions("left", [0.0] * 7)

        controller.prepare_joint_impedance_control(("left",))
        controller.hold_joint_positions("left", [0.1] * 7)
        self.assertEqual(robot.requests[-1], ([0.1] * 7, 0.01, 0, False))

        with self.assertRaisesRegex(SafetyError, "seven finite"):
            controller.hold_joint_positions("left", [0.0] * 6)

    def test_rebase_to_measured_pose_reanchors_only_on_fresh_feedback(self):
        def left_stream():
            return LeftEndEffectorPositionStream(
                object(), rate_hz=50.0, life_time_s=0.10, max_translation_velocity_mps=0.01
            )

        def right_stream():
            return RightEndEffectorPositionStream(
                object(), rate_hz=50.0, life_time_s=0.10, max_translation_velocity_mps=0.01
            )

        for factory in (left_stream, right_stream):
            with self.subTest(stream=factory().__class__.__name__):
                stream = factory()
                stale_target = CartesianPoseTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
                measured = CartesianPoseTarget((0.1, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

                stream._commanded = stale_target
                stream._actual = measured
                stream._actual_at = time.monotonic()
                stream.rebase_to_measured_pose()
                self.assertEqual(stream._commanded, measured)

                stream._commanded = stale_target
                stream._actual_at = time.monotonic() - 10.0
                stream.rebase_to_measured_pose()
                self.assertEqual(stream._commanded, stale_target)

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

    def test_left_stream_can_rotate_without_translating(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        target = CartesianPoseTarget(
            initial.position_m,
            (0.0, 0.0, math.sin(math.pi / 4.0), math.cos(math.pi / 4.0)),
        )
        stream = controller.start_left_end_effector_position_stream(
            initial,
            max_translation_velocity_mps=0.001,
            allow_orientation_change=True,
            max_angular_velocity_rad_s=1.0,
        )
        try:
            stream.update_target(target)
            self.assertTrue(stream.wait_until_target(2.0))
        finally:
            stream.stop()

        poses = [request.left_end_effector_pose for request in robot.requests]
        self.assertGreater(len(poses), 1)
        self.assertEqual(
            [(pose.position.x, pose.position.y, pose.position.z) for pose in poses[-3:]],
            [(0.1, 0.2, 0.3)] * min(3, len(poses)),
        )
        self.assertAlmostEqual(poses[-1].orientation.z, math.sin(math.pi / 4.0), places=6)
        self.assertAlmostEqual(poses[-1].orientation.w, math.cos(math.pi / 4.0), places=6)

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
            robot.control_mode = 1
            stream.update_target(CartesianPoseTarget((0.101, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
            with self.assertRaises(SafetyError):
                stream.wait_until_target(0.5)
        finally:
            stream.stop()

    def test_left_stream_stops_when_control_mode_changes(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_left_end_effector_position_stream(initial)
        try:
            deadline = time.monotonic() + 0.5
            while not robot.requests and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertGreaterEqual(len(robot.requests), 1)

            robot.control_mode = 1
            stream.update_target(CartesianPoseTarget(
                (0.101, 0.2, 0.3), initial.orientation_xyzw,
            ))
            with self.assertRaises(SafetyError):
                stream.wait_until_target(0.5)
            request_count = len(robot.requests)
            time.sleep(0.05)
            self.assertEqual(len(robot.requests), request_count)
        finally:
            stream.stop()

    def test_left_actual_arrival_wait_rejects_control_mode_change(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_left_end_effector_position_stream(initial)
        try:
            self.assertTrue(stream.wait_until_target(0.5))
            original_status = robot.get_motion_control_status

            def status_with_healthy_writer():
                status = original_status()
                if threading.current_thread().name == "g2-left-end-effector-position":
                    status.control_mode = 3
                return status

            robot.get_motion_control_status = status_with_healthy_writer
            robot.control_mode = 1
            with self.assertRaises(SafetyError):
                stream.wait_until_actual_target(
                    initial,
                    0.2,
                    position_tolerance_m=0.00001,
                    orientation_tolerance_deg=0.1,
                    stable_samples=1,
                )
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

    def test_paused_position_stream_stop_returns_only_after_writer_exits(self):
        robot = FakeRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        initial = CartesianPoseTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_right_end_effector_position_stream(initial)
        gate.pause()
        robot.requests.clear()
        worker = stream._thread
        time.sleep(0.12)
        stream.stop()
        self.assertFalse(worker.is_alive())
        before = len(robot.requests)
        gate.resume()
        time.sleep(0.05)
        self.assertEqual(len(robot.requests), before)

    def test_right_stream_rebases_on_feedback_after_manual_takeover(self):
        robot = FakeRobot()
        robot.follow_right_pose = False
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        initial = CartesianPoseTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_right_end_effector_position_stream(
            initial, max_translation_velocity_mps=0.01,
        )
        try:
            stream.update_target(CartesianPoseTarget((0.2, 0.0, 0.0), initial.orientation_xyzw))
            time.sleep(0.12)
            gate.pause()
            robot.right_end_pose.position.x = 0.1
            time.sleep(0.12)
            request_count = len(robot.requests)
            gate.resume()
            deadline = time.monotonic() + 1.0
            while len(robot.requests) == request_count and time.monotonic() < deadline:
                time.sleep(0.01)
            resumed_x = robot.requests[request_count].right_end_effector_pose.position.x
            self.assertGreaterEqual(resumed_x, 0.1)
            self.assertLessEqual(resumed_x, 0.10021)
        finally:
            gate.resume()
            stream.stop()

    def test_resume_waits_for_fresh_feedback_before_any_automatic_write(self):
        robot = FakeRobot()
        robot.follow_right_pose = False
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        feedback_allowed = threading.Event()
        feedback_allowed.set()
        original_read = controller.read_end_effector_pose

        def controlled_read(frame_name):
            feedback_allowed.wait()
            return original_read(frame_name)

        controller.read_end_effector_pose = controlled_read
        initial = CartesianPoseTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_right_end_effector_position_stream(
            initial, max_translation_velocity_mps=0.01,
        )
        try:
            stream.update_target(CartesianPoseTarget((0.2, 0.0, 0.0), initial.orientation_xyzw))
            time.sleep(0.12)
            feedback_allowed.clear()
            gate.pause()
            robot.right_end_pose.position.x = 0.1
            request_count = len(robot.requests)
            gate.resume()
            time.sleep(0.08)
            self.assertEqual(len(robot.requests), request_count)
            feedback_allowed.set()
            deadline = time.monotonic() + 1.0
            while len(robot.requests) == request_count and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertAlmostEqual(robot.requests[request_count].right_end_effector_pose.position.x, 0.1)
        finally:
            feedback_allowed.set()
            gate.resume()
            stream.stop()

    def test_new_pause_generation_cannot_be_consumed_by_previous_rebase(self):
        robot = FakeRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        stream = RightEndEffectorPositionStream(
            controller,
            rate_hz=50.0,
            life_time_s=0.1,
            max_translation_velocity_mps=0.01,
        )
        target = CartesianPoseTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        stream._actual = target
        stream._actual_at = time.monotonic()
        stream._feedback_sequence = 1

        gate.pause()
        gate.resume()
        with self.assertRaises(MotionRebasePending):
            stream._refresh_target_before_send(target)

        # Feedback for generation 1 arrives, but a second takeover begins
        # before that generation can be committed by an automatic send.
        stream._feedback_sequence = 2
        stream._actual_at = time.monotonic()
        gate.pause()
        gate.resume()
        with self.assertRaises(MotionRebasePending):
            stream._refresh_target_before_send(target)
        self.assertEqual(stream._pending_rebase_generation, 2)

        stream._feedback_sequence = 3
        stream._actual_at = time.monotonic()
        self.assertIs(stream._refresh_target_before_send(target), target)
        self.assertEqual(stream._gate_generation, 2)

    def test_stop_waits_for_an_inflight_native_write_to_finish(self):
        robot = FakeRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        entered = threading.Event()
        release = threading.Event()
        original_send = robot.end_effector_pose_control

        def blocking_send(request):
            entered.set()
            release.wait()
            return original_send(request)

        robot.end_effector_pose_control = blocking_send
        initial = CartesianPoseTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_right_end_effector_position_stream(initial)
        self.assertTrue(entered.wait(1.0))
        stopped = threading.Event()
        stopper = threading.Thread(target=lambda: (stream.stop(), stopped.set()))
        stopper.start()
        self.assertFalse(stopped.wait(1.1))
        release.set()
        self.assertTrue(stopped.wait(1.0))
        stopper.join(1.0)

    def test_planning_request_uses_g2_names(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.move_preset("right", [0.0] * 7, velocity_rad_s=0.05)
        request = robot.requests[0]
        self.assertEqual(request.joint_names[0], "idx61_arm_r_joint1")
        self.assertEqual(request.joint_names[-1], "idx67_arm_r_joint7")
        self.assertEqual(request.joint_velocities, [0.05] * 7)

    def test_bilateral_planning_request_uses_all_arm_names(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        target = [0.1] * 14
        controller.move_bilateral_preset(target, velocity_rad_s=0.05)
        request = robot.requests[0]
        self.assertEqual(request.joint_names, list(ARM_JOINT_NAMES))
        self.assertEqual(request.joint_positions, target)
        self.assertEqual(request.joint_velocities, [0.05] * 14)
        self.assertEqual(robot.control_mode, 3)
        self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 1])

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
        self.assertEqual(robot.control_mode, 3)
        self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 1])

    def test_servo_to_stops_when_control_mode_changes(self):
        robot = FakeRobot()
        controller = G2Controller(
            FakeGdk,
            robot,
            allow_motion=True,
            control_period_s=0.02,
            max_servo_velocity_rad_s=0.15,
        )
        original_servo = robot.move_arm_joint_servo

        def drop_mode_after_first_command(positions, period, mode, flag):
            original_servo(positions, period, mode, flag)
            if len(robot.requests) == 1:
                robot.control_mode = 1

        robot.move_arm_joint_servo = drop_mode_after_first_command

        with self.assertRaises(SafetyError):
            controller.servo_to([0.02] * 14, duration_s=0.24)
        self.assertGreaterEqual(len(robot.requests), 1)
        self.assertLess(len(robot.requests), 12)

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
        self.assertEqual(robot.control_mode, 3)
        self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 1])

    def test_servo_waypoints_stop_when_control_mode_changes(self):
        robot = FakeRobot()
        controller = G2Controller(
            FakeGdk,
            robot,
            allow_motion=True,
            control_period_s=0.02,
            max_servo_velocity_rad_s=0.15,
        )
        original_servo = robot.move_arm_joint_servo

        def drop_mode_after_first_command(positions, period, mode, flag):
            original_servo(positions, period, mode, flag)
            if len(robot.requests) == 1:
                robot.control_mode = 1

        robot.move_arm_joint_servo = drop_mode_after_first_command

        with self.assertRaises(SafetyError):
            controller.servo_waypoints(([0.02] * 14,), duration_s=0.24)
        self.assertGreaterEqual(len(robot.requests), 1)
        self.assertLess(len(robot.requests), 12)

    def test_cartesian_impedance_uses_mode_2_and_50hz_stream(self):
        robot = FakeRobot()
        robot.control_mode = 3
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
                initial_y = 0.2 if side == "left" else -0.2
                stream = factory(CartesianPoseTarget((0.1, initial_y, 0.3), (0, 0, 0, 1)))
                try:
                    gate.request_advance()
                    self.assertTrue(stream._stop.wait(1.0))
                    with self.assertRaises(AdvanceRightRequested):
                        stream.raise_if_failed()
                finally:
                    stream.stop()

    def test_end_effector_position_stream_uses_mode_3(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_left_end_effector_position_stream(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            rate_hz=50.0,
            life_time_s=0.10,
        )
        time.sleep(0.05)
        stream.stop()

        self.assertEqual(robot.control_mode, 3)
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

    def test_single_end_effector_position_setpoint_requires_mode_3(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        controller.send_left_end_effector_position_setpoint(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            life_time_s=0.10,
        )

        self.assertEqual(robot.control_mode, 3)
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

    def test_left_stream_can_confirm_a_hold_target_was_commanded_before_stop(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        hold = CartesianPoseTarget((0.101, 0.2, 0.3), initial.orientation_xyzw)
        stream = controller.start_left_end_effector_position_stream(
            initial, max_translation_velocity_mps=0.01,
        )
        try:
            stream.update_target(hold)
            self.assertTrue(stream.wait_until_commanded_target(hold, 0.5))
        finally:
            stream.stop()
        self.assertAlmostEqual(robot.requests[-1].left_end_effector_pose.position.x, 0.101)

    def test_left_stream_pauses_command_progress_when_actual_pose_lags(self):
        robot = FakeRobot()
        robot.left_end_pose.position.x, robot.left_end_pose.position.y, robot.left_end_pose.position.z = (0.1, 0.2, 0.3)
        robot.follow_left_pose = False
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_left_end_effector_position_stream(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            max_translation_velocity_mps=0.03,
        )
        try:
            stream.update_target(CartesianPoseTarget((0.15, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
            time.sleep(0.45)
            x_positions = [request.left_end_effector_pose.position.x for request in robot.requests]
            self.assertLessEqual(max(x_positions) - 0.1, 0.006)
            self.assertFalse(stream.is_target_reached())
        finally:
            stream.stop()

    def test_left_stream_xy_remains_responsive_during_long_descent(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_left_end_effector_position_stream(
            initial, max_translation_velocity_mps=0.03,
        )
        try:
            stream.set_independent_xy_z_slew(True)
            stream.update_target(CartesianPoseTarget((0.101, 0.2, 0.05), initial.orientation_xyzw))
            time.sleep(0.2)
            self.assertGreater(robot.requests[-1].left_end_effector_pose.position.x, 0.1003)
            self.assertLess(robot.requests[-1].left_end_effector_pose.position.z, 0.3)
        finally:
            stream.stop()

    def test_left_stream_holds_z_until_xy_priority_is_released(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_left_end_effector_position_stream(
            initial, max_translation_velocity_mps=0.03,
        )
        try:
            stream.set_independent_xy_z_slew(True)
            stream.set_xy_priority(True)
            stream.update_target(CartesianPoseTarget((0.101, 0.2, 0.05), initial.orientation_xyzw))
            time.sleep(0.2)
            positions = [request.left_end_effector_pose.position for request in robot.requests]
            self.assertGreater(positions[-1].x, 0.1009)
            self.assertTrue(all(abs(position.z - 0.3) < 1e-9 for position in positions))
            stream.set_xy_priority(False)
            time.sleep(0.15)
            self.assertLess(robot.requests[-1].left_end_effector_pose.position.z, 0.299)
        finally:
            stream.stop()

    def test_left_stream_does_not_report_arrival_from_command_only(self):
        robot = FakeRobot()
        robot.left_end_pose.position.x, robot.left_end_pose.position.y, robot.left_end_pose.position.z = (0.1, 0.2, 0.3)
        robot.follow_left_pose = False
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_left_end_effector_position_stream(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
            max_translation_velocity_mps=0.03,
        )
        try:
            stream.update_target(CartesianPoseTarget((0.115, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)))
            self.assertFalse(stream.wait_until_target(0.25))
        finally:
            stream.stop()

    def test_left_stream_does_not_fabricate_arrival_before_feedback(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        feedback_gate = threading.Event()
        def blocked_read(_frame):
            feedback_gate.wait()
            return CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        controller.read_end_effector_pose = blocked_read
        stream = controller.start_left_end_effector_position_stream(
            CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        )
        try:
            self.assertFalse(stream.is_target_reached())
            self.assertFalse(stream.wait_until_target(0.05))
        finally:
            feedback_gate.set()
            stream.stop()

    def test_left_stream_revokes_arrival_after_actual_drift(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_left_end_effector_position_stream(initial, max_translation_velocity_mps=0.01)
        try:
            stream.update_target(CartesianPoseTarget((0.101, 0.2, 0.3), initial.orientation_xyzw))
            self.assertTrue(stream.wait_until_target(0.5))
            robot.follow_left_pose = False
            robot.left_end_pose.position.x += 0.1
            time.sleep(0.15)
            self.assertFalse(stream.is_target_reached())
        finally:
            stream.stop()

    def test_left_stream_rejects_restart_while_feedback_reader_is_blocked(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        feedback_gate = threading.Event()
        def blocked_read(_frame):
            feedback_gate.wait()
            return CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        controller.read_end_effector_pose = blocked_read
        initial = CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_left_end_effector_position_stream(initial)
        try:
            stream.stop()
            with self.assertRaises(RuntimeError):
                stream.start(initial)
        finally:
            feedback_gate.set()
            stream.stop()

    def test_position_stream_does_not_burst_after_delayed_feedback(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                robot = FakeRobot()
                controller = G2Controller(FakeGdk, robot, allow_motion=True)
                feedback_gate = threading.Event()
                first_feedback = True
                feedback_times = []
                initial = CartesianPoseTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))

                def delayed_read(_frame):
                    nonlocal first_feedback
                    feedback_times.append(time.monotonic())
                    if first_feedback:
                        first_feedback = False
                        feedback_gate.wait()
                    time.sleep(0.005)
                    return initial

                controller.read_end_effector_pose = delayed_read
                send_times = []
                original_send = robot.end_effector_pose_control

                def capture_send(request):
                    send_times.append(time.monotonic())
                    time.sleep(0.005)
                    return original_send(request)

                robot.end_effector_pose_control = capture_send
                factory = getattr(controller, f"start_{side}_end_effector_position_stream")
                stream = factory(initial, max_translation_velocity_mps=0.01)
                try:
                    stream.update_target(CartesianPoseTarget((0.02, 0.0, 0.0), initial.orientation_xyzw))
                    time.sleep(0.22)
                    feedback_gate.set()
                    time.sleep(0.35)
                finally:
                    feedback_gate.set()
                    stream.stop()
                intervals = [second - first for first, second in zip(send_times, send_times[1:])]
                self.assertGreaterEqual(len(intervals), 3)
                self.assertGreaterEqual(min(intervals), 0.014)
                median_interval = sorted(intervals)[len(intervals) // 2]
                self.assertGreaterEqual(median_interval, 0.018)
                self.assertLessEqual(median_interval, 0.024)
                feedback_intervals = [
                    second - first for first, second in zip(feedback_times, feedback_times[1:])
                ]
                regular_feedback_intervals = [interval for interval in feedback_intervals if interval < 0.15]
                self.assertGreaterEqual(len(regular_feedback_intervals), 2)
                self.assertGreaterEqual(min(regular_feedback_intervals), 0.08)
                self.assertLessEqual(sum(regular_feedback_intervals) / len(regular_feedback_intervals), 0.14)

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
            self.assertFalse(stream.wait_until_target(0.5))
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
