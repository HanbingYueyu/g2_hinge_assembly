import json
import io
from contextlib import redirect_stdout
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from safety import ARM_JOINT_NAMES, LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES, SafetyError
from teach_trajectory import build_trajectory
from teach_trajectory import trajectory_document
from teach_trajectory_gdk import G2TrajectorySession, main


class FakeImpedance:
    def __init__(self, *, kp):
        self.kp = kp


class FakeJointControlReq:
    pass


class FakeGdk:
    Impedance = FakeImpedance
    JointControlReq = FakeJointControlReq


class FakeMotionStatus:
    mode = 0
    error_code = 0
    error_msg = ""

    def __init__(self, control_mode):
        self.control_mode = control_mode


class FakeRobot:
    def __init__(self, positions=None):
        self.positions = list(positions if positions is not None else [0.0] * 14)
        self.control_modes = []
        self.impedance_calls = []
        self.planned_requests = []
        self.servo_commands = []
        self.control_mode = 1
        self.error_code = 0

    def get_joint_states(self):
        return {
            "states": [
                {"name": name, "motor_position": position}
                for name, position in reversed(list(zip(ARM_JOINT_NAMES, self.positions)))
            ]
        }

    def get_motion_control_status(self):
        return FakeMotionStatus(self.control_mode)

    def set_control_mode(self, mode):
        self.control_modes.append(mode)
        self.control_mode = mode
        return 0

    def set_joint_impedance(self, impedances, group):
        self.impedance_calls.append((impedances, group))
        return 0

    def joint_control_request(self, request):
        self.planned_requests.append(request)
        offset = 0 if tuple(request.joint_names) == LEFT_ARM_JOINT_NAMES else 7
        self.positions[offset : offset + 7] = request.joint_positions

    def move_arm_joint_servo(self, command, period_s, mode, relative):
        self.servo_commands.append((tuple(command), period_s, mode, relative))
        self.positions = list(command)


class G2TrajectorySessionTests(unittest.TestCase):
    def test_teach_save_trims_initial_idle_and_reports_removed_samples(self):
        trajectory = build_trajectory("left", [(0, [0]*7), (10, [0]*7), (20, [.001]*7), (30, [.02]*7), (40, [.02]*7)])
        gdk = FakeGdk()
        gdk.gdk_release = lambda: None
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "recorded.json"
            with patch("teach_trajectory_gdk._open_gdk", return_value=(gdk, FakeRobot())), patch.object(G2TrajectorySession, "record", return_value=trajectory), redirect_stdout(io.StringIO()):
                code = main(["teach", "--side", "left", "--output", str(output), "--allow-motion"])
            saved = json.loads(output.read_text())
        self.assertEqual(code, 0)
        self.assertEqual(len(saved["samples"]), 3)
        self.assertEqual([s["t_ns"] for s in saved["samples"]], [0, 10, 20])
        self.assertEqual(saved["leading_stationary_trim"]["removed_samples"], 2)

    def test_record_restores_standard_impedance_without_returning_to_position_mode(self):
        robot = FakeRobot()
        session = G2TrajectorySession(FakeGdk(), robot, "left")

        trajectory = session.record(duration_s=0.0, rate_hz=100.0)

        self.assertEqual(trajectory.side, "left")
        self.assertEqual(trajectory.samples[0].joint_positions_rad, (0.0,) * 7)
        self.assertEqual(robot.control_modes, [3])
        self.assertEqual(
            [
                tuple(item.kp for item in impedances)
                for impedances, group in robot.impedance_calls
                if group == 0
            ],
            [
                (50, 50, 50, 60, 30, 30, 30),
                (400, 400, 400, 500, 300, 200, 200),
            ],
        )

    def test_record_interruption_also_restores_standard_impedance(self):
        robot = FakeRobot()
        session = G2TrajectorySession(
            FakeGdk(),
            robot,
            "left",
            sleep_fn=lambda _: None,
        )

        def interrupt_recording():
            raise KeyboardInterrupt

        trajectory = session.record(
            rate_hz=100.0,
            stop_requested=interrupt_recording,
        )

        self.assertEqual(len(trajectory.samples), 1)
        self.assertEqual(robot.control_modes, [3])
        self.assertEqual(
            [tuple(item.kp for item in impedances) for impedances, _ in robot.impedance_calls],
            [
                (50, 50, 50, 60, 30, 30, 30),
                (400, 400, 400, 500, 300, 200, 200),
            ],
        )
        self.assertEqual([group for _, group in robot.impedance_calls], [0, 0])

    def test_record_sets_current_selected_angles_as_the_impedance_reference(self):
        robot = FakeRobot([0.15] * 7 + [0.25] * 7)
        session = G2TrajectorySession(FakeGdk(), robot, "right")

        session.record(duration_s=0.0, rate_hz=100.0)

        reference = robot.planned_requests[0]
        self.assertEqual(tuple(reference.joint_names), RIGHT_ARM_JOINT_NAMES)
        self.assertEqual(reference.joint_positions, [0.25] * 7)

    def test_record_selects_only_right_arm_feedback(self):
        robot = FakeRobot([0.1] * 7 + [0.2] * 7)
        session = G2TrajectorySession(FakeGdk(), robot, "right")

        trajectory = session.record(duration_s=0.0, rate_hz=100.0)

        self.assertEqual(trajectory.joint_names, RIGHT_ARM_JOINT_NAMES)
        self.assertEqual(trajectory.samples[0].joint_positions_rad, (0.2,) * 7)
        self.assertEqual([group for _, group in robot.impedance_calls], [1, 1])
        self.assertEqual(
            tuple(item.kp for item in robot.impedance_calls[-1][0]),
            (400, 400, 400, 500, 300, 200, 200),
        )

    def test_record_stops_after_external_stop_request_and_keeps_samples(self):
        robot = FakeRobot()
        stop_answers = iter((True,))
        session = G2TrajectorySession(
            FakeGdk(),
            robot,
            "left",
            sleep_fn=lambda _: None,
        )

        trajectory = session.record(rate_hz=100.0, stop_requested=lambda: next(stop_answers))

        self.assertEqual(len(trajectory.samples), 1)
        self.assertEqual(robot.control_modes, [3])

    def test_replay_sends_seven_selected_right_arm_joint_values(self):
        robot = FakeRobot([0.3] * 7 + [0.4] * 7)
        events = []
        session = G2TrajectorySession(
            FakeGdk(),
            robot,
            "right",
            sleep_fn=lambda _: None,
            event_sink=events.append,
        )
        trajectory = build_trajectory(
            "right",
            [
                *[(index * 1_000_000, [index * 0.01] * 7) for index in range(10)],
                (20_000_000, [0.1] * 7),
            ],
        )

        sent = session.replay(trajectory, speed_scale=0.2)

        request = robot.planned_requests[0]
        self.assertEqual(tuple(request.joint_names), RIGHT_ARM_JOINT_NAMES)
        self.assertEqual(request.joint_positions, [0.0] * 7)
        self.assertGreater(sent, 1)
        self.assertTrue(all(len(command[0]) == 7 for command in robot.servo_commands))
        self.assertEqual(robot.servo_commands[-1][0], (0.1,) * 7)
        self.assertTrue(all(command[1] == 0.01 for command in robot.servo_commands))
        self.assertTrue(all(command[2] == 1 for command in robot.servo_commands))
        self.assertEqual(
            [tuple(item.kp for item in impedances) for impedances, group in robot.impedance_calls if group == 1],
            [(400, 400, 400, 500, 300, 200, 200)],
        )
        self.assertTrue(any("起点已到达" in event for event in events))
        self.assertEqual(
            [event for event in events if event.startswith("回放点进度：")],
            [f"回放点进度：{percent}%" for percent in (0, 20, 40, 60, 80, 90, 100)],
        )
        self.assertEqual(robot.control_modes, [3])

    def test_replay_stops_when_joint_impedance_mode_is_lost(self):
        class SteppingClock:
            def __init__(self, step):
                self.value = 0.0
                self.step = step

            def __call__(self):
                self.value += self.step
                return self.value

        class Robot(FakeRobot):
            def move_arm_joint_servo(self, command, period_s, mode, relative):
                super().move_arm_joint_servo(command, period_s, mode, relative)
                # Simulate losing the persistent control mode after one target.
                self.control_mode = 1

        session = G2TrajectorySession(
            FakeGdk(),
            Robot(),
            "left",
            monotonic_fn=SteppingClock(0.2),
            sleep_fn=lambda _: None,
        )
        trajectory = build_trajectory("left", [(0, [0.0] * 7), (10_000_000, [0.1] * 7)])

        with self.assertRaisesRegex(SafetyError, "control_mode=3"):
            session.replay(trajectory, speed_scale=0.5)

    def test_replay_prepares_impedance_through_the_provided_controller(self):
        calls = []

        class Controller:
            def prepare_joint_impedance_control(self, sides=("left", "right"), kp=None):
                calls.append(("prepare", tuple(sides)))

            def _require_joint_impedance_mode(self, operation):
                calls.append(("require", operation))

        robot = FakeRobot()
        session = G2TrajectorySession(
            FakeGdk(), robot, "left", controller=Controller(), sleep_fn=lambda _: None
        )
        session.replay(build_trajectory("left", [(0, [0.0] * 7)]), speed_scale=0.2)

        self.assertEqual(calls[0], ("prepare", ("left",)))
        self.assertTrue(any(kind == "require" for kind, _ in calls))
        self.assertEqual(robot.impedance_calls, [])

    def test_record_routes_low_and_standard_gains_through_the_provided_controller(self):
        writes = []

        class Controller:
            def set_joint_impedance(self, side, kp):
                writes.append((side, tuple(float(value) for value in kp)))

        robot = FakeRobot()
        session = G2TrajectorySession(
            FakeGdk(), robot, "right", controller=Controller(), sleep_fn=lambda _: None
        )
        session.record(duration_s=0.0, rate_hz=100.0)

        self.assertEqual(
            writes,
            [
                ("right", (50, 50, 50, 60, 30, 30, 30)),
                ("right", (400, 400, 400, 500, 300, 200, 200)),
            ],
        )
        self.assertEqual(robot.impedance_calls, [])

    def test_replay_rejects_a_trajectory_for_the_other_arm(self):
        session = G2TrajectorySession(FakeGdk(), FakeRobot(), "left")
        trajectory = build_trajectory("right", [(0, [0.0] * 7)])

        with self.assertRaisesRegex(ValueError, "does not match"):
            session.replay(trajectory, speed_scale=0.2)

    def test_replay_allows_a_single_stationary_sample(self):
        robot = FakeRobot()
        session = G2TrajectorySession(FakeGdk(), robot, "left", sleep_fn=lambda _: None)
        trajectory = build_trajectory("left", [(0, [0.0] * 7)])

        sent = session.replay(trajectory, speed_scale=0.2)

        self.assertEqual(sent, 1)
        self.assertEqual(robot.servo_commands[0][0], (0.0,) * 7)
        self.assertEqual(robot.servo_commands[0][2], 0)
        self.assertEqual(robot.control_modes, [3])
        self.assertEqual(robot.impedance_calls[0][1], 0)


class G2TrajectoryCliTests(unittest.TestCase):
    def test_replay_requires_explicit_motion_authorization(self):
        result = main(["replay", "--input", "trajectory.json"])

        self.assertEqual(result, 2)

    def test_teach_requires_explicit_motion_authorization(self):
        result = main(["teach", "--side", "left", "--output", "trajectory.json"])

        self.assertEqual(result, 2)

    def test_inspect_loads_trajectory_without_opening_gdk(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "left.json"
            path.write_text(
                json.dumps(trajectory_document(build_trajectory("left", [(0, [0.0] * 7), (1_000_000_000, [0.2] * 7)]))),
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                result = main(["inspect", "--input", str(path)])

        self.assertEqual(result, 0)
        self.assertIn('"samples": 2', output.getvalue())


if __name__ == "__main__":
    unittest.main()
