import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import contextmanager

from left_teach_placement import _GatedRobot, read_left_trajectory, replay_left_placement


class PlacementTests(unittest.TestCase):
    def test_arrival_wait_excludes_long_spacemouse_pause(self):
        from teach_trajectory_gdk import G2TrajectorySession
        from test_teach_trajectory_gdk import FakeGdk, FakeRobot
        now = [0.0]
        class Robot(FakeRobot):
            reads = 0
            def get_joint_states(self):
                now[0] = 45.0
                self.reads += 1
                if self.reads > 1:
                    self.positions[:7] = [.1]*7
                return super().get_joint_states()
        class Gate:
            @contextmanager
            def automatic_motion(self):
                now[0] = 45.0
                yield
        robot = _GatedRobot(Robot(), Gate())
        session = G2TrajectorySession(FakeGdk(), robot, "left", monotonic_fn=robot.clock, sleep_fn=lambda _: None)
        with patch("left_teach_placement.time.monotonic", side_effect=lambda: now[0]):
            session._wait_for_start([.1]*7)
        self.assertEqual(robot.paused_s, 45.0)

    def test_real_replay_keeps_other_arm_and_reaches_final_sample(self):
        from teach_trajectory import build_trajectory, trajectory_document
        from test_teach_trajectory_gdk import FakeGdk, FakeRobot
        class Robot(FakeRobot):
            def move_arm_joint_servo(self, command, period_s, mode, relative):
                self.servo_commands.append((tuple(command), period_s, mode, relative))
                self.positions[:7] = command
        robot = Robot([0.0]*7 + [.2]*7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "left.json"
            path.write_text(json.dumps(trajectory_document(build_trajectory("left", [(0, [0]*7), (10_000_000, [.1]*7)]))))
            args = SimpleNamespace(left_teach_trajectory=path, left_teach_speed_scale=.5, left_teach_start_velocity_rad_s=.05)
            services = SimpleNamespace(gdk=FakeGdk(), robot=robot, gate=None)
            with patch("sys.stdout"):
                result = replay_left_placement(services, args)
        self.assertEqual(result["sent"], 3)
        self.assertEqual(robot.positions, [.1]*7 + [.2]*7)
        self.assertTrue(all(len(command[0]) == 7 for command in robot.servo_commands))
        self.assertEqual(robot.control_modes, [3])
        self.assertEqual(robot.impedance_calls[0][1], 0)

    def test_right_trajectory_is_rejected(self):
        from teach_trajectory import build_trajectory, trajectory_document
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "right.json"
            path.write_text(json.dumps(trajectory_document(build_trajectory("right", [(0, [0]*7)]))))
            with self.assertRaisesRegex(ValueError, "左臂"):
                read_left_trajectory(path)

    def test_replay_borrows_robot_and_preserves_gripper(self):
        from teach_trajectory import build_trajectory, trajectory_document
        writes = []
        class Robot:
            def set_control_mode(self, mode):
                writes.append(("mode", mode))
            def set_joint_impedance(self, impedances, group):
                writes.append(("impedance", tuple(item.kp for item in impedances), group))
            def move_arm_joint_servo(self, *args):
                writes.append(("servo", args))
        class Gate:
            @contextmanager
            def automatic_motion(self):
                writes.append(("lock",))
                yield
        class Session:
            def __init__(self, gdk, robot, side, **kwargs):
                self.robot = robot
                self.assert_side = side
            def replay(self, trajectory, **kwargs):
                if self.assert_side != "left" or kwargs["speed_scale"] != .5:
                    raise AssertionError("wrong replay settings")
                self.robot.set_control_mode(3)
                self.robot.set_joint_impedance(
                    [SimpleNamespace(kp=value) for value in (400, 400, 400, 500, 300, 200, 200)],
                    0,
                )
                self.robot.move_arm_joint_servo([0]*7, .01, 0, False)
                return 1
            def _wait_for_start(self, target):
                pass
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "left.json"
            path.write_text(json.dumps(trajectory_document(build_trajectory("left", [(0, [0]*7)]))))
            args = SimpleNamespace(left_teach_trajectory=path, left_teach_speed_scale=.5, left_teach_start_velocity_rad_s=.05)
            services = SimpleNamespace(gdk=object(), robot=Robot(), gate=Gate())
            with patch("left_teach_placement.G2TrajectorySession", Session):
                record = replay_left_placement(services, args)
        self.assertEqual(record["sent"], 1)
        self.assertEqual(
            [item[0] for item in writes],
            ["lock", "mode", "lock", "impedance", "lock", "servo"],
        )
        self.assertEqual(writes[1], ("mode", 3))
        self.assertEqual(writes[3][1:], ((400, 400, 400, 500, 300, 200, 200), 0))


if __name__ == "__main__":
    unittest.main()
