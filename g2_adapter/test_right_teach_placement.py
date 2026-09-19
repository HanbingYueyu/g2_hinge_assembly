import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from right_teach_placement import validate_right_teach_args, replay_right_placement
from teach_trajectory import build_trajectory, trajectory_document


class RightPlacementTests(unittest.TestCase):
    def test_four_segments_wait_before_each_next_segment_even_after_takeover(self):
        from right_teach_placement import run_right_teach_sequence
        files = [Path(f"right_{i}.json") for i in range(1, 5)]
        args = SimpleNamespace(right_teach_trajectories=files, right_teach_trajectory=None)
        events = []
        class Runtime:
            def wait_for_advance(self, spec):
                events.append(("wait", spec.stage))
            def emit(self, event, **fields):
                pass
        def replay(services, segment_args):
            events.append(("replay", segment_args.right_teach_trajectory))
            return {"sent": 3, "interrupted_by_takeover": True}
        with patch("right_teach_placement.replay_right_placement", side_effect=replay), patch("sys.stdout"):
            result = run_right_teach_sequence(object(), args, Runtime())
        self.assertEqual(events, [("replay", files[0]), ("wait", "right_teach_gate_2"), ("replay", files[1]), ("wait", "right_teach_gate_3"), ("replay", files[2]), ("wait", "right_teach_gate_4"), ("replay", files[3])])
        self.assertEqual(len(result["segments"]), 4)

    def test_all_four_files_are_validated_before_motion(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"right_{i}.json" for i in range(4)]
            for path in paths[:3]:
                path.write_text(json.dumps(trajectory_document(build_trajectory("right", [(0, [0]*7)]))))
            args = SimpleNamespace(right_teach_trajectory=None, right_teach_trajectories=paths, right_teach_speed_scale=.5, right_teach_start_velocity_rad_s=.05)
            with self.assertRaises(OSError):
                validate_right_teach_args(args)
            paths[3].write_text(json.dumps(trajectory_document(build_trajectory("left", [(0, [0]*7)]))))
            with self.assertRaisesRegex(ValueError, "右臂"):
                validate_right_teach_args(args)

    def test_wrong_arm_and_invalid_speed_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "demo.json"
            args = SimpleNamespace(right_teach_trajectory=path, right_teach_speed_scale=.5, right_teach_start_velocity_rad_s=.05)
            path.write_text(json.dumps(trajectory_document(build_trajectory("left", [(0, [0]*7)]))))
            with self.assertRaisesRegex(ValueError, "右臂"):
                validate_right_teach_args(args)
            path.write_text(json.dumps(trajectory_document(build_trajectory("right", [(0, [0]*7)]))))
            for value in (0, float("nan"), -1):
                args.right_teach_speed_scale = value
                with self.assertRaises(ValueError):
                    validate_right_teach_args(args)

    def test_real_right_replay_preserves_left_arm(self):
        from test_teach_trajectory_gdk import FakeGdk, FakeRobot
        class Robot(FakeRobot):
            def move_arm_joint_servo(self, command, period_s, mode, relative):
                if mode != 1 or len(command) != 7:
                    raise AssertionError("wrong selected arm")
                self.positions[7:] = command
        robot = Robot([.2]*7 + [0]*7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "right.json"
            path.write_text(json.dumps(trajectory_document(build_trajectory("right", [(0, [0]*7), (10_000_000, [.1]*7)]))))
            args = SimpleNamespace(right_teach_trajectory=path, right_teach_speed_scale=.5, right_teach_start_velocity_rad_s=.05)
            with patch("sys.stdout"):
                result = replay_right_placement(SimpleNamespace(robot=robot, gdk=FakeGdk(), gate=None), args)
        self.assertEqual(result["sent"], 3)
        self.assertEqual(robot.positions, [.2]*7 + [.1]*7)
        self.assertEqual(robot.control_modes, [3])
        self.assertEqual(robot.impedance_calls[0][1], 1)
        self.assertEqual(
            tuple(item.kp for item in robot.impedance_calls[0][0]),
            (400, 400, 400, 500, 300, 200, 200),
        )


if __name__ == "__main__":
    unittest.main()
