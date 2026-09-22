import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from right_teach_placement import validate_right_teach_args, replay_right_placement
from teach_trajectory import build_trajectory, trajectory_document
from cartesian_teach import JointTrajectory, JointTrajectoryPoint, ValidationReport, joint_document
from safety import RIGHT_ARM_JOINT_NAMES


class RightPlacementTests(unittest.TestCase):
    def test_all_four_solved_cartesian_documents_are_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(4):
                path = Path(directory) / f"right_{index}.json"
                trajectory = JointTrajectory(
                    side="right",
                    joint_names=RIGHT_ARM_JOINT_NAMES,
                    fixed_joint_positions={f"idx0{i}_body_joint{i}": 0.0 for i in range(1, 6)},
                    source_sha256="0" * 64,
                    urdf_sha256="1" * 64,
                    points=(JointTrajectoryPoint(0, (0.0,) * 7, 0.0, 0.0),),
                )
                path.write_text(json.dumps(joint_document(
                    trajectory, ValidationReport(True, (), 0.0, 0.0, 0.0, 0.0),
                )), encoding="utf-8")
                paths.append(path)
            args = SimpleNamespace(
                right_teach_trajectory=None,
                right_teach_trajectories=paths,
                right_teach_speed_scale=.5,
                right_teach_start_velocity_rad_s=.05,
            )
            specs = validate_right_teach_args(args)
        self.assertEqual(len(specs), 4)
        self.assertTrue(all(spec.kind == "cartesian_solved" for spec in specs))

    def test_right_solved_document_uses_position_replay_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "right_cartesian_solved.json"
            trajectory = JointTrajectory(
                side="right",
                joint_names=RIGHT_ARM_JOINT_NAMES,
                fixed_joint_positions={f"idx0{i}_body_joint{i}": 0.0 for i in range(1, 6)},
                source_sha256="0" * 64,
                urdf_sha256="1" * 64,
                points=(JointTrajectoryPoint(0, (0.0,) * 7, 0.0, 0.0),),
            )
            path.write_text(json.dumps(joint_document(
                trajectory, ValidationReport(True, (), 0.0, 0.0, 0.0, 0.0),
            )), encoding="utf-8")
            args = SimpleNamespace(
                right_teach_trajectory=path,
                right_teach_trajectories=None,
                right_teach_speed_scale=.5,
                right_teach_start_velocity_rad_s=.05,
                teach_max_velocity_rad_s=1.0,
                teach_max_acceleration_rad_s2=15.0,
            )
            services = SimpleNamespace(gdk=object(), robot=object(), gate=None)
            with patch("right_teach_placement.replay_solved_trajectory", return_value=9) as replay:
                result = replay_right_placement(services, args)
        replay.assert_called_once()
        self.assertEqual(result["trajectory_type"], "cartesian_solved")
        self.assertEqual(result["sent"], 9)
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
