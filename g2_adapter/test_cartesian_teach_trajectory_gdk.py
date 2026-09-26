import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cartesian_teach import (
    CartesianSample,
    CartesianTrajectory,
    JointTrajectory,
    JointTrajectoryPoint,
    ValidationReport,
    cartesian_document,
    joint_document,
    load_cartesian_document,
    load_joint_document,
)
from cartesian_teach_trajectory_gdk import (
    BODY_JOINT_NAMES,
    ensure_replayable,
    enter_position_control,
    iter_scaled_joint_targets,
    record_trajectory,
    replay_trajectory,
    main,
    _parser,
    verify_live_model,
)


class ReplayGateTests(unittest.TestCase):
    def setUp(self):
        self.trajectory = JointTrajectory(
            side="right",
            joint_names=tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)),
            fixed_joint_positions={name: 0.0 for name in BODY_JOINT_NAMES},
            source_sha256="0" * 64,
            urdf_sha256="1" * 64,
            points=(
                JointTrajectoryPoint(0, (0.0,) * 7, 0.0, 0.0),
                JointTrajectoryPoint(1_000_000_000, (1.0,) * 7, 0.0, 0.0),
            ),
        )

    def test_replay_gate_rejects_a_file_whose_offline_validation_failed(self):
        report = ValidationReport(False, ("IK failed at point 3",), 0.0, 0.0, 0.0, 0.0)

        with self.assertRaisesRegex(ValueError, "not replay-ready"):
            ensure_replayable(self.trajectory, report)

    def test_speed_scaling_resamples_joint_targets_at_100_hz(self):
        points = tuple(iter_scaled_joint_targets(self.trajectory, speed_scale=0.5, rate_hz=100.0))

        self.assertEqual(points[0], (0.0,) * 7)
        self.assertEqual(points[-1], (1.0,) * 7)
        self.assertEqual(len(points), 201)
        self.assertAlmostEqual(points[100][0], 0.5)

    def test_json_documents_preserve_absolute_frames_and_validation_gate(self):
        raw = CartesianTrajectory(
            "left", "base_link", "arm_l_end_link", tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)),
            {name: 0.25 for name in BODY_JOINT_NAMES},
            (CartesianSample(0, (0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0), (0.0,) * 7),),
        )
        loaded_raw = load_cartesian_document(cartesian_document(raw, urdf_sha256="a" * 64))
        self.assertEqual(loaded_raw.base_frame, "base_link")
        self.assertEqual(loaded_raw.tip_frame, "arm_l_end_link")
        report = ValidationReport(True, (), 0.4, 0.8, 0.00001, 0.0001)

        loaded_joint, loaded_report = load_joint_document(joint_document(self.trajectory, report))

        self.assertEqual(loaded_joint.points[-1].joint_positions_rad, (1.0,) * 7)
        self.assertTrue(loaded_report.replay_ready)

    def test_solved_document_rejects_non_increasing_timestamps_even_if_marked_ready(self):
        report = ValidationReport(True, (), 0.1, 0.1, 0.0, 0.0)
        document = joint_document(self.trajectory, report)
        document["points"][1]["t_ns"] = 0

        with self.assertRaisesRegex(ValueError, "timestamps"):
            load_joint_document(document)

    def test_documents_reject_right_joint_names_claimed_as_left_arm(self):
        raw = CartesianTrajectory(
            "left", "base_link", "arm_l_end_link", tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8)),
            {name: 0.0 for name in BODY_JOINT_NAMES},
            (CartesianSample(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0,) * 7),),
        )
        with self.assertRaisesRegex(ValueError, "joint names"):
            load_cartesian_document(cartesian_document(raw, urdf_sha256="a" * 64))

        document = joint_document(self.trajectory, ValidationReport(True, (), 0.0, 0.0, 0.0, 0.0))
        document["side"] = "left"
        with self.assertRaisesRegex(ValueError, "joint names"):
            load_joint_document(document)

    def test_documents_reject_non_finite_fixed_body_joint(self):
        document = joint_document(self.trajectory, ValidationReport(True, (), 0.0, 0.0, 0.0, 0.0))
        document["fixed_joint_positions"][BODY_JOINT_NAMES[0]] = float("nan")
        with self.assertRaisesRegex(ValueError, "fixed body"):
            load_joint_document(document)


class RecordingTests(unittest.TestCase):
    def test_recording_uses_mode_one_and_reads_absolute_end_link_without_motion_commands(self):
        names = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
        states = [
            {"name": name, "motor_position": 0.1 * index, "error_code": 0}
            for index, name in enumerate((*BODY_JOINT_NAMES, *names), 1)
        ]
        pose = SimpleNamespace(
            position=SimpleNamespace(x=0.4, y=0.2, z=0.8),
            orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )

        class Robot:
            def __init__(self):
                self.control_modes = []

            def set_control_mode(self, value):
                self.control_modes.append(value)
                return 0

            def get_motion_control_status(self):
                return SimpleNamespace(control_mode=1, mode=0, error_code=0, error_msg="",
                                       frame_names=["arm_l_end_link"], frame_poses=[pose])

            def get_joint_states(self):
                return {"states": states}

        robot = Robot()
        trajectory = record_trajectory(
            robot, "left", rate_hz=100.0, duration_s=0.0,
            monotonic_ns_fn=lambda: 123, sleep_fn=lambda _: None,
        )

        self.assertEqual(robot.control_modes, [])
        self.assertEqual(trajectory.base_frame, "base_link")
        self.assertEqual(trajectory.tip_frame, "arm_l_end_link")
        self.assertEqual(trajectory.samples[0].position_m, (0.4, 0.2, 0.8))
        self.assertEqual(trajectory.samples[0].reference_joint_positions_rad, tuple(0.1 * i for i in range(6, 13)))
        self.assertEqual(set(trajectory.fixed_joint_positions), set(BODY_JOINT_NAMES))

    def test_position_mode_switch_waits_for_lagging_status(self):
        statuses = iter((3, 3, 1))

        class Robot:
            def __init__(self):
                self.control_modes = []

            def get_motion_control_status(self):
                return SimpleNamespace(control_mode=next(statuses), error_code=0, error_msg="")

            def set_control_mode(self, value):
                self.control_modes.append(value)
                return 0

        robot = Robot()
        clock = iter((0.0, 0.1))
        enter_position_control(robot, monotonic_fn=lambda: next(clock), sleep_fn=lambda _: None)
        self.assertEqual(robot.control_modes, [1])

    def test_position_mode_switch_rejects_nonzero_setter_result(self):
        class Robot:
            def get_motion_control_status(self):
                return SimpleNamespace(control_mode=3, error_code=0, error_msg="")

            def set_control_mode(self, _):
                return 9

        with self.assertRaisesRegex(RuntimeError, "returned 9"):
            enter_position_control(Robot())


class ReplayTests(unittest.TestCase):
    def test_replay_enters_position_mode_plans_to_start_then_streams_seven_joints(self):
        names = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
        trajectory = JointTrajectory(
            "right", names, {name: 0.0 for name in BODY_JOINT_NAMES}, "0" * 64, "1" * 64,
            (JointTrajectoryPoint(0, (0.0,) * 7, 0.0, 0.0),),
        )
        report = ValidationReport(True, (), 0.0, 0.0, 0.0, 0.0)

        class Request:
            pass

        class Robot:
            def __init__(self):
                self.control_modes, self.plans, self.servo = [], [], []

            def set_control_mode(self, value):
                self.control_modes.append(value)

            def get_motion_control_status(self):
                return SimpleNamespace(control_mode=1, mode=0, error_code=0, error_msg="")

            def get_joint_states(self):
                all_names = (*BODY_JOINT_NAMES, *(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)), *names)
                return {"states": [{"name": name, "motor_position": 0.0, "error_code": 0} for name in all_names]}

            def joint_control_request(self, request):
                self.plans.append(request)

            def move_arm_joint_servo(self, positions, timestep, group, low_latency):
                self.servo.append((positions, timestep, group, low_latency))

        robot = Robot()
        sent = replay_trajectory(
            SimpleNamespace(JointControlReq=Request), robot, trajectory, report,
            speed_scale=1.0, sleep_fn=lambda _: None, monotonic_fn=lambda: 0.0,
        )

        self.assertEqual(robot.control_modes, [])
        self.assertEqual(robot.plans[0].joint_names, list(names))
        self.assertEqual(robot.servo, [([0.0] * 7, 0.01, 1, False)])
        self.assertEqual(sent, 1)

        # Split stages: the first sends only a plan; the second only servo targets.
        robot = Robot()
        options = dict(speed_scale=1.0, sleep_fn=lambda _: None, monotonic_fn=lambda: 0.0)
        self.assertEqual(replay_trajectory(SimpleNamespace(JointControlReq=Request), robot,
            trajectory, report, start_only=True, **options), 0)
        self.assertEqual(len(robot.plans), 1)
        self.assertEqual(robot.servo, [])
        self.assertEqual(replay_trajectory(SimpleNamespace(JointControlReq=Request), robot,
            trajectory, report, move_to_start=False, **options), 1)
        self.assertEqual(len(robot.plans), 1)
        self.assertEqual(len(robot.servo), 1)

    def test_live_model_check_rejects_a_wrong_end_link_transform(self):
        root = Path(__file__).resolve().parents[1]
        model = __import__("g2_kinematics").UrdfKinematics(root / "robot_description/G2_t2v2_crsB_omnipicker.urdf")
        names = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
        fixed = {name: value for name, value in zip(BODY_JOINT_NAMES, (-0.2, 0.4, -0.2, 0.0, 0.0))}
        joints = (0.1, -0.4, 0.2, -0.8, 0.1, 0.3, -0.2)
        pose_matrix = model.forward("base_link", "arm_l_end_link", {**fixed, **dict(zip(names, joints))})
        pose = SimpleNamespace(
            position=SimpleNamespace(x=pose_matrix[0, 3] + 0.02, y=pose_matrix[1, 3], z=pose_matrix[2, 3]),
            orientation=SimpleNamespace(**dict(zip(("x", "y", "z", "w"), __import__("cartesian_teach").matrix_to_quaternion(pose_matrix[:3, :3])))),
        )

        class Robot:
            def get_joint_states(self):
                values = {**fixed, **dict(zip(names, joints))}
                return {"states": [{"name": name, "motor_position": value, "error_code": 0} for name, value in values.items()]}

            def get_motion_control_status(self):
                return SimpleNamespace(error_code=0, frame_names=["arm_l_end_link"], frame_poses=[pose])

        with self.assertRaisesRegex(RuntimeError, "URDF/GDK end-link mismatch"):
            verify_live_model(model, Robot(), "left")

    def test_replay_revalidates_loaded_points_before_switching_mode(self):
        names = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
        trajectory = JointTrajectory(
            "right", names, {name: 0.0 for name in BODY_JOINT_NAMES}, "0" * 64, "1" * 64,
            (JointTrajectoryPoint(0, (99.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0, 0.0),),
        )
        report = ValidationReport(True, (), 0.0, 0.0, 0.0, 0.0)

        class Robot:
            def __init__(self):
                self.control_modes = []

            def set_control_mode(self, value):
                self.control_modes.append(value)

            def get_joint_states(self):
                all_names = (*BODY_JOINT_NAMES, *(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)), *names)
                return {"states": [{"name": name, "motor_position": 0.0, "error_code": 0} for name in all_names]}

        robot = Robot()
        with self.assertRaisesRegex(ValueError, "joint limit"):
            replay_trajectory(SimpleNamespace(), robot, trajectory, report, speed_scale=1.0)
        self.assertEqual(robot.control_modes, [])

    def test_replay_rejects_invalid_speed_parameters_before_reading_or_moving_robot(self):
        names = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
        trajectory = JointTrajectory(
            "right", names, {name: 0.0 for name in BODY_JOINT_NAMES}, "0" * 64, "1" * 64,
            (JointTrajectoryPoint(0, (0.0,) * 7, 0.0, 0.0),),
        )
        report = ValidationReport(True, (), 0.0, 0.0, 0.0, 0.0)

        class Robot:
            def get_joint_states(self):
                raise AssertionError("robot must not be read for invalid replay parameters")

        for kwargs in ({"speed_scale": float("nan")}, {"speed_scale": 1.0, "start_velocity_rad_s": 0.0}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "positive and finite"):
                replay_trajectory(SimpleNamespace(), Robot(), trajectory, report, **kwargs)

    def test_replay_reports_progress_in_ten_percent_steps(self):
        names = tuple(f"idx6{i}_arm_r_joint{i}" for i in range(1, 8))
        trajectory = JointTrajectory(
            "right", names, {name: 0.0 for name in BODY_JOINT_NAMES}, "0" * 64, "1" * 64,
            tuple(
                JointTrajectoryPoint(index * 100_000_000, (index * 0.01,) * 7, 0.0, 0.0)
                for index in range(11)
            ),
        )
        report = ValidationReport(True, (), 0.1, 0.0, 0.0, 0.0)

        class Robot:
            def __init__(self):
                self.values = {name: 0.0 for name in (*BODY_JOINT_NAMES, *(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)), *names)}

            def get_motion_control_status(self):
                return SimpleNamespace(control_mode=1, error_code=0, error_msg="")

            def get_joint_states(self):
                return {"states": [{"name": name, "motor_position": value, "error_code": 0}
                                    for name, value in self.values.items()]}

            def joint_control_request(self, request):
                for name, value in zip(request.joint_names, request.joint_positions):
                    self.values[name] = value

            def move_arm_joint_servo(self, positions, _timestep, _group, _low_latency):
                for name, value in zip(names, positions):
                    self.values[name] = value

        progress = []
        replay_trajectory(
            SimpleNamespace(JointControlReq=type("Request", (), {})), Robot(), trajectory, report,
            speed_scale=1.0, sleep_fn=lambda _: None, monotonic_fn=lambda: 0.0,
            progress_sink=progress.append,
        )

        self.assertEqual(progress, [f"回放点进度：{value}%" for value in range(0, 101, 10)])


class OfflineCliTests(unittest.TestCase):
    def test_record_parser_exposes_leading_motion_thresholds(self):
        args = _parser().parse_args([
            "record", "--side", "left", "--output", "record.json",
            "--leading-motion-mm", "1.2", "--leading-motion-deg", "2.3",
        ])

        self.assertAlmostEqual(args.leading_motion_mm, 1.2)
        self.assertAlmostEqual(args.leading_motion_deg, 2.3)

    def test_solve_command_writes_a_replay_ready_joint_file_without_gdk(self):
        root = Path(__file__).resolve().parents[1]
        urdf = root / "robot_description/G2_t2v2_crsB_omnipicker.urdf"
        model = __import__("g2_kinematics").UrdfKinematics(urdf)
        names = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
        fixed = {name: value for name, value in zip(BODY_JOINT_NAMES, (-0.2, 0.4, -0.2, 0.0, 0.0))}
        joints = (0.1, -0.4, 0.2, -0.8, 0.1, 0.3, -0.2)
        pose = model.forward("base_link", "arm_l_end_link", {**fixed, **dict(zip(names, joints))})
        raw = CartesianTrajectory(
            "left", "base_link", "arm_l_end_link", names, fixed,
            (CartesianSample(0, tuple(pose[:3, 3]), __import__("cartesian_teach").matrix_to_quaternion(pose[:3, :3]), joints),),
        )
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "raw.json", Path(directory) / "solved.json"
            source.write_text(json.dumps(cartesian_document(raw, urdf_sha256="ignored")), encoding="utf-8")

            code = main(["solve", "--input", str(source), "--output", str(output), "--urdf", str(urdf)])

            self.assertEqual(code, 0)
            _, report = load_joint_document(json.loads(output.read_text(encoding="utf-8")))
            self.assertTrue(report.replay_ready, report.errors)


if __name__ == "__main__":
    unittest.main()
