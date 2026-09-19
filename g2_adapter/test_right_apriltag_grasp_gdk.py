import math
import json
import unittest
import contextlib
import io
import tempfile
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from right_apriltag_grasp_gdk import wait_for_manual_finish, return_screwdriver
from unittest.mock import Mock


class RightEntryPreparationTests(unittest.TestCase):
    def test_execute_prepares_right_before_worker_premove_and_grasp(self):
        self._run_entry(execute=True, worker=True, premove=True)

    def test_preparation_failure_prevents_worker_and_motion(self):
        self._run_entry(execute=True, worker=True, premove=True, fail=True)

    def test_read_only_with_or_without_allow_motion_does_not_prepare(self):
        for allow_motion in (False, True):
            with self.subTest(allow_motion=allow_motion):
                self._run_entry(execute=False, allow_motion=allow_motion)

    def test_mode_three_is_verified_before_gripper_motion(self):
        self._run_entry(execute=True, verified_mode=1)

    def _run_entry(self, *, execute, worker=False, premove=False, fail=False,
                   allow_motion=False, verified_mode=3):
        import right_apriltag_grasp_gdk as module
        from apriltag_workpiece import GraspTargets
        from control import CartesianPoseTarget

        events = []
        controller = Mock()
        controller.read_end_effector_pose.return_value = CartesianPoseTarget((0.4, -0.3, 0.8), (0, 0, 0, 1))
        controller.motion_status_summary.return_value = {"mode": 0, "control_mode": verified_mode, "error_code": 0}
        def prepare(sides):
            self.assertEqual(sides, ("right",))
            events.append("prepare")
            if fail:
                raise RuntimeError("preparation failed")
        controller.prepare_joint_impedance_control.side_effect = prepare
        gdk = Mock()
        gdk.gdk_init.return_value = gdk.GDKRes.kSuccess
        gdk.Camera.return_value.get_latest_image.return_value = SimpleNamespace(timestamp_ns=1, width=8, height=8)
        gdk.TF.return_value.lookup_transform_latest.return_value = (Mock(), 1)
        estimator = Mock(sample_counts={}, last_diagnostics={})
        estimator.estimate.return_value = SimpleNamespace(camera_from_workpiece=np.eye(4))
        targets = GraspTargets(np.eye(4), np.eye(4), np.eye(4))
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            config = json.loads(Path(module.__file__).with_name("right_apriltag_grasp.example.json").read_text())
            config.pop("grasp_distance_mm", None)
            config_path = Path(directory) / "config.json"
            config_path.write_text(json.dumps(config))
            argv = ["--config", str(config_path), "--max-frames", "1", "--web-port", "0",
                    "--terminal-output", "json", "--log", str(Path(directory) / "log.jsonl"), "--stop-after-transport"]
            if execute:
                argv += ["--execute", "--allow-motion"]
            elif allow_motion:
                argv += ["--allow-motion"]
            if worker:
                argv += ["--spacemouse-device", "/dev/fake"]
            if premove:
                argv += ["--premove-pr1"]
            stack.enter_context(patch.dict("sys.modules", {"agibot_gdk": gdk}))
            patches = {
                "G2Controller": controller, "detector_for_41h12": Mock(),
                "RollingWorkpiecePoseEstimator": estimator,
                "decode_color_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
                "intrinsic_to_dict": {"fx": 1, "fy": 1, "ppx": 0, "ppy": 0, "distortion": ()},
                "transform_to_dict": {}, "inverse_transform_dict": {}, "transform_from_dict": np.eye(4),
                "camera_to_base_at": ({}, {"mode": "exact_image_time"}),
                "detect_corners": {}, "g2_grasp_targets": targets, "horizontal_grasp_targets": targets,
                "target_record": {}, "load_right_pose": [0.0] * 7,
            }
            for name, result in patches.items():
                stack.enter_context(patch.object(module, name, return_value=result))
            gate = stack.enter_context(patch.object(module, "MotionPauseGate"))
            gate.return_value.is_paused.return_value = False
            worker_class = stack.enter_context(patch.object(module, "SpaceMouseTakeoverWorker"))
            worker_class.return_value.start.side_effect = lambda: events.append("worker")
            stack.enter_context(patch.object(module, "DianpiCycleRunner"))
            stack.enter_context(patch.object(module, "move_right_preset", side_effect=lambda *a, **k: events.append("premove") or {}))
            stack.enter_context(patch.object(module, "move_right_gripper_with_feedback", side_effect=lambda *a, **k: events.append("open") or {}))
            stack.enter_context(patch.object(module, "execute_local_grasp", side_effect=lambda *a, **k: events.append("grasp") or {}))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            errors = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            result = module.main(argv)
            self.assertEqual(result, 1 if fail or (execute and verified_mode != 3) else 0, errors.getvalue())
            if not execute:
                self.assertEqual(events, [])
            elif fail:
                self.assertEqual(events, ["prepare"])
                worker_class.assert_not_called()
            elif verified_mode != 3:
                self.assertEqual(events, ["prepare", "prepare"])
            else:
                self.assertEqual(events, ["prepare", "worker", "premove", "prepare", "open", "grasp"])
            controller.set_control_mode.assert_not_called()
            gdk.gdk_release.assert_called_once()


class ManualFinishTests(unittest.TestCase):
    def test_facing_distance_aligns_red_center_in_base_frame(self):
        from right_apriltag_grasp_gdk import facing_distance_targets, RED_GRIP_CENTER_M
        from apriltag_workpiece import GraspTargets
        grasp = np.eye(4)
        grasp[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
        workpiece = np.eye(4)
        workpiece[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        workpiece[:3, 3] = [0.5, -0.3, 0.8]
        center = workpiece[:3, :3] @ RED_GRIP_CENTER_M + workpiece[:3, 3]
        for distance in (10, 100, 300):
            result = facing_distance_targets(GraspTargets(workpiece, grasp, grasp), distance)
            np.testing.assert_allclose(result.base_from_pregrasp_end[:3, 3] + grasp[:3, 2] * distance / 1000, center)
            np.testing.assert_allclose(result.base_from_grasp_end[:3, 3], center)
            np.testing.assert_allclose(result.base_from_pregrasp_end[:3, :3], grasp[:3, :3])
        for distance in (0, 301, float("nan")):
            with self.assertRaises(ValueError):
                facing_distance_targets(GraspTargets(np.eye(4), grasp, grasp), distance)

    def test_horizontal_grasp_preserves_tcp_position_and_horizontal_approach(self):
        from right_apriltag_grasp_gdk import horizontal_grasp_targets
        from apriltag_workpiece import GraspTargets
        angle = 0.2
        grasp = np.eye(4)
        grasp[:3, :3] = [[0, -math.sin(angle), math.cos(angle)], [1, 0, 0], [0, math.cos(angle), math.sin(angle)]]
        grasp[:3, 3] = [0.4, -0.3, 0.8]
        pre = grasp.copy()
        pre[:3, 3] -= 0.03 * grasp[:3, 2]
        tcp = np.eye(4)
        tcp[:3, 3] = [0, 0.028, 0.145]
        result = horizontal_grasp_targets(GraspTargets(np.eye(4), pre, grasp), tcp)
        actual = result.base_from_grasp_end
        self.assertTrue(np.allclose(actual[:3, 1], [0, 0, 1]))
        self.assertAlmostEqual(actual[2, 2], 0)
        self.assertAlmostEqual(np.linalg.det(actual[:3, :3]), 1)
        self.assertTrue(np.allclose((actual @ tcp)[:3, 3], (grasp @ tcp)[:3, 3]))
        self.assertTrue(np.allclose(actual[:3, 3] - result.base_from_pregrasp_end[:3, 3], 0.03 * actual[:3, 2]))

    def test_return_visits_clearance_then_saved_pose_before_release(self):
        saved = CartesianPoseTarget((0.4, -0.3, 0.8), (0, 0, 0, 1))
        controller = Mock()
        controller.read_end_effector_pose.return_value = saved
        stream = controller.start_right_end_effector_position_stream.return_value
        visited = []
        def arrival(target, *_args, **_kwargs):
            visited.append(target)
            return target
        stream.wait_until_actual_target.side_effect = arrival
        config = {
            "autonomous_approach": {"speed_mps": 0.018, "angular_speed_deg_s": 18},
            "local_motion": {"final_actual_position_tolerance_m": 0.0005, "final_actual_orientation_tolerance_deg": 0.5},
            "gripper": {"open_rad": -0.785},
        }
        def opened(*_args):
            self.assertEqual(len(visited), 2)
            stream.stop.assert_called_once()
            return {}
        with patch("right_apriltag_grasp_gdk.move_right_gripper_with_feedback", side_effect=opened):
            return_screwdriver(controller, saved, config, lambda event: None)
        self.assertAlmostEqual(visited[0].position_m[2], 0.9)
        self.assertEqual(visited[0].orientation_xyzw, saved.orientation_xyzw)
        self.assertEqual(visited[1], saved)
        for failed_stage in (0, 1):
            stream.wait_until_actual_target.side_effect = [None] if failed_stage == 0 else [visited[0], None]
            with patch("right_apriltag_grasp_gdk.move_right_gripper_with_feedback") as release:
                with self.assertRaisesRegex(RuntimeError, "不松爪"):
                    return_screwdriver(controller, saved, config, lambda event: None)
                release.assert_not_called()

    def test_return_can_defer_the_gripper_release_to_the_final_stage(self):
        saved = CartesianPoseTarget((0.4, -0.3, 0.8), (0, 0, 0, 1))
        controller = Mock()
        controller.read_end_effector_pose.return_value = saved
        stream = controller.start_right_end_effector_position_stream.return_value
        stream.wait_until_actual_target.side_effect = lambda target, *_a, **_k: target
        config = {
            "autonomous_approach": {"speed_mps": 0.018, "angular_speed_deg_s": 18},
            "local_motion": {"final_actual_position_tolerance_m": 0.0005, "final_actual_orientation_tolerance_deg": 0.5},
            "gripper": {"open_rad": -0.785},
        }
        events = []
        with patch("right_apriltag_grasp_gdk.move_right_gripper_with_feedback") as release:
            return_screwdriver(controller, saved, config, events.append, release=False)
        release.assert_not_called()
        self.assertEqual(events[-1]["event"], "right_return_complete_gripper_held")
        self.assertEqual([event["event"] for event in events].count("right_return_arrived"), 2)

    def test_parse_args_accepts_an_explicit_argv_and_the_staged_stop_flag(self):
        from right_apriltag_grasp_gdk import parse_args
        args = parse_args(["--execute", "--allow-motion", "--stop-after-transport",
                           "--premove-pose", "RR1", "--post-grasp-pose", "RR2"])
        self.assertTrue(args.stop_after_transport)
        self.assertEqual(args.premove_pose, "RR1")
        self.assertEqual(args.post_grasp_pose, "RR2")
        default = parse_args([])
        self.assertFalse(default.stop_after_transport)
        self.assertIsNone(default.grasp_distance_mm)
        self.assertEqual(parse_args(["--grasp-distance-mm", "165"]).grasp_distance_mm, 165.0)

    def test_standalone_main_delegates_to_the_injectable_stage(self):
        import right_apriltag_grasp_gdk as module
        with patch.object(module, "run_right_grasp_stage", return_value=7) as stage:
            self.assertEqual(module.main(["--max-frames", "1"]), 7)
        self.assertIsNone(stage.call_args.kwargs["services"])
        self.assertEqual(stage.call_args.kwargs["argv"], ["--max-frames", "1"])

    def test_every_args_reference_in_the_module_is_a_defined_cli_flag(self):
        import inspect
        import pathlib
        import re
        import right_apriltag_grasp_gdk as module
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        used = set(re.findall(r"\bargs\.([A-Za-z_][A-Za-z0-9_]*)", source))
        defined = set(vars(module.parse_args([])))
        self.assertEqual(sorted(used - defined), [])

    def test_wait_keeps_worker_alive_until_manual_finish(self):
        worker = SimpleNamespace(allow_advance_right=False, advance_requested=False)
        checks = []
        worker.raise_if_failed = lambda: checks.append(True)

        def sleep(_seconds):
            self.assertTrue(worker.allow_advance_right)
            if len(checks) == 3:
                worker.advance_requested = True

        wait_for_manual_finish(worker, sleep=sleep)
        self.assertEqual(len(checks), 4)

    def test_wait_refreshes_preview_until_manual_finish(self):
        worker = SimpleNamespace(allow_advance_right=False, advance_requested=False)
        worker.raise_if_failed = lambda: None
        previews = []

        def preview():
            previews.append(True)
            if len(previews) == 3:
                worker.advance_requested = True

        wait_for_manual_finish(worker, preview=preview)
        self.assertEqual(len(previews), 3)

from control import CartesianPoseTarget, G2Controller
from right_apriltag_grasp_gdk import (
    bounded_pose_waypoints,
    close_right_gripper,
    close_right_gripper_on_pose_stream,
    execute_autonomous_pregrasp,
    execute_live_autonomous_pregrasp,
    execute_local_grasp,
    apply_auto_motion_overrides,
    move_right_gripper_with_feedback,
    quaternion_angle_deg,
    VisualLockTimeout,
    load_tag_geometry,
    calibration_terminal_message,
    format_terminal_status,
    SharedView,
    publish_shared_view,
    orientation_relock_needed,
    orientation_relock_progress,
    targets_with_locked_orientation,
    visual_lock_pose_ready,
)
from test_control import FakeGdk, FakeRobot


class RightAprilTagApproachTests(unittest.TestCase):
    def test_auto_motion_overrides_set_speeds_without_scaling_timeouts(self):
        config = {
            "autonomous_approach": {
                "speed_mps": 0.010,
                "angular_speed_deg_s": 10.0,
                "waypoint_timeout_s": 8.0,
                "max_total_duration_s": 120.0,
            },
            "local_motion": {
                "final_speed_mps": 0.005,
                "final_timeout_s": 10.0,
            },
        }
        updated = apply_auto_motion_overrides(
            config,
            speed_mm_s=2.5,
            rotation_deg_s=6.0,
        )
        self.assertAlmostEqual(updated["autonomous_approach"]["speed_mps"], 0.0025)
        self.assertAlmostEqual(updated["autonomous_approach"]["angular_speed_deg_s"], 6.0)
        self.assertAlmostEqual(updated["autonomous_approach"]["waypoint_timeout_s"], 8.0)
        self.assertAlmostEqual(updated["autonomous_approach"]["max_total_duration_s"], 120.0)
        self.assertAlmostEqual(updated["local_motion"]["final_speed_mps"], 0.0025)
        self.assertAlmostEqual(updated["local_motion"]["final_timeout_s"], 10.0)
        self.assertAlmostEqual(config["local_motion"]["final_speed_mps"], 0.005)

    def test_example_calibration_gate_is_three_pixels_for_grasp_and_pregrasp(self):
        config = json.loads(
            (Path(__file__).with_name("right_apriltag_grasp.example.json")).read_text(encoding="utf-8")
        )
        self.assertEqual(config["calibration_max_reprojection_error_px"], 3.0)

    def test_terminal_status_is_one_line_and_includes_pnp_rejection(self):
        line = format_terminal_status(
            frame=20,
            visible_tag_ids=[0, 1, 2],
            sample_counts={0: 20, 1: 20, 2: 20},
            pnp_diagnostics={
                "state": "pnp_reprojection_rejected",
                "mean_reprojection_error_px": 3.21,
                "max_reprojection_error_px": 9.30,
                "inlier_count": 8,
            },
        )
        self.assertNotIn("\n", line)
        self.assertIn("帧 20", line)
        self.assertIn("Tag 0,1,2", line)
        self.assertIn("sample 0:20/20 1:20/20 2:20/20", line)
        self.assertIn("rejected mean=3.21px max=9.30px inliers=8", line)

    def test_terminal_calibration_message_reports_rejection_instead_of_success(self):
        message = calibration_terminal_message(
            "pregrasp_calibration",
            "pregrasp_calibration_rejected",
            "横向偏差 10.6 mm；要求不超过 3.0 mm",
        )
        self.assertIn("\n", message)
        self.assertIn("未通过", message)
        self.assertIn("横向偏差 10.6 mm", message)
        self.assertNotIn("候选已生成", message)

    def test_terminal_calibration_message_reports_only_valid_candidate_as_success(self):
        message = calibration_terminal_message(
            "pregrasp_calibration",
            "right_pregrasp_calibration_candidate",
            None,
        )
        self.assertEqual(message, "\n右手 pregrasp 标定候选已生成。")

    def test_shared_view_sequence_changes_on_each_frame_update(self):
        shared = SharedView()
        shared.update(b"first", {"state": "frame", "sequence": 20})
        first_image, first_status = shared.snapshot()
        shared.update(b"second", {"state": "visual_lock_waiting", "sequence": 20})
        second_image, second_status = shared.snapshot()
        self.assertEqual(first_image, b"first")
        self.assertEqual(second_image, b"second")
        self.assertEqual(first_status["sequence"], 1)
        self.assertEqual(second_status["sequence"], 2)

    def test_shared_view_status_heartbeat_keeps_last_image_and_advances_sequence(self):
        shared = SharedView()
        shared.update(b"last-frame", {"state": "frame"})
        shared.update_status({"state": "visual_lock_waiting", "motion": "hold_current_pose"})
        image, status = shared.snapshot()
        self.assertEqual(image, b"last-frame")
        self.assertEqual(status["state"], "visual_lock_waiting")
        self.assertEqual(status["motion"], "hold_current_pose")
        self.assertEqual(status["sequence"], 2)

    def test_publish_shared_view_exposes_visual_lock_motion(self):
        shared = SharedView()
        with patch("right_apriltag_grasp_gdk.draw_overlay", return_value=b"jpeg"):
            publish_shared_view(
                shared,
                np.zeros((32, 32, 3), dtype=np.uint8),
                {},
                {
                    "event": "frame",
                    "visible_tag_ids": [],
                    "sample_counts": {0: 20, 1: 20, 2: 20},
                    "pnp_diagnostics": {"state": "collecting_samples"},
                },
                motion="hold_current_pose",
            )
        image, status = shared.snapshot()
        self.assertIsInstance(image, bytes)
        self.assertEqual(status["state"], "frame")
        self.assertEqual(status["motion"], "hold_current_pose")

    def test_visual_lock_accepts_current_workpiece_inlier_count_but_rejects_weak_pose(self):
        self.assertTrue(visual_lock_pose_ready(SimpleNamespace(inlier_count=10), 8))
        self.assertFalse(visual_lock_pose_ready(SimpleNamespace(inlier_count=7), 8))
        self.assertFalse(visual_lock_pose_ready(None, 8))

    def test_visual_servo_keeps_locked_orientation_while_following_translation(self):
        raw_pregrasp = np.eye(4, dtype=np.float64)
        raw_grasp = np.eye(4, dtype=np.float64)
        raw_pregrasp[:3, 3] = (0.30, -0.20, 0.40)
        raw_grasp[:3, 3] = (0.30, -0.20, 0.35)
        angle = math.radians(8.0)
        raw_pregrasp[:3, :3] = np.array(
            [[math.cos(angle), -math.sin(angle), 0.0],
             [math.sin(angle), math.cos(angle), 0.0],
             [0.0, 0.0, 1.0]],
        )
        raw_grasp[:3, :3] = raw_pregrasp[:3, :3]
        raw_targets = SimpleNamespace(
            base_from_workpiece=np.eye(4, dtype=np.float64),
            base_from_pregrasp_end=raw_pregrasp,
            base_from_grasp_end=raw_grasp,
        )
        locked = targets_with_locked_orientation(raw_targets, np.eye(3, dtype=np.float64))
        self.assertTrue(np.allclose(locked.base_from_pregrasp_end[:3, 3], raw_pregrasp[:3, 3]))
        self.assertTrue(np.allclose(locked.base_from_grasp_end[:3, 3], raw_grasp[:3, 3]))
        self.assertTrue(np.allclose(locked.base_from_pregrasp_end[:3, :3], np.eye(3)))
        self.assertTrue(np.allclose(locked.base_from_grasp_end[:3, :3], np.eye(3)))

    def test_visual_servo_relocks_only_after_persistent_orientation_change(self):
        identity_q = (0.0, 0.0, 0.0, 1.0)
        one_degree_q = (0.0, 0.0, math.sin(math.radians(0.5)), math.cos(math.radians(0.5)))
        three_degree_q = (0.0, 0.0, math.sin(math.radians(1.5)), math.cos(math.radians(1.5)))
        self.assertFalse(orientation_relock_needed(identity_q, one_degree_q, threshold_deg=2.0))
        self.assertTrue(orientation_relock_needed(identity_q, three_degree_q, threshold_deg=2.0))
        previous = None
        count = 0
        for _ in range(2):
            previous, count, ready = orientation_relock_progress(
                three_degree_q, previous, count,
                consistency_deg=1.0, required_count=3,
            )
            self.assertFalse(ready)
        previous, count, ready = orientation_relock_progress(
            three_degree_q, previous, count,
            consistency_deg=1.0, required_count=3,
        )
        self.assertTrue(ready)
        self.assertEqual(count, 3)

    def test_loads_electric_screwdriver_geometry_without_changing_legacy_defaults(self):
        geometry = load_tag_geometry({
            "tag_geometry_profile": "g2_adapter/tag_geometry_electric_screwdriver.json",
        })
        self.assertTrue(np.allclose(geometry.centers_m[0], [0.0, -0.0415, -0.0075]))
        self.assertTrue(np.allclose(geometry.centers_m[1], [-0.03475, -0.05148543023, -0.01379941798]))

    def test_bounded_waypoints_limit_translation_and_orientation(self):
        start = CartesianPoseTarget((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        target = CartesianPoseTarget(
            (0.051, 0.0, 0.0),
            (0.0, 0.0, math.sin(math.radians(10.0)), math.cos(math.radians(10.0))),
        )
        waypoints = bounded_pose_waypoints(
            start, target, max_translation_m=0.020, max_orientation_deg=3.0,
        )
        self.assertEqual(len(waypoints), 7)
        previous = start
        for waypoint in waypoints:
            self.assertLessEqual(
                np.linalg.norm(np.asarray(waypoint.position_m) - np.asarray(previous.position_m)),
                0.020 + 1e-9,
            )
            self.assertLessEqual(
                quaternion_angle_deg(previous.orientation_xyzw, waypoint.orientation_xyzw),
                3.0 + 1e-9,
            )
            previous = waypoint
        self.assertAlmostEqual(waypoints[-1].position_m[0], target.position_m[0], places=9)
        for actual, expected in zip(waypoints[-1].orientation_xyzw, target.orientation_xyzw):
            self.assertAlmostEqual(actual, expected, places=9)

    def test_autonomous_pregrasp_starts_from_actual_coarse_pose(self):
        robot = FakeRobot()
        robot.right_end_pose.position.x = 0.0
        robot.right_end_pose.position.y = -0.1
        robot.right_end_pose.position.z = 0.3
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        current = CartesianPoseTarget((0.0, -0.1, 0.3), (0.0, 0.0, 0.0, 1.0))
        target = CartesianPoseTarget(
            (0.041, -0.1, 0.3),
            (0.0, 0.0, math.sin(math.radians(10.0)), math.cos(math.radians(10.0))),
        )
        config = {
            "autonomous_approach": {
                "enabled": True,
                "max_start_distance_m": 0.2,
                "max_start_orientation_change_deg": 20.0,
                "max_waypoint_translation_m": 0.02,
                "max_waypoint_orientation_deg": 3.0,
                "speed_mps": 0.03,
                "angular_speed_deg_s": 30.0,
                "waypoint_timeout_s": 2.0,
                "actual_position_tolerance_m": 0.0001,
                "actual_orientation_tolerance_deg": 0.2,
                "stable_samples": 2,
            },
        }
        result = execute_autonomous_pregrasp(controller, current, target, config)
        self.assertLess(np.linalg.norm(np.asarray(result.position_m) - np.asarray(target.position_m)), 1e-9)
        self.assertLess(quaternion_angle_deg(result.orientation_xyzw, target.orientation_xyzw), 1e-6)
        self.assertGreater(len(robot.requests), 1)

    def test_autonomous_pregrasp_allows_far_and_rotated_coarse_start(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        current = CartesianPoseTarget((0.0, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))
        target = CartesianPoseTarget(
            (0.25, 0.0, 0.3),
            (0.0, 0.0, math.sin(math.radians(12.5)), math.cos(math.radians(12.5))),
        )
        config = {
            "autonomous_approach": {
                "enabled": True,
                "max_waypoint_translation_m": 0.02,
                "max_waypoint_orientation_deg": 3.0,
                "speed_mps": 0.03,
                "angular_speed_deg_s": 30.0,
                "waypoint_timeout_s": 2.0,
                "actual_position_tolerance_m": 0.0001,
                "actual_orientation_tolerance_deg": 0.2,
                "stable_samples": 2,
            },
        }
        result = execute_autonomous_pregrasp(controller, current, target, config)
        self.assertLess(np.linalg.norm(np.asarray(result.position_m) - np.asarray(target.position_m)), 1e-9)
        self.assertLess(quaternion_angle_deg(result.orientation_xyzw, target.orientation_xyzw), 1e-6)
        self.assertGreaterEqual(len(robot.requests), 13)

    def test_local_grasp_accepts_float_roundoff_between_target_rotations(self):
        robot = FakeRobot()
        robot.right_end_pose.position.x = 0.1
        robot.right_end_pose.position.y = -0.2
        robot.right_end_pose.position.z = 0.3
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        pregrasp = np.eye(4, dtype=np.float64)
        pregrasp[:3, 3] = (0.1, -0.2, 0.3)
        grasp = pregrasp.copy()
        grasp[0, 0] = 1.0 - 2e-16
        grasp[0, 1] = -2e-8
        grasp[1, 0] = 2e-8
        grasp[1, 1] = 1.0 - 2e-16
        grasp[0, 3] += 0.001
        config = {
            "calibrated": True,
            "right_mode1_probe_verified": True,
            "local_motion": {
                "max_initial_pregrasp_error_m": 0.003,
                "max_initial_orientation_error_deg": 0.5,
                "max_final_approach_m": 0.05,
                "final_speed_mps": 0.01,
                "final_timeout_s": 1.0,
                "final_actual_position_tolerance_m": 0.0005,
                "final_actual_orientation_tolerance_deg": 0.5,
                "final_actual_stable_samples": 2,
            },
            "gripper": {"closed_rad": 0.0, "close_settle_s": 0.0, "close_drain_s": 0.0},
        }
        execute_local_grasp(
            controller,
            SimpleNamespace(base_from_pregrasp_end=pregrasp, base_from_grasp_end=grasp),
            config,
        )
        self.assertEqual(robot.gripper_requests, [])
        self.assertEqual(robot.requests[-1].end_effector_joint_names, ["right_omnipicker"])
        self.assertEqual(robot.requests[-1].end_effector_joint_positions, [0.0])

    def test_right_gripper_close_on_pose_stream_reaches_target(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        initial = CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
        stream = controller.start_right_end_effector_position_stream(initial)
        try:
            closed = close_right_gripper_on_pose_stream(
                controller, stream, 0.0, settle_s=0.0,
            )
        finally:
            stream.stop()
        self.assertEqual(closed["command_path"], "right_end_effector_pose_stream")
        self.assertEqual(closed["close_state"], "target_reached")
        self.assertAlmostEqual(closed["actual_position_rad"], 0.0, places=6)
        self.assertEqual(robot.gripper_requests, [])

    def test_right_gripper_close_on_pose_stream_rejects_no_motion(self):
        robot = FakeRobot()
        robot.follow_right_gripper = False
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        stream = controller.start_right_end_effector_position_stream(
            CartesianPoseTarget((0.1, -0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
        )
        try:
            with self.assertRaises(RuntimeError):
                close_right_gripper_on_pose_stream(
                    controller, stream, 0.0, settle_s=0.12,
                )
        finally:
            stream.stop()

    def test_right_gripper_opens_before_close_and_reports_feedback(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        opened = move_right_gripper_with_feedback(
            controller, -0.785, tolerance_rad=0.05, timeout_s=0.5,
        )
        self.assertAlmostEqual(opened["actual_position_rad"], -0.785, places=6)
        closed = close_right_gripper(controller, 0.0, settle_s=0.0)
        self.assertAlmostEqual(closed["actual_position_rad"], 0.0, places=6)
        self.assertEqual(closed["close_state"], "target_reached")
        self.assertEqual([item[0][0] for item in robot.gripper_requests[:2]], [-0.785, 0.0])
        self.assertEqual(closed["attempts"], 1)

    def test_right_gripper_close_repeats_nonblocking_command_until_settled(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        closed = close_right_gripper(controller, 0.0, settle_s=0.21)
        self.assertGreaterEqual(closed["attempts"], 3)
        self.assertAlmostEqual(closed["motion_toward_target_rad"], 0.785, places=3)

    def test_right_gripper_close_rejects_feedback_that_does_not_move(self):
        robot = FakeRobot()
        robot.follow_right_gripper = False
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        with self.assertRaises(RuntimeError):
            close_right_gripper(controller, 0.0, settle_s=0.0, min_motion_rad=0.02)

    def test_live_autonomous_pregrasp_reacquires_target_after_each_step(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        identity = np.eye(4, dtype=np.float64)

        def make_targets(x):
            pregrasp = identity.copy()
            pregrasp[0, 3] = x
            grasp = pregrasp.copy()
            grasp[0, 3] += 0.001
            return SimpleNamespace(
                base_from_pregrasp_end=pregrasp,
                base_from_grasp_end=grasp,
            )

        initial_targets = make_targets(0.009)
        lock_targets = [make_targets(0.009), make_targets(0.011), make_targets(0.012), make_targets(0.012)]
        lock_workpieces = []
        for index in range(len(lock_targets)):
            workpiece = identity.copy()
            workpiece[0, 3] = index * 0.001
            lock_workpieces.append(workpiece)
        calls = []

        def acquire_lock(*, moving=False):
            index = min(len(calls), len(lock_targets) - 1)
            calls.append(index)
            return {
                "targets": lock_targets[index],
                "base_from_workpiece": lock_workpieces[index],
            }

        config = {
            "local_motion": {
                "max_initial_pregrasp_error_m": 0.003,
                "max_initial_orientation_error_deg": 0.5,
            },
            "autonomous_approach": {
                "enabled": True,
                "max_start_distance_m": 0.2,
                "max_start_orientation_change_deg": 20.0,
                "max_waypoint_translation_m": 0.005,
                "max_waypoint_orientation_deg": 2.0,
                "speed_mps": 0.03,
                "angular_speed_deg_s": 30.0,
                "waypoint_timeout_s": 1.0,
                "actual_position_tolerance_m": 0.0001,
                "actual_orientation_tolerance_deg": 0.2,
                "stable_samples": 2,
                "max_target_delta_m": 0.03,
                "max_target_delta_deg": 5.0,
                "max_iterations": 10,
                "max_total_duration_s": 5.0,
            },
        }

        actual, final_targets = execute_live_autonomous_pregrasp(
            controller, initial_targets, acquire_lock, config,
        )
        self.assertGreaterEqual(len(calls), 4)
        self.assertLessEqual(abs(actual.position_m[0] - 0.012), 0.003 + 1e-9)
        self.assertAlmostEqual(final_targets.base_from_pregrasp_end[0, 3], 0.012, places=9)

    def test_live_autonomous_holds_and_retries_after_temporary_visual_loss(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        identity = np.eye(4, dtype=np.float64)
        pregrasp = identity.copy()
        pregrasp[0, 3] = 0.006
        targets = SimpleNamespace(base_from_pregrasp_end=pregrasp, base_from_grasp_end=pregrasp.copy())
        calls = []

        def acquire_lock(*, moving=False):
            calls.append(len(calls))
            if len(calls) == 1:
                raise VisualLockTimeout("temporary loss")
            return {"targets": targets, "base_from_workpiece": identity.copy()}

        config = {
            "local_motion": {
                "max_initial_pregrasp_error_m": 0.003,
                "max_initial_orientation_error_deg": 0.5,
            },
            "autonomous_approach": {
                "enabled": True,
                "max_start_distance_m": 0.2,
                "max_start_orientation_change_deg": 20.0,
                "max_waypoint_translation_m": 0.005,
                "max_waypoint_orientation_deg": 2.0,
                "speed_mps": 0.03,
                "angular_speed_deg_s": 30.0,
                "waypoint_timeout_s": 1.0,
                "actual_position_tolerance_m": 0.0001,
                "actual_orientation_tolerance_deg": 0.2,
                "stable_samples": 2,
                "visual_lock_hold_timeout_s": 0.5,
                "max_target_delta_m": 0.03,
                "max_target_delta_deg": 5.0,
                "max_iterations": 10,
                "max_total_duration_s": 5.0,
            },
        }

        actual, _ = execute_live_autonomous_pregrasp(controller, targets, acquire_lock, config)
        self.assertGreaterEqual(len(calls), 3)
        self.assertLessEqual(abs(actual.position_m[0] - 0.006), 0.003 + 1e-9)

    def test_live_autonomous_confirms_jump_before_following_new_target(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        identity = np.eye(4, dtype=np.float64)

        def make_targets(x):
            pregrasp = identity.copy()
            pregrasp[0, 3] = x
            return SimpleNamespace(base_from_pregrasp_end=pregrasp, base_from_grasp_end=pregrasp.copy())

        old_targets = make_targets(0.009)
        new_targets = make_targets(0.012)
        jump_workpiece = identity.copy()
        jump_workpiece[0, 3] = 0.050
        locks = [
            (old_targets, identity.copy()),
            (new_targets, jump_workpiece),
            (new_targets, jump_workpiece),
            (new_targets, jump_workpiece),
        ]
        calls = []

        def acquire_lock(*, moving=False):
            index = min(len(calls), len(locks) - 1)
            calls.append(index)
            targets, workpiece = locks[index]
            return {"targets": targets, "base_from_workpiece": workpiece}

        config = {
            "local_motion": {
                "max_initial_pregrasp_error_m": 0.003,
                "max_initial_orientation_error_deg": 0.5,
            },
            "autonomous_approach": {
                "enabled": True,
                "max_start_distance_m": 0.2,
                "max_start_orientation_change_deg": 20.0,
                "max_waypoint_translation_m": 0.005,
                "max_waypoint_orientation_deg": 2.0,
                "speed_mps": 0.03,
                "angular_speed_deg_s": 30.0,
                "waypoint_timeout_s": 1.0,
                "actual_position_tolerance_m": 0.0001,
                "actual_orientation_tolerance_deg": 0.2,
                "stable_samples": 2,
                "visual_lock_hold_timeout_s": 0.5,
                "max_target_delta_m": 0.03,
                "max_target_delta_deg": 5.0,
                "jump_reacquire_confirmations": 3,
                "jump_cluster_delta_m": 0.01,
                "jump_cluster_delta_deg": 3.0,
                "max_iterations": 10,
                "max_total_duration_s": 5.0,
            },
        }

        actual, final_targets = execute_live_autonomous_pregrasp(
            controller, old_targets, acquire_lock, config,
        )
        self.assertGreaterEqual(len(calls), 6)
        self.assertLessEqual(abs(actual.position_m[0] - 0.012), 0.003 + 1e-9)
        self.assertAlmostEqual(final_targets.base_from_pregrasp_end[0, 3], 0.012, places=9)


if __name__ == "__main__":
    unittest.main()
