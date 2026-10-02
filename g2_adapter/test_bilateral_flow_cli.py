"""Orchestrator plumbing: stage plan CLI, dry-run validation and stage argv."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import left_grasp_gdk
import right_apriltag_grasp_gdk
import run_bilateral_flow_gdk as flow
from bilateral_services import StageServices
from bilateral_stages import DEFAULT_PLAN, Stage, stage_names


def write_poses(path: Path, *, complete: bool) -> Path:
    poses = {"workflow_home": {"side": "both", "joint_names": None, "joint_positions_rad": [0.0] * 14}}
    if complete:
        for name, side in (("LR1", "right"), ("LR2", "right"), ("RR1", "right"), ("RR2", "right"), ("LL1", "left"), ("LL2", "left")):
            poses[name] = {"side": side, "joint_positions_rad": [0.0] * 7}
    path.write_text(json.dumps({"schema_version": 1, "poses": poses}), encoding="utf-8")
    return path


class OrchestratorCliTests(unittest.TestCase):
    def test_tts_can_start_for_workflow_speech_when_vlm_is_disabled(self):
        args = flow.parse_args(['--tts-enable', 'true', '--vlm-enable', 'false'])
        self.assertTrue(flow.should_start_tts(args))

    def test_child_stdout_forwards_selected_arrival_events_without_json_on_console(self):
        output = io.StringIO()
        events = []
        stream = flow.JsonLineSuppressingStream(output, event_sink=events.append)
        stream.write('{"event":"left_above_hinge_arrived"}\n')
        stream.write('{"event":"left_stage_complete_above_hole"}\n')
        stream.write('{"event":"right_pr1_premove_arrived","preset":"RR1"}\n')
        stream.write('{"event":"frame","state":"regrasp_reacquiring","reason":"xy_misaligned_near_contact","regrasp_attempt":1}\n')
        stream.write('{"event":"left_visual_regrasp_requested","regrasp_attempt":2}\n')
        stream.write('operator text\n')
        self.assertEqual([event['event'] for event in events], [
            'left_above_hinge_arrived', 'left_stage_complete_above_hole',
            'right_pr1_premove_arrived', 'left_visual_regrasp_requested',
            'left_visual_regrasp_requested'])
        self.assertEqual(events[-1]['regrasp_attempt'], 2)
        self.assertEqual(events[-2]['regrasp_reason'], 'xy_misaligned_near_contact')
        self.assertEqual(output.getvalue(), 'operator text\n')
    def test_features_are_explicitly_disabled_by_default(self):
        args = flow.parse_args([])
        self.assertFalse(args.vlm_enable)
        self.assertFalse(args.teach_enable)

    def test_repeat_demo_does_not_enable_taught_placement(self):
        args = flow.parse_args(['--repeat-grasp-demo', '--teach-enable', 'true'])
        self.assertFalse(flow.teach_enabled(args))

    def test_feature_switches_accept_explicit_boolean_values(self):
        args = flow.parse_args(["--vlm-enable", "true", "--teach-enable", "false"])
        self.assertTrue(args.vlm_enable)
        self.assertFalse(args.teach_enable)

        args = flow.parse_args(["--vlm-enable", "false", "--teach-enable", "true"])
        self.assertFalse(args.vlm_enable)
        self.assertTrue(args.teach_enable)

    def test_vlm_mode_and_tts_switch_are_selectable(self):
        try:
            args = flow.parse_args([
                "--vlm-mode", "online", "--vlm-enable", "true", "--tts-enable", "true",
            ])
        except SystemExit as exc:
            self.fail(f"VLM mode and TTS switches should parse: {exc}")
        self.assertEqual(args.vlm_mode, "online")
        self.assertTrue(args.tts_enable)

        args = flow.parse_args(["--vlm-mode", "local", "--tts-enable", "false"])
        self.assertEqual(args.vlm_mode, "local")
        self.assertFalse(args.tts_enable)

    def test_vlm_mode_and_tts_default_to_local_and_disabled(self):
        args = flow.parse_args([])
        self.assertEqual(args.vlm_mode, "local")
        self.assertFalse(args.tts_enable)

    def test_gdk_is_not_released_while_tts_worker_is_using_it(self):
        class Announcer:
            is_alive = True

        class StoppedAnnouncer:
            is_alive = False

        should_release = getattr(flow, "should_release_gdk", None)
        self.assertIsNotNone(should_release, "GDK release policy for TTS shutdown is missing")
        self.assertFalse(should_release(Announcer()))
        self.assertTrue(should_release(StoppedAnnouncer()))
        self.assertTrue(should_release(None))

    def test_teach_enable_requires_right_files(self):
        args = flow.parse_args(["--device", "/dev/test", "--teach-enable"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(flow.validate_args(args), 2)

    def test_four_right_trajectory_arguments_preserve_file_order(self):
        files = ["right1.json", "right2.json", "right3.json", "right4.json"]
        args = flow.parse_args(["--teach-enable", "--right-teach-trajectories", *files])
        self.assertEqual(args.right_teach_trajectories, [Path(path) for path in files])
        self.assertIsNone(args.right_teach_trajectory)

    def test_teach_placement_preserves_original_hole_transport(self):
        args = flow.parse_args(["--teach-enable", "--left-teach-trajectory", "demo.json", "--left-teach-speed-scale", "0.5"])
        argv = flow.StageSettings(args).left_argv()
        original = flow.StageSettings(flow.parse_args([])).left_argv()
        self.assertEqual(argv, original)
        self.assertIn("--move-above-hole", argv)
        self.assertEqual(args.left_teach_speed_scale, .5)

    def test_original_placement_is_default(self):
        args = flow.parse_args([])
        self.assertFalse(args.teach_enable)
        self.assertNotIn("--stop-before-hole-transfer", flow.StageSettings(args).left_argv())

    def test_teach_requires_trajectory_before_robot_start(self):
        args = flow.parse_args(["--device", "/dev/test", "--teach-enable"])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(flow.validate_args(args), 2)

    def test_teach_stage_does_not_execute_ll1_or_force_descent(self):
        from types import SimpleNamespace
        ctx = SimpleNamespace(args=flow.parse_args(["--teach-enable", "--left-teach-trajectory", "demo.json"]), services=StageServices())
        with patch("left_teach_placement.replay_left_placement", return_value={"method": "teach", "sent": 5}) as replay:
            self.assertEqual(flow.stage_left_to_ll1_and_seat(ctx), {"method": "teach", "sent": 5})
        replay.assert_called_once()

    def test_stage_console_filter_hides_json_but_keeps_operator_text(self):
        output = io.StringIO()
        stream = flow.JsonLineSuppressingStream(output)
        stream.write('{"event":"frame","details":[1,2,3]}\n')
        stream.write("左手已到 hole 上方；本阶段完成。\n")
        stream.flush()
        self.assertEqual(output.getvalue(), "左手已到 hole 上方；本阶段完成。\n")

    def test_list_stages_prints_the_plan_without_a_device(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.assertEqual(flow.main(["--list-stages"]), 0)
        text = buffer.getvalue()
        for spec in DEFAULT_PLAN:
            self.assertIn(spec.stage.value, text)
        self.assertIn("[SpaceMouse 门]", text)

    def test_running_without_allow_motion_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(flow.main(["--device", "/dev/hidraw-test"]), 2)

    def test_running_without_a_device_is_refused(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(flow.main(["--allow-motion"]), 2)

    def test_missing_gdk_is_reported_instead_of_crashing(self):
        import sys
        with patch.dict(sys.modules, {"agibot_gdk": None}):
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(flow.main(["--device", "/dev/hidraw-test", "--allow-motion"]), 2)

    def test_defaults_match_the_operator_spec(self):
        args = flow.parse_args([])
        self.assertEqual(args.approach_yaw_deg, 0.0)
        self.assertEqual(args.approach_pitch_deg, 42.0)
        self.assertEqual(args.approach_roll_deg, 180.0)
        self.assertEqual(args.left_hole_yaw_deg, 20.0)
        self.assertEqual(args.left_hole_pitch_deg, 42.0)
        self.assertEqual(args.left_hole_roll_deg, 180.0)
        self.assertEqual(args.left_hole_x_mm, 0.0)
        self.assertEqual(args.left_hole_y_mm, 0.0)
        self.assertEqual(args.left_hole_z_mm, 100.0)
        self.assertEqual(args.right_hole_yaw_deg, 20.0)
        self.assertEqual(args.right_hole_pitch_deg, 0.0)
        self.assertEqual(args.right_hole_roll_deg, 180.0)
        self.assertEqual(args.right_hole_x_mm, 0.0)
        self.assertEqual(args.right_hole_y_mm, 20.0)
        self.assertEqual(args.right_hole_z_mm, 200.0)
        self.assertEqual(args.right_grasp_distance_mm, 152.0)
        self.assertEqual(args.approach_above_mm, 100.0)
        self.assertEqual(args.right_premove_pose, "RR1")
        self.assertEqual(args.right_post_grasp_pose, "RR2")
        self.assertEqual(args.from_stage, None)
        self.assertEqual(args.through_stage, None)
        self.assertEqual(args.left_seat_max_descent_mm, 10.0)
        self.assertEqual(args.left_seat_force_threshold_n, 3.0)
        self.assertEqual(set(stage_names()), set(stage_names()))


class DryRunTests(unittest.TestCase):
    def test_teach_dry_run_checks_trajectory_and_does_not_need_ll1(self):
        from teach_trajectory import build_trajectory, trajectory_document
        with tempfile.TemporaryDirectory() as directory:
            poses = write_poses(Path(directory) / "poses.json", complete=True)
            document = json.loads(poses.read_text())
            del document["poses"]["LL1"]
            poses.write_text(json.dumps(document))
            trajectory = Path(directory) / "left.json"
            trajectory.write_text(json.dumps(trajectory_document(build_trajectory("left", [(0, [0]*7)]))))
            right_trajectories = []
            for index in range(4):
                right = Path(directory) / f"right{index}.json"
                right.write_text(json.dumps(trajectory_document(build_trajectory("right", [(0, [0]*7)]))))
                right_trajectories.append(str(right))
            with contextlib.redirect_stdout(io.StringIO()):
                code = flow.main([
                    "--dry-run", "--poses", str(poses), "--teach-enable",
                    "--left-teach-trajectory", str(trajectory),
                    "--right-teach-trajectories", *right_trajectories,
                ])
            self.assertEqual(code, 0)
            with contextlib.redirect_stderr(io.StringIO()):
                code = flow.main(["--dry-run", "--poses", str(poses), "--teach-enable"])
            self.assertEqual(code, 2)

    def test_missing_staged_presets_are_reported_with_exit_1(self):
        with tempfile.TemporaryDirectory() as directory:
            poses = write_poses(Path(directory) / "poses.json", complete=False)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = flow.main(["--dry-run", "--poses", str(poses)])
        self.assertEqual(code, 1)
        text = buffer.getvalue()
        self.assertIn("missing_presets", text)
        for name in ("LR1", "LR2", "RR1", "LL1", "LL2"):
            self.assertIn(name, text)

    def test_complete_preset_file_passes_the_dry_run(self):
        with tempfile.TemporaryDirectory() as directory:
            poses = write_poses(Path(directory) / "poses.json", complete=True)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = flow.main(["--dry-run", "--poses", str(poses)])
        self.assertEqual(code, 0)
        self.assertIn('"missing_presets": []', buffer.getvalue().replace("\n", " ").replace("  ", " "))


class StageArgvTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = flow.StageSettings(args=flow.parse_args([]))

    def test_left_argv_defers_the_hole_move_and_skips_the_duplicate_home(self):
        argv = self.settings.left_argv()
        self.assertIn("--move-above-hole", argv)
        self.assertIn("--defer-hole-move", argv)
        self.assertIn("--skip-workflow-home", argv)
        self.assertIn("--full-cycle", argv)
        self.assertIn("--head-coarse-approach", argv)
        self.assertIn("--allow-motion", argv)
        self.assertEqual(argv[argv.index("--hole-forward-yaw-deg") + 1], "20.0")
        self.assertEqual(argv[argv.index("--hole-x-mm") + 1], "0.0")
        self.assertEqual(argv[argv.index("--hole-y-mm") + 1], "0.0")
        self.assertEqual(argv[argv.index("--hole-z-mm") + 1], "100.0")
        self.assertEqual(argv[argv.index("--head-forward-yaw-deg") + 1], "0.0")
        self.assertEqual(argv[argv.index("--web-port") + 1], "0")
        # SAM3 is gone from the staged flow.
        for retired in ("--checkpoint", "--prompt", "--sam-review-interval-s"):
            self.assertNotIn(retired, argv)

    def test_left_argv_defaults_to_the_one_class_hinge_weight(self):
        # The left grasp only consumes the hinge class; run_bilateral_alignment
        # keeps the two-class weight because its direct mode needs the shaft.
        argv = self.settings.left_argv()
        model = argv[argv.index("--yolo-model") + 1]
        self.assertTrue(model.endswith("models/left_d405_hinge_yolo/best.pt"), model)

    def test_left_argv_keeps_the_tuned_grasp_behaviour_of_the_old_command(self):
        # These four flags were part of the proven bilateral command; dropping
        # them makes the final descent miss the live XY visual servo.
        argv = self.settings.left_argv()
        self.assertIn("--final-descent-visual-servo", argv)
        self.assertEqual(argv[argv.index("--final-descent-visual-tolerance-mm") + 1], "2.0")
        self.assertEqual(argv[argv.index("--yolo-center-fraction") + 1], "0.7")
        self.assertEqual(argv[argv.index("--post-grasp-target-lost-timeout-s") + 1], "1.0")

    def test_left_argv_is_accepted_by_the_left_stage_parser(self):
        # The staged flow injects the left stage in-process, so a flag the left
        # module does not define only explodes on the robot.  Parse the exact
        # argv the flow builds and check the empty-grasp detector defaults.
        argv = self.settings.left_argv()
        parsed = left_grasp_gdk.parse_args(argv)
        self.assertEqual(parsed.hole_forward_yaw_deg, 20.0)
        self.assertEqual(parsed.post_grasp_lift_area_grace_s, 1.0)
        self.assertEqual(parsed.post_grasp_lift_area_window_frames, 5)
        self.assertEqual(parsed.post_grasp_lift_area_spread, 0.20)
        self.assertEqual(parsed.post_grasp_lift_area_confirmations, 2)
        self.assertEqual(parsed.max_travel_mm, 450.0)

    def test_right_argv_uses_the_staged_presets_and_stops_after_transport(self):
        argv = self.settings.right_argv()
        self.assertEqual(argv[argv.index("--premove-pose") + 1], "RR1")
        self.assertNotIn("--post-grasp-pose", argv)
        self.assertIn("--stop-after-transport", argv)
        self.assertIn("--premove-pr1", argv)
        self.assertNotIn("--post-grasp-pr2", argv)
        self.assertIn("--autonomous-approach", argv)
        self.assertEqual(argv[argv.index("--web-port") + 1], "0")
        self.assertEqual(argv[argv.index("--grasp-distance-mm") + 1], "152.0")

    def test_right_grasp_distance_override_is_forwarded(self):
        settings = flow.StageSettings(args=flow.parse_args(["--right-grasp-distance-mm", "165"]))
        argv = settings.right_argv()
        self.assertEqual(argv[argv.index("--grasp-distance-mm") + 1], "165.0")


class HeadTargetGateTests(unittest.TestCase):
    """Only the hinge is mandatory; a missing hole is a logged downgrade."""

    @staticmethod
    def localization(*, hinge=True, hole=True, hole_samples=12, hole_reason="头部 hole 没有可用三维样本"):
        targets = {}
        if hinge:
            targets["hinge"] = {"position_base_link_m": [0.4, 0.1, 0.8], "samples": 12}
        if hole:
            targets["hole"] = {"position_base_link_m": [0.5, 0.15, 0.79], "samples": hole_samples}
        rejected = {}
        if not hole:
            rejected["hole"] = {"samples": hole_samples, "reason": hole_reason}
        return {"frames": 12, "targets": targets, "rejected_targets": rejected}

    def test_both_targets_present_are_returned(self):
        targets = flow.require_head_targets(self.localization())
        self.assertEqual(set(targets), {"hinge", "hole"})

    def test_missing_hole_is_a_downgrade_not_a_stop(self):
        targets = flow.require_head_targets(self.localization(hole=False, hole_samples=0))
        self.assertEqual(set(targets), {"hinge"})
        self.assertIsNone(targets.get("hole"))

    def test_missing_hole_produces_a_warning_record(self):
        warning = flow.hole_missing_warning(self.localization(hole=False, hole_samples=0))
        self.assertEqual(warning["event"], "head_hole_missing")
        self.assertEqual(warning["level"], "warning")
        self.assertIn("等待", warning["action"])
        self.assertIsNone(flow.hole_missing_warning(self.localization()))

    def test_scattered_hole_is_reported_in_the_warning_reason(self):
        warning = flow.hole_missing_warning(
            self.localization(hole=False, hole_samples=15, hole_reason="头部 hole 样本没有足够一致的主簇"),
        )
        self.assertIn("主簇", warning["reason"])

    def test_missing_hinge_is_still_refused(self):
        from safety import SafetyError

        with self.assertRaises(SafetyError):
            flow.require_head_targets(self.localization(hinge=False))

    def test_raw_preview_defaults_are_on(self):
        args = flow.parse_args([])
        self.assertGreater(args.raw_preview_fps, 0.0)
        self.assertGreater(args.raw_preview_hold_s, 0.0)


class DispatchTests(unittest.TestCase):
    def test_vlm_failure_during_placement_restarts_left_grasp_without_relocalizing_head_or_opening_gripper(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from vlm_gripper_monitor import GripperMonitor, VlmRegraspRequested
        context, runtime = self.make_context()
        context.services.put('head_targets', {'hinge': {'position': [1,2,3]}})
        monitor = GripperMonitor(None, None, lambda record: None)
        context.services.put('vlm_gripper_monitor', monitor)
        sent = []
        context.services.robot = SimpleNamespace(move_arm_joint_servo=lambda *args: sent.append(args))
        prepared = []
        context.services.controller = SimpleNamespace(
            checked_arm_state=lambda: (0.,)*14,
            _motion_write=nullcontext,
            prepare_joint_impedance_control=lambda sides: prepared.append(sides),
            hold_joint_positions=lambda side, positions: (
                self.assertEqual(prepared, [("left",)]),
                context.services.robot.move_arm_joint_servo(positions, .01, 0, False),
            ),
        )
        visits = []
        failed = [False]
        def handler(stage):
            def run(_ctx):
                visits.append(stage)
                if stage is Stage.LEFT_TO_LL1_AND_SEAT and not failed[0]:
                    failed[0] = True
                    raise VlmRegraspRequested('dropped')
                return {}
            return run
        with patch.dict(flow.STAGE_HANDLERS, {stage: handler(stage) for stage in flow.STAGE_HANDLERS}), patch('run_bilateral_flow_gdk.time.sleep'), patch('left_seat_motion.move_left_vertical', return_value={'event': 'left_vertical_lift_complete'}) as lift:
            self.assertEqual(flow._drive(context, context.args), 0)
        lift.assert_called_once_with(context.services.controller, distance_mm=50.0, speed_mm_s=context.args.auto_speed_mm_s)
        self.assertEqual(visits.count(Stage.LEFT_GRASP), 2)
        self.assertEqual(visits.count(Stage.HEAD_LOCALIZE), 1)
        self.assertEqual(len(sent), 20)
        self.assertEqual(prepared, [("left",)])
        self.assertTrue(all(args == ([0.]*7, .01, 0, False) for args in sent))
        self.assertEqual(runtime.left_gripper_opens, 0)

    def test_vlm_regrasp_lift_failure_stops_before_restarting_pickup(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from vlm_gripper_monitor import GripperMonitor, VlmRegraspRequested
        context, runtime = self.make_context()
        context.services.put('head_targets', {'hinge': {'position': [1, 2, 3]}})
        context.services.put('vlm_gripper_monitor', GripperMonitor(None, None, lambda record: None))
        context.services.robot = SimpleNamespace(move_arm_joint_servo=lambda *args: None)
        context.services.controller = SimpleNamespace(
            checked_arm_state=lambda: (0.,)*14,
            _motion_write=nullcontext,
            prepare_joint_impedance_control=lambda sides: None,
            hold_joint_positions=lambda side, positions: context.services.robot.move_arm_joint_servo(positions, .01, 0, False),
        )
        visits = []
        def handler(stage):
            def run(_ctx):
                visits.append(stage)
                if stage is Stage.LEFT_TO_LL1_AND_SEAT:
                    raise VlmRegraspRequested('dropped')
                return {}
            return run
        with patch.dict(flow.STAGE_HANDLERS, {stage: handler(stage) for stage in flow.STAGE_HANDLERS}), patch('run_bilateral_flow_gdk.time.sleep'), patch('left_seat_motion.move_left_vertical', side_effect=RuntimeError('lift not reached')):
            self.assertEqual(flow._drive(context, context.args), 1)
        self.assertEqual(visits.count(Stage.LEFT_GRASP), 1)
        self.assertEqual(runtime.left_gripper_opens, 0)

    def test_left_motion_stages_keep_live_feed_in_both_placement_modes(self):
        from bilateral_web import BilateralWeb, RawPaneFeed
        for teach_enabled in (False, True):
            with self.subTest(teach_enabled=teach_enabled):
                context, _runtime = self.make_context()
                context.args.teach_enable = teach_enabled
                web = BilateralWeb(host="127.0.0.1", port=0)
                feed = RawPaneFeed("left", lambda: b"current-left-frame", web, hold_s=0)
                context.feeds["left"] = feed
                checked = []
                def handler(stage):
                    def run(_ctx):
                        if stage is Stage.LEFT_GRASP:
                            self.assertEqual(feed.step(), "published")
                        if stage in (Stage.LEFT_TO_LL1_AND_SEAT, Stage.LEFT_RELEASE_LIFT_TO_LL2):
                            self.assertEqual(feed.step(), "published")
                            self.assertEqual(web.snapshot_pane("left")[0], b"current-left-frame")
                            checked.append(stage)
                        return {}
                    return run
                with patch.dict(flow.STAGE_HANDLERS, {stage: handler(stage) for stage in flow.STAGE_HANDLERS}):
                    self.assertEqual(flow._drive(context, context.args), 0)
                self.assertEqual(checked, [Stage.LEFT_TO_LL1_AND_SEAT, Stage.LEFT_RELEASE_LIFT_TO_LL2])

    def make_context(self):
        class FakeRuntime:
            def __init__(self) -> None:
                self.events: list[tuple] = []
                self.advances: list[str] = []
                self.left_gripper_opens = 0
                self.left_presets: list[tuple[str, ...]] = []
                self.right_presets: list[str] = []
                self.right_preset_loads: list[str] = []
                self.sleep_calls: list[float] = []

            def emit(self, event, **fields):
                self.events.append((event, fields))
                return {"event": event, **fields}

            def emit_event(self, record):
                self.events.append((record["event"], record))
                return record

            def wait_for_advance(self, spec):
                self.advances.append(spec.stage.value)

            def open_left_gripper(self, **_kwargs):
                self.left_gripper_opens += 1
                return {
                    "event": "left_gripper_opened",
                    "actual_position_rad": {"left": -0.785},
                    "command_attempts": {"left": 2},
                }

            def move_left_presets(self, names, **_kwargs):
                self.left_presets.append(tuple(names))
                return [{"preset": name} for name in names]

            def move_right_preset(self, name, **_kwargs):
                self.right_presets.append(str(name))
                return {"event": "right_preset_arrived", "preset": str(name)}

            def load_right_preset(self, name):
                self.right_preset_loads.append(str(name))
                return [0.0] * 7

            def sleep(self, seconds):
                self.sleep_calls.append(float(seconds))

            def open_all_grippers(self, **_kwargs):
                return {"event": "all_grippers_opened"}

            def move_both_to_home(self, **_kwargs):
                return {"event": "workflow_home_arrived"}

        runtime = FakeRuntime()
        context = flow.FlowContext(
            gdk=None, services=StageServices(), runtime=runtime,
            settings=flow.StageSettings(args=flow.parse_args([])),
        )
        return context, runtime

    def test_ll1_force_descent_then_release_lift_and_ll2(self):
        context, runtime = self.make_context()
        context.services.robot = object()
        with patch("left_seat_motion.force_guarded_left_descent", return_value={
            "event": "left_seat_descent_complete", "stop_reason": "force_threshold",
        }) as descent, patch("left_seat_motion.move_left_vertical", return_value={
            "event": "left_release_lift_complete", "distance_mm": 100.0,
        }) as lift:
            seated = flow.stage_left_to_ll1_and_seat(context)
            released = flow.stage_left_release_lift_to_ll2(context)
        self.assertEqual(runtime.left_presets, [("LL1",), ("LL2",)])
        self.assertEqual(seated["descent"]["stop_reason"], "force_threshold")
        self.assertEqual(released["lift"]["distance_mm"], 100.0)
        self.assertEqual(runtime.left_gripper_opens, 1)
        self.assertEqual(descent.call_args.kwargs["max_distance_mm"], 10.0)
        self.assertEqual(descent.call_args.kwargs["force_threshold_n"], 3.0)
        self.assertEqual(lift.call_args.kwargs["distance_mm"], 100.0)

    def test_pre_release_vlm_warning_is_read_only(self):
        context, runtime = self.make_context()
        context.services.robot = object()
        with patch("run_bilateral_flow_gdk._one_shot_gripper_check", return_value="dropped"), \
             patch("left_seat_motion.move_left_vertical", return_value={"event": "lift"}):
            released = flow.stage_left_release_lift_to_ll2(context)
        # The warning must not block the release or request a regrasp.
        self.assertEqual(runtime.left_gripper_opens, 1)
        self.assertEqual(released["pre_release_vlm"], "dropped")
        warnings = [record for name, record in runtime.events if name == "vlm_pre_release_warning"]
        self.assertEqual(len(warnings), 1)
        self.assertTrue(warnings[0]["read_only"])
        self.assertEqual(warnings[0]["action"], "continue_release")

    def test_right_lr2_cycle_closes_holds_one_second_then_opens(self):
        context, runtime = self.make_context()
        context.services.put("right_config", {
            "gripper": {"closed_rad": 0.0, "open_rad": -0.785, "close_settle_s": 0.8},
        })
        context.services.controller = Mock()
        stream = Mock()
        context.services.controller.start_right_end_effector_position_stream = lambda *a, **k: stream
        target_pose = object()
        with patch("right_hole_motion.right_preset_end_pose", return_value=target_pose) as pose, \
             patch("right_hole_motion.move_right_via_vertical_clearance", return_value={
            "event": "right_lr2_clearance_round_trip", "clearance_mm": 50.0,
        }) as clearance, \
             patch("right_apriltag_grasp_gdk.close_right_gripper_on_pose_stream", return_value={
                 "target_rad": 0.0,
                 "actual_position_rad": -0.16667076388888893,
                 "close_state": "blocked_before_target",
             }) as close, \
             patch("right_apriltag_grasp_gdk.move_right_gripper_on_pose_stream", return_value={
                 "target_rad": -0.785,
             }) as open:
            result = flow.stage_right_to_lr2_and_gripper_cycle(context)

        self.assertEqual(runtime.right_presets, [])
        self.assertEqual(runtime.right_preset_loads, ["LR2"])
        pose.assert_called_once_with(context.services.controller, [0.0] * 7)
        clearance.assert_called_once_with(
            context.services.controller,
            target_pose=target_pose,
            clearance_mm=50.0,
            speed_mm_s=context.args.auto_speed_mm_s,
        )
        close.assert_called_once_with(
            context.services.controller, stream, 0.0, settle_s=0.8,
        )
        open.assert_called_once_with(
            context.services.controller, stream, -0.785,
            tolerance_rad=0.05, timeout_s=3.0,
        )
        self.assertEqual(runtime.sleep_calls, [1.0])
        self.assertEqual(result["preset"], "LR2")
        self.assertEqual(result["gripper_cycle"], "closed_held_opened")

    def test_finish_does_not_check_after_right_return(self):
        context, runtime = self.make_context()
        with patch("run_bilateral_flow_gdk._one_shot_assembly_check") as check:
            result = flow.stage_finish(context)
        self.assertEqual(result["grippers"], "all_grippers_opened")
        self.assertNotIn("final_vlm", result)
        check.assert_not_called()

    def test_screw_check_moves_to_configured_hole_after_lift_before_images(self):
        context, _ = self.make_context()
        context.args.teach_enable = True
        context.services.controller = Mock()
        context.services.put('right_saved_grasp_pose', 'saved')
        context.services.put('right_saved_workpiece_pose', 'workpiece')
        context.services.put('head_targets', {'hole': {'position_base_link_m': [1, 2, 3]}})
        context.services.put('right_config', {})
        context.args.right_hole_x_mm = 17
        calls = []
        with patch('right_hole_motion.lift_right_before_return', side_effect=lambda *a, **k: calls.append('lift') or {'event': 'lift'}) as lift, \
             patch('right_hole_motion.move_right_tool_above_hole', side_effect=lambda *a, **k: calls.append('hole') or {'event': 'hole'}) as move, \
             patch('run_bilateral_flow_gdk._one_shot_four_screws_check', side_effect=lambda *a, **k: calls.append('images') or {}), \
             patch('right_apriltag_grasp_gdk.return_screwdriver', side_effect=lambda *a, **k: calls.append('return')):
            flow.stage_right_return(context)
        self.assertEqual(calls, ['lift', 'hole', 'images', 'return'])
        self.assertEqual(lift.call_args.kwargs['distance_mm'], 30.)
        self.assertFalse(move.call_args.kwargs['lift_before_move'])
        for axis in ('x', 'y', 'z'):
            self.assertEqual(move.call_args.kwargs[axis + '_mm'], getattr(context.args, 'right_hole_' + axis + '_mm'))
        for axis in ('yaw', 'pitch', 'roll'):
            self.assertEqual(move.call_args.kwargs[axis + '_deg'], getattr(context.args, 'right_hole_' + axis + '_deg'))

    def test_handler_table_covers_every_non_gate_stage(self):
        expected = {spec.stage for spec in DEFAULT_PLAN if not spec.waits_for_advance}
        self.assertEqual(set(flow.STAGE_HANDLERS), expected)

    def test_interrupted_grasp_stage_is_never_treated_as_success(self):
        from safety import SafetyError

        context, _runtime = self.make_context()
        context.services.put("head_targets", {
            "hinge": {"position_base_link_m": [0.4, 0.1, 0.8]},
        })
        with patch.object(
            left_grasp_gdk, "run_left_cycle",
            return_value=left_grasp_gdk.INTERRUPTED_EXIT_CODE,
        ), self.assertRaises(SafetyError):
            flow.stage_left_grasp(context)
        with patch.object(
            right_apriltag_grasp_gdk, "run_right_grasp_stage",
            return_value=right_apriltag_grasp_gdk.INTERRUPTED_EXIT_CODE,
        ), self.assertRaises(SafetyError):
            flow.stage_right_grasp(context)

    def test_grasp_and_hole_transport_are_separate_stages(self):
        context, runtime = self.make_context()
        context.services.controller = object()
        context.services.put('head_targets', {
            'hinge': {'position_base_link_m': [0.4, 0.1, 0.8]},
            'hole': {'position_base_link_m': [0.5, 0.2, 0.8]},
        })
        with tempfile.TemporaryDirectory() as directory:
            context.args.log = Path(directory) / 'left.jsonl'
            with patch.object(left_grasp_gdk, 'run_left_cycle', return_value=0), \
                 patch.object(left_grasp_gdk, 'move_left_above_hole', return_value={'event': 'left_above_hole_arrived'}) as move:
                flow.stage_left_grasp(context)
                move.assert_not_called()
                self.assertFalse(any(event == 'left_stage_complete_above_hole' for event, _ in runtime.events))
                flow.stage_left_to_hole_offset(context)
        move.assert_called_once()
        self.assertEqual(move.call_args.kwargs['hole_base_m'], [0.5, 0.2, 0.8])
        self.assertEqual(move.call_args.kwargs['args'].hole_y_mm, context.args.left_hole_y_mm)
        self.assertTrue(any(event == 'left_stage_complete_above_hole' for event, _ in runtime.events))

    def test_right_grasp_transports_tool_axis_to_head_hole_instead_of_rr2(self):
        from control import CartesianPoseTarget
        import numpy as np

        context, _runtime = self.make_context()
        saved = CartesianPoseTarget((0.4, -0.2, 0.8), (0.0, 0.0, 0.0, 1.0))
        context.services.controller = object()
        context.services.put("head_targets", {
            "hole": {"position_base_link_m": [0.5, 0.15, 0.79]},
        })

        def successful_grasp(*, services, argv):
            services.put("right_saved_grasp_pose", saved)
            services.put("right_saved_workpiece_pose", np.eye(4))
            return 0

        with patch.object(right_apriltag_grasp_gdk, "run_right_grasp_stage", side_effect=successful_grasp), patch(
            "right_hole_motion.move_right_tool_above_hole",
            return_value={"event": "right_tool_above_hole_arrived"},
        ):
            result = flow.stage_right_grasp(context)
        self.assertEqual(result["transport"], "right_tool_above_hole_arrived")

    def test_right_teach_runs_only_after_hole_transport_arrives(self):
        from bilateral_web import BilateralWeb, RawPaneFeed
        import numpy as np
        context, _runtime = self.make_context()
        context.args.teach_enable = True
        context.args.right_teach_trajectory = Path("right_demo.json")
        context.services.controller = object()
        context.services.put("head_targets", {"hole": {"position_base_link_m": [0.5, .15, .79]}})
        web = BilateralWeb(host="127.0.0.1", port=0)
        feed = RawPaneFeed("right", lambda: b"live-right-frame", web, hold_s=0)
        feed.pause()
        context.feeds["right"] = feed
        order = []
        def grasp(*, services, argv):
            self.assertEqual(feed.step(), "paused")
            order.append("grasp")
            services.put("right_saved_grasp_pose", object())
            services.put("right_saved_workpiece_pose", np.eye(4))
            return 0
        def transport(*args, **kwargs):
            self.assertEqual(feed.step(), "published")
            self.assertEqual(web.snapshot_pane("right")[0], b"live-right-frame")
            order.append("hole_arrived")
            return {"event": "right_tool_above_hole_arrived"}
        def replay(*args):
            self.assertEqual(feed.step(), "published")
            order.append("teach")
            return {"method": "teach", "sent": 3}
        with patch.object(right_apriltag_grasp_gdk, "run_right_grasp_stage", side_effect=grasp), patch("right_hole_motion.move_right_tool_above_hole", side_effect=transport), patch("right_teach_placement.replay_right_placement", side_effect=replay):
            result = flow.stage_right_grasp(context)
        self.assertEqual(order, ["grasp", "hole_arrived", "teach"])
        self.assertEqual(result["placement"]["sent"], 3)

    def test_drive_walks_every_stage_and_hits_both_gates_in_order(self):
        context, runtime = self.make_context()
        handled: list[str] = []

        def make_handler(stage):
            def handler(_ctx):
                handled.append(stage.value)
                return {"stage": stage.value}
            return handler

        table = {stage: make_handler(stage) for stage in flow.STAGE_HANDLERS}
        with patch.dict(flow.STAGE_HANDLERS, table, clear=True):
            code = flow._drive(context, flow.parse_args([]))
        self.assertEqual(code, 0)
        self.assertEqual(handled, [s.stage.value for s in DEFAULT_PLAN if not s.waits_for_advance])
        self.assertEqual(runtime.advances, ["gate_left", "gate_right_pre_grasp", "gate_right"])
        completed = [fields["stage"] for event, fields in runtime.events if event == "stage_complete"]
        self.assertEqual(completed, [s.stage.value for s in DEFAULT_PLAN])
        self.assertIn("bilateral_flow_complete", [event for event, _ in runtime.events])

    def test_the_gate_sequence_is_identical_with_or_without_a_hole(self):
        # A missing hole only removes the hole move; all SpaceMouse gates and
        # everything else stay exactly the same.
        for hole_detected in (True, False):
            with self.subTest(hole_detected=hole_detected):
                context, runtime = self.make_context()

                def make_handler(stage, found=hole_detected):
                    def handler(_ctx):
                        if stage is Stage.LEFT_GRASP:
                            return {"hole_detected": found}
                        return {"stage": stage.value}
                    return handler

                table = {stage: make_handler(stage) for stage in flow.STAGE_HANDLERS}
                with patch.dict(flow.STAGE_HANDLERS, table, clear=True):
                    code = flow._drive(context, flow.parse_args([]))
                self.assertEqual(code, 0)
                self.assertEqual(runtime.advances, ["gate_left", "gate_right_pre_grasp", "gate_right"])
                self.assertNotIn("gate_skipped", [event for event, _ in runtime.events])
                # Gate 1 only confirms; the immediately following stage owns release.
                self.assertEqual(runtime.left_gripper_opens, 0)
                completed = {
                    fields["stage"]: fields
                    for event, fields in runtime.events if event == "stage_complete"
                }
                self.assertNotIn("left_gripper_position_rad", completed["gate_left"])

    def test_a_failing_stage_stops_the_sequence(self):
        context, runtime = self.make_context()

        def boom(_ctx):
            raise RuntimeError("hole 目标缺失")

        table = {stage: (lambda _ctx: {}) for stage in flow.STAGE_HANDLERS}
        table[flow.Stage.HEAD_LOCALIZE] = boom
        with patch.dict(flow.STAGE_HANDLERS, table, clear=True):
            with contextlib.redirect_stderr(io.StringIO()):
                code = flow._drive(context, flow.parse_args([]))
        self.assertEqual(code, 1)
        failures = [fields for event, fields in runtime.events if event == "stage_failed"]
        self.assertEqual(failures[0]["stage"], "head_localize")
        self.assertIn("hole", failures[0]["error"])
        self.assertNotIn("bilateral_flow_complete", [event for event, _ in runtime.events])

    def test_from_stage_starts_the_walk_later(self):
        context, runtime = self.make_context()
        prerequisites = []
        handlers = {stage: (lambda _ctx: {}) for stage in flow.STAGE_HANDLERS}
        handlers[flow.Stage.HEAD_LOCALIZE] = lambda _ctx: prerequisites.append("head_localize") or {}
        with patch.dict(
            flow.STAGE_HANDLERS, handlers, clear=True,
        ):
            code = flow._drive(context, flow.parse_args(["--from-stage", "gate_right"]))
        self.assertEqual(code, 0)
        self.assertEqual(prerequisites, ["head_localize"])
        completed = [fields["stage"] for event, fields in runtime.events if event == "stage_complete"]
        self.assertEqual(completed, ["gate_right", "right_return", "finish"])

    def test_starting_at_right_grasp_localizes_head_before_any_right_motion(self):
        context, _runtime = self.make_context()
        order = []
        handlers = {stage: (lambda _ctx: {}) for stage in flow.STAGE_HANDLERS}
        handlers[flow.Stage.HEAD_LOCALIZE] = lambda _ctx: order.append("head_localize") or {}
        handlers[flow.Stage.RIGHT_GRASP] = lambda _ctx: order.append("right_grasp") or {}
        with patch.dict(flow.STAGE_HANDLERS, handlers, clear=True):
            code = flow._drive(context, flow.parse_args([
                "--from-stage", "right_grasp", "--through-stage", "right_grasp",
            ]))
        self.assertEqual(code, 0)
        self.assertEqual(order, ["head_localize", "right_grasp"])

    def test_through_stage_stops_after_including_the_requested_stage(self):
        context, runtime = self.make_context()
        with patch.dict(
            flow.STAGE_HANDLERS, {stage: (lambda _ctx: {}) for stage in flow.STAGE_HANDLERS}, clear=True,
        ):
            code = flow._drive(context, flow.parse_args(["--through-stage", "right_to_lr1"]))
        self.assertEqual(code, 0)
        completed = [fields["stage"] for event, fields in runtime.events if event == "stage_complete"]
        self.assertEqual(completed, ["home", "head_localize", "left_grasp", "left_to_hole_offset", "vlm_gripper_check", "right_to_lr1"])
        events = [event for event, _fields in runtime.events]
        self.assertIn("bilateral_flow_stopped_at_stage", events)
        self.assertNotIn("bilateral_flow_complete", events)

    def test_from_and_through_can_select_one_stage(self):
        context, runtime = self.make_context()
        with patch.dict(
            flow.STAGE_HANDLERS, {stage: (lambda _ctx: {}) for stage in flow.STAGE_HANDLERS}, clear=True,
        ):
            code = flow._drive(context, flow.parse_args([
                "--from-stage", "left_to_ll1_and_seat",
                "--through-stage", "left_to_ll1_and_seat",
            ]))
        self.assertEqual(code, 0)
        completed = [fields["stage"] for event, fields in runtime.events if event == "stage_complete"]
        self.assertEqual(completed, ["left_to_ll1_and_seat"])

    def test_through_stage_before_from_stage_is_rejected(self):
        args = flow.parse_args([
            "--device", "/dev/hidraw-test",
            "--from-stage", "gate_right",
            "--through-stage", "left_grasp",
        ])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(flow.validate_args(args), 2)


class CameraWarmupTests(unittest.TestCase):
    """A pane must not stay blank because its camera was not ready yet."""

    def test_warmup_succeeds_on_the_first_available_frame(self):
        import numpy as np

        class Camera:
            def __init__(self) -> None:
                self.calls = 0

            def get_latest_image(self, *_args):
                self.calls += 1
                return object()

        camera = Camera()
        with patch("camera.decode_color_rgb", return_value=np.zeros((4, 5, 3), dtype=np.uint8)):
            report = flow.warmup_camera(object(), camera, "left", "left")
        self.assertTrue(report["ok"])
        self.assertEqual(report["attempts"], 1)
        self.assertEqual(report["shape"], [4, 5, 3])

    def test_warmup_reports_a_late_camera_without_raising(self):
        class Camera:
            def __init__(self) -> None:
                self.calls = 0

            def get_latest_image(self, *_args):
                self.calls += 1
                raise RuntimeError("no frame yet")

        camera = Camera()
        report = flow.warmup_camera(object(), camera, "left", "left", attempts=2, timeout_ms=1.0)
        self.assertFalse(report["ok"])
        self.assertEqual(report["attempts"], 2)
        self.assertEqual(camera.calls, 2)
        self.assertIn("no frame yet", report["error"])

    def test_reader_uses_a_short_timeout_so_one_pane_cannot_stall(self):
        class Camera:
            def __init__(self) -> None:
                self.timeouts: list[float] = []

            def get_latest_image(self, _type, timeout_ms):
                self.timeouts.append(timeout_ms)
                raise RuntimeError("stop")

        camera = Camera()
        reader = flow._make_jpeg_reader(object(), camera, "left", timeout_ms=250.0)
        with self.assertRaises(RuntimeError):
            reader()
        self.assertEqual(camera.timeouts, [250.0])


if __name__ == "__main__":
    unittest.main()
