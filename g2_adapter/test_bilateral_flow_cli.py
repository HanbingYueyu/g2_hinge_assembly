"""Orchestrator plumbing: stage plan CLI, dry-run validation and stage argv."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import left_grasp_gdk
import right_apriltag_grasp_gdk
import run_bilateral_flow_gdk as flow
from bilateral_services import StageServices
from bilateral_stages import DEFAULT_PLAN, Stage, stage_names


def write_poses(path: Path, *, complete: bool) -> Path:
    poses = {"workflow_home": {"side": "both", "joint_names": None, "joint_positions_rad": [0.0] * 14}}
    if complete:
        for name, side in (("LR1", "right"), ("RR1", "right"), ("RR2", "right"), ("LL1", "left"), ("LL2", "left")):
            poses[name] = {"side": side, "joint_positions_rad": [0.0] * 7}
    path.write_text(json.dumps({"schema_version": 1, "poses": poses}), encoding="utf-8")
    return path


class OrchestratorCliTests(unittest.TestCase):
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
    def test_missing_staged_presets_are_reported_with_exit_1(self):
        with tempfile.TemporaryDirectory() as directory:
            poses = write_poses(Path(directory) / "poses.json", complete=False)
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                code = flow.main(["--dry-run", "--poses", str(poses)])
        self.assertEqual(code, 1)
        text = buffer.getvalue()
        self.assertIn("missing_presets", text)
        for name in ("LR1", "RR1", "LL1", "LL2"):
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

    def test_left_argv_requests_the_hole_move_and_skips_the_duplicate_home(self):
        argv = self.settings.left_argv()
        self.assertIn("--move-above-hole", argv)
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
        self.assertEqual(argv[argv.index("--final-descent-visual-tolerance-mm") + 1], "5.0")
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
    def make_context(self):
        class FakeRuntime:
            def __init__(self) -> None:
                self.events: list[tuple] = []
                self.advances: list[str] = []
                self.left_gripper_opens = 0
                self.left_presets: list[tuple[str, ...]] = []

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
        self.assertEqual(runtime.advances, ["gate_left", "gate_right"])
        completed = [fields["stage"] for event, fields in runtime.events if event == "stage_complete"]
        self.assertEqual(completed, [s.stage.value for s in DEFAULT_PLAN])
        self.assertIn("bilateral_flow_complete", [event for event, _ in runtime.events])

    def test_the_gate_sequence_is_identical_with_or_without_a_hole(self):
        # A missing hole only removes the hole move; both SpaceMouse gates and
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
                self.assertEqual(runtime.advances, ["gate_left", "gate_right"])
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
        self.assertEqual(completed, ["home", "head_localize", "left_grasp", "right_to_lr1"])
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
