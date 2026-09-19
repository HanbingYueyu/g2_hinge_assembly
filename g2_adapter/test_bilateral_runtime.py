import json
import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock, patch

# Import OpenCV before any ``patch.dict("sys.modules", ...)`` below.  run_flow()
# lazily imports vlm_flow_checks -> cv2; when that first import happens inside
# the mocked sys.modules it leaves cv2 partially initialised, so the next
# test's cv2 import fails with "module 'cv2.dnn' has no attribute 'DictValue'".
import cv2  # noqa: F401

from bilateral_runtime import (
    OPEN_GRIPPER_RAD,
    STAGED_PRESETS,
    BilateralRuntime,
    console_event_summary,
)
from safety import SafetyError


class ProductionEntryPreparationTests(unittest.TestCase):
    def test_flow_prepares_both_arms_before_workers_and_stages(self):
        self._run_flow_preparation(fail=False)

    def test_flow_preparation_failure_prevents_workers_and_stages(self):
        self._run_flow_preparation(fail=True)

    def _run_flow_preparation(self, *, fail):
        import run_bilateral_flow_gdk as module

        events = []
        controller = Mock()
        controller.checked_arm_state.side_effect = lambda: events.append("state")
        def prepare(sides=("left", "right")):
            self.assertEqual(sides, ("left", "right"))
            events.append("prepare")
            if fail:
                raise SafetyError("preparation failed")
        controller.prepare_joint_impedance_control.side_effect = prepare
        gdk = Mock()
        gdk.gdk_init.return_value = gdk.GDKRes.kSuccess
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            args = module.parse_args(["--allow-motion", "--disable-vlm-checks", "--web-port", "0",
                                      "--log", str(Path(directory) / "flow.jsonl")])
            stack.enter_context(patch.dict("sys.modules", {"agibot_gdk": gdk, "torch": Mock()}))
            stack.enter_context(patch("control.G2Controller", side_effect=lambda *a, **k: events.append("controller") or controller))
            worker = stack.enter_context(patch("spacemouse_takeover.SpaceMouseTakeoverWorker"))
            worker.return_value.start.side_effect = lambda: events.append("worker")
            dianpi = stack.enter_context(patch("dianpi_runner.DianpiCycleRunner"))
            dianpi.return_value.start_p10_monitor.side_effect = lambda: events.append("dianpi")
            stack.enter_context(patch.object(module, "KeyboardStageAdvance"))
            stack.enter_context(patch.object(module.time, "sleep"))
            stack.enter_context(patch("yolo_hinge.load_yolo_segmentation_model"))
            stack.enter_context(patch("right_apriltag_grasp_gdk.detector_for_41h12"))
            drive = stack.enter_context(patch.object(module, "_drive", side_effect=lambda *a: events.append("stage") or 0))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if fail:
                with self.assertRaisesRegex(SafetyError, "preparation failed"):
                    module.run_flow(args)
                worker.assert_not_called()
                dianpi.assert_not_called()
                drive.assert_not_called()
            else:
                self.assertEqual(module.run_flow(args), 0)
                self.assertEqual(events, ["controller", "state", "prepare", "dianpi", "worker", "stage"])
            gdk.gdk_release.assert_called_once()

    def test_preset_prepares_only_the_moving_arm(self):
        for side in ("left", "right"):
            with self.subTest(side=side):
                self._run_preset_entry(side=side, allow_motion=True)

    def test_preset_preview_does_not_prepare_or_move(self):
        self._run_preset_entry(side="left", allow_motion=False)

    def test_preset_preparation_failure_prevents_planning(self):
        self._run_preset_entry(side="right", allow_motion=True, fail=True)

    def _run_preset_entry(self, *, side, allow_motion, fail=False):
        import run_preset as module

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            poses = Path(directory) / "poses.json"
            poses.write_text(json.dumps({"poses": {"selected": {"side": side, "joint_positions_rad": [0.0] * 7}}}))
            args = Namespace(poses=poses, preset="selected", allow_motion=allow_motion, velocity_rad_s=0.08)
            controller, events = self._entry_controller(stack, module, args, fail=fail)
            self.assertEqual(module.main(), 1 if fail else 0)
            self.assertEqual(events, (["prepare"] if fail else ["prepare", "move"]) if allow_motion else [])
            if allow_motion:
                controller.prepare_joint_impedance_control.assert_called_once_with((side,))

    def test_sequence_prepares_left_before_presets_or_handoff(self):
        for handoff in (False, True):
            with self.subTest(handoff=handoff):
                self._run_sequence_entry(allow_motion=True, handoff=handoff)

    def test_sequence_preview_does_not_prepare_or_move(self):
        self._run_sequence_entry(allow_motion=False)

    def test_sequence_preparation_failure_prevents_handoff_and_planning(self):
        self._run_sequence_entry(allow_motion=True, handoff=True, fail=True)

    def _run_sequence_entry(self, *, allow_motion, handoff=False, fail=False):
        import run_left_premove_sequence as module

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            args = Namespace(poses=Path("unused"), allow_motion=allow_motion, handoff=handoff,
                             velocity_rad_s=0.08, arrival_tolerance_rad=0.03, timeout_per_preset_s=25.0,
                             log=Path(directory) / "sequence.jsonl")
            controller, events = self._entry_controller(stack, module, args, fail=fail)
            stack.enter_context(patch.object(module, "load_presets", return_value=[("p1", [0.0] * 7), ("p2", [0.0] * 7)]))
            stack.enter_context(patch.object(module, "open_handoff_gripper", side_effect=lambda c: events.append("gripper") or {}))
            self.assertEqual(module.main(), 1 if fail else 0)
            expected = ["prepare", *(["gripper"] if handoff else []), "move", "move"]
            self.assertEqual(events, (["prepare"] if fail else expected) if allow_motion else [])
            if allow_motion:
                controller.prepare_joint_impedance_control.assert_called_once_with(("left",))

    def _entry_controller(self, stack, module, args, *, fail):
        events = []
        controller = Mock()
        controller.checked_arm_state.return_value = [0.0] * 14
        def prepare(*_args):
            events.append("prepare")
            if fail:
                raise SafetyError("preparation failed")
        controller.prepare_joint_impedance_control.side_effect = prepare
        controller.move_preset.side_effect = lambda *a, **k: events.append("move")
        gdk = Mock()
        gdk.gdk_init.return_value = gdk.GDKRes.kSuccess
        stack.enter_context(patch.dict("sys.modules", {"agibot_gdk": gdk}))
        stack.enter_context(patch.object(module, "parse_args", return_value=args))
        stack.enter_context(patch.object(module, "G2Controller", return_value=controller))
        stack.enter_context(patch.object(module.time, "sleep"))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        return controller, events


class FakeController:
    def __init__(self) -> None:
        self.motion_gate = object()
        self.gripper_targets: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.gripper_positions: dict[str, float] = {"left": 0.0, "right": 0.0}
        self.gripper_reach_after = 0
        self.gripper_reads = 0
        self.gripper_moves: dict[str, int] = {"left": 0, "right": 0}
        self.control_modes: list[int] = []
        self.left_current = [0.0] * 7
        self.track_left_preset = True
        self.preset_requests: list[tuple] = []

    def set_control_mode(self, mode: int) -> None:
        self.control_modes.append(int(mode))

    def move_gripper(self, side: str, position_rad: float) -> None:
        self.gripper_moves[side] += 1
        self.gripper_targets[side] = float(position_rad)

    def read_gripper_position(self, side: str):
        self.gripper_reads += 1
        if self.gripper_reads > self.gripper_reach_after:
            return float(self.gripper_targets.get(side, 0.0))
        return self.gripper_positions.get(side)

    def checked_arm_state(self):
        return tuple(list(self.left_current) + [0.0] * 7)

    def move_preset(self, side, positions, **kwargs) -> None:
        self.preset_requests.append((side, list(positions), kwargs))
        if side == "left" and self.track_left_preset:
            self.left_current = list(positions)


class FakeWorker:
    def __init__(self) -> None:
        self.allow_advance_right = False
        self.advance_requested = False
        self.consume_calls = 0
        self.failed: BaseException | None = None
        self.stop_for_final_motion_calls = 0

    def raise_if_failed(self) -> None:
        if self.failed is not None:
            raise SafetyError("fake worker failed")

    def consume_advance(self) -> None:
        self.consume_calls += 1
        self.advance_requested = False

    def stop_for_final_motion(self) -> None:
        self.stop_for_final_motion_calls += 1


def write_poses(path: Path) -> Path:
    document = {
        "schema_version": 1,
        "poses": {
            "workflow_home": {
                "side": "both",
                "joint_names": None,
                "joint_positions_rad": [0.0] * 14,
            },
            "LL1": {"side": "left", "joint_positions_rad": [0.1] * 7},
            "LL2": {"side": "left", "joint_positions_rad": [0.2] * 7},
            "LR1": {"side": "right", "joint_positions_rad": [0.3] * 7},
            "RR1": {"side": "right", "joint_positions_rad": [0.4] * 7},
            "RR2": {"side": "right", "joint_positions_rad": [0.5] * 7},
        },
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


class RuntimeFixture(unittest.TestCase):
    def setUp(self) -> None:
        self._temp = tempfile.TemporaryDirectory()
        self.poses = write_poses(Path(self._temp.name) / "poses.json")
        self.controller = FakeController()
        self.worker = FakeWorker()
        self.ticks = {"now": 0.0}

        def clock() -> float:
            self.ticks["now"] += 0.25
            return self.ticks["now"]

        self.runtime = BilateralRuntime(
            args=Namespace(poses=self.poses, home_pose_name="workflow_home"),
            controller=self.controller,
            gate=object(),
            worker=self.worker,
            clock=clock,
            sleep=lambda _seconds: None,
        )

    def tearDown(self) -> None:
        self._temp.cleanup()


class ConsoleSummaryTests(unittest.TestCase):
    def test_stage_and_force_events_have_short_operator_messages(self):
        self.assertEqual(
            console_event_summary({"event": "stage_start", "stage": "left", "title": "左臂下探"}),
            "[阶段开始] 左臂下探 (left)",
        )
        self.assertEqual(
            console_event_summary({
                "event": "left_seat_descent_complete", "stop_reason": "force_threshold",
                "descent_mm": 3.2, "delta_force_n": 3.1,
            }),
            "[下探完成] 原因=力阈值 距离=3.2 mm 力变化=3.1 N",
        )
        self.assertIsNone(console_event_summary({"event": "verbose_internal", "value": [1, 2, 3]}))

    def test_emit_writes_full_json_but_prints_only_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "flow.jsonl"
            runtime = BilateralRuntime(args=Namespace(poses="unused"), controller=None, journal_path=log)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                runtime.emit("stage_complete", stage="right_to_lr1", preset="LR1")
            self.assertEqual(output.getvalue().strip(), "[阶段完成] right_to_lr1")
            stored = json.loads(log.read_text(encoding="utf-8"))
            self.assertEqual(stored["preset"], "LR1")
            self.assertIn("utc", stored)


class GripperAndHomeTests(RuntimeFixture):
    def test_open_all_grippers_drives_both_arms_and_confirms_feedback(self):
        event = self.runtime.open_all_grippers()
        self.assertEqual(set(self.controller.gripper_targets), {"left", "right"})
        self.assertAlmostEqual(self.controller.gripper_targets["left"], OPEN_GRIPPER_RAD)
        self.assertAlmostEqual(self.controller.gripper_targets["right"], OPEN_GRIPPER_RAD)
        self.assertEqual(event["event"], "all_grippers_opened")

    def test_open_all_grippers_repeats_the_command_until_feedback_matches(self):
        # A single non-blocking move_end_effector_joint request does not open
        # the jaw; the proven set_omnipicker.py tool re-sends it every ~100 ms.
        self.controller.gripper_reach_after = 2
        event = self.runtime.open_all_grippers(resend_period_s=0.1)
        self.assertEqual(event["event"], "all_grippers_opened")
        self.assertGreaterEqual(self.controller.gripper_moves["left"], 2)
        self.assertGreaterEqual(self.controller.gripper_moves["right"], 2)
        self.assertGreaterEqual(event["command_attempts"]["left"], 2)

    def test_open_all_grippers_rejects_an_out_of_range_angle(self):
        with self.assertRaises(ValueError):
            self.runtime.open_all_grippers(open_rad=0.5)

    def test_open_all_grippers_times_out_without_feedback(self):
        self.controller.gripper_reach_after = 10 ** 9
        self.ticks["now"] = 1000.0
        with self.assertRaises(SafetyError):
            self.runtime.open_all_grippers(timeout_s=1.0)


class PresetTests(RuntimeFixture):
    def test_left_presets_run_in_order_and_check_arrival(self):
        records = self.runtime.move_left_presets(("LL1", "LL2"))
        self.assertEqual([record["preset"] for record in records], ["LL1", "LL2"])
        self.assertEqual([request[1] for request in self.controller.preset_requests],
                         [[0.1] * 7, [0.2] * 7])

    def test_left_preset_timeout_is_reported(self):
        self.controller.track_left_preset = False
        self.ticks["now"] = 500.0
        with self.assertRaises(SafetyError):
            self.runtime.move_left_presets(("LL1",), timeout_s=1.0)

    def test_right_preset_uses_the_recorded_right_arm_values(self):
        with patch("bilateral_runtime.move_right_preset", return_value={"event": "RR1_arrived"}) as mocked:
            record = self.runtime.move_right_preset("RR1")
        self.assertEqual(record["event"], "RR1_arrived")
        self.assertEqual(mocked.call_args[0][1], [0.4] * 7)
        self.assertEqual(mocked.call_args[1]["preset_name"], "RR1")

    def test_load_preset_rejects_a_name_outside_the_staged_flow(self):
        with self.assertRaises(ValueError):
            self.runtime.load_preset("left_p3")

    def test_validate_plan_assets_lists_missing_presets(self):
        self.poses.write_text(json.dumps({"schema_version": 1, "poses": {}}), encoding="utf-8")
        report = self.runtime.validate_plan_assets()
        self.assertEqual(sorted(report["missing"]), sorted(STAGED_PRESETS))
        self.assertTrue(all(not item["ok"] for item in report["presets"].values()))

    def test_validate_plan_assets_accepts_a_complete_file(self):
        report = self.runtime.validate_plan_assets()
        self.assertEqual(report["missing"], [])


class GateTests(RuntimeFixture):
    def test_wait_for_advance_consumes_exactly_one_request(self):
        self.worker.advance_requested = False
        original_sleep = self.runtime.sleep
        calls = {"n": 0}

        def sleep(_seconds):
            calls["n"] += 1
            if calls["n"] == 1:
                self.worker.advance_requested = True
            original_sleep(_seconds)

        self.runtime.sleep = sleep
        self.runtime.wait_for_advance("gate_left")
        self.assertEqual(self.worker.consume_calls, 1)
        self.assertFalse(self.worker.advance_requested)
        self.assertFalse(self.worker.allow_advance_right)
        self.assertEqual([event["event"] for event in self.runtime.events],
                         ["stage_waiting_for_advance", "stage_advance_received"])

    def test_wait_for_advance_requires_a_worker(self):
        self.runtime.worker = None
        with self.assertRaises(SafetyError):
            self.runtime.wait_for_advance("gate_left")

    def test_release_gate_stops_the_worker_and_drops_the_gate(self):
        self.runtime.release_gate()
        self.assertEqual(self.worker.stop_for_final_motion_calls, 1)
        self.assertIsNone(self.controller.motion_gate)


if __name__ == "__main__":
    unittest.main()
