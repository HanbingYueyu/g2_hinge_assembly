import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from bilateral_runtime import (
    OPEN_GRIPPER_RAD,
    STAGED_PRESETS,
    BilateralRuntime,
)
from safety import SafetyError


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
