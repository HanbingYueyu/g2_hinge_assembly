import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from run_left_premove_sequence import HANDOFF_PRESETS, load_presets, open_handoff_gripper


class LeftPremoveSequenceTests(unittest.TestCase):
    def test_handoff_open_waits_for_feedback_and_stops_stream(self):
        controller = Mock()
        controller.read_gripper_position.side_effect = [None, 0.0, -0.78]
        event = open_handoff_gripper(controller, clock=lambda: 0, sleep=lambda _: None)
        self.assertEqual(event["actual_position_rad"], -0.78)
        controller.start_left_gripper_position_stream.assert_called_once_with(-0.785)
        controller.start_left_gripper_position_stream.return_value.stop.assert_called_once()

    def test_handoff_open_timeout_stops_stream(self):
        controller = Mock()
        controller.read_gripper_position.return_value = 0.0
        times = iter([0.0, 6.1])
        with self.assertRaisesRegex(RuntimeError, "不执行交接点"):
            open_handoff_gripper(controller, clock=lambda: next(times))
        controller.start_left_gripper_position_stream.return_value.stop.assert_called_once()

    def test_loads_p1_then_p2_ignoring_post_grasp_p3(self):
        document = {
            "schema_version": 1,
            "poses": {
                "left_p1": {"side": "left", "joint_positions_rad": [1.0] * 7},
                "left_p2": {"side": "left", "joint_positions_rad": [0.2] * 7},
                "left_p3": {"side": "left", "joint_positions_rad": [0.3] * 7},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                load_presets(path),
                [("left_p1", [1.0] * 7), ("left_p2", [0.2] * 7)],
            )

    def test_handoff_order_and_invalid_second_point(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            document = {"poses": {name: {"side": "left", "joint_positions_rad": [0.2] * 7} for name in reversed(HANDOFF_PRESETS)}}
            path.write_text(json.dumps(document))
            self.assertEqual([name for name, _ in load_presets(path, HANDOFF_PRESETS)], list(HANDOFF_PRESETS))
            document["poses"][HANDOFF_PRESETS[1]]["joint_positions_rad"][0] = float("nan")
            path.write_text(json.dumps(document))
            with self.assertRaises(RuntimeError):
                load_presets(path, HANDOFF_PRESETS)


if __name__ == "__main__":
    unittest.main()
