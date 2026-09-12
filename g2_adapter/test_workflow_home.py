import json
import tempfile
import unittest
from pathlib import Path

from workflow_home import WORKFLOW_HOME_POSE_NAME, load_workflow_home, move_workflow_home
from safety import ARM_JOINT_NAMES, SafetyError


class FakeHomeController:
    def __init__(self):
        self.target = None
        self.state = [0.0] * 14

    def move_bilateral_preset(self, target, **kwargs):
        self.target = list(target)
        self.state = list(target)

    def checked_arm_state(self):
        return tuple(self.state)


class WorkflowHomeTests(unittest.TestCase):
    def test_loads_and_validates_fourteen_joint_home_pose(self):
        document = {
            "schema_version": 1,
            "poses": {
                WORKFLOW_HOME_POSE_NAME: {
                    "side": "both",
                    "joint_names": list(ARM_JOINT_NAMES),
                    "joint_positions_rad": [0.1] * 14,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(load_workflow_home(path), [0.1] * 14)

    def test_missing_or_one_arm_home_pose_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps({"poses": {}}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_workflow_home(path)
            path.write_text(json.dumps({"poses": {
                WORKFLOW_HOME_POSE_NAME: {
                    "side": "left", "joint_positions_rad": [0.0] * 7,
                },
            }}), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_workflow_home(path)

    def test_moves_both_arms_and_confirms_arrival(self):
        controller = FakeHomeController()
        result = move_workflow_home(
            controller, [0.1] * 14, velocity_rad_s=0.05,
            arrival_tolerance_rad=0.03, timeout_s=1.0,
        )
        self.assertEqual(controller.target, [0.1] * 14)
        self.assertEqual(result["preset"], WORKFLOW_HOME_POSE_NAME)
        self.assertEqual(result["max_joint_error_rad"], 0.0)


if __name__ == "__main__":
    unittest.main()
