import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from control import CartesianPoseTarget, G2Controller
from right_post_grasp_sequence import (
    POST_GRASP_LIFT_M,
    PR1_POSE_NAME,
    PR2_POSE_NAME,
    execute_post_grasp_transport,
    lift_target_from_pose,
    load_pr1_pose,
    load_pr2_pose,
    move_right_pr1,
    move_right_pr2,
    pose_snapshot,
)
import right_post_grasp_sequence as sequence
from test_control import FakeGdk, FakeRobot


class RightPostGraspSequenceTests(unittest.TestCase):
    def test_pr1_pose_is_loaded_as_a_right_arm_preset(self):
        document = {
            "schema_version": 1,
            "poses": {
                PR1_POSE_NAME: {
                    "side": "right",
                    "joint_positions_rad": [0.1] * 7,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(load_pr1_pose(path), [0.1] * 7)

    def test_pr2_pose_is_loaded_as_a_right_arm_preset(self):
        document = {
            "schema_version": 1,
            "poses": {
                PR2_POSE_NAME: {
                    "side": "right",
                    "joint_positions_rad": [0.2] * 7,
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(load_pr2_pose(path), [0.2] * 7)

    def test_pr1_pose_range_is_checked_before_transport(self):
        document = {
            "schema_version": 1,
            "poses": {
                PR1_POSE_NAME: {
                    "side": "right",
                    "joint_positions_rad": [0.1, 0.1, 0.1, 1.5, 0.1, 0.1, 0.1],
                },
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                load_pr1_pose(path)

    def test_lift_target_adds_100_mm_in_base_z_and_keeps_orientation(self):
        before = CartesianPoseTarget(
            (0.2, -0.3, 0.4),
            (0.0, 0.0, 0.2, 0.9797958971),
        )
        target = lift_target_from_pose(before)
        self.assertAlmostEqual(target.position_m[0], before.position_m[0])
        self.assertAlmostEqual(target.position_m[1], before.position_m[1])
        self.assertAlmostEqual(POST_GRASP_LIFT_M, 0.10)
        self.assertAlmostEqual(target.position_m[2] - before.position_m[2], 0.10)
        self.assertEqual(target.orientation_xyzw, before.orientation_xyzw)

    def test_pose_snapshot_saves_current_right_end_link_and_joints(self):
        robot = FakeRobot()
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        snapshot = pose_snapshot(controller)
        self.assertEqual(snapshot["frame"], "arm_r_end_link")
        self.assertEqual(len(snapshot["joint_positions_rad"]), 7)
        self.assertEqual(snapshot["position_m"], [0.0, 0.0, 0.0])

    def test_move_right_pr1_uses_planned_preset_and_checks_arrival(self):
        target = [0.1] * 7

        class StubController:
            def __init__(self):
                self.request = None

            def move_preset(self, side, positions, **kwargs):
                self.request = (side, list(positions), kwargs)

            def checked_arm_state(self):
                return tuple([0.0] * 7 + target)

        controller = StubController()
        result = move_right_pr1(controller, target, timeout_s=1.0)
        self.assertEqual(controller.request[0], "right")
        self.assertEqual(controller.request[1], target)
        self.assertEqual(result["event"], "right_pr1_arrived")
        self.assertAlmostEqual(result["max_joint_error_rad"], 0.0)

    def test_move_right_pr2_uses_planned_preset_and_checks_arrival(self):
        target = [0.1] * 7

        class StubController:
            def __init__(self):
                self.request = None

            def move_preset(self, side, positions, **kwargs):
                self.request = (side, list(positions), kwargs)

            def checked_arm_state(self):
                return tuple([0.0] * 7 + target)

        controller = StubController()
        result = move_right_pr2(controller, target, timeout_s=1.0)
        self.assertEqual(controller.request[0], "right")
        self.assertEqual(controller.request[1], target)
        self.assertEqual(result["event"], "right_pr2_arrived")
        self.assertAlmostEqual(result["max_joint_error_rad"], 0.0)

    def test_move_right_pr1_forwards_explicit_velocity_without_legacy_cap(self):
        target = [0.1] * 7

        class StubController:
            def __init__(self):
                self.request = None

            def move_preset(self, side, positions, **kwargs):
                self.request = (side, list(positions), kwargs)

            def checked_arm_state(self):
                return tuple([0.0] * 7 + target)

        controller = StubController()
        move_right_pr1(controller, target, velocity_rad_s=0.2, timeout_s=1.0)
        self.assertAlmostEqual(controller.request[2]["velocity_rad_s"], 0.2)

    def test_post_grasp_transport_saves_then_lifts_then_moves_to_pr2(self):
        events = []
        saved = {
            "frame": "arm_r_end_link",
            "position_m": [0.1, -0.2, 0.3],
            "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "joint_positions_rad": [0.0] * 7,
        }
        with (
            patch.object(sequence, "pose_snapshot", side_effect=lambda _controller: events.append("save") or saved.copy()),
            patch.object(sequence, "pose_from_snapshot", return_value=object()),
            patch.object(sequence, "lift_after_grasp", side_effect=lambda _controller, _before: events.append("lift") or {"event": "lift"}),
            patch.object(sequence, "move_right_pr2", side_effect=lambda _controller, _target: events.append("pr2") or {"event": "pr2"}),
        ):
            result = execute_post_grasp_transport(object(), [0.1] * 7)
        self.assertEqual(events, ["save", "lift", "pr2"])
        self.assertEqual([item["event"] for item in result], [
            "right_post_grasp_pose_saved", "lift", "pr2",
        ])


if __name__ == "__main__":
    unittest.main()
