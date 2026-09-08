import argparse
import subprocess
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from spacemouse_input import InputFrame
from run_bilateral_workflow_gdk import (
    EXIT_ADVANCE_RIGHT,
    build_stage_commands,
    main,
    parse_args,
    wait_for_right_advance,
)


class BilateralWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.poses = Path(directory.name) / "poses.json"
        self.poses.write_text(json.dumps({"poses": {
            name: {"side": "left", "joint_positions_rad": [0, 0, 0, -1, 0, 0, 0]}
            for name in ("left_handoff_1", "left_handoff_2")
        }}))

    def test_default_web_ports_are_enabled_for_both_stages(self):
        args = parse_args(["--device", "/dev/hidraw3", "--allow-motion"])
        self.assertEqual(args.left_web_port, 8767)
        self.assertEqual(args.right_web_port, 8767)

    def args(self, **overrides):
        values = {
            "device": "/dev/hidraw3",
            "left_config": "g2_adapter/left_grasp.example.json",
            "right_config": "g2_adapter/right_apriltag_grasp.example.json",
            "poses": str(self.poses),
            "checkpoint": "checkpoints/sam3.1_multiplex.pt",
            "yolo_model": "models/left_d405_hinge_shaft_yolo/best.pt",
            "left_web_port": 0,
            "right_web_port": 0,
            "speed_mm_s": 10.0,
            "rotation_deg_s": 10.0,
            "auto_speed_mm_s": 6.0,
            "auto_rotation_deg_s": 8.0,
            "auto_preset_velocity_rad_s": 0.05,
            "deadzone": 0.12,
            "allow_motion": True,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_stage_commands_include_takeover_and_left_full_cycle_right_execute(self):
        left, right = build_stage_commands(self.args())
        self.assertIn("--full-cycle", left)
        self.assertIn("--wait-for-next", left)
        self.assertIn("--spacemouse-device", left)
        self.assertIn("--speed-mm-s", left)
        self.assertIn("--auto-preset-velocity-rad-s", left)
        self.assertIn("--premove-pr1", right)
        self.assertIn("--post-grasp-pr2", right)
        self.assertIn("--autonomous-approach", right)
        geometry_path = right[right.index("--tag-geometry-profile") + 1]
        self.assertTrue(geometry_path.endswith("/g2_adapter/tag_geometry_electric_screwdriver.json"))
        from pathlib import Path
        self.assertTrue(Path(geometry_path).is_file())
        self.assertIn("--execute", right)
        self.assertIn("--spacemouse-device", right)
        self.assertIn("--auto-speed-mm-s", right)
        self.assertIn("--auto-rotation-deg-s", right)
        self.assertIn("--auto-preset-velocity-rad-s", right)

    def test_auto_speed_options_are_omitted_when_not_overridden(self):
        left, right = build_stage_commands(self.args(
            auto_speed_mm_s=None,
            auto_rotation_deg_s=None,
            auto_preset_velocity_rad_s=None,
        ))
        self.assertNotIn("--auto-speed-mm-s", left + right)
        self.assertNotIn("--auto-rotation-deg-s", left + right)
        self.assertNotIn("--auto-preset-velocity-rad-s", left + right)

    def test_normal_left_completion_waits_for_manual_advance_before_right(self):
        completed = [
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
        ]
        with (
            patch("run_bilateral_workflow_gdk.subprocess.run", side_effect=completed) as run,
            patch("run_bilateral_workflow_gdk.wait_for_right_advance", return_value=True) as wait,
        ):
            self.assertEqual(main(self.args()), 0)
        self.assertEqual(run.call_count, 3)
        self.assertIn("--handoff", run.call_args_list[1].args[0])
        self.assertIn("--execute", run.call_args_list[2].args[0])
        wait.assert_called_once_with("/dev/hidraw3")

        completed = [
            subprocess.CompletedProcess([], EXIT_ADVANCE_RIGHT),
            subprocess.CompletedProcess([], 0),
            subprocess.CompletedProcess([], 0),
        ]
        with (
            patch("run_bilateral_workflow_gdk.subprocess.run", side_effect=completed) as run,
            patch("run_bilateral_workflow_gdk.wait_for_right_advance") as wait,
        ):
            self.assertEqual(main(self.args()), 0)
        wait.assert_not_called()
        self.assertEqual(run.call_count, 3)

    def test_handoff_failure_blocks_right(self):
        with patch("run_bilateral_workflow_gdk.subprocess.run", side_effect=[
            subprocess.CompletedProcess([], EXIT_ADVANCE_RIGHT),
            subprocess.CompletedProcess([], 1),
        ]) as run:
            self.assertEqual(main(self.args()), 1)
        self.assertEqual(run.call_count, 2)

    def test_missing_handoff_point_blocks_all_motion(self):
        self.poses.write_text('{"poses": {}}')
        with patch("run_bilateral_workflow_gdk.subprocess.run") as run:
            self.assertEqual(main(self.args()), 2)
        run.assert_not_called()

    def test_wait_for_right_advance_detects_two_second_right_hold(self):
        frames = iter([
            InputFrame((0.0,) * 6, (False, False), (), (0.0, 0.0), True),
            InputFrame((0.0,) * 6, (False, True), (1,), (0.1, 0.1), True),
            InputFrame((0.0,) * 6, (False, True), (), (2.2, 2.2), True),
        ])
        times = iter([0.0, 0.1, 2.2])

        class FakeHID:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def poll(self):
                return next(frames)

        self.assertTrue(wait_for_right_advance(
            "/dev/hidraw3",
            hid_factory=lambda _device: FakeHID(),
            clock=lambda: next(times),
            sleep=lambda _seconds: None,
        ))

    def test_left_failure_stops_before_right(self):
        with patch(
            "run_bilateral_workflow_gdk.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1),
        ) as run:
            self.assertEqual(main(self.args()), 1)
        self.assertEqual(run.call_count, 1)

    def test_motion_must_be_explicit(self):
        with patch("run_bilateral_workflow_gdk.subprocess.run") as run:
            self.assertEqual(main(self.args(allow_motion=False)), 2)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
