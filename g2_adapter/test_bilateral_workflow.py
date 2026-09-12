import argparse
import subprocess
import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from run_bilateral_workflow_gdk import (
    EXIT_ADVANCE_RIGHT,
    build_stage_commands,
    main,
    parse_args,
)
from safety import ARM_JOINT_NAMES


class BilateralWorkflowTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.poses = Path(directory.name) / "poses.json"
        self.poses.write_text(json.dumps({"poses": {
            **{
                name: {"side": "left", "joint_positions_rad": [0, 0, 0, -1, 0, 0, 0]}
                for name in ("left_handoff_1", "left_handoff_2")
            },
            "workflow_home": {
                "side": "both",
                "joint_names": list(ARM_JOINT_NAMES),
                "joint_positions_rad": [0.0] * 14,
            },
        }}))

    def test_default_web_ports_are_enabled_for_both_stages(self):
        args = parse_args(["--device", "/dev/hidraw3", "--allow-motion"])
        self.assertEqual(args.left_web_port, 8767)
        self.assertEqual(args.right_web_port, 8767)
        self.assertAlmostEqual(args.head_above_mm, 100.0)
        self.assertAlmostEqual(args.yolo_center_fraction, 0.70)
        self.assertAlmostEqual(args.final_descent_visual_tolerance_mm, 5.0)

    def args(self, **overrides):
        values = {
            "device": "/dev/hidraw3",
            "left_config": "g2_adapter/left_grasp.example.json",
            "right_config": "g2_adapter/right_apriltag_grasp.example.json",
            "poses": str(self.poses),
            "checkpoint": "checkpoints/sam3.1_multiplex.pt",
            "yolo_model": "models/left_d405_hinge_shaft_yolo/best.pt",
            "left_height_reference": "g2_adapter/left_step_reference.json",
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
        self.assertIn("--head-coarse-approach", left)
        self.assertIn("--head-above-mm", left)
        self.assertEqual(left[left.index("--head-above-mm") + 1], "100.0")
        self.assertIn("--final-descent-visual-servo", left)
        self.assertIn("--height-reference", left)
        self.assertIn("--final-descent-visual-tolerance-mm", left)
        self.assertIn("--home-pose-name", left)
        self.assertIn("workflow_home", left)
        self.assertIn("--wait-for-next", left)
        self.assertIn("--spacemouse-device", left)
        self.assertIn("--speed-mm-s", left)
        self.assertIn("--xy-speed-mm-s", left)
        self.assertEqual(left[left.index("--xy-speed-mm-s") + 1], "6.0")
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

    def test_stage_commands_ignore_retired_left_step_model_and_reference(self):
        left, _ = build_stage_commands(self.args(
            left_step_profile="g2_adapter/left_step_profile.json",
            left_step_reference="g2_adapter/left_step_reference.json",
        ))
        self.assertNotIn("--step-profile", left)
        self.assertNotIn("--step-reference", left)

    def test_auto_speed_options_are_omitted_when_not_overridden(self):
        left, right = build_stage_commands(self.args(
            auto_speed_mm_s=None,
            auto_rotation_deg_s=None,
            auto_preset_velocity_rad_s=None,
        ))
        self.assertNotIn("--auto-speed-mm-s", left + right)
        self.assertNotIn("--auto-rotation-deg-s", left + right)
        self.assertNotIn("--auto-preset-velocity-rad-s", left + right)

    def test_completed_in_process_handoff_starts_right(self):
        completed = [
            subprocess.CompletedProcess([], EXIT_ADVANCE_RIGHT),
            subprocess.CompletedProcess([], 0),
        ]
        with patch("run_bilateral_workflow_gdk.subprocess.run", side_effect=completed) as run:
            self.assertEqual(main(self.args()), 0)
        self.assertEqual(run.call_count, 2)
        self.assertIn("--execute", run.call_args_list[1].args[0])

    def test_left_exit_without_completed_handoff_blocks_right(self):
        with patch("run_bilateral_workflow_gdk.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 0)) as run:
            self.assertEqual(main(self.args()), 1)
        self.assertEqual(run.call_count, 1)

    def test_missing_handoff_point_blocks_all_motion(self):
        self.poses.write_text('{"poses": {}}')
        with patch(
            "run_bilateral_workflow_gdk.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1),
        ) as run:
            self.assertEqual(main(self.args()), 2)
        run.assert_not_called()

    def test_missing_workflow_home_blocks_all_motion(self):
        document = json.loads(self.poses.read_text())
        del document["poses"]["workflow_home"]
        self.poses.write_text(json.dumps(document))
        with patch(
            "run_bilateral_workflow_gdk.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1),
        ) as run:
            self.assertEqual(main(self.args()), 2)
        run.assert_not_called()

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
