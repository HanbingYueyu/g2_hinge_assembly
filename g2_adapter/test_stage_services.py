"""Injection seams that let the staged orchestrator reuse both demos in-process."""

import unittest
from unittest.mock import patch

import left_grasp_gdk
import right_apriltag_grasp_gdk
from bilateral_services import StageServices


class LeftStageExtractionTests(unittest.TestCase):
    def test_parse_args_accepts_an_explicit_argv_and_the_skip_home_flag(self):
        args = left_grasp_gdk.parse_args(["--skip-workflow-home", "--full-cycle", "--allow-motion"])
        self.assertTrue(args.skip_workflow_home)
        self.assertFalse(left_grasp_gdk.parse_args([]).skip_workflow_home)

    def test_standalone_main_delegates_without_services(self):
        with patch.object(left_grasp_gdk, "run_left_cycle", return_value=3) as stage:
            self.assertEqual(left_grasp_gdk.main(["--max-frames", "1"]), 3)
        self.assertIsNone(stage.call_args.kwargs["services"])
        self.assertEqual(stage.call_args.kwargs["argv"], ["--max-frames", "1"])


class RightStageExtractionTests(unittest.TestCase):
    def test_parse_args_accepts_an_explicit_argv(self):
        args = right_apriltag_grasp_gdk.parse_args(["--premove-pose", "RR1"])
        self.assertEqual(args.premove_pose, "RR1")

    def test_standalone_main_delegates_without_services(self):
        with patch.object(right_apriltag_grasp_gdk, "run_right_grasp_stage", return_value=4) as stage:
            self.assertEqual(right_apriltag_grasp_gdk.main([]), 4)
        self.assertIsNone(stage.call_args.kwargs["services"])


class LeftModelSelectionTests(unittest.TestCase):
    """The grasp chain is hinge-only; the shaft model stays for alignment."""

    def test_left_grasp_prefers_the_one_class_hinge_weight(self):
        self.assertTrue(
            str(left_grasp_gdk.DEFAULT_G2_NATIVE_YOLO).endswith("models/left_d405_hinge_yolo/best.pt"),
            left_grasp_gdk.DEFAULT_G2_NATIVE_YOLO,
        )

    def test_the_two_class_hinge_shaft_weight_stays_available_for_alignment(self):
        import run_bilateral_alignment_gdk

        self.assertTrue(
            str(run_bilateral_alignment_gdk.DEFAULT_LEFT_TARGET_MODEL).endswith(
                "models/left_d405_hinge_shaft_yolo/best.pt"
            ),
            run_bilateral_alignment_gdk.DEFAULT_LEFT_TARGET_MODEL,
        )


class ServicesContractTests(unittest.TestCase):
    """Injected stages must only touch attributes StageServices actually has."""

    def test_every_services_attribute_used_by_a_stage_exists(self):
        import dataclasses
        import importlib
        import pathlib
        import re

        available = {field.name for field in dataclasses.fields(StageServices)} | {"put", "get"}
        for module_name in ("left_grasp_gdk", "right_apriltag_grasp_gdk"):
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
                used = set(re.findall(r"\bservices\.([A-Za-z_][A-Za-z0-9_]*)", source))
                self.assertEqual(sorted(used - available), [], f"{module_name} 引用了不存在的 services 字段")


class StageServicesTests(unittest.TestCase):
    def test_services_carry_cross_stage_state(self):
        services = StageServices()
        marker = object()
        self.assertIs(services.put("right_saved_grasp_pose", marker), marker)
        self.assertIs(services.get("right_saved_grasp_pose"), marker)
        self.assertIsNone(services.get("missing"))

    def test_a_fresh_services_object_owns_nothing(self):
        services = StageServices()
        for name in ("gdk", "robot", "controller", "gate", "worker", "tf",
                     "head_yolo", "left_yolo", "right_detector",
                     "head_camera", "left_camera", "right_camera", "web"):
            self.assertIsNone(getattr(services, name), name)


if __name__ == "__main__":
    unittest.main()
