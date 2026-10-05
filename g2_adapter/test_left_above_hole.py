import inspect
import contextlib
import io
import pathlib
import re
import unittest
from argparse import Namespace
from unittest.mock import Mock, patch

import left_grasp_gdk
from control import CartesianPoseTarget
from safety import SafetyError


def make_gate_monitor(*, now: float = 0.0) -> dict:
    return {
        "reference_center_px": [100.0, 100.0],
        "reference_area_px": 1000,
        "awaiting_fresh_observation": True,
        "last_observation_at": now,
        "stale_since": None,
        "paused": False,
        "failure": None,
    }


def make_observation(*, valid=True, sequence: int = 1) -> dict:
    return {
        "sequence": sequence,
        "valid": valid,
        "reason": "locked_target_lost",
        "center_px": [100.0, 100.0],
        "area_px": 1000,
    }


def gate_step(monitor: dict, observation, *, now: float, max_stale_s: float = 2.0) -> str:
    return left_grasp_gdk.held_part_gate_step(
        monitor,
        observation,
        now=now,
        max_stale_s=max_stale_s,
        center_drift_px=12.0,
        area_change_fraction=0.10,
    )


class HeldPartGateTests(unittest.TestCase):
    """The hole move must not start before a fresh "still held" confirmation."""

    def test_gate_waits_until_a_fresh_frame_arrives(self):
        monitor = make_gate_monitor(now=0.0)
        self.assertEqual(gate_step(monitor, None, now=0.5), "wait")
        self.assertTrue(monitor["awaiting_fresh_observation"])

    def test_gate_goes_stale_and_holds_instead_of_moving_blind(self):
        monitor = make_gate_monitor(now=0.0)
        self.assertEqual(gate_step(monitor, None, now=3.0), "stale")
        self.assertTrue(monitor["paused"])

    def test_gate_proceeds_on_a_healthy_fresh_frame(self):
        monitor = make_gate_monitor(now=0.0)
        self.assertEqual(gate_step(monitor, make_observation(valid=True), now=0.5), "proceed")
        self.assertIsNone(monitor["failure"])

    def test_gate_reports_lost_on_a_timeout_confirmed_loss(self):
        monitor = make_gate_monitor(now=0.0)
        self.assertEqual(gate_step(monitor, make_observation(valid=False), now=3.0), "lost")
        self.assertIsNotNone(monitor["failure"])

    def test_gate_waits_for_positive_held_target_evidence_after_temporary_miss(self):
        # A temporary miss is not a drop, but also cannot authorize transport.
        monitor = make_gate_monitor(now=0.0)
        self.assertEqual(gate_step(monitor, make_observation(valid=None), now=0.5), "wait")
        self.assertEqual(gate_step(monitor, make_observation(valid=True), now=0.6), "proceed")

    def test_gate_recovers_after_stale_with_a_fresh_frame(self):
        monitor = make_gate_monitor(now=0.0)
        self.assertEqual(gate_step(monitor, None, now=3.0), "stale")
        self.assertEqual(gate_step(monitor, make_observation(valid=True, sequence=7), now=3.2), "proceed")
        self.assertFalse(monitor["paused"])
        self.assertIsNone(monitor["stale_since"])


def make_args(**overrides) -> Namespace:
    base = dict(
        step_tcp_end_mm="0,0,145",
        hole_speed_mm_s=10.0,
        auto_preset_velocity_rad_s=.2,
        preset_velocity_rad_s=.08,
        hole_x_mm=0.0,
        hole_y_mm=0.0,
        hole_z_mm=100.0,
        hole_forward_yaw_deg=20.0,
        hole_gripper_pitch_deg=42.0,
        hole_gripper_roll_deg=180.0,
        orientation_speed_deg_s=20.0,
        max_travel_mm=450.0,
        head_xy_tolerance_mm=50.0,
    )
    base.update(overrides)
    return Namespace(**base)


class LeftAboveHoleTests(unittest.TestCase):
    def test_successful_hole_move_logs_details_without_terminal_json(self):
        terminal = io.StringIO()
        with contextlib.redirect_stdout(terminal):
            self._run()
        self.assertEqual(terminal.getvalue(), "")
        self.assertIn('"event": "left_above_hole_motion_start"', self.log.getvalue())
        self.assertIn('"event": "left_above_hole_arrived"', self.log.getvalue())

    def test_head_approach_uses_endpoint_ik_and_preset_velocity(self):
        args = left_grasp_gdk.parse_args([])
        args.auto_preset_velocity_rad_s = .23
        args.step_tcp_end_mm = '0,0,0'
        args.head_above_mm = 100.
        controller = Mock()
        controller.read_end_effector_pose.return_value = CartesianPoseTarget((.4,.2,.8), (0.,0.,0.,1.))
        with patch('left_hole_joint_move.move_left_hole_joint_target', side_effect=lambda c, target, **kw: target) as move:
            _, _, result = left_grasp_gdk.run_head_coarse_approach_in_process(
                gdk=None, controller=controller, tf=None, args=args,
                head_model_path=pathlib.Path('unused'), log=io.StringIO(),
                prelocalized_targets={'hinge': {'position_base_link_m': [.4,.2,.7]}})
        self.assertEqual(move.call_args.kwargs['velocity_rad_s'], .23)
        self.assertEqual(result['target_end_link_base_m'], [.4,.2,.7999999999999999])
        controller.start_left_end_effector_position_stream.assert_not_called()

    def setUp(self) -> None:
        self.log = io.StringIO()
        self.hole = [0.5, 0.1, 0.8]
        self.current = CartesianPoseTarget((0.4, 0.0, 0.9), (0, 0, 0, 1))
        self.controller = Mock()
        self.controller.read_end_effector_pose.return_value = self.current
        self.stream = self.controller.start_left_end_effector_position_stream.return_value
        self.stream.wait_until_target.return_value = True

    def _run(self, actual=None, args=None):
        if actual is None:
            def reached_target(_controller, target, **_kwargs):
                return target
            wait_patch = patch("left_hole_joint_move.move_left_hole_joint_target", side_effect=reached_target)
        else:
            wait_patch = patch("left_hole_joint_move.move_left_hole_joint_target", return_value=actual)
        with wait_patch as waiter:
            record = left_grasp_gdk.move_left_above_hole(
                controller=self.controller, hole_base_m=self.hole,
                args=args or make_args(), log=self.log,
            )
        return record, waiter

    def test_move_commends_the_measured_hole_above_height_and_pose(self):
        record, waiter = self._run()
        self.assertEqual(record["event"], "left_above_hole_arrived")
        self.assertEqual(record["travel_mm"] > 0, True)
        self.assertEqual(waiter.call_count, 1)
        self.assertEqual(waiter.call_args.kwargs['velocity_rad_s'], .2)
        self.controller.start_left_end_effector_position_stream.assert_not_called()
        self.assertIn("left_above_hole_motion_start", self.log.getvalue())

    def test_xyz_offsets_shift_the_target_in_base_link(self):
        actual = CartesianPoseTarget((0.52, 0.07, 0.92), (0, 0, 0, 1))
        _, waiter = self._run(
            actual=actual,
            args=make_args(
                step_tcp_end_mm="0,0,0",
                hole_x_mm=20.0,
                hole_y_mm=-30.0,
                hole_z_mm=120.0,
            ),
        )
        target = waiter.call_args.args[1]
        self.assertEqual(target.position_m, (0.52, 0.07, 0.92))

    def test_horizontal_error_beyond_tolerance_is_rejected(self):
        actual = CartesianPoseTarget((0.6, 0.1, 1.0), (0, 0, 0, 1))
        with self.assertRaises(SafetyError):
            self._run(actual=actual, args=make_args(head_xy_tolerance_mm=5.0))

    def test_invalid_hole_coordinates_are_rejected(self):
        with self.assertRaises(SafetyError):
            left_grasp_gdk.move_left_above_hole(
                controller=self.controller, hole_base_m=[0.0, float("nan"), 0.5],
                args=make_args(), log=self.log,
            )

    def test_out_of_range_speed_and_height_are_rejected(self):
        with self.assertRaises(SafetyError):
            left_grasp_gdk.move_left_above_hole(
                controller=self.controller, hole_base_m=self.hole,
                args=make_args(auto_preset_velocity_rad_s=0.0), log=self.log,
            )
        with self.assertRaises(SafetyError):
            left_grasp_gdk.move_left_above_hole(
                controller=self.controller, hole_base_m=self.hole,
                args=make_args(hole_z_mm=0.0), log=self.log,
            )

    def test_invalid_tcp_definition_is_rejected(self):
        with self.assertRaises(SafetyError):
            left_grasp_gdk.move_left_above_hole(
                controller=self.controller, hole_base_m=self.hole,
                args=make_args(step_tcp_end_mm="0,0"), log=self.log,
            )


class HoleTargetFromContextTests(unittest.TestCase):
    """A missing hole downgrades to "hold and wait", it no longer aborts."""

    def test_returns_the_head_hole_when_present(self):
        self.assertEqual(
            left_grasp_gdk.hole_target_from_context({"hole_position_base_link_m": [0.5, 0.1, 0.8]}),
            [0.5, 0.1, 0.8],
        )

    def test_returns_none_when_the_hole_was_not_detected(self):
        for context in (None, {}, {"hole_position_base_link_m": None}):
            with self.subTest(context=context):
                self.assertIsNone(left_grasp_gdk.hole_target_from_context(context))

    def test_orchestrator_targets_become_coarse_context_without_relocalizing(self):
        targets = {
            "hinge": {"position_base_link_m": [0.4, 0.2, 0.7], "samples": 5},
            "hole": {"position_base_link_m": [0.5, 0.1, 0.6], "samples": 4},
        }
        self.assertEqual(
            left_grasp_gdk.head_context_from_targets(targets),
            {
                "hinge_position_base_link_m": [0.4, 0.2, 0.7],
                "hole_position_base_link_m": [0.5, 0.1, 0.6],
            },
        )

    def test_missing_hole_is_a_json_null_log_value(self):
        self.assertIsNone(left_grasp_gdk.optional_point(None))

    def test_prelocalized_path_needs_no_camera_model_or_sensor_tf(self):
        class ForbiddenGdk:
            class Camera:
                def __init__(self, *_args):
                    raise AssertionError("camera must not be opened")

        class ForbiddenTf:
            def get_tf_from_sensor(self, *_args):
                raise AssertionError("sensor TF must not be read")

        events = []
        class Controller:
            def set_control_mode(self, _mode):
                events.append(("mode", _mode))

            def prepare_joint_impedance_control(self, sides):
                events.append(("prepare", sides))

            def read_end_effector_pose(self, _frame):
                raise RuntimeError("motion boundary reached")

        args = Namespace(
            step_tcp_end_mm="0,0,0", head_forward_yaw_deg=0.0,
            head_gripper_pitch_deg=42.0, head_gripper_roll_deg=180.0,
            yolo_device="cuda:0", head_samples=3, hole_sample_timeout_s=1.0,
            head_max_sample_spread_mm=20.0, head_min_inlier_samples=2,
            head_yolo_confidence=0.8,
        )
        targets = {"hinge": {"position_base_link_m": [0.4, 0.2, 0.7]}}
        with self.assertRaisesRegex(RuntimeError, "motion boundary reached"):
            left_grasp_gdk.run_head_coarse_approach_in_process(
                gdk=ForbiddenGdk, controller=Controller(), tf=ForbiddenTf(), args=args,
                head_model_path=pathlib.Path("unused.pt"), log=io.StringIO(),
                prelocalized_targets=targets,
            )
        self.assertEqual(events, [("prepare", ("left",))])


class HoleFinalizePlanTests(unittest.TestCase):
    """A missing hole only removes the move; the rest of the flow is identical."""

    def test_a_localized_hole_moves_above_it(self):
        action, event = left_grasp_gdk.hole_finalize_plan([0.5, 0.1, 0.8])
        self.assertEqual(action, "move_to_hole")
        self.assertEqual(event, "left_stage_complete_above_hole")

    def test_staged_grasp_finishes_before_hole_transfer(self):
        action, event = left_grasp_gdk.hole_finalize_plan([0.5, 0.1, 0.8], defer_move=True)
        self.assertEqual(action, "defer_hole_move")
        self.assertEqual(event, "left_grasp_ready_for_hole_move")

    def test_a_missing_hole_only_skips_the_move(self):
        for missing in (None,):
            with self.subTest(hole=missing):
                action, event = left_grasp_gdk.hole_finalize_plan(missing)
                self.assertEqual(action, "skip_hole_move")
                self.assertEqual(event, "left_stage_complete_without_hole_move")


class FakeGripperController:
    """Acts as both controller and its left gripper stream."""

    def __init__(self, positions: list[float]) -> None:
        self.positions = list(positions)
        self.stream_target: float | None = None
        self.stream_stopped = False

    def read_gripper_position(self, side: str):
        if len(self.positions) > 1:
            return self.positions.pop(0)
        return self.positions[0]

    def start_left_gripper_position_stream(self, target_rad: float, **_kwargs):
        self.stream_target = float(target_rad)
        return self

    def raise_if_failed(self) -> None:
        return None

    def stop(self) -> None:
        self.stream_stopped = True


class LeftGripperReleaseTests(unittest.TestCase):
    """The hold path releases the part only after feedback confirms the jaw."""

    def test_open_confirms_the_measured_jaw_angle(self):
        controller = FakeGripperController([0.0, -0.78])
        record = left_grasp_gdk.open_left_gripper_with_feedback(
            controller, target_rad=-0.785, tolerance_rad=0.035, timeout_s=2.0,
        )
        self.assertEqual(record["event"], "gripper_open_confirmed")
        self.assertAlmostEqual(record["actual_position_rad"], -0.78)
        self.assertGreaterEqual(record["attempts"], 1)
        self.assertTrue(controller.stream_stopped)

    def test_open_times_out_without_feedback(self):
        controller = FakeGripperController([0.0])
        with self.assertRaises(SafetyError):
            left_grasp_gdk.open_left_gripper_with_feedback(
                controller, target_rad=-0.785, tolerance_rad=0.035, timeout_s=0.15,
            )
        self.assertTrue(controller.stream_stopped)

    def test_open_rejects_non_positive_tolerance_or_timeout(self):
        controller = FakeGripperController([0.0])
        with self.assertRaises(ValueError):
            left_grasp_gdk.open_left_gripper_with_feedback(
                controller, target_rad=-0.785, tolerance_rad=0.0, timeout_s=1.0,
            )
        with self.assertRaises(ValueError):
            left_grasp_gdk.open_left_gripper_with_feedback(
                controller, target_rad=-0.785, tolerance_rad=0.03, timeout_s=0.0,
            )


class CliArgumentContractTests(unittest.TestCase):
    """Guard against referencing a CLI flag that this module never defines.

    ``move_left_above_hole`` was first written against the head-localization
    module's argument names, which raised ``AttributeError`` on the robot after
    a successful grasp.
    """

    # Attributes assigned onto ``args`` at runtime instead of via argparse.
    RUNTIME_ASSIGNED = {
        "step_alignment", "step_compensation_m", "step_geometry", "step_height_samples", "step_tcp_m",
    }
    # Pre-existing latent bug inside the retired ``detector == "sam3"`` branch.
    # The staged flow always runs ``--detector yolo`` and never enters it.
    RETIRED_PATH_MISSING = {"sam_grasp_height_fraction"}

    @staticmethod
    def _used_names(source: str) -> set[str]:
        return set(re.findall(r"\bargs\.([A-Za-z_][A-Za-z0-9_]*)", source))

    def test_move_left_above_hole_only_uses_defined_cli_arguments(self):
        defined = set(vars(left_grasp_gdk.parse_args([])))
        used = self._used_names(inspect.getsource(left_grasp_gdk.move_left_above_hole))
        self.assertEqual(sorted(used - defined), [])

    def test_module_argument_references_match_the_known_allow_list(self):
        source = pathlib.Path(left_grasp_gdk.__file__).read_text(encoding="utf-8")
        defined = set(vars(left_grasp_gdk.parse_args([])))
        missing = self._used_names(source) - defined - self.RUNTIME_ASSIGNED
        self.assertEqual(missing, self.RETIRED_PATH_MISSING)


if __name__ == "__main__":
    unittest.main()
