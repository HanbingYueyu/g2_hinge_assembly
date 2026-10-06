"""Checks for assembly-anchor registration and the fixed-orientation hole route."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from .top_hinge_four_hole_workflow import (
    HOLE_OFFSETS_BASE_LINK_M, LIFT_M, _append_stages, _current_cad_rotation, _first_hole_translation,
    _followup_target,
    _settle_replay_final_joints,
)
from .top_hinge_step_geometry import measure_top_hinge
from .control import CartesianPoseTarget


def _pair(geometry, theta, table=(0.5, 0.1)):
    cad = np.asarray(geometry.red_xy_mm) - np.asarray(geometry.assembly_table_xy_mm)
    rotate = np.array(((math.cos(theta), -math.sin(theta)),
                       (math.sin(theta), math.cos(theta))))
    tip = np.asarray(table) + rotate @ cad / 1000.0
    return {
        "table_base_link_m": [*table, 0.72],
        "tip_base_link_m": [float(tip[0]), float(tip[1]), 0.72],
        "midpoint_xy_m": [float((tip[0] + table[0]) / 2),
                          float((tip[1] + table[1]) / 2)],
        "yaw_rad": math.atan2(tip[1]-table[1], tip[0]-table[0]),
        "spacing_m": float(np.linalg.norm(tip - np.asarray(table))),
    }


class TopHingeFourHoleWorkflowTest(unittest.TestCase):
    def test_hole_biases_use_robot_axes_without_inheriting_first_hole_bias(self):
        geometry = measure_top_hinge()
        nominal_first = np.array((0.45, -0.01286, 0.75))
        expected_biases = ((0.002, -0.003, 0), (0, 0, 0),
                           (0, 0, 0), (0.002, -0.003, 0))
        self.assertEqual(HOLE_OFFSETS_BASE_LINK_M, expected_biases)
        first = nominal_first + np.asarray(expected_biases[0])
        # A rotated workpiece must not rotate the robot-relative correction.
        for theta in (0.0, math.pi / 2, -0.4):
            rotation, mapping = _current_cad_rotation(
                geometry, _pair(geometry, 0), _pair(geometry, theta),
            )
            stages, _ = _append_stages(
                geometry, tuple(first), (0, 0, 0, 1), rotation, mapping,
            )
            np.testing.assert_allclose(stages[0]["position_base_link_m"],
                                       first + (0, 0, LIFT_M), atol=1e-12)
            for index, (_, cad_xy) in enumerate(geometry.next_holes_xy_mm, start=1):
                nominal_xy = (nominal_first[:2] + rotation @ (
                    np.asarray(cad_xy) - np.asarray(geometry.green_xy_mm)
                ) / 1000.0)
                np.testing.assert_allclose(
                    stages[index * 3 - 1]["position_base_link_m"],
                    (*(nominal_xy + np.asarray(expected_biases[index][:2])), nominal_first[2]),
                    atol=1e-12,
                )

    def test_followup_keeps_stream_start_orientation_after_feedback_changes(self):
        initial = CartesianPoseTarget((0.4, -0.1, 0.74), (0.0, 0.0, 0.0, 1.0))
        settled_feedback = CartesianPoseTarget(
            (0.4001, -0.1001, 0.7401),
            (0.0, 0.0, math.sin(math.radians(0.02) / 2),
             math.cos(math.radians(0.02) / 2)),
        )
        target = _followup_target(
            np.array((0.42, -0.12, 0.76)),
            np.asarray(settled_feedback.position_m),
            np.asarray(initial.position_m),
            initial,
        )
        self.assertEqual(target.orientation_xyzw, initial.orientation_xyzw)
        np.testing.assert_allclose(target.position_m, (0.4201, -0.1201, 0.7601))

    def test_first_hole_joint_replay_waits_for_actual_feedback(self):
        names = tuple(f"right_joint_{index}" for index in range(7))
        solved = SimpleNamespace(
            joint_names=names,
            points=(SimpleNamespace(joint_positions_rad=(0.1,) * 7),),
        )

        class FakeRobot:
            commands = 0

            def get_motion_control_status(self):
                return SimpleNamespace(control_mode=1, error_code=0)

            def move_arm_joint_servo(self, positions, duration, group, flag):
                assert positions == [0.1] * 7 and duration == 0.01 and group == 1 and not flag
                self.commands += 1

            def get_joint_states(self):
                error = 0.02 if self.commands == 1 else 0.002
                return {"states": [
                    {"name": name, "motor_position": 0.1 - error, "error_code": 0}
                    for name in names
                ]}

        robot = FakeRobot()
        self.assertAlmostEqual(_settle_replay_final_joints(robot, solved), 0.002)
        self.assertEqual(robot.commands, 4)

    def test_first_hole_motion_preserves_fixed_tool_offset(self):
        geometry = measure_top_hinge()
        old = _pair(geometry, -0.1)
        new = _pair(geometry, 0.2, table=(0.53, 0.08))
        cad_first_from_table = (
            np.asarray(geometry.green_xy_mm) - np.asarray(geometry.assembly_table_xy_mm)
        ) / 1000.0

        def hole_xy(pair, angle):
            rotation = np.array(((math.cos(angle), -math.sin(angle)),
                                 (math.sin(angle), math.cos(angle))))
            return np.asarray(pair["table_base_link_m"][:2]) + rotation @ cad_first_from_table

        tool_offset = np.array((-0.08, -0.17))
        taught = (*map(float, hole_xy(old, -0.1) + tool_offset), 0.74)
        result, translation, _ = _first_hole_translation(
            geometry, old, new, taught, "proper",
        )
        np.testing.assert_allclose(result[:2], hole_xy(new, 0.2) + tool_offset, atol=1e-9)
        np.testing.assert_allclose(
            translation[:2], hole_xy(new, 0.2) - hole_xy(old, -0.1), atol=1e-9,
        )
        self.assertEqual(result[2], taught[2])
        stationary, offset, _ = _first_hole_translation(
            geometry, old, old, taught, "proper",
        )
        np.testing.assert_allclose(stationary, taught, atol=1e-9)
        np.testing.assert_allclose(offset, (0.0, 0.0, 0.0), atol=1e-9)

    def test_both_step_models_supply_the_recognized_pair_and_four_plate_holes(self):
        measured = measure_top_hinge()
        self.assertAlmostEqual(measured.red_green_distance_mm, 89.29634, places=3)
        self.assertAlmostEqual(
            math.dist(measured.red_xy_mm, measured.assembly_table_xy_mm),
            110.733, delta=0.01,
        )
        self.assertEqual([name for name, _ in measured.next_holes_xy_mm],
                         ["bottom_middle", "top_right", "top_left"])

    def test_two_recognized_anchors_determine_absolute_yaw_after_relocation(self):
        geometry = measure_top_hinge()
        theta = 0.4
        old = _pair(geometry, -0.1)
        new = _pair(geometry, theta, table=(0.55, 0.08))
        rotation, mapping = _current_cad_rotation(geometry, old, new)
        self.assertAlmostEqual(mapping["current_cad_to_base_yaw_deg"],
                               math.degrees(theta), places=5)
        self.assertAlmostEqual(mapping["measured_pair_yaw_change_deg"],
                               math.degrees(theta + 0.1), places=5)
        expected = np.array(((math.cos(theta), -math.sin(theta)),
                             (math.sin(theta), math.cos(theta))))
        np.testing.assert_allclose(rotation, expected, atol=1e-9)

    def test_every_hole_lifts_moves_xy_and_descends_without_orientation_change(self):
        geometry = measure_top_hinge()
        rotation, mapping = _current_cad_rotation(geometry, _pair(geometry, 0),
                                                   _pair(geometry, 0.4))
        # The end-link need not coincide with the green hole; its fixed offset
        # cancels when only CAD differences are added to this endpoint.
        endpoint = (0.45, -0.01286, 0.75)
        orientation = (0.1, 0.2, 0.3, math.sqrt(0.86))
        stages, _ = _append_stages(geometry, endpoint, orientation, rotation, mapping)
        self.assertEqual(len(stages), 9)
        previous = np.asarray(endpoint)
        for index, stage in enumerate(stages):
            target = np.asarray(stage["position_base_link_m"])
            self.assertEqual(tuple(stage["orientation_xyzw"]), orientation)
            if index % 3 == 0:
                self.assertEqual(stage["stage"], "lift_20mm")
                np.testing.assert_allclose(target[:2], previous[:2], atol=1e-9)
                self.assertAlmostEqual(target[2] - previous[2], LIFT_M)
            elif index % 3 == 1:
                self.assertEqual(stage["stage"], "xy_at_lifted_z")
                self.assertAlmostEqual(target[2], previous[2])
            else:
                self.assertEqual(stage["stage"], "descend_20mm")
                np.testing.assert_allclose(target[:2], previous[:2], atol=1e-9)
                self.assertAlmostEqual(target[2] - previous[2], -LIFT_M)
            previous = target

    def test_inconsistent_recognized_target_spacing_is_rejected(self):
        geometry = measure_top_hinge()
        old = _pair(geometry, 0)
        bad = _pair(geometry, 0)
        bad["tip_base_link_m"][1] += 0.03
        with self.assertRaisesRegex(ValueError, "spacing"):
            _current_cad_rotation(geometry, old, bad)

    def test_mirror_keeps_anchor_direction_but_swaps_first_hole_side(self):
        geometry = measure_top_hinge()
        rotation, mapping = _current_cad_rotation(geometry, _pair(geometry, 0),
                                                   _pair(geometry, 0.4))
        endpoint = (0.45, -0.01286, 0.75)
        proper, proper_map = _append_stages(geometry, endpoint, (0, 0, 0, 1),
                                             rotation, mapping, "proper")
        mirrored, mirrored_map = _append_stages(geometry, endpoint, (0, 0, 0, 1),
                                                 rotation, mapping, "mirrored")
        self.assertAlmostEqual(proper_map["cad_to_base_xy_determinant"], 1.0)
        self.assertAlmostEqual(mirrored_map["cad_to_base_xy_determinant"], -1.0)
        self.assertGreater(
            np.linalg.norm(np.asarray(proper[2]["position_base_link_m"][:2]) -
                           np.asarray(mirrored[2]["position_base_link_m"][:2])),
            0.01,
        )


if __name__ == "__main__":
    unittest.main()
