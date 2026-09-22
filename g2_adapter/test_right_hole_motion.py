import unittest

import numpy as np

from control import CartesianPoseTarget
from arm_orientation import fixed_forward_down_orientation
from right_hole_motion import (
    _rotation_from_quaternion,
    end_target_for_tool_axis,
    move_right_via_vertical_clearance,
    right_hole_orientation,
)


class RightHoleTargetTests(unittest.TestCase):
    def test_right_lr2_clearance_goes_up_then_returns_to_original_pose(self):
        current = CartesianPoseTarget((0.4, -0.2, 0.4), (0.0, 0.0, 0.0, 1.0))
        origin = CartesianPoseTarget((0.4, -0.2, 0.6), (0.0, 0.0, 0.0, 1.0))

        class Stream:
            def __init__(self):
                self.targets = []
                self.stopped = False

            def update_target(self, target):
                self.targets.append(target)

            def wait_until_actual_target(self, target, *_args, **_kwargs):
                return target

            def stop(self):
                self.stopped = True

        class Controller:
            def __init__(self):
                self.stream = Stream()

            def read_end_effector_pose(self, _frame):
                return current

            def start_right_end_effector_position_stream(self, initial, **kwargs):
                self.initial = initial
                self.stream_kwargs = kwargs
                return self.stream

        controller = Controller()
        result = move_right_via_vertical_clearance(
            controller, target_pose=origin, clearance_mm=50.0, speed_mm_s=10.0,
        )

        self.assertEqual([target.position_m for target in controller.stream.targets], [
            (0.4, -0.2, 0.65),
            (0.4, -0.2, 0.6),
        ])
        self.assertEqual(result["clearance_mm"], 50.0)
        self.assertTrue(controller.stream.stopped)

    def test_right_hole_uses_the_shared_full_flow_orientation(self):
        actual = right_hole_orientation(yaw_deg=20.0, pitch_deg=0.0, roll_deg=180.0)
        expected = fixed_forward_down_orientation(
            forward_yaw_deg=20.0, gripper_pitch_deg=0.0, gripper_roll_deg=180.0,
        )
        self.assertEqual(actual, expected)

    def test_places_cad_axis_origin_at_requested_hole_point(self):
        saved_end = CartesianPoseTarget((1.0, 2.0, 3.0), (0.0, 0.0, 0.0, 1.0))
        base_from_workpiece = np.eye(4)
        base_from_workpiece[:3, 3] = (1.1, 2.2, 3.3)

        target = end_target_for_tool_axis(
            saved_end, base_from_workpiece,
            tool_axis_target_base_m=(0.5, 0.6, 0.7),
            target_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        )

        self.assertTrue(np.allclose(target.position_m, (0.4, 0.4, 0.4)))
        self.assertEqual(target.orientation_xyzw, (0.0, 0.0, 0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
