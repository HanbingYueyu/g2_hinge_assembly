import unittest
from unittest.mock import Mock, patch

import numpy as np

import left_seat_motion
from control import CartesianPoseTarget


class LeftSeatMotionTests(unittest.TestCase):
    def setUp(self):
        self.controller = Mock()
        self.stream = self.controller.start_left_end_effector_position_stream.return_value

    def test_descent_stops_when_relative_force_reaches_threshold(self):
        drifted_orientation = (0.001, 0.0, 0.0, 0.9999995)
        poses = iter([
            CartesianPoseTarget((0.4, 0.1, 0.8), (0, 0, 0, 1)),
            CartesianPoseTarget((0.4, 0.1, 0.797), drifted_orientation),
            CartesianPoseTarget((0.4, 0.1, 0.797), drifted_orientation),
        ])
        forces = iter([np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 3.1])])
        with patch.object(left_seat_motion, "measured_left_end_link_pose", side_effect=lambda _c: next(poses)), \
             patch.object(left_seat_motion, "capture_left_force_baseline", return_value=np.zeros(3)), \
             patch.object(left_seat_motion, "read_left_force", side_effect=lambda _r: next(forces)):
            result = left_seat_motion.force_guarded_left_descent(
                self.controller, object(), max_distance_mm=10, force_threshold_n=3,
                speed_mm_s=10, sleep=lambda _s: None,
            )
        self.assertEqual(result["stop_reason"], "force_threshold")
        self.assertAlmostEqual(result["descent_mm"], 3.0)
        hold = self.stream.hold_at_pose.call_args.args[0]
        self.assertEqual(hold.position_m, (0.4, 0.1, 0.797))
        self.assertEqual(hold.orientation_xyzw, (0, 0, 0, 1))
        self.stream.hold_at_pose.assert_called_once()
        self.stream.wait_until_commanded_target.assert_called_once()
        self.stream.stop.assert_called_once()

    def test_descent_stops_successfully_at_maximum_distance(self):
        poses = iter([
            CartesianPoseTarget((0.4, 0.1, 0.8), (0, 0, 0, 1)),
            CartesianPoseTarget((0.4, 0.1, 0.790), (0, 0, 0, 1)),
            CartesianPoseTarget((0.4, 0.1, 0.790), (0, 0, 0, 1)),
        ])
        with patch.object(left_seat_motion, "measured_left_end_link_pose", side_effect=lambda _c: next(poses)), \
             patch.object(left_seat_motion, "capture_left_force_baseline", return_value=np.zeros(3)), \
             patch.object(left_seat_motion, "read_left_force", return_value=np.array([0.0, 0.0, 1.0])):
            result = left_seat_motion.force_guarded_left_descent(
                self.controller, object(), max_distance_mm=10, force_threshold_n=3,
                speed_mm_s=10, sleep=lambda _s: None,
            )
        self.assertEqual(result["stop_reason"], "max_distance")
        self.assertAlmostEqual(result["descent_mm"], 10.0)

    def test_demo_descent_can_finish_near_saved_pose_without_exact_100_mm(self):
        poses = iter([
            CartesianPoseTarget((0.4, 0.1, 0.8), (0, 0, 0, 1)),
            CartesianPoseTarget((0.4, 0.1, 0.704), (0, 0, 0, 1)),
        ])
        with patch.object(left_seat_motion, "measured_left_end_link_pose", side_effect=lambda _c: next(poses)), \
             patch.object(left_seat_motion, "capture_left_force_baseline", return_value=np.zeros(3)), \
             patch.object(left_seat_motion, "read_left_force", return_value=np.array([0.0, 0.0, 0.2])):
            result = left_seat_motion.force_guarded_left_descent(
                self.controller, object(), max_distance_mm=100, force_threshold_n=2,
                speed_mm_s=10, completion_tolerance_mm=5, timeout_s=30,
                sleep=lambda _s: None,
            )
        self.assertEqual(result['stop_reason'], 'max_distance')
        self.assertAlmostEqual(result['descent_mm'], 96.0)

    def test_release_retract_moves_up_exactly_100_mm(self):
        start = CartesianPoseTarget((0.4, 0.1, 0.7), (0, 0, 0, 1))
        with patch.object(left_seat_motion, "measured_left_end_link_pose", return_value=start), \
             patch.object(left_seat_motion, "wait_actual_left_target", side_effect=lambda _c, _s, target, **_k: target):
            result = left_seat_motion.move_left_vertical(
                self.controller, distance_mm=100, speed_mm_s=30,
            )
        target = self.stream.update_target.call_args.args[0]
        self.assertEqual(target.position_m[:2], (0.4, 0.1))
        self.assertAlmostEqual(target.position_m[2], 0.8)
        self.assertEqual(result["distance_mm"], 100.0)


if __name__ == "__main__":
    unittest.main()
