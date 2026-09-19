import unittest
from unittest.mock import patch
from types import SimpleNamespace

from control import CartesianPoseTarget
from left_grasp_gdk import recover_for_visual_regrasp, lift_before_visual_reacquire


class VisualRetryTests(unittest.TestCase):
    def test_streamed_reacquire_lift_is_vertical_and_failure_stops_stream(self):
        origin = CartesianPoseTarget((.2,.1,.3), (0.,0.,0.,1.))
        targets, stopped = [], []
        stream = SimpleNamespace(update_target=targets.append, stop=lambda: stopped.append(True))
        controller = SimpleNamespace(start_left_end_effector_position_stream=lambda *a,**kw: stream)
        with patch('left_grasp_gdk.measured_left_end_link_pose', return_value=origin), patch('left_grasp_gdk.wait_actual_left_target', side_effect=RuntimeError('not reached')):
            with self.assertRaisesRegex(RuntimeError, 'not reached'):
                lift_before_visual_reacquire(controller, speed_mm_s=30.)
        self.assertEqual(targets[0].position_m[:2], origin.position_m[:2])
        self.assertAlmostEqual(targets[0].position_m[2], .33)
        self.assertEqual(targets[0].orientation_xyzw, origin.orientation_xyzw)
        self.assertEqual(stopped, [True])

    def test_retreat_is_at_least_three_cm_and_arrival_precedes_open(self):
        events = []
        origin = CartesianPoseTarget((.2,.1,.3), (0.,0.,0.,1.))
        stream = SimpleNamespace(raise_if_failed=lambda: None,
                                 rebase_to_measured_pose=lambda: None,
                                 update_target=lambda target: events.append(('target',target)),
                                 wait_until_target=lambda _: True,
                                 stop=lambda: events.append(('stop',)))
        controller = SimpleNamespace(start_left_end_effector_position_stream=lambda *a,**kw: stream)
        with patch('left_grasp_gdk.measured_left_end_link_pose', return_value=origin), patch('left_grasp_gdk.wait_actual_left_target', side_effect=lambda *a,**kw: events.append(('arrived',))), patch('left_grasp_gdk.open_left_gripper_with_feedback', side_effect=lambda *a,**kw: events.append(('open',)) or {}):
            recover_for_visual_regrasp(controller,stream,retract_m=0.,translation_speed_mps=.01,fixed_orientation_xyzw=origin.orientation_xyzw,gripper_open_rad=-.7,gripper_tolerance_rad=.02,gripper_timeout_s=2.)
        targets = [e[1] for e in events if e[0] == 'target']
        self.assertAlmostEqual(targets[-1].position_m[2], .33)
        self.assertEqual(targets[-1].position_m[:2], origin.position_m[:2])
        self.assertLess(events.index(('arrived',)), events.index(('open',)))

    def test_failed_arrival_does_not_open(self):
        origin = CartesianPoseTarget((.2,.1,.3), (0.,0.,0.,1.))
        stream = SimpleNamespace(raise_if_failed=lambda: None,rebase_to_measured_pose=lambda: None,update_target=lambda target: None,wait_until_target=lambda _: True,stop=lambda: None)
        controller = SimpleNamespace(start_left_end_effector_position_stream=lambda *a,**kw: stream)
        with patch('left_grasp_gdk.measured_left_end_link_pose', return_value=origin), patch('left_grasp_gdk.wait_actual_left_target', side_effect=RuntimeError('not reached')), patch('left_grasp_gdk.open_left_gripper_with_feedback') as opening:
            with self.assertRaisesRegex(RuntimeError,'not reached'):
                recover_for_visual_regrasp(controller,stream,retract_m=.03,translation_speed_mps=.01,fixed_orientation_xyzw=origin.orientation_xyzw,gripper_open_rad=-.7,gripper_tolerance_rad=.02,gripper_timeout_s=2.)
            opening.assert_not_called()
