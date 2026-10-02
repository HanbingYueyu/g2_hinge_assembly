import unittest
import contextlib
import io
import json
from unittest.mock import patch
from types import SimpleNamespace

from control import CartesianPoseTarget
from left_grasp_gdk import recover_for_visual_regrasp, lift_before_visual_reacquire
from move_left_above_head_hinge_gdk import LeftTargetArrivalTimeout
from safety import SafetyError


class VisualRetryTests(unittest.TestCase):
    def test_completed_visual_reacquire_lift_emits_numbered_tts_event(self):
        origin = CartesianPoseTarget((.2, .1, .3), (0., 0., 0., 1.))
        stream = SimpleNamespace(update_target=lambda target: None, stop=lambda: None)
        controller = SimpleNamespace(start_left_end_effector_position_stream=lambda *args, **kwargs: stream)
        output = io.StringIO()
        with patch('left_grasp_gdk.measured_left_end_link_pose', return_value=origin), \
             patch('left_grasp_gdk.wait_actual_left_target'), contextlib.redirect_stdout(output):
            lift_before_visual_reacquire(controller, speed_mm_s=30., regrasp_attempt=2)
        records = [json.loads(line) for line in output.getvalue().splitlines() if line.startswith('{')]
        self.assertEqual(records, [{'event': 'left_visual_regrasp_requested', 'regrasp_attempt': 2}])

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
        stopped = []
        stream = SimpleNamespace(raise_if_failed=lambda: None,rebase_to_measured_pose=lambda: None,update_target=lambda target: None,wait_until_target=lambda _: True,stop=lambda: stopped.append(True))
        controller = SimpleNamespace(start_left_end_effector_position_stream=lambda *a,**kw: stream)
        with patch('left_grasp_gdk.measured_left_end_link_pose', return_value=origin), patch('left_grasp_gdk.wait_actual_left_target', side_effect=LeftTargetArrivalTimeout('not reached')) as wait, patch('left_grasp_gdk.open_left_gripper_with_feedback') as opening:
            with self.assertRaisesRegex(RuntimeError,'not reached'):
                recover_for_visual_regrasp(controller,stream,retract_m=.03,translation_speed_mps=.01,fixed_orientation_xyzw=origin.orientation_xyzw,gripper_open_rad=-.7,gripper_tolerance_rad=.02,gripper_timeout_s=2.)
            opening.assert_not_called()
        self.assertEqual(wait.call_count, 2)
        self.assertEqual(len(stopped), 2)

    def test_retract_timeout_retries_from_measured_pose_before_opening(self):
        origin = CartesianPoseTarget((.2, .1, .3), (0., 0., 0., 1.))
        stalled = CartesianPoseTarget((.201, .101, .301), origin.orientation_xyzw)
        events = []
        first = SimpleNamespace(
            raise_if_failed=lambda: None, rebase_to_measured_pose=lambda: None,
            update_target=lambda target: events.append(('first_target', target)),
            stop=lambda: events.append(('first_stop',)),
        )
        retry = SimpleNamespace(
            update_target=lambda target: events.append(('retry_target', target)),
            stop=lambda: events.append(('retry_stop',)),
        )
        resumed = SimpleNamespace()
        streams = iter((retry, resumed))
        controller = SimpleNamespace(start_left_end_effector_position_stream=lambda initial, **kw:
                                     events.append(('start', initial)) or next(streams))
        poses = iter((origin, stalled, stalled))
        def wait(_controller, stream, target, **kwargs):
            events.append(('wait', stream, target))
            if stream is first:
                raise LeftTargetArrivalTimeout('not reached')
        with patch('left_grasp_gdk.measured_left_end_link_pose', side_effect=lambda _: next(poses)), \
             patch('left_grasp_gdk.wait_actual_left_target', side_effect=wait), \
             patch('left_grasp_gdk.open_left_gripper_with_feedback', side_effect=lambda *a, **kw:
                   events.append(('open',)) or {}):
            recover_for_visual_regrasp(
                controller, first, retract_m=.03, translation_speed_mps=.01,
                fixed_orientation_xyzw=origin.orientation_xyzw,
                gripper_open_rad=-.7, gripper_tolerance_rad=.02, gripper_timeout_s=2.,
            )
        self.assertLess(events.index(('first_stop',)), events.index(('start', stalled)))
        retry_target = next(event[1] for event in events if event[0] == 'retry_target')
        self.assertEqual(retry_target.position_m[:2], stalled.position_m[:2])
        self.assertAlmostEqual(retry_target.position_m[2], .33)
        self.assertLess(events.index(('retry_stop',)), events.index(('open',)))

    def test_stream_safety_failure_stops_without_retry_or_open(self):
        origin = CartesianPoseTarget((.2, .1, .3), (0., 0., 0., 1.))
        stopped = []
        stream = SimpleNamespace(
            raise_if_failed=lambda: None, rebase_to_measured_pose=lambda: None,
            update_target=lambda _: None, stop=lambda: stopped.append(True),
        )
        controller = SimpleNamespace(start_left_end_effector_position_stream=lambda *a, **kw:
                                     self.fail('must not restart after a safety fault'))
        with patch('left_grasp_gdk.measured_left_end_link_pose', return_value=origin), \
             patch('left_grasp_gdk.wait_actual_left_target', side_effect=SafetyError('stream fault')), \
             patch('left_grasp_gdk.open_left_gripper_with_feedback') as opening:
            with self.assertRaisesRegex(SafetyError, 'stream fault'):
                recover_for_visual_regrasp(
                    controller, stream, retract_m=.03, translation_speed_mps=.01,
                    fixed_orientation_xyzw=origin.orientation_xyzw,
                    gripper_open_rad=-.7, gripper_tolerance_rad=.02, gripper_timeout_s=2.,
                )
            opening.assert_not_called()
        self.assertEqual(stopped, [True])
