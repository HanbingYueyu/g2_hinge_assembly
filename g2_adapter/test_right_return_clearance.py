import unittest
from types import SimpleNamespace
from unittest.mock import patch

from bilateral_services import StageServices
from control import CartesianPoseTarget
import run_bilateral_flow_gdk as flow


class RightReturnClearanceTests(unittest.TestCase):
    def context(self, method, arrived=True):
        events = []
        origin = CartesianPoseTarget((.3, -.2, .5), (0., 0., 1., 0.))
        class Stream:
            def update_target(self, target):
                events.append(('target', target))
            def wait_until_actual_target(self, target, timeout, **kwargs):
                events.append(('arrived', target))
                return target if arrived else None
            def stop(self):
                events.append(('stop',))
        class Controller:
            def read_end_effector_pose(self, name):
                self.name = name
                return origin
            def start_right_end_effector_position_stream(self, current, **kwargs):
                events.append(('start', current, kwargs))
                return Stream()
        services = StageServices(controller=Controller())
        services.put('right_saved_grasp_pose', origin)
        services.put('right_config', {})
        runtime = SimpleNamespace(emit_event=lambda event: events.append(('log', event)))
        ctx = SimpleNamespace(services=services, runtime=runtime, args=SimpleNamespace(teach_enable=(method == 'teach'), auto_speed_mm_s=30))
        return ctx, events, origin

    def test_teach_lifts_thirty_mm_then_checks_screws_before_return(self):
        ctx, events, origin = self.context('teach')
        with patch('run_bilateral_flow_gdk._one_shot_four_screws_check', side_effect=lambda *a, **kw: events.append(('screws', kw)) or {'task': 'four-screws', 'ok': True}), \
             patch('right_apriltag_grasp_gdk.return_screwdriver', side_effect=lambda *a, **kw: events.append(('return', kw))):
            flow.stage_right_return(ctx)
        self.assertEqual([e[0] for e in events], ['start', 'target', 'arrived', 'stop', 'log', 'screws', 'return'])
        self.assertEqual(events[5][1]['phase'], 'before_right_return')
        target = events[1][1]
        self.assertEqual(target.position_m, (.3, -.2, .53))
        self.assertEqual(target.orientation_xyzw, origin.orientation_xyzw)
        self.assertEqual(events[0][2]['max_translation_velocity_mps'], .03)
        self.assertFalse(events[-1][1]['release'])

    def test_original_mode_has_no_extra_lift(self):
        ctx, events, _ = self.context('original')
        with patch('run_bilateral_flow_gdk._one_shot_four_screws_check', side_effect=lambda *a, **kw: events.append(('screws', kw)) or {'task': 'four-screws', 'ok': True}), \
             patch('right_apriltag_grasp_gdk.return_screwdriver', side_effect=lambda *a, **kw: events.append(('return',))):
            result = flow.stage_right_return(ctx)
        self.assertEqual([e[0] for e in events], ['screws', 'return'])
        self.assertEqual(result['four_screws']['task'], 'four-screws')

    def test_failed_lift_stops_stream_and_does_not_start_return(self):
        ctx, events, _ = self.context('teach', arrived=False)
        with patch('right_apriltag_grasp_gdk.return_screwdriver', side_effect=lambda *a, **kw: events.append(('return',))):
            with self.assertRaisesRegex(RuntimeError, '上抬'):
                flow.stage_right_return(ctx)
        self.assertEqual([e[0] for e in events], ['start', 'target', 'arrived', 'stop'])
