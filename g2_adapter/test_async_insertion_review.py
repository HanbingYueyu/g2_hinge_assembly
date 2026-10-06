import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import run_bilateral_flow_gdk as flow


class AsyncInsertionTests(unittest.TestCase):
    def test_review_runs_in_background_and_is_consumed_once(self):
        values = {}
        started, release = threading.Event(), threading.Event()
        class Services:
            web = object()
            def get(self, key):
                return values.get(key)
            def put(self, key, value):
                values[key] = value
        ctx = SimpleNamespace(args=SimpleNamespace(vlm_enable=True, disable_vlm_checks=False),
                              services=Services(), runtime=SimpleNamespace(emit_event=lambda _: None))
        ctx.services.controller = SimpleNamespace(check_automatic_guard=lambda: None)
        values['vlm_client'] = object()
        def review(*_args, **_kwargs):
            started.set()
            release.wait(2)
            return [{'task': 'convex-inserted', 'ok': True}]
        with patch('vlm_flow_checks.run_assembly_review', side_effect=review) as run:
            flow._start_insertion_review(ctx)
            self.assertTrue(started.wait(1))
            self.assertFalse(values['insertion_review_job'].done.is_set())
            release.set()
            result = flow.stage_vlm_assembly_check(ctx)
            self.assertTrue(result['ok'])
            self.assertEqual(run.call_count, 1)
            self.assertIsNone(values['insertion_review_job'])

    def test_disabled_vlm_does_not_start_job(self):
        ctx = SimpleNamespace(args=SimpleNamespace(vlm_enable=False, disable_vlm_checks=False))
        flow._start_insertion_review(ctx)

    def test_start_is_before_lift_and_ll2_motion(self):
        from unittest.mock import Mock
        events = []
        ctx = SimpleNamespace(
            args=SimpleNamespace(auto_speed_mm_s=30, auto_preset_velocity_rad_s=.2,
                                 preset_arrival_tolerance_rad=.01, preset_timeout_s=30),
            services=SimpleNamespace(get=lambda _: None, controller=object()),
            runtime=SimpleNamespace(
                open_left_gripper=lambda: (events.append('release') or {'event': 'released'}),
                emit_event=lambda _: None,
                move_left_presets=lambda *_args, **_kwargs: (events.append('LL2') or [{'preset': 'LL2'}]),
            ),
        )
        with patch('run_bilateral_flow_gdk._one_shot_gripper_check', return_value='attached'), \
             patch('run_bilateral_flow_gdk._start_insertion_review', side_effect=lambda _: events.append('review')), \
             patch('left_seat_motion.move_left_vertical', side_effect=lambda *_args, **_kwargs: (events.append('lift') or {'event': 'lift'})):
            flow.stage_left_release_lift_to_ll2(ctx)
        self.assertEqual(events, ['release', 'review', 'lift', 'LL2'])
