import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import repeated_grasp_demo as demo
from spacemouse_takeover import MotionPauseGate


class EscapeTests(unittest.TestCase):
    def test_escape_cancels_active_round_and_reset_allows_new_selection(self):
        selection = demo.DemoModeSelection()
        selection.choose('6')
        selection.wait_first()
        selection.choose('esc')
        with self.assertRaises(demo.DemoCancelled):
            selection.check_cancelled()
        selection.return_to_menu()
        selection.check_cancelled()
        selection.choose('3')
        self.assertEqual(selection.wait_first(), '3')

    def test_q_remains_deferred(self):
        selection = demo.DemoModeSelection()
        selection.choose('1')
        selection.wait_first()
        selection.choose('q')
        selection.check_cancelled()
        self.assertIsNone(selection.next_round())

    def test_cancel_wakes_paused_motion_waiter(self):
        selection = demo.DemoModeSelection()
        selection.choose('1')
        selection.wait_first()
        gate = MotionPauseGate()
        gate.cancel_guard = selection.check_cancelled
        gate.pause()
        errors = []
        def wait():
            try:
                gate.wait_for_auto()
            except demo.DemoCancelled:
                errors.append('cancelled')
        thread = threading.Thread(target=wait)
        thread.start()
        selection.choose('esc')
        thread.join(timeout=.5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ['cancelled'])
        self.assertTrue(gate.is_paused())

    def test_active_action_unwinds_and_returns_to_menu_without_home(self):
        selection = demo.DemoModeSelection()
        selection.choose('6')
        values = {}
        class Services:
            controller = SimpleNamespace(motion_gate=None)
            def get(self, name): return values.get(name)
            def put(self, name, value): values[name] = value
        ctx = SimpleNamespace(services=Services(), runtime=SimpleNamespace(emit=Mock()))
        original_wait = selection.wait_first
        waits = []
        def wait():
            waits.append(1)
            if len(waits) == 2:
                self.assertIsNone(selection.active)
                selection.choose('q')
            return original_wait()
        def action(*_args):
            selection.choose('esc')
            ctx.services.controller.demo_cancel_guard()
        with patch.object(selection, 'wait_first', side_effect=wait), \
             patch('repeated_grasp_demo.run_demo_round', side_effect=action), \
             patch('repeated_grasp_demo._stop_cancelled_round') as stop, \
             patch('run_bilateral_flow_gdk.stage_home') as home:
            self.assertEqual(demo.run_repeated_demo(ctx, selection), 0)
        stop.assert_called_once_with(ctx)
        home.assert_not_called()
        self.assertEqual(len(waits), 2)
        self.assertIsNone(ctx.services.controller.demo_cancel_guard)

    def test_escape_interrupts_waiting_vlm_request(self):
        selection = demo.DemoModeSelection()
        selection.choose('6')
        selection.wait_first()
        started, release = threading.Event(), threading.Event()
        def request():
            started.set()
            release.wait(2)
            return 'old result'
        client = demo._CancelableVlmClient(SimpleNamespace(run_bytes=request), selection)
        errors = []
        def call():
            try:
                client.run_bytes()
            except demo.DemoCancelled:
                errors.append('cancelled')
        thread = threading.Thread(target=call)
        thread.start()
        self.assertTrue(started.wait(.5))
        selection.choose('esc')
        thread.join(.5)
        release.set()
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, ['cancelled'])

    def test_stop_tool_even_when_arm_hold_fails(self):
        from contextlib import nullcontext
        controller = SimpleNamespace(
            demo_cancel_guard=Mock(), manual_motion=nullcontext,
            prepare_joint_impedance_control=Mock(side_effect=RuntimeError('hold failed')))
        runner = SimpleNamespace(stop_current_cycle=Mock())
        services = SimpleNamespace(get=lambda name: None, controller=controller, dianpi_runner=runner)
        old_guard = controller.demo_cancel_guard
        with self.assertRaises(RuntimeError):
            demo._stop_cancelled_round(SimpleNamespace(services=services))
        runner.stop_current_cycle.assert_called_once()
        self.assertIs(controller.demo_cancel_guard, old_guard)
