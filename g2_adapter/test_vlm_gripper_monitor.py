import unittest
from unittest.mock import patch

from vlm_gripper_monitor import GripperMonitor, VlmRegraspRequested


class MonitorTests(unittest.TestCase):
    def test_unknown_and_network_failure_do_not_trigger_regrasp(self):
        monitor = GripperMonitor(None, None, lambda record: None)
        monitor.set_phase('post_grasp')
        with patch('vlm_gripper_monitor._snapshots', return_value=({'left': b'full'}, {})), patch('vlm_gripper_monitor.crop_jpeg', return_value=b'crop'), patch('vlm_gripper_monitor.run_gripper_check', side_effect=[{'ok': True, 'result': {'state': 'dropped'}}, {'ok': False}, {'ok': True, 'result': {'state': 'unknown'}}]):
            for _ in range(3):
                monitor.sample()
        monitor.raise_if_regrasp_requested()
        monitor.finish_before_release()
        self.assertEqual(monitor.phase, 'idle')

    def test_automatic_guard_is_not_run_in_background_writer(self):
        import threading
        from types import SimpleNamespace
        from control import G2Controller
        calls = []
        controller = SimpleNamespace(automatic_guard=lambda: calls.append('guard'), motion_gate=None)
        thread = threading.Thread(target=lambda: G2Controller.check_automatic_guard(controller))
        thread.start()
        thread.join()
        self.assertEqual(calls, [])
        G2Controller.check_automatic_guard(controller)
        self.assertEqual(calls, ['guard'])

    def test_mixed_failure_requests_regrasp_after_two_post_grasp_votes(self):
        monitor = GripperMonitor(None, None, lambda record: None)
        monitor.set_phase('post_grasp')
        with patch('vlm_gripper_monitor._snapshots', return_value=({'left': b'full'}, {})), patch('vlm_gripper_monitor.crop_jpeg', return_value=b'crop'), patch('vlm_gripper_monitor.run_gripper_check', side_effect=[{'ok': True, 'result': {'state': 'dropped'}}, {'ok': True, 'result': {'state': 'not_attached'}}]):
            monitor.sample()
            monitor.raise_if_regrasp_requested()
            monitor.sample()
            with self.assertRaises(VlmRegraspRequested):
                monitor.raise_if_regrasp_requested()
        monitor.set_phase('pre_grasp')
        monitor.raise_if_regrasp_requested()

    def test_pre_grasp_failure_does_not_request_regrasp(self):
        monitor = GripperMonitor(None, None, lambda record: None)
        monitor.set_phase('pre_grasp')
        with patch('vlm_gripper_monitor.run_gripper_check', return_value={'ok': True, 'result': {'state': 'dropped'}}):
            for _ in range(3):
                monitor.sample()
        monitor.raise_if_regrasp_requested()
    def test_pre_grasp_cached_frame_cannot_become_post_grasp_reference(self):
        monitor = GripperMonitor(None, None, lambda record: None)
        monitor.set_phase('post_grasp')
        with patch('vlm_gripper_monitor._snapshots', return_value=({'left': b'old'}, {'left': {'raw_age_s': .9}})), patch('vlm_gripper_monitor.run_gripper_check') as check:
            self.assertIsNone(monitor.sample())
            self.assertIsNone(monitor.reference)
            check.assert_not_called()

    def test_late_publication_of_pre_phase_read_is_not_a_reference(self):
        monitor = GripperMonitor(None, None, lambda record: None)
        monitor.set_phase('post_grasp')
        status = {'left': {'raw_age_s': 0., 'frame_read_started_s': monitor.phase_started - .1}}
        with patch('vlm_gripper_monitor._snapshots', return_value=({'left': b'old'}, status)), patch('vlm_gripper_monitor.run_gripper_check') as check:
            self.assertIsNone(monitor.sample())
            self.assertIsNone(monitor.reference)
            check.assert_not_called()

    def test_post_grasp_reference_is_history_not_automatic_success_and_votes_require_three_consecutive(self):
        monitor = GripperMonitor(None, None, lambda record: None)
        monitor.set_phase('post_grasp')
        with patch('vlm_gripper_monitor._snapshots', return_value=({'left': b'full'}, {})), patch('vlm_gripper_monitor.crop_jpeg', return_value=b'crop'), patch('vlm_gripper_monitor.run_gripper_check', return_value={'ok': True, 'result': {'state': 'attached'}}) as check:
            first = monitor.sample()
            self.assertEqual(first['stable_state'], 'unknown')
            self.assertIsNone(check.call_args.kwargs.get('reference'))
            monitor.sample()
            self.assertEqual(monitor.sample()['stable_state'], 'attached')
        monitor.set_phase('pre_grasp')
        self.assertIsNone(monitor.reference)
        self.assertEqual(list(monitor.votes), [])

    def test_phase_change_discards_inflight_result(self):
        events = []
        monitor = GripperMonitor(None, None, events.append)
        monitor.set_phase('pre_grasp')
        def response(*args, **kwargs):
            monitor.set_phase('post_grasp')
            return {'ok': True, 'result': {'state': 'dropped'}}
        with patch('vlm_gripper_monitor.run_gripper_check', side_effect=response):
            self.assertIsNone(monitor.sample())
        self.assertEqual(events, [])
