import unittest
from unittest.mock import Mock
from types import SimpleNamespace
from repeated_grasp_demo import DemoModeSelection, toggle_demo_grippers

class ToggleTests(unittest.TestCase):
    def test_zero_queued_after_active_round(self):
        selection=DemoModeSelection();selection.active='1'
        selection.choose('0')
        self.assertEqual(selection.active,'1')
        self.assertEqual(selection.next_round(),'0')
    def test_both_open_close(self):
        controller=Mock();controller.read_gripper_position.return_value=-.785
        ctx=SimpleNamespace(services=SimpleNamespace(controller=controller),runtime=Mock())
        toggle_demo_grippers(ctx)
        self.assertEqual(ctx.runtime._open_grippers.call_args.kwargs['open_rad'],0.)
    def test_mixed_state_opens_both(self):
        controller=Mock();controller.read_gripper_position.side_effect=[-.785,0.]
        ctx=SimpleNamespace(services=SimpleNamespace(controller=controller),runtime=Mock())
        toggle_demo_grippers(ctx)
        self.assertLess(ctx.runtime._open_grippers.call_args.kwargs['open_rad'],-.78)
    def test_missing_feedback_does_not_move(self):
        controller=Mock();controller.read_gripper_position.return_value=None
        ctx=SimpleNamespace(services=SimpleNamespace(controller=controller),runtime=Mock())
        with self.assertRaises(Exception):toggle_demo_grippers(ctx)
        ctx.runtime._open_grippers.assert_not_called()
