import os
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import repeated_grasp_demo as demo
from repeated_grasp_demo import DemoKeyboard, DemoModeSelection, check_left_return_arrival, run_demo_round


class DemoModeSelectionTests(unittest.TestCase):
    def test_switch_is_consumed_only_between_complete_rounds(self):
        selection = DemoModeSelection()
        self.assertEqual(selection.choose('1'), '抓取铰链')
        self.assertEqual(selection.wait_first(), '1')
        self.assertEqual(selection.active, '1')
        self.assertEqual(selection.choose('2'), '抓取铰链并移动')
        self.assertEqual(selection.active, '1')
        self.assertEqual(selection.next_round(), '2')
        self.assertEqual(selection.active, '2')
        self.assertEqual(selection.next_round(), '2')

    def test_stop_is_deferred_until_round_end(self):
        selection = DemoModeSelection()
        selection.choose('3')
        self.assertEqual(selection.wait_first(), '3')
        self.assertEqual(selection.choose('q'), '完成当前轮后返回选择界面')
        self.assertEqual(selection.active, '3')
        self.assertEqual(selection.pending, 'q')
        self.assertIsNone(selection.next_round())
        self.assertIsNone(selection.active)
        selection.choose('2')
        self.assertEqual(selection.wait_first(), '2')

    def test_keyboard_queues_switch_while_round_is_active(self):
        read_fd, write_fd = os.pipe()
        stream = os.fdopen(read_fd, 'rb', buffering=0)
        selection = DemoModeSelection()
        keyboard = DemoKeyboard(selection, stream=stream)
        try:
            keyboard.start()
            os.write(write_fd, b'1\n')
            self.assertEqual(selection.wait_first(), '1')
            os.write(write_fd, b'3\n')
            deadline = time.monotonic() + 1.0
            while selection.pending != '3' and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(selection.active, '1')
            self.assertEqual(selection.next_round(), '3')
        finally:
            keyboard.stop()
            os.close(write_fd)
            stream.close()

    def test_invalid_choice_does_not_replace_pending_mode(self):
        selection = DemoModeSelection()
        selection.choose('1')
        self.assertEqual(selection.wait_first(), '1')
        selection.choose('3')
        with self.assertRaises(ValueError):
            selection.choose('7')
        self.assertEqual(selection.next_round(), '3')

    def test_four_selects_screwdriver_transport_for_next_round(self):
        selection = DemoModeSelection()
        selection.choose('3')
        self.assertEqual(selection.wait_first(), '3')
        self.assertEqual(selection.choose('4'), '抓取电批并移动')
        self.assertEqual(selection.active, '3')
        self.assertEqual(selection.next_round(), '4')


class LeftReturnTests(unittest.TestCase):
    def test_early_force_contact_does_not_release_hinge(self):
        with self.assertRaisesRegex(Exception, '提前接触'):
            check_left_return_arrival(
                saved_position=(0.4, 0.1, 0.6),
                actual_position=(0.4, 0.1, 0.68),
                descent={'stop_reason': 'force_threshold', 'descent_mm': 20.0},
            )

    def test_arrival_near_original_pickup_allows_release(self):
        check_left_return_arrival(
            saved_position=(0.4, 0.1, 0.6),
            actual_position=(0.402, 0.1, 0.604),
            descent={'stop_reason': 'force_threshold', 'descent_mm': 96.0},
        )


class DemoRoundTests(unittest.TestCase):
    def test_left_pick_waits_for_confirmation_before_round_can_return(self):
        events = []
        with patch('run_bilateral_flow_gdk._confirm_left_grasp', side_effect=lambda _: events.append('confirmed')):
            result = demo._pick_left_with_confirmation(object(), lambda _: (events.append('picked') or {'ok': True}))
        self.assertEqual(events, ['picked', 'confirmed'])
        self.assertEqual(result, {'ok': True})


    def test_hinge_only_returns_without_hole_transport(self):
        events = []
        actions = {name: (lambda name=name: events.append(name)) for name in (
            'head', 'require_hole', 'left_pick', 'to_hole', 'left_return',
        )}
        run_demo_round('1', actions)
        self.assertEqual(events, ['head', 'left_pick', 'left_return'])

    def test_hinge_move_returns_to_saved_pickup_before_release(self):
        events = []
        actions = {name: (lambda name=name: events.append(name)) for name in (
            'head', 'require_hole', 'left_pick', 'to_hole', 'left_return', 'right_pick', 'right_return',
        )}
        run_demo_round('2', actions)
        self.assertEqual(events, ['head', 'require_hole', 'left_pick', 'to_hole', 'left_return'])

    def test_screwdriver_lifts_and_returns_without_hole_transport(self):
        events = []
        actions = {name: (lambda name=name: events.append(name)) for name in (
            'head', 'require_hole', 'left_pick', 'to_hole', 'left_return',
            'right_pick', 'right_vlm', 'right_lift', 'right_return',
        )}
        run_demo_round('3', actions)
        self.assertEqual(events, ['right_pick', 'right_vlm', 'right_return'])

    def test_screwdriver_transport_locates_hole_before_pick_and_returns_after_arrival(self):
        events = []
        actions = {name: (lambda name=name: events.append(name)) for name in (
            'head', 'require_hole', 'right_pick', 'right_vlm', 'right_lift',
            'right_to_hole', 'right_return',
        )}
        run_demo_round('4', actions)
        self.assertEqual(events, [
            'head', 'require_hole', 'right_pick', 'right_vlm', 'right_lift',
            'right_to_hole', 'right_return',
        ])

    def test_screwdriver_hole_transport_uses_half_joint_speed_without_second_lift(self):
        saved = object()
        workpiece = np.eye(4)
        services = Mock()
        services.get.side_effect = lambda key: {
            'right_saved_grasp_pose': saved,
            'right_saved_workpiece_pose': workpiece,
            'head_targets': {'hole': {'position_base_link_m': [.5, .2, .8]}},
        }.get(key)
        args = SimpleNamespace(
            right_hole_x_mm=0, right_hole_y_mm=60, right_hole_z_mm=100,
            right_hole_yaw_deg=55, right_hole_pitch_deg=0, right_hole_roll_deg=0,
            auto_speed_mm_s=30, auto_preset_velocity_rad_s=.2,
        )
        ctx = SimpleNamespace(services=services, args=args, runtime=Mock())
        with patch('right_hole_motion.move_right_tool_above_hole', return_value={'event': 'right_tool_above_hole_arrived'}) as transport:
            demo._move_right_to_hole(ctx)
        self.assertEqual(transport.call_args.args[1:4], (saved, workpiece, [.5, .2, .8]))
        self.assertEqual(transport.call_args.kwargs['joint_velocity_rad_s'], .1)
        self.assertFalse(transport.call_args.kwargs['lift_before_move'])
        ctx.runtime.emit_event.assert_called_once_with({'event': 'right_tool_above_hole_arrived'})


class ReturnMotionTests(unittest.TestCase):
    def test_left_return_uses_configured_speed(self):
        import tempfile
        from pathlib import Path
        from control import CartesianPoseTarget
        saved = CartesianPoseTarget((.4, .1, .7), (0, 0, 0, 1))
        upper = CartesianPoseTarget((.4, .1, .8), saved.orientation_xyzw)
        services = Mock()
        services.get.side_effect = lambda key: saved if key == 'left_demo_grasp_pose' else None
        services.controller.read_end_effector_pose.side_effect = [upper, upper, saved]
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'left.json'
            config.write_text('{"contact":{"force_delta_threshold_n":3}}')
            ctx = SimpleNamespace(services=services, runtime=Mock(), args=SimpleNamespace(
                left_config=config, auto_speed_mm_s=30, auto_preset_velocity_rad_s=.2))
            with patch('left_hole_joint_move.move_left_hole_joint_target'), \
                 patch('left_seat_motion.force_guarded_left_descent', return_value={'stop_reason':'max_distance'}) as descent, \
                 patch('left_seat_motion.move_left_vertical', return_value={}):
                demo._move_left_to_saved_pose(ctx)
            self.assertEqual(descent.call_args.kwargs['speed_mm_s'], 30)
            ctx.runtime.open_left_gripper.assert_called_once()
