import os
import time
import unittest

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
        selection.choose('q')
        self.assertEqual(selection.active, '3')
        self.assertIsNone(selection.next_round())

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
            selection.choose('4')
        self.assertEqual(selection.next_round(), '3')


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
        self.assertEqual(events, ['right_pick', 'right_vlm', 'right_lift', 'right_return'])
