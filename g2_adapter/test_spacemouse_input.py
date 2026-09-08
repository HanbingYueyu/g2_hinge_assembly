import struct
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from spacemouse_input import (
    CompactHID,
    CompactReports,
    InputError,
    discover_devices,
    parse_axis_map,
    resolve_discovered_device,
)


def report(channel, *values):
    return bytes([channel]) + struct.pack('<hhh', *values)


class CompactInputTests(unittest.TestCase):
    def test_signed_axes_and_separate_report_clocks(self):
        reader = CompactReports()
        reader.feed(report(1, 350, -175, 0), 1.0)
        reader.feed(report(2, -350, 0, 175), 1.1)
        reader.feed(bytes([3, 0]), 1.2)
        frame = reader.snapshot()
        self.assertEqual(frame.axes, (1, -.5, 0, -1, 0, .5))
        self.assertEqual(frame.axis_times, (1.0, 1.1))
        self.assertTrue(frame.ready)

    def test_buttons_are_edges_and_first_held_report_is_not_a_click(self):
        reader = CompactReports()
        reader.feed(bytes([3, 2]), 1)
        self.assertEqual(reader.snapshot().pressed, ())
        reader.feed(bytes([3, 0]), 2)
        reader.feed(bytes([3, 2]), 3)
        reader.feed(bytes([3, 2]), 4)
        reader.feed(bytes([3, 0]), 5)
        self.assertEqual(reader.snapshot().pressed, (1,))
        self.assertEqual(reader.snapshot().pressed, ())

    def test_empty_poll_never_refreshes_axis_time(self):
        reader = CompactReports()
        reader.feed(report(1, 350, 0, 0), 1)
        reader.feed(bytes([3, 0]), 9)
        self.assertEqual(reader.snapshot().axis_times, (1, None))
        self.assertFalse(reader.snapshot().ready)

    def test_short_motion_report_is_rejected(self):
        with self.assertRaises(InputError):
            CompactReports().feed(bytes([1, 0]), 1)

    def test_axis_mapping_requires_signed_permutation_within_each_triplet(self):
        self.assertEqual(parse_axis_map('2,-1,3,5,-4,6'), (2, -1, 3, 5, -4, 6))
        for text in ('1,1,3,4,5,6', '1,2,3,4,5,0', '4,2,3,1,5,6'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_axis_map(text)

    def test_discovery_only_accepts_compact_usb_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, identity in [('hidraw0', '0003:0000256F:0000C635'), ('hidraw1', '0003:0000256F:0000C626')]:
                device = root / name / 'device'
                device.mkdir(parents=True)
                (device / 'uevent').write_text(f'HID_ID={identity}\n')
            self.assertEqual(discover_devices(root), ['/dev/hidraw0'])

    def test_stable_udev_symlink_resolves_to_discovered_hidraw_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hidraw = root / 'hidraw0'
            alias = root / 'spacemouse-compact'
            hidraw.touch()
            alias.symlink_to(hidraw)
            self.assertEqual(
                resolve_discovered_device(str(alias), [str(hidraw)]),
                str(hidraw),
            )

    def open_fake(self):
        # Only the hardware boundary is substituted; decoding and lifecycle
        # are the production implementation.
        stack = __import__('contextlib').ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch('spacemouse_input.discover_devices', return_value=['/dev/hidraw7']))
        opened = stack.enter_context(patch('spacemouse_input.os.open', return_value=7))
        stack.enter_context(patch('spacemouse_input.fcntl.flock'))
        closed = stack.enter_context(patch('spacemouse_input.os.close'))
        reader = CompactHID()
        self.addCleanup(reader.close)
        return reader, opened, closed

    def test_poll_drains_reports_and_does_not_stamp_empty_reads(self):
        reader, _, _ = self.open_fake()
        with patch('spacemouse_input.os.read', side_effect=[report(1, 350, 0, 0), report(2, 0, 0, 0), bytes([3, 0]), BlockingIOError()]), patch('spacemouse_input.time.monotonic', return_value=5):
            self.assertTrue(reader.poll().ready)
        with patch('spacemouse_input.os.read', side_effect=BlockingIOError()), patch('spacemouse_input.time.monotonic', return_value=10):
            frame = reader.poll()
            self.assertEqual(frame.axes[0], 1)
            self.assertEqual(frame.axis_times, (5, 5))

    def test_device_is_read_only_and_close_is_idempotent(self):
        import os
        reader, opened, closed = self.open_fake()
        flags = opened.call_args.args[1]
        self.assertEqual(flags & os.O_ACCMODE, os.O_RDONLY)
        self.assertTrue(flags & os.O_NONBLOCK)
        reader.close()
        reader.close()
        closed.assert_called_once_with(7)

    def test_disconnect_and_report_backlog_fail_closed(self):
        reader, _, _ = self.open_fake()
        for side_effect in (OSError('unplugged'), [b'']):
            with self.subTest(side_effect=side_effect), patch('spacemouse_input.os.read', side_effect=side_effect), self.assertRaises(InputError):
                reader.poll()
        with patch('spacemouse_input.os.read', return_value=bytes([3, 0])), self.assertRaises(InputError):
            reader.poll()


if __name__ == '__main__':
    unittest.main()
