import contextlib
import io
import unittest
from unittest.mock import patch

from run_spacemouse_gdk import ensure_position_mode, main, run_loop
from spacemouse_input import InputError, InputFrame
from spacemouse_teleop import PreviewController, TeleopConfig, TeleopSession


class Clock:
    def __init__(self):
        self.now = 10.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class Source:
    path = 'offline-test'

    def __init__(self, clock, fail_at=None):
        self.clock, self.fail_at = clock, fail_at
        self.count = 0

    def poll(self):
        self.count += 1
        if self.count == self.fail_at:
            raise InputError('disconnected')
        axes = (0,) * 6 if self.count == 1 else (1, 0, 0, 0, 0, 0)
        return InputFrame(axes, (False, False), (), (self.clock(), self.clock()), True)


class CLITests(unittest.TestCase):
    def test_robot_start_switches_control_mode_3_to_position_mode_1(self):
        robot = __import__('test_control').FakeRobot()
        robot.control_mode = 3
        controller = __import__('control').G2Controller(__import__('test_control').FakeGdk, robot, allow_motion=True)
        ensure_position_mode(controller)
        self.assertEqual(robot.control_mode, 1)

    def test_robot_start_rejects_planning_mode_before_switching(self):
        robot = __import__('test_control').FakeRobot()
        robot.motion_mode = 2
        robot.control_mode = 3
        controller = __import__('control').G2Controller(__import__('test_control').FakeGdk, robot, allow_motion=True)
        with self.assertRaises(Exception):
            ensure_position_mode(controller)
        self.assertEqual(robot.control_mode, 3)

    def test_loop_runs_at_50hz_and_stops_at_duration(self):
        clock = Clock()
        source = Source(clock)
        robot = PreviewController()
        session = TeleopSession(robot, TeleopConfig(axis_map=(1, 2, 3, 4, 5, 6)), clock=clock)
        run_loop(source, session, duration_s=.1, clock=clock, sleep=clock.sleep, emit=lambda _: None)
        self.assertIn(source.count, (5, 6))
        self.assertGreater(robot.poses['right'].position_m[0], 0)
        self.assertLessEqual(robot.poses['right'].position_m[0], .0011)
        with self.assertRaises(Exception):
            session.tick(source.poll(), clock())

    def test_disconnect_stops_session_and_does_not_auto_reconnect(self):
        clock = Clock()
        source = Source(clock, fail_at=3)
        session = TeleopSession(PreviewController(), TeleopConfig(), clock=clock)
        with self.assertRaises(InputError):
            run_loop(source, session, clock=clock, sleep=clock.sleep, emit=lambda _: None)
        self.assertEqual(source.count, 3)
        with self.assertRaises(Exception):
            session.tick(source.poll(), clock())

    def test_list_devices_never_initializes_gdk(self):
        with patch.dict('sys.modules', {'agibot_gdk': None}), patch('run_spacemouse_gdk.discover_devices', return_value=['/dev/hidraw7']):
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                self.assertEqual(main(['--list-devices']), 0)
            self.assertIn('/dev/hidraw7', output.getvalue())

    def test_default_preview_never_initializes_gdk(self):
        class IdleSource:
            path = 'offline-test'
            def __init__(self, path):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def poll(self):
                return InputFrame((0,) * 6, (False, False), (), (0.0, 0.0), True)
        with patch.dict('sys.modules', {'agibot_gdk': None}), patch('run_spacemouse_gdk.CompactHID', IdleSource):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(['--duration-s', '.04']), 0)

    def test_invalid_parameters_fail_before_device_open(self):
        for args in (['--speed-mm-s', 'nan'], ['--duration-s', '-1'], ['--axis-map=1,1,3,4,5,6']):
            with self.subTest(args=args), patch('run_spacemouse_gdk.CompactHID', side_effect=AssertionError('must not open')):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as exc:
                    main(args)
                self.assertEqual(exc.exception.code, 2)

    def test_cumulative_workspace_options_are_removed(self):
        with self.assertRaises(SystemExit):
            __import__('run_spacemouse_gdk').parse_args(['--max-displacement-mm', '1'])


if __name__ == '__main__':
    unittest.main()
