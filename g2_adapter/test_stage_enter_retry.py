import unittest
import os
import threading
import time
import tempfile
import json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

from dianpi_runner import DianpiCycleRunner
from left_teach_placement import end_on_takeover
from safety import AdvanceRightRequested
from spacemouse_takeover import MotionPauseGate, SpaceMouseTakeoverWorker
from spacemouse_teleop import TeleopConfig
from test_spacemouse_takeover import frame
from keyboard_stage_advance import KeyboardStageAdvance
import keyboard_stage_advance
from left_teach_placement import replay_left_placement
from right_teach_placement import replay_right_placement
from teach_trajectory import build_trajectory, trajectory_document
from test_teach_trajectory_gdk import FakeGdk, FakeRobot


class StageEnterRetryTests(unittest.TestCase):
    def test_enter_can_confirm_when_spacemouse_is_disconnected(self):
        attempted = threading.Event()
        def disconnected(device):
            attempted.set()
            raise OSError('HID unplugged')
        gate = MotionPauseGate()
        worker = SpaceMouseTakeoverWorker(None, gate, device='unused', config=TeleopConfig(side='left'), allow_advance_right=True, hid_factory=disconnected)
        worker.start()
        try:
            self.assertTrue(attempted.wait(1))
            worker.request_keyboard_advance()
            deadline = time.monotonic() + 1.5
            while not worker.advance_requested and time.monotonic() < deadline:
                time.sleep(.01)
            self.assertTrue(worker.advance_requested)
        finally:
            worker.stop()

    def test_enter_during_real_replay_stops_both_arms_without_endpoint_wait(self):
        for side, replay in [('left', replay_left_placement), ('right', replay_right_placement)]:
            with self.subTest(side=side), tempfile.TemporaryDirectory() as directory:
                gate = MotionPauseGate()
                worker = SpaceMouseTakeoverWorker(None, gate, device='unused', config=TeleopConfig(side=side))
                class Robot(FakeRobot):
                    def move_arm_joint_servo(self, command, *args):
                        self.servo_commands.append(command)
                        worker.request_keyboard_advance()
                        worker.step(frame(1), 1)
                robot = Robot()
                path = Path(directory) / 'demo.json'
                path.write_text(json.dumps(trajectory_document(build_trajectory(side, [(0, [0]*7), (100_000_000, [.1]*7)]))))
                args = SimpleNamespace(**{f'{side}_teach_trajectory': path, f'{side}_teach_speed_scale': .5, f'{side}_teach_start_velocity_rad_s': .05})
                with patch('sys.stdout'):
                    result = replay(SimpleNamespace(robot=robot, gdk=FakeGdk(), gate=gate, worker=worker), args)
                self.assertEqual(len(robot.servo_commands), 1)
                self.assertTrue(result['interrupted_by_advance'])
                self.assertTrue(worker.advance_requested)
                self.assertFalse(worker.allow_advance_right)
                worker.consume_advance()
                gate.wait_for_auto()

    def test_terminal_enter_requests_confirmation_but_other_text_does_not(self):
        read_fd, write_fd = os.pipe()
        received = threading.Event()
        worker = SimpleNamespace(request_keyboard_advance=received.set)
        with os.fdopen(read_fd, 'r') as stream:
            listener = KeyboardStageAdvance(worker, stream)
            listener.start()
            try:
                os.write(write_fd, b'not enter\n')
                self.assertFalse(received.wait(.15))
                os.write(write_fd, b'\n')
                self.assertTrue(received.wait(1))
            finally:
                listener.stop()
                os.close(write_fd)

    def test_terminal_space_requests_dianpi_immediately(self):
        read_fd, write_fd = os.pipe()
        received = threading.Event()
        worker = SimpleNamespace(request_keyboard_dianpi=received.set)
        with os.fdopen(read_fd, 'r') as stream:
            with patch.object(keyboard_stage_advance.termios, 'tcgetattr', return_value=[]), \
                 patch.object(keyboard_stage_advance.tty, 'setcbreak'):
                listener = KeyboardStageAdvance(worker, stream)
                listener.start()
                try:
                    os.write(write_fd, b' ')
                    self.assertTrue(received.wait(1))
                finally:
                    listener.stop()
                    os.close(write_fd)

    def test_replay_enables_confirmation_only_inside_segment(self):
        worker = SimpleNamespace(allow_advance_right=False)
        with end_on_takeover({}, SimpleNamespace(sent=0), '左手', worker):
            self.assertTrue(worker.allow_advance_right)
        self.assertFalse(worker.allow_advance_right)

    def test_read_failure_reopens_connection_without_replaying_cycle(self):
        controllers = []
        cycles = []
        class Stop:
            stopped = False
            def is_set(self):
                return self.stopped
            def wait(self, duration):
                if len(controllers) == 2:
                    self.stopped = True
        class Controller:
            closed = False
            def open(self):
                pass
            def read_inputs(self):
                return None if self is controllers[0] else {10: 0}
            def close(self):
                self.closed = True
            def run_cycle(self):
                raise AssertionError('reconnect must not replay a cycle')
        def factory():
            controller = Controller()
            controllers.append(controller)
            return controller
        runner = DianpiCycleRunner(controller_factory=factory, event_sink=lambda e: None)
        runner.trigger = lambda: cycles.append('cycle')
        runner._monitor_stop = Stop()
        runner._monitor_loop(.05)
        self.assertEqual(len(controllers), 2)
        self.assertTrue(controllers[0].closed)
        self.assertFalse(controllers[1].closed)
        self.assertEqual(cycles, [], '重连后持续低电平不能自动重跑周期')

    def test_failed_connection_waits_five_seconds_before_second_attempt(self):
        attempts = []
        waits = []
        class Stop:
            stopped = False
            def is_set(self):
                return self.stopped
            def wait(self, duration):
                waits.append(duration)
                self.stopped = len(attempts) >= 2
        def factory():
            attempts.append(len(waits))
            raise OSError('not connected')
        runner = DianpiCycleRunner(controller_factory=factory, event_sink=lambda e: None)
        runner._monitor_stop = Stop()
        runner._monitor_loop(.05)
        self.assertEqual(attempts, [0, 1])
        self.assertEqual(waits, [5.0, 5.0])

    def test_enter_is_consumed_once_and_not_queued_from_other_stages(self):
        gate = MotionPauseGate()
        worker = SpaceMouseTakeoverWorker(None, gate, device='unused', config=TeleopConfig(side='left'))
        worker.request_keyboard_advance()
        worker.allow_advance_right = True
        worker.step(frame(1), 1)
        self.assertFalse(worker.advance_requested)
        worker.request_keyboard_advance()
        worker.step(frame(2), 2)
        self.assertTrue(worker.advance_requested)
        worker.consume_advance()
        worker.step(frame(3), 3)
        self.assertFalse(worker.advance_requested)
        gate.wait_for_auto()

    def test_replay_advance_ends_segment_and_keeps_confirmation_for_next_gate(self):
        gate = MotionPauseGate()
        generation = gate.pause_generation
        gate.request_advance()
        result = {}
        with end_on_takeover(result, SimpleNamespace(sent=4), '右手'):
            with gate.automatic_teach_motion(generation):
                self.fail('must not write another target')
        self.assertTrue(result['interrupted_by_advance'])
        with self.assertRaises(AdvanceRightRequested):
            gate.wait_for_auto()
        gate.consume_advance()
        gate.wait_for_auto()
