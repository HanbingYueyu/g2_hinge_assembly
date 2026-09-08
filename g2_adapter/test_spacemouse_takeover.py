import threading
import time
import unittest

from spacemouse_input import InputFrame
from spacemouse_takeover import (
    AdvanceRightRequested,
    DOUBLE_CLICK_WINDOW_S,
    INACTIVITY_TIMEOUT_S,
    LONG_PRESS_S,
    RIGHT_LONG_PRESS_S,
    GestureResult,
    MotionPauseGate,
    SpaceMouseTakeoverWorker,
    SpaceMouseGestureState,
    TakeoverEvent,
)
from control import G2Controller
from spacemouse_teleop import TeleopConfig
from test_control import FakeGdk
from test_spacemouse_teleop import BilateralRobot


def frame(now, *, axes=(0.0,) * 6, buttons=(False, False), pressed=()):
    return InputFrame(axes, buttons, pressed, (now, now), True)


class GestureTests(unittest.TestCase):
    def test_left_double_click_requests_left_takeover(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.0), 1.0)
        state.update(frame(1.02, buttons=(True, False), pressed=(0,)), 1.02)
        state.update(frame(1.06), 1.06)
        result = state.update(frame(1.30, buttons=(True, False), pressed=(0,)), 1.30)
        result = state.update(frame(1.34), 1.34)
        self.assertEqual(result.events, (TakeoverEvent.TAKEOVER_LEFT,))
        self.assertIn(0, result.consumed_buttons)

    def test_right_double_click_requests_right_takeover(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.0), 1.0)
        state.update(frame(1.02, buttons=(False, True), pressed=(1,)), 1.02)
        state.update(frame(1.06), 1.06)
        state.update(frame(1.30, buttons=(False, True), pressed=(1,)), 1.30)
        result = state.update(frame(1.34), 1.34)
        self.assertEqual(result.events, (TakeoverEvent.TAKEOVER_RIGHT,))
        self.assertIn(1, result.consumed_buttons)

    def test_right_long_press_requests_advance_and_suppresses_gripper_click(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.0), 1.0)
        state.update(frame(1.02, buttons=(False, True), pressed=(1,)), 1.02)
        result = state.update(frame(1.02 + RIGHT_LONG_PRESS_S + 0.01, buttons=(False, True)), 1.03 + RIGHT_LONG_PRESS_S)
        self.assertEqual(result.events, (TakeoverEvent.ADVANCE_RIGHT,))
        self.assertIn(1, result.consumed_buttons)
        released = state.update(frame(3.1), 3.1)
        self.assertEqual(released.events, ())

    def test_click_after_window_is_two_single_clicks(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.0), 1.0)
        state.update(frame(1.02, buttons=(True, False), pressed=(0,)), 1.02)
        state.update(frame(1.06), 1.06)
        state.update(frame(1.06 + DOUBLE_CLICK_WINDOW_S + 0.1), 1.06 + DOUBLE_CLICK_WINDOW_S + 0.1)
        state.update(frame(1.50, buttons=(True, False), pressed=(0,)), 1.50)
        result = state.update(frame(1.54), 1.54)
        self.assertEqual(result.events, ())

    def test_activity_is_reported_for_axes_and_buttons(self):
        state = SpaceMouseGestureState()
        result = state.update(frame(1.0, axes=(0.2, 0, 0, 0, 0, 0)), 1.0)
        self.assertTrue(result.activity)
        result = state.update(frame(1.02), 1.02)
        self.assertFalse(result.activity)


class MotionPauseGateTests(unittest.TestCase):
    def test_pause_waits_for_an_inflight_automatic_write_before_manual_access(self):
        gate = MotionPauseGate()
        entered = threading.Event()
        release = threading.Event()

        def automatic():
            with gate.automatic_motion():
                entered.set()
                release.wait(1.0)

        thread = threading.Thread(target=automatic)
        thread.start()
        self.assertTrue(entered.wait(1.0))
        pausing = threading.Thread(target=gate.pause)
        pausing.start()
        time.sleep(0.02)
        self.assertTrue(pausing.is_alive())
        release.set()
        pausing.join(1.0)
        thread.join(1.0)
        self.assertFalse(pausing.is_alive())
        self.assertTrue(gate.is_paused())
        gate.resume()

    def test_automatic_waits_until_resume_but_manual_context_bypasses(self):
        gate = MotionPauseGate()
        gate.pause()
        entered = threading.Event()
        released = threading.Event()

        def automatic():
            entered.set()
            gate.wait_for_auto()
            released.set()

        thread = threading.Thread(target=automatic)
        thread.start()
        self.assertTrue(entered.wait(1.0))
        self.assertFalse(released.wait(0.03))
        with gate.manual_motion():
            gate.wait_for_auto()
        gate.resume()
        self.assertTrue(released.wait(1.0))
        thread.join(1.0)

    def test_abort_wakes_automatic_waiters_with_error(self):
        gate = MotionPauseGate()
        gate.pause()
        errors = []

        def automatic():
            try:
                gate.wait_for_auto()
            except Exception as exc:  # noqa: BLE001 - assert the propagated gate error
                errors.append(exc)

        thread = threading.Thread(target=automatic)
        thread.start()
        time.sleep(0.02)
        gate.abort(RuntimeError("input failed"))
        thread.join(1.0)
        self.assertEqual(len(errors), 1)
        self.assertIn("input failed", str(errors[0]))

    def test_advance_wakes_automatic_waiters_with_stage_transition(self):
        gate = MotionPauseGate()
        gate.pause()
        errors = []

        def automatic():
            try:
                gate.wait_for_auto()
            except Exception as exc:  # noqa: BLE001 - assert the transition signal
                errors.append(exc)

        thread = threading.Thread(target=automatic)
        thread.start()
        time.sleep(0.02)
        gate.request_advance()
        thread.join(1.0)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], AdvanceRightRequested)
        self.assertTrue(gate.is_paused())


class WorkerTests(unittest.TestCase):
    def test_worker_samples_clock_after_hid_poll_before_takeover_tick(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)

        class Clock:
            def __init__(self):
                self.now = 0.0

            def __call__(self):
                return self.now

        clock = Clock()
        frames = [
            frame(0.02),
            frame(0.04, buttons=(True, False), pressed=(0,)),
            frame(0.06),
            frame(0.08, buttons=(True, False), pressed=(0,)),
            frame(0.10),
            frame(0.12),
        ]

        class FakeHID:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def poll(self):
                # HID report timestamps are produced during poll, after the
                # loop's previous clock sample.
                clock.now += 0.02
                result = frames.pop(0)
                if not frames:
                    worker._stop.set()
                return InputFrame(
                    result.axes,
                    result.buttons,
                    result.pressed,
                    (clock.now, clock.now),
                    result.ready,
                    result.peak_axes,
                )

        worker = SpaceMouseTakeoverWorker(
            controller,
            gate,
            device="/dev/hidraw-test",
            config=TeleopConfig(side="left", axis_map=(1, 2, 3, 4, 5, 6)),
            clock=clock,
            hid_factory=lambda _device: FakeHID(),
        )
        worker._run()
        self.assertIsNone(worker._failed)
        self.assertEqual(worker.active_side, "left")
        worker.stop()

    def test_double_click_activates_requested_side_until_explicit_second_double_click(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        worker = SpaceMouseTakeoverWorker(
            controller,
            gate,
            device=None,
            config=TeleopConfig(side="right", axis_map=(1, 2, 3, 4, 5, 6)),
            clock=lambda: 1.34,
        )
        worker.step(frame(1.0), 1.0)
        worker.step(frame(1.02, buttons=(True, False), pressed=(0,)), 1.02)
        worker.step(frame(1.06), 1.06)
        worker.step(frame(1.30, buttons=(True, False), pressed=(0,)), 1.30)
        result = worker.step(frame(1.34), 1.34)
        self.assertEqual(result.events, (TakeoverEvent.TAKEOVER_LEFT,))
        self.assertEqual(worker.active_side, "left")
        self.assertTrue(gate.is_paused())

        # Idleness no longer releases manual control.
        for index in range(1, int(INACTIVITY_TIMEOUT_S / 0.02) + 3):
            now = 1.34 + index * 0.02
            worker.step(frame(now), now)
        self.assertEqual(worker.active_side, "left")
        self.assertTrue(gate.is_paused())

        # The active hand's second double-click is the explicit release.
        worker.step(frame(4.40, buttons=(True, False), pressed=(0,)), 4.40)
        worker.step(frame(4.42), 4.42)
        worker.step(frame(4.46, buttons=(True, False), pressed=(0,)), 4.46)
        worker.step(frame(4.48), 4.48)
        self.assertIsNone(worker.active_side)
        self.assertFalse(gate.is_paused())

    def test_left_stage_right_long_press_requests_advance_without_resuming_gate(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        worker = SpaceMouseTakeoverWorker(
            controller,
            gate,
            device=None,
            config=TeleopConfig(side="left", axis_map=(1, 2, 3, 4, 5, 6)),
            allow_advance_right=True,
            clock=lambda: 1.02,
        )
        worker.step(frame(1.0), 1.0)
        worker.step(frame(1.02, buttons=(False, True), pressed=(1,)), 1.02)
        result = worker.step(
            frame(1.02 + RIGHT_LONG_PRESS_S + 0.01, buttons=(False, True)),
            1.02 + RIGHT_LONG_PRESS_S + 0.01,
        )
        self.assertEqual(result.events, (TakeoverEvent.ADVANCE_RIGHT,))
        self.assertTrue(worker.advance_requested)
        self.assertTrue(gate.is_paused())


if __name__ == "__main__":
    unittest.main()
