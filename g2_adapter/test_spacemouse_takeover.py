import threading
import time
import unittest

from spacemouse_input import InputError, InputFrame
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
from control import G2Controller, STANDARD_JOINT_IMPEDANCE_KP
from safety import SafetyError
from spacemouse_teleop import TeleopConfig, TransientGDKReadTimeout
from dianpi_runner import DianpiTriggerResult
from test_control import FakeGdk
from test_spacemouse_teleop import BilateralRobot


def frame(now, *, axes=(0.0,) * 6, buttons=(False, False), pressed=()):
    return InputFrame(axes, buttons, pressed, (now, now), True)


class FakeDianpiRunner:
    def __init__(self, result=DianpiTriggerResult.STARTED):
        self.result = result
        self.calls = 0

    def trigger(self):
        self.calls += 1
        return self.result


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

    def test_left_long_hold_then_right_press_emits_dianpi_once(self):
        state = SpaceMouseGestureState()
        state.update(frame(0.00, buttons=(True, False), pressed=(0,)), 0.00)
        result = state.update(frame(LONG_PRESS_S + 0.01, buttons=(True, False)), LONG_PRESS_S + 0.01)
        self.assertEqual(result.events, ())
        result = state.update(
            frame(LONG_PRESS_S + 0.02, buttons=(True, True), pressed=(1,)),
            LONG_PRESS_S + 0.02,
        )
        self.assertEqual(result.events, (TakeoverEvent.START_DIANPI,))
        self.assertEqual(result.consumed_buttons, (0, 1))
        result = state.update(frame(LONG_PRESS_S + 0.50, buttons=(True, True)), LONG_PRESS_S + 0.50)
        self.assertEqual(result.events, ())
        state.update(frame(LONG_PRESS_S + 0.60, buttons=(False, False)), LONG_PRESS_S + 0.60)

    def test_left_long_hold_then_right_dianpi_does_not_require_neutral_axes(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.00, buttons=(True, False), pressed=(0,)), 1.00)
        state.update(frame(1.00 + LONG_PRESS_S + 0.01, buttons=(True, False)), 1.00 + LONG_PRESS_S + 0.01)
        result = state.update(
            frame(1.00 + LONG_PRESS_S + 0.02, axes=(0.5, 0, 0, 0, 0, 0), buttons=(True, True), pressed=(1,)),
            1.00 + LONG_PRESS_S + 0.02,
        )
        self.assertEqual(result.events, (TakeoverEvent.START_DIANPI,))
        self.assertEqual(result.consumed_buttons, (0, 1))

    def test_early_right_press_is_suppressed_until_both_buttons_release(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.00, buttons=(True, False), pressed=(0,)), 1.00)
        state.update(frame(1.05, buttons=(True, True), pressed=(1,)), 1.05)
        self.assertEqual(
            state.update(frame(1.05 + LONG_PRESS_S + 0.1, buttons=(True, True)), 1.05 + LONG_PRESS_S + 0.1).events,
            (),
        )
        state.update(frame(2.00, buttons=(False, False)), 2.00)
        state.update(frame(2.10, buttons=(True, False), pressed=(0,)), 2.10)
        state.update(frame(2.10 + LONG_PRESS_S + 0.01, buttons=(True, False)), 2.10 + LONG_PRESS_S + 0.01)
        result = state.update(frame(2.10 + LONG_PRESS_S + 0.02, buttons=(True, True), pressed=(1,)), 2.10 + LONG_PRESS_S + 0.02)
        self.assertEqual(result.events, (TakeoverEvent.START_DIANPI,))

    def test_dianpi_chord_suppresses_all_single_button_gestures(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.00, buttons=(True, False), pressed=(0,)), 1.00)
        state.update(frame(1.00 + LONG_PRESS_S + 0.01, buttons=(True, False)), 1.00 + LONG_PRESS_S + 0.01)
        result = state.update(frame(1.00 + LONG_PRESS_S + 0.02, buttons=(True, True), pressed=(1,)), 1.00 + LONG_PRESS_S + 0.02)
        self.assertEqual(result.events, (TakeoverEvent.START_DIANPI,))
        state.update(frame(1.50, buttons=(False, False)), 1.50)
        result = state.update(frame(1.60, pressed=(1,)), 1.60)
        self.assertEqual(result.events, ())

    def test_held_left_allows_repeated_right_taps_to_run_repeated_cycles(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.00, buttons=(True, False), pressed=(0,)), 1.00)
        state.update(frame(1.00 + LONG_PRESS_S + 0.01, buttons=(True, False)), 1.00 + LONG_PRESS_S + 0.01)
        first = state.update(frame(1.50, buttons=(True, True), pressed=(1,)), 1.50)
        self.assertEqual(first.events, (TakeoverEvent.START_DIANPI,))
        # Tap the right button again without releasing the left modifier: the
        # second release must re-arm the chord instead of locking the operator
        # out until both buttons are fully up.
        released = state.update(frame(1.60, buttons=(True, False)), 1.60)
        self.assertEqual(released.events, ())
        second = state.update(frame(1.70, buttons=(True, True), pressed=(1,)), 1.70)
        self.assertEqual(second.events, (TakeoverEvent.START_DIANPI,))
        third = state.update(frame(1.80, buttons=(True, True), pressed=(1,)), 1.80)
        self.assertEqual(third.events, ())

    def test_rearmed_chord_never_leaks_a_takeover_on_release(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.00, buttons=(True, False), pressed=(0,)), 1.00)
        state.update(frame(1.00 + LONG_PRESS_S + 0.01, buttons=(True, False)), 1.00 + LONG_PRESS_S + 0.01)
        state.update(frame(1.50, buttons=(True, True), pressed=(1,)), 1.50)
        state.update(frame(1.60, buttons=(True, False)), 1.60)
        # Releasing the still-held left modifier after a re-armed chord is a
        # chord release, not a click or a takeover.
        self.assertEqual(state.update(frame(1.70), 1.70).events, ())
        self.assertEqual(state.update(frame(1.80, buttons=(False, True), pressed=(1,)), 1.80).events, ())
        self.assertEqual(state.update(frame(1.90), 1.90).events, ())

    def test_early_right_press_is_not_rearmed_into_a_chord_right_away(self):
        state = SpaceMouseGestureState()
        state.update(frame(1.00, buttons=(True, False), pressed=(0,)), 1.00)
        # Both buttons are pressed before the left modifier is long enough.
        state.update(frame(1.05, buttons=(True, True), pressed=(1,)), 1.05)
        state.update(frame(1.06, buttons=(True, False)), 1.06)
        # A right press that still predates the left long press stays silent.
        self.assertEqual(
            state.update(frame(1.10, buttons=(True, True), pressed=(1,)), 1.10).events,
            (),
        )
        # Once the left hold has genuinely crossed the threshold, the next
        # right press is a real chord again.
        self.assertEqual(state.update(frame(1.45, buttons=(True, False)), 1.45).events, ())
        self.assertEqual(
            state.update(frame(1.46, buttons=(True, True), pressed=(1,)), 1.46).events,
            (TakeoverEvent.START_DIANPI,),
        )


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
        self.assertTrue(gate.is_paused())
        self.assertEqual(gate.pause_generation, 0)
        release.set()
        pausing.join(1.0)
        thread.join(1.0)
        self.assertFalse(pausing.is_alive())
        self.assertTrue(gate.is_paused())
        self.assertEqual(gate.pause_generation, 1)
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

    def test_consume_advance_clears_the_request_and_reopens_automatic_motion(self):
        gate = MotionPauseGate()
        gate.request_advance()
        self.assertTrue(gate.is_paused())
        with self.assertRaises(AdvanceRightRequested):
            gate.wait_for_auto()
        gate.consume_advance()
        self.assertFalse(gate.is_paused())
        # The next automatic writer proceeds instead of raising again, so one
        # gate can serve several "next stage" boundaries.
        gate.wait_for_auto()

    def test_consume_advance_keeps_an_aborted_gate_closed(self):
        gate = MotionPauseGate()
        error = SafetyError("worker failed")
        gate.abort(error)
        gate.consume_advance()
        self.assertTrue(gate.is_paused())
        with self.assertRaises(SafetyError):
            gate.wait_for_auto()

    def test_pause_that_wins_before_writer_lock_prevents_the_queued_write(self):
        gate = MotionPauseGate()
        entered = threading.Event()
        gate._writer_lock.acquire()
        try:
            automatic = threading.Thread(target=lambda: self._enter_automatic(gate, entered))
            automatic.start()
            time.sleep(0.02)
            pausing = threading.Thread(target=gate.pause)
            pausing.start()
            time.sleep(0.02)
        finally:
            gate._writer_lock.release()
        pausing.join(1.0)
        self.assertFalse(entered.wait(0.05))
        gate.resume()
        self.assertTrue(entered.wait(1.0))
        automatic.join(1.0)

    @staticmethod
    def _enter_automatic(gate, entered):
        with gate.automatic_motion():
            entered.set()


class WorkerTests(unittest.TestCase):
    def test_activation_uses_manual_position_control_and_restores_joint_impedance(self):
        for side, group in (("left", 0), ("right", 1)):
            with self.subTest(side=side):
                robot = BilateralRobot()
                robot.control_mode = 3
                gate = MotionPauseGate()
                controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
                worker = SpaceMouseTakeoverWorker(
                    controller,
                    gate,
                    device=None,
                    config=TeleopConfig(side=side, axis_map=(1, 2, 3, 4, 5, 6)),
                    clock=lambda: 1.0,
                )

                worker._activate(side, frame(1.0), 1.0, ())

                # Manual takeover is rigid position control, not impedance.
                self.assertEqual(robot.control_mode, 1)
                self.assertTrue(gate.is_paused())
                self.assertEqual(robot.joint_impedance_calls, [])

                worker._deactivate(resume=False)

                # Automatic stages need joint impedance restored on that arm.
                self.assertEqual(robot.control_mode, 3)
                self.assertEqual([configured_group for _, configured_group in robot.joint_impedance_calls], [group])
                self.assertEqual(
                    [item.kp for item in robot.joint_impedance_calls[0][0]],
                    list(STANDARD_JOINT_IMPEDANCE_KP),
                )

    def test_worker_routes_left_hold_right_press_to_dianpi_without_takeover(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        runner = FakeDianpiRunner()
        worker = SpaceMouseTakeoverWorker(
            controller,
            gate,
            device=None,
            config=TeleopConfig(side="left", axis_map=(1, 2, 3, 4, 5, 6)),
            dianpi_runner=runner,
            clock=lambda: LONG_PRESS_S + 0.02,
        )

        worker.step(frame(0.00, buttons=(True, False), pressed=(0,)), 0.00)
        worker.step(frame(LONG_PRESS_S + 0.01, buttons=(True, False)), LONG_PRESS_S + 0.01)
        result = worker.step(frame(LONG_PRESS_S + 0.02, buttons=(True, True), pressed=(1,)), LONG_PRESS_S + 0.02)

        self.assertEqual(result.events, (TakeoverEvent.START_DIANPI,))
        self.assertEqual(runner.calls, 1)
        self.assertIsNone(worker.active_side)

    def test_busy_dianpi_trigger_does_not_fail_worker(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        runner = FakeDianpiRunner(DianpiTriggerResult.BUSY)
        worker = SpaceMouseTakeoverWorker(
            controller,
            gate,
            device=None,
            config=TeleopConfig(side="left", axis_map=(1, 2, 3, 4, 5, 6)),
            dianpi_runner=runner,
        )

        worker.step(frame(0.00, buttons=(True, False), pressed=(0,)), 0.00)
        worker.step(frame(LONG_PRESS_S + 0.01, buttons=(True, False)), LONG_PRESS_S + 0.01)
        worker.step(frame(LONG_PRESS_S + 0.02, buttons=(True, True), pressed=(1,)), LONG_PRESS_S + 0.02)
        worker.raise_if_failed()
        self.assertEqual(runner.calls, 1)

    def test_worker_reports_every_dianpi_trigger_result(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        events = []

        class CountingRunner:
            def __init__(self):
                self.calls = 0

            def trigger(self):
                self.calls += 1
                return DianpiTriggerResult.STARTED if self.calls == 1 else DianpiTriggerResult.BUSY

        runner = CountingRunner()
        worker = SpaceMouseTakeoverWorker(
            controller,
            gate,
            device=None,
            config=TeleopConfig(side="left", axis_map=(1, 2, 3, 4, 5, 6)),
            dianpi_runner=runner,
            event_sink=events.append,
        )

        worker.step(frame(0.00, buttons=(True, False), pressed=(0,)), 0.00)
        worker.step(frame(LONG_PRESS_S + 0.01, buttons=(True, False)), LONG_PRESS_S + 0.01)
        worker.step(frame(LONG_PRESS_S + 0.02, buttons=(True, True), pressed=(1,)), LONG_PRESS_S + 0.02)
        # Tap the right button a second time without releasing the left one.
        worker.step(frame(LONG_PRESS_S + 0.10, buttons=(True, False)), LONG_PRESS_S + 0.10)
        worker.step(frame(LONG_PRESS_S + 0.20, buttons=(True, True), pressed=(1,)), LONG_PRESS_S + 0.20)

        self.assertEqual(runner.calls, 2)
        self.assertEqual(
            [event["action"] for event in events],
            ["cycle_started", "ignored_previous_cycle_still_running"],
        )
        self.assertTrue(all(event["event"] == "dianpi_trigger_result" for event in events))
        # The operator-facing sink must stay optional for standalone users.
        silent = SpaceMouseTakeoverWorker(
            controller, gate, device=None,
            config=TeleopConfig(side="left", axis_map=(1, 2, 3, 4, 5, 6)),
            dianpi_runner=FakeDianpiRunner(),
        )
        silent.step(frame(0.00, buttons=(True, False), pressed=(0,)), 0.00)
        silent.step(frame(LONG_PRESS_S + 0.01, buttons=(True, False)), LONG_PRESS_S + 0.01)
        silent.step(frame(LONG_PRESS_S + 0.02, buttons=(True, True), pressed=(1,)), LONG_PRESS_S + 0.02)
        self.assertIsNone(silent.event_sink)

    def test_dianpi_trigger_sink_failure_cannot_break_the_worker(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)

        def broken_sink(_event):
            raise RuntimeError("log pipe closed")

        worker = SpaceMouseTakeoverWorker(
            controller, gate, device=None,
            config=TeleopConfig(side="left", axis_map=(1, 2, 3, 4, 5, 6)),
            dianpi_runner=FakeDianpiRunner(),
            event_sink=broken_sink,
        )
        worker.step(frame(0.00, buttons=(True, False), pressed=(0,)), 0.00)
        worker.step(frame(LONG_PRESS_S + 0.01, buttons=(True, False)), LONG_PRESS_S + 0.01)
        result = worker.step(frame(LONG_PRESS_S + 0.02, buttons=(True, True), pressed=(1,)), LONG_PRESS_S + 0.02)
        self.assertEqual(result.events, (TakeoverEvent.START_DIANPI,))
        worker.raise_if_failed()

    def test_transient_gdk_read_timeout_keeps_takeover_and_auto_paused(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)

        class FakeHID:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def poll(self):
                return frame(1.0)

        worker = SpaceMouseTakeoverWorker(
            controller, gate, device="/dev/hidraw-test",
            config=TeleopConfig(side="left", axis_map=(1, 2, 3, 4, 5, 6)),
            clock=lambda: 1.0, hid_factory=lambda _device: FakeHID(),
        )
        gate.pause()
        worker.active_side = "left"
        worker._session = object()
        calls = []

        def step(_frame, _now):
            calls.append(True)
            if len(calls) == 1:
                raise TransientGDKReadTimeout("slow status")
            worker._stop.set()

        worker.step = step
        worker._run()
        self.assertIsNone(worker._failed)
        self.assertEqual(worker.active_side, "left")
        self.assertIsNotNone(worker._session)
        self.assertTrue(gate.is_paused())

    def test_disconnect_holds_and_reopens_device_without_failing_workflow(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        opened = []

        class FakeHID:
            def __init__(self, attempt):
                self.attempt = attempt

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def poll(self):
                if self.attempt == 1:
                    raise InputError("device disconnected")
                worker._stop.set()
                return frame(1.0)

        def factory(_device):
            opened.append(True)
            return FakeHID(len(opened))

        worker = SpaceMouseTakeoverWorker(
            controller, gate, device="/dev/spacemouse-compact",
            config=TeleopConfig(side="right", axis_map=(1, 2, 3, 4, 5, 6)),
            clock=lambda: 1.0, hid_factory=factory,
        )
        worker._run()
        self.assertEqual(len(opened), 2)
        self.assertIsNone(worker._failed)
        self.assertTrue(gate.is_paused())
        self.assertIn("disconnected", worker.last_disconnect)

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
        gripper_positions = [
            tuple(request.end_effector_joint_positions) for request in robot.requests
            if request.end_effector_joint_positions
        ]
        self.assertEqual(gripper_positions, [])

    def test_dianpi_chord_holds_measured_pose_until_both_buttons_release(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        worker = SpaceMouseTakeoverWorker(
            controller, gate, device=None,
            config=TeleopConfig(side="left", axis_map=(1, 2, 3, 4, 5, 6)),
            dianpi_runner=FakeDianpiRunner(),
            clock=lambda: 1.0,
        )
        worker._activate("left", frame(1.0), 1.0, ())
        worker.step(frame(1.02, buttons=(True, False), pressed=(0,)), 1.02)
        for index in range(1, 22):
            now = 1.02 + index * .02
            worker.step(frame(now, buttons=(True, False)), now)
        worker.step(frame(1.46, axes=(.8, 0, 0, 0, 0, 0), buttons=(True, True), pressed=(1,)), 1.46)
        x_after_trigger = robot.requests[-1].left_end_effector_pose.position.x
        worker.step(frame(1.48, axes=(.8, 0, 0, 0, 0, 0), buttons=(True, True)), 1.48)
        self.assertEqual(robot.requests[-1].left_end_effector_pose.position.x, x_after_trigger)

    def test_right_double_click_never_opens_the_right_gripper(self):
        robot = BilateralRobot()
        gate = MotionPauseGate()
        controller = G2Controller(FakeGdk, robot, allow_motion=True, motion_gate=gate)
        worker = SpaceMouseTakeoverWorker(
            controller, gate, device=None,
            config=TeleopConfig(side="right", axis_map=(1, 2, 3, 4, 5, 6)),
            clock=lambda: 1.34,
        )
        worker.step(frame(1.0), 1.0)
        worker.step(frame(1.02, buttons=(False, True), pressed=(1,)), 1.02)
        worker.step(frame(1.06), 1.06)
        worker.step(frame(1.30, buttons=(False, True), pressed=(1,)), 1.30)
        worker.step(frame(1.34), 1.34)
        self.assertEqual(worker.active_side, "right")
        for index in range(1, 33):
            now = 1.34 + index * .02
            worker.step(frame(now), now)
        worker.step(frame(2.00, buttons=(False, True), pressed=(1,)), 2.00)
        worker.step(frame(2.04), 2.04)
        worker.step(frame(2.08, buttons=(False, True), pressed=(1,)), 2.08)
        worker.step(frame(2.12), 2.12)
        self.assertIsNone(worker.active_side)
        self.assertFalse(any(request.end_effector_joint_positions for request in robot.requests))

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

    def test_worker_consume_advance_clears_the_flag_and_reopens_the_gate(self):
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
        worker.step(
            frame(1.02 + RIGHT_LONG_PRESS_S + 0.01, buttons=(False, True)),
            1.02 + RIGHT_LONG_PRESS_S + 0.01,
        )
        self.assertTrue(worker.advance_requested)
        worker.consume_advance()
        self.assertFalse(worker.advance_requested)
        self.assertFalse(gate.is_paused())
        gate.wait_for_auto()


if __name__ == "__main__":
    unittest.main()
