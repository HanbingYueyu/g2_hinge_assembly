"""Pure SpaceMouse takeover gestures and a shared automatic-motion pause gate.

The gesture state is deliberately independent from GDK.  The pause gate is
used by a single GDK process so a temporary SpaceMouse writer can pause the
visual writer without creating a second Robot/GDK owner.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import math
import threading
import time
from typing import Iterator

try:
    from .safety import SafetyError, AdvanceRightRequested, MotionWriteCancelled
    from .spacemouse_input import InputFrame
except ImportError:
    from safety import SafetyError, AdvanceRightRequested, MotionWriteCancelled
    from spacemouse_input import InputFrame


DOUBLE_CLICK_WINDOW_S = 0.55
LONG_PRESS_S = 0.40
RIGHT_LONG_PRESS_S = 2.0
# Kept as a compatibility constant for callers that imported the old policy;
# takeover release is now explicit (double-click the active hand) and never
# occurs solely because the SpaceMouse is idle.
INACTIVITY_TIMEOUT_S = 3.0


class TeachReplayTakenOver(RuntimeError):
    """A taught replay was ended by manual takeover, not temporarily paused."""


class TakeoverEvent(str, Enum):
    NONE = "none"
    TAKEOVER_LEFT = "takeover_left"
    TAKEOVER_RIGHT = "takeover_right"
    ADVANCE_RIGHT = "advance_right"
    START_DIANPI = "start_dianpi"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class GestureResult:
    events: tuple[TakeoverEvent, ...] = ()
    consumed_buttons: tuple[int, ...] = ()
    activity: bool = False


class SpaceMouseGestureState:
    """Convert sampled button reports into takeover/advance gestures."""

    def __init__(self, *, deadzone: float = 0.12) -> None:
        if not math.isfinite(deadzone) or not 0.0 <= deadzone < 1.0:
            raise ValueError("deadzone must be in [0, 1)")
        self.deadzone = float(deadzone)
        self._was_down = [False, False]
        self._down_since: list[float | None] = [None, None]
        self._long = [False, False]
        self._pending_click: list[float | None] = [None, None]
        self._chord_blocked = False
        self._chord_triggered = False

    def _short_release(self, button: int, now: float) -> tuple[TakeoverEvent | None, bool]:
        down_since = self._down_since[button]
        duration = now - down_since if down_since is not None else 0.0
        self._down_since[button] = None
        if self._long[button] or duration >= LONG_PRESS_S:
            self._long[button] = False
            self._pending_click[button] = None
            return None, False
        self._long[button] = False
        pending = self._pending_click[button]
        if pending is not None and now - pending <= DOUBLE_CLICK_WINDOW_S:
            self._pending_click[button] = None
            return (
                TakeoverEvent.TAKEOVER_LEFT if button == 0 else TakeoverEvent.TAKEOVER_RIGHT,
                True,
            )
        self._pending_click[button] = now
        return None, False

    def _instant_edges(self, button: int, count: int, now: float) -> tuple[TakeoverEvent | None, bool]:
        if count <= 0:
            return None, False
        pending = self._pending_click[button]
        if count >= 2 or (pending is not None and now - pending <= DOUBLE_CLICK_WINDOW_S):
            self._pending_click[button] = None
            return (
                TakeoverEvent.TAKEOVER_LEFT if button == 0 else TakeoverEvent.TAKEOVER_RIGHT,
                True,
            )
        self._pending_click[button] = now
        return None, False

    def update(self, frame: InputFrame, now: float) -> GestureResult:
        if not math.isfinite(now):
            raise ValueError("gesture timestamp must be finite")
        events: list[TakeoverEvent] = []
        consumed: set[int] = set()
        activity = bool(any(frame.buttons) or frame.pressed)
        if any(abs(float(value)) > self.deadzone for value in frame.axes):
            activity = True

        # A pending first click is allowed to expire without becoming an
        # action. This prevents two separated single clicks from switching a
        # hand after the double-click window.
        for button, pending in enumerate(self._pending_click):
            if pending is not None and now - pending > DOUBLE_CLICK_WINDOW_S:
                self._pending_click[button] = None

        # The overlap itself is a conflict boundary.  Once both buttons are
        # down, never let a partially completed single-button gesture leak
        # through.  Dianpi is deliberately a directional gesture: the left
        # button must already have been held long enough to be a 6D modifier,
        # then a right-button press while it remains held starts one cycle.
        # The state is held until both buttons are released.
        both_down = bool(frame.buttons[0]) and bool(frame.buttons[1])
        left_down = bool(frame.buttons[0])
        right_down = bool(frame.buttons[1])
        left_pressed = left_down and not self._was_down[0]
        right_pressed = right_down and not self._was_down[1]

        # Update the left hold before checking the overlap.  Without this
        # small pre-check, a report containing the right-button edge would
        # lose the fact that the left button had already crossed the long
        # press threshold on the preceding report.
        if left_pressed:
            self._down_since[0] = now
            self._long[0] = False
        if left_down and self._down_since[0] is not None:
            if now - self._down_since[0] >= LONG_PRESS_S:
                self._long[0] = True
                self._pending_click[0] = None

        if both_down and not self._chord_blocked:
            left_long = self._long[0]
            self._chord_triggered = False
            self._chord_blocked = True
            self._pending_click = [None, None]
            self._long = [left_long, False]
            if left_long and right_pressed:
                self._chord_triggered = True
                events.append(TakeoverEvent.START_DIANPI)
        if self._chord_blocked:
            consumed.update((0, 1))
            if right_down:
                self._was_down = [True, True]
                return GestureResult(tuple(events), tuple(sorted(consumed)), activity)
            # The right-button press edge is what triggers one Dianpi cycle, so
            # the chord re-arms as soon as that edge is released.  An operator
            # can then keep the left modifier held and tap the right button
            # again for the next cycle instead of being locked out until both
            # buttons are fully up.  Releases are still swallowed here so no
            # click or takeover can leak out of the overlap.
            self._chord_blocked = False
            self._chord_triggered = False
            if not left_down:
                self._was_down = [False, False]
                self._down_since = [None, None]
                self._long = [False, False]
                self._pending_click = [None, None]
            else:
                # Left is still held: keep its measured press state (the
                # pre-chord long-press tracker keeps updating it every tick),
                # so the next right press is another chord only once the left
                # button has genuinely been held long enough.  Drop any pending
                # single click it may have accumulated.
                self._was_down = [True, False]
                self._pending_click[0] = None
            return GestureResult((), tuple(sorted(consumed)), activity)

        for button in (0, 1):
            down = bool(frame.buttons[button])
            if down and not self._was_down[button]:
                self._down_since[button] = now
                self._long[button] = False
            if down and self._down_since[button] is not None:
                held = now - self._down_since[button]
                if button == 1 and held >= RIGHT_LONG_PRESS_S and not self._long[button]:
                    self._long[button] = True
                    self._pending_click[button] = None
                    events.append(TakeoverEvent.ADVANCE_RIGHT)
                    consumed.add(button)
                elif button == 0 and held >= LONG_PRESS_S:
                    # Left long press is the 6D modifier. It is not a
                    # takeover event, but it must not become a click later.
                    self._long[button] = True
                    self._pending_click[button] = None
            if not down and self._was_down[button]:
                event, was_double = self._short_release(button, now)
                if event is not None:
                    events.append(event)
                if was_double:
                    consumed.add(button)
            if not down and not self._was_down[button]:
                event, was_double = self._instant_edges(button, frame.pressed.count(button), now)
                if event is not None:
                    events.append(event)
                if was_double:
                    consumed.add(button)
            self._was_down[button] = down

        # A right long press wins over any pending click and is emitted once.
        if TakeoverEvent.ADVANCE_RIGHT in events:
            events = [TakeoverEvent.ADVANCE_RIGHT]
            consumed.add(1)
        return GestureResult(tuple(events), tuple(sorted(consumed)), activity)


class MotionPauseGate:
    """Block automatic GDK sends while a manual writer owns the controller."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._writer_lock = threading.RLock()
        self._paused = False
        self._advance_requested = False
        self._aborted: BaseException | None = None
        self._cancelled_thread_ids: set[int] = set()
        self._pause_generation = 0
        self._local = threading.local()

    def pause(self) -> None:
        with self._condition:
            if self._aborted is not None:
                raise SafetyError(f"motion gate aborted: {self._aborted}")
            self._paused = True
        # Do not let a manual command race an automatic command that passed
        # the pause check immediately before this call. Publish the new pause
        # generation only after that in-flight writer has finished, so its
        # refresh callback cannot establish a pre-takeover feedback baseline.
        with self._writer_lock:
            with self._condition:
                self._pause_generation += 1

    def resume(self) -> None:
        with self._condition:
            if self._aborted is not None:
                return
            self._paused = False
            self._condition.notify_all()

    def request_advance(self) -> None:
        """Close automatic motion and wake waiters for a left→right switch."""
        with self._condition:
            if self._aborted is not None:
                raise SafetyError(f"motion gate aborted: {self._aborted}")
            self._paused = True
            self._advance_requested = True
            self._condition.notify_all()
        # Finish any automatic write that had already passed the pause check.
        # New writers see _advance_requested and raise immediately.
        with self._writer_lock:
            return

    def consume_advance(self) -> None:
        """Clear one consumed advance request and re-open automatic motion.

        The staged bilateral flow uses the same gate for several "next stage"
        gates, so each gate must consume exactly one request instead of the
        sticky single-shot behaviour of the original two-stage runner.
        """
        with self._condition:
            self._advance_requested = False
            if self._aborted is None:
                self._paused = False
            self._condition.notify_all()

    def abort(self, error: BaseException) -> None:
        with self._condition:
            self._aborted = error
            # Keep the gate closed after a worker failure. Waiting automatic
            # writers wake and raise instead of silently resuming motion.
            self._paused = True
            self._condition.notify_all()

    def is_paused(self) -> bool:
        with self._condition:
            return self._paused

    @property
    def pause_generation(self) -> int:
        with self._condition:
            return self._pause_generation

    def wait_for_auto(self) -> None:
        if getattr(self._local, "manual", False):
            return
        with self._condition:
            thread_id = threading.get_ident()
            while (
                self._paused and not self._advance_requested and self._aborted is None
                and thread_id not in self._cancelled_thread_ids
            ):
                self._condition.wait()
            if thread_id in self._cancelled_thread_ids:
                raise MotionWriteCancelled("automatic motion writer was stopped")
            if self._aborted is not None:
                raise SafetyError(f"motion gate aborted: {self._aborted}")
            if self._advance_requested:
                raise AdvanceRightRequested("SpaceMouse 请求从左手阶段进入右手阶段")

    @contextmanager
    def automatic_teach_motion(self, generation: int) -> Iterator[None]:
        """Keep normal writer ownership, but never wait/resume a taken-over replay."""
        with self._writer_lock:
            with self._condition:
                if self._advance_requested:
                    raise AdvanceRightRequested("确认进入下一阶段")
                if self._paused or self._pause_generation != generation:
                    raise TeachReplayTakenOver("SpaceMouse 接管，结束示教回放")
                self.wait_for_auto()
            yield
            with self._condition:
                if self._advance_requested:
                    raise AdvanceRightRequested("确认进入下一阶段")
                if self._paused or self._pause_generation != generation:
                    raise TeachReplayTakenOver("SpaceMouse 接管，结束示教回放")

    @contextmanager
    def automatic_motion(self) -> Iterator[None]:
        """Reserve the single robot writer for one automatic GDK request."""
        while True:
            self.wait_for_auto()
            self._writer_lock.acquire()
            with self._condition:
                thread_id = threading.get_ident()
                manual = getattr(self._local, "manual", False)
                blocked = not manual and (
                    self._paused or self._advance_requested or self._aborted is not None
                )
                cancelled = thread_id in self._cancelled_thread_ids
            if not blocked and not cancelled:
                break
            self._writer_lock.release()
        try:
            yield
        finally:
            self._writer_lock.release()

    def cancel_automatic_thread(self, thread: threading.Thread | None) -> None:
        """Wake and cancel one stream writer without resuming other writers."""
        if thread is None or thread.ident is None:
            return
        with self._condition:
            self._cancelled_thread_ids.add(thread.ident)
            self._condition.notify_all()

    def clear_cancelled_thread(self, thread: threading.Thread | None) -> None:
        if thread is None or thread.ident is None:
            return
        with self._condition:
            self._cancelled_thread_ids.discard(thread.ident)

    @contextmanager
    def manual_motion(self) -> Iterator[None]:
        previous = getattr(self._local, "manual", False)
        with self._writer_lock:
            self._local.manual = True
            try:
                yield
            finally:
                self._local.manual = previous


class SpaceMouseTakeoverWorker:
    """Run manual control beside an existing visual loop in one GDK process."""

    def __init__(
        self,
        controller,
        gate: MotionPauseGate,
        *,
        device: str | None,
        config,
        allow_advance_right: bool = False,
        clock=time.monotonic,
        sleep=time.sleep,
        hid_factory=None,
        dianpi_runner=None,
        event_sink=None,
    ) -> None:
        self.controller = controller
        self.gate = gate
        self.device = device
        self.config = config
        self.allow_advance_right = bool(allow_advance_right)
        self.clock = clock
        self.sleep = sleep
        self.hid_factory = hid_factory
        self.dianpi_runner = dianpi_runner
        self.event_sink = event_sink
        self.gestures = SpaceMouseGestureState(deadzone=config.deadzone)
        self.active_side: str | None = None
        self._session = None
        self._advance_requested = False
        self._keyboard_advance = threading.Event()
        self._failed: BaseException | None = None
        self._connected = False
        self._last_disconnect: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_dianpi_result = None
        self._dianpi_error: BaseException | None = None

    @property
    def advance_requested(self) -> bool:
        return self._advance_requested

    def request_keyboard_advance(self) -> None:
        # Only the HID worker may release a manual session or write to GDK.
        if self.allow_advance_right:
            self._keyboard_advance.set()

    def _process_keyboard_advance(self) -> bool:
        if not self._keyboard_advance.is_set():
            return False
        self._keyboard_advance.clear()
        if not self.allow_advance_right:
            return False
        if self._session is not None:
            self._deactivate(resume=False)
        self.gate.request_advance()
        self._advance_requested = True
        return True

    def consume_advance(self) -> None:
        """Consume one pending advance request and let automatic motion resume."""
        self._advance_requested = False
        self.gate.consume_advance()

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def last_disconnect(self) -> str | None:
        return self._last_disconnect

    @property
    def last_dianpi_result(self):
        return self._last_dianpi_result

    @property
    def dianpi_error(self) -> BaseException | None:
        return self._dianpi_error

    def raise_if_failed(self) -> None:
        if self._failed is not None:
            raise SafetyError(f"SpaceMouse 接管失败: {self._failed}") from self._failed

    @staticmethod
    def _without_consumed_buttons(frame: InputFrame, consumed: tuple[int, ...]) -> InputFrame:
        blocked = set(consumed)
        return InputFrame(
            frame.axes,
            frame.buttons,
            tuple(button for button in frame.pressed if button not in blocked),
            frame.axis_times,
            frame.ready,
            frame.peak_axes,
        )

    @staticmethod
    def _neutral_frame(now: float) -> InputFrame:
        """Build a valid hold report for a button-conflict event.

        A Dianpi trigger owns both buttons.  Feeding the original axes into
        the manual session at that instant could turn a trigger into an
        unintended robot step, so the session receives a measured-pose hold
        while retaining its current gripper goal.
        """
        return InputFrame(
            (0.0,) * 6,
            (False, False),
            (),
            (now, now),
            True,
            (0.0,) * 6,
        )

    def _new_session(self, side: str):
        from dataclasses import replace

        try:
            from .spacemouse_teleop import TeleopSession
        except ImportError:
            from spacemouse_teleop import TeleopSession

        config = replace(self.config, side=side)
        return TeleopSession(
            self.controller,
            config,
            clock=self.clock,
            allow_side_switch=False,
        )

    def _activate(self, side: str, frame: InputFrame, now: float, consumed: tuple[int, ...]) -> None:
        if not self.controller.allow_motion:
            raise SafetyError("SpaceMouse 接管需要 --allow-motion")
        if self._session is not None:
            self._deactivate(resume=False)
        self.gate.pause()
        self.active_side = side
        self._session = self._new_session(side)
        with self.controller.manual_motion():
            status = self.controller.motion_status_summary()
            if status["control_mode"] != 1:
                self.controller.set_control_mode(1)
            self._session.tick(self._without_consumed_buttons(frame, consumed), now)

    def _deactivate(self, *, resume: bool) -> None:
        if self._session is not None:
            with self.controller.manual_motion():
                self._session.stop()
        self._session = None
        self.active_side = None
        if resume:
            self.gate.resume()

    def _emit_dianpi(self, event: dict) -> None:
        sink = self.event_sink
        if sink is None:
            return
        try:
            sink(event)
        except Exception:
            # Operator feedback must never affect G2 or SpaceMouse control.
            pass

    def step(self, frame: InputFrame, now: float) -> GestureResult:
        result = self.gestures.update(frame, now)
        if self._process_keyboard_advance():
            return result
        if TakeoverEvent.START_DIANPI in result.events:
            if self.dianpi_runner is None:
                self._emit_dianpi({
                    "event": "dianpi_trigger_result",
                    "result": "ignored",
                    "action": "no_dianpi_runner",
                })
            else:
                try:
                    outcome = self.dianpi_runner.trigger()
                    self._last_dianpi_result = outcome
                    self._dianpi_error = None
                    value = getattr(outcome, "value", None) or str(outcome)
                    self._emit_dianpi({
                        "event": "dianpi_trigger_result",
                        "result": value,
                        "action": (
                            "cycle_started"
                            if value == "started"
                            else "ignored_previous_cycle_still_running"
                            if value == "busy"
                            else "hardware_error"
                        ),
                    })
                except BaseException as exc:
                    # The supplied runner is expected to swallow hardware
                    # failures; keep this boundary defensive so a logging or
                    # test runner can never abort G2 motion.
                    self._dianpi_error = exc
                    self._emit_dianpi({
                        "event": "dianpi_trigger_result",
                        "result": "raised",
                        "action": "hardware_error",
                        "error": str(exc),
                    })
            if self._session is not None:
                # The chord owns both buttons and must not feed stale axes or
                # click state into TeleopSession.  A neutral tick bases the
                # hold on current feedback while preserving the active
                # gripper goal.
                with self.controller.manual_motion():
                    self._session.tick(self._neutral_frame(now), now)
            return result
        if TakeoverEvent.ADVANCE_RIGHT in result.events:
            if self.allow_advance_right:
                if self._session is not None:
                    self._deactivate(resume=False)
                self.gate.request_advance()
                self._advance_requested = True
            return result

        requested_side = None
        if TakeoverEvent.TAKEOVER_LEFT in result.events:
            requested_side = "left"
        elif TakeoverEvent.TAKEOVER_RIGHT in result.events:
            requested_side = "right"
        chord_active = set(result.consumed_buttons) == {0, 1}
        sanitized = self._neutral_frame(now) if chord_active else self._without_consumed_buttons(frame, result.consumed_buttons)
        if requested_side is not None:
            if self._session is not None:
                self._session.cancel_pending_gripper_click()
            if self._session is not None and self.active_side == requested_side:
                # A second double-click on the active hand is the explicit
                # takeover release gesture.  Stop with a measured hold and
                # let the visual writer resume; never release on inactivity.
                self._deactivate(resume=True)
                return result
            self._activate(requested_side, sanitized, now, ())
        elif self._session is not None:
            with self.controller.manual_motion():
                self._session.tick(sanitized, now)
        return result

    def _run(self) -> None:
        try:
            if self.hid_factory is None:
                try:
                    from .spacemouse_input import CompactHID, InputError
                except ImportError:
                    from spacemouse_input import CompactHID, InputError

                hid_factory = CompactHID
            else:
                hid_factory = self.hid_factory
                try:
                    from .spacemouse_input import InputError
                except ImportError:
                    from spacemouse_input import InputError
            while not self._stop.is_set():
                self._process_keyboard_advance()
                try:
                    with hid_factory(self.device) as source:
                        self.gestures = SpaceMouseGestureState(deadzone=self.config.deadzone)
                        self._connected = True
                        deadline = self.clock()
                        while not self._stop.is_set():
                            # CompactHID timestamps each report while ``poll`` drains
                            # the kernel queue. Read the control-loop time *after*
                            # polling; sampling it before the read can make fresh
                            # axis timestamps appear to be from the future.
                            frame = source.poll()
                            now = self.clock()
                            try:
                                self.step(frame, now)
                            except Exception as exc:
                                try:
                                    from .spacemouse_teleop import TransientGDKReadTimeout
                                except ImportError:
                                    from spacemouse_teleop import TransientGDKReadTimeout
                                if not isinstance(exc, TransientGDKReadTimeout):
                                    raise
                                # Keep exclusive ownership. Rebuild the manual
                                # target from feedback on a later healthy tick;
                                # never resume automatic motion on a read stall.
                                side = self.active_side
                                self._session = None if side is None else self._new_session(side)
                                self.gestures = SpaceMouseGestureState(deadzone=self.config.deadzone)
                            deadline += 0.02
                            self._stop.wait(max(0.0, deadline - self.clock()))
                except (InputError, OSError) as exc:
                    if self._stop.is_set():
                        return
                    self._connected = False
                    self._last_disconnect = str(exc)
                    # Cancel an active manual target using its measured-pose
                    # stop, then keep the automatic writer paused. Reopening a
                    # HID node never resumes motion or reuses old axis state.
                    if self._session is not None:
                        self._deactivate(resume=False)
                    self.gate.pause()
                    self.gestures = SpaceMouseGestureState(deadzone=self.config.deadzone)
                    self._stop.wait(0.5)
        except BaseException as exc:  # Surface all worker/GDK failures to the owner loop.
            if not self._stop.is_set():
                self._failed = exc
                self.gate.abort(exc)
        finally:
            self._connected = False

    def start(self) -> None:
        if self.device is None:
            raise ValueError("SpaceMouse worker requires a device path")
        if self._thread is not None:
            raise RuntimeError("SpaceMouse worker is already running")
        self._thread = threading.Thread(target=self._run, name="g2-spacemouse-takeover", daemon=True)
        self._thread.start()

    def stop_for_final_motion(self) -> None:
        """Require the manual writer to finish before a final automatic return."""
        thread = self._thread
        self.stop()
        if thread is not None and thread.is_alive():
            raise SafetyError("SpaceMouse 接管线程尚未停止；拒绝启动放回动作")
        self.raise_if_failed()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._thread = None
        if self._session is not None:
            try:
                self._deactivate(resume=True)
            except Exception as exc:  # Keep cleanup best-effort and preserve the original failure.
                if self._failed is None:
                    self._failed = exc
        elif self.gate.is_paused() and not self._advance_requested:
            self.gate.resume()
