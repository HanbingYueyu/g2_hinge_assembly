"""Asynchronous bridge from SpaceMouse gestures to the RM1010 cycle."""

from __future__ import annotations

from enum import Enum
import importlib
import threading
from typing import Callable


class DianpiTriggerResult(str, Enum):
    STARTED = "started"
    BUSY = "busy"
    ERROR = "error"


def _default_controller_factory():
    # Keep the RM1010 shared-library import out of G2 startup.  The first
    # accepted chord is the explicit point at which the device is needed.
    module = importlib.import_module("dianpi.dp.Control")
    return module.DianpiController()


class DianpiCycleRunner:
    """Own one RM1010 controller and run cycles away from G2 threads.

    A controller is opened lazily on the first trigger and remains open until
    ``close`` so later chords do not rescan the USB bus.  Hardware exceptions
    are stored and emitted as events; ``trigger`` never raises them into the
    SpaceMouse worker.
    """

    def __init__(
        self,
        *,
        controller_factory: Callable[[], object] | None = None,
        event_sink: Callable[[dict], None] = print,
        thread_factory=threading.Thread,
    ) -> None:
        self._controller_factory = controller_factory or _default_controller_factory
        self._event_sink = event_sink
        self._thread_factory = thread_factory
        self._lock = threading.Lock()
        self._controller = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._closed = False
        self._last_error: BaseException | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def last_error(self) -> BaseException | None:
        with self._lock:
            return self._last_error

    def _emit(self, event: dict) -> None:
        try:
            self._event_sink(event)
        except Exception:
            # Logging must never affect G2 or SpaceMouse control.
            pass

    def trigger(self) -> DianpiTriggerResult:
        with self._lock:
            if self._closed:
                self._emit({"event": "dianpi_error", "error": "runner closed"})
                return DianpiTriggerResult.ERROR
            if self._running:
                self._emit({"event": "dianpi_busy"})
                return DianpiTriggerResult.BUSY
            self._running = True
            self._last_error = None
            try:
                thread = self._thread_factory(
                    target=self._run,
                    name="g2-dianpi-cycle",
                    daemon=True,
                )
                self._thread = thread
                thread.start()
            except BaseException as exc:
                self._running = False
                self._thread = None
                self._last_error = exc
                self._emit({"event": "dianpi_error", "error": str(exc)})
                return DianpiTriggerResult.ERROR
        return DianpiTriggerResult.STARTED

    def _run(self) -> None:
        local_controller = None
        try:
            with self._lock:
                controller = self._controller
            if controller is None:
                local_controller = self._controller_factory()
                local_controller.open()
                with self._lock:
                    self._controller = local_controller
                controller = local_controller
            self._emit({"event": "dianpi_started"})
            controller.run_cycle()
            self._emit({"event": "dianpi_finished"})
        except BaseException as exc:
            with self._lock:
                self._last_error = exc
            self._emit({"event": "dianpi_error", "error": str(exc)})
            if local_controller is not None:
                # An open failure may occur before the controller is retained;
                # still give the device a chance to restore safe outputs.
                with self._lock:
                    retained = self._controller is local_controller
                if not retained:
                    try:
                        local_controller.close()
                    except Exception:
                        pass
        finally:
            with self._lock:
                self._running = False
                self._thread = None

    def close(self, timeout_s: float = 1.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
            controller = self._controller
        if thread is not None and thread.is_alive() and controller is not None:
            stop_cycle = getattr(controller, "stop_cycle", None)
            if stop_cycle is not None:
                try:
                    stop_cycle()
                except BaseException as exc:
                    with self._lock:
                        self._last_error = exc
                    self._emit({"event": "dianpi_error", "error": str(exc)})
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_s)))
        if thread is not None and thread.is_alive():
            self._emit({"event": "dianpi_error", "error": "周期线程未能在关闭期限内停止"})
            return
        with self._lock:
            controller = self._controller
            self._controller = None
        if controller is not None:
            try:
                controller.close()
            except BaseException as exc:
                with self._lock:
                    self._last_error = exc
                self._emit({"event": "dianpi_error", "error": str(exc)})
