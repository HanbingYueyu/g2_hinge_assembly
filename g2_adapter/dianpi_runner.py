"""Asynchronous bridge from SpaceMouse gestures to the RM1010 cycle."""

from __future__ import annotations

from enum import Enum
import importlib
import sys
import threading
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class DianpiTriggerResult(str, Enum):
    STARTED = "started"
    BUSY = "busy"
    ERROR = "error"


def _default_controller_factory():
    # Keep the RM1010 shared-library import inside the runner.  The integrated
    # workflow opens it from the P10 monitor or the first SpaceMouse trigger.
    try:
        module = importlib.import_module("dianpi.dp.Control")
    except ModuleNotFoundError:
        # Started as ``python g2_adapter/<script>.py`` the project root is not
        # on sys.path, so the sibling ``dianpi`` package is invisible.  Add it
        # once and retry.
        root = str(PROJECT_ROOT)
        if root not in sys.path:
            sys.path.append(root)
        module = importlib.import_module("dianpi.dp.Control")
    return module.DianpiController()


class DianpiCycleRunner:
    """Own one RM1010 controller and run cycles away from G2 threads.

    A controller is opened lazily by the P10 monitor or the first trigger and
    remains open until ``close`` so later events do not rescan the USB bus.
    Hardware exceptions are stored and emitted as events; ``trigger`` never
    raises them into the SpaceMouse worker.
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
        self._open_lock = threading.Lock()
        self._controller = None
        self._thread: threading.Thread | None = None
        self._monitor_thread: threading.Thread | None = None
        self._monitor_stop = threading.Event()
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

    def _ensure_controller(self):
        """Return the shared controller, opening it at most once."""
        with self._lock:
            if self._closed:
                raise RuntimeError("runner closed")
            controller = self._controller
        if controller is not None:
            return controller

        # The monitor and a SpaceMouse trigger can start at the same time.
        # Serialize the scan/open sequence so they cannot create two RM1010
        # owners or close one another's controller.
        with self._open_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("runner closed")
                controller = self._controller
            if controller is not None:
                return controller

            local_controller = None
            try:
                local_controller = self._controller_factory()
                local_controller.open()
                with self._lock:
                    if self._closed:
                        raise RuntimeError("runner closed")
                    self._controller = local_controller
                return local_controller
            except BaseException:
                if local_controller is not None:
                    try:
                        local_controller.close()
                    except Exception:
                        pass
                raise

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

    def start_p10_monitor(self, *, poll_period_s: float = 0.05) -> None:
        """Poll the RM1010 P10 falling edge using the shared controller.

        The monitor is intentionally opt-in so standalone unit users and the
        SpaceMouse-only tool do not open an RM1010 unexpectedly.  Integrated
        G2 workflows start it after creating the runner; a missing device is
        reported and retried without affecting G2 motion.
        """
        period = float(poll_period_s)
        if period <= 0.0:
            raise ValueError("poll_period_s must be positive")
        with self._lock:
            if self._closed:
                raise RuntimeError("runner closed")
            if self._monitor_thread is not None and self._monitor_thread.is_alive():
                return
            self._monitor_stop.clear()
            thread = self._thread_factory(
                target=self._monitor_loop,
                args=(period,),
                name="g2-dianpi-p10-monitor",
                daemon=True,
            )
            self._monitor_thread = thread
            try:
                thread.start()
            except BaseException:
                self._monitor_thread = None
                raise

    def _monitor_loop(self, poll_period_s: float) -> None:
        previous_start = 1
        while not self._monitor_stop.is_set():
            try:
                controller = self._ensure_controller()
                states = controller.read_inputs()
                if states is None:
                    raise RuntimeError("RM1010 输入读取失败")
                current_start = int(states.get(10, 1))
                if current_start == 0 and previous_start == 1:
                    self.trigger()
                previous_start = current_start
            except BaseException as exc:
                if self._monitor_stop.is_set():
                    return
                with self._lock:
                    self._last_error = exc
                if isinstance(exc, (ImportError, ModuleNotFoundError)):
                    # A missing dianpi package cannot be fixed by retrying.
                    # Report once and leave the RM1010 monitor off instead of
                    # logging an error every poll period.
                    self._emit({
                        "event": "dianpi_unavailable",
                        "source": "p10_monitor",
                        "error": str(exc),
                        "hint": "请确认工程根目录在 PYTHONPATH 中，或用 scripts/run_gdk_sam3_python.sh 启动",
                    })
                    return
                self._emit({"event": "dianpi_error", "source": "p10_monitor", "error": str(exc)})
                # A missed read must not suppress a real low edge on the
                # next successful read.
                previous_start = 1
                self._monitor_stop.wait(max(0.5, poll_period_s))
                continue
            self._monitor_stop.wait(poll_period_s)

    def _run(self) -> None:
        try:
            controller = self._ensure_controller()
            self._emit({"event": "dianpi_started"})
            controller.run_cycle()
            self._emit({"event": "dianpi_finished"})
        except BaseException as exc:
            with self._lock:
                self._last_error = exc
            self._emit({"event": "dianpi_error", "error": str(exc)})
        finally:
            with self._lock:
                self._running = False
                self._thread = None

    def close(self, timeout_s: float = 1.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            monitor_thread = self._monitor_thread
            self._monitor_stop.set()
            thread = self._thread
            controller = self._controller
        if monitor_thread is not None and monitor_thread is not threading.current_thread():
            monitor_thread.join(timeout=max(0.0, float(timeout_s)))
        if monitor_thread is not None and monitor_thread.is_alive():
            self._emit({"event": "dianpi_error", "error": "P10 监控线程未能在关闭期限内停止"})
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
