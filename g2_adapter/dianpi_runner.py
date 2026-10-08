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
    DISABLED = "disabled"


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
        self._trigger_enabled = True
        self._automatic_only = False
        self._gpio_trigger_override = None

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

    def set_trigger_policy(self, *, enabled: bool, automatic_only: bool) -> tuple[bool, bool]:
        with self._lock:
            if self._running:
                raise RuntimeError("电批正在运行，不能切换演示模式")
            previous = (self._trigger_enabled, self._automatic_only)
            self._trigger_enabled = bool(enabled)
            self._automatic_only = bool(automatic_only)
            return previous

    def set_gpio_trigger_override(self, enabled: bool | None) -> bool | None:
        """Override only the physical P10 source; None restores normal policy."""
        with self._lock:
            previous = self._gpio_trigger_override
            self._gpio_trigger_override = None if enabled is None else bool(enabled)
            return previous

    def trigger(self, *, automatic: bool = False, source: str = "manual") -> DianpiTriggerResult:
        with self._lock:
            if source == "gpio" and self._gpio_trigger_override is not None:
                allowed = self._gpio_trigger_override
            else:
                allowed = self._trigger_enabled and (not self._automatic_only or automatic)
            if not allowed:
                return DianpiTriggerResult.DISABLED
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
                    self.trigger(source="gpio")
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
                # Reconnection establishes a baseline, not a falling edge.
                # A held-low start must never replay a failed cycle.
                previous_start = None
                # Drop a broken idle connection so the next attempt really
                # reopens USB. Never close a controller owned by a cycle.
                with self._open_lock:
                    with self._lock:
                        broken = self._controller if not self._running else None
                        if broken is not None:
                            self._controller = None
                    if broken is not None:
                        try:
                            broken.close()
                        except Exception:
                            pass
                self._monitor_stop.wait(5.0)
                continue
            self._monitor_stop.wait(poll_period_s)

    def _run(self) -> None:
        try:
            controller = self._ensure_controller()
            self._emit({"event": "dianpi_started"})
            result = controller.run_cycle()
            if result is False:
                with self._lock:
                    self._last_error = RuntimeError("电批运行超时，未确认完成")
            self._emit({"event": "dianpi_timeout" if result is False else "dianpi_finished"})
        except BaseException as exc:
            with self._lock:
                self._last_error = exc
            self._emit({"event": "dianpi_error", "error": str(exc)})
        finally:
            with self._lock:
                self._running = False
                self._thread = None

    def stop_current_cycle(self, timeout_s: float = 1.0) -> None:
        """Stop the current tool cycle while preserving the reusable connection."""
        with self._lock:
            controller, thread = self._controller, self._thread
        if thread is None:
            return
        if controller is None or not hasattr(controller, "stop_cycle"):
            raise RuntimeError("电批控制器不支持停止当前周期")
        controller.stop_cycle()
        if thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        if thread.is_alive():
            raise RuntimeError("电批周期停止超时")

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
