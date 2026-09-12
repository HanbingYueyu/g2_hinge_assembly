from __future__ import annotations

import sys
import threading
import unittest
import time

import dianpi_runner
from dianpi_runner import DianpiCycleRunner, DianpiTriggerResult


class FakeController:
    def __init__(self, *, block=False, fail_open=False, fail_cycle=False, input_states=None):
        self.block = block
        self.fail_open = fail_open
        self.fail_cycle = fail_cycle
        self.input_states = list(input_states or [])
        self.last_input_state = {3: 1, 4: 1, 10: 1}
        self.open_calls = 0
        self.cycle_calls = 0
        self.read_calls = 0
        self.close_calls = 0
        self.stop_calls = 0
        self.cycle_started = threading.Event()
        self.cycle_finished = threading.Event()
        self.release = threading.Event()

    def open(self):
        self.open_calls += 1
        if self.fail_open:
            raise RuntimeError("scan failed")

    def run_cycle(self):
        self.cycle_calls += 1
        self.cycle_started.set()
        if self.fail_cycle:
            raise RuntimeError("cycle failed")
        if self.block:
            self.release.wait(1.0)
        self.cycle_finished.set()

    def read_inputs(self):
        self.read_calls += 1
        if self.input_states:
            self.last_input_state = self.input_states.pop(0)
        return self.last_input_state

    def close(self):
        self.close_calls += 1

    def stop_cycle(self):
        self.stop_calls += 1
        self.release.set()


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def make_runner(*, controller=None, events=None):
    controller = controller or FakeController()
    factory_calls = []

    def factory():
        factory_calls.append(True)
        return controller

    event_sink = events.append if events is not None else lambda _event: None
    runner = DianpiCycleRunner(controller_factory=factory, event_sink=event_sink)
    return runner, controller, factory_calls


def test_trigger_lazily_opens_and_runs_one_cycle():
    runner, controller, factory_calls = make_runner()
    assert factory_calls == []

    assert runner.trigger() is DianpiTriggerResult.STARTED
    controller.cycle_finished.wait(1.0)

    assert factory_calls == [True]
    assert controller.open_calls == 1
    assert controller.cycle_calls == 1
    runner.close()
    assert controller.close_calls == 1


def test_trigger_while_running_returns_busy_without_second_cycle():
    runner, controller, _factory_calls = make_runner(controller=FakeController(block=True))

    assert runner.trigger() is DianpiTriggerResult.STARTED
    assert controller.cycle_started.wait(1.0)
    assert runner.trigger() is DianpiTriggerResult.BUSY
    assert controller.cycle_calls == 1
    controller.release.set()
    controller.cycle_finished.wait(1.0)
    runner.close()


def test_hardware_error_is_reported_without_raising_to_worker():
    events = []
    runner, controller, _factory_calls = make_runner(
        controller=FakeController(fail_cycle=True), events=events
    )

    assert runner.trigger() is DianpiTriggerResult.STARTED
    wait_until(lambda: runner.last_error is not None)
    assert "cycle failed" in str(runner.last_error)
    wait_until(lambda: any(event["event"] == "dianpi_error" for event in events))
    assert any(event["event"] == "dianpi_error" for event in events)
    runner.close()
    assert controller.close_calls == 1


def test_close_stops_active_cycle_before_closing_controller():
    controller = FakeController(block=True)
    runner, _controller, _factory_calls = make_runner(controller=controller)
    assert runner.trigger() is DianpiTriggerResult.STARTED
    assert controller.cycle_started.wait(1.0)

    runner.close(timeout_s=1.0)

    assert controller.stop_calls == 1
    assert controller.close_calls == 1


def test_p10_monitor_triggers_cycle_on_falling_edge():
    controller = FakeController(input_states=[{3: 1, 4: 1, 10: 1}, {3: 1, 4: 1, 10: 0}])
    runner, _controller, factory_calls = make_runner(controller=controller)

    runner.start_p10_monitor(poll_period_s=0.001)
    wait_until(lambda: controller.cycle_finished.is_set())

    assert factory_calls == [True]
    assert controller.open_calls == 1
    assert controller.cycle_calls == 1
    runner.close()
    assert controller.close_calls == 1


def test_p10_monitor_and_spacemouse_trigger_share_one_controller():
    controller = FakeController(input_states=[{3: 1, 4: 1, 10: 1}])
    runner, _controller, factory_calls = make_runner(controller=controller)

    runner.start_p10_monitor(poll_period_s=0.001)
    wait_until(lambda: controller.open_calls == 1)
    assert runner.trigger() is DianpiTriggerResult.STARTED
    wait_until(lambda: controller.cycle_finished.is_set())

    assert factory_calls == [True]
    assert controller.open_calls == 1
    assert controller.cycle_calls == 1
    runner.close()


class DianpiImportFallbackTests(unittest.TestCase):
    """A plain-script launch must still find the sibling ``dianpi`` package."""

    def test_factory_adds_the_project_root_and_retries_once(self):
        calls = {"count": 0}

        class FakeModule:
            @staticmethod
            def DianpiController():
                return "controller"

        def fake_import(name):
            calls["count"] += 1
            if calls["count"] == 1:
                raise ModuleNotFoundError(name)
            return FakeModule

        original_import = dianpi_runner.importlib.import_module
        original_path = list(sys.path)
        dianpi_runner.importlib.import_module = fake_import
        try:
            self.assertEqual(dianpi_runner._default_controller_factory(), "controller")
        finally:
            dianpi_runner.importlib.import_module = original_import
            sys.path[:] = original_path
        self.assertEqual(calls["count"], 2)
        self.assertIn(str(dianpi_runner.PROJECT_ROOT), sys.path)

    def test_p10_monitor_reports_once_and_stops_when_dianpi_is_missing(self):
        events: list[dict] = []

        def factory():
            raise ModuleNotFoundError("No module named 'dianpi'")

        runner = DianpiCycleRunner(controller_factory=factory, event_sink=events.append)
        runner.start_p10_monitor(poll_period_s=0.001)
        wait_until(lambda: any(event.get("event") == "dianpi_unavailable" for event in events))
        time.sleep(0.05)
        runner.close()
        names = [event["event"] for event in events]
        self.assertEqual(names.count("dianpi_unavailable"), 1)
        self.assertNotIn("dianpi_error", names)
