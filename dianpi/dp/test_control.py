from __future__ import annotations

import threading
import time
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from dianpi.dp.Control import (
    DianpiBusyError,
    DianpiController,
    INPUT_PINS,
    OUTPUT_ALL,
    OUTPUT_PINS,
    OUTPUT_PINS_LOW,
    PIN_ERR,
    PIN_OK,
    PIN_START,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += float(seconds)


class FakeIO:
    def __init__(self) -> None:
        self.clock = FakeClock()
        self.serial_number = 13579
        self.initialized = []
        self.writes = []
        self.reads = []
        self.closed = []
        self.states = {PIN_OK: 1, PIN_ERR: 1, PIN_START: 1}
        self.read_block = threading.Event()
        self.read_started = threading.Event()
        self.block_reads = False

    def scan(self, serial_numbers) -> int:
        serial_numbers._obj[0] = self.serial_number
        return 1

    def open(self, serial_number: int) -> int:
        assert serial_number == self.serial_number
        return 0

    def close(self, serial_number: int) -> int:
        self.closed.append(serial_number)
        return 0

    def init(self, serial_number, tx, rx, count) -> int:
        for index in range(count):
            self.initialized.append((int(tx[index].Pin), int(tx[index].Mode), int(tx[index].Pull)))
            rx[index].Ret = 0
        return 0

    def write(self, serial_number, tx, rx, count) -> int:
        for index in range(count):
            self.writes.append((self.clock.now, int(tx[index].Pin), int(tx[index].PinState)))
            rx[index].Ret = 0
        return 0

    def read(self, serial_number, tx, rx, count) -> int:
        self.reads.append(self.clock.now)
        if self.block_reads and not self.read_block.is_set():
            self.read_started.set()
            self.read_block.wait(1.0)
        for index in range(count):
            rx[index].Ret = 0
            rx[index].PinState = self.states.get(int(tx[index].Pin), 1)
        return 0


def make_controller(io: FakeIO) -> DianpiController:
    return DianpiController(
        scan_fn=io.scan,
        open_fn=io.open,
        close_fn=io.close,
        init_fn=io.init,
        read_fn=io.read,
        write_fn=io.write,
        sleep_fn=io.clock.sleep,
        monotonic_fn=io.clock.monotonic,
        output_fn=lambda *_args, **_kwargs: None,
    )


def opened_controller() -> tuple[DianpiController, FakeIO]:
    io = FakeIO()
    controller = make_controller(io)
    controller.open()
    return controller, io


def writes_for(io: FakeIO, pin: int, state: int):
    return [(timestamp, value) for timestamp, written_pin, value in io.writes if written_pin == pin and value == state]


def test_open_initializes_existing_pins_and_safe_levels():
    controller, io = opened_controller()

    assert [pin for pin, _mode, _pull in io.initialized] == OUTPUT_ALL + INPUT_PINS
    assert [(pin, state) for _timestamp, pin, state in io.writes] == [
        (pin, 1) for pin in OUTPUT_PINS
    ] + [(pin, 0) for pin in OUTPUT_PINS_LOW]
    controller.close()


def test_run_cycle_preserves_pulse_and_delayed_forward_timing():
    controller, io = opened_controller()
    io.writes.clear()

    controller.run_cycle()

    assert writes_for(io, 9, 1)[0][0] == pytest.approx(0.05)
    assert writes_for(io, 8, 0)[0][0] == pytest.approx(2.05)
    assert writes_for(io, 0, 0)[0][0] == pytest.approx(4.05)
    assert writes_for(io, 0, 1)[-1][0] == pytest.approx(12.05)
    assert writes_for(io, 8, 1)[-1][0] == pytest.approx(12.05)


def test_second_cycle_is_rejected_while_first_cycle_is_running():
    io = FakeIO()
    io.block_reads = True
    controller = make_controller(io)
    controller.open()

    cycle_thread = threading.Thread(target=controller.run_cycle)
    cycle_thread.start()
    assert io.read_started.wait(1.0)
    with pytest.raises(DianpiBusyError):
        controller.run_cycle()
    io.read_block.set()
    cycle_thread.join(1.0)
    assert not cycle_thread.is_alive()


def test_close_restores_safe_levels_and_is_idempotent():
    controller, io = opened_controller()
    io.writes.clear()

    controller.close()
    controller.close()

    assert [(pin, state) for _timestamp, pin, state in io.writes] == [
        (pin, 1) for pin in OUTPUT_PINS
    ] + [(pin, 0) for pin in OUTPUT_PINS_LOW]
    assert io.closed == [io.serial_number]
