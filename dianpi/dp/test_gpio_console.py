import threading

from dianpi.dp.gpio_console import GpioConsole, format_gpio_snapshot


class BlockingController:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.cycles = 0
        self.stopped = False
        self.closed = False

    def run_cycle(self):
        self.cycles += 1
        self.started.set()
        self.release.wait(1)

    def stop_cycle(self):
        self.stopped = True
        self.release.set()

    def close(self):
        self.closed = True


def test_initial_gpio_snapshot_prints_all_sixteen_pins_on_one_line():
    line = format_gpio_snapshot({pin: pin % 2 for pin in range(16)})
    assert line.count('P') == 16
    assert 'P00 正转=0' in line
    assert 'P03 OK=1' in line
    assert 'P05=1' in line
    assert 'P15=1' in line
    assert '\n' not in line


def test_later_snapshot_prints_complete_line_even_when_state_does_not_change():
    after = {pin: (0 if pin in (3, 5) else 1) for pin in range(16)}
    line = format_gpio_snapshot(after)
    assert line.count('P') == 16
    assert 'P03 OK=0' in line
    assert 'P05=0' in line
    assert 'P00 正转=1' in line
    assert format_gpio_snapshot(after) == line


def test_space_trigger_rejects_second_cycle_and_close_stops_active_cycle():
    controller = BlockingController()
    messages = []
    console = GpioConsole(controller, messages.append)
    assert console.trigger()
    assert controller.started.wait(1)
    assert not console.trigger()
    assert controller.cycles == 1
    console.close()
    assert controller.stopped
    assert controller.closed
