"""Standalone RM1010 GPIO monitor with Space-triggered screwdriver cycles."""

from __future__ import annotations

import argparse
import select
import sys
import termios
import threading
import tty

from .Control import DianpiController, PIN_NAMES


def format_gpio_snapshot(levels: dict[int, int]) -> str:
    parts = []
    for pin in range(16):
        name = PIN_NAMES.get(pin)
        label = f"P{pin:02d}" + (f" {name.split('-', 1)[1]}" if name else "")
        parts.append(f"{label}={levels[pin]}")
    return "  ".join(parts)


class GpioConsole:
    def __init__(self, controller: DianpiController, output=print) -> None:
        self.controller = controller
        self.output = output
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def trigger(self) -> bool:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._thread = threading.Thread(target=self._run_cycle, name="dianpi-console-cycle")
            self._thread.start()
            return True

    def _run_cycle(self) -> None:
        try:
            self.controller.run_cycle()
        except Exception as exc:
            self.output(f"电批周期失败：{exc}")

    def close(self) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None and thread.is_alive():
            self.controller.stop_cycle()
            thread.join(timeout=3.0)
            if thread.is_alive():
                raise RuntimeError("电批周期尚未停止；设备仍由周期线程占用")
        self.controller.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="监视 RM1010 P0-P15；空格触发一次电批周期，Ctrl+C 退出")
    parser.add_argument("--rate-hz", type=float, default=10.0)
    args = parser.parse_args(argv)
    if not 1.0 <= args.rate_hz <= 50.0:
        parser.error("--rate-hz 必须在 1 到 50 之间")
    if not sys.stdin.isatty():
        parser.error("空格控制需要交互式终端")

    output_lock = threading.Lock()

    def output(message: str) -> None:
        with output_lock:
            sys.stdout.write(f"{message}\n")
            sys.stdout.flush()

    controller = DianpiController(monitor_all_pins=True, output_fn=output)
    controller.open()
    console = GpioConsole(controller, output)
    fd = sys.stdin.fileno()
    original_terminal = None
    try:
        original_terminal = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        output("空格：打一次螺丝；Ctrl+C：停止并退出。请勿同时运行主流程。")
        while True:
            levels = controller.read_gpio_levels()
            output(format_gpio_snapshot(levels))
            readable, _, _ = select.select([fd], [], [], 1.0 / args.rate_hz)
            if readable:
                for key in sys.stdin.buffer.read1(128):
                    if key == 32 and not console.trigger():
                        output("电批周期运行中，忽略重复空格。")
    except KeyboardInterrupt:
        output("收到 Ctrl+C；停止电批并恢复输出电平。")
    finally:
        if original_terminal is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, original_terminal)
        console.close()
        output("已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
