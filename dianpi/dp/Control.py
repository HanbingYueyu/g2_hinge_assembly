#-*- coding: utf-8 -*-
"""
RM1010 电批 IO 控制。

引脚分配:
  输出 (Mode=1): P0 正转、P1 反转、P2 自由、P8 电磁阀、P9 送钉机
  输入 (Mode=0, Pull=1): P3 OK、P4 ERR、P10 启动
  输出 (Mode=1, Pull=2): P11 信号源，初始低电平

``DianpiController`` 提供可复用的打开、单周期和清理 API。直接运行本
文件时仍通过 P10 低电平下降沿触发周期。
"""

from ctypes import *
import os
import sys
import threading
import time

# ---- 解决 Windows Code Runner 中文乱码 ----
if sys.platform == "win32":
    os.system("chcp 65001 > nul")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---- 添加公共库路径 ----
public_path = os.path.normpath(
    os.path.dirname(os.path.abspath(__file__)) + "/../public"
)
if public_path not in sys.path:
    sys.path.append(public_path)
from librockmong import *
from device import *
from gpio import *


# ============================================================
# 引脚定义
# ============================================================
PIN_FORWARD = 0
PIN_REVERSE = 1
PIN_FREE = 2
PIN_OK = 3
PIN_ERR = 4
PIN_SOLENOID = 8
PIN_FEEDER = 9
PIN_START = 10
PIN_SIGNAL = 11

OUTPUT_PINS = [PIN_FORWARD, PIN_REVERSE, PIN_FREE, PIN_SOLENOID, PIN_FEEDER]
OUTPUT_PINS_LOW = [PIN_SIGNAL]
OUTPUT_ALL = OUTPUT_PINS + OUTPUT_PINS_LOW
INPUT_PINS = [PIN_OK, PIN_ERR, PIN_START]

PIN_NAMES = {
    PIN_FORWARD: "P0-正转",
    PIN_REVERSE: "P1-反转",
    PIN_FREE: "P2-自由",
    PIN_OK: "P3-OK",
    PIN_ERR: "P4-ERR",
    PIN_SOLENOID: "P8-电磁阀",
    PIN_FEEDER: "P9-送钉机",
    PIN_START: "P10-启动",
    PIN_SIGNAL: "P11-信号源",
}


class DianpiError(RuntimeError):
    """RM1010 扫描、初始化或 IO 操作失败。"""


class DianpiBusyError(DianpiError):
    """已有一个电批周期正在运行。"""


def write_single_pin(sn, pin, state):
    """使用默认 RM1010 API 写单个引脚。"""
    tx = (IO_WriteMultiPin_TxStruct_t * 1)()
    rx = (IO_WriteMultiPin_RxStruct_t * 1)()
    tx[0].Pin = pin
    tx[0].PinState = state
    return IO_WriteMultiPin(sn, tx, rx, 1)


def read_all_inputs(sn, read_tx, read_rx, count):
    """读取输入引脚；底层失败或单路失败时返回 None。"""
    ret = IO_ReadMultiPin(sn, read_tx, read_rx, count)
    if ret < 0:
        return None
    states = {}
    for i in range(count):
        if read_rx[i].Ret != 0:
            return None
        states[read_tx[i].Pin] = read_rx[i].PinState
    return states


class DianpiController:
    """可复用的 RM1010 控制器，默认使用仓库中的 ctypes API。

    所有底层函数和时钟都可注入，使电批时序可以在不连接 RM1010 的情况
    下测试。控制器实例可以执行多个串行周期，生命周期由 ``open``/``close``
    管理。
    """

    def __init__(
        self,
        *,
        scan_fn=Device_Scan,
        open_fn=Device_Open,
        close_fn=Device_Close,
        init_fn=IO_InitMultiPin,
        read_fn=IO_ReadMultiPin,
        write_fn=IO_WriteMultiPin,
        sleep_fn=time.sleep,
        monotonic_fn=time.monotonic,
        output_fn=print,
    ) -> None:
        self._scan_fn = scan_fn
        self._open_fn = open_fn
        self._close_fn = close_fn
        self._init_fn = init_fn
        self._read_fn = read_fn
        self._write_fn = write_fn
        self._sleep_fn = sleep_fn
        self._monotonic_fn = monotonic_fn
        self._output_fn = output_fn
        self._serial_number: int | None = None
        self._opened = False
        self._state_lock = threading.Lock()
        # ``read_inputs`` is used by both the P10 monitor and the cycle
        # thread.  Protect the reusable ctypes buffers and serialize reads
        # with output writes at the RM1010 boundary.
        self._io_lock = threading.Lock()
        self._cycle_lock = threading.Lock()
        self._cycle_running = False
        self._stop_event = threading.Event()
        self._input_report_lock = threading.Lock()
        self._reported_input_states = {PIN_OK: 1, PIN_ERR: 1}
        self._read_tx = None
        self._read_rx = None

    @property
    def serial_number(self) -> int | None:
        return self._serial_number

    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def cycle_running(self) -> bool:
        with self._state_lock:
            return self._cycle_running

    def _require_open(self) -> int:
        if not self._opened or self._serial_number is None:
            raise DianpiError("RM1010 设备尚未打开")
        return self._serial_number

    def _write_pin(self, pin: int, state: int) -> None:
        with self._io_lock:
            sn = self._require_open()
            tx = (IO_WriteMultiPin_TxStruct_t * 1)()
            rx = (IO_WriteMultiPin_RxStruct_t * 1)()
            tx[0].Pin = pin
            tx[0].PinState = state
            ret = self._write_fn(sn, tx, rx, 1)
            if int(ret) < 0 or rx[0].Ret != 0:
                raise DianpiError(
                    f"RM1010 写入 {PIN_NAMES.get(pin, f'P{pin}')}={state} 失败: "
                    f"ret={int(ret)}, pin_ret={int(rx[0].Ret)}"
                )

    def _restore_cycle_outputs(self) -> None:
        for pin in (PIN_FORWARD, PIN_SOLENOID, PIN_FEEDER):
            try:
                self._write_pin(pin, 1)
            except DianpiError as exc:
                self._output_fn(f"dianpi_error: 清理 {PIN_NAMES[pin]} 失败: {exc}")

    def open(self) -> None:
        """扫描并打开第一台设备，配置既有引脚并写安全初始电平。"""
        with self._state_lock:
            if self._opened:
                return

        serial_numbers = (c_int * 20)()
        ret = self._scan_fn(byref(serial_numbers))
        if int(ret) < 0:
            raise DianpiError(f"设备扫描错误: {int(ret)}")
        if int(ret) == 0:
            raise DianpiError("未发现 RM1010 设备")
        serial_number = int(serial_numbers[0])
        opened = False
        try:
            ret = self._open_fn(serial_number)
            if int(ret) < 0:
                raise DianpiError(f"设备打开错误: {int(ret)}")
            opened = True
            self._serial_number = serial_number
            self._opened = True

            total_pins = len(OUTPUT_ALL) + len(INPUT_PINS)
            init_tx = (IO_InitMultiPin_TxStruct_t * total_pins)()
            init_rx = (IO_InitMultiPin_RxStruct_t * total_pins)()
            index = 0
            for pin in OUTPUT_PINS:
                init_tx[index].Pin = pin
                init_tx[index].Mode = 1
                init_tx[index].Pull = 0
                index += 1
            for pin in OUTPUT_PINS_LOW:
                init_tx[index].Pin = pin
                init_tx[index].Mode = 1
                init_tx[index].Pull = 2
                index += 1
            for pin in INPUT_PINS:
                init_tx[index].Pin = pin
                init_tx[index].Mode = 0
                init_tx[index].Pull = 1
                index += 1

            ret = self._init_fn(serial_number, init_tx, init_rx, total_pins)
            failed = [
                (int(init_tx[i].Pin), int(init_rx[i].Ret))
                for i in range(total_pins)
                if init_rx[i].Ret != 0
            ]
            if int(ret) < 0 or failed:
                raise DianpiError(f"RM1010 引脚初始化失败: ret={int(ret)}, pins={failed}")

            for pin in OUTPUT_PINS:
                self._write_pin(pin, 1)
            for pin in OUTPUT_PINS_LOW:
                self._write_pin(pin, 0)

            self._read_tx = (IO_ReadMultiPin_TxStruct_t * len(INPUT_PINS))()
            self._read_rx = (IO_ReadMultiPin_RxStruct_t * len(INPUT_PINS))()
            for index, pin in enumerate(INPUT_PINS):
                self._read_tx[index].Pin = pin
            self._output_fn(f"设备连接成功! SN={serial_number}")
        except Exception:
            self._opened = False
            self._serial_number = None
            self._read_tx = None
            self._read_rx = None
            if opened:
                try:
                    self._close_fn(serial_number)
                except Exception:
                    pass
            raise

    def read_inputs(self) -> dict[int, int] | None:
        """读取 P3/P4/P10；底层失败返回 None。"""
        with self._io_lock:
            sn = self._require_open()
            if self._read_tx is None or self._read_rx is None:
                raise DianpiError("RM1010 输入结构尚未初始化")
            ret = self._read_fn(sn, self._read_tx, self._read_rx, len(INPUT_PINS))
            if int(ret) < 0:
                return None
            states = {}
            for index, pin in enumerate(INPUT_PINS):
                if self._read_rx[index].Ret != 0:
                    return None
                states[pin] = int(self._read_rx[index].PinState)
            return states

    def _report_inputs(self, states: dict[int, int]) -> None:
        messages = []
        with self._input_report_lock:
            for pin, label in ((PIN_OK, "OK"), (PIN_ERR, "ERR")):
                current = int(states.get(pin, 1))
                previous = self._reported_input_states[pin]
                if previous != 0 and current == 0:
                    messages.append(f">>> [{label}] 信号收到! <<<")
                self._reported_input_states[pin] = current
        for message in messages:
            self._output_fn(message)

    def _wait_and_report(self, duration_s: float) -> None:
        deadline = self._monotonic_fn() + duration_s
        while True:
            if self._stop_event.is_set():
                raise DianpiError("电批周期已停止")
            remaining = deadline - self._monotonic_fn()
            if remaining <= 0:
                return
            states = self.read_inputs()
            if states is None:
                raise DianpiError("RM1010 输入读取失败")
            self._report_inputs(states)
            self._sleep_fn(min(0.05, remaining))

    def run_cycle(self) -> None:
        """执行一个完整电批周期；运行中拒绝并发周期。"""
        self._require_open()
        if not self._cycle_lock.acquire(blocking=False):
            raise DianpiBusyError("电批周期正在运行")
        with self._state_lock:
            self._cycle_running = True
        self._stop_event.clear()
        completed = False
        try:
            self._write_pin(PIN_FEEDER, 0)
            self._sleep_fn(0.05)
            self._write_pin(PIN_FEEDER, 1)
            self._output_fn("[步骤 1/4] 送钉机脉冲: P9 LOW 50ms -> HIGH  ✓")

            self._output_fn("[步骤 2/4] 等待 2 秒...")
            self._wait_and_report(2.0)

            self._output_fn("[步骤 3/4] 电磁阀开启: P8 LOW (10s)")
            self._write_pin(PIN_SOLENOID, 0)
            self._wait_and_report(2.0)

            self._write_pin(PIN_FORWARD, 0)
            self._output_fn("  [> 正转启动: P0 LOW (8s)")
            self._wait_and_report(8.0)

            self._write_pin(PIN_FORWARD, 1)
            self._write_pin(PIN_SOLENOID, 1)
            self._output_fn("  [> 正转停止 (8s) + 电磁阀关闭 (10s)  ✓")
            completed = True
        finally:
            if not completed:
                self._restore_cycle_outputs()
            with self._state_lock:
                self._cycle_running = False
            self._cycle_lock.release()

    def stop_cycle(self) -> None:
        """请求当前周期尽快停止，并由周期清理输出引脚。"""
        with self._state_lock:
            if self._cycle_running:
                self._stop_event.set()

    def close(self) -> None:
        """恢复安全电平并关闭设备；可重复调用。"""
        with self._state_lock:
            if not self._opened or self._serial_number is None:
                return
            if self._cycle_running:
                raise DianpiBusyError("电批周期正在运行，不能关闭设备")
            serial_number = self._serial_number

        cleanup_errors = []
        for pin in OUTPUT_PINS:
            try:
                self._write_pin(pin, 1)
            except DianpiError as exc:
                cleanup_errors.append(str(exc))
        try:
            self._write_pin(PIN_SIGNAL, 0)
        except DianpiError as exc:
            cleanup_errors.append(str(exc))
        try:
            ret = self._close_fn(serial_number)
            if int(ret) < 0:
                cleanup_errors.append(f"设备关闭错误: {int(ret)}")
        finally:
            self._opened = False
            self._serial_number = None
            self._read_tx = None
            self._read_rx = None
        if cleanup_errors:
            self._output_fn("dianpi_error: " + "; ".join(cleanup_errors))


def main() -> None:
    controller = DianpiController()
    prev_state = {PIN_START: 1}
    try:
        controller.open()
        print("=" * 55)
        print("  系统就绪, 等待启动信号 (P10 下降沿)...")
        print("  按 Ctrl+C 退出")
        print("=" * 55)
        while True:
            states = controller.read_inputs()
            if states is None:
                print("读取错误!")
                time.sleep(0.1)
                continue

            controller._report_inputs(states)
            curr_start = states.get(PIN_START, 1)
            if curr_start == 0 and prev_state[PIN_START] == 1:
                print("\n*** 收到启动信号! 开始执行时序... ***\n")
                try:
                    controller.run_cycle()
                except DianpiBusyError as exc:
                    print(f"dianpi_error: {exc}")
                print("\n*** 时序执行完毕! 继续等待启动信号... ***\n")
                states = controller.read_inputs()
                if states:
                    controller._report_inputs(states)
                    prev_state[PIN_START] = states.get(PIN_START, 1)
                continue

            prev_state[PIN_START] = curr_start
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")
    except DianpiError as exc:
        print(f"dianpi_error: {exc}")
    finally:
        print("正在清理...")
        controller.close()


if __name__ == "__main__":
    main()
