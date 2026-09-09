#-*- coding: utf-8 -*-
"""
控制程序 — 基于 RM1010 IO 多引脚控制

引脚分配:
  输出 (Mode=1, 初始 HIGH):
    P0  — 正转 (Forward)
    P1  — 反转 (Reverse)
    P2  — 自由 (Free)
    P8  — 电磁阀 (Solenoid valve)
    P9  — 送钉机 (Screw feeder)

  输入 (Mode=0, Pull=1 上拉):
    P3  — OK
    P4  — ERR
    P10 — 启动 (Start)

控制逻辑:
  1. 自动扫描并连接设备
  2. 循环读取输入引脚状态
  3. 检测到启动信号 (P10 下降沿) 后执行:
     a. P9 送钉机低电平 1ms
     b. 延时 2s
     c. P8 电磁阀低电平 10s
     d. P0 正转低电平 8s
  4. 检测到 OK / ERR 信号时输出到终端
"""

from ctypes import *
import os
import sys
import time

# ---- 解决 Windows Code Runner 中文乱码 ----
if sys.platform == "win32":
    # 方法1: 将控制台代码页切换为 UTF-8
    os.system("chcp 65001 > nul")
    # 方法2: 如果 stdout 已打开, 用 UTF-8 重新包装
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---- 添加公共库路径 ----
public_path = os.path.normpath(
    os.path.dirname(os.path.abspath(__file__)) + "/../public"
)
sys.path.append(public_path)
from librockmong import *
from device import *
from gpio import *

# ============================================================
# 引脚定义
# ============================================================
PIN_FORWARD  = 0   # P0: 正转
PIN_REVERSE  = 1   # P1: 反转
PIN_FREE     = 2   # P2: 自由
PIN_OK       = 3   # P3: OK (输入)
PIN_ERR      = 4   # P4: ERR (输入)
PIN_SOLENOID = 8   # P8: 电磁阀
PIN_FEEDER   = 9   # P9: 送钉机
PIN_START    = 10  # P10: 启动 (输入)
PIN_SIGNAL   = 11  # P11: 信号源 (输出, 下拉, 初始低电平)

OUTPUT_PINS      = [PIN_FORWARD, PIN_REVERSE, PIN_FREE, PIN_SOLENOID, PIN_FEEDER]  # 初始高电平
OUTPUT_PINS_LOW  = [PIN_SIGNAL]   # 初始低电平 (含下拉)
OUTPUT_ALL       = OUTPUT_PINS + OUTPUT_PINS_LOW
INPUT_PINS       = [PIN_OK, PIN_ERR, PIN_START]

PIN_NAMES = {
    PIN_FORWARD:  "P0-正转",
    PIN_REVERSE:  "P1-反转",
    PIN_FREE:     "P2-自由",
    PIN_OK:       "P3-OK",
    PIN_ERR:      "P4-ERR",
    PIN_SOLENOID: "P8-电磁阀",
    PIN_FEEDER:   "P9-送钉机",
    PIN_START:    "P10-启动",
    PIN_SIGNAL:   "P11-信号源",
}


# ============================================================
# 辅助函数
# ============================================================
def write_single_pin(sn, pin, state):
    """写单个引脚: state=0 低电平, state=1 高电平"""
    tx = (IO_WriteMultiPin_TxStruct_t * 1)()
    rx = (IO_WriteMultiPin_RxStruct_t * 1)()
    tx[0].Pin = pin
    tx[0].PinState = state
    return IO_WriteMultiPin(sn, tx, rx, 1)


def read_all_inputs(sn, read_tx, read_rx, count):
    """读取所有输入引脚, 返回 {pin: state} 字典; 失败返回 None"""
    ret = IO_ReadMultiPin(sn, read_tx, read_rx, count)
    if ret < 0:
        return None
    states = {}
    for i in range(count):
        states[read_tx[i].Pin] = read_rx[i].PinState
    return states


# ============================================================
# 主程序
# ============================================================
if __name__ == '__main__':

    # ---------- 1. 自动扫描设备 ----------
    SerialNumbers = (c_int * 20)()
    ret = Device_Scan(byref(SerialNumbers))

    if ret < 0:
        print("设备扫描错误: %d" % ret)
        exit()
    elif ret == 0:
        print("未发现设备!")
        exit()
    else:
        for i in range(ret):
            print("Dev%d 序列号: %d" % (i, SerialNumbers[i]))

    sn = SerialNumbers[0]
    print("正在连接设备 SN=%d ..." % sn)

    # ---------- 2. 打开设备 ----------
    ret = Device_Open(sn)
    if ret < 0:
        print("设备打开错误: %d" % ret)
        exit()
    print("设备连接成功!")

    # ---------- 3. 初始化所有 IO 引脚 ----------
    total_pins = len(OUTPUT_ALL) + len(INPUT_PINS)
    init_tx = (IO_InitMultiPin_TxStruct_t * total_pins)()
    init_rx = (IO_InitMultiPin_RxStruct_t * total_pins)()

    idx = 0
    # 输出引脚 (初始高电平): Mode=1 (输出), Pull=0 (无上下拉)
    for pin in OUTPUT_PINS:
        init_tx[idx].Pin  = pin
        init_tx[idx].Mode = 1
        init_tx[idx].Pull = 0
        idx += 1
    # 输出引脚 (初始低电平): Mode=1 (输出), Pull=2 (下拉)
    for pin in OUTPUT_PINS_LOW:
        init_tx[idx].Pin  = pin
        init_tx[idx].Mode = 1
        init_tx[idx].Pull = 2
        idx += 1
    # 输入引脚: Mode=0 (输入), Pull=1 (上拉)
    for pin in INPUT_PINS:
        init_tx[idx].Pin  = pin
        init_tx[idx].Mode = 0
        init_tx[idx].Pull = 1
        idx += 1

    ret = IO_InitMultiPin(sn, init_tx, init_rx, total_pins)
    if ret < 0:
        for i in range(total_pins):
            if init_rx[i].Ret != 0:
                print("初始化错误 引脚%d: %d" % (init_tx[i].Pin, init_rx[i].Ret))

    # ---------- 4. 输出引脚初始电平 ----------
    write_tx = (IO_WriteMultiPin_TxStruct_t * len(OUTPUT_ALL))()
    write_rx = (IO_WriteMultiPin_RxStruct_t * len(OUTPUT_ALL))()

    # P0~P9: 初始高电平
    for i, pin in enumerate(OUTPUT_PINS):
        write_tx[i].Pin      = pin
        write_tx[i].PinState = 1

    # P11: 初始低电平
    for i, pin in enumerate(OUTPUT_PINS_LOW):
        idx_w = len(OUTPUT_PINS) + i
        write_tx[idx_w].Pin      = pin
        write_tx[idx_w].PinState = 0

    ret = IO_WriteMultiPin(sn, write_tx, write_rx, len(OUTPUT_ALL))
    if ret < 0:
        for i in range(len(OUTPUT_ALL)):
            if write_rx[i].Ret != 0:
                print("写错误 引脚%d: %d" % (write_tx[i].Pin, write_rx[i].Ret))

    # ---------- 5. 准备输入读取结构 ----------
    read_tx = (IO_ReadMultiPin_TxStruct_t * len(INPUT_PINS))()
    read_rx = (IO_ReadMultiPin_RxStruct_t * len(INPUT_PINS))()

    for i, pin in enumerate(INPUT_PINS):
        read_tx[i].Pin = pin

    # 上一次引脚状态 (初始化为高电平 = 未触发)
    prev_state = {
        PIN_OK:    1,
        PIN_ERR:   1,
        PIN_START: 1,
    }

    print("=" * 55)
    print("  系统就绪, 等待启动信号 (P10 下降沿)...")
    print("  按 Ctrl+C 退出")
    print("=" * 55)

    # ---------- 6. 主循环 ----------
    try:
        while True:

            # --- 读取所有输入引脚 ---
            states = read_all_inputs(sn, read_tx, read_rx, len(INPUT_PINS))
            if states is None:
                print("读取错误!")
                time.sleep(0.1)
                continue

            curr_ok    = states.get(PIN_OK, 1)
            curr_err   = states.get(PIN_ERR, 1)
            curr_start = states.get(PIN_START, 1)

            # --- OK 信号检测 (下降沿, 低电平有效) ---
            if curr_ok == 0 and prev_state[PIN_OK] == 1:
                print(">>> [OK] 信号收到! <<<")

            # --- ERR 信号检测 (下降沿, 低电平有效) ---
            if curr_err == 0 and prev_state[PIN_ERR] == 1:
                print(">>> [ERR] 信号收到! <<<")

            # --- 启动信号检测 (下降沿, 低电平有效) ---
            if curr_start == 0 and prev_state[PIN_START] == 1:
                print("\n*** 收到启动信号! 开始执行时序... ***\n")

                # ------ a. 送钉机信号: P9 低电平 50ms ------
                write_single_pin(sn, PIN_FEEDER, 0)
                # 精确延时 ~50ms
                t_start = time.perf_counter()
                while time.perf_counter() - t_start < 0.05:
                    pass
                write_single_pin(sn, PIN_FEEDER, 1)
                print("[步骤 1/4] 送钉机脉冲: P9 LOW 50ms -> HIGH  ✓")

                # ------ b. 延时 2s ------
                print("[步骤 2/4] 等待 2 秒...")
                # 在等待期间也监测 OK/ERR
                t0 = time.time()
                while time.time() - t0 < 2.0:
                    states = read_all_inputs(sn, read_tx, read_rx, len(INPUT_PINS))
                    if states:
                        s_ok  = states.get(PIN_OK, 1)
                        s_err = states.get(PIN_ERR, 1)
                        if s_ok == 0:
                            print("  >>> [OK] 信号收到! <<<")
                        if s_err == 0:
                            print("  >>> [ERR] 信号收到! <<<")
                    time.sleep(0.05)

                # ------ c. 电磁阀 + 正转 ------
                # 电磁阀 P8 LOW 10s; 2s 后正转 P0 LOW 8s (同时结束)
                print("[步骤 3/4] 电磁阀开启: P8 LOW (10s)")
                write_single_pin(sn, PIN_SOLENOID, 0)

                forward_started = False
                solenoid_stopped = False
                t0 = time.time()
                while not solenoid_stopped:
                    elapsed = time.time() - t0

                    # 2s 后: 正转启动
                    if not forward_started and elapsed >= 2.0:
                        write_single_pin(sn, PIN_FORWARD, 0)
                        forward_started = True
                        print("  [> 正转启动: P0 LOW (8s)")

                    # 10s 到: 正转 + 电磁阀 同时关闭
                    if not solenoid_stopped and elapsed >= 10.0:
                        write_single_pin(sn, PIN_FORWARD, 1)
                        write_single_pin(sn, PIN_SOLENOID, 1)
                        solenoid_stopped = True
                        print("  [> 正转停止 (8s) + 电磁阀关闭 (10s)  ✓")

                    # 监控 OK/ERR
                    states = read_all_inputs(sn, read_tx, read_rx, len(INPUT_PINS))
                    if states:
                        s_ok  = states.get(PIN_OK, 1)
                        s_err = states.get(PIN_ERR, 1)
                        if s_ok == 0:
                            print("  >>> [OK] 信号收到! <<<")
                        if s_err == 0:
                            print("  >>> [ERR] 信号收到! <<<")
                    time.sleep(0.05)

                print("\n*** 时序执行完毕! 继续等待启动信号... ***\n")

                # 刷新上一次状态, 避免退出时序后重复触发
                states = read_all_inputs(sn, read_tx, read_rx, len(INPUT_PINS))
                if states:
                    prev_state[PIN_OK]    = states.get(PIN_OK, 1)
                    prev_state[PIN_ERR]   = states.get(PIN_ERR, 1)
                    prev_state[PIN_START] = states.get(PIN_START, 1)
                continue   # 跳过本次循环末尾的状态更新

            # --- 更新上一次状态 ---
            prev_state[PIN_OK]    = curr_ok
            prev_state[PIN_ERR]   = curr_err
            prev_state[PIN_START] = curr_start

            time.sleep(0.05)   # 主循环轮询间隔 50ms

    except KeyboardInterrupt:
        print("\n\n程序被用户中断。")

    # ---------- 7. 清理: 输出引脚恢复初始电平 ----------
    print("正在清理...")
    for pin in OUTPUT_PINS:
        write_single_pin(sn, pin, 1)       # P0~P9 恢复高电平
    for pin in OUTPUT_PINS_LOW:
        write_single_pin(sn, pin, 0)       # P11 恢复低电平

    ret = Device_Close(sn)
    if ret < 0:
        print("设备关闭错误: %d" % ret)
    else:
        print("设备已关闭。")
