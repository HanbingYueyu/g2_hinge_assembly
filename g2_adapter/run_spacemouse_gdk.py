#!/usr/bin/env python3
"""SpaceMouse Compact -> selected G2 end link; default is offline preview."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import signal
import sys
import time

try:
    from .control import G2Controller
    from .spacemouse_input import CompactHID, discover_devices, parse_axis_map
    from .spacemouse_teleop import PERIOD_S, SWITCH_DELAY_S, PreviewController, TeleopConfig, TeleopSession
except ImportError:
    from control import G2Controller
    from spacemouse_input import CompactHID, discover_devices, parse_axis_map
    from spacemouse_teleop import PERIOD_S, SWITCH_DELAY_S, PreviewController, TeleopConfig, TeleopSession


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='SpaceMouse 控制 G2 末端；左按钮松开仅平移、按住完整 6D、双击切手，右按钮开合夹爪。默认不连接机器人。')
    parser.add_argument('--list-devices', action='store_true', help='列出 USB SpaceMouse Compact 后退出')
    parser.add_argument('--device', help='指定 /dev/hidrawN 或 udev 固定别名（如 /dev/spacemouse-compact）；仅接受 Compact 256f:c635')
    parser.add_argument('--side', choices=('left', 'right'), default='right', help='启动时选中的手，默认 right')
    parser.add_argument('--allow-motion', action='store_true', help='连接 GDK 并允许末端和夹爪运动')
    parser.add_argument('--speed-mm-s', type=float, default=10, help='平移合速度上限 (0,30] mm/s')
    parser.add_argument('--rotation-deg-s', type=float, default=10, help='旋转合速度上限 (0,28.6] deg/s')
    parser.add_argument('--deadzone', type=float, default=.12, help='轴死区 (0,0.5]')
    parser.add_argument('--axis-map', default='-2,-1,-3,-5,-4,-6', help='输出 XYZ/RXYZ 对应的原始轴；负号反向，使用 --axis-map=...')
    parser.add_argument('--duration-s', type=float, default=0, help='运行时长；0 表示直到 Ctrl+C')
    args = parser.parse_args(argv)
    try:
        if not math.isfinite(args.duration_s) or args.duration_s < 0:
            raise ValueError('--duration-s 必须是非负有限数')
        config = TeleopConfig(
            side=args.side, speed_m_s=args.speed_mm_s / 1000,
            angular_speed_rad_s=math.radians(args.rotation_deg_s),
            deadzone=args.deadzone, axis_map=parse_axis_map(args.axis_map),
        )
    except ValueError as exc:
        parser.error(str(exc))
    return args, config


def emit_json(document):
    print(json.dumps(document, ensure_ascii=False, allow_nan=False), flush=True)


def ensure_position_mode(controller):
    """Enter GDK mode 1 before the first teleoperation pose command."""
    before = controller.motion_status_summary()
    if before['mode'] not in {0, 1} or before['error_code'] != 0:
        raise RuntimeError(f"启动前拒绝切换：GDK 状态异常 {before}")
    if before['control_mode'] != 1:
        controller.set_control_mode(1)
    after = controller.motion_status_summary()
    if after['mode'] not in {0, 1} or after['control_mode'] != 1 or after['error_code'] != 0:
        raise RuntimeError(f"启动前未进入健康 control_mode=1：{after}")
    return before, after


def run_loop(source, session, *, duration_s=0, clock=time.monotonic, sleep=time.sleep, emit=emit_json):
    start = deadline = clock()
    last_output = -math.inf
    last_event = None
    try:
        while not duration_s or clock() - start < duration_s:
            # Check timing *before* draining the kernel queue, so a paused
            # process cannot relabel old queued motion as fresh and execute it.
            now = clock()
            if now - deadline > .06:
                raise RuntimeError('50 Hz 循环延迟过大，停止遥操作')
            frame = source.poll()
            now = clock()
            event = session.tick(frame, now)
            if event != last_event or now - last_output >= .2:
                emit({'event': event, 'side': session.side, 'raw_axes': frame.axes,
                      'buttons': frame.buttons,
                      'target': asdict(session.target) if session.target is not None else None})
                last_output, last_event = now, event
            deadline += PERIOD_S
            sleep(max(0, deadline - clock()))
    finally:
        try:
            session.stop()
        except Exception as exc:
            print(f'结束保持命令失败，等待末条命令有效期结束：{exc}', file=sys.stderr)
        # No robot calls remain in flight: the loop is the only writer.
        sleep(SWITCH_DELAY_S)


def main(argv=None) -> int:
    args, config = parse_args(argv)
    if args.list_devices:
        emit_json({'devices': discover_devices()})
        return 0
    initialized = False
    previous_sigterm = None
    try:
        with CompactHID(args.device) as source:
            if args.allow_motion:
                try:
                    import agibot_gdk
                except ImportError as exc:
                    raise RuntimeError('agibot_gdk 不可用；请通过 scripts/run_gdk_python.sh 启动') from exc
                if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
                    raise RuntimeError('GDK 初始化失败')
                initialized = True
                robot = agibot_gdk.Robot()
                time.sleep(.5)
                controller = G2Controller(agibot_gdk, robot, allow_motion=True)
                mode_before, mode_after = ensure_position_mode(controller)
                controller.checked_arm_state()
            else:
                controller = PreviewController()
                mode_before = mode_after = None
            session = TeleopSession(controller, config, clock=time.monotonic)

            def terminate(signum, frame):
                raise KeyboardInterrupt

            previous_sigterm = signal.signal(signal.SIGTERM, terminate)
            emit_json({'mode': 'robot' if args.allow_motion else 'preview_no_robot',
                       'device': source.path, 'config': asdict(config),
                       'control_mode_before': mode_before,
                       'control_mode_after': mode_after,
                       'instructions': '先轻推/扭动后回中，并按下再松开任一按钮以完成初始输入检测；ready 后推动即运动。左按钮松开只平移、按住完整 6D，左按钮双击切手，右按钮单击开合夹爪，Ctrl+C 退出。预览位姿原点为零。'})
            run_loop(source, session, duration_s=args.duration_s)
        return 0
    except KeyboardInterrupt:
        print('SpaceMouse 遥操作已退出。', file=sys.stderr)
        return 0
    except Exception as exc:
        print(f'SpaceMouse 遥操作失败：{exc}', file=sys.stderr)
        return 1
    finally:
        if previous_sigterm is not None:
            signal.signal(signal.SIGTERM, previous_sigterm)
        if initialized:
            agibot_gdk.gdk_release()


if __name__ == '__main__':
    raise SystemExit(main())
