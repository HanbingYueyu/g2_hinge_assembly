#!/usr/bin/env python3
"""Standalone left-arm Cartesian impedance motion; not constant-force control."""

import argparse
import math
import sys
import time

try:
    from .control import CartesianPoseTarget, G2Controller
except ImportError:
    from control import CartesianPoseTarget, G2Controller


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='左臂笛卡尔阻抗移动：保持姿态，沿基座 XYZ 相对移动；不是恒力控制')
    parser.add_argument('--x-mm', type=float, default=0)
    parser.add_argument('--y-mm', type=float, default=0)
    parser.add_argument('--z-mm', type=float, default=20)
    parser.add_argument('--speed-mm-s', type=float, default=5)
    parser.add_argument('--kp', type=float, nargs=6, default=(2000, 2000, 2000, 200, 200, 200), metavar='K', help='XYZ 平移及 XYZ 旋转刚度，共六个值；默认采用实测右臂配置')
    parser.add_argument('--allow-motion', action='store_true', help='确认切换阻抗模式并移动；不指定则只读预览')
    return parser.parse_args(argv)


def validate_args(args):
    if not all(math.isfinite(v) for v in (args.x_mm, args.y_mm, args.z_mm, args.speed_mm_s)) or args.speed_mm_s <= 0:
        raise ValueError('XYZ 必须为有限值，速度必须为正有限值')
    lower = (600, 600, 600, 30, 30, 30)
    upper = (6000, 6000, 6000, 300, 300, 300)
    if len(args.kp) != 6 or any(not math.isfinite(v) or not lo <= v <= hi for v, lo, hi in zip(args.kp, lower, upper)):
        raise ValueError('刚度需六个值：平移 600–6000，旋转 30–300（沿用现有控制接口范围）')


def move_left_impedance(controller, *, delta_mm, speed_mm_s, kp, clock=time.monotonic, sleep=time.sleep):
    origin = controller.read_end_effector_pose('arm_l_end_link')
    delta = tuple(v / 1000.0 for v in delta_mm)
    duration = math.sqrt(sum(v*v for v in delta_mm)) / speed_mm_s
    stream = None
    try:
        controller.enable_left_cartesian_impedance(kp)
        stream = controller.start_left_cartesian_impedance_stream(origin, rate_hz=50, life_time_s=.10)
        started = clock()
        while True:
            stream.raise_if_failed()
            fraction = min(1.0, (clock() - started) / duration) if duration else 1.0
            target = CartesianPoseTarget(tuple(a + fraction*b for a,b in zip(origin.position_m, delta)), origin.orientation_xyzw)
            stream.update_target(target)
            if fraction >= 1:
                break
            sleep(.02)
        # Allow the 50 Hz sender to publish the final target and settle.
        # External force may leave actual position different from the target.
        sleep(.5)
        stream.raise_if_failed()
        actual = controller.read_end_effector_pose('arm_l_end_link')
        return {'duration_s': duration, 'target_position_m': list(target.position_m), 'actual_position_m': list(actual.position_m)}
    finally:
        if stream is not None:
            stream.stop()
        # Rebase to measured feedback before returning to position mode.
        # If this hold fails, do not switch using an unconfirmed target.
        if stream is not None or controller.motion_status_summary()['control_mode'] == 2:
            hold = controller.read_end_effector_pose('arm_l_end_link')
            controller._send_left_cartesian_pose(hold, .10)
            controller.set_control_mode(1)


def main(argv=None):
    args = parse_args(argv)
    gdk = None
    initialized = False
    try:
        validate_args(args)
        import agibot_gdk as gdk
        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise RuntimeError('GDK 初始化失败')
        initialized = True
        robot = gdk.Robot()
        time.sleep(1)
        controller = G2Controller(gdk, robot, allow_motion=args.allow_motion)
        current = controller.read_end_effector_pose('arm_l_end_link')
        print(f'左臂当前 XYZ(m)：{current.position_m}；基座偏移(mm)：{(args.x_mm, args.y_mm, args.z_mm)}', flush=True)
        if not args.allow_motion:
            print('只读预览；添加 --allow-motion 才切换阻抗模式并移动。')
            return 0
        print('开始左臂笛卡尔阻抗移动；请确认末端负载正确，不要同时运行其他控制脚本。', flush=True)
        result = move_left_impedance(controller, delta_mm=(args.x_mm, args.y_mm, args.z_mm), speed_mm_s=args.speed_mm_s, kp=args.kp)
        print(f"移动目标发送完成，已回到位控；实际 XYZ(m)：{result['actual_position_m']}", flush=True)
        return 0
    except KeyboardInterrupt:
        print('已中断。', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'阻抗移动 demo 失败：{exc}', file=sys.stderr)
        return 1
    finally:
        if initialized:
            gdk.gdk_release()


if __name__ == '__main__':
    raise SystemExit(main())
