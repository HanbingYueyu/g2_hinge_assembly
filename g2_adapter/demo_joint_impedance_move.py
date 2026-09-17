#!/usr/bin/env python3
"""Standalone selected-arm joint impedance motion, not constant-force control."""

import argparse
import math
import sys
import time

try:
    from .control import G2Controller
    from .safety import validate_arm_target
except ImportError:
    from control import G2Controller
    from safety import validate_arm_target


LOW_KP = (50, 50, 50, 60, 30, 30, 30)
HIGH_KP = (500, 500, 500, 600, 300, 300, 300)
DEFAULT_KP = (400, 400, 400, 500, 300, 200, 200)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='单臂关节阻抗移动（模式 3）；不是恒力控制')
    parser.add_argument('--side', choices=('left', 'right'), default='left')
    parser.add_argument('--delta-deg', nargs=7, type=float, default=(0,0,0,0,0,0,5), help='关节 1–7 相对当前角度的偏移（度）')
    parser.add_argument('--velocity-rad-s', type=float, default=.05)
    parser.add_argument('--kp', nargs=7, type=float, default=DEFAULT_KP, help='关节 1–7 刚度，单位 Nm/rad；默认采用实测右臂配置')
    parser.add_argument('--allow-motion', action='store_true', help='允许切换模式并移动；否则只读预览')
    return parser.parse_args(argv)


def validate_args(args):
    if len(args.delta_deg) != 7 or not all(math.isfinite(v) for v in args.delta_deg):
        raise ValueError('角度偏移必须是七个有限值')
    if not math.isfinite(args.velocity_rad_s) or not 0 < args.velocity_rad_s <= .5:
        raise ValueError('速度需在 (0, 0.5] rad/s 内')
    if len(args.kp) != 7 or any(not math.isfinite(v) or not lo <= v <= hi for v, lo, hi in zip(args.kp, LOW_KP, HIGH_KP)):
        raise ValueError('七关节刚度范围：1–3 为 50–500，4 为 60–600，5–7 为 30–300 Nm/rad')


def target_for_args(controller, args):
    current = list(controller.checked_arm_state())
    offset = 0 if args.side == 'left' else 7
    target = tuple(q + math.radians(d) for q,d in zip(current[offset:offset+7], args.delta_deg))
    current[offset:offset+7] = target
    validate_arm_target(current)
    return target


def move_joint_impedance(controller, args, *, clock=time.monotonic, sleep=time.sleep):
    validate_args(args)
    target = target_for_args(controller, args)
    offset = 0 if args.side == 'left' else 7
    group = 0 if args.side == 'left' else 1
    entered = False
    try:
        controller.set_control_mode(3)
        entered = True
        controller.robot.set_joint_impedance([controller.gdk.Impedance(kp=v) for v in args.kp], group)
        controller.move_preset(args.side, target, velocity_rad_s=args.velocity_rad_s,
                               life_time_s=30., detail='joint impedance demo')
        deadline = clock() + 30.
        stable = 0
        while True:
            actual = controller.checked_arm_state()[offset:offset+7]
            stable = stable + 1 if max(abs(a-b) for a,b in zip(actual, target)) <= math.radians(.5) else 0
            if stable >= 3:
                return actual
            if clock() >= deadline:
                raise RuntimeError('30 秒内未到达目标（阻抗模式受外力时可能存在角度误差）')
            sleep(.05)
    finally:
        if entered:
            actual = controller.checked_arm_state()[offset:offset+7]
            # Replace the old destination before returning to position mode.
            # A failed hold deliberately prevents the mode switch.
            controller.robot.move_arm_joint_servo(list(actual), .01, group, False)
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
        target = target_for_args(controller, args)
        print(f'{args.side} 臂目标角度(度)：{[round(math.degrees(v), 2) for v in target]}', flush=True)
        if not args.allow_motion:
            print('只读预览；添加 --allow-motion 才切换关节阻抗并移动。')
            return 0
        print('开始关节阻抗移动；请确认末端负载正确，且没有其他控制脚本同时运行。', flush=True)
        actual = move_joint_impedance(controller, args)
        print(f'移动完成，已回到位控；实测角度(度)：{[round(math.degrees(v), 2) for v in actual]}', flush=True)
        return 0
    except KeyboardInterrupt:
        print('已由操作者中断。', file=sys.stderr)
        return 130
    except Exception as exc:
        print(f'关节阻抗 demo 失败：{exc}', file=sys.stderr)
        return 1
    finally:
        if initialized:
            gdk.gdk_release()


if __name__ == '__main__':
    raise SystemExit(main())
