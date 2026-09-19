#!/usr/bin/env python3
"""Standalone right-arm joint-impedance kp probe (control_mode 3).

Sets the right arm's seven joint kp, drives one small end-effector move through
the same 50 Hz position stream the right grasp uses, measures the steady-state
commanded-vs-actual following error, then returns to the start pose and restores
the standard stiffness.

This does not touch the bilateral flow, the grasp stages, or any other entry
point: it only writes the right arm's kp while it runs.
"""

import argparse
import math
import sys
import time

import numpy as np

try:
    from .control import CartesianPoseTarget, G2Controller, STANDARD_JOINT_IMPEDANCE_KP
except ImportError:
    from control import CartesianPoseTarget, G2Controller, STANDARD_JOINT_IMPEDANCE_KP


RIGHT_END_LINK = "arm_r_end_link"
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
# Same empirically safe kp envelope as demo_joint_impedance_move.py.
LOW_KP = (50.0, 50.0, 50.0, 60.0, 30.0, 30.0, 30.0)
HIGH_KP = (500.0, 500.0, 500.0, 600.0, 300.0, 300.0, 300.0)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="右臂关节阻抗 kp 探针（模式 3）：小位移 + 稳态跟随误差测量；不修改主流程",
    )
    parser.add_argument(
        "--kp", nargs=7, type=float, default=list(STANDARD_JOINT_IMPEDANCE_KP),
        help="右臂关节 1–7 刚度 Nm/rad；默认标准 400 400 400 500 300 200 200",
    )
    parser.add_argument("--axis", choices=("x", "y", "z"), default="z", help="探针方向（base_link），默认 z")
    parser.add_argument("--delta-mm", type=float, default=20.0, help="探针位移，默认 20 mm")
    parser.add_argument("--speed-mm-s", type=float, default=5.0, help="位姿流速度，默认 5 mm/s")
    parser.add_argument("--settle-s", type=float, default=2.0, help="稳态误差采样时长，默认 2 s")
    parser.add_argument("--allow-motion", action="store_true", help="允许切模式、写 kp 并移动；否则只读预览")
    return parser.parse_args(argv)


def validate_args(args):
    if len(args.kp) != 7 or any((not math.isfinite(value)) or value <= 0.0 for value in args.kp):
        raise ValueError("--kp 需要七个正的有限值")
    if any(
        not (low <= value <= high)
        for value, low, high in zip(args.kp, LOW_KP, HIGH_KP)
    ):
        raise ValueError(
            "kp 超出经验安全范围：关节 1–3 为 50–500，关节 4 为 60–600，关节 5–7 为 30–300 Nm/rad"
        )
    if not math.isfinite(args.delta_mm) or not 0.0 < abs(args.delta_mm) <= 50.0:
        raise ValueError("--delta-mm 需要 (0, 50] mm")
    if not math.isfinite(args.speed_mm_s) or not 0.0 < args.speed_mm_s <= 30.0:
        raise ValueError("--speed-mm-s 需要 (0, 30] mm/s")
    if not math.isfinite(args.settle_s) or not 0.0 <= args.settle_s <= 10.0:
        raise ValueError("--settle-s 需要 [0, 10] s")


def follow_errors_mm(controller, target, *, samples, period_s):
    errors = []
    for _ in range(samples):
        actual = controller.read_end_effector_pose(RIGHT_END_LINK)
        errors.append(
            float(np.linalg.norm(np.asarray(actual.position_m) - np.asarray(target.position_m))) * 1000.0
        )
        time.sleep(period_s)
    return errors


def _wait(stream, target, timeout_s):
    return stream.wait_until_actual_target(
        target, timeout_s,
        position_tolerance_m=0.005,
        orientation_tolerance_deg=1.0,
        stable_samples=3,
    )


def main(argv=None):
    args = parse_args(argv)
    gdk = None
    initialized = False
    controller = None
    try:
        validate_args(args)
        import agibot_gdk as gdk
        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(gdk, robot, allow_motion=args.allow_motion)

        current = controller.read_end_effector_pose(RIGHT_END_LINK)
        delta = [0.0, 0.0, 0.0]
        delta[AXIS_INDEX[args.axis]] = abs(args.delta_mm) / 1000.0
        target = CartesianPoseTarget(
            tuple(float(current.position_m[index] + delta[index]) for index in range(3)),
            current.orientation_xyzw,
        )
        print(f"右末端当前 base_link: {[round(float(v), 6) for v in current.position_m]}", flush=True)
        print(f"探针目标 {args.axis} +{abs(args.delta_mm):.1f} mm: {[round(float(v), 6) for v in target.position_m]}", flush=True)
        print(f"kp = {[float(v) for v in args.kp]}", flush=True)
        if not args.allow_motion:
            print("只读预览；添加 --allow-motion 才会切模式、写 kp 并移动。")
            return 0

        # The stream factory prepares the standard stiffness for this arm; the
        # explicit write right after it replaces that with the probe kp.
        stream = controller.start_right_end_effector_position_stream(
            current, rate_hz=50.0, life_time_s=0.10,
            max_translation_velocity_mps=args.speed_mm_s / 1000.0,
        )
        try:
            controller.set_joint_impedance("right", args.kp)
            print("已写入右臂探针 kp（模式 3）。", flush=True)
            stream.update_target(target)
            travel_m = abs(args.delta_mm) / 1000.0
            timeout_s = max(5.0, travel_m / (args.speed_mm_s / 1000.0) + 3.0)
            arrived = _wait(stream, target, timeout_s)
            print(f"到位判定: {'到达' if arrived else f'超时(>{timeout_s:.1f}s)'}", flush=True)

            samples = max(1, int(round(args.settle_s / 0.1)))
            errors = sorted(follow_errors_mm(controller, target, samples=samples, period_s=0.1))
            print(
                "稳态跟随误差(mm): "
                f"min={errors[0]:.3f} median={errors[len(errors) // 2]:.3f} max={errors[-1]:.3f}",
                flush=True,
            )

            stream.update_target(current)
            _wait(stream, current, timeout_s)
        finally:
            stream.stop()
        return 0
    except KeyboardInterrupt:
        print("已由操作者中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"右臂 kp 探针失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if controller is not None and args.allow_motion:
            try:
                actual = controller.checked_arm_state()[7:14]
                controller.hold_joint_positions("right", actual)
                controller.prepare_joint_impedance_control(("right",))
                print("已恢复右臂标准 kp。", flush=True)
            except Exception as exc:
                print(f"恢复标准 kp 失败：{exc}", file=sys.stderr)
        if initialized:
            gdk.gdk_release()


if __name__ == "__main__":
    raise SystemExit(main())
