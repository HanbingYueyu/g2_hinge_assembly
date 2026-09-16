#!/usr/bin/env python3
"""Keep current XYZ and apply the selected arm's full-flow orientation convention."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time

try:
    from .arm_orientation import fixed_forward_down_orientation
    from .control import CartesianPoseTarget, G2Controller, quaternion_angle_deg
except ImportError:
    from arm_orientation import fixed_forward_down_orientation
    from control import CartesianPoseTarget, G2Controller, quaternion_angle_deg


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="保持末端 XYZ 不变，按全流程同一约定调整 G2 手臂姿态")
    parser.add_argument("--side", choices=("left", "right"), default="right")
    parser.add_argument("--yaw-deg", type=float, required=True)
    parser.add_argument("--pitch-deg", type=float, required=True)
    parser.add_argument("--roll-deg", type=float, required=True)
    parser.add_argument("--angular-speed-deg-s", type=float, default=10.0)
    parser.add_argument("--allow-motion", action="store_true", help="唯一运动确认参数；未指定时只读预览。")
    return parser.parse_args(argv)


def orientation_target(
    current, *, side: str, yaw_deg: float, pitch_deg: float, roll_deg: float,
) -> CartesianPoseTarget:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    orientation = fixed_forward_down_orientation(
        forward_yaw_deg=yaw_deg,
        gripper_pitch_deg=pitch_deg,
        gripper_roll_deg=roll_deg,
    )
    return CartesianPoseTarget(
        tuple(float(value) for value in current.position_m),
        orientation,
    )


def rotate_arm_orientation(
    controller,
    *,
    side: str,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    angular_speed_deg_s: float,
) -> dict:
    end_link = "arm_l_end_link" if side == "left" else "arm_r_end_link"
    current = controller.read_end_effector_pose(end_link)
    target = orientation_target(
        current, side=side, yaw_deg=yaw_deg, pitch_deg=pitch_deg, roll_deg=roll_deg,
    )
    angular_distance_deg = quaternion_angle_deg(current.orientation_xyzw, target.orientation_xyzw)
    factory = getattr(controller, f"start_{side}_end_effector_position_stream")
    stream = factory(
        current, rate_hz=50.0, life_time_s=0.10,
        max_translation_velocity_mps=0.005,
        allow_orientation_change=True,
        max_angular_velocity_rad_s=math.radians(angular_speed_deg_s),
    )
    try:
        stream.update_target(target)
        actual = stream.wait_until_actual_target(
            target, angular_distance_deg / angular_speed_deg_s + 3.0,
            position_tolerance_m=0.005,
            orientation_tolerance_deg=0.5,
            stable_samples=3,
        )
        if actual is None:
            raise RuntimeError("末端姿态未在限定时间内实际到位")
    finally:
        stream.stop()
    return {
        "side": side,
        "position_m": list(current.position_m),
        "yaw_pitch_roll_deg": [float(yaw_deg), float(pitch_deg), float(roll_deg)],
        "target_orientation_xyzw": list(target.orientation_xyzw),
        "angular_distance_deg": angular_distance_deg,
        "angular_speed_deg_s": float(angular_speed_deg_s),
    }


def main(argv=None) -> int:
    args = parse_args(argv)
    angles = (args.yaw_deg, args.pitch_deg, args.roll_deg)
    if not all(math.isfinite(value) and -180.0 <= value <= 180.0 for value in angles):
        print("参数错误：yaw、pitch、roll 必须是 [-180, 180] 内的有限角度。", file=sys.stderr)
        return 2
    if not math.isfinite(args.angular_speed_deg_s) or not 0.6 <= args.angular_speed_deg_s <= 28.6:
        print("参数错误：--angular-speed-deg-s 必须在 [0.6, 28.6]。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_python.sh 运行。", file=sys.stderr)
        return 2

    initialized = False
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(0.5)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        controller.checked_arm_state()
        end_link = "arm_l_end_link" if args.side == "left" else "arm_r_end_link"
        current = controller.read_end_effector_pose(end_link)
        target = orientation_target(
            current, side=args.side,
            yaw_deg=args.yaw_deg, pitch_deg=args.pitch_deg, roll_deg=args.roll_deg,
        )
        if not args.allow_motion:
            print(json.dumps({
                "side": args.side,
                "position_m_preserved": list(current.position_m),
                "target_orientation_xyzw": list(target.orientation_xyzw),
                "command_sent": False,
                "note": "追加 --allow-motion 才会转动手臂。",
            }, ensure_ascii=False))
            return 0
        result = rotate_arm_orientation(
            controller, side=args.side,
            yaw_deg=args.yaw_deg, pitch_deg=args.pitch_deg, roll_deg=args.roll_deg,
            angular_speed_deg_s=args.angular_speed_deg_s,
        )
        print(json.dumps({"event": "arm_orientation_arrived", **result}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"设置手臂姿态失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
