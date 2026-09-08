#!/usr/bin/env python3
"""Supervised 1 mm capability probe for G2 right mode-1 pose control.

It performs one tiny translation and a return using the same 50-Hz GDK path
as the AprilTag final approach.  It has no perception, no gripper operation,
and sends no command unless ``--allow-motion`` is supplied.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

try:
    from .control import CartesianPoseTarget, G2Controller
except ImportError:
    from control import CartesianPoseTarget, G2Controller


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2 右 arm_r_end_link mode-1 受看护 1 mm 往返探针")
    parser.add_argument("--axis", choices=("x", "y", "z"), default="x")
    parser.add_argument("--distance-mm", type=float, default=1.0)
    parser.add_argument("--speed-mm-s", type=float, default=2.0)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=0.20, help="实际到位位置容差，0.10..0.50 mm")
    parser.add_argument("--orientation-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--allow-motion", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_motion:
        print("只读退出：该探针需要 --allow-motion，未发送任何 GDK 指令。")
        return 2
    if (
        not 0.5 <= args.distance_mm <= 1.0 or not 1.0 <= args.speed_mm_s <= 5.0
        or not 1.0 <= args.timeout_s <= 10.0 or not 0.10 <= args.position_tolerance_mm <= 0.50
        or not 0.1 <= args.orientation_tolerance_deg <= 1.0
    ):
        print("仅允许 0.5–1.0 mm、1–5 mm/s、1–10 s、0.10–0.50 mm 实际到位容差。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(0.5)
        controller = G2Controller(agibot_gdk, robot, allow_motion=True)
        controller.checked_arm_state()
        start = controller.read_end_effector_pose("arm_r_end_link")
        offset = {"x": (args.distance_mm / 1000.0, 0.0, 0.0), "y": (0.0, args.distance_mm / 1000.0, 0.0), "z": (0.0, 0.0, args.distance_mm / 1000.0)}[args.axis]
        target = CartesianPoseTarget(tuple(a + b for a, b in zip(start.position_m, offset)), start.orientation_xyzw)
        stream = controller.start_right_end_effector_position_stream(start, max_translation_velocity_mps=args.speed_mm_s / 1000.0)
        try:
            stream.update_target(target)
            moved = stream.wait_until_actual_target(
                target, args.timeout_s, position_tolerance_m=args.position_tolerance_mm / 1000.0,
                orientation_tolerance_deg=args.orientation_tolerance_deg, stable_samples=10,
            )
            stream.update_target(start)
            returned = stream.wait_until_actual_target(
                start, args.timeout_s, position_tolerance_m=args.position_tolerance_mm / 1000.0,
                orientation_tolerance_deg=args.orientation_tolerance_deg, stable_samples=10,
            )
        finally:
            stream.stop()
        if moved is None or returned is None:
            raise RuntimeError("实际末端未在容差内连续稳定 10 次；探针未通过，未设置验证标记")
        print(json.dumps({
            "event": "right_mode1_probe", "axis": args.axis, "requested_displacement_m": offset,
            "start_m": start.position_m, "after_outbound_m": moved.position_m, "after_return_m": returned.position_m,
            "outbound_actual_m": [b - a for a, b in zip(start.position_m, moved.position_m)],
            "return_residual_m": [b - a for a, b in zip(start.position_m, returned.position_m)],
            "actual_position_tolerance_m": args.position_tolerance_mm / 1000.0,
            "actual_orientation_tolerance_deg": args.orientation_tolerance_deg,
            "stable_samples_required": 10,
            "result": "operator_must_review_before_setting_right_mode1_probe_verified_true",
        }, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"右手 mode-1 探针失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if 'initialized' in locals() and initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
