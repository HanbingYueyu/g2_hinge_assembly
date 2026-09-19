#!/usr/bin/env python3
"""Lift a held object, then move the left arm to its recorded P3 then P4 poses.

P4 is a deliberately operator-recorded workcell pose.  This program never
opens, closes, or otherwise commands the Omnipicker.  It first makes one
local, fixed-orientation 150 mm upward end-link motion, then sends only the
seven left-arm joints through GDK's planned joint request and verifies each
arrival. Omitting ``--allow-motion`` is a complete read-only preflight.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .control import CartesianPoseTarget, G2Controller
    from .record_pose import DEFAULT_OUTPUT
except ImportError:
    from control import CartesianPoseTarget, G2Controller
    from record_pose import DEFAULT_OUTPUT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESETS = ("left_p3", "left_p4")
LEFT_END_LINK = "arm_l_end_link"
PRE_P4_LIFT_M = 0.15
PRE_P4_LIFT_SPEED_MPS = 0.01
PRE_P4_LIFT_TOLERANCE_M = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="上抬后移动已夹住物体的左臂至 P3→P4；不操作夹爪")
    parser.add_argument("--poses", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--velocity-rad-s", type=float, default=0.08)
    parser.add_argument("--arrival-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--timeout-per-preset-s", type=float, default=25.0)
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/left_post_grasp_*.jsonl")
    parser.add_argument("--allow-motion", action="store_true", help="唯一运动确认参数；不指定时严格只读。")
    return parser.parse_args()


def load_presets(path: Path) -> list[tuple[str, list[float]]]:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    poses = document.get("poses")
    if not isinstance(poses, dict):
        raise RuntimeError(f"姿态文件无效：{path}")
    result: list[tuple[str, list[float]]] = []
    for name in PRESETS:
        item = poses.get(name)
        if not isinstance(item, dict) or item.get("side") != "left":
            raise RuntimeError(f"姿态文件缺少左手 {name}")
        values = [float(value) for value in item.get("joint_positions_rad", ())]
        if len(values) != 7:
            raise RuntimeError(f"{name} 必须含 7 个左臂关节值")
        result.append((name, values))
    return result


def write_log(stream, event: dict) -> None:
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()


def lift_before_p4(controller: G2Controller) -> dict:
    """Raise the measured left flange 150 mm in base +Z before P4.

    This uses GDK's mode-3 end-effector position stream at the same 50 Hz as
    the grasp servo. It preserves the current flange orientation and never
    includes an end-effector-joint request, so the held gripper state is left
    untouched. P4 is not requested unless the measured lift is within 10 mm
    of the requested 150 mm and has negligible lateral drift.
    """
    before = controller.read_end_effector_pose(LEFT_END_LINK)
    target = CartesianPoseTarget(
        (before.position_m[0], before.position_m[1], before.position_m[2] + PRE_P4_LIFT_M),
        before.orientation_xyzw,
    )
    stream = controller.start_left_end_effector_position_stream(
        before,
        rate_hz=50.0,
        life_time_s=0.10,
        max_translation_velocity_mps=PRE_P4_LIFT_SPEED_MPS,
    )
    try:
        stream.update_target(target)
        timeout_s = PRE_P4_LIFT_M / PRE_P4_LIFT_SPEED_MPS + 0.50
        if not stream.wait_until_target(timeout_s):
            raise RuntimeError("P4 前 150 mm 上抬的 GDK 位姿流未在限定时间内发送到目标；拒绝请求 P4")
        time.sleep(0.10)
        after = controller.read_end_effector_pose(LEFT_END_LINK)
        delta = tuple(float(actual - origin) for actual, origin in zip(after.position_m, before.position_m))
        lateral_m = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
        if abs(delta[2] - PRE_P4_LIFT_M) > PRE_P4_LIFT_TOLERANCE_M or lateral_m > PRE_P4_LIFT_TOLERANCE_M:
            raise RuntimeError(
                "P4 前上抬实测位移不满足要求；"
                f"actual_delta_m={list(delta)}, expected_z_m={PRE_P4_LIFT_M:.3f}, "
                f"tolerance_m={PRE_P4_LIFT_TOLERANCE_M:.3f}；拒绝请求 P4"
            )
        return {
            "event": "pre_p4_lift_arrived",
            "before_end_link_base_link": list(before.position_m),
            "target_end_link_base_link": list(target.position_m),
            "actual_end_link_base_link": list(after.position_m),
            "actual_delta_base_m": list(delta),
            "lift_m": PRE_P4_LIFT_M,
            "speed_mps": PRE_P4_LIFT_SPEED_MPS,
            "tolerance_m": PRE_P4_LIFT_TOLERANCE_M,
            "gripper": "preserved; no gripper request is issued",
        }
    finally:
        stream.stop()


def main() -> int:
    args = parse_args()
    if not (
        math.isfinite(args.velocity_rad_s)
        and args.velocity_rad_s > 0.0
        and 0.001 <= args.arrival_tolerance_rad <= 0.10
        and 1.0 <= args.timeout_per_preset_s <= 30.0
    ):
        print("参数错误：速度、到位容差或超时超出安全范围。", file=sys.stderr)
        return 2
    try:
        presets = load_presets(args.poses)
    except Exception as exc:
        print(f"读取 P3/P4 姿态失败：{exc}", file=sys.stderr)
        return 2
    log_path = args.log or PROJECT_ROOT / "logs" / f"left_post_grasp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2
    initialized = False
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        current = controller.checked_arm_state()
        preview = {
            "mode": "motion" if args.allow_motion else "read_only_preview",
            "presets": [{"name": name, "target_rad": target} for name, target in presets],
            "current_left_arm_rad": list(current[:7]),
            "arrival_tolerance_rad": args.arrival_tolerance_rad,
            "pre_p4_lift": {
                "frame": "base_link",
                "delta_m": [0.0, 0.0, PRE_P4_LIFT_M],
                "speed_mps": PRE_P4_LIFT_SPEED_MPS,
                "measured_tolerance_m": PRE_P4_LIFT_TOLERANCE_M,
            },
            "gripper": "preserved: this program does not send any gripper request",
            "log": str(log_path),
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        if not args.allow_motion:
            print("只读预检完成：未发送运动或夹爪命令。")
            return 0
        with log_path.open("a", encoding="utf-8") as log:
            write_log(log, {
                "event": "start",
                "utc": datetime.now(timezone.utc).isoformat(),
                "presets": [name for name, _ in presets],
                "pre_p4_lift_m": PRE_P4_LIFT_M,
                "gripper": "preserved; no gripper request is issued by this program",
            })
            lift_event = lift_before_p4(controller)
            lift_event["utc"] = datetime.now(timezone.utc).isoformat()
            write_log(log, lift_event)
            print("P4 前已实测上抬 150 mm；开始依次请求 P3、P4，夹爪未操作。")
            for name, target in presets:
                controller.move_preset("left", target, velocity_rad_s=args.velocity_rad_s, life_time_s=args.timeout_per_preset_s, detail=f"traditional held-object {name}")
                write_log(log, {"event": "request_sent", "preset": name, "utc": datetime.now(timezone.utc).isoformat(), "target_rad": target})
                deadline = time.monotonic() + args.timeout_per_preset_s
                while time.monotonic() < deadline:
                    measured = controller.checked_arm_state()[:7]
                    max_error = max(abs(actual - expected) for actual, expected in zip(measured, target))
                    if max_error <= args.arrival_tolerance_rad:
                        write_log(log, {"event": "arrived", "preset": name, "utc": datetime.now(timezone.utc).isoformat(), "max_joint_error_rad": max_error})
                        print(f"{name} 已到位，最大关节误差 {max_error:.4f} rad；夹爪未操作。")
                        break
                    time.sleep(0.10)
                else:
                    raise RuntimeError(f"{name} 在 {args.timeout_per_preset_s:.1f} s 内未到位；停止后续命令，夹爪未操作")
        print("P4 前已上抬 150 mm，P3、P4 已依次到位；夹爪状态保持原样，未启动检测或力控。")
        return 0
    except Exception as exc:
        print(f"左手 P4 运输失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
