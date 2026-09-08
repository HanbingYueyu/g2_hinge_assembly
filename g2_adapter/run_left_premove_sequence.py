#!/usr/bin/env python3
"""Run the recorded left P1 -> P2 path with arrival checks.

P2 is the final detection pose.  The script does not start SAM3, force mode,
or grasping after arrival.  Omitting ``--allow-motion`` always performs only
the complete read-only preflight.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from .control import G2Controller
    from .record_pose import DEFAULT_OUTPUT
    from .safety import ARM_POSITION_LIMITS, SafetyError
except ImportError:
    from control import G2Controller
    from record_pose import DEFAULT_OUTPUT
    from safety import ARM_POSITION_LIMITS, SafetyError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESETS = ("left_p1", "left_p2")
HANDOFF_PRESETS = ("left_handoff_1", "left_handoff_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 P1→P2 顺序到达左手最终检测位")
    parser.add_argument("--poses", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--handoff", action="store_true", help="执行左手两个交接点，供切换右手前使用")
    parser.add_argument("--velocity-rad-s", type=float, default=0.08)
    parser.add_argument("--arrival-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--timeout-per-preset-s", type=float, default=25.0)
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/left_premove_*.jsonl")
    parser.add_argument("--allow-motion", action="store_true", help="唯一运动确认参数；不指定时严格只读。")
    return parser.parse_args()


def load_presets(path: Path, names=PRESETS) -> list[tuple[str, list[float]]]:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    poses = document.get("poses")
    if not isinstance(poses, dict):
        raise RuntimeError(f"姿态文件无效：{path}")
    result = []
    for name in names:
        item = poses.get(name)
        if not isinstance(item, dict) or item.get("side") != "left":
            raise RuntimeError(f"姿态文件缺少左手 {name}")
        values = [float(value) for value in item.get("joint_positions_rad", ())]
        if len(values) != 7:
            raise RuntimeError(f"{name} 必须含 7 个左臂关节值")
        for value, (lower, upper) in zip(values, ARM_POSITION_LIMITS[:7]):
            if not lower <= value <= upper:
                raise SafetyError(f"{name} 关节值无效或超限：{value}")
        result.append((name, values))
    return result


def write_log(stream, event: dict) -> None:
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()


def open_handoff_gripper(controller, *, clock=time.monotonic, sleep=time.sleep):
    """Confirm the left gripper fully open before issuing handoff presets."""
    target = -0.785
    stream = controller.start_left_gripper_position_stream(target)
    deadline = clock() + 6.0
    actual = None
    try:
        while True:
            stream.raise_if_failed()
            actual = controller.read_gripper_position("left")
            if actual is not None and abs(actual - target) <= 0.035:
                return {"event": "left_handoff_gripper_open_confirmed",
                        "target_rad": target, "actual_position_rad": actual}
            if clock() >= deadline:
                raise SafetyError(f"左手交接前夹爪张开反馈超时：actual={actual}；不执行交接点、不启动右手")
            sleep(0.05)
    finally:
        stream.stop()


def main() -> int:
    args = parse_args()
    if not (0 < args.velocity_rad_s <= 0.15 and 0.001 <= args.arrival_tolerance_rad <= 0.10 and 1.0 <= args.timeout_per_preset_s <= 30.0):
        print("参数错误：速度、到位容差或超时超出安全范围。", file=sys.stderr)
        return 2
    try:
        presets = load_presets(args.poses, HANDOFF_PRESETS if args.handoff else PRESETS)
    except Exception as exc:
        print(f"读取预移动姿态失败：{exc}", file=sys.stderr)
        return 2
    log_path = args.log or PROJECT_ROOT / "logs" / f"left_premove_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
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
        time.sleep(1.0)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        current = controller.checked_arm_state()
        preview = {
            "mode": "motion" if args.allow_motion else "read_only_preview",
            "presets": [{"name": name, "target_rad": target} for name, target in presets],
            "current_left_arm_rad": list(current[:7]),
            "arrival_tolerance_rad": args.arrival_tolerance_rad,
            "log": str(log_path),
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        if not args.allow_motion:
            print("只读预检完成：未发送运动或夹爪命令。")
            return 0
        with log_path.open("a", encoding="utf-8") as log:
            write_log(log, {"event": "start", "utc": datetime.now(timezone.utc).isoformat(), "presets": [name for name, _ in presets]})
            if args.handoff:
                opened = open_handoff_gripper(controller)
                write_log(log, opened)
                print(json.dumps(opened, ensure_ascii=False), flush=True)
            for name, target in presets:
                controller.move_preset("left", target, velocity_rad_s=args.velocity_rad_s, life_time_s=args.timeout_per_preset_s, detail=f"traditional ordered {name}")
                write_log(log, {"event": "request_sent", "preset": name, "utc": datetime.now(timezone.utc).isoformat(), "target_rad": target})
                deadline = time.monotonic() + args.timeout_per_preset_s
                while time.monotonic() < deadline:
                    current = controller.checked_arm_state()[:7]
                    max_error = max(abs(measured - expected) for measured, expected in zip(current, target))
                    if max_error <= args.arrival_tolerance_rad:
                        write_log(log, {"event": "arrived", "preset": name, "utc": datetime.now(timezone.utc).isoformat(), "max_joint_error_rad": max_error})
                        print(f"{name} 已到位，最大关节误差 {max_error:.4f} rad。")
                        break
                    time.sleep(0.10)
                else:
                    raise RuntimeError(f"{name} 在 {args.timeout_per_preset_s:.1f} s 内未到位；停止后续点位请求")
        print("→".join(name for name, _ in presets) + " 已全部到位；夹爪状态保持原样。")
        return 0
    except Exception as exc:
        print(f"左手预移动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
