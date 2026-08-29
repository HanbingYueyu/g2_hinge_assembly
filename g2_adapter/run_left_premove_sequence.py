#!/usr/bin/env python3
"""Run the recorded left P2 -> P3 path with arrival checks.

P3 is the final detection pose.  The script does not start SAM3, force mode,
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
except ImportError:
    from control import G2Controller
    from record_pose import DEFAULT_OUTPUT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRESETS = ("left_p2", "left_p3")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按 P2→P3 顺序到达左手最终检测位")
    parser.add_argument("--poses", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--velocity-rad-s", type=float, default=0.08)
    parser.add_argument("--arrival-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--timeout-per-preset-s", type=float, default=25.0)
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/left_premove_*.jsonl")
    parser.add_argument("--allow-motion", action="store_true", help="唯一运动确认参数；不指定时严格只读。")
    return parser.parse_args()


def load_presets(path: Path) -> list[tuple[str, list[float]]]:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    poses = document.get("poses")
    if not isinstance(poses, dict):
        raise RuntimeError(f"姿态文件无效：{path}")
    result = []
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


def main() -> int:
    args = parse_args()
    if not (0 < args.velocity_rad_s <= 0.15 and 0.001 <= args.arrival_tolerance_rad <= 0.10 and 1.0 <= args.timeout_per_preset_s <= 30.0):
        print("参数错误：速度、到位容差或超时超出安全范围。", file=sys.stderr)
        return 2
    try:
        presets = load_presets(args.poses)
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
                    raise RuntimeError(f"{name} 在 {args.timeout_per_preset_s:.1f} s 内未到位；停止后续 P3 请求")
        print("P2→P3 已全部到位：当前为最终检测位；未启动检测、力控、抓取或夹爪命令。")
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
