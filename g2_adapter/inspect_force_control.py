#!/usr/bin/env python3
"""Read GDK force-control capability and idle wrench baseline only.

This program never calls a GDK method beginning with ``set_``, never changes
control mode, and never creates a motion/gripper request.  It is the required
baseline before a force-contact grasp is enabled.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读读取 G2 力控状态、阻抗和末端六维力基线")
    parser.add_argument("--samples", type=int, default=100, help="静止采样数，默认 100。")
    parser.add_argument("--period-s", type=float, default=0.02, help="采样周期，默认 20 ms。")
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/force_probe_*.json")
    return parser.parse_args()


def vector3(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def wrench_dict(wrench, frame_name: str) -> dict:
    return {
        "frame_name": str(frame_name),
        "force_n": vector3(wrench.force),
        "torque_nm": vector3(wrench.torque),
    }


def finite_matrix(samples: list[list[float]]) -> dict | None:
    matrix = np.asarray(samples, dtype=np.float64)
    if matrix.size == 0 or matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        return None
    return {
        "mean": [float(value) for value in matrix.mean(axis=0)],
        "std": [float(value) for value in matrix.std(axis=0)],
        "min": [float(value) for value in matrix.min(axis=0)],
        "max": [float(value) for value in matrix.max(axis=0)],
    }


def main() -> int:
    args = parse_args()
    if args.samples < 10 or args.samples > 1000 or not math.isfinite(args.period_s) or not 0.005 <= args.period_s <= 1.0:
        print("参数错误：--samples 必须在 10..1000，--period-s 必须在 0.005..1.0。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_python.sh 运行。", file=sys.stderr)
        return 2
    log_path = args.log or PROJECT_ROOT / "logs" / f"force_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    initialized = False
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        initial = robot.get_motion_control_status()
        frames = list(robot.get_end_pose_control_frame())
        cartesian_impedance = [
            {"kp": float(item.kp), "ki": float(item.ki), "kv": float(item.kv)}
            for item in robot.get_cartesian_impedance()
        ]
        joint_impedance = [
            {"kp": float(item.kp), "ki": float(item.ki), "kv": float(item.kv)}
            for item in robot.get_joint_impedance()
        ]
        wrench_samples: list[list[dict]] = []
        disturbance_samples: list[list[float]] = []
        statuses: list[dict] = []
        deadline = time.monotonic()
        for _ in range(args.samples):
            status = robot.get_motion_control_status()
            frame_names = [str(value) for value in status.frame_names]
            wrenches = [
                wrench_dict(item, frame_names[index] if index < len(frame_names) else f"frame_{index}")
                for index, item in enumerate(status.wrenches)
            ]
            wrench_samples.append(wrenches)
            disturbance = [float(value) for value in status.arm_disturbance_force]
            disturbance_samples.append(disturbance)
            statuses.append({
                "mode": int(status.mode),
                "control_mode": int(status.control_mode),
                "error_code": int(status.error_code),
                "error_msg": str(status.error_msg),
                "frame_names": [str(value) for value in status.frame_names],
            })
            deadline += args.period_s
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
        wrench_count = max((len(sample) for sample in wrench_samples), default=0)
        wrench_summary = []
        for index in range(wrench_count):
            force = [sample[index]["force_n"] for sample in wrench_samples if len(sample) > index]
            torque = [sample[index]["torque_nm"] for sample in wrench_samples if len(sample) > index]
            frame_name = next(
                (sample[index]["frame_name"] for sample in wrench_samples if len(sample) > index),
                f"frame_{index}",
            )
            wrench_summary.append({
                "index": index,
                "frame_name": frame_name,
                "force_n": finite_matrix(force),
                "torque_nm": finite_matrix(torque),
            })
        disturbance_width = max((len(sample) for sample in disturbance_samples), default=0)
        disturbance_summary = [
            finite_matrix([[sample[index]] for sample in disturbance_samples if len(sample) > index])
            for index in range(disturbance_width)
        ]
        document = {
            "schema_version": 1,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "samples": args.samples,
            "period_s": args.period_s,
            "read_only": True,
            "motion_status_initial": {
                "mode": int(initial.mode), "control_mode": int(initial.control_mode),
                "error_code": int(initial.error_code), "error_msg": str(initial.error_msg),
                "frame_names": [str(value) for value in initial.frame_names],
            },
            "end_pose_control_frames": frames,
            "cartesian_impedance": cartesian_impedance,
            "joint_impedance": joint_impedance,
            "wrench_baseline": wrench_summary,
            "arm_disturbance_force_baseline": disturbance_summary,
            "sample_statuses": statuses,
            "safety": "read-only: no mode switch, impedance/compliance write, robot motion, gripper command, or torque-sensor calibration",
        }
        log_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "log": str(log_path),
            "end_pose_control_frames": frames,
            "motion_mode": document["motion_status_initial"]["mode"],
            "control_mode": document["motion_status_initial"]["control_mode"],
            "wrench_baseline": wrench_summary,
            "arm_disturbance_force_baseline": disturbance_summary,
            "safety": document["safety"],
        }, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"力控只读探针失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
