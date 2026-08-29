#!/usr/bin/env python3
"""Supervised 1 mm left end-link position-control probe for GDK mode 1.

The probe neither changes GDK/application modes nor controls the gripper.  It
reads the start pose from ``get_motion_control_status().frame_poses``, linearly
interpolates exactly one local base_link translation in the caller thread at
50 Hz, then records the same status pose after the command lifetime expires.
Without ``--allow-motion`` it only writes the before-state log.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from .control import CartesianPoseTarget, G2Controller
except ImportError:
    from control import CartesianPoseTarget, G2Controller


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEFT_END_LINK = "arm_l_end_link"


@dataclass(frozen=True)
class StatusPose:
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


def pose_from_status(status, frame_name: str) -> StatusPose:
    names = [str(value) for value in status.frame_names]
    try:
        pose = status.frame_poses[names.index(frame_name)]
    except (AttributeError, ValueError, IndexError) as exc:
        raise RuntimeError(f"GDK motion status does not contain frame {frame_name}") from exc
    position = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
    orientation = (
        float(pose.orientation.x), float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    )
    if not all(math.isfinite(value) for value in (*position, *orientation)):
        raise RuntimeError(f"GDK motion status pose for {frame_name} is not finite")
    return StatusPose(position, orientation)


def status_summary(status, pose: StatusPose) -> dict:
    return {
        "mode": int(status.mode),
        "control_mode": int(status.control_mode),
        "error_code": int(status.error_code),
        "error_msg": str(status.error_msg),
        "frame": LEFT_END_LINK,
        "position_m": list(pose.position_m),
        "orientation_xyzw": list(pose.orientation_xyzw),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2 左臂 mode-1 末端位姿 1 mm 受看护探针")
    parser.add_argument("--axis", choices=("x", "y", "z"), required=True, help="base_link 平移轴。")
    parser.add_argument("--delta-m", type=float, default=0.001, help="有符号平移距离，最大绝对值 1 mm。")
    parser.add_argument("--duration-s", type=float, default=0.50, help="50 Hz 插补持续时间，0.2..2.0 秒。")
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/left_end_pose_mode1_*.json")
    parser.add_argument("--allow-motion", action="store_true", help="唯一动作确认参数；未指定时严格只读。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.delta_m) or not 0.0 < abs(args.delta_m) <= 0.001:
        print("参数错误：--delta-m 的绝对值必须在 (0, 0.001] 米。", file=sys.stderr)
        return 2
    if not math.isfinite(args.duration_s) or not 0.2 <= args.duration_s <= 2.0:
        print("参数错误：--duration-s 必须在 0.2..2.0 秒。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    log_path = args.log or PROJECT_ROOT / "logs" / f"left_end_pose_mode1_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    document: dict = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": not args.allow_motion,
        "control": "GDK end_effector_pose_control in existing mode 1 only",
        "group": "kLeftArm",
        "frame": LEFT_END_LINK,
        "requested_base_delta_m": {"x": 0.0, "y": 0.0, "z": 0.0},
        "duration_s": args.duration_s,
        "rate_hz": 50.0,
        "safety": "no gripper; no depth approach; no GDK/application mode change; exactly one interpolated 1 mm translation at most",
    }
    document["requested_base_delta_m"][args.axis] = float(args.delta_m)
    initialized = False
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        controller.checked_arm_state()
        before_status = robot.get_motion_control_status()
        before_pose = pose_from_status(before_status, LEFT_END_LINK)
        document["before"] = status_summary(before_status, before_pose)
        if not args.allow_motion:
            print(json.dumps({"log": str(log_path), "mode": "read_only", "before": document["before"], "safety": document["safety"]}, ensure_ascii=False, indent=2))
            return 0
        if int(before_status.control_mode) != 1 or int(before_status.error_code) != 0:
            raise RuntimeError(
                f"探针要求健康的 control_mode=1，当前 control_mode={int(before_status.control_mode)}, "
                f"error={int(before_status.error_code)}"
            )

        delta = dict(document["requested_base_delta_m"])
        target_position = (
            before_pose.position_m[0] + float(delta["x"]),
            before_pose.position_m[1] + float(delta["y"]),
            before_pose.position_m[2] + float(delta["z"]),
        )
        target = CartesianPoseTarget(target_position, before_pose.orientation_xyzw)
        document["target"] = {"position_m": list(target.position_m), "orientation_xyzw": list(target.orientation_xyzw)}

        steps = max(1, int(round(args.duration_s * 50.0)))
        period_s = args.duration_s / steps
        deadline = time.monotonic()
        sent = 0
        for index in range(1, steps + 1):
            fraction = index / steps
            position = tuple(
                before + fraction * (after - before)
                for before, after in zip(before_pose.position_m, target.position_m)
            )
            controller.send_left_end_effector_position_setpoint(
                CartesianPoseTarget(position, before_pose.orientation_xyzw), life_time_s=0.10
            )
            sent += 1
            deadline += period_s
            delay = deadline - time.monotonic()
            if delay > 0.0:
                time.sleep(delay)
        time.sleep(0.20)
        after_status = robot.get_motion_control_status()
        after_pose = pose_from_status(after_status, LEFT_END_LINK)
        actual_delta = tuple(after - before for before, after in zip(before_pose.position_m, after_pose.position_m))
        document.update({
            "commands_sent": sent,
            "after": status_summary(after_status, after_pose),
            "actual_base_delta_m": list(actual_delta),
            "actual_translation_norm_m": float(math.sqrt(sum(value * value for value in actual_delta))),
        })
        print(json.dumps({"log": str(log_path), "before": document["before"], "target": document["target"], "after": document["after"], "actual_base_delta_m": document["actual_base_delta_m"], "commands_sent": sent, "safety": document["safety"]}, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        document["interrupted"] = True
        print("收到 Ctrl-C；已停止发送后续末端目标。")
        return 0
    except Exception as exc:
        document["error"] = str(exc)
        print(f"左臂 mode-1 末端位姿探针失败：{exc}", file=sys.stderr)
        return 1
    finally:
        document["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            log_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"写入探针日志失败：{exc}", file=sys.stderr)
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
