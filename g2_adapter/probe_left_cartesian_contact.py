#!/usr/bin/env python3
"""Supervised left-arm GDK mode-2 contact-data probe.

Without ``--allow-motion`` this reads and writes one status snapshot only.
With that one confirmation flag, it changes to Cartesian impedance mode, holds
the *currently measured* left TCP pose at 50 Hz, and records raw force data
while the operator applies a gentle manual contact.  It never commands an arm
translation or a gripper.  At the end (including Ctrl-C), it stops the pose
stream and restores the original control mode.
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
    from .camera import transform_to_dict
    from .control import CartesianPoseTarget, G2Controller, LeftCartesianImpedanceStream
except ImportError:
    from camera import transform_to_dict
    from control import CartesianPoseTarget, G2Controller, LeftCartesianImpedanceStream


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEFT_CONTACT_KP = (600.0, 600.0, 600.0, 30.0, 30.0, 30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集左手 GDK mode-2 接触力原始数据")
    parser.add_argument("--duration-s", type=float, default=15.0, help="采集时长，默认 15 秒。")
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/left_contact_probe_*.json")
    parser.add_argument("--allow-motion", action="store_true", help="唯一动作确认参数；允许切换 mode 2 并保持当前位姿。")
    return parser.parse_args()


def vector3(value) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def status_sample(status) -> dict:
    return {
        "monotonic_s": time.monotonic(),
        "mode": int(status.mode),
        "control_mode": int(status.control_mode),
        "error_code": int(status.error_code),
        "error_msg": str(status.error_msg),
        "frame_names": [str(value) for value in status.frame_names],
        "wrenches": [
            {"force_n": vector3(wrench.force), "torque_nm": vector3(wrench.torque)}
            for wrench in status.wrenches
        ],
        "arm_disturbance_force": [float(value) for value in status.arm_disturbance_force],
    }


def measured_left_end_pose(tf) -> CartesianPoseTarget:
    timestamp_ns = int(tf.get_latest_timestamp("arm_l_end_link"))
    if timestamp_ns <= 0:
        raise RuntimeError("GDK has no current arm_l_end_link timestamp")
    transform = transform_to_dict(tf.lookup_transform("base_link", "arm_l_end_link", timestamp_ns))
    return CartesianPoseTarget(
        tuple(float(value) for value in transform["translation_m"]),
        tuple(float(value) for value in transform["rotation_xyzw"]),
    )


def main() -> int:
    args = parse_args()
    if not math.isfinite(args.duration_s) or not 5.0 <= args.duration_s <= 60.0:
        print("参数错误：--duration-s 必须在 5..60 秒。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    log_path = args.log or PROJECT_ROOT / "logs" / f"left_contact_probe_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    initialized = False
    stream: LeftCartesianImpedanceStream | None = None
    controller: G2Controller | None = None
    original_mode: int | None = None
    samples: list[dict] = []
    document: dict = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": not args.allow_motion,
        "duration_s": args.duration_s,
        "left_cartesian_impedance_kp": list(LEFT_CONTACT_KP),
        "safety": "no gripper command; no commanded TCP translation; mode-2 test restores the original control mode after stopping its 50-Hz hold stream",
    }
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        controller.checked_arm_state()
        initial_status = robot.get_motion_control_status()
        original_mode = int(initial_status.control_mode)
        document["initial_status"] = status_sample(initial_status)
        if not args.allow_motion:
            document["samples"] = [status_sample(robot.get_motion_control_status())]
            print(json.dumps({"log": str(log_path), "mode": "read_only", "initial_status": document["initial_status"], "safety": document["safety"]}, ensure_ascii=False, indent=2))
            return 0
        if original_mode != 1:
            raise RuntimeError(f"接触力采集必须从 position mode 1 开始，当前为 mode {original_mode}")

        # Read before the mode switch and use it as the first mode-2 target.
        # This is a hold of the observed TCP, not a displacement command.
        target = measured_left_end_pose(agibot_gdk.TF())
        document["held_left_tcp_target_base_link"] = {
            "position_m": list(target.position_m),
            "orientation_xyzw": list(target.orientation_xyzw),
        }
        controller.enable_left_cartesian_impedance(LEFT_CONTACT_KP)
        stream = controller.start_left_cartesian_impedance_stream(target, rate_hz=50.0, life_time_s=0.10)
        print("已进入 mode 2 并保持当前左手位姿。现在只需轻推左手末端约 2 秒，然后松手；不要推动夹爪以外的部位。")
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline:
            stream.raise_if_failed()
            samples.append(status_sample(robot.get_motion_control_status()))
            time.sleep(0.02)
        document["samples"] = samples
        document["completed"] = True
        return 0
    except KeyboardInterrupt:
        document["samples"] = samples
        document["interrupted"] = True
        print("收到 Ctrl-C；正在停止保持并恢复原控制模式。")
        return 0
    except Exception as exc:
        document["samples"] = samples
        document["error"] = str(exc)
        print(f"左手接触力采集失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if stream is not None:
            stream.stop()
        if controller is not None and args.allow_motion and original_mode is not None:
            try:
                controller.set_control_mode(original_mode)
                document["restored_control_mode"] = original_mode
            except Exception as exc:
                document["restore_control_mode_error"] = str(exc)
                print(f"未能恢复 control_mode={original_mode}：{exc}", file=sys.stderr)
        document["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            log_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"写入接触力日志失败：{exc}", file=sys.stderr)
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
