#!/usr/bin/env python3
"""Explicitly place G2 in control_mode=1 without commanding robot motion.

This is intentionally separate from visual grasp.  It changes only GDK's
persisted control-mode state; it never sends an end-pose, joint, gripper,
application-mode, impedance, or payload command.  Without ``--allow-motion``
it is a read-only status check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def status_dict(status) -> dict:
    return {
        "mode": int(status.mode),
        "control_mode": int(status.control_mode),
        "error_code": int(status.error_code),
        "error_msg": str(status.error_msg),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 G2 切到关节位控 control_mode=1（不发送运动目标）")
    parser.add_argument("--timeout-s", type=float, default=2.0, help="切换后的状态确认超时，0.2..5 秒。")
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/left_position_mode_*.json")
    parser.add_argument("--allow-motion", action="store_true", help="唯一确认参数；未指定时只读取当前模式。")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0.2 <= args.timeout_s <= 5.0:
        print("参数错误：--timeout-s 必须在 0.2..5 秒。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2
    log_path = (args.log or PROJECT_ROOT / "logs" / f"left_position_mode_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json").expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": 1,
        "requested_control_mode": 1,
        "read_only": not args.allow_motion,
        "safety": "no end-pose, joint, gripper, application-mode, impedance, payload, or robot-start command is issued",
    }
    initialized = False
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        before = robot.get_motion_control_status()
        document["before"] = status_dict(before)
        if not args.allow_motion:
            print(json.dumps({"log": str(log_path), **document}, ensure_ascii=False, indent=2))
            return 0
        if int(before.error_code) != 0:
            raise RuntimeError(f"拒绝切换：当前运动错误 {int(before.error_code)}: {before.error_msg}")
        if int(before.control_mode) != 1:
            robot.set_control_mode(1)
        deadline = time.monotonic() + args.timeout_s
        while True:
            after = robot.get_motion_control_status()
            if int(after.control_mode) == 1 and int(after.error_code) == 0:
                document["after"] = status_dict(after)
                document["mode_change_sent"] = int(before.control_mode) != 1
                print(json.dumps({"log": str(log_path), **document}, ensure_ascii=False, indent=2))
                return 0
            if time.monotonic() >= deadline:
                raise RuntimeError(f"切换后未进入健康 control_mode=1：{status_dict(after)}")
            time.sleep(0.1)
    except Exception as exc:
        document["error"] = str(exc)
        print(f"左臂位控模式切换失败：{exc}", file=sys.stderr)
        return 1
    finally:
        document["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        try:
            log_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            print(f"写入模式日志失败：{exc}", file=sys.stderr)
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
