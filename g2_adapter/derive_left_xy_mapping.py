#!/usr/bin/env python3
"""Derive a GDK end-pose D405 pixel Jacobian from two guarded probe logs.

This script is offline: it imports no GDK binding and never contacts or starts
the robot. By default it only prints the proposed mapping. ``--write-config``
updates the supplied local JSON configuration but deliberately leaves
``calibrated`` unchanged, so it cannot enable a full grasp by itself.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from .xy_mapping import pixel_jacobian_from_end_pose_trials
except ImportError:
    from xy_mapping import pixel_jacobian_from_end_pose_trials


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = "gdk_mode1_end_effector_pose_control"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从两次 GDK 末端位姿小步日志生成左手 XY 像素 Jacobian")
    parser.add_argument("--x-log", required=True, type=Path, help="+X 标定的 JSONL 日志。")
    parser.add_argument("--y-log", required=True, type=Path, help="+Y 标定的 JSONL 日志。")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "left_grasp.example.json")
    parser.add_argument("--write-config", action="store_true", help="确认后才把计算出的映射写入本地配置；不改变 calibrated。")
    return parser.parse_args()


def result_from_log(path: Path) -> dict:
    records = []
    for number, line in enumerate(path.expanduser().read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path}:{number} 不是 JSONL") from exc
        if record.get("event") == "result":
            records.append(record)
    if len(records) != 1:
        raise RuntimeError(f"{path} 必须且只能包含一个 result 记录，实际为 {len(records)}")
    result = records[0]
    if result.get("controller") != CONTROLLER or result.get("frame") != "arm_l_end_link":
        raise RuntimeError(f"{path} 不是 GDK mode-1 arm_l_end_link 标定日志")
    for key in ("actual_end_link_base_delta_m", "pixel_delta_post_minus_pre_px"):
        if key not in result:
            raise RuntimeError(f"{path} 缺少 {key}")
    return result


def main() -> int:
    args = parse_args()
    try:
        x_result = result_from_log(args.x_log)
        y_result = result_from_log(args.y_log)
        jacobian = pixel_jacobian_from_end_pose_trials(
            x_actual_base_delta_m=x_result["actual_end_link_base_delta_m"],
            y_actual_base_delta_m=y_result["actual_end_link_base_delta_m"],
            x_pixel_delta_px=x_result["pixel_delta_post_minus_pre_px"],
            y_pixel_delta_px=y_result["pixel_delta_post_minus_pre_px"],
        )
        output = {
            "mapping_controller": CONTROLLER,
            "x_trial": {"actual_end_link_base_delta_m": x_result["actual_end_link_base_delta_m"], "pixel_delta_px": x_result["pixel_delta_post_minus_pre_px"]},
            "y_trial": {"actual_end_link_base_delta_m": y_result["actual_end_link_base_delta_m"], "pixel_delta_px": y_result["pixel_delta_post_minus_pre_px"]},
            "pixel_jacobian_px_per_m": jacobian.tolist(),
            "write_config": bool(args.write_config),
        }
        if args.write_config:
            config_path = args.config.expanduser().resolve()
            document = json.loads(config_path.read_text(encoding="utf-8"))
            alignment = document.setdefault("alignment", {})
            alignment["pixel_jacobian_px_per_m"] = jacobian.tolist()
            alignment["mapping_controller"] = CONTROLLER
            config_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            output["config_written"] = str(config_path)
            output["calibrated_remains"] = document.get("calibrated")
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"左手 XY 映射生成失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
