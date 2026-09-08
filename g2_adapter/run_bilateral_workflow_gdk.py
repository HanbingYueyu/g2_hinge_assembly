#!/usr/bin/env python3
"""Run the existing left visual workflow followed by the right visual workflow.

Each child owns its own GDK, camera, and SpaceMouse reader.  The parent never
opens hidraw or creates a Robot, so serial stage transitions preserve the
single-writer invariant.  A left child exit code of ``20`` means that the
SpaceMouse right-button 2-second hold requested the same transition as normal
left completion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXIT_ADVANCE_RIGHT = 20


def validate_handoff_poses(path):
    try:
        from .run_left_premove_sequence import HANDOFF_PRESETS, load_presets
    except ImportError:
        from run_left_premove_sequence import HANDOFF_PRESETS, load_presets
    return load_presets(Path(path), HANDOFF_PRESETS)


def build_handoff_command(args):
    velocity = args.auto_preset_velocity_rad_s
    return [
        sys.executable, str(PROJECT_ROOT / "g2_adapter" / "run_left_premove_sequence.py"),
        "--handoff", "--poses", str(Path(args.poses).expanduser()),
        "--velocity-rad-s", str(0.08 if velocity is None else velocity),
        "--allow-motion",
    ]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="左手视觉流程完成后串联右手视觉流程，并支持 SpaceMouse 临时接管")
    parser.add_argument("--device", required=True, help="SpaceMouse Compact 的 /dev/hidrawN 路径或 udev 固定别名")
    parser.add_argument("--left-config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "left_grasp.example.json")
    parser.add_argument("--right-config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "right_apriltag_grasp.example.json")
    parser.add_argument("--poses", type=Path, default=PROJECT_ROOT / "g2_adapter" / "recorded_poses.json")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt")
    parser.add_argument("--yolo-model", type=Path, default=PROJECT_ROOT / "models" / "left_d405_hinge_shaft_yolo" / "best.pt")
    parser.add_argument("--left-web-port", type=int, default=8767)
    parser.add_argument("--right-web-port", type=int, default=8767)
    parser.add_argument(
        "--speed-mm-s", "--spacemouse-speed-mm-s",
        dest="speed_mm_s", type=float, default=10.0,
        help="SpaceMouse 接管平移速度上限",
    )
    parser.add_argument(
        "--rotation-deg-s", "--spacemouse-rotation-deg-s",
        dest="rotation_deg_s", type=float, default=10.0,
        help="SpaceMouse 接管旋转速度上限",
    )
    parser.add_argument("--auto-speed-mm-s", type=float, default=None, help="视觉自动末端平移速度；不指定则使用各 demo 配置")
    parser.add_argument("--auto-rotation-deg-s", type=float, default=None, help="视觉自动末端旋转速度；不指定则使用右手配置")
    parser.add_argument("--auto-preset-velocity-rad-s", type=float, default=None, help="视觉自动关节预设速度；不指定则使用各 demo 默认值")
    parser.add_argument("--deadzone", type=float, default=0.12)
    parser.add_argument("--allow-motion", action="store_true", help="允许左右 demo 发送真实 GDK 运动")
    return parser.parse_args(argv)


def _common_spacemouse_args(args: argparse.Namespace) -> list[str]:
    return [
        "--spacemouse-device", args.device,
        "--spacemouse-speed-mm-s", str(args.speed_mm_s),
        "--spacemouse-rotation-deg-s", str(args.rotation_deg_s),
        "--spacemouse-deadzone", str(args.deadzone),
    ]


def _optional_auto_args(args: argparse.Namespace) -> list[str]:
    result: list[str] = []
    if getattr(args, "auto_speed_mm_s", None) is not None:
        result.extend(("--auto-speed-mm-s", str(args.auto_speed_mm_s)))
    if getattr(args, "auto_rotation_deg_s", None) is not None:
        result.extend(("--auto-rotation-deg-s", str(args.auto_rotation_deg_s)))
    if getattr(args, "auto_preset_velocity_rad_s", None) is not None:
        result.extend(("--auto-preset-velocity-rad-s", str(args.auto_preset_velocity_rad_s)))
    return result


def wait_for_right_advance(
    device: str,
    *,
    hid_factory=None,
    clock=time.monotonic,
    sleep=time.sleep,
) -> bool:
    """Wait for the manual right-button 2-second stage-advance gesture.

    The left child has already exited before this function opens hidraw, so
    the parent never competes with a running GDK/SpaceMouse owner.
    """
    try:
        from .spacemouse_input import CompactHID
        from .spacemouse_takeover import RIGHT_LONG_PRESS_S, SpaceMouseGestureState, TakeoverEvent
    except ImportError:
        from spacemouse_input import CompactHID
        from spacemouse_takeover import RIGHT_LONG_PRESS_S, SpaceMouseGestureState, TakeoverEvent

    factory = CompactHID if hid_factory is None else hid_factory
    gestures = SpaceMouseGestureState()
    with factory(device) as source:
        while True:
            now = clock()
            result = gestures.update(source.poll(), now)
            if TakeoverEvent.ADVANCE_RIGHT in result.events:
                return True
            sleep(min(0.02, RIGHT_LONG_PRESS_S / 10.0))


def build_stage_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    python = sys.executable
    left = [
        python, str(PROJECT_ROOT / "g2_adapter" / "left_grasp_gdk.py"),
        "--config", str(Path(args.left_config).expanduser()),
        "--checkpoint", str(Path(args.checkpoint).expanduser()),
        "--yolo-model", str(Path(args.yolo_model).expanduser()),
        "--poses", str(Path(args.poses).expanduser()),
        "--full-cycle",
        "--wait-for-next",
        "--web-port", str(args.left_web_port),
        *(["--speed-mm-s", str(args.auto_speed_mm_s)] if getattr(args, "auto_speed_mm_s", None) is not None else []),
        *(["--auto-preset-velocity-rad-s", str(args.auto_preset_velocity_rad_s)] if getattr(args, "auto_preset_velocity_rad_s", None) is not None else []),
        *_common_spacemouse_args(args),
        "--allow-motion",
    ]
    right = [
        python, str(PROJECT_ROOT / "g2_adapter" / "right_apriltag_grasp_gdk.py"),
        "--config", str(Path(args.right_config).expanduser()),
        "--tag-geometry-profile", str(PROJECT_ROOT / "g2_adapter" / "tag_geometry_electric_screwdriver.json"),
        "--poses", str(Path(args.poses).expanduser()),
        "--web-port", str(args.right_web_port),
        "--execute", "--autonomous-approach", "--allow-motion",
        "--premove-pr1", "--post-grasp-pr2",
        *_common_spacemouse_args(args),
        *_optional_auto_args(args),
    ]
    return left, right


def main(argv=None) -> int:
    args = parse_args(argv) if argv is None or isinstance(argv, (list, tuple)) else argv
    if not args.allow_motion:
        print("双手串联流程必须显式指定 --allow-motion；否则不会启动动作。", file=sys.stderr)
        return 2
    if not args.device.startswith("/dev/"):
        print("--device 必须是 /dev/hidrawN 路径。", file=sys.stderr)
        return 2
    if not (
        1.0 <= args.speed_mm_s <= 30.0
        and 1.0 <= args.rotation_deg_s <= 28.6
        and 0.0 <= args.deadzone <= 0.5
        and (args.auto_speed_mm_s is None or 1.0 <= args.auto_speed_mm_s <= 18.0)
        and (args.auto_rotation_deg_s is None or 1.0 <= args.auto_rotation_deg_s <= 28.6)
        and (args.auto_preset_velocity_rad_s is None or 0.001 <= args.auto_preset_velocity_rad_s <= 0.15)
    ):
        print("SpaceMouse 或视觉自动速度参数无效。", file=sys.stderr)
        return 2
    left_command, right_command = build_stage_commands(args)
    try:
        validate_handoff_poses(args.poses)
    except Exception as exc:
        print(f"左手交接点未就绪：{exc}；请先记录 left_handoff_1 和 left_handoff_2。", file=sys.stderr)
        return 2
    print(json.dumps({"event": "left_stage_start", "command": left_command}, ensure_ascii=False), flush=True)
    left_result = subprocess.run(left_command, cwd=PROJECT_ROOT)
    if left_result.returncode not in (0, EXIT_ADVANCE_RIGHT):
        print(json.dumps({"event": "left_stage_failed", "returncode": left_result.returncode}, ensure_ascii=False), file=sys.stderr, flush=True)
        return int(left_result.returncode)
    if left_result.returncode == 0:
        print(json.dumps({
            "event": "left_stage_complete_waiting_for_manual_next",
            "source": "spacemouse_right_long_press_2s",
        }, ensure_ascii=False), flush=True)
        try:
            wait_for_right_advance(args.device)
        except KeyboardInterrupt:
            print("等待右手启动时收到 Ctrl-C；不启动右手流程。", file=sys.stderr, flush=True)
            return 130
        except Exception as exc:
            print(json.dumps({
                "event": "manual_right_advance_failed",
                "error": str(exc),
            }, ensure_ascii=False), file=sys.stderr, flush=True)
            return 1
    handoff_command = build_handoff_command(args)
    print(json.dumps({"event": "left_handoff_start", "command": handoff_command}, ensure_ascii=False), flush=True)
    handoff_result = subprocess.run(handoff_command, cwd=PROJECT_ROOT)
    if handoff_result.returncode != 0:
        print(json.dumps({"event": "left_handoff_failed", "returncode": handoff_result.returncode}), file=sys.stderr, flush=True)
        return int(handoff_result.returncode)
    print(json.dumps({"event": "right_stage_start", "left_returncode": left_result.returncode, "command": right_command}, ensure_ascii=False), flush=True)
    right_result = subprocess.run(right_command, cwd=PROJECT_ROOT)
    if right_result.returncode != 0:
        print(json.dumps({"event": "right_stage_failed", "returncode": right_result.returncode}, ensure_ascii=False), file=sys.stderr, flush=True)
    else:
        print(json.dumps({"event": "bilateral_workflow_complete"}, ensure_ascii=False), flush=True)
    return int(right_result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
