#!/usr/bin/env python3
"""Run the existing left visual workflow followed by the right visual workflow.

Each child owns its own GDK, camera, and SpaceMouse reader. The left child
first returns both arms to the recorded workflow home pose, then opens its
gripper and reaches both handoff presets before releasing its GDK.
A left child exit code of ``20`` means that handoff completed and the right
stage may start.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

try:
    from .workflow_home import WORKFLOW_HOME_POSE_NAME, load_workflow_home
except ImportError:
    from workflow_home import WORKFLOW_HOME_POSE_NAME, load_workflow_home


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXIT_ADVANCE_RIGHT = 20


def validate_handoff_poses(path):
    try:
        from .run_left_premove_sequence import HANDOFF_PRESETS, load_presets
    except ImportError:
        from run_left_premove_sequence import HANDOFF_PRESETS, load_presets
    return load_presets(Path(path), HANDOFF_PRESETS)


def validate_workflow_home(path):
    return load_workflow_home(Path(path), WORKFLOW_HOME_POSE_NAME)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="左手视觉流程完成后串联右手视觉流程，并支持 SpaceMouse 临时接管")
    parser.add_argument("--device", required=True, help="SpaceMouse Compact 的 /dev/hidrawN 路径或 udev 固定别名")
    parser.add_argument("--left-config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "left_grasp.example.json")
    # Kept only so older shell history still parses; the bilateral workflow
    # always uses the YOLO box landmark and never forwards STEP/PnP options.
    parser.add_argument("--left-step-profile", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--left-step-reference", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--left-height-reference",
        type=Path,
        default=PROJECT_ROOT / "g2_adapter" / "left_step_reference.json",
        help="左手 10 cm/5 cm/抓取点三点像素标定，不使用 STEP/PnP 几何。",
    )
    parser.add_argument("--right-config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "right_apriltag_grasp.example.json")
    parser.add_argument("--poses", type=Path, default=PROJECT_ROOT / "g2_adapter" / "recorded_poses.json")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt")
    parser.add_argument("--yolo-model", type=Path, default=PROJECT_ROOT / "models" / "left_d405_hinge_yolo" / "best.pt")
    parser.add_argument("--head-yolo-model", type=Path, default=PROJECT_ROOT / "models" / "head_hinge_yolo" / "best.pt")
    parser.add_argument("--head-yolo-confidence", type=float, default=0.50,
                        help="头部 YOLO 接受检测的最低置信度，范围 0.50～1.00。")
    parser.add_argument("--yolo-center-fraction", type=float, default=0.70,
                        help="左手 YOLO 全图推理时的中央有效目标区域宽高比例，默认 0.70。")
    parser.add_argument("--post-grasp-target-lost-timeout-s", type=float, default=3.0,
                        help="抓取后同一目标连续不可见多久才重新抓取，默认 3 秒。")
    parser.add_argument("--final-descent-visual-tolerance-mm", type=float, default=5.0,
                        help="左手下探视觉伺服的估计水平误差死区，默认 5 mm。")
    parser.add_argument("--head-forward-yaw-deg", type=float, default=0.0,
                        help="头部粗定位时末端 +Z 相对 base_link +X 的水平偏航角。")
    parser.add_argument("--head-gripper-pitch-deg", type=float, default=30.0,
                        help="头部粗定位时末端 +Z 向下俯角，正值向下。")
    parser.add_argument("--head-gripper-roll-deg", type=float, default=180.0,
                        help="头部粗定位时绕末端 +Z 的滚转角，默认 180 度。")
    parser.add_argument("--head-above-mm", type=float, default=100.0,
                        help="头部粗定位后夹爪中心位于铰链上方的距离，默认 100 mm。")
    parser.add_argument("--head-samples", type=int, default=15,
                        help="头部粗定位候选三维样本数。")
    parser.add_argument("--head-min-inlier-samples", type=int, default=8,
                        help="头部粗定位空间一致主簇所需最少样本数。")
    parser.add_argument("--head-max-sample-spread-mm", type=float, default=50.0,
                        help="头部粗定位主簇相对中位数的最大距离。")
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
    parser.add_argument(
        "--auto-speed-mm-s", type=float, default=None,
        help="统一视觉自动平移速度（左手头部粗移/XY/下探及右手笛卡尔路径）；不指定则使用各 demo 配置",
    )
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


def build_stage_commands(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    python = sys.executable
    head_yolo_model = getattr(
        args, "head_yolo_model",
        PROJECT_ROOT / "models" / "head_hinge_yolo" / "best.pt",
    )
    left = [
        python, str(PROJECT_ROOT / "g2_adapter" / "left_grasp_gdk.py"),
        "--config", str(Path(args.left_config).expanduser()),
        "--checkpoint", str(Path(args.checkpoint).expanduser()),
        "--detector", "yolo",
        "--yolo-model", str(Path(args.yolo_model).expanduser()),
        "--head-coarse-approach",
        "--head-forward-yaw-deg", str(getattr(args, "head_forward_yaw_deg", 0.0)),
        "--head-gripper-pitch-deg", str(getattr(args, "head_gripper_pitch_deg", 30.0)),
        "--head-gripper-roll-deg", str(getattr(args, "head_gripper_roll_deg", 180.0)),
        "--head-above-mm", str(getattr(args, "head_above_mm", 100.0)),
        "--head-samples", str(getattr(args, "head_samples", 15)),
        "--head-min-inlier-samples", str(getattr(args, "head_min_inlier_samples", 8)),
        "--head-max-sample-spread-mm", str(getattr(args, "head_max_sample_spread_mm", 50.0)),
        "--head-yolo-model", str(Path(head_yolo_model).expanduser()),
        "--head-yolo-confidence", str(getattr(args, "head_yolo_confidence", 0.50)),
        "--height-reference", str(Path(args.left_height_reference).expanduser()),
        "--yolo-center-fraction", str(getattr(args, "yolo_center_fraction", 0.70)),
        "--post-grasp-target-lost-timeout-s", str(getattr(args, "post_grasp_target_lost_timeout_s", 3.0)),
        "--final-descent-visual-servo",
        "--final-descent-visual-tolerance-mm", str(getattr(args, "final_descent_visual_tolerance_mm", 5.0)),
        "--home-pose-name", WORKFLOW_HOME_POSE_NAME,
        "--poses", str(Path(args.poses).expanduser()),
        "--full-cycle",
        "--wait-for-next",
        "--web-port", str(args.left_web_port),
        # Keep every left automatic Cartesian path on the same speed setting:
        # coarse head approach/continuous XY use --xy-speed-mm-s, while the
        # final descent uses --speed-mm-s below.
        *(["--xy-speed-mm-s", str(args.auto_speed_mm_s)] if getattr(args, "auto_speed_mm_s", None) is not None else []),
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
        and 0.5 <= getattr(args, "head_yolo_confidence", 0.50) <= 1.0
        and math.isfinite(getattr(args, "head_above_mm", 100.0))
        and 1.0 <= getattr(args, "head_above_mm", 100.0) <= 500.0
        and 0.0 < getattr(args, "yolo_center_fraction", 0.70) <= 1.0
        and 0.10 <= getattr(args, "post_grasp_target_lost_timeout_s", 3.0) <= 30.0
        and math.isfinite(getattr(args, "final_descent_visual_tolerance_mm", 5.0))
        and getattr(args, "final_descent_visual_tolerance_mm", 5.0) > 0.0
        and (args.auto_speed_mm_s is None or (math.isfinite(args.auto_speed_mm_s) and args.auto_speed_mm_s > 0.0))
        and (args.auto_rotation_deg_s is None or (math.isfinite(args.auto_rotation_deg_s) and args.auto_rotation_deg_s > 0.0))
        and (args.auto_preset_velocity_rad_s is None or (math.isfinite(args.auto_preset_velocity_rad_s) and args.auto_preset_velocity_rad_s > 0.0))
    ):
        print("SpaceMouse 或视觉自动速度参数无效。", file=sys.stderr)
        return 2
    left_command, right_command = build_stage_commands(args)
    try:
        validate_handoff_poses(args.poses)
    except Exception as exc:
        print(f"左手交接点未就绪：{exc}；请先记录 left_handoff_1 和 left_handoff_2。", file=sys.stderr)
        return 2
    try:
        validate_workflow_home(args.poses)
    except Exception as exc:
        print(f"双臂归位点未就绪：{exc}；请先记录 workflow_home。", file=sys.stderr)
        return 2
    print(json.dumps({"event": "left_stage_start", "command": left_command}, ensure_ascii=False), flush=True)
    left_result = subprocess.run(left_command, cwd=PROJECT_ROOT)
    if left_result.returncode not in (0, EXIT_ADVANCE_RIGHT):
        print(json.dumps({"event": "left_stage_failed", "returncode": left_result.returncode}, ensure_ascii=False), file=sys.stderr, flush=True)
        return int(left_result.returncode)
    if left_result.returncode != EXIT_ADVANCE_RIGHT:
        print("左手进程未完成进程内交接；拒绝启动右手。", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"event": "right_stage_start", "left_returncode": left_result.returncode, "command": right_command}, ensure_ascii=False), flush=True)
    right_result = subprocess.run(right_command, cwd=PROJECT_ROOT)
    if right_result.returncode != 0:
        print(json.dumps({"event": "right_stage_failed", "returncode": right_result.returncode}, ensure_ascii=False), file=sys.stderr, flush=True)
    else:
        print(json.dumps({"event": "bilateral_workflow_complete"}, ensure_ascii=False), flush=True)
    return int(right_result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
