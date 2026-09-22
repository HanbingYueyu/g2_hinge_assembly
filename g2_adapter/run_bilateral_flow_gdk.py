#!/usr/bin/env python3
"""Single-process staged bilateral workflow for the G2 electric-screwdriver cell.

One process owns exactly one GDK session, one ``G2Controller``, one
``MotionPauseGate``, one SpaceMouse worker, three cameras and three preloaded
models (head hinge+hole YOLO, left D405 YOLO, right AprilTag detector).  SAM3 is
never imported or loaded here.

The stage order and the two SpaceMouse "next stage" gates live in
``bilateral_stages.DEFAULT_PLAN``; ``--list-stages`` prints them and ``--dry-run``
validates every recorded preset without touching the robot.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    from .bilateral_runtime import BilateralRuntime, PreloadedModels, append_jsonl_record, console_event_summary
    from .bilateral_services import StageServices
    from .bilateral_stages import DEFAULT_PLAN, Stage, StageSequencer, stage_names
    from .bilateral_web import BilateralWeb, RawPaneFeed
    from .safety import SafetyError
    from .keyboard_stage_advance import KeyboardStageAdvance
except ImportError:
    from bilateral_runtime import BilateralRuntime, PreloadedModels, append_jsonl_record, console_event_summary
    from bilateral_services import StageServices
    from bilateral_stages import DEFAULT_PLAN, Stage, StageSequencer, stage_names
    from bilateral_web import BilateralWeb, RawPaneFeed
    from safety import SafetyError
    from keyboard_stage_advance import KeyboardStageAdvance


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class JsonLineSuppressingStream:
    """Pass human-readable child output through while hiding JSON event lines."""

    def __init__(self, destination) -> None:
        self.destination = destination
        self._buffer = ""

    def write(self, text: str) -> int:
        self._buffer += str(text)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not line.lstrip().startswith("{"):
                self.destination.write(line + "\n")
        return len(text)

    def flush(self) -> None:
        if self._buffer and not self._buffer.lstrip().startswith("{"):
            self.destination.write(self._buffer)
        self._buffer = ""
        self.destination.flush()


@contextlib.contextmanager
def concise_child_stdout():
    stream = JsonLineSuppressingStream(sys.stdout)
    try:
        with contextlib.redirect_stdout(stream):
            yield
    finally:
        stream.flush()


@dataclass
class StageSettings:
    """Values the orchestrator forwards to the borrowed stage CLI parsers."""

    args: argparse.Namespace

    # ------------------------------------------------------------- left stage
    def left_argv(self) -> list[str]:
        args = self.args
        argv = [
            "--config", str(Path(args.left_config).expanduser()),
            "--detector", "yolo",
            "--yolo-model", str(Path(args.left_yolo_model).expanduser()),
            "--head-coarse-approach",
            "--head-yolo-model", str(Path(args.head_yolo_model).expanduser()),
            "--head-yolo-confidence", str(args.head_yolo_confidence),
            "--head-forward-yaw-deg", str(args.approach_yaw_deg),
            "--head-gripper-pitch-deg", str(args.approach_pitch_deg),
            "--head-gripper-roll-deg", str(args.approach_roll_deg),
            "--head-above-mm", str(args.approach_above_mm),
            "--head-samples", str(args.head_samples),
            "--head-min-inlier-samples", str(args.head_min_inlier_samples),
            "--head-max-sample-spread-mm", str(args.head_max_sample_spread_mm),
            # Keep the tuned left-grasp behaviour of the original bilateral
            # command: live XY visual servo during the final descent, the
            # central YOLO acceptance window, and the post-grasp target-loss
            # timeout.  Without the servo the descent only follows the
            # interpolated screen target and can push the part.
            "--final-descent-visual-servo",
            "--final-descent-visual-tolerance-mm", str(args.final_descent_visual_tolerance_mm),
            "--yolo-center-fraction", str(args.yolo_center_fraction),
            "--post-grasp-target-lost-timeout-s", str(args.post_grasp_target_lost_timeout_s),
            "--move-above-hole",
            "--hole-x-mm", str(args.left_hole_x_mm),
            "--hole-y-mm", str(args.left_hole_y_mm),
            "--hole-z-mm", str(args.left_hole_z_mm),
            "--hole-forward-yaw-deg", str(args.left_hole_yaw_deg),
            "--hole-gripper-pitch-deg", str(args.left_hole_pitch_deg),
            "--hole-gripper-roll-deg", str(args.left_hole_roll_deg),
            "--hole-speed-mm-s", str(args.auto_speed_mm_s),
            "--hole-sample-timeout-s", str(args.hole_sample_timeout_s),
            "--home-pose-name", args.home_pose_name,
            "--poses", str(Path(args.poses).expanduser()),
            "--full-cycle",
            "--skip-workflow-home",
            "--web-port", "0",
            "--xy-speed-mm-s", str(args.auto_speed_mm_s),
            "--speed-mm-s", str(args.auto_speed_mm_s),
            "--auto-preset-velocity-rad-s", str(args.auto_preset_velocity_rad_s),
            "--preset-arrival-tolerance-rad", str(args.preset_arrival_tolerance_rad),
            "--preset-timeout-s", str(args.preset_timeout_s),
            "--allow-motion",
        ]
        if args.left_height_reference is not None:
            argv.extend(("--height-reference", str(Path(args.left_height_reference).expanduser())))
        if getattr(args, "relief_extra_lift_mm", None) is not None:
            argv.extend(("--relief-extra-lift-mm", str(args.relief_extra_lift_mm)))
        if args.log is not None:
            argv.extend(("--log", str(Path(args.log).expanduser())))
        return argv

    # ------------------------------------------------------------ right stage
    def right_argv(self) -> list[str]:
        args = self.args
        argv = [
            "--config", str(Path(args.right_config).expanduser()),
            "--tag-geometry-profile", str(Path(args.tag_geometry_profile).expanduser()),
            "--poses", str(Path(args.poses).expanduser()),
            "--web-port", "0",
            "--execute", "--autonomous-approach", "--allow-motion",
            "--premove-pr1",
            "--premove-pose", args.right_premove_pose,
            "--grasp-distance-mm", str(args.right_grasp_distance_mm),
            "--stop-after-transport",
            "--auto-speed-mm-s", str(args.auto_speed_mm_s),
            "--auto-preset-velocity-rad-s", str(args.auto_preset_velocity_rad_s),
        ]
        if args.log is not None:
            argv.extend(("--log", str(Path(args.log).expanduser())))
        return argv


class FlowContext:
    """Everything a stage function needs, without a second GDK session."""

    def __init__(self, *, gdk, services: StageServices, runtime: BilateralRuntime,
                 settings: StageSettings, feeds: dict | None = None) -> None:
        self.gdk = gdk
        self.services = services
        self.runtime = runtime
        self.settings = settings
        # pane name -> RawPaneFeed, so a stage can own its camera's pane.
        self.feeds = dict(feeds or {})

    @property
    def args(self) -> argparse.Namespace:
        return self.settings.args


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("布尔参数必须是 true/false")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="SpaceMouse Compact 的 /dev/hidrawN 路径或 udev 固定别名")
    parser.add_argument("--poses", type=Path, default=PROJECT_ROOT / "g2_adapter" / "recorded_poses.json")
    parser.add_argument("--home-pose-name", default="workflow_home")
    parser.add_argument("--left-config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "left_grasp.example.json")
    parser.add_argument("--left-height-reference", type=Path, default=PROJECT_ROOT / "g2_adapter" / "left_step_reference.json")
    parser.add_argument("--right-config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "right_apriltag_grasp.example.json")
    parser.add_argument(
        "--tag-geometry-profile", type=Path,
        default=PROJECT_ROOT / "g2_adapter" / "tag_geometry_electric_screwdriver.json",
    )
    parser.add_argument("--head-yolo-model", type=Path, default=PROJECT_ROOT / "models" / "head_hinge_yolo" / "best.pt")
    parser.add_argument(
        "--left-yolo-model", type=Path,
        default=PROJECT_ROOT / "models" / "left_d405_hinge_yolo" / "best.pt",
        help="左手 D405 抓取检测权重；抓取只消费 hinge 类，默认单类权重。",
    )
    parser.add_argument("--head-yolo-confidence", type=float, default=0.50)
    parser.add_argument("--yolo-device", default="cuda:0")
    parser.add_argument("--head-samples", type=int, default=15)
    parser.add_argument("--head-min-inlier-samples", type=int, default=8)
    parser.add_argument("--head-max-sample-spread-mm", type=float, default=50.0)
    parser.add_argument("--head-localize-timeout-s", type=float, default=60.0)
    parser.add_argument("--yolo-center-fraction", type=float, default=0.70,
                        help="左手 D405 YOLO 全图推理时中央有效目标区域宽高比例，默认 0.70。")
    parser.add_argument("--final-descent-visual-tolerance-mm", type=float, default=5.0,
                        help="下探视觉伺服的估计水平误差死区，默认 5 mm。")
    parser.add_argument("--post-grasp-target-lost-timeout-s", type=float, default=1.0,
                        help="抓取后同一目标连续不可见多久才重新抓取，默认 1 秒。")
    parser.add_argument("--relief-extra-lift-mm", type=float, default=None,
                        help="泄力阈值达到后额外上抬的毫米数；省略时用 left config 的 contact.relief_extra_lift_m（默认 0），允许 [0, 20]。")
    parser.add_argument("--approach-yaw-deg", type=float, default=0.0, help="到铰链上方时的偏航角，默认 0（正对前方）。")
    parser.add_argument("--approach-pitch-deg", type=float, default=42.0, help="到铰链上方时的下俯角，默认 42。")
    parser.add_argument("--approach-roll-deg", type=float, default=180.0, help="到铰链上方时的滚转角，默认 180。")
    parser.add_argument("--approach-above-mm", type=float, default=100.0, help="铰链上方距离，默认 100 mm。")
    parser.add_argument("--left-hole-x-mm", type=float, default=0.0, help="左手相对 hole 的 base_link X 偏移，默认 0 mm。")
    parser.add_argument(
        "--teach-enable", nargs="?", const=True, default=False, type=_parse_bool,
        help="统一启用左右手示教轨迹，可写 true/false；不指定时使用原流程。",
    )
    parser.add_argument("--left-teach-trajectory", type=Path, help="左臂示教 JSON；支持旧关节轨迹或已 solve 的 *_cartesian_solved.json。")
    parser.add_argument("--left-teach-speed-scale", type=float, default=0.2)
    parser.add_argument("--left-teach-start-velocity-rad-s", type=float, default=0.05)
    parser.add_argument("--left-hole-y-mm", type=float, default=0.0, help="左手相对 hole 的 base_link Y 偏移，默认 0 mm。")
    parser.add_argument("--left-hole-z-mm", type=float, default=100.0, help="左手相对 hole 的 base_link Z 偏移，默认 100 mm。")
    parser.add_argument("--left-hole-yaw-deg", type=float, default=20.0)
    parser.add_argument("--left-hole-pitch-deg", type=float, default=42.0)
    parser.add_argument("--left-hole-roll-deg", type=float, default=180.0)
    parser.add_argument("--right-hole-x-mm", type=float, default=0.0, help="右手批头相对 hole 的 base_link X 偏移，默认 0 mm。")
    right_teach_files = parser.add_mutually_exclusive_group()
    right_teach_files.add_argument("--right-teach-trajectory", type=Path, help="单段右臂示教 JSON；支持旧关节轨迹或已 solve 的末端轨迹。")
    right_teach_files.add_argument("--right-teach-trajectories", type=Path, nargs=4, metavar="JSON", help="按顺序指定4个旧关节或 *_cartesian_solved.json 轨迹文件；第一段自动开始，后续右键长按2秒开始。")
    parser.add_argument("--right-teach-speed-scale", type=float, default=0.2)
    parser.add_argument("--right-teach-start-velocity-rad-s", type=float, default=0.05)
    parser.add_argument(
        "--teach-max-velocity-rad-s", type=float, default=1.0,
        help="末端示教回放的实际关节速度上限，默认 1.0 rad/s。",
    )
    parser.add_argument(
        "--teach-max-acceleration-rad-s2", type=float, default=15.0,
        help="末端示教回放的实际关节加速度上限，默认 15.0 rad/s²。",
    )
    parser.add_argument("--right-hole-y-mm", type=float, default=20.0, help="右手批头相对 hole 的 base_link Y 偏移，默认 20 mm。")
    parser.add_argument("--right-hole-z-mm", type=float, default=200.0, help="右手批头相对 hole 的 base_link Z 偏移，默认 200 mm。")
    parser.add_argument("--right-hole-yaw-deg", type=float, default=20.0)
    parser.add_argument("--right-hole-pitch-deg", type=float, default=0.0)
    parser.add_argument("--right-hole-roll-deg", type=float, default=180.0)
    parser.add_argument("--hole-sample-timeout-s", type=float, default=20.0)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8767, help="三相机合并页面端口；0 表示关闭。")
    parser.add_argument("--raw-preview-fps", type=float, default=5.0,
                        help="某一路没有阶段叠加图时，原画面兜底推送帧率；0 表示关闭兜底。")
    parser.add_argument("--raw-preview-hold-s", type=float, default=1.0,
                        help="阶段叠加图停推多久后恢复原画面，默认 1 秒。")
    parser.add_argument("--right-premove-pose", default="RR1")
    parser.add_argument("--right-post-grasp-pose", default="RR2")
    parser.add_argument(
        "--right-grasp-distance-mm", type=float, default=152.0,
        help="右手末端原点到电批红色抓取中心的距离，默认 152 mm。",
    )
    parser.add_argument(
        "--vlm-enable", nargs="?", const=True, default=False, type=_parse_bool,
        help="启用全流程 VLM 判别，可写 true/false；不指定时跳过 VLM。",
    )
    # Kept as a hidden compatibility switch for old launch wrappers.  The
    # public interface is now the positive --vlm-enable flag.
    parser.add_argument("--disable-vlm-checks", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--vlm-env-file", type=Path, default=PROJECT_ROOT / "vlm_checks" / "config" / "vlm.env")
    parser.add_argument("--vlm-model", default=os.environ.get("VLM_MODEL", "qwen3-vl-8b-local"))
    parser.add_argument(
        "--vlm-api-url",
        default=os.environ.get("VLM_API_URL", "http://192.168.30.12:8000/v1"),
        help="OpenAI-compatible VLM /v1 基地址或完整 /chat/completions 地址。",
    )
    parser.add_argument("--vlm-timeout-s", type=float, default=60.0)
    parser.add_argument("--vlm-gripper-crop", default=None, help="可选像素框 x0,y0,x1,y1；默认居中保留高度60%%、宽度40%%")
    parser.add_argument("--auto-speed-mm-s", type=float, default=10.0)
    parser.add_argument("--auto-preset-velocity-rad-s", type=float, default=0.08)
    parser.add_argument("--left-seat-max-descent-mm", type=float, default=10.0,
                        help="左臂从 LL1 向 base_link -Z 力保护下探的最大距离，默认 10 mm。")
    parser.add_argument("--left-seat-force-threshold-n", type=float, default=3.0,
                        help="左臂 LL1 下探相对基线三轴合力阈值，默认 3 N。")
    parser.add_argument("--preset-arrival-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--preset-timeout-s", type=float, default=25.0)
    parser.add_argument("--spacemouse-speed-mm-s", type=float, default=30.0)
    parser.add_argument("--spacemouse-rotation-deg-s", type=float, default=20.0)
    parser.add_argument("--spacemouse-deadzone", type=float, default=0.12)
    parser.add_argument("--from-stage", choices=stage_names(), default=None, help="从指定阶段开始（调试用）。")
    parser.add_argument("--through-stage", choices=stage_names(), default=None,
                        help="执行完指定阶段后正常结束（调试用，包含该阶段）。")
    parser.add_argument("--list-stages", action="store_true", help="只打印阶段计划后退出。")
    parser.add_argument("--dry-run", action="store_true", help="只读：校验姿态文件与阶段计划，不初始化 GDK。")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--allow-motion", action="store_true", help="唯一运动确认参数；不指定时拒绝启动动作。")
    return parser.parse_args(argv)


def print_stage_plan() -> None:
    for index, spec in enumerate(DEFAULT_PLAN, start=1):
        gate = "  [SpaceMouse 门]" if spec.waits_for_advance else ""
        pane = "" if spec.camera_pane is None else f"  cam={spec.camera_pane}"
        print(f"{index:2d}. {spec.stage.value:14s} step {spec.user_step:4s}{gate}{pane}\n    {spec.title}")


def teach_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "teach_enable", False))


def vlm_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "vlm_enable", False)) and not bool(
        getattr(args, "disable_vlm_checks", False)
    )


def run_dry_run(args: argparse.Namespace) -> int:
    if teach_enabled(args):
        try:
            try:
                from .right_teach_placement import validate_right_teach_args
            except ImportError:
                from right_teach_placement import validate_right_teach_args
            validate_right_teach_args(args)
        except (ValueError, OSError) as exc:
            print(f"右手示教参数错误：{exc}", file=sys.stderr)
            return 2
    if teach_enabled(args):
        try:
            try:
                from .left_teach_placement import validate_teach_args
            except ImportError:
                from left_teach_placement import validate_teach_args
            validate_teach_args(args)
        except (ValueError, OSError) as exc:
            print(f"左手示教参数错误：{exc}", file=sys.stderr)
            return 2
    runtime = BilateralRuntime(args=args, controller=None)
    report = runtime.validate_plan_assets()
    home_ok, home_error = True, None
    try:
        try:
            from .workflow_home import load_workflow_home
        except ImportError:
            from workflow_home import load_workflow_home
        load_workflow_home(Path(args.poses), args.home_pose_name)
    except Exception as exc:  # noqa: BLE001 - reported to the operator
        home_ok, home_error = False, str(exc)
    print_stage_plan()
    print(json.dumps({
        "event": "bilateral_flow_dry_run",
        "poses": str(Path(args.poses).expanduser()),
        "presets": report["presets"],
        "missing_presets": report["missing"],
        "home_pose": args.home_pose_name,
        "home_ok": home_ok,
        "home_error": home_error,
    }, ensure_ascii=False, indent=2))
    return 0 if not report["missing"] and home_ok else 1


def validate_args(args: argparse.Namespace) -> int | None:
    if teach_enabled(args):
        try:
            try:
                from .right_teach_placement import validate_right_teach_args
            except ImportError:
                from right_teach_placement import validate_right_teach_args
            validate_right_teach_args(args)
        except (ValueError, OSError) as exc:
            print(f"右手示教参数错误：{exc}", file=sys.stderr)
            return 2
    if teach_enabled(args):
        try:
            try:
                from .left_teach_placement import validate_teach_args
            except ImportError:
                from left_teach_placement import validate_teach_args
            validate_teach_args(args)
        except (ValueError, OSError) as exc:
            print(f"左手示教参数错误：{exc}", file=sys.stderr)
            return 2
    if args.vlm_enable and args.disable_vlm_checks:
        print("--vlm-enable 与旧的 --disable-vlm-checks 不能同时指定。", file=sys.stderr)
        return 2
    if args.device is None or not str(args.device).startswith("/dev/"):
        print("--device 必须是 /dev/hidrawN 路径或 udev 固定别名。", file=sys.stderr)
        return 2
    if args.from_stage is not None and args.through_stage is not None:
        order = stage_names()
        if order.index(args.through_stage) < order.index(args.from_stage):
            print("--through-stage 不能位于 --from-stage 之前。", file=sys.stderr)
            return 2
    checks = (
        math.isfinite(args.head_samples) and 1 <= args.head_samples <= 30,
        1 <= args.head_min_inlier_samples <= args.head_samples,
        math.isfinite(args.head_max_sample_spread_mm) and 0 < args.head_max_sample_spread_mm <= 200,
        math.isfinite(args.auto_speed_mm_s) and 0 < args.auto_speed_mm_s <= 50,
        math.isfinite(args.auto_preset_velocity_rad_s) and 0 < args.auto_preset_velocity_rad_s <= 0.5,
        math.isfinite(args.left_seat_max_descent_mm) and 0 < args.left_seat_max_descent_mm <= 100,
        math.isfinite(args.left_seat_force_threshold_n) and 0 < args.left_seat_force_threshold_n <= 50,
        math.isfinite(args.left_hole_x_mm) and abs(args.left_hole_x_mm) <= 500,
        math.isfinite(args.left_hole_y_mm) and abs(args.left_hole_y_mm) <= 500,
        math.isfinite(args.left_hole_z_mm) and 0 < args.left_hole_z_mm <= 500,
        math.isfinite(args.right_hole_x_mm) and abs(args.right_hole_x_mm) <= 500,
        math.isfinite(args.right_hole_y_mm) and abs(args.right_hole_y_mm) <= 500,
        math.isfinite(args.right_hole_z_mm) and 0 < args.right_hole_z_mm <= 500,
        math.isfinite(args.right_grasp_distance_mm) and 10 <= args.right_grasp_distance_mm <= 300,
        math.isfinite(args.teach_max_velocity_rad_s) and 0 < args.teach_max_velocity_rad_s <= 2.0,
        math.isfinite(args.teach_max_acceleration_rad_s2) and 0 < args.teach_max_acceleration_rad_s2 <= 30.0,
        math.isfinite(args.vlm_timeout_s) and 1 <= args.vlm_timeout_s <= 300,
        math.isfinite(args.approach_above_mm) and 0 < args.approach_above_mm <= 500,
        1.0 <= args.spacemouse_speed_mm_s <= 30.0,
        1.0 <= args.spacemouse_rotation_deg_s <= 28.6,
        0.0 <= args.spacemouse_deadzone <= 0.5,
        0.5 <= args.head_yolo_confidence <= 1.0,
        0.0 < args.yolo_center_fraction <= 1.0,
        math.isfinite(args.final_descent_visual_tolerance_mm) and args.final_descent_visual_tolerance_mm > 0.0,
        0.1 <= args.post_grasp_target_lost_timeout_s <= 30.0,
        args.relief_extra_lift_mm is None or (
            math.isfinite(args.relief_extra_lift_mm) and 0.0 <= args.relief_extra_lift_mm <= 20.0
        ),
        math.isfinite(args.web_port) and 0 <= args.web_port <= 65535,
    )
    if not all(checks):
        print("阶段参数超出安全范围。", file=sys.stderr)
        return 2
    return None


# --------------------------------------------------------------------- stages
def stage_home(ctx: FlowContext) -> dict:
    opened = ctx.runtime.open_all_grippers()
    home = ctx.runtime.move_both_to_home(
        velocity_rad_s=ctx.args.auto_preset_velocity_rad_s,
        arrival_tolerance_rad=ctx.args.preset_arrival_tolerance_rad,
        timeout_s=ctx.args.preset_timeout_s,
    )
    return {"grippers": opened["event"], "home": home["event"]}


def stage_head_localize(ctx: FlowContext) -> dict:
    try:
        from .move_left_above_head_hinge_gdk import localize_head_targets
    except ImportError:
        from move_left_above_head_hinge_gdk import localize_head_targets
    gdk = ctx.gdk
    tf = ctx.services.tf
    localization, overlay = localize_head_targets(
        camera=ctx.services.head_camera, tf=tf, model=ctx.services.head_yolo, gdk=gdk,
        color_type=gdk.CameraType.kHeadColor, depth_type=gdk.CameraType.kHeadDepth,
        depth_to_color=_transform_dict(tf, gdk.SensorExtrinsicType.kHeadDepthToHeadColor),
        head_from_color=_transform_dict(tf, gdk.SensorExtrinsicType.kHeadRGBDToHeadLink3),
        samples=ctx.args.head_samples,
        min_inlier_samples=ctx.args.head_min_inlier_samples,
        max_sample_spread_mm=ctx.args.head_max_sample_spread_mm,
        confidence=ctx.args.head_yolo_confidence,
        imgsz=640, device=ctx.args.yolo_device,
        timeout_s=ctx.args.head_localize_timeout_s,
        on_frame=_head_live_publisher(ctx),
    )
    _publish_head_overlay(ctx, overlay, localization)
    targets = require_head_targets(localization)
    ctx.services.put("head_targets", targets)
    hole_target = targets.get("hole")
    warning = hole_missing_warning(localization)
    if warning is not None:
        ctx.runtime.emit_event(warning)
        print("警告：头部没有识别到 hole；将正常抓取并上抬，然后停在上抬位置等待下一阶段指令。", flush=True)
    return {
        "hinge_position_base_link_m": targets["hinge"]["position_base_link_m"],
        "hole_position_base_link_m": None if hole_target is None else hole_target["position_base_link_m"],
        "hole_samples": 0 if hole_target is None else hole_target["samples"],
        "hole_missing": hole_target is None,
        "frames": localization["frames"],
    }


def require_head_targets(localization: dict) -> dict:
    """Return the head targets; only the hinge is mandatory.

    A missing hole is a downgrade, not a stop: the left stage logs a warning,
    grasps and lifts as usual, then holds and waits for the operator's next
    stage command instead of moving above the hole.
    """
    targets = localization.get("targets") or {}
    rejected = localization.get("rejected_targets") or {}
    if "hinge" not in targets:
        raise SafetyError(
            "头部没有定位到最左 hinge；停止后续自动流程。原因："
            + str(rejected.get("hinge", {}).get("reason", "无有效三维样本"))
        )
    return targets


def hole_missing_warning(localization: dict) -> dict | None:
    """Warning record for a head localization that found no hole, else None."""
    if (localization.get("targets") or {}).get("hole") is not None:
        return None
    reason = (localization.get("rejected_targets") or {}).get("hole", {}).get(
        "reason", "无有效三维样本",
    )
    return {
        "event": "head_hole_missing",
        "level": "warning",
        "reason": reason,
        "action": "继续抓取；上抬到位后停在原地等待 SpaceMouse 下一阶段指令，不做 hole 移动",
    }


def _make_jpeg_reader(gdk, camera, color_type, *, timeout_ms: float = 300.0):
    """Return a callable that grabs the newest frame of one camera as JPEG.

    The timeout is deliberately short: three pane feeds run side by side, so a
    single stalled camera must not hold a whole second of its own thread.
    """
    import cv2
    try:
        from .camera import decode_color_rgb
    except ImportError:
        from camera import decode_color_rgb

    def read():
        frame = camera.get_latest_image(color_type, float(timeout_ms))
        bgr = cv2.cvtColor(decode_color_rgb(frame, gdk), cv2.COLOR_RGB2BGR)
        ok, buffer = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buffer.tobytes() if ok else None

    return read


def warmup_camera(gdk, camera, color_type, label: str, *, attempts: int = 3,
                  timeout_ms: float = 1000.0) -> dict:
    """Read one frame before the panes start so the first page load is not blank.

    Returns a report instead of raising: a camera that is late is a diagnostic,
    not a reason to abort the workflow.
    """
    try:
        from .camera import decode_color_rgb
    except ImportError:
        from camera import decode_color_rgb

    last_error: BaseException | None = None
    for attempt in range(1, max(1, int(attempts)) + 1):
        try:
            frame = camera.get_latest_image(color_type, float(timeout_ms))
            image = decode_color_rgb(frame, gdk)
            return {
                "camera": label, "ok": True, "attempts": attempt,
                "shape": list(getattr(image, "shape", ())),
            }
        except BaseException as exc:  # noqa: BLE001 - reported, never raised
            last_error = exc
    return {
        "camera": label, "ok": False, "attempts": int(attempts),
        "error": str(last_error)[:200],
    }


def _head_live_publisher(ctx: "FlowContext"):
    """Publish the head hinge/hole overlay for every localization frame."""
    try:
        from .move_left_above_head_hinge_gdk import draw_head_targets_overlay
    except ImportError:
        from move_left_above_head_hinge_gdk import draw_head_targets_overlay

    def publish(bgr, masks, scores, classes) -> None:
        web = getattr(ctx.services, "web", None)
        if web is None:
            return
        import cv2

        overlay = draw_head_targets_overlay(bgr, masks, scores, classes)
        ok, buffer = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            web.pane("head").update(buffer.tobytes(), {
                "state": "head_coarse_localizing",
                "source": "head_yolo",
                "note": "正在采样最左 hinge 与最高置信度 hole",
            })

    return publish


def _transform_dict(tf, sensor_type) -> dict:
    try:
        from .camera import transform_to_dict
    except ImportError:
        from camera import transform_to_dict
    return transform_to_dict(tf.get_tf_from_sensor(sensor_type))


def _publish_head_overlay(ctx: FlowContext, overlay, localization: dict) -> None:
    """Push the head hinge/hole overlay into the shared page, if one is running."""
    web = getattr(ctx.services, "web", None)
    if web is None or overlay is None:
        return
    import cv2
    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return
    web.pane("head").update(encoded.tobytes(), {
        "state": "head_hinge_hole_localized",
        "source": "head_yolo",
        "frames": localization.get("frames"),
        "targets": localization.get("targets"),
        "safety": "只读头部定位；未发送运动。",
    })


def stage_left_grasp(ctx: FlowContext) -> dict:
    ctx.services.put('left_raw_feed', ctx.feeds.get('left'))
    monitor = ctx.services.get('vlm_gripper_monitor')
    if monitor is not None:
        # VLM is not polled before a grasp exists: the monitor stays idle until
        # the gripper actually closes, then left_grasp switches it to
        # 'post_grasp'.  This avoids burning remote VLM calls on the approach
        # and descent, where "not attached" is expected and meaningless.
        monitor.set_phase('idle')
    try:
        from .left_grasp_gdk import run_left_cycle
    except ImportError:
        from left_grasp_gdk import run_left_cycle
    # A missing hole is handled inside the left stage: warn, grasp, lift, then
    # hold and wait for the next stage command instead of moving to the hole.
    targets = ctx.services.get("head_targets") or {}
    with concise_child_stdout():
        code = run_left_cycle(services=ctx.services, argv=ctx.settings.left_argv())
    if monitor is not None:
        monitor.raise_if_regrasp_requested()
    if code != 0:
        raise SafetyError(f"左手抓取阶段返回 {code}")
    return {"returncode": code, "hole_detected": "hole" in targets}


def stage_left_to_ll1_and_seat(ctx: FlowContext) -> dict:
    monitor = ctx.services.get('vlm_gripper_monitor')
    try:
        if teach_enabled(ctx.args):
            try:
                from .left_teach_placement import replay_left_placement
            except ImportError:
                from left_teach_placement import replay_left_placement
            return replay_left_placement(ctx.services, ctx.args)
        try:
            from .left_seat_motion import force_guarded_left_descent
        except ImportError:
            from left_seat_motion import force_guarded_left_descent
        records = ctx.runtime.move_left_presets(
            ("LL1",),
            velocity_rad_s=ctx.args.auto_preset_velocity_rad_s,
            arrival_tolerance_rad=ctx.args.preset_arrival_tolerance_rad,
            timeout_s=ctx.args.preset_timeout_s,
        )
        descent = force_guarded_left_descent(
            ctx.services.controller, ctx.services.robot,
            max_distance_mm=ctx.args.left_seat_max_descent_mm,
            force_threshold_n=ctx.args.left_seat_force_threshold_n,
            speed_mm_s=ctx.args.auto_speed_mm_s,
        )
        ctx.runtime.emit_event(descent)
        return {"presets": [record["preset"] for record in records], "descent": descent}
    finally:
        if monitor is not None:
            # The hinge has been seated.  From here the left arm only releases
            # and lifts away, so a later "not attached" vote (which is expected
            # once the jaws open) must never request another pickup.  Leaving
            # the monitor in post_grasp kept polling through the gate and
            # turned a finished placement into a regrasp.
            monitor.set_phase('idle')


def stage_left_release_lift_to_ll2(ctx: FlowContext) -> dict:
    monitor = ctx.services.get('vlm_gripper_monitor')
    if monitor is not None:
        monitor.finish_before_release()
    # Read-only pre-release confirmation: the hinge has just been seated by the
    # force-guarded descent.  A remote VLM may still not see it between the
    # jaws, so this only warns; it never blocks the release and never requests
    # a regrasp (that veto caused repeated false-negative regrasps before).
    pre_release_state = _one_shot_gripper_check(ctx, phase="pre_release")
    if pre_release_state in {"not_attached", "dropped"}:
        print(
            f"[VLM 放件前] 只读告警：VLM 判定 {pre_release_state}；不触发重抓，继续按原流程放件。",
            flush=True,
        )
        ctx.runtime.emit_event({
            "event": "vlm_pre_release_warning",
            "state": pre_release_state,
            "read_only": True,
            "action": "continue_release",
            "note": "只读告警：不阻止放件、不请求重抓",
        })
    try:
        from .left_seat_motion import move_left_vertical
    except ImportError:
        from left_seat_motion import move_left_vertical
    released = ctx.runtime.open_left_gripper()
    lift = move_left_vertical(
        ctx.services.controller, distance_mm=100.0, speed_mm_s=ctx.args.auto_speed_mm_s,
    )
    ctx.runtime.emit_event(lift)
    records = ctx.runtime.move_left_presets(
        ("LL2",),
        velocity_rad_s=ctx.args.auto_preset_velocity_rad_s,
        arrival_tolerance_rad=ctx.args.preset_arrival_tolerance_rad,
        timeout_s=ctx.args.preset_timeout_s,
    )
    return {
        "release": released["event"], "lift": lift,
        "presets": [record["preset"] for record in records],
        "pre_release_vlm": pre_release_state,
    }


def stage_right_to_lr1(ctx: FlowContext) -> dict:
    record = ctx.runtime.move_right_preset(
        "LR1",
        velocity_rad_s=ctx.args.auto_preset_velocity_rad_s,
        arrival_tolerance_rad=ctx.args.preset_arrival_tolerance_rad,
        timeout_s=ctx.args.preset_timeout_s,
    )
    return {"preset": record["preset"]}


def _vlm_client(ctx: FlowContext):
    if not vlm_enabled(ctx.args):
        return None
    return ctx.services.get("vlm_client")


def _one_shot_gripper_check(ctx: FlowContext, *, phase: str) -> str:
    """Read-only left-gripper check; returns the VLM state, never commands."""
    client = _vlm_client(ctx)
    if client is None:
        return "unknown"
    try:
        from .vlm_flow_checks import run_gripper_check
    except ImportError:
        from vlm_flow_checks import run_gripper_check
    record = {**run_gripper_check(ctx.services.web, client), "workflow_phase": phase}
    ctx.runtime.emit_event(record)
    if not record.get("ok"):
        return "unknown"
    return str((record.get("result") or {}).get("state", "unknown"))


def _one_shot_assembly_check(ctx: FlowContext, *, phase: str) -> list[dict]:
    """Read-only pin + C-hole check; returns records, never commands."""
    client = _vlm_client(ctx)
    if client is None:
        return []
    try:
        from .vlm_flow_checks import run_assembly_checks
    except ImportError:
        from vlm_flow_checks import run_assembly_checks
    records = run_assembly_checks(ctx.services.web, client)
    for record in records:
        ctx.runtime.emit_event({**record, "workflow_phase": phase})
    return records


def _one_shot_right_fixture_check(ctx: FlowContext, *, phase: str) -> dict:
    """Read-only V32 right-fixture check; it never changes the motion path."""
    client = _vlm_client(ctx)
    if client is None:
        return {"task": "right-fixture-grasp", "ok": False, "skipped": True, "reason": "disabled_or_unavailable"}
    try:
        from .vlm_flow_checks import run_right_fixture_grasp_check
    except ImportError:
        from vlm_flow_checks import run_right_fixture_grasp_check
    record = {**run_right_fixture_grasp_check(ctx.services.web, client), "workflow_phase": phase}
    ctx.runtime.emit_event(record)
    return record


def _one_shot_four_screws_check(ctx: FlowContext, *, phase: str) -> dict:
    """Read-only four-screw check; it never gates the closing motion."""
    client = _vlm_client(ctx)
    if client is None:
        return {"task": "four-screws", "ok": False, "skipped": True, "reason": "disabled_or_unavailable"}
    try:
        from .vlm_flow_checks import run_four_screws_check
    except ImportError:
        from vlm_flow_checks import run_four_screws_check
    record = {**run_four_screws_check(ctx.services.web, client), "workflow_phase": phase}
    ctx.runtime.emit_event(record)
    return record


def stage_vlm_gripper_check(ctx: FlowContext) -> dict:
    if not vlm_enabled(ctx.args):
        return {"skipped": True, "reason": "disabled"}
    monitor = ctx.services.get('vlm_gripper_monitor')
    if monitor is not None:
        # Background serial requests continue through transport/placement.
        # Do not wait for a remote response or declare this stage grasp success.
        monitor.set_phase('post_grasp')
        return {'task': 'gripper', 'monitoring': True}
    try:
        from .vlm_flow_checks import run_gripper_check
    except ImportError:
        from vlm_flow_checks import run_gripper_check
    record = run_gripper_check(ctx.services.web, ctx.services.get("vlm_client"))
    ctx.runtime.emit_event(record)
    return {"task": "gripper", "ok": record["ok"]}


def stage_vlm_assembly_check(ctx: FlowContext) -> dict:
    if not vlm_enabled(ctx.args):
        return {"skipped": True, "reason": "disabled"}
    try:
        from .vlm_flow_checks import run_assembly_review
    except ImportError:
        from vlm_flow_checks import run_assembly_review
    records = run_assembly_review(
        ctx.services.web,
        ctx.services.get("vlm_client"),
        ctx.runtime.emit_event,
        include_convex=True,
    )
    return {"tasks": [record["task"] for record in records], "ok": all(record["ok"] for record in records)}


def stage_right_grasp(ctx: FlowContext) -> dict:
    try:
        from .right_apriltag_grasp_gdk import run_right_grasp_stage
    except ImportError:
        from right_apriltag_grasp_gdk import run_right_grasp_stage
    with concise_child_stdout():
        code = run_right_grasp_stage(services=ctx.services, argv=ctx.settings.right_argv())
    # The child no longer reads/publishes camera frames after returning.
    # Restore the raw feed now, not after the subsequent transport and replay.
    feed = ctx.feeds.get("right")
    if feed is not None:
        feed.resume()
    if code != 0:
        raise SafetyError(f"右手抓取阶段返回 {code}")
    fixture_vlm = _one_shot_right_fixture_check(ctx, phase="right_grasp")
    if ctx.services.get("right_saved_grasp_pose") is None:
        raise SafetyError("右手抓取阶段没有保存抓取位姿；无法在门2后回位")
    workpiece = ctx.services.get("right_saved_workpiece_pose")
    if workpiece is None:
        raise SafetyError("右手抓取阶段没有保存电批工件位姿；无法定位批头轴线")
    hole = (ctx.services.get("head_targets") or {}).get("hole")
    if hole is None:
        raise SafetyError("头部没有可用的 hole 位置；停止右手运输")
    try:
        from .right_hole_motion import move_right_tool_above_hole
    except ImportError:
        from right_hole_motion import move_right_tool_above_hole
    transport = move_right_tool_above_hole(
        ctx.services.controller,
        ctx.services.get("right_saved_grasp_pose"),
        workpiece,
        hole["position_base_link_m"],
        x_mm=ctx.args.right_hole_x_mm,
        y_mm=ctx.args.right_hole_y_mm,
        z_mm=ctx.args.right_hole_z_mm,
        yaw_deg=ctx.args.right_hole_yaw_deg,
        pitch_deg=ctx.args.right_hole_pitch_deg,
        roll_deg=ctx.args.right_hole_roll_deg,
        speed_mm_s=ctx.args.auto_speed_mm_s,
    )
    ctx.runtime.emit_event(transport)
    result = {
        "returncode": code,
        "right_fixture_vlm": fixture_vlm,
        "transport": transport["event"],
    }
    if teach_enabled(ctx.args):
        try:
            from .right_teach_placement import run_right_teach_sequence
        except ImportError:
            from right_teach_placement import run_right_teach_sequence
        print("右手已到 hole 偏移点；开始示教回放。", flush=True)
        result["placement"] = run_right_teach_sequence(ctx.services, ctx.args, ctx.runtime)
    return result


def stage_right_return(ctx: FlowContext) -> dict:
    try:
        from .right_apriltag_grasp_gdk import return_screwdriver
        from .right_apriltag_grasp_gdk import require_mapping
    except ImportError:
        from right_apriltag_grasp_gdk import return_screwdriver, require_mapping
    saved = ctx.services.get("right_saved_grasp_pose")
    if saved is None:
        raise SafetyError("没有可用的右手抓取位姿；拒绝回位")
    config = ctx.services.get("right_config")
    if config is None:
        config = require_mapping(
            json.loads(Path(ctx.args.right_config).expanduser().read_text(encoding="utf-8")), "config",
        )
    events: list[dict] = []
    if teach_enabled(ctx.args):
        try:
            from .right_hole_motion import lift_right_before_return
        except ImportError:
            from right_hole_motion import lift_right_before_return
        print("右手归位前先上抬 30 mm，随后检查四颗螺丝…", flush=True)
        ctx.runtime.emit_event(lift_right_before_return(
            ctx.services.controller, speed_mm_s=ctx.args.auto_speed_mm_s, distance_mm=30.0,
        ))
    four_screws = _one_shot_four_screws_check(ctx, phase="before_right_return")
    return_screwdriver(
        ctx.services.controller, saved, config, events.append, release=False,
    )
    return {
        "four_screws": four_screws,
        "events": [event.get("event") for event in events],
        "released": False,
    }


def stage_finish(ctx: FlowContext) -> dict:
    opened = ctx.runtime.open_all_grippers()
    controller = ctx.services.controller
    if controller is not None:
        # The electric screwdriver has been released; leave the right-arm
        # post-grasp position-control phase so later motion is impedance again.
        controller.set_position_control_phase(False)
    home = ctx.runtime.move_both_to_home(
        velocity_rad_s=ctx.args.auto_preset_velocity_rad_s,
        arrival_tolerance_rad=ctx.args.preset_arrival_tolerance_rad,
        timeout_s=ctx.args.preset_timeout_s,
    )
    return {"grippers": opened["event"], "home": home["event"]}


STAGE_HANDLERS = {
    Stage.HOME: stage_home,
    Stage.HEAD_LOCALIZE: stage_head_localize,
    Stage.LEFT_GRASP: stage_left_grasp,
    Stage.VLM_GRIPPER_CHECK: stage_vlm_gripper_check,
    Stage.RIGHT_TO_LR1: stage_right_to_lr1,
    Stage.LEFT_TO_LL1_AND_SEAT: stage_left_to_ll1_and_seat,
    Stage.LEFT_RELEASE_LIFT_TO_LL2: stage_left_release_lift_to_ll2,
    Stage.VLM_ASSEMBLY_CHECK: stage_vlm_assembly_check,
    Stage.RIGHT_GRASP: stage_right_grasp,
    Stage.RIGHT_RETURN: stage_right_return,
    Stage.FINISH: stage_finish,
}


def run_flow(args: argparse.Namespace) -> int:
    try:
        from .vlm_flow_checks import start_vlm_run
    except ImportError:
        from vlm_flow_checks import start_vlm_run
    start_vlm_run()
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    from spacemouse_teleop import TeleopConfig
    from spacemouse_takeover import MotionPauseGate, SpaceMouseTakeoverWorker
    from dianpi_runner import DianpiCycleRunner

    try:
        from .control import G2Controller
        from .yolo_hinge import load_yolo_segmentation_model
        from .right_apriltag_grasp_gdk import detector_for_41h12
    except ImportError:
        from control import G2Controller
        from yolo_hinge import load_yolo_segmentation_model
        from right_apriltag_grasp_gdk import detector_for_41h12

    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败", file=sys.stderr)
        return 2
    settings = StageSettings(args=args)
    log_path = (
        Path(args.log).expanduser() if args.log is not None
        else PROJECT_ROOT / "logs" / f"bilateral_flow_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    )
    args.log = log_path
    cameras: list = []
    worker = None
    keyboard = None
    dianpi = None
    web: BilateralWeb | None = None
    runtime: BilateralRuntime | None = None
    models: PreloadedModels | None = None
    services: StageServices | None = None
    ctx: FlowContext | None = None
    feeds: dict = {}
    try:
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        gate = MotionPauseGate()
        controller = G2Controller(agibot_gdk, robot, allow_motion=True, motion_gate=gate)
        controller.checked_arm_state()
        controller.prepare_joint_impedance_control()
        def flow_event_sink(event: dict) -> None:
            record = {"utc": datetime.now(timezone.utc).isoformat(), **event}
            append_jsonl_record(log_path, record)
            summary = console_event_summary(record)
            if summary is not None:
                print(summary, flush=True)

        dianpi = DianpiCycleRunner(event_sink=flow_event_sink)
        dianpi.start_p10_monitor()
        worker = SpaceMouseTakeoverWorker(
            controller, gate,
            device=args.device,
            config=TeleopConfig(
                side="left",
                speed_m_s=args.spacemouse_speed_mm_s / 1000.0,
                angular_speed_rad_s=math.radians(args.spacemouse_rotation_deg_s),
                deadzone=args.spacemouse_deadzone,
            ),
            allow_advance_right=False,
            dianpi_runner=dianpi,
            event_sink=flow_event_sink,
        )
        worker.start()
        keyboard = KeyboardStageAdvance(worker)
        keyboard.start()

        head_camera = agibot_gdk.Camera([agibot_gdk.CameraType.kHeadColor, agibot_gdk.CameraType.kHeadDepth])
        left_camera = agibot_gdk.Camera([agibot_gdk.CameraType.kHandLeftColor, agibot_gdk.CameraType.kHandLeftDepth])
        right_camera = agibot_gdk.Camera([agibot_gdk.CameraType.kHandRightColor])
        cameras = [head_camera, left_camera, right_camera]
        tf = agibot_gdk.TF()

        flow_event_sink({"event": "bilateral_models_preload_start"})
        models = PreloadedModels(
            head_yolo=load_yolo_segmentation_model(Path(args.head_yolo_model), device=args.yolo_device),
            left_yolo=load_yolo_segmentation_model(Path(args.left_yolo_model), device=args.yolo_device),
            apriltag_detector=detector_for_41h12(),
        )
        flow_event_sink({"event": "bilateral_models_preloaded", "sam3": "not_loaded"})

        if args.web_port:
            web = BilateralWeb(host=args.web_host, port=args.web_port)
            url = web.start()
            print(f"三相机合并页面：{url}（HEAD / LEFT / RIGHT，可勾选隐藏）", flush=True)
        if web is not None and args.raw_preview_fps > 0:
            # Warm each stream once before the feeds start: GDK needs a moment
            # after Camera() before the first colour frame is available, and a
            # pane that never gets its first frame stays blank on the page.
            camera_sources = (
                ("head", head_camera, agibot_gdk.CameraType.kHeadColor),
                ("left", left_camera, agibot_gdk.CameraType.kHandLeftColor),
                ("right", right_camera, agibot_gdk.CameraType.kHandRightColor),
            )
            for label, pane_camera, color_type in camera_sources:
                report = warmup_camera(agibot_gdk, pane_camera, color_type, label)
                flow_event_sink({"event": "camera_warmup", **report})
                if not report["ok"]:
                    print(
                        f"警告：{label} 相机预热失败，页面该格会显示错误信息并持续重试：{report['error']}",
                        flush=True,
                    )
            # Every pane stays live: the stages push annotated frames while they
            # own a camera, and the raw feed covers the rest of the run.
            for pane_name, pane_camera, color_type in camera_sources:
                feed = RawPaneFeed(
                    pane_name,
                    _make_jpeg_reader(agibot_gdk, pane_camera, color_type),
                    web,
                    hold_s=args.raw_preview_hold_s,
                    fps=args.raw_preview_fps,
                )
                feeds[pane_name] = feed
                feed.start()
            print(
                f"原画面兜底：三路相机常显，{args.raw_preview_fps:g} fps；"
                f"阶段叠加图停推 {args.raw_preview_hold_s:g} 秒后自动切回原画面。",
                flush=True,
            )

        services = StageServices(
            gdk=agibot_gdk, robot=robot, controller=controller, gate=gate, worker=worker,
            dianpi_runner=dianpi, tf=tf,
            head_yolo=models.head_yolo, left_yolo=models.left_yolo, right_detector=models.apriltag_detector,
            head_camera=head_camera, left_camera=left_camera, right_camera=right_camera,
            web=web,
        )
        if vlm_enabled(args):
            try:
                from vlm_checks.src.vlm_three_checks import VlmClient
                services.put("vlm_client", VlmClient(
                    args.vlm_env_file, args.vlm_model, args.vlm_api_url, args.vlm_timeout_s,
                ))
            except Exception as exc:
                services.put("vlm_client", None)
                flow_event_sink({"event": "vlm_check_warning", "task": "init", "error": str(exc)})
        services.extra["right_color_type"] = agibot_gdk.CameraType.kHandRightColor
        runtime = BilateralRuntime(
            args=args, controller=controller, gate=gate, worker=worker,
            models=models, journal_path=log_path,
        )
        ctx = FlowContext(
            gdk=agibot_gdk, services=services, runtime=runtime, settings=settings, feeds=feeds,
        )
        if vlm_enabled(args) and services.get('vlm_client') is not None:
            try:
                from .vlm_gripper_monitor import GripperMonitor
                from .vlm_flow_checks import parse_crop_xyxy
            except ImportError:
                from vlm_gripper_monitor import GripperMonitor
                from vlm_flow_checks import parse_crop_xyxy
            monitor = GripperMonitor(web, services.get('vlm_client'), flow_event_sink,
                                     parse_crop_xyxy(args.vlm_gripper_crop) if args.vlm_gripper_crop else None)
            services.put('vlm_gripper_monitor', monitor)
            controller.automatic_guard = monitor.raise_if_regrasp_requested
            monitor.start()
        return _drive(ctx, args)
    finally:
        if services is not None:
            monitor = services.get('vlm_gripper_monitor')
            if monitor is not None:
                monitor.stop()
        # Deterministic teardown: stop every producer thread first, then drop
        # the vision assets, and only then release the GDK session.  Closing
        # cameras or releasing GDK while a worker is still ticking has been
        # observed to abort inside the native binding.
        for feed in feeds.values():
            try:
                feed.stop()
            except Exception:
                pass
        if keyboard is not None:
            keyboard.stop()
        if worker is not None:
            try:
                worker.stop()
            except Exception:
                pass
        if dianpi is not None:
            try:
                # The P10 monitor can be inside a slow native import/open when
                # the run ends; give it longer than the 1 s default before
                # reporting that it did not stop.
                dianpi.close(timeout_s=3.0)
            except Exception:
                pass
        if web is not None:
            try:
                web.stop()
            except Exception:
                pass
        for camera in cameras:
            try:
                camera.close_camera()
            except Exception:
                pass
        models = services = runtime = ctx = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        time.sleep(0.2)
        try:
            agibot_gdk.gdk_release()
        except Exception:
            pass


def _drive(ctx: FlowContext, args: argparse.Namespace) -> int:
    try:
        from .vlm_gripper_monitor import VlmRegraspRequested
    except ImportError:
        from vlm_gripper_monitor import VlmRegraspRequested
    if args.from_stage is not None:
        order = stage_names()
        if order.index(args.from_stage) > order.index(Stage.HEAD_LOCALIZE.value):
            ctx.runtime.emit(
                "stage_prerequisite_start",
                stage=Stage.HEAD_LOCALIZE.value,
                reason=f"--from-stage {args.from_stage} 仍依赖本次运行的 hole 定位",
            )
            print("[前置定位] 正在重新定位 head hinge / hole…", flush=True)
            feed = ctx.feeds.get("head")
            if feed is not None:
                feed.pause()
            try:
                payload = STAGE_HANDLERS[Stage.HEAD_LOCALIZE](ctx)
            except Exception as exc:  # noqa: BLE001 - prerequisite failure stops all later motion
                ctx.runtime.emit("stage_failed", stage=Stage.HEAD_LOCALIZE.value, error=str(exc))
                print(f"前置阶段 {Stage.HEAD_LOCALIZE.value} 失败：{exc}", file=sys.stderr)
                return 1
            finally:
                if feed is not None:
                    feed.resume()
            ctx.runtime.emit(
                "stage_prerequisite_complete",
                stage=Stage.HEAD_LOCALIZE.value,
                for_stage=args.from_stage,
                **(payload or {}),
            )
            print("[前置定位完成] 已取得本次运行的 hole 位置。", flush=True)
    sequencer = StageSequencer(start_at=args.from_stage)
    while not sequencer.is_done:
        spec = sequencer.current
        ctx.runtime.emit(
            "stage_start", stage=spec.stage.value, user_step=spec.user_step, title=spec.title,
        )
        # A stage that uses a camera owns that pane and pushes its own overlay;
        # pause the raw fallback for it so the two never read the same camera.
        feed = ctx.feeds.get(spec.camera_pane) if spec.camera_pane else None
        if feed is not None and spec.stage is not Stage.LEFT_GRASP:
            feed.pause()
        try:
            if spec.waits_for_advance:
                ctx.runtime.wait_for_advance(spec)
                payload = {"gate": spec.stage.value}
            else:
                payload = STAGE_HANDLERS[spec.stage](ctx)
        except VlmRegraspRequested as exc:
            monitor = ctx.services.get('vlm_gripper_monitor')
            if monitor is None or not (ctx.services.get('head_targets') or {}).get('hinge'):
                ctx.runtime.emit('stage_failed', stage=spec.stage.value, error='VLM 重抓缺少本次头部铰链定位')
                return 1
            monitor.set_phase('idle')
            ctx.runtime.emit('vlm_regrasp_requested', stage=spec.stage.value, reason=str(exc))
            print('[VLM 重抓] 停止当前动作，先竖直上抬 50 mm，再返回铰链预接近点重新夹取。', flush=True)
            try:
                controller = ctx.services.controller
                # All left stage streams have unwound/stopped before here.
                # Replace the previous destination with measured joints;
                # do not open the gripper while returning to the tray.
                current = list(controller.checked_arm_state()[:7])
                for _ in range(20):
                    controller.hold_joint_positions("left", current)
                    time.sleep(.01)
                try:
                    from .left_seat_motion import move_left_vertical
                except ImportError:
                    from left_seat_motion import move_left_vertical
                lift = move_left_vertical(
                    controller, distance_mm=50.0, speed_mm_s=ctx.args.auto_speed_mm_s,
                )
                ctx.runtime.emit_event(lift)
            except Exception as recovery_error:
                ctx.runtime.emit('stage_failed', stage=spec.stage.value, error=f'VLM 重抓停止或上抬失败：{recovery_error}')
                print(f'VLM 重抓停止或上抬失败：{recovery_error}', file=sys.stderr)
                return 1
            sequencer.skip_to(Stage.LEFT_GRASP.value)
            continue
        except Exception as exc:  # noqa: BLE001 - reported and stops the sequence
            sequencer.fail_current(str(exc))
            ctx.runtime.emit("stage_failed", stage=spec.stage.value, error=str(exc))
            print(f"阶段 {spec.stage.value} 失败：{exc}", file=sys.stderr)
            return 1
        finally:
            if feed is not None:
                feed.resume()
        sequencer.complete_current(payload)
        stage_fields = dict(payload or {})
        stage_fields["stage"] = spec.stage.value
        ctx.runtime.emit("stage_complete", **stage_fields)
        if args.through_stage == spec.stage.value:
            ctx.runtime.emit(
                "bilateral_flow_stopped_at_stage",
                stage=spec.stage.value,
                completed=sequencer.results,
            )
            return 0
    ctx.runtime.emit("bilateral_flow_complete", **sequencer.summary())
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.list_stages:
        print_stage_plan()
        return 0
    if args.dry_run:
        return run_dry_run(args)
    if not args.allow_motion:
        print("分阶段流程必须显式指定 --allow-motion；否则不会启动动作。", file=sys.stderr)
        return 2
    invalid = validate_args(args)
    if invalid is not None:
        return invalid
    return run_flow(args)


if __name__ == "__main__":
    raise SystemExit(main())
