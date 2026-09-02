#!/usr/bin/env python3
"""Final-position left D405 visual grasp using only GDK and SAM3.

By default P2/P3 remain outside this program, so it can be used as a
final-detection-position-only grasp tool. ``--full-cycle`` makes this the
single GDK/SAM3 process for P2→P3→grasp→+Z lift→P4: SAM3 is loaded before
P2 is requested and GDK is initialized exactly once. Without ``--allow-motion``
it is still a perception preview: it never changes a GDK control mode or sends
an arm/Omnipicker request. With that one flag it refuses an uncalibrated
configuration file and stops sending new commands on every bad frame/error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

try:
    from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from .contact_detection import ForceContactDetector, ForceContactParameters
    from .control import CartesianPoseTarget, G2Controller, LeftEndEffectorPositionStream
    from .left_grasp_state import GraspParameters, GraspPhase, LeftGraspState, Observation, requires_calibrated_config
    from .live_left_d405_sam3_web import SharedView, handler_class
    from .localize_left_d405_sam3 import draw_overlay, eligible_candidates, parse_roi, stats_for_mask
    from .safety import SafetyError
    from .sam_crop import center_crop_bounds, center_crop_image, draw_crop_box, expand_crop_masks
    from .visual_servo import base_position_after_delta, empirical_horizontal_base_delta_from_pixel_error
    from .xy_mapping import choose_candidate_by_left_rank
    from .yolo_hinge import associate_shaft_to_hinge, infer_yolo_masks, load_yolo_segmentation_model, review_matches, split_yolo_candidates
    from .record_pose import DEFAULT_OUTPUT as DEFAULT_POSES_OUTPUT
    from .run_left_premove_sequence import load_presets as load_premove_presets
    from .run_left_post_grasp_sequence import (
        PRE_P4_LIFT_M,
        PRE_P4_LIFT_TOLERANCE_M,
        load_presets as load_post_grasp_presets,
    )
except ImportError:
    from camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from contact_detection import ForceContactDetector, ForceContactParameters
    from control import CartesianPoseTarget, G2Controller, LeftEndEffectorPositionStream
    from left_grasp_state import GraspParameters, GraspPhase, LeftGraspState, Observation, requires_calibrated_config
    from live_left_d405_sam3_web import SharedView, handler_class
    from localize_left_d405_sam3 import draw_overlay, eligible_candidates, parse_roi, stats_for_mask
    from safety import SafetyError
    from sam_crop import center_crop_bounds, center_crop_image, draw_crop_box, expand_crop_masks
    from visual_servo import base_position_after_delta, empirical_horizontal_base_delta_from_pixel_error
    from xy_mapping import choose_candidate_by_left_rank
    from yolo_hinge import associate_shaft_to_hinge, infer_yolo_masks, load_yolo_segmentation_model, review_matches, split_yolo_candidates
    from record_pose import DEFAULT_OUTPUT as DEFAULT_POSES_OUTPUT
    from run_left_premove_sequence import load_presets as load_premove_presets
    from run_left_post_grasp_sequence import (
        PRE_P4_LIFT_M,
        PRE_P4_LIFT_TOLERANCE_M,
        load_presets as load_post_grasp_presets,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_SCALE_M = 0.001
# GDK's kLeftHandRGBDToArmLEndLink external calibration terminates here.  GDK
# mode-1 end-effector pose control uses the same flange link by default.
LEFT_VISUAL_TIP_LINK = "arm_l_end_link"
# SAM3 inference itself sets the cadence (about a few Hz on the current GPU).
# Do not add a material artificial delay between display-only descent frames.
FORCE_DESCENT_SAM_PREVIEW_PERIOD_S = 0.01
_PREFERRED_G2_HINGE_SHAFT_YOLO = PROJECT_ROOT / "models" / "left_d405_hinge_shaft_yolo" / "best.pt"
_ROLLBACK_G2_HINGE_YOLO = PROJECT_ROOT / "models" / "left_d405_hinge_yolo" / "best.pt"
# Prefer the newly trained two-class model when it is present, while retaining
# the known one-class weight as a safe checkout/rollback fallback.
DEFAULT_G2_NATIVE_YOLO = (
    _PREFERRED_G2_HINGE_SHAFT_YOLO
    if _PREFERRED_G2_HINGE_SHAFT_YOLO.is_file()
    else _ROLLBACK_G2_HINGE_YOLO
)
HINGE_CLASS_ID = 0
SHAFT_CLASS_ID = 1


@dataclass(frozen=True)
class AdaptiveGripperCloseParameters:
    """Per-grasp position-only Omnipicker closure policy."""

    seek_target_rad: float
    minimum_close_travel_rad: float
    stall_window_s: float
    stall_position_delta_rad: float
    additional_close_rad: float

    def validate(self) -> None:
        lower, upper = -math.pi / 4.0, 0.0
        values = (
            self.seek_target_rad,
            self.minimum_close_travel_rad,
            self.stall_window_s,
            self.stall_position_delta_rad,
            self.additional_close_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("adaptive_close 包含非有限数值")
        if not lower <= self.seek_target_rad <= upper:
            raise ValueError("adaptive_close.seek_target_rad 必须在 Omnipicker 行程内")
        if not 0.02 <= self.minimum_close_travel_rad <= 0.50:
            raise ValueError("adaptive_close.minimum_close_travel_rad 必须在 [0.02, 0.50]")
        if not 0.20 <= self.stall_window_s <= 2.0:
            raise ValueError("adaptive_close.stall_window_s 必须在 [0.20, 2.0]")
        if not 0.001 <= self.stall_position_delta_rad <= 0.03:
            raise ValueError("adaptive_close.stall_position_delta_rad 必须在 [0.001, 0.03]")
        if not 0.0 <= self.additional_close_rad <= 0.10:
            raise ValueError("adaptive_close.additional_close_rad 必须在 [0, 0.10]")


class FinalDescentVisualMailbox:
    """Thread-safe latest-only visual correction handoff.

    The camera/SAM worker never sends GDK commands.  It can only publish the
    newest correction for the force loop to accept or reject.  Keeping only
    one item avoids a delayed inference result building up a stale motion
    queue during descent.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._latest: dict | None = None

    def publish(self, item: dict) -> None:
        with self._lock:
            self._sequence += 1
            self._latest = {**item, "sequence": self._sequence, "published_monotonic_s": time.monotonic()}

    def newer_than(self, sequence: int) -> dict | None:
        with self._lock:
            if self._latest is None or self._sequence <= sequence:
                return None
            return dict(self._latest)


def held_part_monitor_failure(
    observation: dict,
    *,
    reference_center_px: list[float],
    reference_area_px: int,
    max_center_drift_px: float,
    max_area_change_fraction: float,
) -> dict | None:
    """Return a transport-veto record when the locked held part is unhealthy.

    This is deliberately a visual *continuity* check, not a new detection or
    a request to substitute another hinge.  During transport the only valid
    object is the one that was verified immediately after the grasp.
    """
    # ``valid=None`` means that this is a fresh frame in which the locked
    # instance is temporarily missing.  It is evidence to count, but not
    # evidence that the part has already been dropped.
    if observation.get("valid") is None:
        return None
    if observation.get("valid") is not True:
        return {
            "reason": observation.get("reason", "locked_target_lost"),
            "observation": observation,
        }
    try:
        center_drift_px = float(np.linalg.norm(
            np.asarray(observation["center_px"], dtype=np.float64)
            - np.asarray(reference_center_px, dtype=np.float64)
        ))
        area_change_fraction = abs(
            float(observation["area_px"]) / float(reference_area_px) - 1.0
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        return {
            "reason": "invalid_held_target_monitor_observation",
            "error": str(exc),
            "observation": observation,
        }
    if not math.isfinite(center_drift_px) or not math.isfinite(area_change_fraction):
        return {
            "reason": "non_finite_held_target_monitor_observation",
            "observation": observation,
        }
    if center_drift_px > max_center_drift_px or area_change_fraction > max_area_change_fraction:
        return {
            "reason": "held_target_geometry_changed",
            "center_drift_px": center_drift_px,
            "area_change_fraction": area_change_fraction,
            "observation": observation,
        }
    return None


@dataclass
class NonBlockingLeftPresetSequence:
    """Send one recorded joint target at a time and poll it from the main loop.

    Unlike the legacy helper this class never sleeps or waits for an arrival.
    It lets the web/SAM worker and safety monitor continue while P2/P3/P4
    targets are in flight.
    """

    presets: list[tuple[str, list[float]]]
    velocity_rad_s: float
    arrival_tolerance_rad: float
    timeout_s: float
    stage: str
    index: int = 0
    sent_at: float | None = None

    def tick(self, controller: G2Controller, log) -> tuple[bool, dict | None]:
        now = time.monotonic()
        if self.index >= len(self.presets):
            return True, None
        name, target = self.presets[self.index]
        if self.sent_at is None:
            controller.move_preset(
                "left", target, velocity_rad_s=self.velocity_rad_s,
                life_time_s=self.timeout_s,
                detail=f"traditional nonblocking {self.stage} {name}",
            )
            self.sent_at = now
            event = {
                "event": "preset_request_sent",
                "stage": self.stage,
                "preset": name,
                "utc": datetime.now(timezone.utc).isoformat(),
                "target_rad": target,
                "nonblocking": True,
            }
            json_record(log, event)
            return False, event
        measured = controller.checked_arm_state()[:7]
        max_error = max(abs(actual - expected) for actual, expected in zip(measured, target))
        if max_error <= self.arrival_tolerance_rad:
            event = {
                "event": "preset_arrived",
                "stage": self.stage,
                "preset": name,
                "utc": datetime.now(timezone.utc).isoformat(),
                "max_joint_error_rad": max_error,
                "nonblocking": True,
            }
            json_record(log, event)
            self.index += 1
            self.sent_at = None
            return self.index >= len(self.presets), event
        if now - self.sent_at > self.timeout_s:
            raise SafetyError(
                f"{self.stage} {name} 在 {self.timeout_s:.1f} s 内未到位；停止后续命令，夹爪状态保持原样"
            )
        return False, None


def stalled_gripper_position(
    samples: list[tuple[float, float]],
    *,
    now: float,
    window_s: float,
    position_delta_rad: float,
) -> float | None:
    """Return the latest position after one complete stable feedback window.

    SAM inference limits this loop to roughly 4 Hz.  Do not require a sample
    to land *exactly* on ``now - window_s``: retain the last sample immediately
    before that boundary as the window baseline.  Otherwise a rolling history
    can continually discard the only sample that proves the full interval,
    even while the physical gripper has been motionless for several seconds.
    """
    if len(samples) < 2:
        return None

    ordered = sorted(
        (timestamp, position)
        for timestamp, position in samples
        if timestamp <= now
    )
    cutoff = now - window_s
    baseline_index = next(
        (index for index in range(len(ordered) - 1, -1, -1) if ordered[index][0] <= cutoff),
        None,
    )
    if baseline_index is None:
        return None
    window = ordered[baseline_index:]
    if len(window) < 2:
        return None
    positions = [position for _, position in window]
    if max(positions) - min(positions) > position_delta_rad:
        return None
    return positions[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2 左手最终检测位 D405/SAM3 闭环抓取")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "left_grasp.example.json")
    parser.add_argument("--prompt", default="silver triangular hinge")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--detector",
        choices=("yolo-sam", "sam3"),
        default="yolo-sam",
        help="默认使用 G2 原生 YOLO 主检测并由 SAM3 复核；sam3 是只用于回退对比的旧路径。",
    )
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_G2_NATIVE_YOLO)
    parser.add_argument("--yolo-confidence", type=float, default=0.25)
    parser.add_argument("--yolo-target-confidence", type=float, default=0.25)
    parser.add_argument(
        "--shaft-association-max-distance-px",
        type=float,
        default=320.0,
        help="选定 hinge 附近允许关联 shaft 的最大像素距离；超出或不明确时只保留 hinge 抓取目标。",
    )
    parser.add_argument(
        "--shaft-association-max-horizontal-distance-px",
        type=float,
        default=140.0,
        help="关联 shaft 与 hinge 时允许的最大水平像素差，避免串到相邻铰链。",
    )
    parser.add_argument(
        "--shaft-association-ambiguity-margin-px",
        type=float,
        default=20.0,
        help="最近两个 shaft 的距离差小于该值时拒绝关联。",
    )
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-device", default="cuda:0")
    parser.add_argument("--sam-review-interval-s", type=float, default=0.50)
    parser.add_argument("--sam-review-ttl-s", type=float, default=1.50)
    parser.add_argument("--sam-review-max-pixel-error", type=float, default=80.0)
    parser.add_argument("--sam-review-max-depth-error-m", type=float, default=0.05)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=50)
    parser.add_argument("--roi", default="none", metavar="X0,Y0,X1,Y1")
    parser.add_argument("--selection", choices=("largest", "leftmost", "second-leftmost", "rightmost"), default="leftmost")
    parser.add_argument("--track-max-distance-px", type=float, default=120.0)
    parser.add_argument(
        "--track-max-depth-jump-m",
        type=float,
        default=0.06,
        help="仅记录跟踪期间相对首次锁定深度的异常跳变阈值；不参与运动决策，默认 60 mm。",
    )
    parser.add_argument("--track-hold-frames", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示持续运行。")
    parser.add_argument("--single-xy-step", action="store_true", help="等待有效目标；最多只发送一次 XY 纠偏后立即停止。")
    parser.add_argument("--checkpoint", default="checkpoints/sam3.1_multiplex.pt")
    parser.add_argument("--control-period-s", type=float, default=0.02)
    parser.add_argument(
        "--pre-final-settle-timeout-s",
        type=float,
        default=5.0,
        help="最终下探前等待最后一个 XY 末端设定点完成的最长时间；默认 5 秒，只等待不发送下探。",
    )
    parser.add_argument(
        "--final-descent-visual-servo",
        action="store_true",
        help="下探期间对已锁定的同一铰链执行受限 XY 复位；力接触仍然优先并会冻结纠偏。",
    )
    parser.add_argument(
        "--final-descent-visual-max-step-m",
        type=float,
        default=0.001,
        help="下探视觉复位每次最多的水平位移，默认 1 mm。",
    )
    parser.add_argument(
        "--final-descent-visual-max-total-m",
        type=float,
        default=0.0,
        help="下探视觉复位相对下探起点的累计水平位移上限；0 表示不设累计上限（默认）。",
    )
    parser.add_argument(
        "--final-descent-target-lost-frames",
        type=int,
        default=2,
        help="下探期间同一锁定目标连续丢失多少个 SAM3 帧后撤回并重新对准，默认 2。",
    )
    parser.add_argument(
        "--regrasp-retract-m",
        type=float,
        default=0.0,
        help="目标丢失或空夹时，重新张爪/对准前的 base_link +Z 撤回量，默认 0 mm（保持当前位置）。",
    )
    parser.add_argument(
        "--post-grasp-verify-lift-m",
        type=float,
        default=0.020,
        help="闭爪后用于视觉验夹的 base_link +Z 小幅上抬，默认 20 mm。",
    )
    parser.add_argument(
        "--post-grasp-verify-frames",
        type=int,
        default=3,
        help="验夹连续比较帧数，默认 3。",
    )
    parser.add_argument(
        "--post-grasp-verify-center-drift-px",
        type=float,
        default=12.0,
        help="验夹中同一实例允许的像素中心漂移，默认 12 px。",
    )
    parser.add_argument(
        "--post-grasp-verify-area-change-fraction",
        type=float,
        default=0.10,
        help="验夹中同一实例允许的相对掩膜面积变化，默认 10%%。",
    )
    parser.add_argument(
        "--post-grasp-verify-depth-change-m",
        type=float,
        default=0.015,
        help="验夹中同一实例允许的相对深度变化，默认 15 mm；用于抵抗夹爪遮挡造成的掩膜面积变化。",
    )
    parser.add_argument(
        "--post-grasp-monitor-max-stale-s",
        type=float,
        default=2.0,
        help="抓取后的运输监控最长允许多久没有新的同一目标观测，默认 2 秒；超时只暂停并保持闭爪，明确丢失才重夹。",
    )
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/left_grasp_*.jsonl")
    parser.add_argument("--web-host", default="127.0.0.1", help="内嵌只读网页监听地址。")
    parser.add_argument("--web-port", type=int, default=0, help="内嵌 SAM3 网页端口；0 表示关闭。")
    parser.add_argument("--stop-after", choices=("none", "xy", "descend"), default="none", help="受看护测试的停止阶段；descend 会完成配置的定时下探但不闭爪。")
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help="单进程执行 P2→P3、SAM3 抓取、+Z 上抬 100 mm、P4；要求 --allow-motion，且不能与受看护停止选项合用。",
    )
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES_OUTPUT, help="包含左 P2/P3/P4 的 recorded_poses.json。")
    parser.add_argument("--preset-velocity-rad-s", type=float, default=0.08, help="全流程 P2/P3/P4 的规划关节速度。")
    parser.add_argument(
        "--speed-mm-s",
        type=float,
        default=None,
        help="整体机器人运动速度（mm/s）；当前配置默认 10 mm/s，可设置 1～18 mm/s。",
    )
    parser.add_argument("--speed-scale", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--preset-arrival-tolerance-rad", type=float, default=0.03, help="全流程预设点到位关节容差。")
    parser.add_argument("--preset-timeout-s", type=float, default=25.0, help="全流程每个预设点的到位超时。")
    parser.add_argument("--allow-motion", action="store_true", help="唯一运动确认参数；未指定时严格只读。")
    return parser.parse_args()


def load_config(path: Path, *, require_calibrated: bool) -> tuple[dict, GraspParameters | None, float | None, float | None, float | None, np.ndarray | None, ForceContactParameters | None, AdaptiveGripperCloseParameters | None]:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise RuntimeError("抓取配置 schema_version 必须为 1")
    alignment = document.get("alignment", {})
    approach = document.get("approach", {})
    gripper = document.get("gripper", {})
    contact = document.get("contact", {})
    adaptive_close = document.get("adaptive_close", {})
    if not isinstance(adaptive_close, dict):
        adaptive_close = {}
    required = {
        "alignment.target_pixel_px": alignment.get("target_pixel_px"),
        "alignment.pixel_jacobian_px_per_m": alignment.get("pixel_jacobian_px_per_m"),
        "approach.pre_descent_depth_m": approach.get("pre_descent_depth_m"),
        "approach.pre_descent_confirmations": approach.get("pre_descent_confirmations"),
        "approach.pre_descent_step_m": approach.get("pre_descent_step_m"),
        "approach.pre_descent_duration_s": approach.get("pre_descent_duration_s"),
        "approach.final_descent_speed_mps": approach.get("final_descent_speed_mps"),
        "approach.final_descent_duration_s": approach.get("final_descent_duration_s"),
        "gripper.open_rad": gripper.get("open_rad"),
        "adaptive_close.seek_target_rad": adaptive_close.get("seek_target_rad"),
        "adaptive_close.minimum_close_travel_rad": adaptive_close.get("minimum_close_travel_rad"),
        "adaptive_close.stall_window_s": adaptive_close.get("stall_window_s"),
        "adaptive_close.stall_position_delta_rad": adaptive_close.get("stall_position_delta_rad"),
        "adaptive_close.additional_close_rad": adaptive_close.get("additional_close_rad"),
    }
    incomplete = any(value is None for value in required.values())
    if require_calibrated and (not document.get("calibrated") or incomplete):
        missing = [name for name, value in required.items() if value is None]
        suffix = ", ".join(missing) if missing else "calibrated=true"
        raise RuntimeError(f"拒绝真机抓取：配置未完成实测标定（{suffix}）。")
    # Preview mode deliberately permits an incomplete file.  It still checks
    # camera/SAM/target tracking, but does not pretend that unknown hardware
    # thresholds describe a valid descent or grasp plan.
    if incomplete:
        return document, None, None, None, None, None, None, None
    target_pixel = alignment.get("target_pixel_px")
    if not isinstance(target_pixel, list) or len(target_pixel) != 2:
        raise RuntimeError("alignment.target_pixel_px 必须为 [u, v]")
    if alignment.get("mapping_controller") != "gdk_mode1_end_effector_pose_control":
        raise RuntimeError("alignment.mapping_controller 必须是 gdk_mode1_end_effector_pose_control；旧关节伺服标定禁止用于末端位姿控制")
    empirical_pixel_jacobian = np.asarray(alignment["pixel_jacobian_px_per_m"], dtype=np.float64)
    if empirical_pixel_jacobian.shape != (2, 2) or not np.all(np.isfinite(empirical_pixel_jacobian)):
        raise RuntimeError("alignment.pixel_jacobian_px_per_m 必须为有限 2x2 矩阵")
    if np.linalg.matrix_rank(empirical_pixel_jacobian) < 2:
        raise RuntimeError("alignment.pixel_jacobian_px_per_m 不可逆")
    parameters = GraspParameters(
        pixel_tolerance=float(alignment.get("pixel_tolerance_px", 12.0)),
        center_confirmations=int(alignment.get("center_confirmations", 3)),
        max_xy_step_m=float(alignment.get("max_xy_step_m", 0.002)),
        target_u_px=float(target_pixel[0]),
        target_v_px=float(target_pixel[1]),
        pre_descent_depth_m=float(approach["pre_descent_depth_m"]),
        pre_descent_confirmations=int(approach["pre_descent_confirmations"]),
        pre_descent_step_m=float(approach["pre_descent_step_m"]),
        pre_descent_duration_s=float(approach["pre_descent_duration_s"]),
        final_descent_speed_mps=float(approach["final_descent_speed_mps"]),
        final_descent_duration_s=float(approach["final_descent_duration_s"]),
        final_descent_completion_tolerance_m=float(approach.get("final_descent_completion_tolerance_m", 0.0005)),
    )
    values = (*parameters.__dict__.values(), float(gripper["open_rad"]), float(gripper.get("close_tolerance_rad", 0.03)), float(gripper.get("close_timeout_s", 3.0)))
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("抓取配置包含非有限数值")
    if not (parameters.pixel_tolerance > 0 and parameters.center_confirmations > 0 and parameters.max_xy_step_m > 0 and parameters.pre_descent_depth_m > 0 and parameters.pre_descent_confirmations > 0 and parameters.pre_descent_step_m > 0 and parameters.pre_descent_duration_s > 0 and parameters.final_descent_speed_mps > 0 and parameters.final_descent_duration_s > 0 and parameters.final_descent_completion_tolerance_m > 0):
        raise RuntimeError("抓取配置中的距离、时间和速度必须为正数")
    contact_parameters = None
    if contact:
        try:
            contact_parameters = ForceContactParameters(
                frame_name=str(contact["frame_name"]),
                force_delta_threshold_n=float(contact["force_delta_threshold_n"]),
                consecutive_samples=int(contact["consecutive_samples"]),
                baseline_samples=int(contact["baseline_samples"]),
                sample_period_s=float(contact["sample_period_s"]),
                require_for_grasp=bool(contact.get("require_for_grasp", True)),
                relief_max_lift_m=float(contact["relief_max_lift_m"]),
                relief_force_delta_threshold_n=float(contact["relief_force_delta_threshold_n"]),
                relief_consecutive_samples=int(contact["relief_consecutive_samples"]),
            )
            contact_parameters.validate()
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"contact 配置无效：{exc}") from exc
    if require_calibrated and (contact_parameters is None or not contact_parameters.require_for_grasp):
        raise RuntimeError("拒绝真机抓取：contact 必须启用 require_for_grasp，并提供已验证的左手力阈值。")
    try:
        adaptive_parameters = AdaptiveGripperCloseParameters(
            seek_target_rad=float(adaptive_close["seek_target_rad"]),
            minimum_close_travel_rad=float(adaptive_close["minimum_close_travel_rad"]),
            stall_window_s=float(adaptive_close["stall_window_s"]),
            stall_position_delta_rad=float(adaptive_close["stall_position_delta_rad"]),
            additional_close_rad=float(adaptive_close["additional_close_rad"]),
        )
        adaptive_parameters.validate()
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"adaptive_close 配置无效：{exc}") from exc
    return document, parameters, float(gripper["open_rad"]), float(gripper.get("close_tolerance_rad", 0.03)), float(gripper.get("close_timeout_s", 3.0)), empirical_pixel_jacobian, contact_parameters, adaptive_parameters


def choose_candidate(statistics: list[dict], candidates: list[int], selection: str) -> int | None:
    if not candidates:
        return None
    if selection == "largest":
        return max(candidates, key=lambda index: statistics[index]["area_px"])
    if selection == "leftmost":
        return min(candidates, key=lambda index: statistics[index]["center_px"][0])
    if selection == "second-leftmost":
        return choose_candidate_by_left_rank(statistics, candidates, rank=2)
    return max(candidates, key=lambda index: statistics[index]["center_px"][0])


def intersect_roi(
    first: tuple[int, int, int, int] | None,
    second: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    """Return an intersection ROI, preserving the legacy SAM 70% safety area."""
    if first is None:
        return second
    x0, y0 = max(first[0], second[0]), max(first[1], second[1])
    x1, y1 = min(first[2], second[2]), min(first[3], second[3])
    return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)


def sam_review_for_yolo_target(
    processor: Sam3Processor,
    color_rgb: np.ndarray,
    aligned_depth: np.ndarray,
    *,
    color_intrinsic: dict,
    end_link_transform: dict,
    prompt: str,
    min_area: int,
    min_valid_depth_pixels: int,
    roi: tuple[int, int, int, int] | None,
    reference_center_px: list[float],
) -> tuple[list[dict], int | None]:
    """Return the nearest valid SAM3 instance to the selected YOLO instance.

    SAM3 is a review gate only in the YOLO path.  It never substitutes a
    different leftmost instance, which is essential when the tray contains
    several visually similar hinges.
    """
    sam_input, crop_bounds = center_crop_image(color_rgb)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_state = processor.set_image(Image.fromarray(sam_input))
        result = processor.set_text_prompt(prompt, image_state)
    crop_masks = [np.asarray(mask, dtype=bool).squeeze() for mask in result["masks"].detach().cpu().numpy()]
    masks = expand_crop_masks(crop_masks, full_shape=color_rgb.shape[:2], bounds=crop_bounds)
    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
    statistics = [stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform) for mask in masks]
    for item, score in zip(statistics, scores):
        item["sam3_confidence"] = float(score)
    candidates = eligible_candidates(
        statistics,
        min_area=min_area,
        min_valid_depth_pixels=min_valid_depth_pixels,
        roi=roi,
        require_depth=False,
    )
    reference = np.asarray(reference_center_px, dtype=np.float64)
    selected = min(
        candidates,
        key=lambda index: float(np.linalg.norm(np.asarray(statistics[index]["center_px"], dtype=np.float64) - reference)),
        default=None,
    )
    return statistics, selected


def measured_left_end_link_pose(controller: G2Controller) -> CartesianPoseTarget:
    """Read the commanded flange pose from GDK motion-status frame_poses."""
    return controller.read_end_effector_pose(LEFT_VISUAL_TIP_LINK)


def read_wrench_force(robot, frame_name: str) -> tuple[object, np.ndarray]:
    """Read one finite GDK wrench force, paired by the documented frame order."""
    status = robot.get_motion_control_status()
    if int(status.error_code) != 0:
        raise SafetyError(f"读取接触力失败：GDK motion error {int(status.error_code)}: {status.error_msg}")
    frame_names = [str(value) for value in status.frame_names]
    try:
        index = frame_names.index(frame_name)
    except ValueError as exc:
        raise SafetyError(f"GDK 状态未提供接触力参考帧 {frame_name}") from exc
    if index >= len(status.wrenches):
        raise SafetyError(f"GDK 状态未提供 {frame_name} 对应的 wrench")
    wrench = status.wrenches[index]
    force = np.asarray((float(wrench.force.x), float(wrench.force.y), float(wrench.force.z)), dtype=np.float64)
    if force.shape != (3,) or not np.all(np.isfinite(force)):
        raise SafetyError(f"{frame_name} wrench force 含无效数值")
    return status, force


def capture_force_baseline(robot, parameters: ForceContactParameters) -> np.ndarray:
    """Capture a stationary pre-contact baseline without moving or re-zeroing G2."""
    samples: list[np.ndarray] = []
    deadline = time.monotonic()
    for _ in range(parameters.baseline_samples):
        _status, force = read_wrench_force(robot, parameters.frame_name)
        samples.append(force)
        deadline += parameters.sample_period_s
        delay = deadline - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
    baseline = np.mean(np.asarray(samples, dtype=np.float64), axis=0)
    if baseline.shape != (3,) or not np.all(np.isfinite(baseline)):
        raise SafetyError("接触力基线无效")
    return baseline


def compact_record(record: dict) -> dict:
    """Keep only the operationally useful fields of a high-rate frame record."""
    summary = {
        key: record[key]
        for key in (
            "event", "frame", "utc", "state", "reason", "error",
            "command_source", "motion_command_sent", "regrasp_attempt", "test_stop",
            "post_grasp_verify", "perception_gate",
        )
        if key in record
    }
    selected = record.get("selected")
    if isinstance(selected, dict):
        summary["target"] = {
            key: selected[key]
            for key in ("center_px", "depth_median_m", "yolo_confidence")
            if key in selected
        }
    force = record.get("force_contact")
    if isinstance(force, dict) and "maximum_delta_force_n" in force:
        summary["force_delta_n"] = round(float(force["maximum_delta_force_n"]), 3)
    visual = record.get("final_descent_visual_servo")
    if isinstance(visual, dict):
        summary["visual_xy"] = {
            "enabled": visual.get("enabled"),
            "accepted_total_base_delta_m": visual.get("accepted_total_base_delta_m"),
        }
    return summary


def json_record(stream, payload: dict) -> None:
    # Frame records arrive at camera/SAM rate and formerly duplicated large
    # masks, candidate lists and model-review data into JSONL. Keep the full
    # payload only for discrete lifecycle events; the frame timeline uses the
    # same compact summary shown in the terminal.
    stored = compact_record(payload) if payload.get("event") == "frame" else payload
    stream.write(json.dumps(stored, ensure_ascii=False) + "\n")
    stream.flush()


def print_compact_record(record: dict, *, file=None) -> None:
    """Print the same compact summary that is written for each frame."""
    summary = compact_record(record)
    print(json.dumps(summary, ensure_ascii=False), file=file, flush=True)


def run_recorded_left_presets(
    controller: G2Controller,
    presets: list[tuple[str, list[float]]],
    *,
    velocity_rad_s: float,
    arrival_tolerance_rad: float,
    timeout_s: float,
    stage: str,
    log,
) -> None:
    """Execute already-validated recorded left-arm poses using the shared GDK robot.

    The helper intentionally owns neither GDK initialization nor a gripper
    command. Failure to arrive at any point raises before the next request,
    preserving the one-process full-cycle safety boundary.
    """
    for name, target in presets:
        controller.move_preset(
            "left",
            target,
            velocity_rad_s=velocity_rad_s,
            life_time_s=timeout_s,
            detail=f"traditional full-cycle {stage} {name}",
        )
        json_record(log, {
            "event": "preset_request_sent",
            "stage": stage,
            "preset": name,
            "utc": datetime.now(timezone.utc).isoformat(),
            "target_rad": target,
        })
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            measured = controller.checked_arm_state()[:7]
            max_error = max(abs(actual - expected) for actual, expected in zip(measured, target))
            if max_error <= arrival_tolerance_rad:
                json_record(log, {
                    "event": "preset_arrived",
                    "stage": stage,
                    "preset": name,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "max_joint_error_rad": max_error,
                })
                print(f"{stage} {name} 已到位，最大关节误差 {max_error:.4f} rad。")
                break
            time.sleep(0.10)
        else:
            raise SafetyError(
                f"{stage} {name} 在 {timeout_s:.1f} s 内未到位；停止后续命令，夹爪状态保持原样"
            )


def open_left_gripper_with_feedback(
    controller: G2Controller,
    *,
    target_rad: float,
    tolerance_rad: float,
    timeout_s: float,
) -> dict:
    """Request the open target until GDK reports the real Omnipicker angle.

    ``move_end_effector_joint`` is non-blocking. A successful call only means
    GDK accepted a request, not that the jaw has physically opened, so this
    function intentionally gates every later arm/vision action on end-state
    feedback.
    """
    if tolerance_rad <= 0.0 or timeout_s <= 0.0:
        raise ValueError("gripper feedback tolerance and timeout must be positive")
    before = controller.read_gripper_position("left")
    attempts = 0
    deadline = time.monotonic() + timeout_s
    latest: float | None = before
    while True:
        controller.move_gripper("left", target_rad)
        attempts += 1
        time.sleep(0.10)
        latest = controller.read_gripper_position("left")
        if latest is not None and abs(latest - target_rad) <= tolerance_rad:
            return {
                "event": "gripper_open_confirmed",
                "target_rad": target_rad,
                "before_position_rad": before,
                "actual_position_rad": latest,
                "tolerance_rad": tolerance_rad,
                "attempts": attempts,
            }
        if time.monotonic() >= deadline:
            raise SafetyError(
                "夹爪张开反馈超时；未进入视觉伺服、下探或后续预设动作："
                f"target={target_rad:.6f}rad, actual={latest}, attempts={attempts}"
            )


def recover_for_visual_regrasp(
    controller: G2Controller,
    position_stream: LeftEndEffectorPositionStream,
    *,
    retract_m: float,
    translation_speed_mps: float,
    fixed_orientation_xyzw: tuple[float, float, float, float],
    gripper_open_rad: float,
    gripper_tolerance_rad: float,
    gripper_timeout_s: float,
) -> tuple[LeftEndEffectorPositionStream, CartesianPoseTarget, dict]:
    """Cancel descent, optionally retract, open, then reacquire tracking.

    Recovery first replaces any queued descent by a measured-pose hold. With
    the default ``retract_m=0`` it keeps the camera at that pose so a nearby
    temporarily lost hinge remains visible; a positive value is an
    operator-selected base ``+Z`` retreat. The tracker resets after open
    feedback is confirmed.
    """
    position_stream.raise_if_failed()
    current = measured_left_end_link_pose(controller)
    # GDK status pose quaternions vary at floating-point precision between
    # reads.  The live stream deliberately rejects any orientation change, so
    # recovery must retain the stream's fixed visual-servo orientation rather
    # than reuse the newly measured quaternion.
    hold = CartesianPoseTarget(current.position_m, fixed_orientation_xyzw)
    position_stream.update_target(hold)
    retract_target = CartesianPoseTarget(
        base_position_after_delta(hold.position_m, np.asarray((0.0, 0.0, retract_m))),
        hold.orientation_xyzw,
    )
    position_stream.update_target(retract_target)
    timeout_s = max(2.0, retract_m / translation_speed_mps + 1.0)
    if not position_stream.wait_until_target(timeout_s):
        raise SafetyError(
            "重新抓取撤回目标未在安全时间内发布完成；停止且不再尝试重新对准："
            f"retract={retract_m:.3f}m, waited={timeout_s:.1f}s"
        )
    position_stream.stop()
    # GDK's direct Omnipicker command is the same confirmed-open operation
    # used before the first visual-servo pass.  Stop the pose stream first so
    # its old close-joint request cannot race the open request.
    open_event = open_left_gripper_with_feedback(
        controller,
        target_rad=gripper_open_rad,
        tolerance_rad=gripper_tolerance_rad,
        timeout_s=gripper_timeout_s,
    )
    resumed_pose = measured_left_end_link_pose(controller)
    resumed_stream = controller.start_left_end_effector_position_stream(
        CartesianPoseTarget(resumed_pose.position_m, fixed_orientation_xyzw),
        rate_hz=50.0,
        life_time_s=0.10,
        max_translation_velocity_mps=translation_speed_mps,
    )
    return resumed_stream, CartesianPoseTarget(resumed_pose.position_m, fixed_orientation_xyzw), {
        "hold_base_link": list(hold.position_m),
        "retract_target_base_link": list(retract_target.position_m),
        "retract_m": retract_m,
        "stream_target_reached": True,
        "gripper_open": open_event,
        "resumed_base_link": list(resumed_pose.position_m),
    }


def publish_web_preview(
    shared: SharedView | None,
    *,
    sequence: int,
    color_rgb: np.ndarray,
    masks: list[np.ndarray],
    statistics: list[dict],
    selected: int | None,
    record: dict,
) -> int:
    """Publish the exact masks selected by this grasp process, if requested."""
    if shared is None:
        return sequence
    crop_bounds = center_crop_bounds(color_rgb.shape)
    overlay = draw_overlay(cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR), masks, statistics, selected)
    overlay = draw_crop_box(overlay, crop_bounds)
    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("无法编码内嵌网页叠加图")
    sequence += 1
    shared.update(encoded.tobytes(), {
        "state": record.get("state", "detecting"),
        "sequence": sequence,
        "frame": record["frame"],
        "selected_mask_index": selected,
        "selected": record.get("selected"),
        "shaft": record.get("shaft"),
        "shaft_association": record.get("shaft_association"),
        "tracking": record.get("tracking"),
        "detector": record.get("detector", "sam3"),
        "sam_center_crop_xyxy": list(crop_bounds),
        "sam_center_crop_fraction": 0.70,
        "grasp_reason": record.get("reason"),
        "motion_enabled": record["motion_enabled"],
        "safety": "网页仅显示本抓取进程的同一帧检测结果；不提供机器人或夹爪控制接口。",
    })
    return sequence


def compact_detection(statistics: list[dict], index: int | None) -> dict | None:
    """Keep JSONL shaft audit records small and JSON-serializable."""
    if index is None or index < 0 or index >= len(statistics):
        return None
    item = statistics[index]
    return {
        "mask_index": int(index),
        "center_px": item.get("center_px"),
        "bbox_xyxy": item.get("bbox_xyxy"),
        "yolo_confidence": item.get("yolo_confidence"),
        "yolo_class_id": item.get("yolo_class_id"),
        "yolo_class_name": item.get("yolo_class_name"),
        "valid_depth_px": item.get("valid_depth_px"),
        "depth_median_m": item.get("depth_median_m"),
    }


class PersistentPerception:
    """The sole owner of the hand camera and the YOLO/SAM3 model instances.

    Motion code must never call GDK Camera or either model.  It waits for a
    snapshot from this object instead.  YOLO runs for every acquired frame;
    SAM3 runs at the configured review cadence (or every frame in SAM3-only
    mode).  The same loop is also the only publisher for the web preview.
    """

    def __init__(self, *, camera, color_type, depth_type, gdk, tf, processor, yolo_model,
                 args: argparse.Namespace, shared: SharedView | None, yolo_model_path: Path):
        self.camera, self.color_type, self.depth_type = camera, color_type, depth_type
        self.gdk, self.tf, self.processor, self.yolo_model = gdk, tf, processor, yolo_model
        self.args, self.shared, self.yolo_model_path = args, shared, yolo_model_path
        self._condition = threading.Condition()
        self._latest: dict | None = None
        self._sequence = 0
        self._phase = {"state": "perception_starting", "reason": ""}
        self._stop = threading.Event()
        self._failure: Exception | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="left-persistent-perception", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def set_phase(self, state: str, reason: str = "") -> None:
        with self._condition:
            self._phase = {"state": state, "reason": reason}

    def newer_than(self, sequence: int, timeout_s: float = 1.5) -> dict | None:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._latest is None or int(self._latest["sequence"]) <= sequence:
                if self._failure is not None:
                    raise SafetyError(f"常驻感知线程失败：{self._failure}")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)
            return self._latest

    def _run(self) -> None:
        color_intrinsic = depth_intrinsic = depth_to_color = end_link_transform = None
        last_sam_review_at = float("-inf")
        sam_review: dict = {"accepted": False, "fresh": False, "reason": "awaiting_first_sam_review"}
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                color = self.camera.get_latest_image(self.color_type, 1000.0)
                depth = self.camera.get_nearest_image(self.depth_type, int(color.timestamp_ns), 1000.0)
                color_rgb = decode_color_rgb(color, self.gdk)
                depth_u16 = decode_depth_u16(depth, self.gdk)
                if color_intrinsic is None:
                    color_intrinsic = intrinsic_to_dict(self.camera.get_camera_intrinsic(self.color_type), width=int(color.width), height=int(color.height))
                    depth_intrinsic = intrinsic_to_dict(self.camera.get_camera_intrinsic(self.depth_type), width=int(depth.width), height=int(depth.height))
                    depth_to_color = transform_to_dict(self.tf.get_tf_from_sensor(self.gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor))
                    end_link_transform = transform_to_dict(self.tf.get_tf_from_sensor(self.gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink))
                aligned_depth = align_depth_to_color(depth_u16, depth_intrinsic=depth_intrinsic, color_intrinsic=color_intrinsic, depth_to_color=depth_to_color, depth_scale_m=DEPTH_SCALE_M)
                crop_bounds = center_crop_bounds(color_rgb.shape)
                roi = intersect_roi(parse_roi(self.args.roi, color_rgb.shape[:2]), crop_bounds)
                if self.args.detector == "sam3":
                    sam_input, _ = center_crop_image(color_rgb)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        image_state = self.processor.set_image(Image.fromarray(sam_input))
                        result = self.processor.set_text_prompt(self.args.prompt, image_state)
                    masks = expand_crop_masks([np.asarray(mask, dtype=bool).squeeze() for mask in result["masks"].detach().cpu().numpy()], full_shape=color_rgb.shape[:2], bounds=crop_bounds)
                    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
                    statistics = [stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform) for mask in masks]
                    for item, score in zip(statistics, scores): item["sam3_confidence"] = float(score)
                    candidates = eligible_candidates(statistics, min_area=self.args.min_area, min_valid_depth_pixels=self.args.min_valid_depth_pixels, roi=roi, require_depth=False)
                    sam_review = {"accepted": True, "fresh": True, "reason": "sam3_primary"}
                else:
                    masks, scores, class_ids = infer_yolo_masks(
                        self.yolo_model,
                        cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR),
                        confidence=self.args.yolo_confidence,
                        imgsz=self.args.yolo_imgsz,
                        device=self.args.yolo_device,
                    )
                    statistics = [
                        stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform)
                        for mask in masks
                    ]
                    names = getattr(self.yolo_model, "names", {})
                    for item, score, class_id in zip(statistics, scores, class_ids):
                        class_id = int(class_id)
                        item["yolo_confidence"] = float(score)
                        item["yolo_class_id"] = class_id
                        item["yolo_class_name"] = names.get(class_id, str(class_id)) if isinstance(names, dict) else str(class_id)
                    geometry = eligible_candidates(
                        statistics,
                        min_area=self.args.min_area,
                        min_valid_depth_pixels=self.args.min_valid_depth_pixels,
                        roi=roi,
                        require_depth=False,
                    )
                    # Only a class-0 hinge may enter the grasp state machine.
                    # Shaft detections are published separately for future
                    # axis/assembly use and can never become the grasp target.
                    hinge_candidates, shaft_candidates = split_yolo_candidates(
                        statistics,
                        geometry,
                        confidence_threshold=self.args.yolo_target_confidence,
                        hinge_class_id=HINGE_CLASS_ID,
                        shaft_class_id=SHAFT_CLASS_ID,
                    )
                    candidates = hinge_candidates
                    provisional = choose_candidate(statistics, candidates, self.args.selection)
                    now = time.monotonic()
                    with self._condition:
                        transport_phase = self._phase.get("state") in {
                            "post_grasp_monitored_lift",
                            "post_grasp_p4",
                        }
                    # Transport monitoring consumes YOLO continuity frames;
                    # do not let a synchronous periodic SAM3 review starve
                    # that mailbox and turn a healthy hold into a stale alarm.
                    if (
                        provisional is not None
                        and not transport_phase
                        and now - last_sam_review_at >= self.args.sam_review_interval_s
                    ):
                        sam_statistics, sam_index = sam_review_for_yolo_target(self.processor, color_rgb, aligned_depth, color_intrinsic=color_intrinsic, end_link_transform=end_link_transform, prompt=self.args.prompt, min_area=self.args.min_area, min_valid_depth_pixels=self.args.min_valid_depth_pixels, roi=roi, reference_center_px=statistics[provisional]["center_px"])
                        accepted, detail = review_matches(statistics[provisional], None if sam_index is None else sam_statistics[sam_index], max_pixel_error=self.args.sam_review_max_pixel_error, max_depth_error_m=self.args.sam_review_max_depth_error_m, allow_missing_depth=True)
                        sam_review = {**detail, "accepted": accepted, "timestamp_monotonic": now, "yolo_center_px": statistics[provisional]["center_px"], "sam_selected": None if sam_index is None else sam_statistics[sam_index]}
                        last_sam_review_at = now
                    age = None if "timestamp_monotonic" not in sam_review else now - float(sam_review["timestamp_monotonic"])
                    sam_review = {**sam_review, "fresh": age is not None and age <= self.args.sam_review_ttl_s, "age_s": None if age is None else round(max(age, 0.0), 3)}
                selected = choose_candidate(statistics, candidates, self.args.selection)
                shaft_association = associate_shaft_to_hinge(
                    statistics,
                    selected,
                    max_center_distance_px=self.args.shaft_association_max_distance_px,
                    max_horizontal_distance_px=self.args.shaft_association_max_horizontal_distance_px,
                    ambiguity_margin_px=self.args.shaft_association_ambiguity_margin_px,
                    shaft_indices=shaft_candidates if self.args.detector == "yolo-sam" else None,
                ) if self.args.detector == "yolo-sam" else {"shaft_index": None, "reason": "sam3_mode"}
                with self._condition:
                    self._sequence += 1
                    snapshot = {"sequence": self._sequence, "color_rgb": color_rgb, "masks": masks, "statistics": statistics, "candidates": candidates, "hinge_candidates": candidates, "shaft_candidates": shaft_candidates if self.args.detector == "yolo-sam" else [], "selected": selected, "shaft_association": shaft_association, "aligned_depth": aligned_depth, "color_intrinsic": color_intrinsic, "depth_intrinsic": depth_intrinsic, "depth_to_color": depth_to_color, "end_link_transform": end_link_transform, "roi": roi, "crop_bounds": crop_bounds, "sam_review": dict(sam_review), "inference_ms": (time.monotonic() - started) * 1000.0}
                    self._latest = snapshot
                    phase = dict(self._phase)
                    self._condition.notify_all()
                if self.shared is not None:
                    overlay = draw_crop_box(draw_overlay(cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR), masks, statistics, selected), crop_bounds)
                    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        self.shared.update(encoded.tobytes(), {"state": phase["state"], "sequence": self._sequence, "selected_mask_index": selected, "selected": None if selected is None else statistics[selected], "shaft_association": shaft_association, "detector": self.args.detector, "inference_ms": round(snapshot["inference_ms"], 1), "grasp_reason": phase["reason"], "safety": "常驻感知线程独占相机、YOLO/SAM3 与网页更新；动作状态机仅消费最新快照。"})
        except Exception as exc:
            with self._condition:
                self._failure = exc
                self._condition.notify_all()


class SnapshotVisualMailbox:
    """Same-target observation adapter backed by persistent perception."""

    def __init__(self, perception: PersistentPerception, *, initial_center_px, track_max_distance_px,
                 target_lost_frames, target_pixel_px=None, pixel_jacobian=None,
                 pixel_tolerance_px=0.0, max_xy_step_m=0.0):
        self.perception = perception
        self.center = np.asarray(initial_center_px, dtype=np.float64)
        self.track_max_distance_px = track_max_distance_px
        self.target_lost_frames = target_lost_frames
        self.target_pixel_px, self.pixel_jacobian = target_pixel_px, pixel_jacobian
        self.pixel_tolerance_px, self.max_xy_step_m = pixel_tolerance_px, max_xy_step_m
        self.lost_frames = 0

    def newer_than(self, sequence: int) -> dict | None:
        snapshot = self.perception.newer_than(sequence, timeout_s=0.0)
        if snapshot is None:
            return None
        nearby = [index for index in snapshot["candidates"] if float(np.linalg.norm(np.asarray(snapshot["statistics"][index]["center_px"], dtype=np.float64) - self.center)) <= self.track_max_distance_px]
        selected = min(nearby, key=lambda index: float(np.linalg.norm(np.asarray(snapshot["statistics"][index]["center_px"], dtype=np.float64) - self.center)), default=None)
        if selected is None:
            self.lost_frames += 1
            return {
                "sequence": snapshot["sequence"],
                "valid": False if self.lost_frames >= self.target_lost_frames else None,
                "reason": (
                    "locked_target_lost"
                    if self.lost_frames >= self.target_lost_frames
                    else "locked_target_missing_pending"
                ),
                "lost_frames": self.lost_frames,
                "last_center_px": self.center.tolist(),
            }
        self.lost_frames = 0
        item = snapshot["statistics"][selected]
        self.center = np.asarray(item["center_px"], dtype=np.float64)
        result = {"sequence": snapshot["sequence"], "valid": True, "selected_mask_index": selected, "center_px": self.center.tolist(), "area_px": int(item["area_px"]), "sam3_confidence": item.get("sam3_confidence"), "yolo_confidence": item.get("yolo_confidence")}
        if self.target_pixel_px is not None and self.pixel_jacobian is not None:
            error = self.center - np.asarray(self.target_pixel_px, dtype=np.float64)
            result["pixel_error_px"] = error.tolist()
            if float(np.linalg.norm(error)) > self.pixel_tolerance_px:
                result["delta_base_m"] = empirical_horizontal_base_delta_from_pixel_error(error, self.pixel_jacobian, max_step_m=self.max_xy_step_m).tolist()
        return result


def start_force_descent_sam_preview(
    shared: SharedView | None,
    *,
    camera,
    color_type,
    depth_type,
    gdk,
    processor: Sam3Processor,
    prompt: str,
    color_intrinsic: dict,
    depth_intrinsic: dict,
    depth_to_color: dict,
    end_link_transform: dict,
    selection: str,
    min_area: int,
    min_valid_depth_pixels: int,
    roi: tuple[int, int, int, int] | None,
    phase: dict[str, str],
    mailbox: FinalDescentVisualMailbox | None = None,
    target_pixel_px: tuple[float, float] | None = None,
    pixel_jacobian: np.ndarray | None = None,
    pixel_tolerance_px: float = 0.0,
    max_xy_step_m: float = 0.0,
    initial_center_px: tuple[float, float] | None = None,
    track_max_distance_px: float = 0.0,
    target_lost_frames: int = 2,
):
    """Run the already-loaded SAM3 model during force-guarded descent.

    The force loop remains in the caller at 50 Hz.  This worker owns the
    single preloaded SAM3 processor while final descent is in progress.  When
    a mailbox is supplied it may publish a bounded *horizontal* suggestion
    for the one target that was locked before descent; only the force loop can
    accept it, and it never changes vertical descent or gripper decisions.
    """
    if shared is None and mailbox is None:
        return None, None
    stop = threading.Event()

    def run() -> None:
        sequence = time.monotonic_ns()
        tracked_center = None if initial_center_px is None else np.asarray(initial_center_px, dtype=np.float64)
        lost_frames = 0
        while not stop.is_set():
            try:
                started = time.monotonic()
                image = camera.get_latest_image(color_type, 1000.0)
                depth = camera.get_nearest_image(depth_type, int(image.timestamp_ns), 1000.0)
                rgb = decode_color_rgb(image, gdk)
                depth_u16 = decode_depth_u16(depth, gdk)
                aligned_depth = align_depth_to_color(
                    depth_u16,
                    depth_intrinsic=depth_intrinsic,
                    color_intrinsic=color_intrinsic,
                    depth_to_color=depth_to_color,
                    depth_scale_m=DEPTH_SCALE_M,
                )
                sam_input, crop_bounds = center_crop_image(rgb)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    image_state = processor.set_image(Image.fromarray(sam_input))
                    result = processor.set_text_prompt(prompt, image_state)
                crop_masks = [np.asarray(mask, dtype=bool).squeeze() for mask in result["masks"].detach().cpu().numpy()]
                masks = expand_crop_masks(crop_masks, full_shape=rgb.shape[:2], bounds=crop_bounds)
                scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
                statistics = [
                    stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform)
                    for mask in masks
                ]
                for item, score in zip(statistics, scores):
                    item["sam3_confidence"] = float(score)
                candidates = eligible_candidates(
                    statistics,
                    min_area=min_area,
                    min_valid_depth_pixels=min_valid_depth_pixels,
                    roi=roi,
                    require_depth=True,
                )
                # Never use left/right ranking once descent has begun: the
                # tracked item is the target locked by the preceding
                # YOLO+SAM gate.  A missed frame merely yields no correction;
                # it must not make the robot jump to a neighbour hinge.
                if tracked_center is not None:
                    nearby = [
                        index for index in candidates
                        if float(np.linalg.norm(np.asarray(statistics[index]["center_px"], dtype=np.float64) - tracked_center))
                        <= track_max_distance_px
                    ]
                    selected = min(
                        nearby,
                        key=lambda index: float(np.linalg.norm(
                            np.asarray(statistics[index]["center_px"], dtype=np.float64) - tracked_center
                        )),
                        default=None,
                    )
                else:
                    selected = choose_candidate(statistics, candidates, selection)
                if selected is not None:
                    lost_frames = 0
                    selected_center = np.asarray(statistics[selected]["center_px"], dtype=np.float64)
                    tracked_center = selected_center
                    if mailbox is not None:
                        # A mailbox is also used as a held-part monitor during
                        # the post-grasp lift.  Publish every same-target
                        # observation, not only those that need an XY update.
                        # The consumer remains the only place that may decide
                        # whether to stop/regrasp; this worker never commands
                        # GDK or the gripper.
                        observation = {
                            "valid": True,
                            "selected_mask_index": int(selected),
                            "center_px": [float(value) for value in selected_center],
                            "area_px": int(statistics[selected]["area_px"]),
                            "sam3_confidence": statistics[selected].get("sam3_confidence"),
                        }
                        if target_pixel_px is not None and pixel_jacobian is not None:
                            pixel_error = selected_center - np.asarray(target_pixel_px, dtype=np.float64)
                            observation["pixel_error_px"] = [float(value) for value in pixel_error]
                            if float(np.linalg.norm(pixel_error)) > pixel_tolerance_px:
                                delta_base = empirical_horizontal_base_delta_from_pixel_error(
                                    pixel_error,
                                    pixel_jacobian,
                                    max_step_m=max_xy_step_m,
                                )
                                observation["delta_base_m"] = [float(value) for value in delta_base]
                        mailbox.publish(observation)
                elif tracked_center is not None:
                    lost_frames += 1
                    if mailbox is not None and lost_frames >= target_lost_frames:
                        mailbox.publish({
                            "valid": False,
                            "reason": "locked_target_lost",
                            "lost_frames": lost_frames,
                            "last_center_px": [float(value) for value in tracked_center],
                        })
                overlay = draw_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), masks, statistics, selected)
                overlay = draw_crop_box(overlay, crop_bounds)
                cv2.putText(
                    overlay,
                    "SAM3 locked-target tracking during force-contact descent",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok and shared is not None:
                    raise RuntimeError("无法编码下探期间的 SAM3 叠加图")
                if shared is not None:
                    sequence += 1
                    shared.update(encoded.tobytes(), {
                        "state": phase["state"],
                        "sequence": sequence,
                        "preview": "live_left_d405_sam3",
                        "sam3_overlay": phase["overlay_note"],
                        "selected_mask_index": selected,
                        "selected": None if selected is None else statistics[selected],
                        "eligible_candidate_indices": [int(index) for index in candidates],
                        "mask_count": len(statistics),
                        "sam_center_crop_xyxy": list(crop_bounds),
                        "sam_center_crop_fraction": 0.70,
                        "inference_ms": round((time.monotonic() - started) * 1000.0, 1),
                        "safety": "SAM3 仅提供同一锁定目标的水平复位建议；力监测与末端控制仍在独立的 20 ms/50 Hz 循环中。",
                    })
            except Exception as exc:
                if shared is not None:
                    sequence += 1
                    shared.update(None, {
                        "state": "force_contact_descent_preview_error",
                        "sequence": sequence,
                        "preview": "live_left_d405_sam3",
                        "error": str(exc),
                    })
                return
            stop.wait(FORCE_DESCENT_SAM_PREVIEW_PERIOD_S)

    worker = threading.Thread(target=run, name="left-force-descent-preview", daemon=True)
    worker.start()
    return stop, worker


def stop_force_descent_sam_preview(stop: threading.Event | None, worker: threading.Thread | None) -> None:
    if stop is not None:
        stop.set()
    if worker is not None and worker is not threading.current_thread():
        worker.join(timeout=1.0)


def main() -> int:
    args = parse_args()
    if not (
        0 <= args.threshold <= 1
        and 0 <= args.yolo_confidence <= 1
        and 0 <= args.yolo_target_confidence <= 1
        and args.yolo_imgsz > 0
        and args.shaft_association_max_distance_px > 0
        and args.shaft_association_max_horizontal_distance_px > 0
        and args.shaft_association_ambiguity_margin_px >= 0
        and args.sam_review_interval_s > 0
        and args.sam_review_ttl_s > 0
        and args.sam_review_max_pixel_error > 0
        and args.sam_review_max_depth_error_m > 0
        and args.min_area >= 1
        and args.min_valid_depth_pixels >= 1
        and args.warmup >= 0
        and args.max_frames >= 0
        and args.track_max_distance_px > 0
        and 0.0 < args.track_max_depth_jump_m <= 0.20
        and args.track_hold_frames >= 0
        and 0 <= args.web_port <= 65535
        and 0.0 < args.preset_velocity_rad_s <= 0.15
        and 0.001 <= args.preset_arrival_tolerance_rad <= 0.10
        and 1.0 <= args.preset_timeout_s <= 30.0
        and 1.0 <= args.pre_final_settle_timeout_s <= 10.0
        and 0.0 < args.final_descent_visual_max_step_m <= 0.002
        and 0.0 <= args.final_descent_visual_max_total_m <= 0.050
        and 1 <= args.final_descent_target_lost_frames <= 10
        and 0.0 <= args.regrasp_retract_m <= 0.15
        and 0.005 <= args.post_grasp_verify_lift_m <= 0.050
        and 2 <= args.post_grasp_verify_frames <= 10
        and 1.0 <= args.post_grasp_verify_center_drift_px <= 80.0
        and 0.02 <= args.post_grasp_verify_area_change_fraction <= 0.50
        and 0.002 <= args.post_grasp_verify_depth_change_m <= 0.050
        and 0.50 <= args.post_grasp_monitor_max_stale_s <= 5.0
    ):
        print("参数错误：检测、预热、跟踪参数无效。", file=sys.stderr)
        return 2
    if args.full_cycle and (not args.allow_motion or args.stop_after != "none" or args.single_xy_step):
        print("--full-cycle 必须同时指定 --allow-motion，且不能与 --stop-after 或 --single-xy-step 合用。", file=sys.stderr)
        return 2
    try:
        config, parameters, gripper_open, gripper_tolerance, gripper_close_timeout, empirical_pixel_jacobian, contact_parameters, adaptive_gripper = load_config(
            args.config,
            require_calibrated=requires_calibrated_config(
                allow_motion=args.allow_motion,
                stop_after=args.stop_after,
            ),
        )
    except Exception as exc:
        print(f"抓取配置无效：{exc}", file=sys.stderr)
        return 2
    base_speed_mm_s = 10.0 if parameters is None else parameters.final_descent_speed_mps * 1000.0
    if args.speed_mm_s is not None and args.speed_scale is not None:
        print("--speed-mm-s 与旧兼容参数 --speed-scale 不能同时使用。", file=sys.stderr)
        return 2
    if args.speed_mm_s is None:
        speed_mm_s = base_speed_mm_s if args.speed_scale is None else base_speed_mm_s * args.speed_scale
    else:
        speed_mm_s = args.speed_mm_s
    if not math.isfinite(speed_mm_s) or not 1.0 <= speed_mm_s <= 18.0:
        print("参数错误：--speed-mm-s 必须在 1～18 mm/s 之间。", file=sys.stderr)
        return 2
    speed_scale = speed_mm_s / base_speed_mm_s
    motion_preset_velocity_rad_s = args.preset_velocity_rad_s * speed_scale
    motion_preset_timeout_s = args.preset_timeout_s / speed_scale
    if motion_preset_velocity_rad_s > 0.15:
        print("参数错误：该速度会超过 GDK 预设关节速度上限 0.15 rad/s。", file=sys.stderr)
        return 2
    if parameters is not None:
        # Keep the configured descent distance unchanged while changing only
        # its traversal speed: duration is inversely scaled.
        parameters = replace(
            parameters,
            final_descent_speed_mps=parameters.final_descent_speed_mps * speed_scale,
            final_descent_duration_s=parameters.final_descent_duration_s / speed_scale,
        )
    premove_presets: list[tuple[str, list[float]]] | None = None
    post_grasp_presets: list[tuple[str, list[float]]] | None = None
    if args.full_cycle:
        try:
            premove_presets = load_premove_presets(args.poses)
            post_grasp_presets = load_post_grasp_presets(args.poses)
        except Exception as exc:
            print(f"全流程姿态文件无效：{exc}", file=sys.stderr)
            return 2
    if not torch.cuda.is_available():
        print("未检测到 CUDA，不能运行实时 SAM3。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        print(f"找不到 SAM3 权重：{checkpoint}", file=sys.stderr)
        return 2
    yolo_model_path = args.yolo_model.expanduser().resolve()
    if args.detector == "yolo-sam" and not yolo_model_path.is_file():
        print(f"找不到 G2 原生 YOLO 权重：{yolo_model_path}", file=sys.stderr)
        return 2
    log_path = args.log or PROJECT_ROOT / "logs" / f"left_grasp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "mode": "motion" if args.allow_motion else "read_only_preview",
        "config": str(args.config),
        "detector": args.detector,
        "yolo_model": None if args.detector == "sam3" else str(yolo_model_path),
        "yolo_input": None if args.detector == "sam3" else "native_left_d405_color_image; YOLO internal letterbox",
        "log": str(log_path),
        "selection": args.selection,
        "full_cycle": args.full_cycle,
        "speed_mm_s": speed_mm_s,
        "speed_scale": speed_scale,
        "preset_velocity_rad_s": motion_preset_velocity_rad_s,
        "preset_timeout_s": motion_preset_timeout_s,
        "poses": None if not args.full_cycle else str(args.poses.expanduser().resolve()),
        "safety": (
            "single process: preload SAM3, then P2/P3→visual grasp→+Z 100 mm→P4; "
            "one GDK initialization and no extra gripper command after grasp"
            if args.full_cycle else "P2/P3/P4 are not commanded by this program"
        ),
    }, ensure_ascii=False))

    initialized = False
    camera = None
    perception: PersistentPerception | None = None
    position_stream: LeftEndEffectorPositionStream | None = None
    force_preview_stop: threading.Event | None = None
    force_preview_thread: threading.Thread | None = None
    force_preview_phase: dict[str, str] | None = None
    web_server: ThreadingHTTPServer | None = None
    web_shared: SharedView | None = None
    try:
        if args.web_port:
            web_shared = SharedView()
            web_server = ThreadingHTTPServer((args.web_host, args.web_port), handler_class(web_shared))
            threading.Thread(target=web_server.serve_forever, daemon=True).start()
            print(f"内嵌实时页面：http://{args.web_host}:{args.web_port}（与抓取共用同一 SAM3 模型）")
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion, control_period_s=args.control_period_s)
        controller.checked_arm_state()
        camera_type = agibot_gdk.CameraType.kHandLeftColor
        depth_type = agibot_gdk.CameraType.kHandLeftDepth
        camera = agibot_gdk.Camera([camera_type, depth_type])
        tf = agibot_gdk.TF()
        model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=args.threshold)
        yolo_model = None
        if args.detector == "yolo-sam":
            print(f"加载 G2 原生 YOLO 主检测器：{yolo_model_path}", flush=True)
            yolo_model = load_yolo_segmentation_model(yolo_model_path, device=args.yolo_device)
        # From this point onward the worker is the sole camera/model owner.
        # Do not add a second Camera reader or SAM3 thread for a motion phase.
        perception = PersistentPerception(
            camera=camera, color_type=camera_type, depth_type=depth_type,
            gdk=agibot_gdk, tf=tf, processor=processor, yolo_model=yolo_model,
            args=args, shared=web_shared, yolo_model_path=yolo_model_path,
        )
        perception.start()
        state = None if parameters is None else LeftGraspState(parameters)
        color_intrinsic = depth_intrinsic = depth_to_color = end_link_transform = None
        tracked_center: np.ndarray | None = None
        tracked_depth_m: float | None = None
        missed_frames = 0
        regrasp_attempts = 0
        last_end_link_target: CartesianPoseTarget | None = None
        gripper_sent_at: float | None = None
        adaptive_hold_target_rad: float | None = None
        adaptive_hold_sent_at: float | None = None
        gripper_position_samples: list[tuple[float, float]] = []
        contact_baseline_force_n: np.ndarray | None = None
        contact_detector: ForceContactDetector | None = None
        active_regrasp_mailbox: FinalDescentVisualMailbox | None = None
        active_regrasp_mailbox_sequence = 0
        post_grasp_verify_baseline: dict | None = None
        post_grasp_verify_samples: list[dict] = []
        post_grasp_verified = False
        post_grasp_lift_monitor: dict | None = None
        post_grasp_p4_sequence: NonBlockingLeftPresetSequence | None = None
        post_grasp_p4_monitor: dict | None = None
        post_grasp_regrasp_open: dict | None = None
        post_grasp_verify_motion: dict | None = None
        premove_sequence: NonBlockingLeftPresetSequence | None = None
        initial_gripper_open: dict | None = None
        single_xy_verification: dict | None = None
        last_sam_review_at = float("-inf")
        yolo_sam_review: dict = {
            "accepted": False,
            "fresh": False,
            "reason": "awaiting_first_sam_review",
            "sam_selected": None,
        }
        frame = 0
        perception_sequence = 0
        web_sequence = 0
        with log_path.open("a", encoding="utf-8") as log:
            json_record(log, {
                "event": "start",
                "utc": datetime.now(timezone.utc).isoformat(),
                "mode": "motion" if args.allow_motion else "read_only_preview",
                "config": config,
                "detector": args.detector,
                "yolo_model": None if args.detector == "sam3" else str(yolo_model_path),
                "speed_mm_s": speed_mm_s,
                "speed_scale": speed_scale,
                "preset_velocity_rad_s": motion_preset_velocity_rad_s,
                "preset_timeout_s": motion_preset_timeout_s,
                "final_descent_speed_mps": None if parameters is None else parameters.final_descent_speed_mps,
                "final_descent_duration_s": None if parameters is None else parameters.final_descent_duration_s,
                "full_cycle": args.full_cycle,
                "sam3_preloaded_before_premove": args.full_cycle,
                "perception_architecture": "one persistent owner: camera + YOLO + periodic SAM3 + web; state machine consumes latest snapshots only",
            })
            json_record(log, {"event": "gdk_motion_status_before_perception", **controller.motion_status_summary()})
            if args.full_cycle:
                assert premove_presets is not None
                json_record(log, {
                    "event": "premove_start_after_sam3_preload",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "presets": [name for name, _ in premove_presets],
                    "sam3": "loaded",
                })
                premove_sequence = NonBlockingLeftPresetSequence(
                    premove_presets, motion_preset_velocity_rad_s,
                    args.preset_arrival_tolerance_rad, motion_preset_timeout_s, "premove",
                )
            if not args.full_cycle and args.allow_motion and args.stop_after == "none":
                assert gripper_open is not None and gripper_tolerance is not None and gripper_close_timeout is not None
                initial_gripper_open = {
                    "deadline": time.monotonic() + gripper_close_timeout,
                    "next_command_at": time.monotonic(), "attempts": 0,
                }
            while args.max_frames == 0 or frame < args.max_frames:
                if post_grasp_verify_motion is not None:
                    assert position_stream is not None
                    verify = post_grasp_verify_motion
                    if position_stream.is_target_reached():
                        last_end_link_target = verify["target"]
                        post_grasp_verify_baseline = verify["baseline"]
                        post_grasp_verify_samples.clear()
                        post_grasp_verify_motion = None
                        json_record(log, {
                            "event": "post_grasp_verify_lift_arrived",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "lift_m": args.post_grasp_verify_lift_m,
                            "nonblocking": True,
                        })
                    elif time.monotonic() > verify["deadline"]:
                        current = measured_left_end_link_pose(controller)
                        position_stream.update_target(CartesianPoseTarget(
                            current.position_m, last_end_link_target.orientation_xyzw,
                        ))
                        position_stream.stop()
                        position_stream = None
                        assert gripper_open is not None and gripper_close_timeout is not None
                        controller.move_gripper("left", gripper_open)
                        post_grasp_regrasp_open = {
                            "reason": "post_grasp_verify_lift_target_not_reached",
                            "orientation_xyzw": last_end_link_target.orientation_xyzw,
                            "deadline": time.monotonic() + gripper_close_timeout,
                            "next_command_at": time.monotonic() + 0.10,
                            "attempts": 1,
                        }
                        post_grasp_verify_motion = None
                    continue
                if premove_sequence is not None:
                    complete, event = premove_sequence.tick(controller, log)
                    if event is not None:
                        print(json.dumps(event, ensure_ascii=False))
                    if complete:
                        mode_before_visual = controller.motion_status_summary()
                        controller.set_control_mode(1)
                        mode_after_visual = controller.motion_status_summary()
                        if mode_after_visual["control_mode"] != 1 or mode_after_visual["error_code"] != 0:
                            raise SafetyError(
                                f"P3 后未进入健康 control_mode=1；拒绝张开夹爪或开始视觉伺服：{mode_after_visual}"
                            )
                        json_record(log, {
                            "event": "premove_complete",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "final_detection_preset": "left_p3",
                            "nonblocking": True,
                        })
                        premove_sequence = None
                        if args.allow_motion and args.stop_after == "none":
                            assert gripper_close_timeout is not None
                            initial_gripper_open = {
                                "deadline": time.monotonic() + gripper_close_timeout,
                                "next_command_at": time.monotonic(), "attempts": 0,
                            }
                    continue
                if initial_gripper_open is not None:
                    assert gripper_open is not None and gripper_tolerance is not None
                    now = time.monotonic()
                    if now >= initial_gripper_open["next_command_at"]:
                        controller.move_gripper("left", gripper_open)
                        initial_gripper_open["attempts"] += 1
                        initial_gripper_open["next_command_at"] = now + 0.10
                    position = controller.read_gripper_position("left")
                    if position is not None and abs(position - gripper_open) <= gripper_tolerance:
                        json_record(log, {
                            "event": "gripper_open_confirmed",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "actual_position_rad": position,
                            "attempts": initial_gripper_open["attempts"],
                            "nonblocking": True,
                        })
                        initial_gripper_open = None
                    elif now > initial_gripper_open["deadline"]:
                        raise SafetyError("夹爪张开反馈超时；未进入视觉伺服、下探或后续预设动作")
                    continue
                if post_grasp_regrasp_open is not None:
                    assert gripper_open is not None and gripper_tolerance is not None
                    recovery = post_grasp_regrasp_open
                    now = time.monotonic()
                    if now >= recovery["next_command_at"]:
                        controller.move_gripper("left", gripper_open)
                        recovery["attempts"] += 1
                        recovery["next_command_at"] = now + 0.10
                    position = controller.read_gripper_position("left")
                    if position is not None and abs(position - gripper_open) <= gripper_tolerance:
                        stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                        force_preview_stop = force_preview_thread = None
                        force_preview_phase = None
                        resumed_pose = measured_left_end_link_pose(controller)
                        position_stream = controller.start_left_end_effector_position_stream(
                            CartesianPoseTarget(resumed_pose.position_m, recovery["orientation_xyzw"]),
                            rate_hz=50.0,
                            life_time_s=0.10,
                            max_translation_velocity_mps=parameters.final_descent_speed_mps,
                        )
                        last_end_link_target = CartesianPoseTarget(resumed_pose.position_m, recovery["orientation_xyzw"])
                        state.reset_for_regrasp()
                        regrasp_attempts += 1
                        tracked_center = tracked_depth_m = None
                        missed_frames = 0
                        gripper_sent_at = adaptive_hold_target_rad = adaptive_hold_sent_at = None
                        gripper_position_samples.clear()
                        post_grasp_verify_baseline = None
                        post_grasp_verify_samples.clear()
                        post_grasp_verified = False
                        json_record(log, {
                            "event": "post_grasp_regrasp_open_confirmed",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "reason": recovery["reason"],
                            "regrasp_attempt": regrasp_attempts,
                            "actual_position_rad": position,
                            "attempts": recovery["attempts"],
                            "nonblocking": True,
                        })
                        post_grasp_regrasp_open = None
                    elif now > recovery["deadline"]:
                        raise SafetyError("掉件后夹爪张开反馈超时；停止后续命令并保持当前实测状态")
                    continue
                # Once the part has passed the short visual verification,
                # transport is advanced by polling only.  Do not enter a
                # nested wait/sleep loop: the SAM worker keeps the web page
                # live and the mailbox can still veto the lift.
                if post_grasp_lift_monitor is not None:
                    assert position_stream is not None
                    position_stream.raise_if_failed()
                    monitor = post_grasp_lift_monitor
                    observation = monitor["mailbox"].newer_than(monitor["mailbox_sequence"])
                    if observation is not None:
                        monitor["mailbox_sequence"] = int(observation["sequence"])
                        observed_at = time.monotonic()
                        monitor["last_observation_at"] = observed_at
                        monitor["failure"] = held_part_monitor_failure(
                            observation,
                            reference_center_px=monitor["reference_center_px"],
                            reference_area_px=monitor["reference_area_px"],
                            max_center_drift_px=args.post_grasp_verify_center_drift_px,
                            max_area_change_fraction=args.post_grasp_verify_area_change_fraction,
                        )
                        stale_since = monitor.get("stale_since")
                        if stale_since is not None and monitor.get("failure") is None:
                            was_paused = bool(monitor.get("paused"))
                            if was_paused:
                                monitor["deadline"] += max(0.0, observed_at - stale_since)
                                position_stream.update_target(monitor["target"])
                                monitor["paused"] = False
                            json_record(log, {
                                "event": "post_grasp_lift_monitor_recovered",
                                "utc": datetime.now(timezone.utc).isoformat(),
                                "action": "resume_lift_hold_closed" if was_paused else "continue_lift_hold_closed",
                                "mailbox_sequence": monitor["mailbox_sequence"],
                                "nonblocking": True,
                            })
                            monitor["stale_since"] = None
                    now = time.monotonic()
                    if (
                        observation is None
                        and not monitor.get("paused")
                        and now - monitor["last_observation_at"] > args.post_grasp_monitor_max_stale_s
                    ):
                        # A missing snapshot is an unknown state, not proof of
                        # a dropped part.  Keep the active +Z stream moving
                        # smoothly and keep the gripper closed; only a fresh
                        # confirmed-loss observation may cancel transport.
                        if monitor.get("stale_since") is None:
                            monitor["stale_since"] = now
                        if not monitor.get("stale_logged"):
                            monitor["stale_logged"] = True
                            json_record(log, {
                                "event": "post_grasp_lift_monitor_stale_continue",
                                "utc": datetime.now(timezone.utc).isoformat(),
                                "reason": "perception_stale_unknown_not_confirmed_lost",
                                "stale_age_s": round(now - monitor["last_observation_at"], 3),
                                "max_stale_s": args.post_grasp_monitor_max_stale_s,
                                "action": "continue_lift_and_keep_gripper_closed",
                                "mailbox_sequence": monitor["mailbox_sequence"],
                                "nonblocking": True,
                            })
                    actual = measured_left_end_link_pose(controller)
                    delta = np.asarray(actual.position_m) - np.asarray(monitor["origin"].position_m)
                    arrived = (
                        abs(float(delta[2]) - PRE_P4_LIFT_M) <= PRE_P4_LIFT_TOLERANCE_M
                        and float(np.linalg.norm(delta[:2])) <= PRE_P4_LIFT_TOLERANCE_M
                    )
                    if monitor.get("failure") is not None:
                        # The nonblocking monitor detected a dropped/moved
                        # part. Hold immediately; the existing recovery path
                        # then opens and restarts visual acquisition.
                        position_stream.update_target(CartesianPoseTarget(
                            actual.position_m, last_end_link_target.orientation_xyzw,
                        ))
                        position_stream.stop()
                        position_stream = None
                        assert gripper_open is not None and gripper_tolerance is not None and gripper_close_timeout is not None
                        controller.move_gripper("left", gripper_open)
                        post_grasp_regrasp_open = {
                            "reason": "held_target_lost_or_changed_during_pre_p4_lift",
                            "monitor": monitor["failure"],
                            "orientation_xyzw": last_end_link_target.orientation_xyzw,
                            "deadline": time.monotonic() + gripper_close_timeout,
                            "next_command_at": time.monotonic() + 0.10,
                            "attempts": 1,
                        }
                        json_record(log, {
                            "event": "post_grasp_lift_monitor_regrasp_started",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "monitor": monitor["failure"],
                            "nonblocking": True,
                        })
                        post_grasp_lift_monitor = None
                        continue
                    if monitor.get("paused"):
                        # Do not finish the lift or enter P4 from a stale
                        # frame.  The next fresh sequence will resume it.
                        continue
                    if arrived:
                        position_stream.stop()
                        position_stream = None
                        last_end_link_target = monitor["target"]
                        json_record(log, {
                            "event": "pre_p4_monitored_lift_arrived",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "actual_delta_base_m": [float(value) for value in delta],
                            "same_target_monitor_samples": monitor["mailbox_sequence"],
                            "nonblocking": True,
                        })
                        post_grasp_lift_monitor = None
                        assert post_grasp_presets is not None
                        post_grasp_p4_sequence = NonBlockingLeftPresetSequence(
                            post_grasp_presets, motion_preset_velocity_rad_s,
                            args.preset_arrival_tolerance_rad, motion_preset_timeout_s, "post_grasp",
                        )
                        post_grasp_p4_monitor = {
                            "mailbox": monitor["mailbox"],
                            "mailbox_sequence": monitor["mailbox_sequence"],
                            "reference_center_px": monitor["reference_center_px"],
                            "reference_area_px": monitor["reference_area_px"],
                            "last_observation_at": time.monotonic(),
                            "stale_since": None,
                            "stale_logged": False,
                            "paused": False,
                            # A fresh observation after the lift is mandatory
                            # before P4 is even requested.  It closes the
                            # race where the part drops just as the lift ends.
                            "awaiting_fresh_observation": True,
                        }
                        perception.set_phase(
                            "post_grasp_p4",
                            "same-target monitoring before and during P4; stale pauses without opening",
                        )
                    elif time.monotonic() > monitor["deadline"]:
                        raise SafetyError("受监控的 P4 前上抬超时；保持当前位置且拒绝进入 P4")
                    # This branch is intentionally non-blocking, but a tight
                    # GDK polling loop can starve the perception worker under
                    # CPython's GIL.  Yield briefly so YOLO keeps publishing.
                    time.sleep(0.01)
                    continue
                if post_grasp_p4_sequence is not None:
                    assert post_grasp_p4_monitor is not None
                    monitor = post_grasp_p4_monitor
                    observation = monitor["mailbox"].newer_than(monitor["mailbox_sequence"])
                    if observation is not None:
                        monitor["mailbox_sequence"] = int(observation["sequence"])
                        observed_at = time.monotonic()
                        monitor["last_observation_at"] = observed_at
                        monitor["failure"] = held_part_monitor_failure(
                            observation,
                            reference_center_px=monitor["reference_center_px"],
                            reference_area_px=monitor["reference_area_px"],
                            max_center_drift_px=args.post_grasp_verify_center_drift_px,
                            max_area_change_fraction=args.post_grasp_verify_area_change_fraction,
                        )
                        monitor["awaiting_fresh_observation"] = False
                        if monitor.get("paused"):
                            monitor["paused"] = False
                            monitor["stale_since"] = None
                            json_record(log, {
                                "event": "post_grasp_p4_monitor_recovered",
                                "utc": datetime.now(timezone.utc).isoformat(),
                                "action": "resume_p4_hold_closed",
                                "mailbox_sequence": monitor["mailbox_sequence"],
                                "nonblocking": True,
                            })
                    now = time.monotonic()
                    if (
                        observation is None
                        and not monitor.get("paused")
                        and now - monitor["last_observation_at"] > args.post_grasp_monitor_max_stale_s
                    ):
                        # Stale perception is an unknown state.  Stop issuing
                        # new P4 targets, but never open the gripper without a
                        # fresh confirmed-loss observation.
                        monitor["paused"] = True
                        monitor["stale_since"] = now
                        if not monitor.get("stale_logged"):
                            monitor["stale_logged"] = True
                            json_record(log, {
                                "event": "post_grasp_p4_monitor_stale_hold",
                                "utc": datetime.now(timezone.utc).isoformat(),
                                "reason": "perception_stale_unknown_not_confirmed_lost",
                                "stale_age_s": round(now - monitor["last_observation_at"], 3),
                                "max_stale_s": args.post_grasp_monitor_max_stale_s,
                                "action": "hold_p4_and_keep_gripper_closed",
                                "mailbox_sequence": monitor["mailbox_sequence"],
                                "nonblocking": True,
                            })
                    if monitor.get("failure") is not None:
                        failure = monitor["failure"]
                        # Before P4 is sent, recovery at the verified lift
                        # pose is safe: open and reacquire the dropped part.
                        # Once a planned P4 request is in flight, GDK exposes
                        # no verified pre-empt/stop API.  Never launch a
                        # regrasp against a moving arm; stop issuing commands
                        # and require the operator to recover safely.
                        if post_grasp_p4_sequence.sent_at is None:
                            stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                            force_preview_stop = force_preview_thread = None
                            force_preview_phase = None
                            assert gripper_open is not None and gripper_close_timeout is not None
                            controller.move_gripper("left", gripper_open)
                            post_grasp_regrasp_open = {
                                "reason": "held_target_lost_or_changed_before_p4",
                                "monitor": failure,
                                "orientation_xyzw": last_end_link_target.orientation_xyzw,
                                "deadline": time.monotonic() + gripper_close_timeout,
                                "next_command_at": time.monotonic() + 0.10,
                                "attempts": 1,
                            }
                            json_record(log, {
                                "event": "post_grasp_p4_monitor_regrasp_started",
                                "utc": datetime.now(timezone.utc).isoformat(),
                                "monitor": failure,
                                "nonblocking": True,
                            })
                            post_grasp_p4_sequence = None
                            post_grasp_p4_monitor = None
                            continue
                        raise SafetyError(
                            "P4 运输期间检测到铰链掉落/异常；已停止发送后续命令。"
                            "当前 GDK 未提供经验证的规划运动抢占接口，禁止在运动中自动重抓；"
                            f"请待机器人静止后由操作员安全恢复。monitor={failure}"
                        )
                    if monitor.get("paused"):
                        # Do not advance a preset sequence while perception is
                        # stale.  A fresh sequence above resumes it.
                        time.sleep(0.01)
                        continue
                    if monitor["awaiting_fresh_observation"]:
                        # Do not issue the P4 request from a stale lift frame.
                        time.sleep(0.01)
                        continue
                    complete, event = post_grasp_p4_sequence.tick(controller, log)
                    if event is not None:
                        print(json.dumps(event, ensure_ascii=False))
                    if complete:
                        stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                        force_preview_stop = force_preview_thread = None
                        force_preview_phase = None
                        json_record(log, {
                            "event": "full_cycle_complete",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "gripper": "preserved after confirmed grasp",
                            "nonblocking": True,
                        })
                        print("完整流程完成：P2→P3、抓取、受监控 +Z 上抬 100 mm、P4；夹爪保持闭合。")
                        return 0
                    time.sleep(0.01)
                    continue
                frame += 1
                started = time.monotonic()
                record: dict = {"event": "frame", "frame": frame, "utc": datetime.now(timezone.utc).isoformat(), "motion_enabled": args.allow_motion, "detector": args.detector}
                try:
                    assert perception is not None
                    snapshot = perception.newer_than(perception_sequence)
                    if snapshot is None:
                        record.update({"state": "waiting_for_perception", "reason": "no_fresh_perception_snapshot; holding_current_pose"})
                        json_record(log, record)
                        print_compact_record(record)
                        continue
                    perception_sequence = int(snapshot["sequence"])
                    color_rgb = snapshot["color_rgb"]
                    masks = snapshot["masks"]
                    statistics = snapshot["statistics"]
                    candidates = snapshot["candidates"]
                    color_intrinsic = snapshot["color_intrinsic"]
                    depth_intrinsic = snapshot["depth_intrinsic"]
                    depth_to_color = snapshot["depth_to_color"]
                    end_link_transform = snapshot["end_link_transform"]
                    roi = snapshot["roi"]
                    crop_bounds = snapshot["crop_bounds"]
                    if args.detector == "yolo-sam":
                        record["yolo"] = {
                            "model": str(yolo_model_path),
                            "input": "persistent_native_left_d405_color_image",
                            "hinge_candidate_indices": [int(index) for index in snapshot.get("hinge_candidates", candidates)],
                            "shaft_candidate_indices": [int(index) for index in snapshot.get("shaft_candidates", [])],
                            "eligible_candidate_indices": [int(index) for index in candidates],
                            "shaft_association": snapshot.get("shaft_association"),
                        }
                    record["eligible_candidate_indices"] = [int(index) for index in candidates]
                    record["sam_center_crop_xyxy"] = list(crop_bounds)
                    record["sam_center_crop_fraction"] = 0.70
                    record["effective_roi_xyxy"] = None if roi is None else list(roi)
                    record["candidate_diagnostics"] = [
                        {
                            "mask_index": int(index),
                            "center_px": statistics[index].get("center_px"),
                            "area_px": statistics[index].get("area_px"),
                            "valid_depth_px": statistics[index].get("valid_depth_px"),
                            "depth_median_m": statistics[index].get("depth_median_m"),
                            "sam3_confidence": statistics[index].get("sam3_confidence"),
                            "yolo_confidence": statistics[index].get("yolo_confidence"),
                            "yolo_class_id": statistics[index].get("yolo_class_id"),
                            "yolo_class_name": statistics[index].get("yolo_class_name"),
                        }
                        for index in range(len(statistics))
                    ]
                    depth_jump_rejected: list[int] = []
                    if tracked_center is not None:
                        nearby = [index for index in candidates if float(np.linalg.norm(np.asarray(statistics[index]["center_px"]) - tracked_center)) <= args.track_max_distance_px]
                        if nearby:
                            # Image-space identity is authoritative for this
                            # visual/force grasp.  Metallic holes and
                            # reflections can make a D405 mask depth jump to
                            # the background, so depth must never cause the
                            # tracker to substitute a different SAM mask.
                            selected = min(nearby, key=lambda index: float(np.linalg.norm(np.asarray(statistics[index]["center_px"]) - tracked_center)))
                            raw_selected_depth = statistics[selected].get("depth_median_m")
                            selected_depth_m = None if raw_selected_depth is None else float(raw_selected_depth)
                            depth_outlier = (
                                tracked_depth_m is not None
                                and selected_depth_m is not None
                                and abs(selected_depth_m - tracked_depth_m) > args.track_max_depth_jump_m
                            )
                            if depth_outlier:
                                depth_jump_rejected = [selected]
                            tracked_center = np.asarray(statistics[selected]["center_px"], dtype=np.float64)
                            missed_frames = 0
                            tracking = "locked_depth_outlier" if depth_outlier else "locked"
                        else:
                            missed_frames += 1
                            selected = None
                            tracking = "holding_lock"
                            if missed_frames > args.track_hold_frames:
                                tracked_center = None
                                missed_frames = 0
                                selected = choose_candidate(statistics, candidates, args.selection)
                                tracking = "reacquired" if selected is not None else "no_target"
                                if selected is not None and statistics[selected].get("depth_median_m") is not None:
                                    tracked_depth_m = float(statistics[selected]["depth_median_m"])
                    else:
                        selected = choose_candidate(statistics, candidates, args.selection)
                        tracking = "acquired" if selected is not None else "no_target"
                    if selected is not None and tracked_center is None:
                        tracked_center = np.asarray(statistics[selected]["center_px"], dtype=np.float64)
                        if statistics[selected].get("depth_median_m") is not None:
                            tracked_depth_m = float(statistics[selected]["depth_median_m"])
                    if args.detector == "yolo-sam":
                        yolo_sam_review = dict(snapshot["sam_review"])
                        record["sam_review"] = {key: value for key, value in yolo_sam_review.items() if key != "timestamp_monotonic"}
                        reviewed_same_target = (
                            selected is not None
                            and yolo_sam_review.get("yolo_center_px") is not None
                            and float(np.linalg.norm(np.asarray(statistics[selected]["center_px"], dtype=np.float64) - np.asarray(yolo_sam_review["yolo_center_px"], dtype=np.float64))) <= args.track_max_distance_px
                        )
                        if not (reviewed_same_target and yolo_sam_review.get("accepted") and yolo_sam_review.get("fresh")):
                            selected = None
                            tracking = "withheld_pending_or_failed_sam_review"
                    shaft_association = (
                        associate_shaft_to_hinge(
                            statistics,
                            selected,
                            shaft_indices=[int(index) for index in snapshot.get("shaft_candidates", [])],
                            max_center_distance_px=args.shaft_association_max_distance_px,
                            max_horizontal_distance_px=args.shaft_association_max_horizontal_distance_px,
                            ambiguity_margin_px=args.shaft_association_ambiguity_margin_px,
                        )
                        if args.detector == "yolo-sam"
                        else {"shaft_index": None, "reason": "sam3_mode"}
                    )
                    if args.detector == "yolo-sam" and "yolo" in record:
                        record["yolo"]["shaft_association"] = shaft_association
                    record.update({
                        "tracking": tracking,
                        "tracking_depth_anchor_m": tracked_depth_m,
                        "tracking_depth_max_jump_m": args.track_max_depth_jump_m,
                        "depth_outlier_candidate_indices": [int(index) for index in depth_jump_rejected],
                        "selected_mask_index": selected,
                        "mask_count": len(statistics),
                        "selected": None if selected is None else statistics[selected],
                        "shaft_association": shaft_association,
                        "shaft": compact_detection(
                            statistics,
                            None if shaft_association.get("shaft_index") is None else int(shaft_association["shaft_index"]),
                        ),
                    })
                    if selected is None:
                        # FINAL_DESCENT here means that XY centering has just
                        # completed; the synchronous force descent has not
                        # been commanded yet.  A transient YOLO depth/SAM
                        # review miss must therefore only cancel that pending
                        # descent and return to visual alignment, rather than
                        # end the entire continuous regrasp process.
                        if state is not None and state.phase in {GraspPhase.APPROACH_DEPTH, GraspPhase.FINAL_DESCENT}:
                            state.reset_alignment()
                            record["target_loss_reset_to_xy_alignment"] = True
                        if not candidates:
                            unavailable_reason = "no_eligible_yolo_target"
                        elif args.detector == "yolo-sam":
                            unavailable_reason = "yolo_target_withheld_by_sam3_review"
                        else:
                            unavailable_reason = "tracked_target_not_present_in_latest_snapshot"
                        record["perception_gate"] = {
                            "candidate_count": len(candidates),
                            "tracking": tracking,
                            "sam_review_accepted": record.get("sam_review", {}).get("accepted"),
                            "sam_review_fresh": record.get("sam_review", {}).get("fresh"),
                        }
                        record["state"] = "waiting_for_selected_depth"
                        record["reason"] = f"{unavailable_reason}; holding_current_pose"
                        perception.set_phase(record["state"], record["reason"])
                        json_record(log, record)
                        print_compact_record(record)
                        continue
                    item = statistics[selected]
                    if single_xy_verification is not None:
                        post_center = np.asarray(item["center_px"], dtype=np.float64)
                        pre_center = np.asarray(single_xy_verification["before_center_px"], dtype=np.float64)
                        record.update({
                            "state": "single_xy_post_motion_measurement",
                            "reason": "same_target_remeasured_after_one_completed_xy_stream",
                            **single_xy_verification,
                            "after_center_px": [float(value) for value in post_center],
                            "pixel_delta_post_minus_pre_px": [float(value) for value in (post_center - pre_center)],
                            "test_stop": "single_xy_correction_verified",
                        })
                        json_record(log, record)
                        print_compact_record(record)
                        print("单次 XY 受看护测试完成：已确认末端反馈并复测同一视觉目标；未下探、未操作夹爪。")
                        return 0
                    if state is None:
                        record.update({"state": "calibration_required", "reason": "complete left_grasp.example.json before a descent/grasp plan can be produced"})
                        perception.set_phase(record["state"], record["reason"])
                        json_record(log, record)
                        print_compact_record(record)
                        continue
                    if tracking == "reacquired":
                        # A re-acquired target is safe before a vertical goal
                        # exists: require the normal multi-frame XY centering
                        # sequence again instead of treating it as fatal.
                        state.reset_alignment()
                        record["target_reacquired_reset_to_xy_alignment"] = True
                    observation = Observation(
                        tuple(float(value) for value in item["center_px"]),
                        # The empirical pixel Jacobian below is the only XY
                        # controller.  Depth is diagnostic/optional and no
                        # longer filters out the visually leftmost hinge.
                        float(item["depth_median_m"] if item.get("depth_median_m") is not None else 1.0),
                        float(color_intrinsic["fx"]), float(color_intrinsic["fy"]),
                        float(color_intrinsic["ppx"]), float(color_intrinsic["ppy"]),
                    )
                    step = state.update(observation, now=time.monotonic())
                    record.update({"state": step.phase.value, "reason": step.reason, "camera_delta_m": step.camera_delta_m, "base_delta_m": step.base_delta_m})
                    perception.set_phase(record["state"], record["reason"])
                    if args.stop_after == "xy" and step.phase == GraspPhase.FINAL_DESCENT:
                        record["test_stop"] = "xy_alignment_complete_before_force_guarded_descent"
                        json_record(log, record)
                        print_compact_record(record)
                        print("受看护 XY 测试完成：已停止在目标上方，未进行力接触下探、未闭爪。")
                        return 0
                    if step.camera_delta_m is not None or step.base_delta_m is not None:
                        if step.camera_delta_m is not None:
                            if empirical_pixel_jacobian is None:
                                raise SafetyError("视觉 XY 修正缺少实测 pixel_jacobian_px_per_m")
                            pixel_error = np.asarray((
                                float(item["center_px"][0]) - parameters.target_u_px,
                                float(item["center_px"][1]) - parameters.target_v_px,
                            ), dtype=np.float64)
                            delta_base = empirical_horizontal_base_delta_from_pixel_error(
                                pixel_error, empirical_pixel_jacobian, max_step_m=parameters.max_xy_step_m,
                            )
                            command_source = "gdk_end_effector_pose_mode1_empirical_visual_xy"
                            record["pixel_error_px"] = [float(value) for value in pixel_error]
                            record["pixel_jacobian_px_per_m"] = empirical_pixel_jacobian.tolist()
                        else:
                            assert step.base_delta_m is not None
                            assert step.motion_duration_s is not None
                            delta_base = np.asarray(step.base_delta_m, dtype=np.float64)
                            command_source = "gdk_end_effector_pose_mode1_force_guarded_final_descent" if step.is_final_descent else "gdk_end_effector_pose_mode1_vertical_approach"
                        if step.is_final_descent and position_stream is not None:
                            prior_target_ready = position_stream.is_target_reached()
                            record["pre_final_xy_settle"] = {
                                "timeout_s": args.pre_final_settle_timeout_s,
                                "stream_target_reached": prior_target_ready,
                                "safety": "非阻塞轮询：未到位时仅保持最后一个 XY 设定点，未发送任何下探目标",
                            }
                            if not prior_target_ready:
                                record.update({
                                    "state": "waiting_for_pre_final_xy_arrival",
                                    "reason": "previous_end_effector_target_still_in_flight",
                                })
                                json_record(log, record)
                                print_compact_record(record)
                                continue
                            # The GDK status quaternion can vary by tiny
                            # floating-point amounts between reads.  The
                            # 50 Hz stream intentionally rejects any
                            # orientation change, so use the measured
                            # position as the final-descent origin while
                            # retaining its original fixed orientation.
                            measured_pose = measured_left_end_link_pose(controller)
                            last_end_link_target = CartesianPoseTarget(
                                measured_pose.position_m,
                                last_end_link_target.orientation_xyzw,
                            )
                        if step.is_final_descent and args.allow_motion and contact_parameters is not None:
                            # This is deliberately captured after XY/depth
                            # alignment and before any final downward command.
                            # It reads the existing GDK sensor zero; it never
                            # invokes the GDK torque-sensor calibration API.
                            contact_baseline_force_n = capture_force_baseline(robot, contact_parameters)
                            contact_detector = ForceContactDetector(contact_parameters, contact_baseline_force_n)
                            record["force_contact"] = {
                                "frame_name": contact_parameters.frame_name,
                                "baseline_force_n": [float(value) for value in contact_baseline_force_n],
                                "force_delta_threshold_n": contact_parameters.force_delta_threshold_n,
                                "consecutive_samples_required": contact_parameters.consecutive_samples,
                                "sample_period_s": contact_parameters.sample_period_s,
                            }
                        if last_end_link_target is None:
                            last_end_link_target = measured_left_end_link_pose(controller)
                        target = CartesianPoseTarget(
                            base_position_after_delta(last_end_link_target.position_m, delta_base),
                            last_end_link_target.orientation_xyzw,
                        )
                        record.update({
                            "visual_tip_link": LEFT_VISUAL_TIP_LINK,
                            "command_source": command_source,
                            "gdk_control_mode": 1,
                            "delta_base_m": [float(value) for value in delta_base],
                            "end_link_target_before_base_link": list(last_end_link_target.position_m),
                            "end_link_target_after_base_link": list(target.position_m),
                            "end_link_orientation_xyzw": list(target.orientation_xyzw),
                        })
                        if args.allow_motion:
                            if position_stream is None:
                                record["gdk_motion_status_before_end_pose_stream"] = controller.motion_status_summary()
                                position_stream = controller.start_left_end_effector_position_stream(
                                    last_end_link_target,
                                    rate_hz=50.0,
                                    life_time_s=0.10,
                                    max_translation_velocity_mps=parameters.final_descent_speed_mps,
                                )
                            position_stream.update_target(target)
                            last_end_link_target = target
                            if step.is_final_descent:
                                assert step.motion_duration_s is not None
                                force_contact_detected = False
                                if contact_detector is not None:
                                    # The optional descent visual servo may
                                    # make only bounded horizontal corrections
                                    # for the one target locked before the
                                    # descent.  It never changes Z, and the
                                    # force guard freezes it well before
                                    # contact takes priority.
                                    visual_mailbox = (SnapshotVisualMailbox(
                                        perception,
                                        initial_center_px=item["center_px"],
                                        track_max_distance_px=args.track_max_distance_px,
                                        target_lost_frames=args.final_descent_target_lost_frames,
                                        target_pixel_px=(parameters.target_u_px, parameters.target_v_px),
                                        pixel_jacobian=empirical_pixel_jacobian,
                                        pixel_tolerance_px=parameters.pixel_tolerance,
                                        max_xy_step_m=args.final_descent_visual_max_step_m,
                                    ) if args.final_descent_visual_servo else None)
                                    # Start after the frame used to initiate
                                    # descent; an old frame must not count as
                                    # a fresh same-target observation.
                                    visual_mailbox_sequence = perception_sequence
                                    visual_total_base_delta = np.zeros(3, dtype=np.float64)
                                    visual_corrections: list[dict] = []
                                    force_preview_phase = {
                                        "state": "force_contact_descent",
                                        "overlay_note": (
                                            "bounded same-target XY visual servo; force guard freezes XY before contact"
                                            if args.final_descent_visual_servo
                                            else "live display only; force is the only final-descent controller"
                                        ),
                                    }
                                    perception.set_phase("force_contact_descent", force_preview_phase["overlay_note"])
                                    final_deadline = time.monotonic() + step.motion_duration_s + 0.25
                                    maximum_delta_force_n = 0.0
                                    latest_force_n: np.ndarray | None = None
                                    latest_hits = 0
                                    force_relief_completed = False
                                    final_descent_retry_reason: str | None = None
                                    while time.monotonic() < final_deadline:
                                        position_stream.raise_if_failed()
                                        _status, force_n = read_wrench_force(robot, contact_detector.parameters.frame_name)
                                        detected, delta_force_n, hits = contact_detector.update(force_n)
                                        latest_force_n = force_n
                                        latest_hits = hits
                                        maximum_delta_force_n = max(maximum_delta_force_n, delta_force_n)
                                        if detected:
                                            # Cancel the queued downward goal
                                            # immediately.  Do not wait for a
                                            # synthetic stream target to be
                                            # "reached" here: after a contact
                                            # the GDK measured pose can differ
                                            # from the stream's internal
                                            # interpolated setpoint by more
                                            # than 5 mm, and that wait is not
                                            # a physical hold confirmation.
                                            # The next target is constructed
                                            # from the measured contact pose,
                                            # so it can only ask for relief
                                            # (up), never further descent.
                                            contact_pose = measured_left_end_link_pose(controller)
                                            position_stream.raise_if_failed()
                                            force_contact_detected = True
                                            record["actual_contact_base_delta_m"] = [
                                                float(value)
                                                for value in (
                                                    np.asarray(contact_pose.position_m)
                                                    - np.asarray(record["end_link_target_before_base_link"])
                                                )
                                            ]
                                            record["actual_contact_end_link_base_link"] = list(contact_pose.position_m)
                                            # Always replace the queued down
                                            # target with a hold at the
                                            # measured contact pose first.
                                            # A light contact does not need an
                                            # unloading lift, but it must
                                            # never inherit a downward target.
                                            contact_hold_target = CartesianPoseTarget(
                                                contact_pose.position_m,
                                                last_end_link_target.orientation_xyzw,
                                            )
                                            position_stream.update_target(contact_hold_target)
                                            last_end_link_target = contact_hold_target
                                            if delta_force_n <= contact_detector.parameters.relief_force_delta_threshold_n:
                                                force_relief_completed = True
                                                record["force_relief"] = {
                                                    "required": False,
                                                    "reason": "contact_force_within_close_limit",
                                                    "max_lift_m": contact_detector.parameters.relief_max_lift_m,
                                                    "force_delta_release_threshold_n": contact_detector.parameters.relief_force_delta_threshold_n,
                                                    "consecutive_samples_required": contact_detector.parameters.relief_consecutive_samples,
                                                    "released": True,
                                                    "latest_force_n": [float(value) for value in force_n],
                                                    "latest_delta_force_n": delta_force_n,
                                                    "consecutive_hits": 0,
                                                }
                                                break
                                            relief_target = CartesianPoseTarget(
                                                base_position_after_delta(
                                                    contact_pose.position_m,
                                                    np.asarray((0.0, 0.0, contact_detector.parameters.relief_max_lift_m)),
                                                ),
                                                last_end_link_target.orientation_xyzw,
                                            )
                                            position_stream.update_target(relief_target)
                                            last_end_link_target = relief_target
                                            relief_deadline = time.monotonic() + (
                                                contact_detector.parameters.relief_max_lift_m
                                                / parameters.final_descent_speed_mps
                                                + 0.35
                                            )
                                            relief_hits = 0
                                            latest_relief_force_n: np.ndarray | None = None
                                            latest_relief_delta_force_n: float | None = None
                                            while time.monotonic() < relief_deadline:
                                                position_stream.raise_if_failed()
                                                _status, relief_force_n = read_wrench_force(robot, contact_detector.parameters.frame_name)
                                                relief_delta_force_n = float(np.linalg.norm(
                                                    relief_force_n - contact_detector.baseline_force_n
                                                ))
                                                latest_relief_force_n = relief_force_n
                                                latest_relief_delta_force_n = relief_delta_force_n
                                                if relief_delta_force_n <= contact_detector.parameters.relief_force_delta_threshold_n:
                                                    relief_hits += 1
                                                else:
                                                    relief_hits = 0
                                                if relief_hits >= contact_detector.parameters.relief_consecutive_samples:
                                                    relief_pose = measured_left_end_link_pose(controller)
                                                    relief_hold_target = CartesianPoseTarget(
                                                        relief_pose.position_m,
                                                        last_end_link_target.orientation_xyzw,
                                                    )
                                                    position_stream.update_target(relief_hold_target)
                                                    last_end_link_target = relief_hold_target
                                                    # This is a cancellation of
                                                    # the remaining upward
                                                    # relief target at the
                                                    # measured pose.  As above,
                                                    # a fixed virtual-target
                                                    # wait would reject valid
                                                    # contact cases without
                                                    # checking physical motion.
                                                    position_stream.raise_if_failed()
                                                    force_relief_completed = True
                                                    break
                                                time.sleep(contact_detector.parameters.sample_period_s)
                                            record["force_relief"] = {
                                                "required": True,
                                                "max_lift_m": contact_detector.parameters.relief_max_lift_m,
                                                "force_delta_release_threshold_n": contact_detector.parameters.relief_force_delta_threshold_n,
                                                "consecutive_samples_required": contact_detector.parameters.relief_consecutive_samples,
                                                "released": force_relief_completed,
                                                "latest_force_n": None if latest_relief_force_n is None else [float(value) for value in latest_relief_force_n],
                                                "latest_delta_force_n": latest_relief_delta_force_n,
                                                "consecutive_hits": relief_hits,
                                            }
                                            if not force_relief_completed:
                                                raise SafetyError(
                                                    "检测到接触后已达到配置的最大卸力上提量但接触力未降至闭爪阈值；已停止且不闭爪"
                                                )
                                            break
                                        # Force is the primary guard.  Do not
                                        # accept visual corrections once the
                                        # force magnitude is even approaching
                                        # the configured contact threshold;
                                        # after contact the branch above has
                                        # already replaced the descent target
                                        # by a measured-pose hold.
                                        if visual_mailbox is not None:
                                            correction = visual_mailbox.newer_than(visual_mailbox_sequence)
                                            if correction is not None:
                                                visual_mailbox_sequence = int(correction["sequence"])
                                                if correction.get("valid") is False:
                                                    # The SAM worker only
                                                    # reports this for the
                                                    # exact target that was
                                                    # locked before descent;
                                                    # do not search sideways
                                                    # at table height.  Hold
                                                    # immediately, then the
                                                    # outer recovery returns
                                                    # to a safe height before
                                                    # reacquiring a target.
                                                    hold_pose = measured_left_end_link_pose(controller)
                                                    hold_target = CartesianPoseTarget(
                                                        hold_pose.position_m,
                                                        last_end_link_target.orientation_xyzw,
                                                    )
                                                    position_stream.update_target(hold_target)
                                                    last_end_link_target = hold_target
                                                    final_descent_retry_reason = str(correction.get("reason", "locked_target_lost"))
                                                    visual_corrections.append({
                                                        "sequence": visual_mailbox_sequence,
                                                        "accepted": False,
                                                        "reason": final_descent_retry_reason,
                                                        "lost_frames": correction.get("lost_frames"),
                                                        "action": "hold_then_retract_open_reacquire",
                                                    })
                                                    break
                                                if correction.get("valid") is None:
                                                    # A distinct fresh frame
                                                    # missed the locked target,
                                                    # but the configured
                                                    # consecutive-loss count
                                                    # has not been reached.
                                                    visual_corrections.append({
                                                        "sequence": visual_mailbox_sequence,
                                                        "accepted": False,
                                                        "reason": correction.get("reason", "locked_target_missing_pending"),
                                                        "lost_frames": correction.get("lost_frames"),
                                                    })
                                                    continue
                                                if "delta_base_m" not in correction:
                                                    # The locked target is
                                                    # healthy and already in
                                                    # the XY tolerance band.
                                                    # A monitor observation is
                                                    # not necessarily a motion
                                                    # request.
                                                    visual_corrections.append({
                                                        "sequence": visual_mailbox_sequence,
                                                        "accepted": False,
                                                        "reason": "xy_within_tolerance",
                                                    })
                                                    continue
                                                if delta_force_n >= contact_detector.parameters.force_delta_threshold_n * 0.5:
                                                    visual_corrections.append({
                                                        "sequence": visual_mailbox_sequence,
                                                        "accepted": False,
                                                        "reason": "force_near_contact",
                                                        "delta_force_n": float(delta_force_n),
                                                        "suggested_delta_base_m": correction["delta_base_m"],
                                                    })
                                                else:
                                                    proposed_total = visual_total_base_delta + np.asarray(
                                                        correction["delta_base_m"], dtype=np.float64
                                                    )
                                                    if (
                                                        args.final_descent_visual_max_total_m > 0.0
                                                        and float(np.linalg.norm(proposed_total[:2]))
                                                        > args.final_descent_visual_max_total_m
                                                    ):
                                                        # The object has moved
                                                        # farther than the
                                                        # locally validated
                                                        # descent envelope.
                                                        # Do not chase it at
                                                        # table height: hold,
                                                        # retract, then resume
                                                        # normal full-range XY
                                                        # alignment from above.
                                                        hold_pose = measured_left_end_link_pose(controller)
                                                        hold_target = CartesianPoseTarget(
                                                            hold_pose.position_m,
                                                            last_end_link_target.orientation_xyzw,
                                                        )
                                                        position_stream.update_target(hold_target)
                                                        last_end_link_target = hold_target
                                                        final_descent_retry_reason = "visual_horizontal_limit"
                                                        visual_corrections.append({
                                                            "sequence": visual_mailbox_sequence,
                                                            "accepted": False,
                                                            "reason": final_descent_retry_reason,
                                                            "suggested_delta_base_m": correction["delta_base_m"],
                                                            "cumulative_horizontal_m": float(np.linalg.norm(proposed_total[:2])),
                                                            "action": "hold_then_retract_open_reacquire",
                                                        })
                                                        break
                                                    else:
                                                        correction_target = CartesianPoseTarget(
                                                            base_position_after_delta(
                                                                last_end_link_target.position_m,
                                                                np.asarray(correction["delta_base_m"], dtype=np.float64),
                                                            ),
                                                            last_end_link_target.orientation_xyzw,
                                                        )
                                                        position_stream.update_target(correction_target)
                                                        last_end_link_target = correction_target
                                                        visual_total_base_delta = proposed_total
                                                        visual_corrections.append({
                                                            "sequence": visual_mailbox_sequence,
                                                            "accepted": True,
                                                            "selected_mask_index": correction["selected_mask_index"],
                                                            "pixel_error_px": correction["pixel_error_px"],
                                                            "delta_base_m": correction["delta_base_m"],
                                                            "cumulative_horizontal_m": float(np.linalg.norm(visual_total_base_delta[:2])),
                                                        })
                                        time.sleep(contact_detector.parameters.sample_period_s)
                                    record["final_descent_visual_servo"] = {
                                        "enabled": visual_mailbox is not None,
                                        "target_lock": "same SAM target nearest to pre-descent YOLO/SAM selection; never re-rank left/right during descent",
                                        "max_step_m": args.final_descent_visual_max_step_m,
                                        "max_total_horizontal_m": (
                                            None
                                            if args.final_descent_visual_max_total_m == 0.0
                                            else args.final_descent_visual_max_total_m
                                        ),
                                        "force_freeze_delta_n": contact_detector.parameters.force_delta_threshold_n * 0.5,
                                        "accepted_total_base_delta_m": [float(value) for value in visual_total_base_delta],
                                        "corrections": visual_corrections,
                                    }
                                    record["force_contact"] = {
                                        **record.get("force_contact", {}),
                                        "detected": force_contact_detected,
                                        "latest_force_n": None if latest_force_n is None else [float(value) for value in latest_force_n],
                                        "latest_delta_force_n": None if latest_force_n is None else float(np.linalg.norm(latest_force_n - contact_detector.baseline_force_n)),
                                        "maximum_delta_force_n": maximum_delta_force_n,
                                        "consecutive_hits": latest_hits,
                                    }
                                    if final_descent_retry_reason is not None:
                                        assert gripper_open is not None and gripper_tolerance is not None and gripper_close_timeout is not None
                                        # The worker owns Camera/SAM during
                                        # descent.  Shut it down before the
                                        # resumed outer loop reads the same
                                        # GDK camera again.
                                        stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                                        force_preview_stop = force_preview_thread = None
                                        force_preview_phase = None
                                        position_stream, last_end_link_target, recovery = recover_for_visual_regrasp(
                                            controller,
                                            position_stream,
                                            retract_m=args.regrasp_retract_m,
                                            translation_speed_mps=parameters.final_descent_speed_mps,
                                            fixed_orientation_xyzw=last_end_link_target.orientation_xyzw,
                                            gripper_open_rad=gripper_open,
                                            gripper_tolerance_rad=gripper_tolerance,
                                            gripper_timeout_s=gripper_close_timeout,
                                        )
                                        regrasp_attempts += 1
                                        state.reset_for_regrasp()
                                        tracked_center = tracked_depth_m = None
                                        missed_frames = 0
                                        yolo_sam_review = {
                                            "accepted": False,
                                            "fresh": False,
                                            "reason": "regrasp_reacquiring_target",
                                            "sam_selected": None,
                                        }
                                        record.update({
                                            "state": "regrasp_reacquiring",
                                            "reason": final_descent_retry_reason,
                                            "regrasp_attempt": regrasp_attempts,
                                            "regrasp_recovery": recovery,
                                        })
                                        json_record(log, record)
                                        print_compact_record(record)
                                        continue
                                    if not force_contact_detected and args.stop_after != "descend":
                                        raise SafetyError(
                                            "最终下探达到已配置的最大行程但未检测到铰链平面接触；已停止且不闭爪"
                                        )
                                    if force_contact_detected and force_preview_phase is not None:
                                        # Keep the same model/camera worker
                                        # running throughout adaptive closure,
                                        # lift, and P4.  The finally block
                                        # owns its single shutdown.
                                        force_preview_phase.update({
                                            "state": "post_contact_grasp_and_transport",
                                            "overlay_note": "live display only during closure/lift/transport; it cannot alter motion or gripper decisions",
                                        })
                                    if force_contact_detected:
                                        # Do not use image displacement while the jaws are
                                        # closing as a grasp-failure signal.  The fingers
                                        # occlude and can locally rotate a correctly held
                                        # hinge, so its pixel centre no longer has the
                                        # pre-contact alignment meaning.  Confirmation is
                                        # deferred to the gripper-stall check followed by
                                        # the dedicated 20 mm held-part verification.
                                        active_regrasp_mailbox = None
                                        active_regrasp_mailbox_sequence = visual_mailbox_sequence
                                elif not position_stream.wait_until_target(step.motion_duration_s + 0.25):
                                    raise SafetyError("定时下探未在规定时间内发送到最终 GDK 末端目标；拒绝闭爪")
                                time.sleep(0.10)
                                actual_end_link = measured_left_end_link_pose(controller)
                                actual_base_delta = np.asarray(actual_end_link.position_m) - np.asarray(record["end_link_target_before_base_link"])
                                record["actual_base_delta_m"] = [float(value) for value in actual_base_delta]
                            record["motion_command_sent"] = True
                        else:
                            record["motion_command_sent"] = False
                        if args.allow_motion and args.single_xy_step and step.camera_delta_m is not None:
                            assert position_stream is not None
                            commanded_distance_m = float(np.linalg.norm(delta_base))
                            stream_timeout_s = max(0.5, commanded_distance_m / parameters.final_descent_speed_mps + 0.5)
                            if not position_stream.wait_until_target(stream_timeout_s):
                                raise SafetyError("单次 XY 目标未在限定时间内完整发布；拒绝进行动作后视觉验证")
                            time.sleep(0.10)
                            actual_end_link = measured_left_end_link_pose(controller)
                            actual_delta = np.asarray(actual_end_link.position_m) - np.asarray(record["end_link_target_before_base_link"])
                            single_xy_verification = {
                                "before_center_px": [float(value) for value in item["center_px"]],
                                "commanded_delta_base_m": [float(value) for value in delta_base],
                                "stream_target_reached": True,
                                "actual_end_link_after_base_link": list(actual_end_link.position_m),
                                "actual_end_link_delta_base_m": [float(value) for value in actual_delta],
                            }
                            record.update({"single_xy_waiting_for_post_motion_measurement": True, **single_xy_verification})
                            json_record(log, record)
                            print_compact_record(record)
                            continue
                        if (
                            args.allow_motion
                            and args.stop_after == "descend"
                            and step.is_final_descent
                        ):
                            record["test_stop"] = (
                                "force_contact_detected_before_gripper_close"
                                if contact_detector is not None and force_contact_detected
                                else "max_final_descent_reached_without_contact_before_gripper_close"
                            )
                            json_record(log, record)
                            print_compact_record(record)
                            print("受看护定时下探测试完成：已停止且未闭爪。")
                            return 0
                        if args.allow_motion and step.is_final_descent:
                            actual_down_m = -float(actual_base_delta[2])
                            actual_lateral_m = float(np.linalg.norm(actual_base_delta[:2]))
                            requested_down_m = -float(delta_base[2])
                            if contact_detector is not None and not force_contact_detected:
                                raise SafetyError("最终下探未检测到接触；拒绝闭爪")
                            if contact_detector is None:
                                # A pure timed descent has no independent
                                # contact proof, therefore it must complete
                                # the requested vertical move without a
                                # material lateral deviation before closure.
                                if (
                                    actual_down_m
                                    < requested_down_m - parameters.final_descent_completion_tolerance_m
                                    or actual_lateral_m
                                    > parameters.final_descent_completion_tolerance_m
                                ):
                                    raise SafetyError(
                                        f"定时下探实测位移不满足已配置闭爪条件：down={actual_down_m:.6f}m, "
                                        f"lateral={actual_lateral_m:.6f}m"
                                    )
                            else:
                                # For force-guarded descent, the verified
                                # contact and three-frame force release above
                                # are the closure conditions.  Contact can
                                # create a small measured XY displacement, so
                                # the timed-descent 0.5 mm straightness check
                                # must not override the force result.
                                record["force_contact_closure_ready"] = {
                                    "actual_down_m": actual_down_m,
                                    "actual_lateral_m": actual_lateral_m,
                                    "criterion": "contact_detected_and_force_released",
                                }
                            step = state.confirm_final_descent()
                            record.update({
                                "state": step.phase.value,
                                "reason": "force_contact_detected" if contact_detector is not None else step.reason,
                            })
                    if step.should_close_gripper:
                        assert adaptive_gripper is not None and gripper_open is not None
                        record["adaptive_gripper_close"] = {
                            "seek_target_rad": adaptive_gripper.seek_target_rad,
                            "minimum_close_travel_rad": adaptive_gripper.minimum_close_travel_rad,
                            "stall_window_s": adaptive_gripper.stall_window_s,
                            "stall_position_delta_rad": adaptive_gripper.stall_position_delta_rad,
                            "additional_close_rad": adaptive_gripper.additional_close_rad,
                        }
                        record["proposed_gripper_seek_rad"] = adaptive_gripper.seek_target_rad
                        if args.allow_motion:
                            if position_stream is None:
                                raise SafetyError("最终下探后末端位姿流未运行；拒绝发送夹爪命令")
                            # Close towards the mechanical endpoint only to
                            # locate this particular hinge's stall angle. The
                            # fixed close_rad calibration is intentionally not
                            # used as a grasp-success criterion: the contact
                            # angle varies with hinge placement.
                            position_stream.set_left_gripper_target(adaptive_gripper.seek_target_rad)
                            gripper_sent_at = time.monotonic()
                            adaptive_hold_target_rad = None
                            adaptive_hold_sent_at = None
                            gripper_position_samples.clear()
                            state.confirm_grasp()
                            record["gripper_command_sent"] = True
                            record["gripper_command_source"] = (
                                "gdk_end_effector_pose_control_50hz_adaptive_stall_seek"
                            )
                        else:
                            record["gripper_command_sent"] = False
                    if state.phase == GraspPhase.GRASPED and gripper_sent_at is not None:
                        if post_grasp_verify_baseline is not None:
                            center_delta_px = float(np.linalg.norm(
                                np.asarray(item["center_px"], dtype=np.float64)
                                - np.asarray(post_grasp_verify_baseline["center_px"], dtype=np.float64)
                            ))
                            area_fraction_change = abs(
                                float(item["area_px"]) / float(post_grasp_verify_baseline["area_px"]) - 1.0
                            )
                            post_grasp_verify_samples.append({
                                "center_delta_px": center_delta_px,
                                "area_fraction_change": area_fraction_change,
                                "depth_change_m": abs(
                                    float(item["depth_median_m"])
                                    - float(post_grasp_verify_baseline["depth_median_m"])
                                ) if (
                                    item.get("depth_median_m") is not None
                                    and post_grasp_verify_baseline.get("depth_median_m") is not None
                                ) else None,
                                "center_px": item["center_px"],
                                "area_px": item["area_px"],
                            })
                            if len(post_grasp_verify_samples) < args.post_grasp_verify_frames:
                                record.update({
                                    "state": "post_grasp_visual_verifying",
                                    "reason": "collecting_same_target_post_lift_frames",
                                    "post_grasp_verify": {
                                        "baseline": post_grasp_verify_baseline,
                                        "samples": post_grasp_verify_samples,
                                    },
                                })
                                json_record(log, record)
                                print_compact_record(record)
                                continue
                            median_center_delta_px = float(np.median([
                                sample["center_delta_px"] for sample in post_grasp_verify_samples
                            ]))
                            median_area_fraction_change = float(np.median([
                                sample["area_fraction_change"] for sample in post_grasp_verify_samples
                            ]))
                            depth_changes = [
                                sample["depth_change_m"] for sample in post_grasp_verify_samples
                                if sample["depth_change_m"] is not None
                            ]
                            median_depth_change_m = (
                                None if not depth_changes else float(np.median(depth_changes))
                            )
                            area_stable = median_area_fraction_change <= args.post_grasp_verify_area_change_fraction
                            depth_stable = (
                                median_depth_change_m is not None
                                and median_depth_change_m <= args.post_grasp_verify_depth_change_m
                            )
                            held_with_gripper = (
                                median_center_delta_px <= args.post_grasp_verify_center_drift_px
                                # A correctly gripped hinge is frequently partly hidden
                                # by the closing fingers.  Its YOLO mask area may change
                                # sharply, whereas its camera-relative depth remains
                                # stable.  A part left on the table fails both checks
                                # after the controlled +Z verification lift.
                                and (area_stable or depth_stable)
                            )
                            verification = {
                                "lift_m": args.post_grasp_verify_lift_m,
                                "frames": args.post_grasp_verify_frames,
                                "median_center_delta_px": median_center_delta_px,
                                "max_center_delta_px": args.post_grasp_verify_center_drift_px,
                                "median_area_fraction_change": median_area_fraction_change,
                                "max_area_fraction_change": args.post_grasp_verify_area_change_fraction,
                                "median_depth_change_m": median_depth_change_m,
                                "max_depth_change_m": args.post_grasp_verify_depth_change_m,
                                "area_stable": area_stable,
                                "depth_stable": depth_stable,
                                "held_with_gripper": held_with_gripper,
                            }
                            if not held_with_gripper:
                                assert position_stream is not None and gripper_open is not None and gripper_tolerance is not None
                                # The verification lift already created the
                                # requested clearance.  Re-open and restart
                                # XY alignment from there; do not add the
                                # retired 100-mm visual-recovery lift.
                                position_stream, last_end_link_target, recovery = recover_for_visual_regrasp(
                                    controller,
                                    position_stream,
                                    retract_m=0.0,
                                    translation_speed_mps=parameters.final_descent_speed_mps,
                                    fixed_orientation_xyzw=last_end_link_target.orientation_xyzw,
                                    gripper_open_rad=gripper_open,
                                    gripper_tolerance_rad=gripper_tolerance,
                                    gripper_timeout_s=gripper_close_timeout,
                                )
                                regrasp_attempts += 1
                                state.reset_for_regrasp()
                                tracked_center = tracked_depth_m = None
                                missed_frames = 0
                                gripper_sent_at = adaptive_hold_target_rad = adaptive_hold_sent_at = None
                                gripper_position_samples.clear()
                                post_grasp_verify_baseline = None
                                post_grasp_verify_samples.clear()
                                post_grasp_verified = False
                                record.update({
                                    "state": "regrasp_reacquiring",
                                    "reason": "post_grasp_visual_verification_failed",
                                    "regrasp_attempt": regrasp_attempts,
                                    "post_grasp_verify": verification,
                                    "regrasp_recovery": recovery,
                                })
                                json_record(log, record)
                                print_compact_record(record)
                                continue
                            post_grasp_verify_baseline = None
                            post_grasp_verify_samples.clear()
                            post_grasp_verified = True
                            record["post_grasp_verify"] = verification
                        # Continue observing the same target while the jaw is
                        # closing.  A loss at this point is treated as a
                        # failed attempt, never as permission to transport an
                        # assumed grasp.  The worker is still using the one
                        # SAM3 model loaded before P2/P3.
                        if active_regrasp_mailbox is not None:
                            close_tracking = active_regrasp_mailbox.newer_than(active_regrasp_mailbox_sequence)
                            if close_tracking is not None:
                                active_regrasp_mailbox_sequence = int(close_tracking["sequence"])
                                close_tracking_failure: str | None = None
                                if close_tracking.get("valid") is False:
                                    close_tracking_failure = "locked_target_lost_while_gripper_closing"
                                elif float(np.linalg.norm(np.asarray(
                                    close_tracking.get("pixel_error_px", (0.0, 0.0)), dtype=np.float64
                                ))) > 2.0 * parameters.pixel_tolerance:
                                    # At the point closure starts the target
                                    # has already passed the multi-frame XY
                                    # centering check.  A subsequent large
                                    # image displacement means the part has
                                    # moved or dropped; do not treat the jaw
                                    # stall alone as a successful grasp.
                                    close_tracking_failure = "locked_target_moved_while_gripper_closing"
                                if close_tracking_failure is not None:
                                    assert position_stream is not None and gripper_open is not None and gripper_tolerance is not None
                                    stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                                    force_preview_stop = force_preview_thread = None
                                    force_preview_phase = None
                                    position_stream, last_end_link_target, recovery = recover_for_visual_regrasp(
                                        controller,
                                        position_stream,
                                        retract_m=args.regrasp_retract_m,
                                        translation_speed_mps=parameters.final_descent_speed_mps,
                                        fixed_orientation_xyzw=last_end_link_target.orientation_xyzw,
                                        gripper_open_rad=gripper_open,
                                        gripper_tolerance_rad=gripper_tolerance,
                                        gripper_timeout_s=gripper_close_timeout,
                                    )
                                    regrasp_attempts += 1
                                    state.reset_for_regrasp()
                                    tracked_center = tracked_depth_m = None
                                    missed_frames = 0
                                    gripper_sent_at = adaptive_hold_target_rad = adaptive_hold_sent_at = None
                                    gripper_position_samples.clear()
                                    active_regrasp_mailbox = None
                                    record.update({
                                        "state": "regrasp_reacquiring",
                                        "reason": close_tracking_failure,
                                        "regrasp_attempt": regrasp_attempts,
                                        "regrasp_recovery": recovery,
                                    })
                                    json_record(log, record)
                                    print_compact_record(record)
                                    continue
                        position = controller.read_gripper_position("left")
                        record["gripper_position_rad"] = position
                        now = time.monotonic()
                        if position is not None:
                            gripper_position_samples.append((now, position))
                            # Keep the baseline sample preceding the stability
                            # window, even at the sparse SAM-loop frame rate.
                            history_limit_s = adaptive_gripper.stall_window_s + max(
                                1.0, adaptive_gripper.stall_window_s
                            )
                            gripper_position_samples[:] = [
                                sample for sample in gripper_position_samples
                                if now - sample[0] <= history_limit_s
                            ]
                        if position is not None and adaptive_hold_target_rad is None:
                            close_travel_rad = position - gripper_open
                            record["adaptive_gripper_close_travel_rad"] = close_travel_rad
                            # Near-zero is the documented fully closed end.
                            # It means the jaws did not encounter a hinge, so
                            # it is a failed attempt, not a reason to
                            # transport an empty gripper.  Retract/open and
                            # return to continuous visual alignment instead.
                            if abs(position - adaptive_gripper.seek_target_rad) <= gripper_tolerance:
                                assert position_stream is not None
                                stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                                force_preview_stop = force_preview_thread = None
                                force_preview_phase = None
                                position_stream, last_end_link_target, recovery = recover_for_visual_regrasp(
                                    controller,
                                    position_stream,
                                    retract_m=args.regrasp_retract_m,
                                    translation_speed_mps=parameters.final_descent_speed_mps,
                                    fixed_orientation_xyzw=last_end_link_target.orientation_xyzw,
                                    gripper_open_rad=gripper_open,
                                    gripper_tolerance_rad=gripper_tolerance,
                                    gripper_timeout_s=gripper_close_timeout,
                                )
                                regrasp_attempts += 1
                                state.reset_for_regrasp()
                                tracked_center = tracked_depth_m = None
                                missed_frames = 0
                                gripper_sent_at = adaptive_hold_target_rad = adaptive_hold_sent_at = None
                                gripper_position_samples.clear()
                                record.update({
                                    "state": "regrasp_reacquiring",
                                    "reason": "empty_gripper_full_close",
                                    "regrasp_attempt": regrasp_attempts,
                                    "regrasp_recovery": recovery,
                                })
                                json_record(log, record)
                                print_compact_record(record)
                                continue
                            if close_travel_rad >= adaptive_gripper.minimum_close_travel_rad:
                                stall_angle = stalled_gripper_position(
                                    gripper_position_samples,
                                    now=now,
                                    window_s=adaptive_gripper.stall_window_s,
                                    position_delta_rad=adaptive_gripper.stall_position_delta_rad,
                                )
                                if stall_angle is not None:
                                    adaptive_hold_target_rad = min(
                                        adaptive_gripper.seek_target_rad,
                                        stall_angle + adaptive_gripper.additional_close_rad,
                                    )
                                    assert position_stream is not None
                                    position_stream.set_left_gripper_target(adaptive_hold_target_rad)
                                    adaptive_hold_sent_at = now
                                    record["adaptive_gripper_stall"] = {
                                        "stall_angle_rad": stall_angle,
                                        "hold_target_rad": adaptive_hold_target_rad,
                                        "additional_close_rad": adaptive_gripper.additional_close_rad,
                                        "window_s": adaptive_gripper.stall_window_s,
                                        "position_delta_rad": adaptive_gripper.stall_position_delta_rad,
                                    }
                        if (
                            position is not None
                            and adaptive_hold_target_rad is not None
                            and adaptive_hold_sent_at is not None
                            and now - adaptive_hold_sent_at >= adaptive_gripper.stall_window_s
                            and abs(position - adaptive_hold_target_rad) <= gripper_tolerance
                        ):
                            record["grasp_confirmed"] = True
                            record["adaptive_gripper_hold_target_rad"] = adaptive_hold_target_rad
                            if not post_grasp_verified:
                                assert position_stream is not None
                                # Stop the display worker before the outer
                                # loop resumes camera reads for verification.
                                stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                                force_preview_stop = force_preview_thread = None
                                force_preview_phase = None
                                verify_origin = measured_left_end_link_pose(controller)
                                verify_target = CartesianPoseTarget(
                                    base_position_after_delta(
                                        verify_origin.position_m,
                                        np.asarray((0.0, 0.0, args.post_grasp_verify_lift_m)),
                                    ),
                                    last_end_link_target.orientation_xyzw,
                                )
                                position_stream.update_target(verify_target)
                                verify_timeout_s = max(
                                    1.0,
                                    args.post_grasp_verify_lift_m / parameters.final_descent_speed_mps + 0.5,
                                )
                                post_grasp_verify_motion = {
                                    "target": verify_target,
                                    "deadline": time.monotonic() + verify_timeout_s,
                                    "baseline": {
                                        "center_px": [float(value) for value in item["center_px"]],
                                        "area_px": int(item["area_px"]),
                                        "depth_median_m": item.get("depth_median_m"),
                                    },
                                }
                                record.update({
                                    "state": "post_grasp_verify_lift_in_flight",
                                    "reason": "nonblocking_lift_before_same_target_verification",
                                    "post_grasp_verify": {
                                        "baseline": post_grasp_verify_motion["baseline"],
                                        "lift_m": args.post_grasp_verify_lift_m,
                                        "frames": args.post_grasp_verify_frames,
                                    },
                                })
                                json_record(log, record)
                                print_compact_record(record)
                                continue
                            # Post-grasp verification needs the outer loop to
                            # own the camera briefly.  Reuse the same already
                            # loaded SAM3 worker for the subsequent +Z lift,
                            # but now hand its locked-target observations to
                            # the lift monitor as well as the web preview.
                            if args.full_cycle:
                                stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                                force_preview_stop = force_preview_thread = None
                                force_preview_phase = {
                                    "state": "post_grasp_monitored_lift",
                                    "overlay_note": (
                                        "same-target held-part monitoring during +Z lift; "
                                        "confirmed loss or geometric change cancels transport and regraspes; stale holds closed"
                                    ),
                                }
                                transport_mailbox = SnapshotVisualMailbox(
                                    perception, initial_center_px=item["center_px"],
                                    track_max_distance_px=args.track_max_distance_px,
                                    target_lost_frames=args.final_descent_target_lost_frames,
                                )
                                perception.set_phase("post_grasp_monitored_lift", force_preview_phase["overlay_note"])
                            json_record(log, record)
                            print_compact_record(record)
                            if args.full_cycle:
                                assert post_grasp_presets is not None
                                assert position_stream is not None
                                # The first transport motion is not a blind,
                                # blocking helper call.  Keep the existing
                                # 50-Hz mode-1 stream alive and inspect every
                                # fresh same-target SAM3 observation while it
                                # raises 100 mm.  A dropped hinge is therefore
                                # a regrasp event, never permission to request
                                # P4.
                                lift_origin = measured_left_end_link_pose(controller)
                                lift_target = CartesianPoseTarget(
                                    base_position_after_delta(
                                        lift_origin.position_m,
                                        np.asarray((0.0, 0.0, PRE_P4_LIFT_M)),
                                    ),
                                    last_end_link_target.orientation_xyzw,
                                )
                                position_stream.update_target(lift_target)
                                json_record(log, {
                                    "event": "post_grasp_monitored_lift_start",
                                    "utc": datetime.now(timezone.utc).isoformat(),
                                    "gripper": "confirmed and preserved while same-target monitoring is healthy",
                                    "pre_p4_lift_m": PRE_P4_LIFT_M,
                                    "monitor_center_drift_px": args.post_grasp_verify_center_drift_px,
                                    "monitor_area_change_fraction": args.post_grasp_verify_area_change_fraction,
                                    "presets": [name for name, _ in post_grasp_presets],
                                })
                                post_grasp_lift_monitor = {
                                    "origin": lift_origin,
                                    "target": lift_target,
                                    "mailbox": transport_mailbox,
                                    # Do not replay the frame that was used
                                    # for grasp verification as a transport
                                    # observation.
                                    "mailbox_sequence": perception_sequence,
                                    "reference_center_px": [float(value) for value in item["center_px"]],
                                    "reference_area_px": int(item["area_px"]),
                                    "last_observation_at": time.monotonic(),
                                    "stale_since": None,
                                    "stale_logged": False,
                                    "paused": False,
                                    "deadline": time.monotonic() + PRE_P4_LIFT_M / parameters.final_descent_speed_mps + 1.0,
                                }
                                # The next main-loop ticks poll this target;
                                # do not wait here.
                                continue
                            print("抓取已确认；保持当前位置和夹爪状态，未自动回预设位。")
                            return 0
                        assert gripper_close_timeout is not None
                        if now - gripper_sent_at > gripper_close_timeout:
                            assert position_stream is not None and gripper_open is not None and gripper_tolerance is not None
                            stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                            force_preview_stop = force_preview_thread = None
                            force_preview_phase = None
                            position_stream, last_end_link_target, recovery = recover_for_visual_regrasp(
                                controller,
                                position_stream,
                                retract_m=args.regrasp_retract_m,
                                translation_speed_mps=parameters.final_descent_speed_mps,
                                fixed_orientation_xyzw=last_end_link_target.orientation_xyzw,
                                gripper_open_rad=gripper_open,
                                gripper_tolerance_rad=gripper_tolerance,
                                gripper_timeout_s=gripper_close_timeout,
                            )
                            regrasp_attempts += 1
                            state.reset_for_regrasp()
                            tracked_center = tracked_depth_m = None
                            missed_frames = 0
                            gripper_sent_at = adaptive_hold_target_rad = adaptive_hold_sent_at = None
                            gripper_position_samples.clear()
                            record.update({
                                "state": "regrasp_reacquiring",
                                "reason": "gripper_no_variable_stall_before_timeout",
                                "regrasp_attempt": regrasp_attempts,
                                "regrasp_recovery": recovery,
                            })
                            json_record(log, record)
                            print_compact_record(record)
                            continue
                    record["inference_ms"] = round((time.monotonic() - started) * 1000.0, 1)
                    json_record(log, record)
                    print_compact_record(record)
                except Exception as exc:
                    record.update({"state": "error_stop_sending_commands", "error": str(exc)})
                    json_record(log, record)
                    print_compact_record(record, file=sys.stderr)
                    return 1
        return 0
    except KeyboardInterrupt:
        print("收到 Ctrl-C；停止发送后续命令。")
        return 0
    except Exception as exc:
        print(f"左手 G2 抓取流程失败：{exc}", file=sys.stderr)
        return 1
    finally:
        stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
        if perception is not None:
            perception.stop()
        if position_stream is not None:
            position_stream.stop()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
