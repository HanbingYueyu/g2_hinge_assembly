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
from dataclasses import dataclass
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
    from .record_pose import DEFAULT_OUTPUT as DEFAULT_POSES_OUTPUT
    from .run_left_premove_sequence import load_presets as load_premove_presets
    from .run_left_post_grasp_sequence import lift_before_p4, load_presets as load_post_grasp_presets
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
    from record_pose import DEFAULT_OUTPUT as DEFAULT_POSES_OUTPUT
    from run_left_premove_sequence import load_presets as load_premove_presets
    from run_left_post_grasp_sequence import lift_before_p4, load_presets as load_post_grasp_presets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_SCALE_M = 0.001
# GDK's kLeftHandRGBDToArmLEndLink external calibration terminates here.  GDK
# mode-1 end-effector pose control uses the same flange link by default.
LEFT_VISUAL_TIP_LINK = "arm_l_end_link"
# SAM3 inference itself sets the cadence (about a few Hz on the current GPU).
# Do not add a material artificial delay between display-only descent frames.
FORCE_DESCENT_SAM_PREVIEW_PERIOD_S = 0.01


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


def json_record(stream, payload: dict) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


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
        "tracking": record.get("tracking"),
        "sam_center_crop_xyxy": list(crop_bounds),
        "sam_center_crop_fraction": 0.70,
        "grasp_reason": record.get("reason"),
        "motion_enabled": record["motion_enabled"],
        "safety": "网页仅显示本抓取进程的同一帧 SAM3 结果；不提供机器人或夹爪控制接口。",
    })
    return sequence


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
):
    """Run the already-loaded SAM3 model during force-guarded descent.

    The force loop remains in the caller at 50 Hz and never reads this
    worker's result.  This worker owns the single preloaded SAM3 processor
    while the main loop is in force-only descent, so it does not load a second
    model and its display-only masks cannot alter motion or gripper decisions.
    """
    if shared is None:
        return None, None
    stop = threading.Event()

    def run() -> None:
        sequence = time.monotonic_ns()
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
                selected = choose_candidate(statistics, candidates, selection)
                overlay = draw_overlay(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), masks, statistics, selected)
                overlay = draw_crop_box(overlay, crop_bounds)
                cv2.putText(
                    overlay,
                    "SAM3 live - display only during force-contact descent",
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if not ok:
                    raise RuntimeError("无法编码下探期间的 SAM3 叠加图")
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
                    "safety": "SAM3 仅用于网页显示；力监测与末端控制仍在独立的 20 ms/50 Hz 循环中。",
                })
            except Exception as exc:
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
    log_path = args.log or PROJECT_ROOT / "logs" / f"left_grasp_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl"
    log_path = log_path.expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "mode": "motion" if args.allow_motion else "read_only_preview",
        "config": str(args.config),
        "log": str(log_path),
        "selection": args.selection,
        "full_cycle": args.full_cycle,
        "poses": None if not args.full_cycle else str(args.poses.expanduser().resolve()),
        "safety": (
            "single process: preload SAM3, then P2/P3→visual grasp→+Z 100 mm→P4; "
            "one GDK initialization and no extra gripper command after grasp"
            if args.full_cycle else "P2/P3/P4 are not commanded by this program"
        ),
    }, ensure_ascii=False))

    initialized = False
    camera = None
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
        for _ in range(args.warmup):
            camera.get_latest_image(camera_type, 1000.0)
        tf = agibot_gdk.TF()
        model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=args.threshold)
        state = None if parameters is None else LeftGraspState(parameters)
        color_intrinsic = depth_intrinsic = depth_to_color = end_link_transform = None
        tracked_center: np.ndarray | None = None
        tracked_depth_m: float | None = None
        missed_frames = 0
        last_end_link_target: CartesianPoseTarget | None = None
        gripper_sent_at: float | None = None
        adaptive_hold_target_rad: float | None = None
        adaptive_hold_sent_at: float | None = None
        gripper_position_samples: list[tuple[float, float]] = []
        contact_baseline_force_n: np.ndarray | None = None
        contact_detector: ForceContactDetector | None = None
        single_xy_verification: dict | None = None
        frame = 0
        web_sequence = 0
        with log_path.open("a", encoding="utf-8") as log:
            json_record(log, {
                "event": "start",
                "utc": datetime.now(timezone.utc).isoformat(),
                "mode": "motion" if args.allow_motion else "read_only_preview",
                "config": config,
                "full_cycle": args.full_cycle,
                "sam3_preloaded_before_premove": args.full_cycle,
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
                run_recorded_left_presets(
                    controller,
                    premove_presets,
                    velocity_rad_s=args.preset_velocity_rad_s,
                    arrival_tolerance_rad=args.preset_arrival_tolerance_rad,
                    timeout_s=args.preset_timeout_s,
                    stage="premove",
                    log=log,
                )
                mode_before_visual = controller.motion_status_summary()
                # GDK joint planning accepts modes 1 and 3. On this robot
                # the P2/P3 request leaves control_mode=3, whereas the
                # documented end_effector_pose_control visual stream requires
                # mode 1. This switch is a control-state transition only: it
                # sends no arm, end-pose, or gripper target.
                controller.set_control_mode(1)
                mode_after_visual = controller.motion_status_summary()
                if mode_after_visual["control_mode"] != 1 or mode_after_visual["error_code"] != 0:
                    raise SafetyError(
                        f"P3 后未进入健康 control_mode=1；拒绝张开夹爪或开始视觉伺服：{mode_after_visual}"
                    )
                json_record(log, {
                    "event": "control_mode_ready_for_visual_servo",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "before": mode_before_visual,
                    "after": mode_after_visual,
                    "safety": "mode transition only; no arm, end-pose, or gripper target is sent here",
                })
                json_record(log, {
                    "event": "premove_complete",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "final_detection_preset": "left_p3",
                })
                print("P2→P3 已到位且已确认 control_mode=1；SAM3 已预加载，开始同一进程内的视觉抓取。")
            # A supervised XY/depth test must not alter the gripper state.
            # Opening belongs only to a complete, calibrated grasp sequence.
            if args.allow_motion and args.stop_after == "none":
                assert gripper_open is not None and gripper_tolerance is not None and gripper_close_timeout is not None
                json_record(log, {
                    "event": "gripper_open_start",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "target_rad": gripper_open,
                    "tolerance_rad": gripper_tolerance,
                    "timeout_s": gripper_close_timeout,
                })
                open_event = open_left_gripper_with_feedback(
                    controller,
                    target_rad=gripper_open,
                    tolerance_rad=gripper_tolerance,
                    timeout_s=gripper_close_timeout,
                )
                open_event["utc"] = datetime.now(timezone.utc).isoformat()
                json_record(log, open_event)
                print(
                    "左夹爪已确认张开："
                    f"actual={open_event['actual_position_rad']:.6f} rad，"
                    f"attempts={open_event['attempts']}；开始视觉伺服。"
                )
            while args.max_frames == 0 or frame < args.max_frames:
                frame += 1
                started = time.monotonic()
                record: dict = {"event": "frame", "frame": frame, "utc": datetime.now(timezone.utc).isoformat(), "motion_enabled": args.allow_motion}
                try:
                    color = camera.get_latest_image(camera_type, 1000.0)
                    depth = camera.get_nearest_image(depth_type, int(color.timestamp_ns), 1000.0)
                    color_rgb = decode_color_rgb(color, agibot_gdk)
                    depth_u16 = decode_depth_u16(depth, agibot_gdk)
                    if color_intrinsic is None:
                        color_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(camera_type), width=int(color.width), height=int(color.height))
                        depth_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height))
                        depth_to_color = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor))
                        end_link_transform = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink))
                    aligned_depth = align_depth_to_color(depth_u16, depth_intrinsic=depth_intrinsic, color_intrinsic=color_intrinsic, depth_to_color=depth_to_color, depth_scale_m=DEPTH_SCALE_M)
                    sam_input, crop_bounds = center_crop_image(color_rgb)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        image_state = processor.set_image(Image.fromarray(sam_input))
                        result = processor.set_text_prompt(args.prompt, image_state)
                    crop_masks = [np.asarray(mask, dtype=bool).squeeze() for mask in result["masks"].detach().cpu().numpy()]
                    masks = expand_crop_masks(crop_masks, full_shape=color_rgb.shape[:2], bounds=crop_bounds)
                    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
                    statistics = [stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform) for mask in masks]
                    for item, score in zip(statistics, scores):
                        item["sam3_confidence"] = float(score)
                    roi = parse_roi(args.roi, color_rgb.shape[:2])
                    candidates = eligible_candidates(statistics, min_area=args.min_area, min_valid_depth_pixels=args.min_valid_depth_pixels, roi=roi, require_depth=True)
                    record["eligible_candidate_indices"] = [int(index) for index in candidates]
                    record["sam_center_crop_xyxy"] = list(crop_bounds)
                    record["sam_center_crop_fraction"] = 0.70
                    record["candidate_diagnostics"] = [
                        {
                            "mask_index": int(index),
                            "center_px": statistics[index].get("center_px"),
                            "area_px": statistics[index].get("area_px"),
                            "valid_depth_px": statistics[index].get("valid_depth_px"),
                            "depth_median_m": statistics[index].get("depth_median_m"),
                            "sam3_confidence": statistics[index].get("sam3_confidence"),
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
                            selected_depth_m = float(statistics[selected]["depth_median_m"])
                            depth_outlier = (
                                tracked_depth_m is not None
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
                                if selected is not None:
                                    tracked_depth_m = float(statistics[selected]["depth_median_m"])
                    else:
                        selected = choose_candidate(statistics, candidates, args.selection)
                        tracking = "acquired" if selected is not None else "no_target"
                    if selected is not None and tracked_center is None:
                        tracked_center = np.asarray(statistics[selected]["center_px"], dtype=np.float64)
                        tracked_depth_m = float(statistics[selected]["depth_median_m"])
                    record.update({
                        "tracking": tracking,
                        "tracking_depth_anchor_m": tracked_depth_m,
                        "tracking_depth_max_jump_m": args.track_max_depth_jump_m,
                        "depth_outlier_candidate_indices": [int(index) for index in depth_jump_rejected],
                        "selected_mask_index": selected,
                        "mask_count": len(statistics),
                        "selected": None if selected is None else statistics[selected],
                    })
                    if selected is None or statistics[selected].get("valid_depth_px", 0) < args.min_valid_depth_pixels:
                        if args.allow_motion and state is not None and state.phase in {GraspPhase.APPROACH_DEPTH, GraspPhase.FINAL_DESCENT}:
                            raise SafetyError("竖直下探前丢失带有效深度的目标；停止且不闭爪")
                        record["state"] = "waiting_for_selected_depth"
                        web_sequence = publish_web_preview(web_shared, sequence=web_sequence, color_rgb=color_rgb, masks=masks, statistics=statistics, selected=selected, record=record)
                        json_record(log, record)
                        print(json.dumps(record, ensure_ascii=False))
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
                        print(json.dumps(record, ensure_ascii=False))
                        print("单次 XY 受看护测试完成：已确认末端反馈并复测同一视觉目标；未下探、未操作夹爪。")
                        return 0
                    if state is None:
                        record.update({"state": "calibration_required", "reason": "complete left_grasp.example.json before a descent/grasp plan can be produced"})
                        web_sequence = publish_web_preview(web_shared, sequence=web_sequence, color_rgb=color_rgb, masks=masks, statistics=statistics, selected=selected, record=record)
                        json_record(log, record)
                        print(json.dumps(record, ensure_ascii=False))
                        continue
                    if tracking == "reacquired":
                        if args.allow_motion and state.phase in {GraspPhase.APPROACH_DEPTH, GraspPhase.FINAL_DESCENT}:
                            raise SafetyError("竖直下探前目标重新获取；停止且不闭爪")
                        state.reset_alignment()
                        record["target_reacquired_reset_to_xy_alignment"] = True
                    observation = Observation(tuple(float(value) for value in item["center_px"]), float(item["depth_median_m"]), float(color_intrinsic["fx"]), float(color_intrinsic["fy"]), float(color_intrinsic["ppx"]), float(color_intrinsic["ppy"]))
                    step = state.update(observation, now=time.monotonic())
                    record.update({"state": step.phase.value, "reason": step.reason, "camera_delta_m": step.camera_delta_m, "base_delta_m": step.base_delta_m})
                    web_sequence = publish_web_preview(web_shared, sequence=web_sequence, color_rgb=color_rgb, masks=masks, statistics=statistics, selected=selected, record=record)
                    if args.stop_after == "xy" and step.phase == GraspPhase.FINAL_DESCENT:
                        record["test_stop"] = "xy_alignment_complete_before_force_guarded_descent"
                        json_record(log, record)
                        print(json.dumps(record, ensure_ascii=False))
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
                            if not position_stream.wait_until_target(1.0):
                                raise SafetyError("前一段末端目标尚未到达；拒绝开始最终下探")
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
                                    force_preview_phase = {
                                        "state": "force_contact_descent",
                                        "overlay_note": "live display only; it cannot alter force descent or gripper decisions",
                                    }
                                    force_preview_stop, force_preview_thread = start_force_descent_sam_preview(
                                        web_shared,
                                        camera=camera,
                                        color_type=camera_type,
                                        depth_type=depth_type,
                                        gdk=agibot_gdk,
                                        processor=processor,
                                        prompt=args.prompt,
                                        color_intrinsic=color_intrinsic,
                                        depth_intrinsic=depth_intrinsic,
                                        depth_to_color=depth_to_color,
                                        end_link_transform=end_link_transform,
                                        selection=args.selection,
                                        min_area=args.min_area,
                                        min_valid_depth_pixels=args.min_valid_depth_pixels,
                                        roi=roi,
                                        phase=force_preview_phase,
                                    )
                                    final_deadline = time.monotonic() + step.motion_duration_s + 0.25
                                    maximum_delta_force_n = 0.0
                                    latest_force_n: np.ndarray | None = None
                                    latest_hits = 0
                                    force_relief_completed = False
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
                                        time.sleep(contact_detector.parameters.sample_period_s)
                                    record["force_contact"] = {
                                        **record.get("force_contact", {}),
                                        "detected": force_contact_detected,
                                        "latest_force_n": None if latest_force_n is None else [float(value) for value in latest_force_n],
                                        "latest_delta_force_n": None if latest_force_n is None else float(np.linalg.norm(latest_force_n - contact_detector.baseline_force_n)),
                                        "maximum_delta_force_n": maximum_delta_force_n,
                                        "consecutive_hits": latest_hits,
                                    }
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
                            print(json.dumps(record, ensure_ascii=False))
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
                            print(json.dumps(record, ensure_ascii=False))
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
                            # do not report a successful empty grasp.
                            if abs(position - adaptive_gripper.seek_target_rad) <= gripper_tolerance:
                                raise SafetyError(
                                    "夹爪接近机械全闭合端但未检测到可变卡住角度；判定未夹到铰链，停止且不运输"
                                )
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
                            json_record(log, record)
                            print(json.dumps(record, ensure_ascii=False))
                            if args.full_cycle:
                                assert post_grasp_presets is not None
                                # The grasp stream continuously holds the
                                # flange and carries the synchronized gripper
                                # target. Stop it before starting the next
                                # independent mode-1 +Z stream; neither the
                                # lift nor the recorded P4 request sends
                                # any gripper command.
                                if position_stream is not None:
                                    position_stream.stop()
                                    position_stream = None
                                json_record(log, {
                                    "event": "post_grasp_transport_start",
                                    "utc": datetime.now(timezone.utc).isoformat(),
                                    "gripper": "confirmed and preserved; no further gripper request",
                                    "pre_p4_lift_m": 0.10,
                                    "presets": [name for name, _ in post_grasp_presets],
                                })
                                lift_event = lift_before_p4(controller)
                                lift_event["utc"] = datetime.now(timezone.utc).isoformat()
                                json_record(log, lift_event)
                                print("夹爪已确认；同一 GDK 进程中已上抬 100 mm，开始 P4。")
                                run_recorded_left_presets(
                                    controller,
                                    post_grasp_presets,
                                    velocity_rad_s=args.preset_velocity_rad_s,
                                    arrival_tolerance_rad=args.preset_arrival_tolerance_rad,
                                    timeout_s=args.preset_timeout_s,
                                    stage="post_grasp",
                                    log=log,
                                )
                                json_record(log, {
                                    "event": "full_cycle_complete",
                                    "utc": datetime.now(timezone.utc).isoformat(),
                                    "gripper": "preserved after confirmed grasp",
                                })
                                print("完整流程完成：P2→P3、抓取、+Z 上抬 100 mm、P4；夹爪保持闭合。")
                                return 0
                            print("抓取已确认；保持当前位置和夹爪状态，未自动回预设位。")
                            return 0
                        assert gripper_close_timeout is not None
                        if now - gripper_sent_at > gripper_close_timeout:
                            raise SafetyError(
                                "夹爪未在限定时间内形成有效的可变卡住角度；已停止发送后续命令，保持现场状态"
                            )
                    record["inference_ms"] = round((time.monotonic() - started) * 1000.0, 1)
                    json_record(log, record)
                    print(json.dumps(record, ensure_ascii=False))
                except Exception as exc:
                    record.update({"state": "error_stop_sending_commands", "error": str(exc)})
                    json_record(log, record)
                    print(json.dumps(record, ensure_ascii=False), file=sys.stderr)
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
