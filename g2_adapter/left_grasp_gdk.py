#!/usr/bin/env python3
"""Final-position left D405 visual grasp using GDK and YOLO only.

By default P2/P3 remain outside this program, so it can be used as a
final-detection-position-only grasp tool. ``--full-cycle`` makes this the
single GDK process for P1→P2→grasp→+Z lift→P4: YOLO runs on every
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
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from .left_step_xy import xy_error, bounded_xy
    from .bilateral_rgbd_geometry import apply_transform
    from .left_step_rgb import OcclusionTrackerPool
    from .apriltag_workpiece import transform_from_dict
    from .left_step_rgb import load_profile, load_alignment_reference, detect_hole_centres, match_holes
    from .left_step_height import (
        camera_distance_from_base_point, interpolate_target_pixel, load_height_samples,
        target_pixel_for_step_pose,
    )
    from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from .contact_detection import ForceContactDetector, ForceContactParameters
    from .control import CartesianPoseTarget, G2Controller, LeftEndEffectorPositionStream, quaternion_angle_deg
    from .left_grasp_state import GraspParameters, GraspPhase, LeftGraspState, Observation, requires_calibrated_config
    from .live_left_d405_sam3_web import CompositeSharedView, SharedView, handler_class
    from .localize_left_d405_sam3 import depth_points, draw_overlay, eligible_candidates, parse_roi, stats_for_mask
    from .move_left_above_head_hinge_gdk import (
        base_from_head_at, dominant_sample_cluster, fixed_forward_down_orientation,
        draw_head_coarse_overlay, draw_head_targets_overlay, gripper_end_target,
        select_highest_confidence_class, select_leftmost_hinge, wait_actual_left_target,
    )
    from .safety import SafetyError
    from .sam_crop import center_crop_bounds, center_crop_image, draw_crop_box, expand_crop_masks
    from .visual_servo import base_position_after_delta, empirical_horizontal_base_delta_from_pixel_error, relief_translation_speed_mps
    from .xy_mapping import choose_candidate_by_left_rank
    from .yolo_hinge import associate_shaft_to_hinge, infer_yolo_masks, load_yolo_segmentation_model, review_matches, reviewed_target_matches_selected, split_yolo_candidates
    from .record_pose import DEFAULT_OUTPUT as DEFAULT_POSES_OUTPUT
    from .workflow_home import WORKFLOW_HOME_POSE_NAME, load_workflow_home, move_workflow_home
    from .inspect_left_gripper_urdf import audit as audit_left_gripper_urdf
    from .run_left_premove_sequence import HANDOFF_PRESETS, load_presets as load_premove_presets, open_handoff_gripper
    from .run_left_post_grasp_sequence import (
        PRE_P4_LIFT_M,
        PRE_P4_LIFT_TOLERANCE_M,
        load_presets as load_post_grasp_presets,
    )
    from .spacemouse_takeover import AdvanceRightRequested, MotionPauseGate, SpaceMouseTakeoverWorker
    from .dianpi_runner import DianpiCycleRunner
    from .spacemouse_teleop import TeleopConfig
except ImportError:
    from left_step_xy import xy_error, bounded_xy
    from bilateral_rgbd_geometry import apply_transform
    from left_step_rgb import OcclusionTrackerPool
    from apriltag_workpiece import transform_from_dict
    from left_step_rgb import load_profile, load_alignment_reference, detect_hole_centres, match_holes
    from left_step_height import (
        camera_distance_from_base_point, interpolate_target_pixel, load_height_samples,
        target_pixel_for_step_pose,
    )
    from camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from contact_detection import ForceContactDetector, ForceContactParameters
    from control import CartesianPoseTarget, G2Controller, LeftEndEffectorPositionStream, quaternion_angle_deg
    from left_grasp_state import GraspParameters, GraspPhase, LeftGraspState, Observation, requires_calibrated_config
    from live_left_d405_sam3_web import CompositeSharedView, SharedView, handler_class
    from localize_left_d405_sam3 import depth_points, draw_overlay, eligible_candidates, parse_roi, stats_for_mask
    from move_left_above_head_hinge_gdk import (
        base_from_head_at, dominant_sample_cluster, fixed_forward_down_orientation,
        draw_head_coarse_overlay, draw_head_targets_overlay, gripper_end_target,
        select_highest_confidence_class, select_leftmost_hinge, wait_actual_left_target,
    )
    from safety import SafetyError
    from sam_crop import center_crop_bounds, center_crop_image, draw_crop_box, expand_crop_masks
    from visual_servo import base_position_after_delta, empirical_horizontal_base_delta_from_pixel_error, relief_translation_speed_mps
    from xy_mapping import choose_candidate_by_left_rank
    from yolo_hinge import associate_shaft_to_hinge, infer_yolo_masks, load_yolo_segmentation_model, review_matches, reviewed_target_matches_selected, split_yolo_candidates
    from record_pose import DEFAULT_OUTPUT as DEFAULT_POSES_OUTPUT
    from workflow_home import WORKFLOW_HOME_POSE_NAME, load_workflow_home, move_workflow_home
    from inspect_left_gripper_urdf import audit as audit_left_gripper_urdf
    from run_left_premove_sequence import HANDOFF_PRESETS, load_presets as load_premove_presets, open_handoff_gripper
    from run_left_post_grasp_sequence import (
        PRE_P4_LIFT_M,
        PRE_P4_LIFT_TOLERANCE_M,
        load_presets as load_post_grasp_presets,
    )
    from spacemouse_takeover import AdvanceRightRequested, MotionPauseGate, SpaceMouseTakeoverWorker
    from dianpi_runner import DianpiCycleRunner
    from spacemouse_teleop import TeleopConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERRUPTED_EXIT_CODE = 130
DEPTH_SCALE_M = 0.001
# GDK's kLeftHandRGBDToArmLEndLink external calibration terminates here.  GDK
# Joint-impedance end-effector pose control uses the same flange link by default.
LEFT_VISUAL_TIP_LINK = "arm_l_end_link"
# Camera/model inference sets the cadence. Do not add an artificial delay
# between display-only frames.
FORCE_DESCENT_SAM_PREVIEW_PERIOD_S = 0.01
# The left grasp only consumes the hinge class (class 0), so the one-class
# hinge weight is preferred here.  The two-class hinge/shaft weight stays on
# disk for run_bilateral_alignment_gdk.py (its post-grasp hole alignment needs
# class 1 "shaft") and as a rollback for this flow.
_PREFERRED_LEFT_HINGE_YOLO = PROJECT_ROOT / "models" / "left_d405_hinge_yolo" / "best.pt"
_ROLLBACK_LEFT_HINGE_SHAFT_YOLO = PROJECT_ROOT / "models" / "left_d405_hinge_shaft_yolo" / "best.pt"
DEFAULT_G2_NATIVE_YOLO = (
    _PREFERRED_LEFT_HINGE_YOLO
    if _PREFERRED_LEFT_HINGE_YOLO.is_file()
    else _ROLLBACK_LEFT_HINGE_SHAFT_YOLO
)
DEFAULT_G2_URDF = PROJECT_ROOT / "robot_description" / "G2_t2v2_crsB_omnipicker.urdf"
EXIT_ADVANCE_RIGHT = 20
HINGE_CLASS_ID = 0
SHAFT_CLASS_ID = 1
SCREEN_GRIPPER_TARGET_HEIGHT_FRACTION = 0.45
# The object grasp landmark is measured upward from the bottom of the YOLO
# box.  Thus 0.55 means 55% up from the box bottom (45% down from its top).
YOLO_GRASP_BOTTOM_UP_FRACTION = 0.55
YOLO_CENTER_CROP_FRACTION = 0.70


def prepare_yolo_input(
    color_rgb: np.ndarray,
    valid_fraction: float = YOLO_CENTER_CROP_FRACTION,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Prepare a full-frame YOLO input and the central valid-target ROI.

    YOLO must see the complete camera image so detections near the image edge
    are available for diagnostics.  Only detections whose center lies in the
    returned ROI are eligible for grasping or visual servoing.
    """
    image = np.asarray(color_rgb)
    if image.ndim < 3 or image.shape[2] != 3:
        raise ValueError("color_rgb must be an HxWx3 image")
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), center_crop_bounds(
        image.shape, fraction=valid_fraction,
    )


def yolo_box_fraction_grasp_pixel(
    item: dict, bottom_up_fraction: float = YOLO_GRASP_BOTTOM_UP_FRACTION,
) -> list[float]:
    """Return the box centre horizontally and a point measured up from bottom."""
    if not 0.0 <= bottom_up_fraction <= 1.0:
        raise ValueError("YOLO box bottom-up grasp fraction must be in [0, 1]")
    box = np.asarray(item.get("bbox_xyxy"), dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("YOLO grasp target requires a finite non-empty bbox_xyxy")
    return [
        float((box[0] + box[2]) / 2.0),
        float(box[3] - bottom_up_fraction * (box[3] - box[1])),
    ]


def screen_gripper_target_pixel(
    image_shape: tuple[int, ...], height_fraction: float = SCREEN_GRIPPER_TARGET_HEIGHT_FRACTION,
) -> np.ndarray:
    """Return the fixed image point where the gripper grasp point should appear.

    The default is 55% of the image height measured upward from the bottom,
    i.e. 45% down from the top.
    """
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0 or not 0.0 <= height_fraction <= 1.0:
        raise ValueError("invalid image shape or screen target height fraction")
    return np.asarray((width * 0.5, height * float(height_fraction)), dtype=np.float64)


def servo_pixel(item: dict) -> list[float]:
    """Use the YOLO box grasp landmark; legacy STEP data is ignored."""
    if "yolo_box_grasp_pixel_px" in item:
        return item["yolo_box_grasp_pixel_px"]
    return item["center_px"]


def urdf_gripper_center_end_m(path: Path, opening_rad: float) -> np.ndarray:
    """Use the midpoint of the two symmetric fixed loop anchors in the URDF."""
    report = audit_left_gripper_urdf(path, opening_rad)
    anchors = [
        np.asarray(loop["anchor1_end_m"], dtype=np.float64)
        for loop in report["closed_loop_checks"]
        if loop["joint"] in {"idx53_gripper_l_outer_joint2", "idx54_gripper_l_inner_joint2"}
    ]
    if len(anchors) != 2 or any(anchor.shape != (3,) or not np.isfinite(anchor).all() for anchor in anchors):
        raise ValueError("URDF does not provide both finite left gripper loop anchors")
    return np.mean(anchors, axis=0)


def project_end_point_to_color_pixel(point_end_m: np.ndarray, camera_to_end: dict, intrinsic: dict) -> list[float]:
    transform = transform_from_dict(camera_to_end)
    point_camera = transform[:3, :3].T @ (np.asarray(point_end_m, dtype=np.float64) - transform[:3, 3])
    if point_camera.shape != (3,) or not np.isfinite(point_camera).all() or point_camera[2] <= 0.0:
        raise ValueError("URDF gripper center projects behind the left color camera")
    return [
        float(intrinsic["fx"] * point_camera[0] / point_camera[2] + intrinsic["ppx"]),
        float(intrinsic["fy"] * point_camera[1] / point_camera[2] + intrinsic["ppy"]),
    ]


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


@dataclass(frozen=True)
class GripperCloseDecision:
    """Decision made from one real gripper feedback sample and its history."""

    action: str
    reason: str
    hold_target_rad: float | None = None
    close_travel_rad: float | None = None
    stall_angle_rad: float | None = None


def decide_gripper_close(
    *,
    position_rad: float | None,
    gripper_open_rad: float,
    tolerance_rad: float,
    parameters: AdaptiveGripperCloseParameters,
    samples: list[tuple[float, float]],
    now: float,
) -> GripperCloseDecision:
    """Choose whether to keep closing, or hold a confirmed close target.

    Reaching the configured mechanical close endpoint is not proof that the
    hinge was missed: a thin hinge may still be held when the reported angle
    is near ``seek_target_rad``.  Keep that endpoint closed and let the
    post-grasp visual/lift monitor decide whether the part is actually held.
    """
    if position_rad is None:
        return GripperCloseDecision(action="wait", reason="gripper_feedback_unavailable")

    close_travel_rad = float(position_rad) - float(gripper_open_rad)
    if abs(float(position_rad) - parameters.seek_target_rad) <= tolerance_rad:
        return GripperCloseDecision(
            action="hold",
            reason="full_close_hold_pending_visual_verification",
            hold_target_rad=parameters.seek_target_rad,
            close_travel_rad=close_travel_rad,
        )

    if close_travel_rad < parameters.minimum_close_travel_rad:
        return GripperCloseDecision(
            action="wait",
            reason="minimum_close_travel_not_reached",
            close_travel_rad=close_travel_rad,
        )

    stall_angle = stalled_gripper_position(
        samples,
        now=now,
        window_s=parameters.stall_window_s,
        position_delta_rad=parameters.stall_position_delta_rad,
    )
    if stall_angle is None:
        return GripperCloseDecision(
            action="wait",
            reason="waiting_for_variable_stall",
            close_travel_rad=close_travel_rad,
        )

    hold_target_rad = min(
        parameters.seek_target_rad,
        stall_angle + parameters.additional_close_rad,
    )
    return GripperCloseDecision(
        action="hold",
        reason="variable_stall_hold",
        hold_target_rad=hold_target_rad,
        close_travel_rad=close_travel_rad,
        stall_angle_rad=stall_angle,
    )


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
    check_geometry: bool = True,
) -> dict | None:
    """Return a transport-veto record when the locked held part is unhealthy.

    This is deliberately a visual *continuity* check, not a new detection or
    a request to substitute another hinge.  During transport the only valid
    object is the one that was verified immediately after the grasp.  Callers
    may disable geometry rejection with ``check_geometry=False`` when mask
    occlusion is expected and only timeout-confirmed loss is actionable.
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
    # Once the gripper has closed, changes in the visible mask are expected:
    # the fingers occlude the part and the view changes during lifting.  The
    # caller can therefore defer the only actionable decision to the explicit
    # target-lost timeout; verification metrics remain diagnostic only.
    if not check_geometry:
        return None
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


def held_part_gate_step(
    monitor: dict,
    observation: dict | None,
    *,
    now: float,
    max_stale_s: float,
    center_drift_px: float,
    area_change_fraction: float,
    check_geometry: bool = False,
) -> str:
    """Advance one post-grasp "is the part still held?" gate step.

    Mirrors the P3/P4 gate: a fresh same-target frame clears
    ``awaiting_fresh_observation``; only a timeout-confirmed loss is
    actionable; stale perception holds position instead of moving blind.

    Returns ``"proceed"`` (part confirmed still held), ``"lost"`` (confirmed
    loss), ``"stale"`` (perception went stale on this step) or ``"wait"``.
    ``monitor`` is updated in place.
    """
    if observation is not None:
        monitor["failure"] = held_part_monitor_failure(
            observation,
            reference_center_px=monitor["reference_center_px"],
            reference_area_px=monitor["reference_area_px"],
            max_center_drift_px=center_drift_px,
            max_area_change_fraction=area_change_fraction,
            check_geometry=check_geometry,
        )
        # A fresh frame with valid=None is only a temporary miss, not
        # affirmative evidence that the hinge is still in the gripper.
        if observation.get("valid") is True:
            monitor["awaiting_fresh_observation"] = False
        monitor["paused"] = False
        monitor["stale_since"] = None
        monitor["last_observation_at"] = now
    if monitor.get("failure") is not None:
        return "lost"
    if not monitor.get("awaiting_fresh_observation"):
        return "proceed"
    if observation is None and not monitor.get("paused") and now - monitor["last_observation_at"] > max_stale_s:
        monitor["paused"] = True
        monitor["stale_since"] = now
        return "stale"
    return "wait"


HELD_PART_AREA_UNSTABLE_REASON = "held_target_mask_area_unstable_during_lift"
def held_part_area_instability(
    samples: Sequence[tuple[float, float]],
    *,
    max_spread: float,
    window_frames: int,
) -> dict | None:
    """Judge a closed-gripper lift from the trailing mask-area history.

    A confirmed close is not proof that the hinge is actually inside the jaws:
    an empty grasp still shows a tracked mask.  The distinguishing signal
    during transport is motion of the part *relative to the gripper*.

    The left wrist camera moves with the gripper, so a part that is really
    held keeps a nearly constant apparent mask area, while a part that was
    never gripped (or is sliding out) keeps changing size frame after frame.
    Areas normally drift right after the close, so only the trailing
    latest ``window_frames`` are judged and the caller feeds this helper only
    after its grace period.

    Returns ``None`` until the trailing window holds ``window_frames``
    finite, positive areas.  Otherwise returns the measured spread together
    with ``unstable`` telling whether it exceeds ``max_spread``.
    """
    valid = [
        (float(timestamp), float(area))
        for timestamp, area in samples
        if math.isfinite(float(area))
        and float(area) > 0.0
    ]
    required = max(1, int(window_frames))
    if len(valid) < required:
        return None
    window = valid[-required:]
    areas = [area for _, area in window]
    median_area = float(np.median(areas))
    spread_fraction = (max(areas) - min(areas)) / median_area
    return {
        "reason": HELD_PART_AREA_UNSTABLE_REASON,
        "unstable": spread_fraction > max_spread,
        "spread_fraction": spread_fraction,
        "max_area_px": max(areas),
        "min_area_px": min(areas),
        "median_area_px": median_area,
        "latest_area_px": areas[-1],
        "samples": len(window),
        "window_frames": required,
        "max_spread": max_spread,
    }


@dataclass
class NonBlockingLeftPresetSequence:
    """Send one recorded joint target at a time and poll it from the main loop.

    Unlike the legacy helper this class never sleeps or waits for an arrival.
    It lets the web/SAM worker and safety monitor continue while P1/P2/P3/P4
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2 左手最终检测位 D405/YOLO 闭环抓取")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "left_grasp.example.json")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_G2_URDF, help="兼容旧命令；当前视觉对准不使用 URDF 夹爪中心。")
    parser.add_argument(
        "--prompt",
        default="silver triangular hinge",
        help="兼容旧命令的参数；YOLO 模式不使用。",
    )
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument(
        "--detector",
        choices=("yolo",),
        default="yolo",
        help="仅使用 YOLO 检测与跟踪。",
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
    parser.add_argument(
        "--yolo-center-fraction",
        type=float,
        default=YOLO_CENTER_CROP_FRACTION,
        help="左手 YOLO 全图推理时的中央有效目标区域宽高比例，默认 0.70（四周各排除 15%%）。",
    )
    parser.add_argument("--sam-review-interval-s", type=float, default=0.50, help="兼容旧命令；YOLO 模式不使用。")
    parser.add_argument("--sam-review-ttl-s", type=float, default=1.50, help="兼容旧命令；YOLO 模式不使用。")
    parser.add_argument("--sam-review-max-pixel-error", type=float, default=80.0, help="兼容旧命令；YOLO 模式不使用。")
    parser.add_argument(
        "--yolo-box-height-fraction",
        type=float,
        default=YOLO_GRASP_BOTTOM_UP_FRACTION,
        help="YOLO 框从底部向上比例，默认 0.55（55%%）；横坐标取检测框中心。",
    )
    parser.add_argument(
        "--height-reference",
        type=Path,
        default=None,
        help="10 cm/5 cm/抓取点三点像素标定；不依赖 STEP/PnP，使用头部三维点估计距离。",
    )
    parser.add_argument("--step-profile", type=Path, help="Use RGB STEP grasp point for XY alignment")
    parser.add_argument("--step-reference", type=Path, help="New manually aligned STEP pixel reference")
    parser.add_argument("--step-metric-xy", action="store_true", help="Use approximate TCP and image-time TF for base XY servo")
    parser.add_argument("--step-tcp-end-mm", default="0,0,145", help="Approximate TCP XYZ in arm_l_end_link, mm; not a calibrated TCP")
    parser.add_argument("--step-compensation-xy-mm", default="0,0", help="Base X,Y target compensation, mm")
    parser.add_argument("--step-xy-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--xy-speed-mm-s", type=float, default=15.0, help="Continuous STEP/coarse XY speed; descent still uses --speed-mm-s")
    parser.add_argument("--sam-review-max-depth-error-m", type=float, default=0.05, help="兼容旧命令；YOLO 模式不使用。")
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
    parser.add_argument("--checkpoint", default="checkpoints/sam3.1_multiplex.pt", help="兼容旧命令；YOLO 模式不加载。")
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
        help="下探期间对已锁定的同一 YOLO 铰链执行受限 XY 复位；力接触仍然优先并会冻结纠偏。",
    )
    parser.add_argument(
        "--final-descent-visual-tolerance-mm",
        type=float,
        default=5.0,
        help="下探视觉伺服的估计水平误差死区，默认 5 mm；超过后才发送 XY 修正。",
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
        help="下探期间同一锁定目标连续丢失多少个 YOLO 帧后撤回并重新对准，默认 2。",
    )
    parser.add_argument(
        "--regrasp-retract-m",
        type=float,
        default=0.03,
        help="视觉失败重新张爪/对准前的 base_link +Z 撤回量，默认且至少 30 mm。",
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
    parser.add_argument(
        "--post-grasp-lift-area-grace-s",
        type=float,
        default=1.0,
        help="闭爪上抬后跳过多久的面积变化（闭爪/起抬瞬间面积本来就会变），默认 1 秒。",
    )
    parser.add_argument(
        "--post-grasp-lift-area-window-frames",
        type=int,
        default=5,
        help="空夹判定使用最近多少个有效面积帧，默认 5 帧。",
    )
    parser.add_argument(
        "--post-grasp-lift-area-spread",
        type=float,
        default=0.20,
        help="窗口内掩膜面积相对中位数的最大波动比例，默认 0.20（20%%）。",
    )
    parser.add_argument(
        "--post-grasp-lift-area-confirmations",
        type=int,
        default=2,
        help="连续多少次窗口判定都超阈值才真正判空夹，默认 2；用于过滤单帧抖动。",
    )
    parser.add_argument(
        "--post-grasp-target-lost-timeout-s",
        type=float,
        default=1.0,
        help="抓取后同一目标连续不可见多久才重新抓取，默认 1 秒。",
    )
    parser.add_argument(
        "--relief-extra-lift-mm",
        type=float,
        default=None,
        help="泄力阈值达到后额外上抬的毫米数；省略时用 contact.relief_extra_lift_m（默认 0），允许 [0, 20]。",
    )
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/left_grasp_*.jsonl")
    parser.add_argument("--web-host", default="127.0.0.1", help="内嵌只读网页监听地址。")
    parser.add_argument("--web-port", type=int, default=0, help="内嵌 YOLO 网页端口；0 表示关闭。")
    parser.add_argument("--stop-after", choices=("none", "xy", "descend"), default="none", help="受看护测试的停止阶段；descend 会完成配置的定时下探但不闭爪。")
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help="执行启动接近、YOLO 抓取、+Z 上抬 150 mm、P3→P4；可用 --head-coarse-approach 替换 P1→P2。",
    )
    parser.add_argument(
        "--skip-workflow-home",
        action="store_true",
        help="分阶段流程使用：调用方已完成双臂归位，本阶段不再重复执行 workflow_home。",
    )
    parser.add_argument(
        "--head-coarse-approach",
        action="store_true",
        help="全流程开始时先用头部 RGB-D 定位并移动到最左铰链上方，跳过旧 P1→P2 预移动。",
    )
    parser.add_argument("--head-yolo-model", type=Path, default=PROJECT_ROOT / "models" / "head_hinge_yolo" / "best.pt")
    parser.add_argument(
        "--head-yolo-confidence",
        type=float,
        default=0.50,
        help="头部 YOLO 接受检测的最低置信度，默认 0.50。",
    )
    parser.add_argument(
        "--head-above-mm",
        type=float,
        default=100.0,
        help="头部粗定位后，夹爪抓取中心位于铰链上方的距离，默认 100 mm。",
    )
    parser.add_argument("--head-xy-tolerance-mm", type=float, default=50.0)
    parser.add_argument("--head-samples", type=int, default=15,
                        help="头部粗定位候选样本数；会从中保留空间一致的主簇。")
    parser.add_argument("--head-min-inlier-samples", type=int, default=8,
                        help="头部粗定位主簇最少样本数。")
    parser.add_argument("--head-max-sample-spread-mm", type=float, default=50.0,
                        help="头部粗定位主簇相对中位数的最大距离。")
    parser.add_argument("--head-forward-yaw-deg", type=float, default=0.0,
                        help="头部粗定位时末端 +Z 相对 base_link +X 的水平偏航角。")
    parser.add_argument("--head-gripper-pitch-deg", type=float, default=30.0,
                        help="头部粗定位时末端 +Z 向下俯角，正值向下。")
    parser.add_argument("--head-gripper-roll-deg", type=float, default=180.0,
                        help="头部粗定位时绕末端 +Z 的滚转角，默认 180 度。")
    parser.add_argument("--hole-sample-timeout-s", type=float, default=20.0,
                        help="头部同时采样 class 1 hole 的最长等待；超时后仍可只用 hinge 继续。")
    parser.add_argument("--move-above-hole", action="store_true",
                        help="分阶段流程：抓取并 +Z 上抬 150 mm 后，以 --hole-* 姿态移动到 hole 上方后结束本阶段。")
    parser.add_argument("--hole-x-mm", type=float, default=0.0,
                        help="左手夹爪中心相对 hole 的 base_link X 偏移，默认 0 mm。")
    parser.add_argument("--hole-y-mm", type=float, default=0.0,
                        help="左手夹爪中心相对 hole 的 base_link Y 偏移，默认 0 mm。")
    parser.add_argument("--hole-z-mm", type=float, default=100.0,
                        help="左手夹爪中心相对 hole 的 base_link Z 偏移，默认 100 mm。")
    parser.add_argument("--hole-forward-yaw-deg", type=float, default=20.0,
                        help="到 hole 上方时末端 +Z 相对 base_link +X 的水平偏航角，默认 20 度。")
    parser.add_argument("--hole-gripper-pitch-deg", type=float, default=42.0,
                        help="到 hole 上方时末端 +Z 向下俯角，默认 42 度。")
    parser.add_argument("--hole-gripper-roll-deg", type=float, default=180.0,
                        help="到 hole 上方时绕末端 +Z 的滚转角，默认 180 度。")
    parser.add_argument("--hole-speed-mm-s", type=float, default=10.0,
                        help="到 hole 上方的自动平移速度上限，默认 10 mm/s。")
    parser.add_argument("--hole-rotation-deg-s", type=float, default=20.0,
                        help="到 hole 上方时的末端角速度上限，默认 20 deg/s。")
    parser.add_argument("--max-travel-mm", type=float, default=450.0,
                        help="到 hole 上方这一步允许的最大直线行程，默认 450 mm；超过即拒绝发送。")
    parser.add_argument(
        "--wait-for-next",
        action="store_true",
        help="全流程完成后保持 GDK、SpaceMouse 接管和 Web 服务，等待右按钮长按 2 秒进入右手流程。",
    )
    parser.add_argument("--poses", type=Path, default=DEFAULT_POSES_OUTPUT, help="包含流程点位和双臂 workflow_home 的 recorded_poses.json。")
    parser.add_argument("--home-pose-name", default=WORKFLOW_HOME_POSE_NAME, help="全流程启动前使用的双臂归位点名称。")
    parser.add_argument("--preset-velocity-rad-s", type=float, default=0.08, help="全流程 P1/P2/P3/P4 的规划关节速度。")
    parser.add_argument(
        "--auto-preset-velocity-rad-s",
        type=float,
        default=None,
        help="覆盖全流程 P1/P2/P3/P4 的自动关节速度，单位 rad/s；不随 --speed-mm-s 缩放。",
    )
    parser.add_argument(
        "--speed-mm-s",
        type=float,
        default=None,
        help="整体机器人运动速度（mm/s）；必须为正数。",
    )
    parser.add_argument("--speed-scale", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--preset-arrival-tolerance-rad", type=float, default=0.03, help="全流程预设点到位关节容差。")
    parser.add_argument("--preset-timeout-s", type=float, default=25.0, help="全流程每个预设点的到位超时。")
    parser.add_argument("--spacemouse-device", help="启用 SpaceMouse 临时接管；指定 /dev/hidrawN。")
    parser.add_argument("--spacemouse-speed-mm-s", type=float, default=10.0, help="SpaceMouse 平移速度上限，单位 mm/s。")
    parser.add_argument("--spacemouse-rotation-deg-s", type=float, default=10.0, help="SpaceMouse 旋转速度上限，单位 deg/s。")
    parser.add_argument("--spacemouse-deadzone", type=float, default=0.12, help="SpaceMouse 轴死区。")
    parser.add_argument("--allow-motion", action="store_true", help="唯一运动确认参数；未指定时严格只读。")
    return parser.parse_args(argv)


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
    # Keep the persisted historical calibration provenance, independent of live mode 3.
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
                relief_extra_lift_m=float(contact.get("relief_extra_lift_m", 0.0)),
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
    return_masks: bool = False,
) -> tuple[list[dict], int | None] | tuple[list[dict], int | None, list[np.ndarray]]:
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
    if return_masks:
        return statistics, selected, masks
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
            "post_grasp_verify", "force_relief_speed", "perception_gate", "tracking", "selected_mask_index",
            "tracking_lock_reset", "step_pose", "pixel_error_px", "delta_base_m", "alignment_stage",
            "actual_xy_error_m", "actual_end_link_before_base_link", "target_follow_error_mm",
            "screen_gripper_target_pixel_px",
            "gripper_position_rad", "adaptive_gripper_close_travel_rad", "adaptive_gripper_stall",
            "height_reference",
            "end_link_target_before_base_link", "end_link_target_after_base_link",
            "final_descent_progress_m", "final_descent_termination", "actual_base_delta_m",
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
    relief = record.get("force_relief")
    if isinstance(relief, dict):
        summary["force_relief"] = {
            key: relief[key]
            for key in (
                "required", "released", "latest_delta_force_n",
                "consecutive_hits", "speed_mm_s", "max_lift_m",
            )
            if key in relief
        }
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
    stream = controller.start_left_gripper_position_stream(target_rad)
    try:
        while True:
            stream.raise_if_failed()
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
    finally:
        stream.stop()


def lift_before_visual_reacquire(controller, *, speed_mm_s):
    """Dedicated retry retreat; no gripper command or XY translation."""
    current = measured_left_end_link_pose(controller)
    target = CartesianPoseTarget(
        (current.position_m[0], current.position_m[1], current.position_m[2] + .03),
        current.orientation_xyzw,
    )
    stream = controller.start_left_end_effector_position_stream(
        current, rate_hz=50., life_time_s=.10,
        max_translation_velocity_mps=speed_mm_s / 1000.,
    )
    try:
        stream.update_target(target)
        wait_actual_left_target(controller, stream, target, timeout_s=30. / speed_mm_s + 5.,
                               tolerance_m=.002, orientation_tolerance_deg=2.)
    finally:
        stream.stop()
    print('[视觉重抓] 已竖直上抬 30 mm，继续重新定位夹取。', flush=True)


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
    """Cancel descent, retreat at least 30 mm in base +Z, then open/reacquire."""
    retract_m = max(.03, retract_m)
    position_stream.raise_if_failed()
    current = measured_left_end_link_pose(controller)
    # GDK status pose quaternions vary at floating-point precision between
    # reads.  The live stream deliberately rejects any orientation change, so
    # recovery must retain the stream's fixed visual-servo orientation rather
    # than reuse the newly measured quaternion.
    hold = CartesianPoseTarget(current.position_m, fixed_orientation_xyzw)
    # A force contact or a stalled stream can leave the internal commanded
    # point far from the measured pose, which freezes the slew and would make
    # this recovery target unexecutable.  Rebase before commanding it.
    position_stream.rebase_to_measured_pose()
    position_stream.update_target(hold)
    retract_target = CartesianPoseTarget(
        base_position_after_delta(hold.position_m, np.asarray((0.0, 0.0, retract_m))),
        hold.orientation_xyzw,
    )
    position_stream.update_target(retract_target)
    timeout_s = max(2.0, retract_m / translation_speed_mps + 1.0)
    try:
        wait_actual_left_target(controller, position_stream, retract_target,
                               timeout_s=timeout_s + 3., tolerance_m=.002,
                               orientation_tolerance_deg=2.)
    finally:
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
        "detector": record.get("detector", "yolo"),
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
    """The sole owner of the hand camera, YOLO model, and left web pane.

    Motion code must never call GDK Camera or the model. It waits for a
    snapshot from this object instead. YOLO runs for every acquired frame and
    this loop publishes the left pane of the composite web preview.
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
        self._target_pixel_override: np.ndarray | None = None
        self._target_pixel_detail: dict | None = None
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

    def set_target_pixel_override(self, target: Sequence[float] | None, detail: dict | None = None) -> None:
        """Update the web marker target used by the dynamic height reference."""
        with self._condition:
            if target is None:
                self._target_pixel_override = None
                self._target_pixel_detail = None
                return
            values = np.asarray(tuple(float(value) for value in target), dtype=np.float64)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise ValueError("target pixel override must contain two finite values")
            self._target_pixel_override = values
            self._target_pixel_detail = None if detail is None else dict(detail)

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
        intrinsic_retry_started = time.monotonic()
        previous_step_timestamp = None
        step_trackers = OcclusionTrackerPool()
        last_sam_review_at = float("-inf")
        sam_review: dict = {"accepted": False, "fresh": False, "reason": "awaiting_first_sam_review"}
        try:
            while not self._stop.is_set():
                started = time.monotonic()
                color = self.camera.get_latest_image(self.color_type, 1000.0)
                color_rgb = decode_color_rgb(color, self.gdk)
                if self.shared is not None and hasattr(self.shared, 'publish_raw'):
                    raw_ok, raw_jpeg = cv2.imencode('.jpg', cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR))
                    if raw_ok:
                        self.shared.publish_raw(raw_jpeg.tobytes(), {'frame_read_started_s': started, 'state': 'left_raw_camera'})
                use_step = getattr(self.args, "step_profile", None) is not None
                if use_step:
                    if color.timestamp_ns == previous_step_timestamp:
                        time.sleep(.01)
                        continue
                    previous_step_timestamp = color.timestamp_ns
                if not use_step:
                    depth = self.camera.get_nearest_image(self.depth_type, int(color.timestamp_ns), 1000.0)
                    depth_u16 = decode_depth_u16(depth, self.gdk)
                if color_intrinsic is None:
                    try:
                        candidate_color_intrinsic = intrinsic_to_dict(
                            self.camera.get_camera_intrinsic(self.color_type),
                            width=int(color.width), height=int(color.height),
                        )
                        candidate_depth_intrinsic = None
                        candidate_depth_to_color = None
                        if not use_step:
                            candidate_depth_intrinsic = intrinsic_to_dict(
                                self.camera.get_camera_intrinsic(self.depth_type),
                                width=int(depth.width), height=int(depth.height),
                            )
                            candidate_depth_to_color = transform_to_dict(self.tf.get_tf_from_sensor(
                                self.gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor
                            ))
                        elif not self.args.step_metric_xy and candidate_color_intrinsic != self.args.step_alignment["intrinsic"]:
                            raise RuntimeError("STEP reference camera calibration differs; recapture reference")
                        candidate_end_link_transform = transform_to_dict(self.tf.get_tf_from_sensor(
                            self.gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink
                        ))
                    except Exception as exc:
                        elapsed = time.monotonic() - intrinsic_retry_started
                        if elapsed >= 5.0:
                            raise RuntimeError(f"左手相机内参/外参连续 5 秒不可用：{exc}") from exc
                        if self.shared is not None:
                            self.shared.update(None, {
                                "state": "waiting_for_left_camera_calibration",
                                "sequence": self._sequence,
                                "retry_elapsed_s": round(elapsed, 2),
                                "error": str(exc),
                            })
                        time.sleep(0.20)
                        continue
                    color_intrinsic = candidate_color_intrinsic
                    depth_intrinsic = candidate_depth_intrinsic
                    depth_to_color = candidate_depth_to_color
                    end_link_transform = candidate_end_link_transform
                aligned_depth = np.zeros(color_rgb.shape[:2], dtype=np.uint16) if use_step else align_depth_to_color(depth_u16, depth_intrinsic=depth_intrinsic, color_intrinsic=color_intrinsic, depth_to_color=depth_to_color, depth_scale_m=DEPTH_SCALE_M)
                screen_target_pixel_px = screen_gripper_target_pixel(color_rgb.shape)
                with self._condition:
                    if self._target_pixel_override is not None:
                        screen_target_pixel_px = self._target_pixel_override.copy()
                        height_reference_detail = dict(self._target_pixel_detail or {})
                    else:
                        height_reference_detail = None
                # Run YOLO on the complete image.  The central region is a
                # target-validity gate only, so edge detections remain visible
                # for diagnostics without entering the grasp state machine.
                crop_bounds = None
                roi = parse_roi(self.args.roi, color_rgb.shape[:2])
                yolo_valid_roi = None
                if self.args.detector in {"yolo", "yolo-sam"}:
                    yolo_valid_roi = center_crop_bounds(
                        color_rgb.shape, fraction=self.args.yolo_center_fraction,
                    )
                    roi = intersect_roi(roi, yolo_valid_roi)
                if self.args.detector == "sam3":
                    sam_input, _ = center_crop_image(color_rgb)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        image_state = self.processor.set_image(Image.fromarray(sam_input))
                        result = self.processor.set_text_prompt(self.args.prompt, image_state)
                    masks = expand_crop_masks([np.asarray(mask, dtype=bool).squeeze() for mask in result["masks"].detach().cpu().numpy()], full_shape=color_rgb.shape[:2], bounds=crop_bounds)
                    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
                    statistics = [stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform) for mask in masks]
                    for item, score, mask in zip(statistics, scores, masks):
                        item["sam3_confidence"] = float(score)
                        item["sam_grasp_pixel_px"] = mask_fraction_grasp_pixel(
                            mask, self.args.sam_grasp_height_fraction
                        )
                    candidates = eligible_candidates(statistics, min_area=self.args.min_area, min_valid_depth_pixels=self.args.min_valid_depth_pixels, roi=roi, require_depth=False)
                    sam_review = {"accepted": True, "fresh": True, "reason": "sam3_primary"}
                else:
                    yolo_bgr, yolo_valid_roi = prepare_yolo_input(
                        color_rgb, valid_fraction=self.args.yolo_center_fraction,
                    )
                    masks, scores, class_ids = infer_yolo_masks(
                        self.yolo_model,
                        yolo_bgr,
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
                    if self.args.detector == "yolo":
                        for item in statistics:
                            item["yolo_box_grasp_pixel_px"] = yolo_box_fraction_grasp_pixel(
                                item, self.args.yolo_box_height_fraction
                            )
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
                        self.args.detector == "yolo-sam"
                        and
                        provisional is not None
                        and not transport_phase
                        and now - last_sam_review_at >= self.args.sam_review_interval_s
                    ):
                        sam_statistics, sam_index, sam_masks = sam_review_for_yolo_target(self.processor, color_rgb, aligned_depth, color_intrinsic=color_intrinsic, end_link_transform=end_link_transform, prompt=self.args.prompt, min_area=self.args.min_area, min_valid_depth_pixels=self.args.min_valid_depth_pixels, roi=roi, reference_center_px=statistics[provisional]["center_px"], return_masks=True)
                        if sam_index is not None:
                            sam_statistics[sam_index]["sam_grasp_pixel_px"] = mask_fraction_grasp_pixel(
                                sam_masks[sam_index], self.args.sam_grasp_height_fraction
                            )
                            sam_statistics[sam_index]["sam_grasp_offset_from_center_px"] = (
                                np.asarray(sam_statistics[sam_index]["sam_grasp_pixel_px"], dtype=np.float64)
                                - np.asarray(sam_statistics[sam_index]["center_px"], dtype=np.float64)
                            ).tolist()
                        accepted, detail = review_matches(statistics[provisional], None if sam_index is None else sam_statistics[sam_index], max_pixel_error=self.args.sam_review_max_pixel_error, max_depth_error_m=self.args.sam_review_max_depth_error_m, allow_missing_depth=True)
                        attempt = {
                            **detail,
                            "accepted": accepted,
                            "timestamp_monotonic": now,
                            "yolo_center_px": statistics[provisional]["center_px"],
                        }
                        if accepted:
                            sam_review = {
                                **attempt,
                                "sam_selected": sam_statistics[sam_index],
                            }
                        elif not sam_review.get("accepted"):
                            # Before the first valid SAM review, fail closed.
                            sam_review = {**attempt, "sam_selected": None}
                        else:
                            # A transient SAM miss must not erase a valid
                            # landmark for the same continuously tracked YOLO
                            # instance. The normal TTL and target-distance gate
                            # below still reject stale or switched targets.
                            sam_review = {**sam_review, "latest_attempt": attempt}
                        last_sam_review_at = now
                    age = None if "timestamp_monotonic" not in sam_review else now - float(sam_review["timestamp_monotonic"])
                    sam_review = {**sam_review, "fresh": age is not None and age <= self.args.sam_review_ttl_s, "age_s": None if age is None else round(max(age, 0.0), 3)}
                    if provisional is not None and sam_review.get("accepted") and sam_review.get("fresh"):
                        sam_selected = sam_review.get("sam_selected")
                        if sam_selected is not None and "sam_grasp_pixel_px" in sam_selected:
                            # SAM is intentionally slower than YOLO. Carry the
                            # latest SAM landmark as an offset on the current
                            # YOLO centre so the control point follows every
                            # frame instead of freezing between SAM reviews.
                            offset = np.asarray(
                                sam_selected.get(
                                    "sam_grasp_offset_from_center_px",
                                    np.asarray(sam_selected["sam_grasp_pixel_px"], dtype=np.float64)
                                    - np.asarray(sam_selected["center_px"], dtype=np.float64),
                                ),
                                dtype=np.float64,
                            )
                            statistics[provisional]["sam_grasp_pixel_px"] = (
                                np.asarray(statistics[provisional]["center_px"], dtype=np.float64) + offset
                            ).tolist()
                if use_step:
                    bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                    base_end = None
                    if self.args.step_metric_xy:
                        try:
                            raw = self.tf.lookup_transform("base_link", "arm_l_end_link", int(color.timestamp_ns))
                            base_end = np.linalg.inv(transform_from_dict(transform_to_dict(raw)))
                        except Exception:
                            pass
                    tracked_poses = step_trackers.update(self.args.step_geometry,
                        [detect_hole_centres(bgr, masks[index]) for index in candidates],
                        [statistics[index]["center_px"] for index in candidates], color_intrinsic)
                    for index, tracked_pose in zip(candidates, tracked_poses):
                        statistics[index]["step_pose"] = dict(tracked_pose)
                        pose = statistics[index]["step_pose"]
                        if self.args.step_metric_xy:
                            if base_end is None:
                                pose.update(accepted=False, reason="image_time_tf_unavailable")
                            elif pose.get("accepted"):
                                pose["base_xy_error_m"] = xy_error(base_end, transform_from_dict(end_link_transform), pose["grasp_xyz_camera_m"], self.args.step_tcp_m, self.args.step_compensation_m)
                selected = choose_candidate(statistics, candidates, self.args.selection)
                height_samples = getattr(self.args, "step_height_samples", None)
                if use_step and selected is not None and height_samples:
                    selected_pose = statistics[selected].get("step_pose", {})
                    if selected_pose.get("accepted"):
                        try:
                            target, height_reference_detail = target_pixel_for_step_pose(
                                height_samples, selected_pose,
                            )
                            screen_target_pixel_px = target
                        except ValueError:
                            # The normal STEP acceptance gate remains in
                            # control; an invalid/missing interpolation input
                            # falls back to the calibrated 0 mm target.
                            height_reference_detail = None
                shaft_association = associate_shaft_to_hinge(
                    statistics,
                    selected,
                    max_center_distance_px=self.args.shaft_association_max_distance_px,
                    max_horizontal_distance_px=self.args.shaft_association_max_horizontal_distance_px,
                    ambiguity_margin_px=self.args.shaft_association_ambiguity_margin_px,
                    shaft_indices=shaft_candidates,
                ) if self.args.detector in {"yolo", "yolo-sam"} else {"shaft_index": None, "reason": "sam3_mode"}
                with self._condition:
                    self._sequence += 1
                    snapshot = {"sequence": self._sequence, "color_rgb": color_rgb, "masks": masks, "statistics": statistics, "candidates": candidates, "hinge_candidates": candidates, "shaft_candidates": shaft_candidates if self.args.detector in {"yolo", "yolo-sam"} else [], "selected": selected, "shaft_association": shaft_association, "aligned_depth": aligned_depth, "color_intrinsic": color_intrinsic, "depth_intrinsic": depth_intrinsic, "depth_to_color": depth_to_color, "end_link_transform": end_link_transform, "screen_gripper_target_pixel_px": screen_target_pixel_px, "height_reference": height_reference_detail, "roi": roi, "sam_review": dict(sam_review), "inference_ms": (time.monotonic() - started) * 1000.0}
                    self._latest = snapshot
                    phase = dict(self._phase)
                    self._condition.notify_all()
                if self.shared is not None:
                    if self.args.detector == "yolo" and selected is not None:
                        overlay = draw_overlay(
                            cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR),
                            [masks[selected]], [statistics[selected]], 0,
                        )
                    else:
                        overlay = draw_overlay(cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR), masks, statistics, selected)
                    if yolo_valid_roi is not None:
                        x0, y0, x1, y1 = yolo_valid_roi
                        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
                        cv2.putText(
                            overlay,
                            f"YOLO valid target region {self.args.yolo_center_fraction:.0%}",
                            (x0 + 8, max(y0 + 24, 24)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                    if selected is not None and (
                        "yolo_box_grasp_pixel_px" in statistics[selected]
                        or statistics[selected].get("step_pose", {}).get("accepted")
                    ):
                        selected_item = statistics[selected]
                        point = tuple(np.round(servo_pixel(selected_item)).astype(int))
                        cv2.drawMarker(overlay, point, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
                        step_pose = selected_item.get("step_pose", {})
                        if step_pose.get("accepted"):
                            for model_index, matched in zip(
                                step_pose.get("matched_model_indices", []),
                                step_pose.get("matched_pixels", []),
                            ):
                                hole_point = tuple(np.round(matched).astype(int))
                                cv2.circle(overlay, hole_point, 5, (0, 255, 0), 2)
                                cv2.putText(
                                    overlay, str(model_index),
                                    (hole_point[0] + 6, hole_point[1] - 6),
                                    cv2.FONT_HERSHEY_SIMPLEX, .45, (0, 255, 0), 2,
                                )
                            cv2.putText(
                                overlay, "STEP grasp projection",
                                (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2,
                            )
                        else:
                            cv2.putText(
                                overlay,
                                f"YOLO grasp bottom-up-{self.args.yolo_box_height_fraction:.0%}",
                                (point[0] + 10, point[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 255), 2,
                            )
                        sam_selected = sam_review.get("sam_selected")
                        if sam_selected is not None and sam_selected.get("bbox_xyxy") is not None:
                            x0, y0, x1, y1 = (int(round(value)) for value in sam_selected["bbox_xyxy"])
                            cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 0, 255), 2)
                            cv2.putText(overlay, "review bbox", (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 0, 255), 2)
                    target_pixel = screen_target_pixel_px
                    if target_pixel is not None:
                        target_point = tuple(np.round(target_pixel).astype(int))
                        cv2.drawMarker(overlay, target_point, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 22, 2)
                        cv2.putText(overlay, "servo target", (target_point[0] + 10, target_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2)
                    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if ok:
                        self.shared.update(encoded.tobytes(), {"state": phase["state"], "sequence": self._sequence, "selected_mask_index": selected, "selected": None if selected is None else statistics[selected], "screen_gripper_target_pixel_px": screen_target_pixel_px.tolist(), "height_reference": height_reference_detail, "shaft_association": shaft_association, "detector": self.args.detector, "yolo_valid_roi_xyxy": None if yolo_valid_roi is None else list(yolo_valid_roi), "yolo_valid_region_fraction": None if yolo_valid_roi is None else self.args.yolo_center_fraction, "inference_ms": round(snapshot["inference_ms"], 1), "grasp_reason": phase["reason"], "safety": "常驻感知线程独占相机、YOLO 与网页更新；动作状态机仅消费最新快照。"})
        except Exception as exc:
            with self._condition:
                self._failure = exc
                self._condition.notify_all()


class SnapshotVisualMailbox:
    """Same-target observation adapter backed by persistent perception."""

    def __init__(self, perception: PersistentPerception, *, initial_center_px, track_max_distance_px,
                 target_lost_frames, target_pixel_px=None, pixel_jacobian=None,
                 pixel_tolerance_px=0.0, max_xy_step_m=0.0, allow_xy_correction=True,
                 target_lost_timeout_s: float | None = None,
                 visual_tolerance_m: float = 0.0, target_pixel_provider=None):
        self.perception = perception
        self.center = np.asarray(initial_center_px, dtype=np.float64)
        self.track_max_distance_px = track_max_distance_px
        self.target_lost_frames = target_lost_frames
        self.target_lost_timeout_s = target_lost_timeout_s
        self.target_pixel_px, self.pixel_jacobian = target_pixel_px, pixel_jacobian
        self.target_pixel_provider = target_pixel_provider
        self.pixel_tolerance_px, self.max_xy_step_m = pixel_tolerance_px, max_xy_step_m
        self.visual_tolerance_m = float(visual_tolerance_m)
        if not math.isfinite(self.visual_tolerance_m) or self.visual_tolerance_m < 0.0:
            raise ValueError("visual_tolerance_m must be finite and non-negative")
        self.lost_frames = 0
        self.lost_since: float | None = None
        self.allow_xy_correction = allow_xy_correction

    def newer_than(self, sequence: int) -> dict | None:
        snapshot = self.perception.newer_than(sequence, timeout_s=0.0)
        if snapshot is None:
            return None
        nearby = [index for index in snapshot["candidates"] if float(np.linalg.norm(np.asarray(snapshot["statistics"][index]["center_px"], dtype=np.float64) - self.center)) <= self.track_max_distance_px]
        selected = min(nearby, key=lambda index: float(np.linalg.norm(np.asarray(snapshot["statistics"][index]["center_px"], dtype=np.float64) - self.center)), default=None)
        if selected is None:
            self.lost_frames += 1
            now = time.monotonic()
            if self.lost_since is None:
                self.lost_since = now
            lost_duration_s = now - self.lost_since
            lost = (
                lost_duration_s >= self.target_lost_timeout_s
                if self.target_lost_timeout_s is not None
                else self.lost_frames >= self.target_lost_frames
            )
            return {
                "sequence": snapshot["sequence"],
                "valid": False if lost else None,
                "reason": (
                    "locked_target_lost"
                    if lost
                    else "locked_target_missing_pending"
                ),
                "lost_frames": self.lost_frames,
                "lost_duration_s": round(lost_duration_s, 3),
                "last_center_px": self.center.tolist(),
            }
        self.lost_frames = 0
        self.lost_since = None
        item = snapshot["statistics"][selected]
        if self.allow_xy_correction and self.target_pixel_px is not None and "step_pose" in item and not item["step_pose"].get("accepted"):
            return {"sequence": snapshot["sequence"], "valid": False, "reason": "step_pose_lost"}
        self.center = np.asarray(item["center_px"], dtype=np.float64)
        result = {"sequence": snapshot["sequence"], "valid": True, "selected_mask_index": selected, "center_px": self.center.tolist(), "area_px": int(item["area_px"]), "sam3_confidence": item.get("sam3_confidence"), "yolo_confidence": item.get("yolo_confidence")}
        if not self.allow_xy_correction:
            return result
        if self.target_pixel_provider is not None:
            target = self.target_pixel_provider(item)
            if target is not None:
                target = np.asarray(target, dtype=np.float64)
                if target.shape != (2,) or not np.isfinite(target).all():
                    raise ValueError("visual target pixel provider returned an invalid target")
                self.target_pixel_px = target
                result["target_pixel_px"] = target.tolist()
        if self.target_pixel_px is not None and self.pixel_jacobian is not None:
            metric = item.get("step_pose", {}).get("base_xy_error_m")
            if metric is not None:
                if np.linalg.norm(metric) > self.perception.args.step_xy_tolerance_mm / 1000:
                    result["delta_base_m"] = bounded_xy(metric, self.max_xy_step_m).tolist()
                    result["pixel_error_px"] = [0.0, 0.0]
                    result["base_xy_error_m"] = metric
                return result
            error = np.asarray(servo_pixel(item)) - np.asarray(self.target_pixel_px, dtype=np.float64)
            result["pixel_error_px"] = error.tolist()
            if self.visual_tolerance_m > 0.0:
                jacobian = np.asarray(self.pixel_jacobian, dtype=np.float64)
                visual_delta = np.linalg.solve(jacobian, -error)
                visual_error_m = float(np.linalg.norm(visual_delta))
                if not math.isfinite(visual_error_m):
                    raise ValueError("visual-servo estimated XY error is not finite")
                result["visual_error_m"] = visual_error_m
                if visual_error_m <= self.visual_tolerance_m:
                    return result
            elif float(np.linalg.norm(error)) <= self.pixel_tolerance_px:
                return result
            result["delta_base_m"] = empirical_horizontal_base_delta_from_pixel_error(
                error, self.pixel_jacobian, max_step_m=self.max_xy_step_m
            ).tolist()
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
                rgb = decode_color_rgb(image, gdk)
                if shared is not None and hasattr(shared, 'publish_raw'):
                    raw_ok, raw_jpeg = cv2.imencode('.jpg', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    if raw_ok:
                        shared.publish_raw(raw_jpeg.tobytes(), {'frame_read_started_s': started, 'state': 'left_raw_camera'})
                depth = camera.get_nearest_image(depth_type, int(image.timestamp_ns), 1000.0)
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


def run_head_coarse_approach_in_process(*, gdk, controller: G2Controller, tf, args, head_model_path: Path, log,
                                        shared: SharedView | None = None, head_camera=None,
                                        head_model=None, prelocalized_targets=None) -> tuple[threading.Event | None, threading.Thread | None, dict]:
    """Locate the leftmost head-camera hinge and move above it in this GDK session.

    ``head_camera`` and ``head_model`` may be supplied by a single-process
    orchestrator that preloaded them; in that case this stage neither creates
    nor closes them.
    """
    try:
        tcp_end_m = np.asarray([float(value) for value in args.step_tcp_end_mm.split(",")], dtype=float) / 1000.0
        if tcp_end_m.shape != (3,) or not np.isfinite(tcp_end_m).all():
            raise ValueError
    except ValueError as exc:
        raise SafetyError("--step-tcp-end-mm 必须是三个有限数值") from exc

    fixed_orientation = fixed_forward_down_orientation(
        forward_yaw_deg=args.head_forward_yaw_deg,
        gripper_pitch_deg=args.head_gripper_pitch_deg,
        gripper_roll_deg=args.head_gripper_roll_deg,
    )
    owns_head_camera = head_camera is None and prelocalized_targets is None
    owns_head_model = head_model is None and prelocalized_targets is None
    if head_camera is None and prelocalized_targets is None:
        head_camera = gdk.Camera([gdk.CameraType.kHeadColor, gdk.CameraType.kHeadDepth])
    head_preview_stop: threading.Event | None = None
    head_preview_thread: threading.Thread | None = None
    head_coarse_context: dict | None = None
    preview_transferred = False
    try:
        if head_model is None and prelocalized_targets is None:
            head_model = load_yolo_segmentation_model(head_model_path, device=args.yolo_device)
        depth_to_color = head_from_color = None
        if prelocalized_targets is None:
            depth_to_color = transform_to_dict(tf.get_tf_from_sensor(gdk.SensorExtrinsicType.kHeadDepthToHeadColor))
            head_from_color = transform_to_dict(tf.get_tf_from_sensor(gdk.SensorExtrinsicType.kHeadRGBDToHeadLink3))
        color_intrinsic = depth_intrinsic = None
        observations: list[np.ndarray] = []
        hole_observations: list[np.ndarray] = []
        tf_modes: list[str] = []
        tf_gaps_ms: list[float] = []
        if prelocalized_targets is not None:
            known = head_context_from_targets(prelocalized_targets)
            observations = [
                np.asarray(known["hinge_position_base_link_m"], dtype=float)
                for _ in range(args.head_samples)
            ]
            if known["hole_position_base_link_m"] is not None:
                hole_observations = [
                    np.asarray(known["hole_position_base_link_m"], dtype=float)
                    for _ in range(args.head_samples)
                ]
            tf_modes = ["orchestrator_head_localize"] * args.head_samples
            tf_gaps_ms = [0.0] * args.head_samples
        last_timestamp = None
        # Keep collecting until the requested number of valid samples is
        # available.  A temporary occlusion or a slow head-camera stream must
        # not abort the whole workflow after an arbitrary wall-clock timeout.
        # The hole (class 1) is sampled independently of the hinge and only
        # until its own deadline, so a frame without a visible hole can never
        # stall the hinge localization.
        hole_deadline = time.monotonic() + float(args.hole_sample_timeout_s)
        while prelocalized_targets is None and (len(observations) < args.head_samples or (
            len(hole_observations) < args.head_samples and time.monotonic() < hole_deadline
        )):
            color = head_camera.get_latest_image(gdk.CameraType.kHeadColor, 1000.0)
            timestamp_ns = int(color.timestamp_ns)
            if timestamp_ns == last_timestamp:
                time.sleep(0.01)
                continue
            last_timestamp = timestamp_ns
            base_from_head, tf_mode, tf_gap_ms = base_from_head_at(tf, timestamp_ns)
            depth = head_camera.get_nearest_image(gdk.CameraType.kHeadDepth, timestamp_ns, 1000.0)
            bgr = cv2.cvtColor(decode_color_rgb(color, gdk), cv2.COLOR_RGB2BGR)
            depth_u16 = decode_depth_u16(depth, gdk)
            if color_intrinsic is None:
                color_intrinsic = intrinsic_to_dict(
                    head_camera.get_camera_intrinsic(gdk.CameraType.kHeadColor), width=int(color.width), height=int(color.height),
                )
                depth_intrinsic = intrinsic_to_dict(
                    head_camera.get_camera_intrinsic(gdk.CameraType.kHeadDepth), width=int(depth.width), height=int(depth.height),
                )
            aligned = align_depth_to_color(
                depth_u16, depth_intrinsic=depth_intrinsic, color_intrinsic=color_intrinsic,
                depth_to_color=depth_to_color, depth_scale_m=DEPTH_SCALE_M,
            )
            masks, scores, classes = infer_yolo_masks(
                head_model, bgr, confidence=args.head_yolo_confidence, imgsz=640, device=args.yolo_device,
            )
            selected = select_leftmost_hinge(masks, scores, classes)
            selected_hole = select_highest_confidence_class(masks, scores, classes, 1)
            if shared is not None:
                overlay = draw_head_targets_overlay(bgr, masks, scores, classes)
                ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if ok:
                    shared.update(encoded.tobytes(), {
                        "state": "head_coarse_localizing",
                        "source": "head_yolo",
                        "sequence": int(time.monotonic_ns()),
                        "selected_mask_index": selected,
                        "selected_hole_mask_index": selected_hole,
                        "collected_samples": len(observations),
                        "collected_hole_samples": len(hole_observations),
                        "required_samples": args.head_samples,
                        "safety": "头部 RGB-D 正在定位最左铰链与 hole；尚未发送左手靠近目标。",
                    })
            if selected is None and selected_hole is None:
                continue
            if selected is not None and len(observations) < args.head_samples:
                _inliers, points = depth_points(masks[selected], aligned, DEPTH_SCALE_M, color_intrinsic)
                if len(points) >= 20:
                    point_color = np.median(points, axis=0)
                    observations.append(np.asarray(apply_transform(
                        base_from_head, apply_transform(head_from_color, point_color),
                    ), dtype=float))
                    tf_modes.append(tf_mode)
                    tf_gaps_ms.append(tf_gap_ms)
            if selected_hole is not None and len(hole_observations) < args.head_samples:
                _hole_inliers, hole_points = depth_points(
                    masks[selected_hole], aligned, DEPTH_SCALE_M, color_intrinsic,
                )
                if len(hole_points) >= 20:
                    hole_color = np.median(hole_points, axis=0)
                    hole_observations.append(np.asarray(apply_transform(
                        base_from_head, apply_transform(head_from_color, hole_color),
                    ), dtype=float))
        hinge_base, inlier_mask, spread_m = dominant_sample_cluster(
            np.asarray(observations), args.head_max_sample_spread_mm / 1000.0,
        )
        inlier_count = int(np.count_nonzero(inlier_mask))
        if inlier_count < args.head_min_inlier_samples:
            raise SafetyError(
                f"头部铰链样本没有足够一致的主簇：{inlier_count}/{len(observations)} 个样本在 "
                f"{args.head_max_sample_spread_mm:.1f} mm 内，至少需要 {args.head_min_inlier_samples} 个"
            )
        localization = {
            "event": "head_leftmost_hinge_localized", "motion_enabled": True,
            "samples": len(observations), "inlier_samples": inlier_count,
            "rejected_samples": len(observations) - inlier_count,
            "sample_spread_mm": round(spread_m * 1000.0, 2),
            "max_sample_spread_mm": args.head_max_sample_spread_mm,
            "tf_modes": [mode for mode, keep in zip(tf_modes, inlier_mask) if keep],
            "max_tf_fallback_gap_ms": round(max(gap for gap, keep in zip(tf_gaps_ms, inlier_mask) if keep), 3),
            "hinge_position_base_link_m": [float(value) for value in hinge_base],
        }
        hole_base = None
        if len(hole_observations) >= args.head_min_inlier_samples:
            try:
                candidate, hole_inliers, hole_spread_m = dominant_sample_cluster(
                    np.asarray(hole_observations), args.head_max_sample_spread_mm / 1000.0,
                )
                hole_inlier_count = int(np.count_nonzero(hole_inliers))
                if hole_inlier_count >= args.head_min_inlier_samples:
                    hole_base = candidate
                    localization.update({
                        "hole_samples": len(hole_observations),
                        "hole_inlier_samples": hole_inlier_count,
                        "hole_sample_spread_mm": round(hole_spread_m * 1000.0, 2),
                        "hole_position_base_link_m": [float(value) for value in candidate],
                    })
            except ValueError:
                hole_base = None
        if hole_base is None:
            localization.update({
                "hole_samples": len(hole_observations),
                "hole_inlier_samples": 0,
                "hole_position_base_link_m": None,
            })
        print(json.dumps(localization, ensure_ascii=False), flush=True)
        json_record(log, localization)

        # Hand the already-open head camera/model to a read-only preview
        # worker before the arm starts moving.  This keeps the head image live
        # during the coarse approach as well as during the later hand motion,
        # without initializing GDK or loading the model a second time.
        if shared is not None and prelocalized_targets is None:
            head_preview_stop, head_preview_thread = start_head_camera_web_preview(
                gdk=gdk, head_model_path=head_model_path, args=args, shared=shared,
                camera=head_camera, model=head_model,
                owns_camera=owns_head_camera, owns_model=owns_head_model,
            )
            head_camera = None
            head_model = None

        controller.prepare_joint_impedance_control(("left",))
        current = controller.read_end_effector_pose(LEFT_VISUAL_TIP_LINK)
        orientation_error_deg = quaternion_angle_deg(current.orientation_xyzw, fixed_orientation)
        # Solve the final position and orientation together, then use the
        # preset joint planner; the intermediate Cartesian path is unconstrained.
        target = gripper_end_target(hinge_base, fixed_orientation, tcp_end_m, args.head_above_mm / 1000.0)
        travel_mm = float(np.linalg.norm(np.asarray(target.position_m) - np.asarray(current.position_m)) * 1000.0)
        if travel_mm > 450.0:
            raise SafetyError(f"当前末端到头部目标直线距离 {travel_mm:.1f} mm，超过 450.0 mm")
        velocity_rad_s = float(args.preset_velocity_rad_s if args.auto_preset_velocity_rad_s is None
                               else args.auto_preset_velocity_rad_s)
        start = {"event": "left_above_hinge_motion_start", "travel_mm": round(travel_mm, 2),
                 "orientation_error_deg": round(orientation_error_deg, 2),
                 "control_method": "endpoint_ik_joint_plan", "control_mode": 3,
                 "velocity_rad_s": velocity_rad_s,
                 "target_end_link_base_m": list(target.position_m), "orientation_xyzw": list(target.orientation_xyzw)}
        print(json.dumps(start, ensure_ascii=False), flush=True)
        json_record(log, start)
        try:
            from .left_hole_joint_move import move_left_hole_joint_target
        except ImportError:
            from left_hole_joint_move import move_left_hole_joint_target
        actual = move_left_hole_joint_target(controller, target, velocity_rad_s=velocity_rad_s)
        actual_center = gripper_end_target(np.asarray(actual.position_m), actual.orientation_xyzw, -tcp_end_m, 0.0).position_m
        xy_error_mm = float(np.linalg.norm(np.asarray(actual_center[:2]) - hinge_base[:2]) * 1000.0)
        if xy_error_mm > args.head_xy_tolerance_mm:
            raise SafetyError(
                f"夹持中心与头部铰链目标水平误差 {xy_error_mm:.1f} mm，"
                f"超过 {args.head_xy_tolerance_mm:.1f} mm；不进入左手视觉伺服"
            )
        arrived = {"event": "left_above_hinge_arrived", "grasp_center_xy_error_mm": round(xy_error_mm, 2),
                   "actual_end_link_base_m": list(actual.position_m), "gdk_session": "reused"}
        print(json.dumps(arrived, ensure_ascii=False), flush=True)
        json_record(log, arrived)
        preview_transferred = True
        return head_preview_stop, head_preview_thread, {
            "hinge_position_base_link_m": [float(value) for value in hinge_base],
            "hole_position_base_link_m": None if hole_base is None else [float(value) for value in hole_base],
            "target_end_link_base_m": [float(value) for value in target.position_m],
            "actual_end_link_base_m": [float(value) for value in actual.position_m],
        }
    finally:
        if not preview_transferred and head_preview_stop is not None:
            stop_head_camera_web_preview(head_preview_stop, head_preview_thread)
        if owns_head_camera and head_camera is not None:
            try:
                head_camera.close_camera()
            except Exception:
                pass
        if owns_head_model and head_model is not None:
            del head_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def hole_target_from_context(head_coarse_context) -> list[float] | None:
    """Return the head-localized hole, or ``None`` when it was not detected.

    A missing hole is a downgrade, not a failure: the flow still grasps and
    lifts, then holds at the lifted pose and waits for the operator instead of
    moving above the hole.
    """
    hole = None if head_coarse_context is None else head_coarse_context.get("hole_position_base_link_m")
    if hole is None:
        return None
    return [float(value) for value in hole]


def optional_point(point) -> list[float] | None:
    """Convert an optional 3-D point for JSON without iterating over None."""
    return None if point is None else [float(value) for value in point]


def head_context_from_targets(targets: dict) -> dict:
    """Adapt the orchestrator's first localization result for the left stage."""
    hinge = (targets.get("hinge") or {}).get("position_base_link_m")
    if hinge is None:
        raise SafetyError("编排器未提供 head_localize 的 hinge 三维坐标")
    hole = (targets.get("hole") or {}).get("position_base_link_m")
    return {
        "hinge_position_base_link_m": [float(value) for value in hinge],
        "hole_position_base_link_m": optional_point(hole),
    }


def hole_finalize_plan(hole_base_m) -> tuple[str, str]:
    """Decide how the left stage ends after the held-part gate.

    The only difference between a localized hole and a missing one is the final
    move: everything before it (grasp, 150 mm lift, same-target verification,
    both SpaceMouse gates) runs identically.
    """
    if hole_base_m is None:
        return "skip_hole_move", "left_stage_complete_without_hole_move"
    return "move_to_hole", "left_stage_complete_above_hole"


def move_left_above_hole(*, controller: G2Controller, hole_base_m, args, log, link: str = LEFT_VISUAL_TIP_LINK) -> dict:
    """Move the left end-link above the head-localized hole with a fixed pose.

    The grasped part stays closed in the gripper; only the end-link pose is
    converted to endpoint IK and sent as a planned joint request. The path is
    not Cartesian-constrained; actual endpoint position and orientation are checked.
    """
    try:
        tcp_end_m = np.asarray([float(value) for value in args.step_tcp_end_mm.split(",")], dtype=float) / 1000.0
        if tcp_end_m.shape != (3,) or not np.isfinite(tcp_end_m).all():
            raise ValueError
    except ValueError as exc:
        raise SafetyError("--step-tcp-end-mm 必须是三个有限数值") from exc
    hole_target = np.asarray(hole_base_m, dtype=float)
    if hole_target.shape != (3,) or not np.isfinite(hole_target).all():
        raise SafetyError(f"hole 目标三维坐标无效：{hole_base_m}")
    velocity_rad_s = float(args.preset_velocity_rad_s if args.auto_preset_velocity_rad_s is None
                           else args.auto_preset_velocity_rad_s)
    if not math.isfinite(velocity_rad_s) or velocity_rad_s <= 0:
        raise SafetyError("--auto-preset-velocity-rad-s 必须为正有限值")
    offsets_mm = np.asarray([args.hole_x_mm, args.hole_y_mm, args.hole_z_mm], dtype=float)
    if not np.isfinite(offsets_mm).all():
        raise SafetyError("--hole-x-mm/--hole-y-mm/--hole-z-mm 必须是有限数值")
    if abs(float(args.hole_x_mm)) > 500.0 or abs(float(args.hole_y_mm)) > 500.0:
        raise SafetyError("--hole-x-mm/--hole-y-mm 必须在 [-500, 500] mm")
    if not 0.0 < float(args.hole_z_mm) <= 500.0:
        raise SafetyError("--hole-z-mm 必须在 (0, 500] mm")

    orientation = fixed_forward_down_orientation(
        forward_yaw_deg=args.hole_forward_yaw_deg,
        gripper_pitch_deg=args.hole_gripper_pitch_deg,
        gripper_roll_deg=args.hole_gripper_roll_deg,
    )
    grasp_center_target = hole_target + offsets_mm / 1000.0
    target = gripper_end_target(grasp_center_target, orientation, tcp_end_m, 0.0)
    current = controller.read_end_effector_pose(link)
    travel_mm = float(np.linalg.norm(np.asarray(target.position_m) - np.asarray(current.position_m)) * 1000.0)
    max_travel_mm = float(getattr(args, "max_travel_mm", 450.0))
    if not (math.isfinite(max_travel_mm) and 0.0 < max_travel_mm <= 800.0):
        raise SafetyError("--max-travel-mm 必须在 (0, 800] mm")
    if travel_mm > max_travel_mm:
        raise SafetyError(
            f"当前末端到 hole 上方目标直线距离 {travel_mm:.1f} mm，超过 --max-travel-mm={max_travel_mm:.1f}"
        )
    start = {
        "event": "left_above_hole_motion_start", "travel_mm": round(travel_mm, 2),
        "control_method": "endpoint_ik_joint_plan", "control_mode": 3,
        "velocity_rad_s": velocity_rad_s,
        "hole_position_base_link_m": [float(value) for value in hole_target],
        "hole_offset_mm": [float(value) for value in offsets_mm],
        "grasp_center_target_base_link_m": [float(value) for value in grasp_center_target],
        "target_end_link_base_m": [float(value) for value in target.position_m],
        "orientation_xyzw": [float(value) for value in target.orientation_xyzw],
        "forward_yaw_deg": args.hole_forward_yaw_deg,
        "gripper_pitch_deg": args.hole_gripper_pitch_deg,
        "gripper_roll_deg": args.hole_gripper_roll_deg,
    }
    print(json.dumps(start, ensure_ascii=False), flush=True)
    json_record(log, start)
    try:
        from .left_hole_joint_move import move_left_hole_joint_target
    except ImportError:
        from left_hole_joint_move import move_left_hole_joint_target
    actual = move_left_hole_joint_target(controller, target, velocity_rad_s=velocity_rad_s)
    actual_center = gripper_end_target(
        np.asarray(actual.position_m), actual.orientation_xyzw, -tcp_end_m, 0.0,
    ).position_m
    xyz_error_mm = float(np.linalg.norm(np.asarray(actual_center) - grasp_center_target) * 1000.0)
    if xyz_error_mm > float(args.head_xy_tolerance_mm):
        raise SafetyError(
            f"夹持中心与偏移后 hole 目标误差 {xyz_error_mm:.1f} mm，"
            f"超过 {args.head_xy_tolerance_mm:.1f} mm"
        )
    arrived = {
        "event": "left_above_hole_arrived", "travel_mm": round(travel_mm, 2),
        "grasp_center_xyz_error_mm": round(xyz_error_mm, 2),
        "actual_end_link_base_m": [float(value) for value in actual.position_m],
    }
    print(json.dumps(arrived, ensure_ascii=False), flush=True)
    json_record(log, arrived)
    return arrived


def start_head_camera_web_preview(*, gdk, head_model_path: Path, args,
                                  shared: SharedView | None, camera=None, model=None,
                                  owns_camera: bool = True, owns_model: bool = True):
    """Keep the head RGB/YOLO view updating during coarse approach.

    Head preview is read-only: it never queries robot state or sends a
    command.  It uses the same initialized GDK session as the grasp process;
    the caller stops it before the left D405 perception stage begins.

    ``owns_camera``/``owns_model`` say whether this preview may close/delete the
    objects handed to it.  A single-process orchestrator injects a camera it
    still owns, so the preview must not release it: closing the same native
    camera twice aborts the interpreter inside the GDK binding.
    """
    if shared is None:
        return None, None
    stop = threading.Event()
    provided_camera = camera
    provided_model = model
    # A camera this function creates itself is always its own to close.
    closes_camera = provided_camera is None or bool(owns_camera)
    drops_model = provided_model is None or bool(owns_model)

    def run() -> None:
        camera = provided_camera
        model = provided_model
        sequence = 0
        last_timestamp = None
        try:
            if camera is None:
                camera = gdk.Camera([gdk.CameraType.kHeadColor])
            if model is None:
                model = load_yolo_segmentation_model(head_model_path, device=args.yolo_device)
            while not stop.is_set():
                try:
                    image = camera.get_latest_image(gdk.CameraType.kHeadColor, 1000.0)
                    timestamp_ns = int(image.timestamp_ns)
                    if timestamp_ns == last_timestamp:
                        stop.wait(0.01)
                        continue
                    last_timestamp = timestamp_ns
                    bgr = cv2.cvtColor(decode_color_rgb(image, gdk), cv2.COLOR_RGB2BGR)
                    masks, scores, classes = infer_yolo_masks(
                        model, bgr, confidence=args.head_yolo_confidence,
                        imgsz=640, device=args.yolo_device,
                    )
                    selected = select_leftmost_hinge(masks, scores, classes)
                    overlay = draw_head_coarse_overlay(bgr, masks, scores, classes, selected, 0, 0)
                    cv2.putText(
                        overlay, "HEAD camera live during coarse approach",
                        (12, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                        (0, 255, 255), 2, cv2.LINE_AA,
                    )
                    ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
                    if not ok:
                        raise RuntimeError("头部相机叠加图 JPEG 编码失败")
                    sequence += 1
                    shared.update(encoded.tobytes(), {
                        "state": "head_camera_live",
                        "source": "head_yolo_preview",
                        "sequence": sequence,
                        "camera_timestamp_ns": timestamp_ns,
                        "selected_mask_index": selected,
                        "detection_count": len(masks),
                        "head_yolo_confidence": args.head_yolo_confidence,
                        "safety": "头部相机仅用于实时网页预览，不参与左手运动控制。",
                    })
                except Exception as exc:
                    sequence += 1
                    shared.update(None, {
                        "state": "head_camera_preview_error",
                        "source": "head_yolo_preview",
                        "sequence": sequence,
                        "error": str(exc),
                    })
                    if stop.wait(0.25):
                        break
        except Exception as exc:
            sequence += 1
            shared.update(None, {
                "state": "head_camera_preview_stopped",
                "source": "head_yolo_preview",
                "sequence": sequence,
                "error": str(exc),
            })
        finally:
            if closes_camera and camera is not None:
                try:
                    camera.close_camera()
                except Exception:
                    pass
            if drops_model:
                del model

    worker = threading.Thread(target=run, name="head-camera-web-preview", daemon=True)
    worker.start()
    return stop, worker


def stop_head_camera_web_preview(stop: threading.Event | None,
                                 worker: threading.Thread | None) -> None:
    if stop is None:
        return
    stop.set()
    if worker is not None and worker is not threading.current_thread():
        worker.join(timeout=2.0)


def main(argv=None) -> int:
    """Standalone CLI: this process owns its own GDK session and SpaceMouse worker."""
    return run_left_cycle(services=None, argv=argv)


def run_left_cycle(*, services=None, argv=None) -> int:
    """Run the left-hand coarse-approach and visual grasp in this process.

    ``services`` carries the orchestrator's GDK session, controller, gate,
    SpaceMouse worker, cameras and preloaded models.  With ``services=None`` the
    stage creates and owns all of them, which keeps the original CLI intact.
    """
    args = parse_args(argv)
    if not (
        0 <= args.threshold <= 1
        and 0 <= args.yolo_confidence <= 1
        and 0 <= args.yolo_target_confidence <= 1
        and 0.5 <= args.head_yolo_confidence <= 1
        and args.yolo_imgsz > 0
        and 0.0 < args.yolo_center_fraction <= 1.0
        and args.shaft_association_max_distance_px > 0
        and args.shaft_association_max_horizontal_distance_px > 0
        and args.shaft_association_ambiguity_margin_px >= 0
        and args.sam_review_interval_s > 0
        and args.sam_review_ttl_s > 0
        and args.sam_review_max_pixel_error > 0
        and args.sam_review_max_depth_error_m > 0
        and 0.0 <= args.yolo_box_height_fraction <= 1.0
        and math.isfinite(args.head_above_mm)
        and 1.0 <= args.head_above_mm <= 500.0
        and args.min_area >= 1
        and args.min_valid_depth_pixels >= 1
        and args.warmup >= 0
        and args.max_frames >= 0
        and args.track_max_distance_px > 0
        and 0.0 < args.track_max_depth_jump_m <= 0.20
        and args.track_hold_frames >= 0
        and 0 <= args.web_port <= 65535
        and 0.0 < args.preset_velocity_rad_s
        and (
            args.auto_preset_velocity_rad_s is None
            or 0.0 < args.auto_preset_velocity_rad_s
        )
        and 0.001 <= args.preset_arrival_tolerance_rad <= 0.10
        and 1.0 <= args.preset_timeout_s <= 30.0
        and 1.0 <= args.pre_final_settle_timeout_s <= 10.0
        and 0.0 < args.final_descent_visual_max_step_m <= 0.002
        and 0.0 <= args.final_descent_visual_max_total_m <= 0.050
        and math.isfinite(args.final_descent_visual_tolerance_mm)
        and args.final_descent_visual_tolerance_mm > 0.0
        and 1 <= args.final_descent_target_lost_frames <= 10
        and 0.0 <= args.regrasp_retract_m <= 0.15
        and 0.005 <= args.post_grasp_verify_lift_m <= 0.050
        and 2 <= args.post_grasp_verify_frames <= 10
        and 1.0 <= args.post_grasp_verify_center_drift_px <= 80.0
        and 0.02 <= args.post_grasp_verify_area_change_fraction <= 0.50
        and 0.002 <= args.post_grasp_verify_depth_change_m <= 0.050
        and 0.50 <= args.post_grasp_monitor_max_stale_s <= 5.0
        and 0.0 <= args.post_grasp_lift_area_grace_s <= 10.0
        and 2 <= args.post_grasp_lift_area_window_frames <= 60
        and 0.05 <= args.post_grasp_lift_area_spread <= 5.0
        and 1 <= args.post_grasp_lift_area_confirmations <= 20
        and 0.10 <= args.post_grasp_target_lost_timeout_s <= 30.0
    ):
        print("参数错误：检测、预热、跟踪参数无效。", file=sys.stderr)
        return 2
    if args.full_cycle and (not args.allow_motion or args.stop_after != "none" or args.single_xy_step):
        print("--full-cycle 必须同时指定 --allow-motion，且不能与 --stop-after 或 --single-xy-step 合用。", file=sys.stderr)
        return 2
    if args.wait_for_next and (not args.full_cycle or not args.spacemouse_device):
        print("--wait-for-next 必须与 --full-cycle 和 --spacemouse-device 共用。", file=sys.stderr)
        return 2
    try:
        config, parameters, gripper_open, gripper_tolerance, gripper_close_timeout, empirical_pixel_jacobian, contact_parameters, adaptive_gripper = load_config(
            args.config,
            require_calibrated=requires_calibrated_config(
                allow_motion=args.allow_motion,
                stop_after=args.stop_after,
            ),
        )
        if args.relief_extra_lift_mm is not None and contact_parameters is not None:
            # Explicit CLI value (mm) wins over contact.relief_extra_lift_m.
            contact_parameters = replace(
                contact_parameters,
                relief_extra_lift_m=float(args.relief_extra_lift_mm) / 1000.0,
            )
            contact_parameters.validate()
        height_samples = None
        if args.height_reference is not None:
            height_samples = load_height_samples(args.height_reference)
        if args.step_profile is not None:
            if (not args.step_metric_xy and args.step_reference is None) or parameters is None or args.detector != "yolo":
                raise ValueError("STEP mode requires --step-reference, calibrated config and YOLO detector")
            args.step_geometry = load_profile(args.step_profile)
            args.final_descent_visual_servo = True
            if args.step_metric_xy:
                args.step_tcp_m = np.asarray([float(x) for x in args.step_tcp_end_mm.split(",")]) / 1000
                args.step_compensation_m = np.asarray([float(x) for x in args.step_compensation_xy_mm.split(",")]) / 1000
                if args.step_tcp_m.shape != (3,) or args.step_compensation_m.shape != (2,) or not np.isfinite(args.step_tcp_m).all() or not np.isfinite(args.step_compensation_m).all() or not 0 < args.step_xy_tolerance_mm <= 10:
                    raise ValueError("invalid approximate TCP / XY compensation / tolerance")
            else:
                args.step_alignment = load_alignment_reference(args.step_reference, args.step_geometry)
                args.step_height_samples = args.step_alignment.get("height_samples")
                parameters = replace(parameters, target_u_px=args.step_alignment["target_pixel_px"][0], target_v_px=args.step_alignment["target_pixel_px"][1])
        elif args.step_metric_xy:
            raise ValueError("--step-metric-xy requires --step-profile")
    except Exception as exc:
        print(f"抓取配置无效：{exc}", file=sys.stderr)
        return 2
    base_speed_mm_s = 10.0 if parameters is None else parameters.final_descent_speed_mps * 1000.0
    if not math.isfinite(args.xy_speed_mm_s) or args.xy_speed_mm_s <= 0.0:
        print("--xy-speed-mm-s must be a positive finite value", file=sys.stderr)
        return 2
    if args.speed_mm_s is not None and args.speed_scale is not None:
        print("--speed-mm-s 与旧兼容参数 --speed-scale 不能同时使用。", file=sys.stderr)
        return 2
    if args.speed_mm_s is None:
        speed_mm_s = base_speed_mm_s if args.speed_scale is None else base_speed_mm_s * args.speed_scale
    else:
        speed_mm_s = args.speed_mm_s
    if not math.isfinite(speed_mm_s) or not speed_mm_s > 0.0:
        print("参数错误：--speed-mm-s 必须是正的有限值。", file=sys.stderr)
        return 2
    if (
        not 1.0 <= args.spacemouse_speed_mm_s <= 30.0
        or not 1.0 <= args.spacemouse_rotation_deg_s <= 28.6
        or not 0.0 <= args.spacemouse_deadzone <= 0.5
    ):
        print("参数错误：SpaceMouse 速度、旋转速度或死区无效。", file=sys.stderr)
        return 2
    if args.spacemouse_device and not args.allow_motion:
        print("--spacemouse-device 必须与 --allow-motion 共用。", file=sys.stderr)
        return 2
    speed_scale = speed_mm_s / base_speed_mm_s
    motion_preset_velocity_rad_s = (
        args.preset_velocity_rad_s
        if args.auto_preset_velocity_rad_s is None
        else args.auto_preset_velocity_rad_s
    )
    motion_preset_timeout_s = args.preset_timeout_s
    if not math.isfinite(motion_preset_velocity_rad_s) or motion_preset_velocity_rad_s <= 0.0:
        print("参数错误：自动关节预设速度必须是正的有限值。", file=sys.stderr)
        return 2
    if parameters is not None:
        # The duration encodes the maximum descent distance, not a separate
        # speed multiplier. Preserve that distance at the explicit speed.
        descent_distance_m = parameters.final_descent_speed_mps * parameters.final_descent_duration_s
        parameters = replace(
            parameters,
            final_descent_speed_mps=speed_mm_s / 1000.0,
            final_descent_duration_s=descent_distance_m / (speed_mm_s / 1000.0),
        )
    premove_presets: list[tuple[str, list[float]]] | None = None
    post_grasp_presets: list[tuple[str, list[float]]] | None = None
    handoff_presets: list[tuple[str, list[float]]] | None = None
    workflow_home_target: list[float] | None = None
    if args.full_cycle:
        try:
            workflow_home_target = load_workflow_home(args.poses, args.home_pose_name)
            if not args.head_coarse_approach:
                premove_presets = load_premove_presets(args.poses)
            post_grasp_presets = load_post_grasp_presets(args.poses)
            if args.wait_for_next:
                handoff_presets = load_premove_presets(args.poses, HANDOFF_PRESETS)
        except Exception as exc:
            print(f"全流程姿态文件无效（含双臂归位点）：{exc}", file=sys.stderr)
            return 2
    if not torch.cuda.is_available():
        print("未检测到 CUDA，当前 YOLO 实时流程无法运行。", file=sys.stderr)
        return 2
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if args.detector in {"yolo-sam", "sam3"} and not checkpoint.is_file():
        print(f"找不到 SAM3 权重：{checkpoint}", file=sys.stderr)
        return 2
    yolo_model_path = args.yolo_model.expanduser().resolve()
    if args.detector in {"yolo", "yolo-sam"} and not yolo_model_path.is_file():
        print(f"找不到 G2 原生 YOLO 权重：{yolo_model_path}", file=sys.stderr)
        return 2
    head_model_path: Path | None = None
    if args.head_coarse_approach:
        if not args.full_cycle or not args.allow_motion:
            print("--head-coarse-approach 必须与 --full-cycle --allow-motion 共用。", file=sys.stderr)
            return 2
        head_model_path = args.head_yolo_model.expanduser().resolve()
        if not head_model_path.is_file():
            print(f"找不到头部 YOLO 权重：{head_model_path}", file=sys.stderr)
            return 2
    # Head coarse localization and the left D405 grasp now share this one GDK
    # client.  Do not create a child process: GDK camera/TF configuration is
    # session-scoped and duplicate initialization can race its DDS responses.
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
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
        "xy_speed_mm_s": args.xy_speed_mm_s,
        "speed_scale": speed_scale,
        "preset_velocity_rad_s": motion_preset_velocity_rad_s,
        "head_yolo_confidence": args.head_yolo_confidence if args.head_coarse_approach else None,
        "head_above_mm": args.head_above_mm if args.head_coarse_approach else None,
        "height_reference": None if args.height_reference is None else str(args.height_reference.expanduser().resolve()),
        "preset_timeout_s": motion_preset_timeout_s,
        "poses": None if not args.full_cycle else str(args.poses.expanduser().resolve()),
        "safety": (
            ((f"head RGB-D coarse approach ({args.head_above_mm:.0f} mm above hinge)→visual grasp→+Z 150 mm→P3→P4; "
              if args.head_coarse_approach else
              "single process: YOLO P1/P2→visual grasp→+Z 150 mm→P3→P4; ")
             + "one GDK initialization and no extra gripper command after grasp")
            if args.full_cycle else "P1/P2/P3/P4 are not commanded by this program"
        ),
    }, ensure_ascii=False))

    initialized = False
    camera = None
    # Only the head-coarse branch assigns this, but later code (height
    # reference, hole gate) tests it, so it must exist for the legacy P1/P2
    # path too.
    head_coarse_context: dict | None = None
    hole_target_base: list[float] | None = None
    # Guarded in ``finally``; objects injected by an orchestrator are reused
    # and must not be stopped or closed by this stage.
    owns_camera = True
    owns_worker = True
    perception: PersistentPerception | None = None
    position_stream: LeftEndEffectorPositionStream | None = None
    initial_gripper_open_stream: LeftEndEffectorPositionStream | None = None
    post_grasp_regrasp_open_stream: LeftEndEffectorPositionStream | None = None
    force_preview_stop: threading.Event | None = None
    force_preview_thread: threading.Thread | None = None
    force_preview_phase: dict[str, str] | None = None
    head_preview_stop: threading.Event | None = None
    head_preview_thread: threading.Thread | None = None
    web_server: ThreadingHTTPServer | None = None
    web_shared: SharedView | CompositeSharedView | None = None
    motion_gate: MotionPauseGate | None = None
    takeover_worker: SpaceMouseTakeoverWorker | None = None
    dianpi_runner: DianpiCycleRunner | None = None
    try:
        if services is not None and services.web is not None:
            # Single-process orchestrator: publish into its shared three-camera
            # page instead of starting a second HTTP server.
            web_shared = services.web
        elif args.web_port:
            web_shared = CompositeSharedView()
            web_server = ThreadingHTTPServer(
                (args.web_host, args.web_port),
                handler_class(web_shared, title="G2 左手 D405 / YOLO（头部粗定位阶段预览）"),
            )
            threading.Thread(target=web_server.serve_forever, daemon=True).start()
            model_note = "YOLO 实时分割" if args.detector == "yolo" else "与抓取共用同一 SAM3 模型"
            print(f"内嵌实时页面：http://{args.web_host}:{args.web_port}（头部粗定位阶段双画面；左手阶段仅左手 D405：{model_note}）")
        if services is None:
            if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
                raise RuntimeError("GDK 初始化失败")
            initialized = True
            robot = agibot_gdk.Robot()
            time.sleep(1.0)
        else:
            robot = services.robot
        if services is not None and services.controller is not None:
            controller = services.controller
            motion_gate = services.gate
        else:
            if args.spacemouse_device:
                motion_gate = MotionPauseGate()
            controller = G2Controller(
                agibot_gdk,
                robot,
                allow_motion=args.allow_motion,
                control_period_s=args.control_period_s,
                motion_gate=motion_gate,
            )
            controller.checked_arm_state()
        if services is not None:
            dianpi_runner = services.dianpi_runner
        if args.spacemouse_device and dianpi_runner is None:
            # Keep the RM1010 P10 monitor alive for the whole integrated
            # process, including the head-camera coarse stage.  The monitor
            # and SpaceMouse chord share this runner/controller.
            dianpi_runner = DianpiCycleRunner(
                event_sink=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
            )
            dianpi_runner.start_p10_monitor()
        if args.full_cycle and not args.skip_workflow_home:
            assert workflow_home_target is not None
            print(json.dumps({
                "event": "workflow_home_start",
                "preset": args.home_pose_name,
                "gdk_session": "reused",
                "target_joint_positions_rad": workflow_home_target,
            }, ensure_ascii=False), flush=True)
            home_log = log_path.open("a", encoding="utf-8")
            try:
                home_record = move_workflow_home(
                    controller,
                    workflow_home_target,
                    velocity_rad_s=motion_preset_velocity_rad_s,
                    arrival_tolerance_rad=args.preset_arrival_tolerance_rad,
                    timeout_s=motion_preset_timeout_s,
                )
                home_record["preset"] = args.home_pose_name
                home_record["utc"] = datetime.now(timezone.utc).isoformat()
                json_record(home_log, home_record)
                print(json.dumps(home_record, ensure_ascii=False), flush=True)
            except Exception as exc:
                print(f"双臂归位失败：{exc}", file=sys.stderr)
                return 1
            finally:
                home_log.close()
        tf = agibot_gdk.TF()
        if args.head_coarse_approach:
            assert head_model_path is not None
            print(json.dumps({"event": "head_coarse_approach_start", "gdk_session": "reused"}, ensure_ascii=False), flush=True)
            # The main perception loop opens ``log`` later in this function.
            # Use a short-lived handle here so head localization is recorded
            # without creating a second GDK client or relying on that later
            # lexical binding.
            head_log = log_path.open("a", encoding="utf-8")
            try:
                head_preview_stop, head_preview_thread, head_coarse_context = run_head_coarse_approach_in_process(
                    gdk=agibot_gdk, controller=controller, tf=tf, args=args,
                    head_model_path=head_model_path, log=head_log,
                    shared=None if web_shared is None else web_shared.pane("head"),
                    head_camera=None if services is None else services.head_camera,
                    head_model=None if services is None else services.head_yolo,
                    prelocalized_targets=None if services is None else services.get("head_targets"),
                )
            except Exception as exc:
                print(f"头部铰链定位/左手预移动失败：{exc}", file=sys.stderr)
                print("头部粗定位未到位；不启动左手 D405 视觉伺服。", file=sys.stderr)
                return 1
            finally:
                head_log.close()
            print(json.dumps({"event": "head_coarse_approach_complete", "xy_tolerance_mm": args.head_xy_tolerance_mm, "gdk_session": "reused"}, ensure_ascii=False), flush=True)
            # The coarse head stage is complete.  Do not keep showing its
            # frame while the left camera/model are being prepared.
            if head_preview_stop is not None:
                stop_head_camera_web_preview(head_preview_stop, head_preview_thread)
                head_preview_stop = head_preview_thread = None
                if hasattr(web_shared, "hide_pane"):
                    web_shared.hide_pane("head")
        # The hole may legitimately be missing: grasp and lift anyway, then hold
        # and wait for the operator instead of moving above the hole.
        hole_target_base = hole_target_from_context(head_coarse_context)
        if services is not None and services.worker is not None:
            takeover_worker = services.worker
            owns_worker = False
        elif args.spacemouse_device:
            takeover_worker = SpaceMouseTakeoverWorker(
                controller,
                motion_gate,
                device=args.spacemouse_device,
                config=TeleopConfig(
                    side="left",
                    speed_m_s=args.spacemouse_speed_mm_s / 1000.0,
                    angular_speed_rad_s=math.radians(args.spacemouse_rotation_deg_s),
                    deadzone=args.spacemouse_deadzone,
                ),
                allow_advance_right=True,
                dianpi_runner=dianpi_runner,
                event_sink=lambda event: print(json.dumps(event, ensure_ascii=False), flush=True),
            )
            takeover_worker.start()
        camera_type = agibot_gdk.CameraType.kHandLeftColor
        depth_type = agibot_gdk.CameraType.kHandLeftDepth
        owns_camera = services is None or services.left_camera is None
        camera = (
            services.left_camera if not owns_camera
            else agibot_gdk.Camera([camera_type] if args.step_profile is not None else [camera_type, depth_type])
        )
        processor = None
        yolo_model = services.left_yolo if services is not None and services.left_yolo is not None else None
        if yolo_model is None and args.detector in {"yolo", "yolo-sam"}:
            print(f"加载 G2 原生 YOLO 主检测器：{yolo_model_path}", flush=True)
            yolo_model = load_yolo_segmentation_model(yolo_model_path, device=args.yolo_device)
        # The left worker remains the sole owner of the hand camera/model.
        if services is not None:
            raw_feed = services.get('left_raw_feed')
            if raw_feed is not None:
                raw_feed.pause()
        perception = PersistentPerception(
            camera=camera, color_type=camera_type, depth_type=depth_type,
            gdk=agibot_gdk, tf=tf, processor=processor, yolo_model=yolo_model,
            args=args, shared=None if web_shared is None else web_shared.pane("left"),
            yolo_model_path=yolo_model_path,
        )
        perception.start()
        coarse_parameters = None
        if args.step_profile is not None and parameters is not None:
            coarse_parameters = replace(parameters,
                target_u_px=float(config["alignment"]["target_pixel_px"][0]),
                target_v_px=float(config["alignment"]["target_pixel_px"][1]))
        state = None if parameters is None else LeftGraspState(parameters, coarse_parameters)
        color_intrinsic = depth_intrinsic = depth_to_color = end_link_transform = None
        tracked_center: np.ndarray | None = None
        tracked_depth_m: float | None = None
        last_height_target_pixel: np.ndarray | None = None
        last_height_target_detail: dict | None = None
        if height_samples:
            last_height_target_pixel, last_height_target_detail = interpolate_target_pixel(
                height_samples, height_samples[-1]["camera_distance_m"],
            )
        missed_frames = 0
        regrasp_attempts = 0
        last_end_link_target: CartesianPoseTarget | None = None
        step_xy_pending = None
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
        # One fresh same-target frame must confirm the part is still held after
        # the monitored lift before the arm moves toward the hole.
        post_grasp_hole_monitor: dict | None = None
        post_grasp_p4_sequence: NonBlockingLeftPresetSequence | None = None
        post_grasp_p4_monitor: dict | None = None
        post_grasp_regrasp_open: dict | None = None
        post_grasp_verify_motion: dict | None = None
        post_grasp_target_lost_since: float | None = None
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
            if args.move_above_hole and hole_target_base is None:
                warning = {
                    "event": "head_hole_missing",
                    "level": "warning",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "reason": "头部定位没有可用的 class 1 hole 三维目标",
                    "action": "继续抓取并上抬 150 mm；随后停在原地等待 SpaceMouse 下一阶段指令，不做 hole 移动",
                }
                json_record(log, warning)
                print(
                    "警告：头部没有识别到 hole；仍会抓取并上抬，然后停在上抬位置等待下一阶段指令。",
                    flush=True,
                )
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
                "height_reference": None if args.height_reference is None else str(args.height_reference.expanduser().resolve()),
                "workflow_home_pose": args.home_pose_name if args.full_cycle else None,
                "final_descent_speed_mps": None if parameters is None else parameters.final_descent_speed_mps,
                "final_descent_duration_s": None if parameters is None else parameters.final_descent_duration_s,
                "final_descent_visual_servo": args.final_descent_visual_servo,
                "final_descent_visual_tolerance_mm": args.final_descent_visual_tolerance_mm,
                "post_grasp_target_lost_timeout_s": args.post_grasp_target_lost_timeout_s,
                "full_cycle": args.full_cycle,
                "perception_architecture": "one GDK session; left perception owns hand camera + YOLO, head preview owns head camera + YOLO, state machine consumes latest snapshots only",
            })
            json_record(log, {"event": "gdk_motion_status_before_perception", **controller.motion_status_summary()})
            if args.full_cycle:
                if args.head_coarse_approach:
                    controller.prepare_joint_impedance_control(("left",))
                    assert gripper_close_timeout is not None and gripper_open is not None
                    initial_gripper_open_stream = controller.start_left_gripper_position_stream(gripper_open)
                    initial_gripper_open = {
                        "deadline": time.monotonic() + gripper_close_timeout,
                        "next_command_at": time.monotonic(), "attempts": 0,
                    }
                    json_record(log, {"event":"legacy_premove_skipped_after_head_coarse_approach", "utc":datetime.now(timezone.utc).isoformat()})
                else:
                    assert premove_presets is not None
                    json_record(log, {
                        "event": "premove_start_after_yolo_load",
                        "utc": datetime.now(timezone.utc).isoformat(),
                        "presets": [name for name, _ in premove_presets],
                        "detector": "yolo",
                    })
                    premove_sequence = NonBlockingLeftPresetSequence(
                        premove_presets, motion_preset_velocity_rad_s,
                        args.preset_arrival_tolerance_rad, motion_preset_timeout_s, "premove",
                    )
            if not args.full_cycle and args.allow_motion and args.stop_after == "none":
                assert gripper_open is not None and gripper_tolerance is not None and gripper_close_timeout is not None
                initial_gripper_open_stream = controller.start_left_gripper_position_stream(gripper_open)
                initial_gripper_open = {
                    "deadline": time.monotonic() + gripper_close_timeout,
                    "next_command_at": time.monotonic(), "attempts": 0,
                }

            def execute_handoff_before_exit() -> int:
                nonlocal position_stream, initial_gripper_open_stream, post_grasp_regrasp_open_stream
                assert handoff_presets is not None
                if takeover_worker is not None:
                    takeover_worker.stop_for_final_motion()
                controller.motion_gate = None
                for stream in (position_stream, initial_gripper_open_stream, post_grasp_regrasp_open_stream):
                    if stream is not None:
                        stream.stop()
                position_stream = initial_gripper_open_stream = post_grasp_regrasp_open_stream = None
                opened = open_handoff_gripper(controller)
                json_record(log, {"utc": datetime.now(timezone.utc).isoformat(), **opened})
                print(json.dumps(opened, ensure_ascii=False), flush=True)
                sequence = NonBlockingLeftPresetSequence(
                    handoff_presets, motion_preset_velocity_rad_s,
                    args.preset_arrival_tolerance_rad, motion_preset_timeout_s, "handoff",
                )
                while True:
                    complete, event = sequence.tick(controller, log)
                    if event is not None:
                        print(json.dumps(event, ensure_ascii=False), flush=True)
                    if complete:
                        json_record(log, {
                            "event": "left_handoff_complete_before_gdk_release",
                            "utc": datetime.now(timezone.utc).isoformat(),
                        })
                        print("左夹爪已张开，left_handoff_1→left_handoff_2 已到位；现在关闭左手 GDK。", flush=True)
                        return EXIT_ADVANCE_RIGHT
                    time.sleep(0.01)

            def wait_for_manual_next() -> int:
                if not args.wait_for_next:
                    return 0
                if takeover_worker is None:
                    raise SafetyError("--wait-for-next 需要已启动的 SpaceMouse 接管线程")
                json_record(log, {
                    "event": "left_stage_complete_waiting_for_manual_next",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "source": "spacemouse_right_long_press_2s",
                    "takeover_available": True,
                })
                print("左手流程已完成；可双击任一按钮临时接管，右按钮长按 2 秒启动右手流程。", flush=True)
                while True:
                    takeover_worker.raise_if_failed()
                    if takeover_worker.advance_requested:
                        return execute_handoff_before_exit()
                    time.sleep(0.01)

            def current_height_reference_target(
                *, end_link_transform: dict | None, item: dict | None = None,
            ) -> tuple[np.ndarray, dict] | None:
                """Interpolate the calibrated screen target without hand depth."""
                nonlocal last_height_target_pixel, last_height_target_detail
                if not height_samples:
                    return None
                distance_m = None
                source = None
                if head_coarse_context is not None and end_link_transform is not None:
                    try:
                        actual_pose = measured_left_end_link_pose(controller)
                        distance_m = camera_distance_from_base_point(
                            head_coarse_context["hinge_position_base_link_m"],
                            actual_pose.position_m,
                            actual_pose.orientation_xyzw,
                            end_link_transform,
                        )
                        source = "head_base_target_and_actual_end_link"
                    except Exception:
                        distance_m = None
                if distance_m is None and item is not None:
                    xyz = item.get("xyz_color_camera_m")
                    if xyz is not None:
                        try:
                            candidate = float(np.linalg.norm(np.asarray(xyz, dtype=np.float64)))
                            if math.isfinite(candidate) and candidate > 0.0:
                                distance_m = candidate
                                source = "left_hand_depth"
                        except (TypeError, ValueError):
                            pass
                if distance_m is None:
                    if last_height_target_pixel is None:
                        return None
                    detail = dict(last_height_target_detail or {})
                    detail.update({"mode": "held_last_valid", "source": "last_valid_height_reference"})
                    return last_height_target_pixel.copy(), detail
                target, detail = interpolate_target_pixel(height_samples, distance_m)
                detail = {**detail, "source": source}
                last_height_target_pixel = target
                last_height_target_detail = detail
                return target.copy(), detail

            while args.max_frames == 0 or frame < args.max_frames:
                controller.check_automatic_guard()
                if takeover_worker is not None:
                    takeover_worker.raise_if_failed()
                    if takeover_worker.advance_requested:
                        json_record(log, {
                            "event": "advance_right_requested",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "source": "spacemouse_right_long_press_2s",
                        })
                        return execute_handoff_before_exit()
                    if motion_gate is not None and motion_gate.is_paused():
                        time.sleep(0.01)
                        continue
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
                        if services is not None and services.get('vlm_gripper_monitor') is not None:
                            services.get('vlm_gripper_monitor').set_phase('idle')
                        post_grasp_regrasp_open_stream = controller.start_left_gripper_position_stream(gripper_open)
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
                        controller.prepare_joint_impedance_control(("left",))
                        mode_after_visual = controller.motion_status_summary()
                        if mode_after_visual["control_mode"] != 3 or mode_after_visual["error_code"] != 0:
                            raise SafetyError(
                                f"P3 后未进入健康关节阻抗 control_mode=3；拒绝张开夹爪或开始视觉伺服：{mode_after_visual}"
                            )
                        json_record(log, {
                            "event": "premove_complete",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "final_detection_preset": "left_p2",
                            "nonblocking": True,
                        })
                        premove_sequence = None
                        if args.allow_motion and args.stop_after == "none":
                            assert gripper_close_timeout is not None
                            assert gripper_open is not None
                            initial_gripper_open_stream = controller.start_left_gripper_position_stream(gripper_open)
                            initial_gripper_open = {
                                "deadline": time.monotonic() + gripper_close_timeout,
                                "next_command_at": time.monotonic(), "attempts": 0,
                            }
                    continue
                if initial_gripper_open is not None:
                    assert gripper_open is not None and gripper_tolerance is not None
                    assert initial_gripper_open_stream is not None
                    now = time.monotonic()
                    initial_gripper_open_stream.raise_if_failed()
                    initial_gripper_open["attempts"] += 1
                    position = controller.read_gripper_position("left")
                    if position is not None and abs(position - gripper_open) <= gripper_tolerance:
                        json_record(log, {
                            "event": "gripper_open_confirmed",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "actual_position_rad": position,
                            "attempts": initial_gripper_open["attempts"],
                            "nonblocking": True,
                        })
                        initial_gripper_open_stream.stop()
                        initial_gripper_open_stream = None
                        initial_gripper_open = None
                    elif now > initial_gripper_open["deadline"]:
                        initial_gripper_open_stream.stop()
                        initial_gripper_open_stream = None
                        raise SafetyError(
                            "夹爪张开反馈超时；未进入视觉伺服、下探或后续预设动作："
                            f"target={gripper_open:.6f}rad, actual={position}, "
                            f"attempts={initial_gripper_open['attempts']}"
                        )
                    continue
                if post_grasp_regrasp_open is not None:
                    assert gripper_open is not None and gripper_tolerance is not None
                    assert post_grasp_regrasp_open_stream is not None
                    recovery = post_grasp_regrasp_open
                    now = time.monotonic()
                    post_grasp_regrasp_open_stream.raise_if_failed()
                    recovery["attempts"] += 1
                    position = controller.read_gripper_position("left")
                    if position is not None and abs(position - gripper_open) <= gripper_tolerance:
                        post_grasp_regrasp_open_stream.stop()
                        post_grasp_regrasp_open_stream = None
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
                        post_grasp_regrasp_open_stream.stop()
                        post_grasp_regrasp_open_stream = None
                        raise SafetyError(
                            "掉件后夹爪张开反馈超时；停止后续命令并保持当前实测状态："
                            f"target={gripper_open:.6f}rad, actual={position}, attempts={recovery['attempts']}"
                        )
                    continue
                # Once the part has passed the short visual verification,
                # transport is advanced by polling only.  Do not enter a
                # nested wait/sleep loop: the perception worker keeps the web
                # page live and the mailbox can still veto the lift.
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
                            check_geometry=False,
                        )
                        # A tracked mask is not proof of a held part.  Feed the
                        # closed-gripper lift with the observed mask areas and
                        # veto transport as soon as the area keeps changing
                        # instead of settling (empty grasp / part sliding out).
                        if (
                            monitor["failure"] is None
                            and observation.get("valid") is True
                            and isinstance(observation.get("area_px"), (int, float))
                            and observed_at - monitor["lift_started_at"]
                            >= args.post_grasp_lift_area_grace_s
                        ):
                            monitor["area_samples"].append(
                                (observed_at, float(observation["area_px"]))
                            )
                            monitor["area_samples"][:] = monitor["area_samples"][-args.post_grasp_lift_area_window_frames:]
                            # Diagnostic only: with a wrist camera a held part
                            # keeps its image centre while a part left behind
                            # slides away, so log the drift next to the area.
                            monitor["center_drift_px"] = float(np.linalg.norm(
                                np.asarray(observation["center_px"], dtype=np.float64)
                                - np.asarray(monitor["reference_center_px"], dtype=np.float64)
                            ))
                            instability = held_part_area_instability(
                                monitor["area_samples"],
                                max_spread=args.post_grasp_lift_area_spread,
                                window_frames=args.post_grasp_lift_area_window_frames,
                            )
                            if instability is not None:
                                monitor["max_area_spread_fraction"] = max(
                                    monitor["max_area_spread_fraction"],
                                    instability["spread_fraction"],
                                )
                                monitor["area_instability"] = instability
                                # "仍在变化" must persist: a single jittery
                                # window is not enough.  Require consecutive
                                # unstable windows before dropping the part.
                                monitor["area_unstable_streak"] = (
                                    monitor["area_unstable_streak"] + 1
                                    if instability["unstable"]
                                    else 0
                                )
                                if (
                                    instability["unstable"]
                                    and monitor["area_unstable_streak"]
                                    >= args.post_grasp_lift_area_confirmations
                                ):
                                    monitor["failure"] = instability
                                    json_record(log, {
                                        "event": "post_grasp_lift_area_unstable",
                                        "utc": datetime.now(timezone.utc).isoformat(),
                                        "reason": instability["reason"],
                                        "area": instability,
                                        "center_drift_px": monitor.get("center_drift_px"),
                                        "unstable_streak": monitor["area_unstable_streak"],
                                        "confirmations": args.post_grasp_lift_area_confirmations,
                                        "action": "cancel_lift_open_gripper_and_regrasp",
                                        "nonblocking": True,
                                    })
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
                        # Either the mailbox confirmed target loss after the
                        # configured timeout, or the trailing mask-area history
                        # showed that the closed gripper never actually held the
                        # part.  Hold immediately; the existing recovery path
                        # then opens and restarts acquisition.
                        failure = monitor["failure"]
                        area_unstable = failure.get("reason") == HELD_PART_AREA_UNSTABLE_REASON
                        position_stream.update_target(CartesianPoseTarget(
                            actual.position_m, last_end_link_target.orientation_xyzw,
                        ))
                        position_stream.stop()
                        position_stream = None
                        assert gripper_open is not None and gripper_tolerance is not None and gripper_close_timeout is not None
                        if services is not None and services.get('vlm_gripper_monitor') is not None:
                            services.get('vlm_gripper_monitor').set_phase('idle')
                        lift_before_visual_reacquire(controller, speed_mm_s=args.xy_speed_mm_s)
                        post_grasp_regrasp_open_stream = controller.start_left_gripper_position_stream(gripper_open)
                        post_grasp_regrasp_open = {
                            "reason": (
                                "held_target_area_unstable_during_pre_p4_lift"
                                if area_unstable
                                else "held_target_lost_after_timeout_during_pre_p4_lift"
                            ),
                            "monitor": failure,
                            "area_monitor": monitor.get("area_instability"),
                            "orientation_xyzw": last_end_link_target.orientation_xyzw,
                            "deadline": time.monotonic() + gripper_close_timeout,
                            "next_command_at": time.monotonic() + 0.10,
                            "attempts": 1,
                        }
                        json_record(log, {
                            "event": "post_grasp_lift_monitor_regrasp_started",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "monitor": failure,
                            "area_unstable": area_unstable,
                            "area_monitor": monitor.get("area_instability"),
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
                            "area_monitor": monitor.get("area_instability"),
                            "area_monitor_center_drift_px": monitor.get("center_drift_px"),
                            "area_monitor_samples": len(monitor["area_samples"]),
                            "area_monitor_max_spread_fraction": monitor["max_area_spread_fraction"],
                            "area_monitor_grace_s": args.post_grasp_lift_area_grace_s,
                            "area_monitor_window_frames": args.post_grasp_lift_area_window_frames,
                            "area_monitor_max_spread": args.post_grasp_lift_area_spread,
                            "nonblocking": True,
                        })
                        post_grasp_lift_monitor = None
                        if args.move_above_hole:
                            hole_base = hole_target_base
                            # Identical flow either way: the same held-part gate
                            # below must still confirm the part is held.  Only
                            # the final move differs when no hole was localized.
                            # Do not start the hole move straight from the lift
                            # arrival.  The P3/P4 path required one fresh
                            # same-target frame here, which closes the race
                            # where the part drops exactly as the lift ends;
                            # the hole move must keep that gate.
                            post_grasp_hole_monitor = {
                                "mailbox": monitor["mailbox"],
                                "mailbox_sequence": monitor["mailbox_sequence"],
                                "reference_center_px": monitor["reference_center_px"],
                                "reference_area_px": monitor["reference_area_px"],
                                "last_observation_at": time.monotonic(),
                                "stale_since": None,
                                "stale_logged": False,
                                "paused": False,
                                "failure": None,
                                "awaiting_fresh_observation": True,
                                "hole_base_m": None if hole_base is None else [float(value) for value in hole_base],
                                "deadline": time.monotonic() + max(
                                    10.0,
                                    3.0 * float(args.post_grasp_target_lost_timeout_s) + 5.0,
                                ),
                            }
                            perception.set_phase(
                                "post_grasp_hole",
                                "同目标确认仍然夹持后才结束本阶段（有 hole 时再移动到 hole 上方）",
                            )
                            json_record(log, {
                                "event": "post_grasp_hole_gate_wait",
                                "utc": datetime.now(timezone.utc).isoformat(),
                                "reason": "require_one_fresh_same_target_frame_after_lift",
                                "hole_position_base_link_m": optional_point(hole_base),
                                "target_lost_timeout_s": args.post_grasp_target_lost_timeout_s,
                                "action": "hold_position_keep_gripper_closed",
                                "nonblocking": True,
                            })
                            continue
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
                            # P4 starts only after one fresh post-lift frame.
                            # Once the first preset request is sent, the
                            # reference geometry is no longer comparable at
                            # preset speed and the action state machine must
                            # stop consuming visual results until P4 arrives.
                            "motion_started": False,
                            # A fresh observation after the lift is mandatory
                            # before P4 is even requested.  It closes the
                            # race where the part drops just as the lift ends.
                            "awaiting_fresh_observation": True,
                        }
                        perception.set_phase(
                            "post_grasp_p4",
                            "same-target monitoring before P4; a fresh frame is required before the request",
                        )
                    elif time.monotonic() > monitor["deadline"]:
                        stalled = measured_left_end_link_pose(controller)
                        stalled_delta = np.asarray(stalled.position_m) - np.asarray(monitor["origin"].position_m)
                        raise SafetyError(
                            "受监控的 P4 前上抬超时；保持当前位置且拒绝进入 P4"
                            f"（目标上抬 {PRE_P4_LIFT_M * 1000:.1f} mm，"
                            f"实测 Δz {float(stalled_delta[2]) * 1000:.1f} mm，"
                            f"横向 {float(np.linalg.norm(stalled_delta[:2])) * 1000:.1f} mm，"
                            f"容差 {PRE_P4_LIFT_TOLERANCE_M * 1000:.1f} mm）"
                        )
                    # This branch is intentionally non-blocking, but a tight
                    # GDK polling loop can starve the perception worker under
                    # CPython's GIL.  Yield briefly so YOLO keeps publishing.
                    time.sleep(0.01)
                    continue
                if post_grasp_hole_monitor is not None:
                    monitor = post_grasp_hole_monitor
                    observation = monitor["mailbox"].newer_than(monitor["mailbox_sequence"])
                    if observation is not None:
                        monitor["mailbox_sequence"] = int(observation["sequence"])
                    now = time.monotonic()
                    gate = held_part_gate_step(
                        monitor,
                        observation,
                        now=now,
                        max_stale_s=args.post_grasp_monitor_max_stale_s,
                        center_drift_px=args.post_grasp_verify_center_drift_px,
                        area_change_fraction=args.post_grasp_verify_area_change_fraction,
                    )
                    if gate == "stale":
                        # Stale perception is an unknown state, never proof of a
                        # dropped part.  Hold position, keep the gripper closed,
                        # and wait for a fresh frame instead of moving blind.
                        json_record(log, {
                            "event": "post_grasp_hole_monitor_stale_hold",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "reason": "perception_stale_unknown_not_confirmed_lost",
                            "stale_age_s": round(now - monitor["last_observation_at"], 3),
                            "max_stale_s": args.post_grasp_monitor_max_stale_s,
                            "action": "hold_before_hole_move_and_keep_gripper_closed",
                            "nonblocking": True,
                        })
                    if gate == "lost":
                        # Confirmed loss right after the lift: open and reacquire
                        # through the same generic recovery path P4 uses.  The
                        # hole move has not started, so this is safe.
                        stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                        force_preview_stop = force_preview_thread = None
                        force_preview_phase = None
                        assert gripper_open is not None and gripper_close_timeout is not None
                        if services is not None and services.get('vlm_gripper_monitor') is not None:
                            services.get('vlm_gripper_monitor').set_phase('idle')
                        if position_stream is not None:
                            position_stream.stop()
                            position_stream = None
                        lift_before_visual_reacquire(controller, speed_mm_s=args.xy_speed_mm_s)
                        post_grasp_regrasp_open_stream = controller.start_left_gripper_position_stream(gripper_open)
                        post_grasp_regrasp_open = {
                            "reason": "held_target_lost_after_timeout_before_hole_move",
                            "monitor": monitor["failure"],
                            "orientation_xyzw": last_end_link_target.orientation_xyzw,
                            "deadline": time.monotonic() + gripper_close_timeout,
                            "next_command_at": time.monotonic() + 0.10,
                            "attempts": 1,
                        }
                        json_record(log, {
                            "event": "post_grasp_hole_monitor_regrasp_started",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "monitor": monitor["failure"],
                            "nonblocking": True,
                        })
                        post_grasp_hole_monitor = None
                        continue
                    if gate == "proceed":
                        hole_base = monitor["hole_base_m"]
                        action, complete_event = hole_finalize_plan(hole_base)
                        json_record(log, {
                            "event": "post_grasp_hole_verified",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "reason": "fresh_same_target_frame_confirms_the_part_is_still_held",
                            "mailbox_sequence": monitor["mailbox_sequence"],
                            "next_action": action,
                            "hole_position_base_link_m": (
                                None if hole_base is None else [float(value) for value in hole_base]
                            ),
                            "nonblocking": True,
                        })
                        post_grasp_hole_monitor = None
                        if action == "skip_hole_move":
                            json_record(log, {
                                "event": complete_event,
                                "utc": datetime.now(timezone.utc).isoformat(),
                                "reason": "head_hole_missing",
                                "note": "除不移动到 hole 上方外，后续流程与识别到 hole 时完全一致",
                                "gripper": "preserved_closed",
                                "nonblocking": True,
                            })
                            print(
                                "未识别到 hole：跳过 hole 上方移动，本阶段结束（其余流程与正常一致）。",
                                flush=True,
                            )
                            return 0
                        move_left_above_hole(
                            controller=controller, hole_base_m=hole_base, args=args, log=log,
                        )
                        done = {
                            "event": complete_event,
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "hole_position_base_link_m": [float(value) for value in hole_base],
                        }
                        json_record(log, done)
                        print("左手已到 hole 上方；本阶段完成。", flush=True)
                        return 0
                    if now >= monitor["deadline"]:
                        raise SafetyError(
                            "抓取并上抬后一直拿不到新的同目标观测；无法确认仍然夹持，"
                            "拒绝盲目移动到 hole 上方"
                        )
                    time.sleep(0.01)
                    continue
                if post_grasp_p4_sequence is not None:
                    assert post_grasp_p4_monitor is not None
                    monitor = post_grasp_p4_monitor
                    if not monitor.get("motion_started"):
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
                                check_geometry=False,
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
                            # Before P4 starts, stale perception is an
                            # unknown state. Do not issue the request until a
                            # fresh frame arrives, but never open the gripper
                            # without a fresh confirmed-loss observation.
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
                                    "action": "hold_before_p4_and_keep_gripper_closed",
                                    "mailbox_sequence": monitor["mailbox_sequence"],
                                    "nonblocking": True,
                                })
                        if monitor.get("failure") is not None:
                            failure = monitor["failure"]
                            # The P4 request has not started yet, so it is
                            # safe to open and reacquire at the verified lift
                            # pose after the configured target-loss timeout.
                            # Once motion_started is latched below, visual
                            # results are deliberately ignored until the
                            # preset sequence arrives.
                            assert not monitor.get("motion_started")
                            stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                            force_preview_stop = force_preview_thread = None
                            force_preview_phase = None
                            assert gripper_open is not None and gripper_close_timeout is not None
                            if services is not None and services.get('vlm_gripper_monitor') is not None:
                                services.get('vlm_gripper_monitor').set_phase('idle')
                            if position_stream is not None:
                                position_stream.stop()
                                position_stream = None
                            lift_before_visual_reacquire(controller, speed_mm_s=args.xy_speed_mm_s)
                            post_grasp_regrasp_open_stream = controller.start_left_gripper_position_stream(gripper_open)
                            post_grasp_regrasp_open = {
                                "reason": "held_target_lost_after_timeout_before_p4",
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
                        if monitor.get("paused") or monitor["awaiting_fresh_observation"]:
                            # A fresh post-lift frame is the last visual gate
                            # before P4. No preset request is sent while it is
                            # stale or absent.
                            time.sleep(0.01)
                            continue
                        # Lock this before calling tick(). If a GDK request
                        # fails, the outer error path must not attempt a visual
                        # regrasp against a possibly moving arm.
                        monitor["motion_started"] = True
                        json_record(log, {
                            "event": "post_grasp_p4_visual_monitor_suspended",
                            "utc": datetime.now(timezone.utc).isoformat(),
                            "reason": "rapid_preset_motion_geometry_not_comparable",
                            "action": "keep_gripper_closed_and_poll_joint_arrival",
                            "mailbox_sequence": monitor["mailbox_sequence"],
                            "nonblocking": True,
                        })
                        perception.set_phase(
                            "post_grasp_p4_in_motion",
                            "preview only; control state machine does not consume detections",
                        )
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
                        print("完整流程完成：P1→P2、抓取、受监控 +Z 上抬 150 mm、P3→P4；夹爪保持闭合。")
                        return wait_for_manual_next()
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
                    screen_target_pixel_px = np.asarray(snapshot["screen_gripper_target_pixel_px"], dtype=np.float64)
                    height_target = current_height_reference_target(
                        end_link_transform=end_link_transform,
                    )
                    if height_target is not None:
                        screen_target_pixel_px, record_height_detail = height_target
                        record["height_reference"] = record_height_detail
                        perception.set_target_pixel_override(screen_target_pixel_px, record_height_detail)
                    record["screen_gripper_target_pixel_px"] = screen_target_pixel_px.tolist()
                    if state is not None and args.step_profile is None:
                        parameters = replace(
                            parameters,
                            target_u_px=float(screen_target_pixel_px[0]),
                            target_v_px=float(screen_target_pixel_px[1]),
                        )
                        state.fine_parameters = parameters
                        if not state.coarse_active:
                            state.parameters = parameters
                    roi = snapshot["roi"]
                    if args.detector in {"yolo", "yolo-sam"}:
                        record["yolo"] = {
                            "model": str(yolo_model_path),
                            "input": "persistent_native_left_d405_color_image",
                            "hinge_candidate_indices": [int(index) for index in snapshot.get("hinge_candidates", candidates)],
                            "shaft_candidate_indices": [int(index) for index in snapshot.get("shaft_candidates", [])],
                            "eligible_candidate_indices": [int(index) for index in candidates],
                            "shaft_association": snapshot.get("shaft_association"),
                        }
                    record["eligible_candidate_indices"] = [int(index) for index in candidates]
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
                        reviewed_same_target, review_distance_px = reviewed_target_matches_selected(
                            None if selected is None else statistics[selected],
                            yolo_sam_review,
                            max_distance_px=args.track_max_distance_px,
                        )
                        record["sam_review"]["current_target_distance_px"] = review_distance_px
                        record["sam_review"]["reviewed_same_target"] = reviewed_same_target
                        if not (reviewed_same_target and yolo_sam_review.get("accepted") and yolo_sam_review.get("fresh")):
                            selected = None
                            tracking = "withheld_pending_or_failed_sam_review"
                            # Do not advance the identity lock with a
                            # candidate that SAM3 did not review. Otherwise a
                            # temporary association miss can lock onto an
                            # unreviewed hinge and fail this gate forever.
                            tracked_center = None
                            tracked_depth_m = None
                            missed_frames = 0
                            record["tracking_lock_reset"] = True
                    shaft_association = (
                        associate_shaft_to_hinge(
                            statistics,
                            selected,
                            shaft_indices=[int(index) for index in snapshot.get("shaft_candidates", [])],
                            max_center_distance_px=args.shaft_association_max_distance_px,
                            max_horizontal_distance_px=args.shaft_association_max_horizontal_distance_px,
                            ambiguity_margin_px=args.shaft_association_ambiguity_margin_px,
                        )
                        if args.detector in {"yolo", "yolo-sam"}
                        else {"shaft_index": None, "reason": "sam3_mode"}
                    )
                    if args.detector in {"yolo", "yolo-sam"} and "yolo" in record:
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
                        if (
                            args.allow_motion
                            and state is not None
                            and state.phase == GraspPhase.GRASPED
                            and gripper_sent_at is not None
                        ):
                            loss_now = time.monotonic()
                            if post_grasp_target_lost_since is None:
                                post_grasp_target_lost_since = loss_now
                                record["post_grasp_target_loss"] = {
                                    "state": "pending",
                                    "lost_duration_s": 0.0,
                                    "timeout_s": args.post_grasp_target_lost_timeout_s,
                                }
                            elif loss_now - post_grasp_target_lost_since >= args.post_grasp_target_lost_timeout_s:
                                assert position_stream is not None
                                assert last_end_link_target is not None
                                assert gripper_open is not None and gripper_tolerance is not None
                                assert gripper_close_timeout is not None and parameters is not None
                                stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                                force_preview_stop = force_preview_thread = None
                                force_preview_phase = None
                                lost_duration_s = loss_now - post_grasp_target_lost_since
                                if services is not None and services.get('vlm_gripper_monitor') is not None:
                                    services.get('vlm_gripper_monitor').set_phase('idle')
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
                                post_grasp_target_lost_since = None
                                post_grasp_verify_baseline = None
                                post_grasp_verify_samples.clear()
                                post_grasp_verified = False
                                active_regrasp_mailbox = None
                                record.update({
                                    "state": "regrasp_reacquiring",
                                    "reason": "post_grasp_target_lost_timeout",
                                    "lost_duration_s": round(lost_duration_s, 3),
                                    "regrasp_timeout_s": args.post_grasp_target_lost_timeout_s,
                                    "regrasp_attempt": regrasp_attempts,
                                    "regrasp_recovery": recovery,
                                })
                                json_record(log, record)
                                print_compact_record(record)
                                continue
                        else:
                            post_grasp_target_lost_since = None
                        if args.step_profile is not None and step_xy_pending is not None:
                            actual = measured_left_end_link_pose(controller)
                            last_end_link_target = CartesianPoseTarget(actual.position_m, last_end_link_target.orientation_xyzw)
                            position_stream.update_target(last_end_link_target)
                            step_xy_pending = None
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
                        }
                        record["state"] = "waiting_for_selected_depth"
                        record["reason"] = f"{unavailable_reason}; holding_current_pose"
                        perception.set_phase(record["state"], record["reason"])
                        json_record(log, record)
                        print_compact_record(record)
                        continue
                    item = statistics[selected]
                    post_grasp_target_lost_since = None
                    coarse_active = state is not None and state.coarse_active
                    gripper_closing = state is not None and state.phase == GraspPhase.GRASPED
                    if state is not None:
                        parameters = state.parameters
                    if coarse_active or gripper_closing:
                        item = dict(item)
                        item.pop("step_pose", None)
                    record["alignment_stage"] = (
                        "coarse_yolo"
                        if coarse_active
                        else (
                            "fine_step"
                            if args.step_profile is not None
                            else ("yolo_box_fraction" if args.detector == "yolo" else "sam_fraction")
                        )
                    )
                    if args.step_profile is not None and not gripper_closing:
                        record["step_pose"] = item.get("step_pose", {})
                        if not coarse_active and not item.get("step_pose", {}).get("accepted"):
                            if step_xy_pending is not None:
                                actual = measured_left_end_link_pose(controller)
                                last_end_link_target = CartesianPoseTarget(actual.position_m, last_end_link_target.orientation_xyzw)
                                position_stream.update_target(last_end_link_target)
                                step_xy_pending = None
                            if state is not None:
                                state.reset_alignment()
                            record.update(state="waiting_for_step_pose", reason="STEP correspondence unavailable; no new target")
                            json_record(log, record)
                            print_compact_record(record)
                            continue
                    if (
                        args.step_profile is not None
                        and state is not None
                        and not state.coarse_active
                        and getattr(args, "step_height_samples", None)
                        and item.get("step_pose", {}).get("accepted")
                    ):
                        dynamic_target, height_detail = target_pixel_for_step_pose(
                            args.step_height_samples, item["step_pose"],
                        )
                        state.set_fine_target_pixel(dynamic_target)
                        parameters = state.parameters
                        record["height_reference"] = height_detail
                    if single_xy_verification is not None:
                        post_center = np.asarray(servo_pixel(item), dtype=np.float64)
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
                        tuple(float(value) for value in servo_pixel(item)),
                        # The empirical pixel Jacobian below is the only XY
                        # controller.  Depth is diagnostic/optional and no
                        # longer filters out the visually leftmost hinge.
                        float(item["depth_median_m"] if item.get("depth_median_m") is not None else 1.0),
                        float(color_intrinsic["fx"]), float(color_intrinsic["fy"]),
                        float(color_intrinsic["ppx"]), float(color_intrinsic["ppy"]),
                        base_xy_error_m=item.get("step_pose", {}).get("base_xy_error_m"),
                        xy_tolerance_m=args.step_xy_tolerance_mm / 1000,
                        target_pixel_px=(float(parameters.target_u_px), float(parameters.target_v_px)),
                    )
                    step = state.update(observation, now=time.monotonic())
                    if args.step_profile is not None and step.camera_delta_m is None and step_xy_pending is not None and not gripper_closing:
                        # Entering the tolerance band: cancel the outstanding
                        # lookahead, hold measured position, then reconfirm.
                        actual = measured_left_end_link_pose(controller)
                        last_end_link_target = CartesianPoseTarget(actual.position_m, last_end_link_target.orientation_xyzw)
                        position_stream.update_target(last_end_link_target)
                        step_xy_pending = None
                        state.reset_alignment()
                        record.update(state="xy_settling", reason="hold_actual_then_reconfirm_visual_alignment")
                        json_record(log, record)
                        continue
                    if args.step_profile is not None and step.phase == GraspPhase.FINAL_DESCENT and position_stream is not None:
                        actual = measured_left_end_link_pose(controller)
                        if not position_stream.is_target_reached() or np.linalg.norm(np.asarray(actual.position_m) - last_end_link_target.position_m) > .0005:
                            state.reset_alignment()
                            record.update(state="xy_settling", reason="waiting_actual_before_descent")
                            json_record(log, record)
                            continue
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
                                float(servo_pixel(item)[0]) - parameters.target_u_px,
                                float(servo_pixel(item)[1]) - parameters.target_v_px,
                            ), dtype=np.float64)
                            delta_base = empirical_horizontal_base_delta_from_pixel_error(
                                pixel_error, empirical_pixel_jacobian, max_step_m=parameters.max_xy_step_m,
                            )
                            if args.step_metric_xy and not coarse_active:
                                delta_base = bounded_xy(item["step_pose"]["base_xy_error_m"], parameters.max_xy_step_m)
                            command_source = "gdk_end_effector_pose_mode3_empirical_visual_xy"
                            if args.step_metric_xy and not coarse_active:
                                command_source = "gdk_end_effector_pose_mode3_step_metric_xy"
                                record["base_xy_error_m"] = item["step_pose"]["base_xy_error_m"]
                            record["pixel_error_px"] = [float(value) for value in pixel_error]
                            record["xy_step_limit_mm"] = parameters.max_xy_step_m * 1000.0
                            record["pixel_jacobian_px_per_m"] = empirical_pixel_jacobian.tolist()
                        else:
                            assert step.base_delta_m is not None
                            assert step.motion_duration_s is not None
                            delta_base = np.asarray(step.base_delta_m, dtype=np.float64)
                            command_source = "gdk_end_effector_pose_mode3_force_guarded_final_descent" if step.is_final_descent else "gdk_end_effector_pose_mode3_vertical_approach"
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
                        # The final descent terminates on measured distance, not
                        # on the nominal speed*duration window: the 50 Hz pose
                        # stream paces itself to measured feedback, so a
                        # compliant arm may need longer than nominal.  Time is
                        # kept only as a runaway guard.
                        descent_completed = False
                        descent_actual_down_m = 0.0
                        descent_actual_lateral_m = 0.0
                        descent_actual_delta_m = np.zeros(3, dtype=np.float64)
                        descent_distance_m = 0.0
                        descent_safety_s = 0.0
                        descent_stall_s = 0.0
                        descent_completion_tolerance_m = 0.0
                        if step.is_final_descent:
                            assert step.motion_duration_s is not None
                            descent_distance_m = max(0.0, -float(delta_base[2]))
                            descent_safety_s = max(15.0, step.motion_duration_s * 3.0 + 5.0)
                            descent_stall_s = max(3.0, step.motion_duration_s * 0.2)
                            descent_completion_tolerance_m = parameters.final_descent_completion_tolerance_m
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
                        # Rebase every visual correction on the measured end-link
                        # pose.  Accumulating from the previous *requested*
                        # target lets a stalled or rate-limited GDK stream run
                        # away from the real robot while the image error stays
                        # unchanged.  The stream still slews at 50 Hz; this
                        # only prevents an unbounded open-loop look-ahead.
                        if step.camera_delta_m is not None and args.allow_motion:
                            actual_before_update = measured_left_end_link_pose(controller)
                            if last_end_link_target is not None:
                                record["target_follow_error_mm"] = round(
                                    float(
                                        np.linalg.norm(
                                            np.asarray(last_end_link_target.position_m, dtype=np.float64)
                                            - np.asarray(actual_before_update.position_m, dtype=np.float64)
                                        )
                                    )
                                    * 1000.0,
                                    3,
                                )
                            record["actual_end_link_before_base_link"] = list(actual_before_update.position_m)
                            orientation = (
                                last_end_link_target.orientation_xyzw
                                if last_end_link_target is not None
                                else actual_before_update.orientation_xyzw
                            )
                            # Rebase X/Y only.  Copying the measured Z as well
                            # let whole-body-IK/compliance drift ratchet the
                            # approach height upward while the XY servo ran
                            # (observed: +146 mm of unintended climb over 234
                            # corrections in mode 3).  The commanded Z stays
                            # the alignment reference so the stream corrects
                            # drift instead of accumulating it.
                            reference_z = (
                                last_end_link_target.position_m[2]
                                if last_end_link_target is not None
                                else actual_before_update.position_m[2]
                            )
                            last_end_link_target = CartesianPoseTarget(
                                (
                                    actual_before_update.position_m[0],
                                    actual_before_update.position_m[1],
                                    reference_z,
                                ),
                                orientation,
                            )
                        elif last_end_link_target is None:
                            last_end_link_target = measured_left_end_link_pose(controller)
                        if args.step_profile is not None and step.camera_delta_m is not None and args.allow_motion:
                            actual = measured_left_end_link_pose(controller)
                            last_end_link_target = CartesianPoseTarget(
                                (
                                    actual.position_m[0],
                                    actual.position_m[1],
                                    last_end_link_target.position_m[2],
                                ),
                                last_end_link_target.orientation_xyzw,
                            )
                        target = CartesianPoseTarget(
                            base_position_after_delta(last_end_link_target.position_m, delta_base),
                            last_end_link_target.orientation_xyzw,
                        )
                        record.update({
                            "visual_tip_link": LEFT_VISUAL_TIP_LINK,
                            "command_source": command_source,
                            "gdk_control_mode": 3,
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
                            if args.step_profile is not None:
                                position_stream.set_translation_velocity(args.xy_speed_mm_s / 1000 if step.camera_delta_m is not None else parameters.final_descent_speed_mps)
                            position_stream.update_target(target)
                            last_end_link_target = target
                            if args.step_profile is not None and step.camera_delta_m is not None and not args.single_xy_step:
                                step_xy_pending = (np.asarray(target.position_m), time.monotonic() + max(2.0, float(np.linalg.norm(delta_base)) / parameters.final_descent_speed_mps + 1.0))
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
                                    height_target_provider = None
                                    if height_samples and args.step_profile is None:
                                        def height_target_provider(_item):
                                            current = current_height_reference_target(
                                                end_link_transform=end_link_transform,
                                                item=_item,
                                            )
                                            return None if current is None else current[0]
                                    elif getattr(args, "step_height_samples", None):
                                        def height_target_provider(step_item):
                                            pose = step_item.get("step_pose", {})
                                            if not pose.get("accepted"):
                                                return None
                                            target, _detail = target_pixel_for_step_pose(
                                                args.step_height_samples, pose,
                                            )
                                            return target
                                    visual_mailbox = (SnapshotVisualMailbox(
                                        perception,
                                        initial_center_px=item["center_px"],
                                        track_max_distance_px=args.track_max_distance_px,
                                        target_lost_frames=args.final_descent_target_lost_frames,
                                        target_pixel_px=(parameters.target_u_px, parameters.target_v_px),
                                        pixel_jacobian=empirical_pixel_jacobian,
                                        pixel_tolerance_px=parameters.pixel_tolerance,
                                        max_xy_step_m=args.final_descent_visual_max_step_m,
                                        visual_tolerance_m=args.final_descent_visual_tolerance_mm / 1000.0,
                                        allow_xy_correction=True,
                                        target_pixel_provider=height_target_provider,
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
                                    maximum_delta_force_n = 0.0
                                    latest_force_n: np.ndarray | None = None
                                    latest_hits = 0
                                    force_relief_completed = False
                                    final_descent_retry_reason: str | None = None
                                    descent_origin_m = np.asarray(
                                        record["end_link_target_before_base_link"], dtype=np.float64
                                    )
                                    descent_safety_deadline = time.monotonic() + descent_safety_s
                                    descent_stalled = False
                                    descent_progress_reference_m = 0.0
                                    descent_progress_reference_at = time.monotonic()
                                    last_progress_read = 0.0
                                    while True:
                                        position_stream.raise_if_failed()
                                        progress_now = time.monotonic()
                                        if progress_now - last_progress_read >= 0.1:
                                            progress_pose = measured_left_end_link_pose(controller)
                                            descent_actual_delta_m = (
                                                np.asarray(progress_pose.position_m, dtype=np.float64)
                                                - descent_origin_m
                                            )
                                            descent_actual_down_m = -float(descent_actual_delta_m[2])
                                            descent_actual_lateral_m = float(
                                                np.linalg.norm(descent_actual_delta_m[:2])
                                            )
                                            last_progress_read = progress_now
                                            record["final_descent_progress_m"] = {
                                                "target_down_m": descent_distance_m,
                                                "actual_down_m": descent_actual_down_m,
                                                "actual_lateral_m": descent_actual_lateral_m,
                                                "completion_tolerance_m": descent_completion_tolerance_m,
                                                "safety_time_s": descent_safety_s,
                                                "stall_time_s": descent_stall_s,
                                            }
                                            if (
                                                descent_actual_down_m
                                                >= descent_distance_m - descent_completion_tolerance_m
                                                or position_stream.is_target_reached()
                                            ):
                                                # Either the strict measured
                                                # distance or the stream's own
                                                # actual-arrival tolerance
                                                # (5 mm) is satisfied; the
                                                # latter absorbs joint-impedance
                                                # steady-state following error.
                                                descent_completed = True
                                                break
                                            if descent_actual_down_m > descent_progress_reference_m + 0.001:
                                                descent_progress_reference_m = descent_actual_down_m
                                                descent_progress_reference_at = progress_now
                                            elif (
                                                progress_now - descent_progress_reference_at
                                                >= descent_stall_s
                                            ):
                                                descent_stalled = True
                                                break
                                        if progress_now >= descent_safety_deadline:
                                            break
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
                                            position_stream.rebase_to_measured_pose()
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
                                            relief_speed_mps = relief_translation_speed_mps(
                                                parameters.final_descent_speed_mps
                                            )
                                            position_stream.set_translation_velocity(relief_speed_mps)
                                            record["force_relief_speed"] = {
                                                "automatic_speed_mps": parameters.final_descent_speed_mps,
                                                "automatic_speed_mm_s": parameters.final_descent_speed_mps * 1000.0,
                                                "relief_speed_mps": relief_speed_mps,
                                                "relief_speed_mm_s": relief_speed_mps * 1000.0,
                                                "ratio": 0.5,
                                            }
                                            position_stream.rebase_to_measured_pose()
                                            position_stream.update_target(relief_target)
                                            last_end_link_target = relief_target
                                            relief_deadline = time.monotonic() + (
                                                contact_detector.parameters.relief_max_lift_m
                                                / relief_speed_mps
                                                + 0.35
                                            )
                                            relief_hits = 0
                                            latest_relief_force_n: np.ndarray | None = None
                                            latest_relief_delta_force_n: float | None = None
                                            try:
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
                                                        extra_lift_m = float(
                                                            contact_detector.parameters.relief_extra_lift_m
                                                        )
                                                        if extra_lift_m > 0.0:
                                                            # Release force confirmed: add
                                                            # the configured extra
                                                            # clearance above the measured
                                                            # release pose.
                                                            relief_hold_target = CartesianPoseTarget(
                                                                base_position_after_delta(
                                                                    relief_pose.position_m,
                                                                    np.asarray((0.0, 0.0, extra_lift_m)),
                                                                ),
                                                                last_end_link_target.orientation_xyzw,
                                                            )
                                                        else:
                                                            relief_hold_target = CartesianPoseTarget(
                                                                relief_pose.position_m,
                                                                last_end_link_target.orientation_xyzw,
                                                            )
                                                        position_stream.rebase_to_measured_pose()
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
                                            finally:
                                                # The slower rate applies only
                                                # to unloading.  All following
                                                # motion, including a failed
                                                # relief retreat, uses the
                                                # configured automatic speed.
                                                position_stream.set_translation_velocity(
                                                    parameters.final_descent_speed_mps
                                                )
                                            record["force_relief"] = {
                                                "required": True,
                                                "max_lift_m": contact_detector.parameters.relief_max_lift_m,
                                                "force_delta_release_threshold_n": contact_detector.parameters.relief_force_delta_threshold_n,
                                                "consecutive_samples_required": contact_detector.parameters.relief_consecutive_samples,
                                                "released": force_relief_completed,
                                                "speed_mps": relief_speed_mps,
                                                "speed_mm_s": relief_speed_mps * 1000.0,
                                                "latest_force_n": None if latest_relief_force_n is None else [float(value) for value in latest_relief_force_n],
                                                "latest_delta_force_n": latest_relief_delta_force_n,
                                                "consecutive_hits": relief_hits,
                                            }
                                            if (
                                                force_relief_completed
                                                and contact_detector.parameters.relief_extra_lift_m > 0.0
                                            ):
                                                # Wait for the extra clearance to settle
                                                # before allowing closure.  Not reaching
                                                # it is recorded but not fatal: the force
                                                # has already been released.
                                                extra_settle_m = float(
                                                    contact_detector.parameters.relief_extra_lift_m
                                                )
                                                extra_deadline = time.monotonic() + max(
                                                    2.0,
                                                    extra_settle_m / parameters.final_descent_speed_mps + 1.5,
                                                )
                                                extra_settled = False
                                                while time.monotonic() < extra_deadline:
                                                    position_stream.raise_if_failed()
                                                    extra_delta = (
                                                        np.asarray(
                                                            measured_left_end_link_pose(controller).position_m
                                                        )
                                                        - np.asarray(last_end_link_target.position_m)
                                                    )
                                                    if float(np.linalg.norm(extra_delta)) <= 0.005:
                                                        extra_settled = True
                                                        break
                                                    time.sleep(contact_detector.parameters.sample_period_s)
                                                record["force_relief"]["extra_lift_m"] = extra_settle_m
                                                record["force_relief"]["extra_lift_settled"] = extra_settled
                                            if not force_relief_completed:
                                                # Return to the pre-descent height before
                                                # permitting another visual alignment.
                                                retry_pose = measured_left_end_link_pose(controller)
                                                retry_z = max(
                                                    retry_pose.position_m[2],
                                                    record["end_link_target_before_base_link"][2],
                                                )
                                                retry_target = CartesianPoseTarget(
                                                    (retry_pose.position_m[0], retry_pose.position_m[1], retry_z),
                                                    last_end_link_target.orientation_xyzw,
                                                )
                                                position_stream.rebase_to_measured_pose()
                                                position_stream.update_target(retry_target)
                                                last_end_link_target = retry_target
                                                retry_deadline = time.monotonic() + (
                                                    (retry_z - retry_pose.position_m[2]) / parameters.final_descent_speed_mps + 3.0
                                                )
                                                retry_hits = 0
                                                actual_retry = retry_pose
                                                retry_arrived = False
                                                retry_released = False
                                                retry_delta_force_n: float | None = None
                                                while time.monotonic() < retry_deadline:
                                                    position_stream.raise_if_failed()
                                                    actual_retry = measured_left_end_link_pose(controller)
                                                    _, retry_force = read_wrench_force(robot, contact_detector.parameters.frame_name)
                                                    retry_delta_force_n = float(np.linalg.norm(retry_force - contact_detector.baseline_force_n))
                                                    arrived = np.linalg.norm(np.asarray(actual_retry.position_m) - retry_target.position_m) <= 0.005
                                                    released = retry_delta_force_n <= contact_detector.parameters.relief_force_delta_threshold_n
                                                    retry_arrived, retry_released = arrived, released
                                                    retry_hits = retry_hits + 1 if arrived and released else 0
                                                    if retry_hits >= contact_detector.parameters.relief_consecutive_samples:
                                                        break
                                                    time.sleep(contact_detector.parameters.sample_period_s)
                                                else:
                                                    record["force_relief"].update({
                                                        "retract_target_base_link": list(retry_target.position_m),
                                                        "retract_actual_base_link": list(actual_retry.position_m),
                                                        "retract_arrived": retry_arrived,
                                                        "retract_released": retry_released,
                                                        "retract_latest_delta_force_n": retry_delta_force_n,
                                                        "retract_hits": retry_hits,
                                                    })
                                                    raise SafetyError(
                                                        "抓取失败后撤回未到位或仍持续受力；停止且不重新下探"
                                                        f"（撤回目标 {list(retry_target.position_m)}，"
                                                        f"实测 {list(actual_retry.position_m)}，"
                                                        f"到位={retry_arrived}，力已释放={retry_released}，"
                                                        f"残余力差={retry_delta_force_n} N，"
                                                        f"释放阈值={contact_detector.parameters.relief_force_delta_threshold_n} N）"
                                                    )
                                                final_descent_retry_reason = "force_relief_failed_retracted_retry"
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
                                                    # The perception mailbox
                                                    # only reports this for the
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
                                                    position_stream.rebase_to_measured_pose()
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
                                                        "visual_error_m": correction.get("visual_error_m"),
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
                                                        position_stream.rebase_to_measured_pose()
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
                                                        "visual_error_m": correction.get("visual_error_m"),
                                                        "delta_base_m": correction["delta_base_m"],
                                                            "cumulative_horizontal_m": float(np.linalg.norm(visual_total_base_delta[:2])),
                                                        })
                                        time.sleep(contact_detector.parameters.sample_period_s)
                                    record["final_descent_visual_servo"] = {
                                        "enabled": visual_mailbox is not None,
                                        "target_lock": "same YOLO target nearest to pre-descent selection; never re-rank left/right during descent",
                                        "max_step_m": args.final_descent_visual_max_step_m,
                                        "tolerance_m": args.final_descent_visual_tolerance_mm / 1000.0,
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
                                        if services is not None and services.get('vlm_gripper_monitor') is not None:
                                            services.get('vlm_gripper_monitor').set_phase('idle')
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
                                        if descent_completed:
                                            termination = "distance_reached_without_contact"
                                        elif descent_stalled:
                                            termination = "stalled_before_configured_distance"
                                        else:
                                            termination = "safety_timeout_before_configured_distance"
                                        record["final_descent_termination"] = termination
                                        record["actual_base_delta_m"] = [
                                            float(value) for value in descent_actual_delta_m
                                        ]
                                        progress_detail = (
                                            f"（目标 {descent_distance_m * 1000:.1f} mm，"
                                            f"实测 {descent_actual_down_m * 1000:.1f} mm，"
                                            f"横向 {descent_actual_lateral_m * 1000:.1f} mm）"
                                        )
                                        if descent_completed:
                                            raise SafetyError(
                                                "最终下探实测已走完配置行程"
                                                + progress_detail
                                                + "仍未检测到铰链平面接触；已停止且不闭爪"
                                            )
                                        if descent_stalled:
                                            raise SafetyError(
                                                "最终下探在安全时限内停止前进"
                                                + progress_detail
                                                + f"，连续 {descent_stall_s:.1f} s 无有效位移且未检测到接触；"
                                                "已停止且不闭爪（位姿流未按实测跟进而冻结）"
                                            )
                                        raise SafetyError(
                                            "最终下探在安全时限内未走完配置行程"
                                            + progress_detail
                                            + f"，安全时限 {descent_safety_s:.1f} s 且未检测到接触；已停止且不闭爪"
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
                                elif not position_stream.wait_until_target(
                                    max(15.0, step.motion_duration_s * 3.0 + 5.0)
                                ):
                                    raise SafetyError(
                                        "定时下探未在安全时限内发送到最终 GDK 末端目标；拒绝闭爪"
                                    )
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
                                "before_center_px": [float(value) for value in servo_pixel(item)],
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
                            if contact_detector is not None and force_contact_detected:
                                test_stop_reason = "force_contact_detected_before_gripper_close"
                            elif contact_detector is not None and descent_completed:
                                test_stop_reason = (
                                    "configured_final_descent_distance_reached_without_contact_before_gripper_close"
                                )
                            elif contact_detector is not None and descent_stalled:
                                test_stop_reason = (
                                    "final_descent_stalled_without_contact_before_gripper_close"
                                )
                            elif contact_detector is not None:
                                test_stop_reason = (
                                    "final_descent_safety_timeout_without_contact_before_gripper_close"
                                )
                            else:
                                test_stop_reason = (
                                    "max_final_descent_reached_without_contact_before_gripper_close"
                                )
                            record["test_stop"] = test_stop_reason
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
                                # by the closing fingers.  Its YOLO mask geometry can
                                # change during the controlled +Z verification lift;
                                # these values are retained for diagnosis only.
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
                            verification["action"] = "transport" if held_with_gripper else "regrasp"
                            post_grasp_verify_baseline = None
                            post_grasp_verify_samples.clear()
                            record["post_grasp_verify"] = verification
                            if not held_with_gripper:
                                assert position_stream is not None and gripper_open is not None and gripper_tolerance is not None
                                stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                                force_preview_stop = force_preview_thread = None
                                force_preview_phase = None
                                if services is not None and services.get('vlm_gripper_monitor') is not None:
                                    services.get('vlm_gripper_monitor').set_phase('idle')
                                position_stream, last_end_link_target, recovery = recover_for_visual_regrasp(
                                    controller, position_stream,
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
                                post_grasp_verified = False
                                record.update({
                                    "state": "regrasp_reacquiring",
                                    "reason": "post_grasp_verify_failed",
                                    "regrasp_attempt": regrasp_attempts,
                                    "regrasp_recovery": recovery,
                                })
                                json_record(log, record)
                                print_compact_record(record)
                                continue
                            post_grasp_verified = True
                        # Continue observing the same target while the jaw is
                        # closing.  A loss at this point is treated as a
                        # failed attempt, never as permission to transport an
                        # assumed grasp.  The perception worker remains the
                        # sole owner of the camera and YOLO model.
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
                                    if services is not None and services.get('vlm_gripper_monitor') is not None:
                                        services.get('vlm_gripper_monitor').set_phase('idle')
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
                            decision = decide_gripper_close(
                                position_rad=position,
                                gripper_open_rad=gripper_open,
                                tolerance_rad=gripper_tolerance,
                                parameters=adaptive_gripper,
                                samples=gripper_position_samples,
                                now=now,
                            )
                            record["adaptive_gripper_close_travel_rad"] = decision.close_travel_rad
                            if decision.hold_target_rad is not None:
                                adaptive_hold_target_rad = decision.hold_target_rad
                                assert position_stream is not None
                                position_stream.set_left_gripper_target(adaptive_hold_target_rad)
                                adaptive_hold_sent_at = now
                                record["adaptive_gripper_stall"] = {
                                    "reason": decision.reason,
                                    "stall_angle_rad": decision.stall_angle_rad,
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
                            if services is not None:
                                vlm_monitor = services.get('vlm_gripper_monitor')
                                if vlm_monitor is not None:
                                    vlm_monitor.set_phase('post_grasp')
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
                                # After contact and closure the internal
                                # commanded point can still be far below the
                                # measured pose, which freezes the slew and
                                # makes this +Z verification lift time out
                                # (seen repeatedly as
                                # post_grasp_verify_lift_target_not_reached).
                                # Rebase so the lift starts from the real arm.
                                position_stream.rebase_to_measured_pose()
                                position_stream.update_target(verify_target)
                                verify_timeout_s = max(
                                    1.5,
                                    args.post_grasp_verify_lift_m / parameters.final_descent_speed_mps + 1.0,
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
                            # own the camera briefly.  Reuse the same
                            # perception worker for the subsequent +Z lift,
                            # but now hand its locked-target observations to
                            # the lift monitor as well as the web preview.
                            if args.full_cycle:
                                stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                                force_preview_stop = force_preview_thread = None
                                force_preview_phase = {
                                    "state": "post_grasp_monitored_lift",
                                    "overlay_note": (
                                        "same-target held-part monitoring during +Z lift; "
                                        "confirmed target loss after timeout cancels transport and regraspes; "
                                        "mask geometry changes are diagnostic only; stale holds closed"
                                    ),
                                }
                                transport_mailbox = SnapshotVisualMailbox(
                                    perception, initial_center_px=item["center_px"],
                                    track_max_distance_px=args.track_max_distance_px,
                                    target_lost_frames=args.final_descent_target_lost_frames,
                                    target_lost_timeout_s=args.post_grasp_target_lost_timeout_s,
                                )
                                perception.set_phase("post_grasp_monitored_lift", force_preview_phase["overlay_note"])
                            json_record(log, record)
                            print_compact_record(record)
                            if args.full_cycle:
                                assert post_grasp_presets is not None
                                assert position_stream is not None
                                # The first transport motion is not a blind,
                                # blocking helper call.  Keep the existing
                                # 50-Hz mode-3 stream alive and inspect every
                                # fresh same-target SAM3 observation while it
                                # raises 150 mm.  A dropped hinge is therefore
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
                                position_stream.rebase_to_measured_pose()
                                position_stream.update_target(lift_target)
                                json_record(log, {
                                    "event": "post_grasp_monitored_lift_start",
                                    "utc": datetime.now(timezone.utc).isoformat(),
                                    "gripper": "confirmed and preserved while same-target monitoring is healthy",
                                    "pre_p4_lift_m": PRE_P4_LIFT_M,
                                    "monitor_center_drift_px": args.post_grasp_verify_center_drift_px,
                                    "monitor_area_change_fraction": args.post_grasp_verify_area_change_fraction,
                                    "monitor_area_grace_s": args.post_grasp_lift_area_grace_s,
                                    "monitor_area_window_frames": args.post_grasp_lift_area_window_frames,
                                    "monitor_area_max_spread": args.post_grasp_lift_area_spread,
                                    "monitor_area_confirmations": args.post_grasp_lift_area_confirmations,
                                    # Diagnostics for the next hardware run: a
                                    # close travel near the full stroke means
                                    # the jaws met nothing, which is the empty
                                    # grasp the area monitor then has to catch.
                                    "gripper_close_travel_rad": record.get("adaptive_gripper_close_travel_rad"),
                                    "gripper_stall": record.get("adaptive_gripper_stall"),
                                    "target_lost_timeout_s": args.post_grasp_target_lost_timeout_s,
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
                                    # Empty-grasp detection: a closed gripper can
                                    # still track a hinge that was never inside
                                    # the jaws, so keep the trailing mask-area
                                    # history of the lift and fail the hold when
                                    # it keeps changing instead of settling.
                                    "lift_started_at": time.monotonic(),
                                    "area_samples": [],
                                    "max_area_spread_fraction": 0.0,
                                    "area_instability": None,
                                    "area_unstable_streak": 0,
                                    # Distance is the goal; time is only a runaway
                                    # guard.  The +1 s margin over the nominal
                                    # travel was too tight: the stream paces
                                    # itself to measured feedback, so a
                                    # compliant arm legitimately needs more.
                                    "deadline": time.monotonic() + max(
                                        10.0,
                                        3.0 * PRE_P4_LIFT_M / parameters.final_descent_speed_mps + 5.0,
                                    ),
                                }
                                # The next main-loop ticks poll this target;
                                # do not wait here.
                                continue
                            print("抓取已确认；保持当前位置和夹爪状态，未自动回预设位。")
                            return wait_for_manual_next()
                        assert gripper_close_timeout is not None
                        if now - gripper_sent_at > gripper_close_timeout:
                            assert position_stream is not None and gripper_open is not None and gripper_tolerance is not None
                            stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
                            force_preview_stop = force_preview_thread = None
                            force_preview_phase = None
                            if services is not None and services.get('vlm_gripper_monitor') is not None:
                                services.get('vlm_gripper_monitor').set_phase('idle')
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
                except AdvanceRightRequested:
                    return execute_handoff_before_exit()
                except Exception as exc:
                    record.update({"state": "error_stop_sending_commands", "error": str(exc)})
                    json_record(log, record)
                    print_compact_record(record, file=sys.stderr)
                    return 1
        return 0
    except KeyboardInterrupt:
        print("收到 Ctrl-C；停止发送后续命令。")
        return INTERRUPTED_EXIT_CODE
    except AdvanceRightRequested:
        return EXIT_ADVANCE_RIGHT
    except Exception as exc:
        print(f"左手 G2 抓取流程失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if owns_worker and takeover_worker is not None:
            takeover_worker.stop()
        if owns_worker and dianpi_runner is not None:
            dianpi_runner.close()
        stop_head_camera_web_preview(head_preview_stop, head_preview_thread)
        stop_force_descent_sam_preview(force_preview_stop, force_preview_thread)
        if perception is not None:
            perception.stop()
        if position_stream is not None:
            position_stream.stop()
        if initial_gripper_open_stream is not None:
            initial_gripper_open_stream.stop()
        if post_grasp_regrasp_open_stream is not None:
            post_grasp_regrasp_open_stream.stop()
        if web_server is not None:
            web_server.shutdown()
            web_server.server_close()
        if owns_camera and camera is not None:
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
