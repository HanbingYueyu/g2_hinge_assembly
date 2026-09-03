#!/usr/bin/env python3
"""Post-P4 bilateral XY alignment: right camera observes, left arm moves.

The operator is responsible for placing the left arm at the intended post-P4
pose and keeping the workpiece held.  This process does not enforce those two
startup conditions; it moves
the right arm once to ``right_post_grasp`` (only with ``--allow-motion``),
then uses the right-hand ``hole`` detector and left-hand ``shaft`` detector to
make bounded, discrete XY updates to the left end-link in GDK mode 1.
``--geometry-mode direct`` uses the official RGB-D extrinsics from both hands
and their dynamic base transforms; legacy ``veto`` keeps pixel-J as the
controller and lets the old right-image RGB-D path deny unsafe corrections.
In ``veto``/right-only read-only use, the hole RGB-D window is accumulated
independently of any left-hand target; ``direct`` still requires a fresh pair.
No Z motion, force action, gripper command, insertion, or automatic regrasp is
implemented here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    from .bilateral_depth_accumulator import FreshHoleDepthAccumulator
    from .bilateral_rgbd_geometry import PairGeometry, apply_transform, estimate_box_depth, estimate_pair_geometry, estimate_point_depth, right_color_point_to_base
    from .bilateral_rgbd_geometry import AdaptiveDepthError, estimate_hole_depth_adaptive
    from .bilateral_servo import (
        AlignmentPolicy,
        BilateralAlignmentGate,
        BilateralMetricAlignmentGate,
        MetricAlignmentPolicy,
        metric_and_pixel_corrections_agree,
    )
    from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, depth_on_color_grid, intrinsic_to_dict, transform_to_dict
    from .control import CartesianPoseTarget, G2Controller
    from .hole_target_detector import (
        BoxDetection,
        center_crop,
        infer_detection,
        load_detection_model,
        pair_status,
        select_locked_hole,
        translate_detection,
    )
    from .record_pose import DEFAULT_OUTPUT
    from .right_hole_target_detect_gdk import RIGHT_ARM_OFFSET, SharedView, handler_class
    from .yolo_hinge import infer_yolo_masks, load_yolo_segmentation_model
except ImportError:
    from bilateral_depth_accumulator import FreshHoleDepthAccumulator
    from bilateral_rgbd_geometry import PairGeometry, apply_transform, estimate_box_depth, estimate_pair_geometry, estimate_point_depth, right_color_point_to_base
    from bilateral_rgbd_geometry import AdaptiveDepthError, estimate_hole_depth_adaptive
    from bilateral_servo import (
        AlignmentPolicy,
        BilateralAlignmentGate,
        BilateralMetricAlignmentGate,
        MetricAlignmentPolicy,
        metric_and_pixel_corrections_agree,
    )
    from camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, depth_on_color_grid, intrinsic_to_dict, transform_to_dict
    from control import CartesianPoseTarget, G2Controller
    from hole_target_detector import (
        BoxDetection,
        center_crop,
        infer_detection,
        load_detection_model,
        pair_status,
        select_locked_hole,
        translate_detection,
    )
    from record_pose import DEFAULT_OUTPUT
    from right_hole_target_detect_gdk import RIGHT_ARM_OFFSET, SharedView, handler_class
    from yolo_hinge import infer_yolo_masks, load_yolo_segmentation_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hole_target_det_v2" / "best.pt"
DEFAULT_LEFT_TARGET_MODEL = PROJECT_ROOT / "models" / "left_d405_hinge_shaft_yolo" / "best.pt"
LEFT_END_LINK = "arm_l_end_link"
LEFT_HINGE_CLASS_ID = 0
LEFT_SHAFT_CLASS_ID = 1
DEFAULT_LEFT_TRACK_MAX_JUMP_PX = 120.0


def parse_jacobian(value: str | None) -> list[list[float]] | None:
    if value is None or not value.strip():
        return None
    try:
        numbers = [float(item) for item in re.split(r"[,;\s]+", value.strip()) if item]
    except ValueError as exc:
        raise ValueError("--pixel-jacobian 必须是 4 个数字：J00,J01,J10,J11") from exc
    if len(numbers) != 4 or not all(math.isfinite(item) for item in numbers):
        raise ValueError("--pixel-jacobian 必须是 4 个有限数字：J00,J01,J10,J11")
    return [numbers[:2], numbers[2:]]


def parse_depth_scales(value: str) -> tuple[float, ...]:
    try:
        scales = tuple(float(item) for item in re.split(r"[,;\s]+", value.strip()) if item)
    except ValueError as exc:
        raise ValueError("--hole-depth-scales 必须是逗号分隔的数字") from exc
    if not scales or scales[0] != 1.0 or any(not math.isfinite(item) or item < 1.0 for item in scales):
        raise ValueError("--hole-depth-scales 必须从 1 开始且为有限非递减正数")
    if any(scales[index] > scales[index + 1] for index in range(len(scales) - 1)):
        raise ValueError("--hole-depth-scales 必须按递增顺序填写")
    return scales


def can_accumulate_hole_depth(
    *,
    selected_hole: bool,
    aligned_depth: bool,
    target_count: int,
    geometry_mode: str,
) -> bool:
    """Return whether the right-hole temporal depth window may accept a frame.

    The right-only ``veto`` path must be able to measure the hole without a
    left-hand target.  The bilateral ``direct`` path still requires exactly
    one left target before consuming a depth batch for motion geometry.
    """
    if geometry_mode not in {"off", "veto", "direct"}:
        raise ValueError(f"unknown geometry mode: {geometry_mode}")
    if not selected_hole or not aligned_depth:
        return False
    return geometry_mode != "direct" or int(target_count) == 1


def load_pose(path: Path, name: str, side: str) -> list[float]:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    item = document.get("poses", {}).get(name)
    if not isinstance(item, dict) or item.get("side") != side:
        raise RuntimeError(f"姿态文件缺少 {side} 侧 {name}")
    values = [float(value) for value in item.get("joint_positions_rad", ())]
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{name} 必须包含 7 个有限关节值")
    return values


@dataclass(frozen=True)
class LeftTargetObservation:
    """A shaft mask associated with one visible hinge; tip is the mating point."""

    box: BoxDetection
    tip_px: tuple[float, float]
    mask: np.ndarray
    hinge_center_px: tuple[float, float] | None = None

    @property
    def class_id(self) -> int:
        return self.box.class_id

    @property
    def class_name(self) -> str:
        return self.box.class_name

    @property
    def confidence(self) -> float:
        return self.box.confidence

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.box.xyxy

    @property
    def center_px(self) -> tuple[float, float]:
        return self.tip_px


def _mask_center(mask: np.ndarray) -> np.ndarray | None:
    mask_bool = np.asarray(mask, dtype=bool)
    rows, cols = np.where(mask_bool)
    if rows.size == 0:
        return None
    return np.array((float(cols.mean()), float(rows.mean())), dtype=np.float64)


def _mask_box(mask: np.ndarray, class_id: int, class_name: str, score: float) -> BoxDetection:
    mask_bool = np.asarray(mask, dtype=bool)
    rows, cols = np.where(mask_bool)
    if rows.size == 0:
        raise ValueError("empty detection mask")
    return BoxDetection(
        class_id=int(class_id),
        class_name=class_name,
        confidence=float(score),
        xyxy=(float(cols.min()), float(rows.min()), float(cols.max()), float(rows.max())),
    )


def _mask_median_depth(mask: np.ndarray, aligned_depth: np.ndarray | None, depth_scale_m: float) -> float | None:
    if aligned_depth is None:
        return None
    depth = np.asarray(aligned_depth)
    mask_bool = np.asarray(mask, dtype=bool)
    if depth.shape != mask_bool.shape:
        return None
    values = depth[mask_bool].astype(np.float64, copy=False)
    values = values[np.isfinite(values) & (values > 0.0)]
    if values.size == 0:
        return None
    median_m = float(np.median(values)) * float(depth_scale_m)
    return median_m if math.isfinite(median_m) and median_m > 0.0 else None


def _left_target_observation(mask: np.ndarray, score: float, hinge_center: np.ndarray) -> LeftTargetObservation:
    """Convert one shaft mask into a tip selected away from its associated hinge."""
    mask_bool = np.asarray(mask, dtype=bool)
    rows, cols = np.where(mask_bool)
    if rows.size == 0:
        raise ValueError("empty shaft mask")
    points = np.column_stack((cols.astype(np.float64), rows.astype(np.float64)))
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    endpoints = (points[int(np.argmin(projections))], points[int(np.argmax(projections))])
    tip = max(endpoints, key=lambda point: float(np.linalg.norm(point - hinge_center)))
    x0, x1 = float(cols.min()), float(cols.max())
    y0, y1 = float(rows.min()), float(rows.max())
    return LeftTargetObservation(
        box=BoxDetection(
            class_id=LEFT_SHAFT_CLASS_ID,
            class_name="target",
            confidence=float(score),
            xyxy=(x0, y0, x1, y1),
        ),
        tip_px=(float(tip[0]), float(tip[1])),
        mask=mask_bool,
        hinge_center_px=(float(hinge_center[0]), float(hinge_center[1])),
    )


def infer_left_target_boxes(
    model,
    frame_bgr,
    *,
    confidence: float,
    imgsz: int,
    device: str,
    selection: str = "nearest-depth",
    aligned_depth: np.ndarray | None = None,
    depth_scale_m: float = 0.001,
    previous_tip_px: tuple[float, float] | None = None,
    previous_hinge_px: tuple[float, float] | None = None,
    max_track_jump_px: float = DEFAULT_LEFT_TRACK_MAX_JUMP_PX,
    max_shaft_distance_px: float = 320.0,
    max_shaft_horizontal_distance_px: float = 140.0,
    shaft_ambiguity_margin_px: float = 20.0,
    overlay_detections: list[BoxDetection] | None = None,
    association_info: dict | None = None,
) -> list[LeftTargetObservation]:
    """Reuse the left grasp lock policy and initialize from the current scene.

    The first frame chooses the nearest-depth hinge (or the largest hinge when
    depth is unavailable), while retaining the left grasp ``leftmost``/
    ``rightmost`` options. Subsequent frames keep the hinge nearest to its
    previous center, then associate the nearest shaft with that locked hinge.
    A shaft is never allowed to replace the locked hinge, and a large jump
    returns no target so the geometry gate holds the arm.
    """
    if selection not in {"nearest-depth", "largest", "leftmost", "second-leftmost", "rightmost"}:
        raise ValueError(f"unsupported left hinge selection: {selection}")
    if not (
        math.isfinite(max_track_jump_px)
        and max_track_jump_px > 0.0
        and math.isfinite(max_shaft_distance_px)
        and max_shaft_distance_px > 0.0
        and math.isfinite(max_shaft_horizontal_distance_px)
        and max_shaft_horizontal_distance_px > 0.0
        and math.isfinite(shaft_ambiguity_margin_px)
        and shaft_ambiguity_margin_px >= 0.0
        and math.isfinite(depth_scale_m)
        and depth_scale_m > 0.0
    ):
        raise ValueError("left hinge/shaft association limits must be positive finite values")
    masks, scores, class_ids = infer_yolo_masks(
        model,
        frame_bgr,
        confidence=confidence,
        imgsz=imgsz,
        device=device,
    )
    hinge_items: list[dict] = []
    shaft_items: list[dict] = []
    raw_detections: list[BoxDetection] = []
    for mask, score, class_id in zip(masks, scores, class_ids):
        mask_bool = np.asarray(mask, dtype=bool)
        center = _mask_center(mask_bool)
        if center is None:
            continue
        if int(class_id) == LEFT_HINGE_CLASS_ID:
            hinge_items.append({
                "center": center,
                "score": float(score),
                "area": int(mask_bool.sum()),
                "mask": mask_bool,
                "depth_m": _mask_median_depth(mask_bool, aligned_depth, depth_scale_m),
            })
            raw_detections.append(_mask_box(mask_bool, LEFT_HINGE_CLASS_ID, "hinge", float(score)))
        elif int(class_id) == LEFT_SHAFT_CLASS_ID:
            shaft_items.append({"center": center, "score": float(score), "mask": mask_bool})
            raw_detections.append(_mask_box(mask_bool, LEFT_SHAFT_CLASS_ID, "shaft", float(score)))
    if overlay_detections is not None:
        overlay_detections.extend(raw_detections)
    if association_info is not None:
        association_info.update({"hinge_count": len(hinge_items), "shaft_count": len(shaft_items)})
    if not hinge_items or not shaft_items:
        if association_info is not None:
            association_info["reason"] = "missing_hinge_or_shaft"
        return []

    previous_tip = None if previous_tip_px is None else np.asarray(previous_tip_px, dtype=np.float64)
    previous_hinge = None if previous_hinge_px is None else np.asarray(previous_hinge_px, dtype=np.float64)
    if previous_tip is not None and (previous_tip.shape != (2,) or not np.all(np.isfinite(previous_tip))):
        previous_tip = None
    if previous_hinge is not None and (previous_hinge.shape != (2,) or not np.all(np.isfinite(previous_hinge))):
        previous_hinge = None
    if previous_hinge is None:
        if selection == "nearest-depth":
            depth_candidates = [item for item in hinge_items if item.get("depth_m") is not None]
            if depth_candidates:
                selected_hinge = min(depth_candidates, key=lambda item: (float(item["depth_m"]), -item["area"], -item["score"]))
                tracking_mode = "initial_nearest_depth"
            else:
                selected_hinge = max(hinge_items, key=lambda item: (item["area"], item["score"]))
                tracking_mode = "initial_largest_depth_unavailable"
        elif selection == "largest":
            selected_hinge = max(hinge_items, key=lambda item: (item["area"], item["score"]))
            tracking_mode = "initial_largest"
        elif selection == "leftmost":
            selected_hinge = min(hinge_items, key=lambda item: float(item["center"][0]))
            tracking_mode = "initial_leftmost"
        elif selection == "second-leftmost":
            ordered = sorted(hinge_items, key=lambda item: float(item["center"][0]))
            selected_hinge = ordered[1] if len(ordered) >= 2 else None
            tracking_mode = "initial_second_leftmost"
        else:
            selected_hinge = max(hinge_items, key=lambda item: float(item["center"][0]))
            tracking_mode = "initial_rightmost"
    else:
        nearby_hinges = [
            item for item in hinge_items
            if float(np.linalg.norm(item["center"] - previous_hinge)) <= max_track_jump_px
        ]
        selected_hinge = min(
            nearby_hinges,
            key=lambda item: float(np.linalg.norm(item["center"] - previous_hinge)),
            default=None,
        )
        tracking_mode = "temporal_lock"
    if selected_hinge is None:
        if association_info is not None:
            association_info["reason"] = "selected_hinge_unavailable"
        return []

    hinge_center = selected_hinge["center"]
    shaft_candidates: list[tuple[float, float, dict]] = []
    for shaft in shaft_items:
        delta = shaft["center"] - hinge_center
        distance = float(np.linalg.norm(delta))
        if distance <= max_shaft_distance_px and abs(float(delta[0])) <= max_shaft_horizontal_distance_px:
            shaft_candidates.append((distance, -float(shaft["score"]), shaft))
    if not shaft_candidates:
        if association_info is not None:
            association_info.update({"reason": "no_nearby_shaft", "tracking": tracking_mode})
        return []
    shaft_candidates.sort(key=lambda item: (item[0], item[1]))
    # On the first frame, reject an ambiguous shaft pair.  Once the hinge and
    # tip are locked, the previous-frame distance is the disambiguator and
    # must be allowed to select the same shaft even when pair distances tie.
    if (
        previous_tip is None
        and len(shaft_candidates) > 1
        and shaft_candidates[1][0] - shaft_candidates[0][0] < shaft_ambiguity_margin_px
    ):
        if association_info is not None:
            association_info.update({"reason": "ambiguous_nearby_shafts", "tracking": tracking_mode})
        return []

    candidates: list[tuple[float, float, float, dict, LeftTargetObservation]] = []
    for pair_distance, _negative_score, shaft in shaft_candidates:
        observation = _left_target_observation(shaft["mask"], shaft["score"], hinge_center)
        tip_jump = float("inf") if previous_tip is None else float(np.linalg.norm(np.asarray(observation.tip_px) - previous_tip))
        hinge_jump = float("inf") if previous_hinge is None else float(np.linalg.norm(hinge_center - previous_hinge))
        track_cost = tip_jump + 0.5 * hinge_jump if math.isfinite(tip_jump) else float("inf")
        candidates.append((track_cost, tip_jump, pair_distance, shaft, observation))
    if previous_tip is not None:
        tracked = [item for item in candidates if item[1] <= max_track_jump_px]
        if not tracked:
            if association_info is not None:
                association_info.update({"reason": "target_jump_exceeds_track_limit", "tracking": tracking_mode})
            return []
        candidates = tracked
        candidates.sort(key=lambda item: (item[0], item[2], -float(item[3]["score"])))
    else:
        candidates.sort(key=lambda item: (item[2], -float(item[3]["score"])))
    selected_pair = candidates[0]
    if association_info is not None:
        association_info.update({
            "reason": "matched_nearest_shaft",
            "tracking": tracking_mode,
            "hinge_center_px": [round(float(value), 1) for value in hinge_center],
            "hinge_depth_m": None if selected_hinge.get("depth_m") is None else round(float(selected_hinge["depth_m"]), 4),
            "shaft_center_px": [round(float(value), 1) for value in selected_pair[3]["center"]],
            "pair_distance_px": round(float(selected_pair[2]), 1),
            "tip_jump_px": None if not math.isfinite(selected_pair[1]) else round(float(selected_pair[1]), 1),
        })
    return [selected_pair[4]]


def left_target_from_offset(initial_pose: CartesianPoseTarget, offset_m: Sequence[float]) -> CartesianPoseTarget:
    """Build every left target from one measured origin; keep Z/orientation fixed."""
    values = tuple(float(value) for value in offset_m)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("left XY offset must contain three finite values")
    if abs(values[2]) > 1e-12:
        raise ValueError("bilateral alignment is XY-only; Z offset must be zero")
    return CartesianPoseTarget(
        tuple(float(initial_pose.position_m[index] + values[index]) for index in range(3)),
        initial_pose.orientation_xyzw,
    )


def draw_alignment_overlay(
    frame_bgr,
    detections,
    selected,
    state: str,
    error: tuple[float, float] | None,
    metric_error: tuple[float, float, float] | None = None,
):
    image = frame_bgr.copy()
    colors = {
        "hole": (0, 255, 0),
        "hinge": (255, 0, 255),
        "shaft": (0, 165, 255),
        "target": (0, 128, 255),
    }
    for item in detections:
        x0, y0, x1, y1 = [int(value) for value in item.xyxy]
        color = colors.get(item.class_name, (255, 0, 0))
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cx, cy = item.center_px
        cv2.circle(image, (int(cx), int(cy)), 4, color, -1)
        cv2.putText(image, f"{item.class_name} {item.confidence:.2f}", (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(image, f"bilateral={state}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if error is not None:
        cv2.putText(image, f"target-hole=({error[0]:.1f},{error[1]:.1f}) px", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if metric_error is not None:
        cv2.putText(
            image,
            "metric target-hole=(%.1f,%.1f,%.1f) mm" % tuple(1000.0 * float(value) for value in metric_error),
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )
    return image


def compact_frame(record: dict) -> dict:
    keys = (
        "event", "frame", "utc", "state", "reason", "motion_enabled",
        "sequence", "detections", "right_hole_lock", "hole_depth", "left_detections", "left_association", "pair_status", "pixel_error_px",
        "target_tip_px_left", "target_source",
        "geometry_mode", "geometry_reason", "geometry_error_m",
        "metric_requested_delta_m", "hole_base_m", "target_base_m",
        "right_hole_xyz_color_camera_m", "right_hole_xyz_right_end_link_m",
        "right_hole_xyz_base_link_m", "right_tf_query_timestamp_ns",
        "right_hole_xyz_source",
        "right_tf_timestamp_ns", "right_tf_age_ms", "right_tf_lookup_mode",
        "right_tf_lookup_error", "right_transform_chain",
        "right_color_timestamp_ns", "right_depth_timestamp_ns", "right_rgbd_skew_ms",
        "hole_depth_support", "hole_depth_stats", "target_depth_support", "left_tf_timestamp_ns", "geometry_agrees",
        "right_depth_registration", "hole_depth_region",
        "requested_delta_m", "accepted_delta_m", "cumulative_offset_m",
        "cumulative_distance_m", "step_index", "p4_verified", "gripper_verified",
        "left_start_precondition_enforced", "right_fixed",
    )
    return {key: record[key] for key in keys if key in record}


def _terminal_number(value, *, scale: float = 1.0, suffix: str = "") -> str:
    """Format an optional scalar for the one-line terminal status."""
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(number):
        return "-"
    return f"{number * scale:.1f}{suffix}"


def _terminal_vector(value, *, scale: float = 1000.0, suffix: str = "mm") -> str:
    """Format a three-dimensional XYZ vector for terminal display."""
    if value is None:
        return "-"
    try:
        numbers = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return "-"
    if len(numbers) != 3 or not all(math.isfinite(item) for item in numbers):
        return "-"
    return "(" + ",".join(f"{item * scale:.1f}" for item in numbers) + f"){suffix}"


def _terminal_point_distance(point: object) -> str:
    """Return the Euclidean distance of an XYZ point from its frame origin."""
    try:
        numbers = tuple(float(item) for item in point)
    except (TypeError, ValueError):
        return "-"
    if len(numbers) != 3 or not all(math.isfinite(item) for item in numbers):
        return "-"
    return _terminal_number(float(np.linalg.norm(numbers)), scale=1000.0, suffix="mm")


def terminal_summary(record: dict) -> str:
    """Return status and right-hole camera/base XYZ for in-place refresh."""
    hole_lock = record.get("right_hole_lock") or {}
    reason = str(record.get("reason") or "-").replace("\n", " ")
    if len(reason) > 34:
        reason = reason[:31] + "..."
    if hole_lock.get("locked"):
        locked = "LOCK"
    elif hole_lock.get("mode") == "disabled" and record.get("right_hole_xyz_color_camera_m") is not None:
        locked = "LIVE"
    elif record.get("right_hole_xyz_color_camera_m") is not None:
        # A raw candidate is visible for diagnosis only; it is never used by
        # the motion gate unless the lock is also true.
        locked = "RAW"
    else:
        locked = "----"
    hole_camera_xyz = record.get("right_hole_xyz_color_camera_m")
    hole_base_xyz = record.get("right_hole_xyz_base_link_m") or record.get("hole_base_m")
    return (
        f"frame={record.get('frame', '-')} state={record.get('state', '-')} "
        f"reason={reason} hole={locked} hole_cam={_terminal_vector(hole_camera_xyz)} "
        f"r_cam={_terminal_point_distance(hole_camera_xyz)} "
        f"hole_base={_terminal_vector(hole_base_xyz)} "
        f"r_base={_terminal_point_distance(hole_base_xyz)}"
    )


def emit_terminal_record(record: dict, output_mode: str) -> bool:
    """Print a frame without scrolling when attached to an interactive TTY.

    JSONL logging remains independent and always receives the full compact
    record.  When stdout is redirected, fall back to one JSON object per line
    so pipes and process supervisors keep their existing behavior.
    """
    live_summary = output_mode == "summary" and bool(getattr(sys.stdout, "isatty", lambda: False)())
    if live_summary:
        # Write the new text before clearing the old tail.  Clearing first
        # makes some terminals visibly blink and looks like a missing frame.
        sys.stdout.write("\r" + terminal_summary(record) + "\033[K")
        sys.stdout.flush()
        return True
    print(json.dumps(compact_frame(record), ensure_ascii=False), flush=True)
    return False


def log_record(stream, record: dict) -> None:
    payload = compact_frame(record) if record.get("event") == "frame" else record
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def latest_transform_to_base(tf, source_frame: str) -> tuple[dict, int | None]:
    """Read the dynamic ``T_base_source`` transform with its GDK timestamp."""
    result = tf.lookup_transform_latest("base_link", source_frame, return_timestamp=True)
    if isinstance(result, tuple) and len(result) == 2:
        transform, timestamp = result
    else:  # Defensive compatibility with older bindings.
        transform, timestamp = result, None
    return transform_to_dict(transform), None if timestamp is None else int(timestamp)


def transform_to_base_at(tf, source_frame: str, timestamp_ns: int) -> tuple[dict, int | None]:
    """Compatibility wrapper returning only the transform and its timestamp."""
    transform, timestamp, _ = transform_to_base_at_with_info(tf, source_frame, timestamp_ns)
    return transform, timestamp


def transform_to_base_at_with_info(
    tf, source_frame: str, timestamp_ns: int, *, static_fallback_max_gap_ms: float | None = None
) -> tuple[dict, int | None, dict]:
    """Read ``T_base_source`` and retain TF provenance for diagnostics.

    GDK rejects a query newer than its newest buffered transform.  Read the
    latest sample first, use an exact image-time lookup when it is available,
    and report a latest fallback instead of hiding it in the frame record.
    """
    latest: tuple[dict, int | None] | None = None
    latest_error: str | None = None
    if hasattr(tf, "lookup_transform_latest"):
        try:
            result = tf.lookup_transform_latest("base_link", source_frame, return_timestamp=True)
            if isinstance(result, tuple) and len(result) == 2:
                latest_transform, latest_timestamp = result
            else:  # Defensive compatibility with older bindings.
                latest_transform, latest_timestamp = result, None
            latest = (
                transform_to_dict(latest_transform),
                None if latest_timestamp is None else int(latest_timestamp),
            )
            # Avoid issuing a known-future query; this is common when the
            # camera timestamp is a few milliseconds newer than the TF buffer.
            if int(timestamp_ns) <= 0 or (
                latest[1] is not None and int(timestamp_ns) > int(latest[1])
            ):
                age_ms = None if latest[1] is None or int(timestamp_ns) <= 0 else (int(timestamp_ns) - int(latest[1])) / 1_000_000.0
                return latest[0], latest[1], {
                    "mode": "latest_static_fallback",
                    "query_timestamp_ns": int(timestamp_ns),
                    "transform_timestamp_ns": latest[1],
                    "age_ms": age_ms,
                    "exact_lookup_error": "image_timestamp_newer_than_latest_tf" if latest[1] is not None else None,
                }
            if (
                static_fallback_max_gap_ms is not None
                and latest[1] is not None
                and int(timestamp_ns) > 0
                and (int(latest[1]) - int(timestamp_ns)) > float(static_fallback_max_gap_ms) * 1_000_000.0
            ):
                gap_ms = (int(latest[1]) - int(timestamp_ns)) / 1_000_000.0
                return latest[0], latest[1], {
                    "mode": "latest_static_fallback",
                    "query_timestamp_ns": int(timestamp_ns),
                    "transform_timestamp_ns": latest[1],
                    "age_ms": -gap_ms,
                    "exact_lookup_error": f"image_timestamp_older_than_latest_tf_by_{gap_ms:.1f}ms",
                }
        except Exception as exc:
            latest_error = str(exc)
    if int(timestamp_ns) > 0 and hasattr(tf, "lookup_transform"):
        try:
            transform = tf.lookup_transform("base_link", source_frame, int(timestamp_ns))
            return transform_to_dict(transform), int(timestamp_ns), {
                "mode": "exact_image_time",
                "query_timestamp_ns": int(timestamp_ns),
                "transform_timestamp_ns": int(timestamp_ns),
                "age_ms": 0.0,
                "exact_lookup_error": None,
            }
        except Exception as exc:
            latest_error = str(exc)
    if latest is not None:
        age_ms = None if latest[1] is None or int(timestamp_ns) <= 0 else (int(timestamp_ns) - int(latest[1])) / 1_000_000.0
        return latest[0], latest[1], {
            "mode": "latest_static_fallback",
            "query_timestamp_ns": int(timestamp_ns),
            "transform_timestamp_ns": latest[1],
            "age_ms": age_ms,
            "exact_lookup_error": latest_error,
        }
    transform, timestamp = latest_transform_to_base(tf, source_frame)
    age_ms = None if timestamp is None or int(timestamp_ns) <= 0 else (int(timestamp_ns) - int(timestamp)) / 1_000_000.0
    return transform, timestamp, {
        "mode": "latest_static_fallback",
        "query_timestamp_ns": int(timestamp_ns),
        "transform_timestamp_ns": timestamp,
        "age_ms": age_ms,
        "exact_lookup_error": latest_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="P4 后右观左动的双臂 XY 定位伺服")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--left-model", type=Path, default=DEFAULT_LEFT_TARGET_MODEL, help="左 D405 hinge/shaft 分割权重；shaft 作为 target")
    parser.add_argument("--poses", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--right-preset", default="right_post_grasp")
    parser.add_argument("--left-p4-preset", default="left_p4")
    parser.add_argument(
        "--p4-tolerance-rad",
        type=float,
        default=0.03,
        help="仅记录左臂相对 P4 的偏差；不再作为启动运动的硬性校验",
    )
    parser.add_argument("--right-drift-tolerance-rad", type=float, default=0.03)
    parser.add_argument(
        "--gripper-closed-max-open-rad",
        type=float,
        default=0.50,
        help="仅记录夹爪开合状态；不再作为启动运动的硬性校验",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--right-crop-percent",
        type=float,
        default=60.0,
        help="右手图像中心裁剪比例；默认 60%%，用于排除画面边缘误检",
    )
    parser.add_argument(
        "--right-hole-track-max-jump-px",
        type=float,
        default=120.0,
        help="右手孔锁定跨帧最大跳变；超出时保持，不切换到另一候选",
    )
    parser.add_argument(
        "--disable-right-hole-lock",
        action="store_true",
        help="只读验证时关闭右孔跨帧锁定；不允许与 --allow-motion 同时使用",
    )
    parser.add_argument("--left-confidence", type=float, default=0.25)
    parser.add_argument(
        "--left-selection",
        choices=("nearest-depth", "largest", "leftmost", "second-leftmost", "rightmost"),
        default="nearest-depth",
        help="左手首帧铰链选择；默认选当前画面深度最近者，后续锁定该铰链",
    )
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--left-imgsz", type=int, default=640)
    parser.add_argument(
        "--left-track-max-jump-px",
        type=float,
        default=DEFAULT_LEFT_TRACK_MAX_JUMP_PX,
        help="左手轴/铰链跨帧最大允许跳变；超出时保持，不切换到邻近目标",
    )
    parser.add_argument("--left-shaft-max-distance-px", type=float, default=320.0)
    parser.add_argument("--left-shaft-max-horizontal-distance-px", type=float, default=140.0)
    parser.add_argument("--left-shaft-ambiguity-margin-px", type=float, default=20.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pixel-jacobian", default=None, help="右图 target-hole 像素 / 左 base XY 米，J00,J01,J10,J11；未提供时只读")
    parser.add_argument(
        "--right-depth-registration",
        choices=("reproject", "registered_same_pixel"),
        default="reproject",
        help="右手深度坐标模式：reproject=按官方外参重投影；registered_same_pixel=GDK 已注册到右彩色网格时直接同像素取样",
    )
    parser.add_argument(
        "--geometry-mode",
        choices=("off", "veto", "direct"),
        default="off",
        help="几何路径：off=旧右图 pixel-J，veto=旧右图 RGB-D 否决，direct=左 shaft/右 hole 双 RGB-D 三维 XY",
    )
    parser.add_argument(
        "--approve-rgbd-direct-motion",
        action="store_true",
        help="允许 direct 模式真实左臂 XY 运动；必须与 --allow-motion 同时使用",
    )
    parser.add_argument(
        "--approve-registered-depth-motion",
        action="store_true",
        help="额外确认右手 registered_same_pixel 深度已经只读验证；必须与 --allow-motion 同时使用",
    )
    parser.add_argument("--metric-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--metric-pair-jump-mm", type=float, default=8.0)
    parser.add_argument("--metric-max-error-mm", type=float, default=60.0)
    parser.add_argument("--min-depth-pixels", type=int, default=8)
    parser.add_argument("--min-depth-support", type=float, default=0.20)
    parser.add_argument("--max-depth-mad-mm", type=float, default=15.0)
    parser.add_argument("--hole-depth-window-frames", type=int, default=5, help="右孔深度时间中值窗口帧数")
    parser.add_argument("--hole-depth-min-temporal-hits", type=int, default=3, help="每个深度像素至少出现的有效帧数")
    parser.add_argument("--hole-depth-max-window-ms", type=float, default=500.0, help="右孔深度窗口最大时间跨度")
    parser.add_argument("--hole-depth-max-center-jitter-px", type=float, default=8.0, help="深度窗口允许的孔中心抖动")
    parser.add_argument("--hole-depth-max-size-change-fraction", type=float, default=0.25, help="深度窗口允许的孔框尺寸变化比例")
    parser.add_argument("--hole-depth-max-temporal-mad-mm", type=float, default=15.0, help="深度窗口逐像素时间 MAD 上限")
    parser.add_argument("--hole-depth-scales", default="1,1.5,2,3,4,6", help="右孔深度采样尺度，逗号分隔且必须从1开始")
    parser.add_argument(
        "--hole-depth-region",
        choices=("ring", "box"),
        default="ring",
        help="右孔有效深度采样区域：ring=孔边缘环带；box=检测框内全部有效像素（不扩张）",
    )
    parser.add_argument("--hole-depth-max-padding-px", type=float, default=300.0, help="右孔深度采样最大扩张像素")
    parser.add_argument("--hole-depth-min-covered-sectors", type=int, default=3, help="右孔环形深度至少覆盖的象限数")
    parser.add_argument("--max-rgbd-skew-ms", type=float, default=100.0)
    parser.add_argument(
        "--right-tf-static-fallback-max-gap-ms",
        type=float,
        default=25.0,
        help="右臂固定观察时，图像时间落后最新 TF 超过该值则直接使用最新 TF，避免无意义的历史缓存报错",
    )
    parser.add_argument("--pixel-tolerance-px", type=float, default=8.0)
    parser.add_argument("--stable-frames", type=int, default=4)
    parser.add_argument("--aligned-frames", type=int, default=5)
    parser.add_argument("--max-pair-jump-px", type=float, default=25.0)
    parser.add_argument("--max-error-px", type=float, default=260.0)
    parser.add_argument("--max-step-mm", type=float, default=1.5)
    parser.add_argument("--max-total-displacement-mm", type=float, default=10.0)
    parser.add_argument("--max-total-travel-mm", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--left-speed-mm-s", type=float, default=10.0)
    parser.add_argument("--step-settle-s", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示持续运行")
    parser.add_argument("--max-duration-s", type=float, default=0.0, help="单次定位最长运行时间；0 表示持续运行")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8776, help="0 表示关闭网页")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument(
        "--terminal-output",
        choices=("summary", "json"),
        default="summary",
        help="终端帧输出：summary 在交互终端原地刷新；json 每帧输出一行 JSON",
    )
    parser.add_argument("--allow-motion", action="store_true", help="允许右臂一次预移动和左臂 XY 伺服")
    parser.add_argument("--skip-right-premove", action="store_true", help="不移动右臂；测试当前观察位")
    args = parser.parse_args()
    if args.disable_right_hole_lock and args.allow_motion:
        parser.error("--disable-right-hole-lock 只允许只读验证，不能与 --allow-motion 同时使用")
    terminal_output_mode = args.terminal_output
    terminal_line_active = False

    try:
        jacobian = parse_jacobian(args.pixel_jacobian)
        depth_scales = parse_depth_scales(args.hole_depth_scales)
        policy = AlignmentPolicy(
            pixel_tolerance_px=args.pixel_tolerance_px,
            stable_frames=args.stable_frames,
            aligned_frames=args.aligned_frames,
            max_pair_jump_px=args.max_pair_jump_px,
            max_error_px=args.max_error_px,
            max_step_m=args.max_step_mm / 1000.0,
            max_total_displacement_m=args.max_total_displacement_mm / 1000.0,
            max_total_travel_m=args.max_total_travel_mm / 1000.0,
            max_steps=args.max_steps,
            min_confidence=args.confidence,
        )
        policy.validate()
        if args.allow_motion and args.geometry_mode in {"off", "veto"} and jacobian is None:
            raise ValueError("允许运动时必须提供右相机实测 --pixel-jacobian；当前没有可复用的左 D405 Jacobian")
        if args.geometry_mode == "direct" and args.pixel_jacobian is not None:
            raise ValueError("direct 模式不接受 --pixel-jacobian；避免两套控制律混用")
        if args.geometry_mode == "direct" and args.allow_motion and not args.approve_rgbd_direct_motion:
            raise ValueError("direct 真实运动必须同时提供 --approve-rgbd-direct-motion")
        if args.approve_rgbd_direct_motion and not args.allow_motion:
            raise ValueError("--approve-rgbd-direct-motion 必须与 --allow-motion 同时使用")
        if args.approve_rgbd_direct_motion and args.geometry_mode != "direct":
            raise ValueError("--approve-rgbd-direct-motion 只能用于 --geometry-mode direct")
        if (
            args.allow_motion
            and args.geometry_mode == "direct"
            and args.right_depth_registration == "registered_same_pixel"
            and not args.approve_registered_depth_motion
        ):
            raise ValueError(
                "registered_same_pixel 真实运动必须先完成只读验证，并同时提供 --approve-registered-depth-motion"
            )
        if args.approve_registered_depth_motion and not args.allow_motion:
            raise ValueError("--approve-registered-depth-motion 必须与 --allow-motion 同时使用")
        if args.approve_registered_depth_motion and (
            args.geometry_mode != "direct" or args.right_depth_registration != "registered_same_pixel"
        ):
            raise ValueError("--approve-registered-depth-motion 只能用于 direct + registered_same_pixel")
        if not (0.0 < args.right_crop_percent <= 100.0) or not math.isfinite(args.right_crop_percent):
            raise ValueError("--right-crop-percent 必须在 0 到 100 之间")
        if not math.isfinite(args.right_hole_track_max_jump_px) or args.right_hole_track_max_jump_px <= 0.0:
            raise ValueError("--right-hole-track-max-jump-px 必须是正数")
        if not (
            0.0 <= args.confidence <= 1.0 and 0.0 <= args.left_confidence <= 1.0
            and 0.0 <= args.iou <= 1.0 and args.imgsz > 0 and args.left_imgsz > 0
            and args.max_frames >= 0 and 0.0 <= args.max_duration_s <= 600.0
            and 0.001 <= args.p4_tolerance_rad <= 0.10
            and 0.001 <= args.right_drift_tolerance_rad <= 0.20
            and 0.05 <= args.gripper_closed_max_open_rad <= 0.785
            and 0.1 <= args.left_speed_mm_s <= 50.0 and 0.0 <= args.step_settle_s <= 5.0
            and 1.0 <= args.right_tf_static_fallback_max_gap_ms <= 5000.0
            and 1.0 <= args.left_track_max_jump_px <= 200.0
            and 1.0 <= args.right_hole_track_max_jump_px <= 1000.0
            and 1 <= args.hole_depth_window_frames <= 30
            and 1 <= args.hole_depth_min_temporal_hits <= args.hole_depth_window_frames
            and 10.0 <= args.hole_depth_max_window_ms <= 5000.0
            and 0.5 <= args.hole_depth_max_center_jitter_px <= 100.0
            and 0.0 <= args.hole_depth_max_size_change_fraction < 1.0
            and 0.1 <= args.hole_depth_max_temporal_mad_mm <= 200.0
            and 0.0 <= args.hole_depth_max_padding_px <= 500.0
            and 1 <= args.hole_depth_min_covered_sectors <= 4
            and 1.0 <= args.left_shaft_max_distance_px <= 1000.0
            and 1.0 <= args.left_shaft_max_horizontal_distance_px <= 500.0
            and 0.0 <= args.left_shaft_ambiguity_margin_px <= 500.0
            and 0 <= args.web_port <= 65535
            and 0.001 <= args.metric_tolerance_mm <= 100.0
            and args.metric_tolerance_mm < args.metric_pair_jump_mm <= 500.0
            and args.metric_tolerance_mm < args.metric_max_error_mm <= 1000.0
            and 1 <= args.min_depth_pixels <= 100000
            and 0.0 < args.min_depth_support <= 1.0
            and 0.1 <= args.max_depth_mad_mm <= 200.0
            and 1.0 <= args.max_rgbd_skew_ms <= 1000.0
        ):
            raise ValueError("检测、P4、速度、伺服或网页参数无效")
        model_path = args.model.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到 hole/target 权重：{model_path}")
        left_model_path = args.left_model.expanduser().resolve()
        if args.geometry_mode == "direct" and not left_model_path.is_file():
            raise FileNotFoundError(f"找不到左 hinge/shaft 权重：{left_model_path}")
        left_p4 = None
        left_start_diagnostic_error = None
        try:
            left_p4 = load_pose(args.poses, args.left_p4_preset, "left")
        except Exception as exc:
            # P4 is diagnostic-only now.  The motion stream starts at the
            # measured current pose, so a missing/invalid left P4 preset must
            # not prevent the operator-authorized run.
            left_start_diagnostic_error = str(exc)
        right_target = load_pose(args.poses, args.right_preset, "right")
    except Exception as exc:
        print(f"参数/姿态无效：{exc}", file=sys.stderr)
        return 2

    log_path = (args.log or PROJECT_ROOT / "logs" / f"bilateral_alignment_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl").expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "mode": "motion" if args.allow_motion else "read_only_preview",
        "stage": "post_p4_bilateral_xy_alignment",
        "camera": "right_hand_color + left_hand_color" if args.geometry_mode == "direct" else "right_hand_color",
        "geometry_mode": args.geometry_mode,
        "moving_arm": "left",
        "model": str(model_path),
        "left_model": str(left_model_path) if args.geometry_mode == "direct" else None,
        "pixel_jacobian": jacobian,
        "right_premove": None if args.skip_right_premove else args.right_preset,
        "p4_preset": args.left_p4_preset,
        "left_start_precondition_policy": "skipped_by_explicit_user_request",
        "left_start_precondition_enforced": False,
        "p4_verification_enforced": False,
        "gripper_verification_enforced": False,
        "left_selection": args.left_selection,
        "right_crop_percent": args.right_crop_percent,
        "right_hole_track_max_jump_px": args.right_hole_track_max_jump_px,
        "hole_depth_scales": depth_scales,
        "hole_depth_region": args.hole_depth_region,
        "right_depth_registration": args.right_depth_registration,
        "right_point_frame": "right_hand_color_optical_assumed",
        "right_transform_chain": "P_base=T_base_arm_r_end(T_arm_r_end_right_rgbd P_right_color)",
        "right_tf_static_fallback_max_gap_ms": args.right_tf_static_fallback_max_gap_ms,
        "registered_depth_motion_approved": args.approve_registered_depth_motion,
        "hole_depth_window_frames": args.hole_depth_window_frames,
        "hole_depth_min_temporal_hits": args.hole_depth_min_temporal_hits,
        "hole_depth_max_temporal_mad_mm": args.hole_depth_max_temporal_mad_mm,
        "max_duration_s": args.max_duration_s,
        "log": str(log_path),
    }, ensure_ascii=False))

    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    initialized = False
    server = None
    shared = None
    stream = None
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        current = controller.checked_arm_state()
        p4_error = None if left_p4 is None else max(abs(actual - expected) for actual, expected in zip(current[:7], left_p4))
        gripper_diagnostic_error = None
        try:
            gripper_position = controller.read_gripper_position("left")
        except Exception as exc:
            gripper_position = None
            gripper_diagnostic_error = str(exc)
        p4_verified = None if p4_error is None else p4_error <= args.p4_tolerance_rad
        gripper_verified = None if gripper_position is None else abs(gripper_position) <= args.gripper_closed_max_open_rad
        # The operator explicitly owns the precondition that the left arm is
        # already in the intended post-P4 pose and is holding the workpiece.
        # Keep these readings for observability, but never block motion on
        # them: the current left pose is the origin of the XY position stream.
        left_start_precondition_policy = "skipped_by_explicit_user_request"
        left_start_precondition_enforced = False
        p4_verification_enforced = False
        gripper_verification_enforced = False
        if args.allow_motion:
            print(
                "左 P4 位置和夹爪闭合状态未作为运动前置条件；"
                "程序不会自动调整左臂姿态，也不会打开夹爪。"
                f" (p4_error={p4_error}, gripper={gripper_position}, "
                f"diagnostic_error={gripper_diagnostic_error or left_start_diagnostic_error})",
                flush=True,
            )
        right_reference = tuple(current[RIGHT_ARM_OFFSET:RIGHT_ARM_OFFSET + 7])
        if not args.skip_right_premove and args.allow_motion:
            controller.move_preset("right", right_target, velocity_rad_s=0.08, life_time_s=25.0, detail="bilateral right observer premove")
            deadline = time.monotonic() + 25.0
            while time.monotonic() < deadline:
                measured = controller.checked_arm_state()[RIGHT_ARM_OFFSET:RIGHT_ARM_OFFSET + 7]
                if max(abs(actual - expected) for actual, expected in zip(measured, right_target)) <= args.right_drift_tolerance_rad:
                    right_reference = tuple(measured)
                    break
                time.sleep(0.10)
            else:
                raise RuntimeError("右臂观察位预移动超时；未启动左臂伺服")
        elif not args.skip_right_premove:
            print("只读预检：读取 right_post_grasp，但未发送右臂预移动。")

        model = load_detection_model(str(model_path), device=args.device)
        left_model = None
        if args.geometry_mode == "direct":
            left_model = load_yolo_segmentation_model(str(left_model_path), device=args.device)
        color_type = agibot_gdk.CameraType.kHandRightColor
        depth_type = agibot_gdk.CameraType.kHandRightDepth
        left_color_type = agibot_gdk.CameraType.kHandLeftColor
        left_depth_type = agibot_gdk.CameraType.kHandLeftDepth
        need_rgbd = args.geometry_mode != "off"
        need_bilateral = args.geometry_mode == "direct"
        camera_types = [color_type, depth_type, left_color_type, left_depth_type] if need_bilateral else [color_type, depth_type] if need_rgbd else [color_type]
        camera = agibot_gdk.Camera(camera_types)
        tf = agibot_gdk.TF() if need_rgbd else None
        depth_to_color = None
        rgbd_to_right_end = None
        left_depth_to_color = None
        left_rgbd_to_end = None
        geometry_startup_error = None
        if need_rgbd:
            startup_errors: list[str] = []
            try:
                depth_to_color = transform_to_dict(
                    tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kRightHandDepthToRightHandColor)
                )
            except Exception as exc:
                startup_errors.append(f"right_depth_to_color: {exc}")
            try:
                rgbd_to_right_end = transform_to_dict(
                    tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kRightHandRGBDToArmREndLink)
                )
            except Exception as exc:
                startup_errors.append(f"right_rgbd_to_end: {exc}")
            if need_bilateral:
                try:
                    left_depth_to_color = transform_to_dict(
                        tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor)
                    )
                except Exception as exc:
                    startup_errors.append(f"left_depth_to_color: {exc}")
                try:
                    left_rgbd_to_end = transform_to_dict(
                        tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink)
                    )
                except Exception as exc:
                    startup_errors.append(f"left_rgbd_to_end: {exc}")
            geometry_startup_error = None if not startup_errors else "official_rgbd_calibration_unavailable: " + "; ".join(startup_errors)
        if args.allow_motion and need_bilateral and (depth_to_color is None or rgbd_to_right_end is None or left_depth_to_color is None or left_rgbd_to_end is None):
            raise RuntimeError(
                "拒绝双相机 RGB-D 运动：GDK 未返回完整左右手官方外参；先用只读 direct 检查运行环境。"
                f" {geometry_startup_error or ''}".strip()
            )
        color_intrinsic = None
        depth_intrinsic = None
        left_color_intrinsic = None
        left_depth_intrinsic = None
        if args.web_port:
            shared = SharedView()
            server = ThreadingHTTPServer((args.web_host, args.web_port), handler_class(shared))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"实时页面：http://{args.web_host}:{args.web_port}")

        pixel_gate = None if jacobian is None else BilateralAlignmentGate(jacobian, policy)
        metric_policy = MetricAlignmentPolicy(
            position_tolerance_m=args.metric_tolerance_mm / 1000.0,
            stable_frames=args.stable_frames,
            aligned_frames=args.aligned_frames,
            max_pair_jump_m=args.metric_pair_jump_mm / 1000.0,
            max_error_m=args.metric_max_error_mm / 1000.0,
            max_step_m=args.max_step_mm / 1000.0,
            max_total_displacement_m=args.max_total_displacement_mm / 1000.0,
            max_total_travel_m=args.max_total_travel_mm / 1000.0,
            max_steps=args.max_steps,
            min_confidence=args.confidence,
        )
        metric_gate = BilateralMetricAlignmentGate(metric_policy) if need_rgbd else None
        initial_pose: CartesianPoseTarget | None = None
        if args.allow_motion:
            if controller.motion_status_summary()["control_mode"] != 1:
                controller.set_control_mode(1)
            initial_pose = controller.read_end_effector_pose(LEFT_END_LINK)
            stream = controller.start_left_end_effector_position_stream(initial_pose, rate_hz=50.0, life_time_s=0.10, max_translation_velocity_mps=args.left_speed_mm_s / 1000.0)
        right_fixed = True
        sequence = 0
        started_at = time.monotonic()
        last_camera_timestamp = 0
        last_left_camera_timestamp = 0
        step_settle_until = 0.0
        aligned = False
        last_state = "acquire"
        fatal_fault = False
        previous_left_target_px: tuple[float, float] | None = None
        previous_left_hinge_px: tuple[float, float] | None = None
        previous_right_hole_px: tuple[float, float] | None = None
        hole_depth_accumulator = FreshHoleDepthAccumulator(
            window_frames=args.hole_depth_window_frames,
            min_temporal_hits=args.hole_depth_min_temporal_hits,
            max_window_ms=args.hole_depth_max_window_ms,
            max_center_jitter_px=args.hole_depth_max_center_jitter_px,
            max_size_change_fraction=args.hole_depth_max_size_change_fraction,
            max_temporal_mad_mm=args.hole_depth_max_temporal_mad_mm,
            admission_min_valid_pixels=args.min_depth_pixels,
            admission_min_support_fraction=args.min_depth_support,
        )
        with log_path.open("a", encoding="utf-8") as log:
            log_record(log, {
                "event": "start", "utc": datetime.now(timezone.utc).isoformat(),
                "stage": "post_p4_bilateral_xy_alignment", "model": str(model_path),
                "camera": "right_hand_color + left_hand_color" if need_bilateral else "right_hand_color",
                "motion_enabled": args.allow_motion, "p4_verified": p4_verified,
                "p4_error_rad": p4_error, "left_gripper_position_rad": gripper_position,
                "gripper_verified": gripper_verified,
                "left_start_precondition_policy": left_start_precondition_policy,
                "left_start_precondition_enforced": left_start_precondition_enforced,
                "p4_verification_enforced": p4_verification_enforced,
                "gripper_verification_enforced": gripper_verification_enforced,
                "left_start_diagnostic_error": left_start_diagnostic_error or gripper_diagnostic_error,
                "left_selection": args.left_selection,
                "right_crop_percent": args.right_crop_percent,
                "right_hole_track_max_jump_px": args.right_hole_track_max_jump_px,
                "right_hole_lock_enabled": not args.disable_right_hole_lock,
                "hole_depth_scales": depth_scales,
                "hole_depth_window_frames": args.hole_depth_window_frames,
                "hole_depth_min_temporal_hits": args.hole_depth_min_temporal_hits,
                "hole_depth_max_temporal_mad_mm": args.hole_depth_max_temporal_mad_mm,
                "hole_depth_admission_min_valid_pixels": args.min_depth_pixels,
                "hole_depth_admission_min_support_fraction": args.min_depth_support,
                "max_duration_s": args.max_duration_s,
                "right_premove": None if args.skip_right_premove else args.right_preset,
                "right_fixed": True, "pixel_jacobian": jacobian,
                "geometry_mode": args.geometry_mode,
                "right_point_frame": "right_hand_color_optical_assumed",
                "right_transform_chain": "P_base=T_base_arm_r_end(T_arm_r_end_right_rgbd P_right_color)",
                "right_tf_static_fallback_max_gap_ms": args.right_tf_static_fallback_max_gap_ms,
                "official_right_calibration": {
                    "depth_to_color": depth_to_color is not None,
                    "rgbd_to_right_end": rgbd_to_right_end is not None,
                    "rgbd_to_right_end_enum": "kRightHandRGBDToArmREndLink",
                    "dynamic_tf": "lookup_transform(base_link, arm_r_end_link, camera_timestamp_ns)",
                },
                "official_left_calibration": {
                    "depth_to_color": left_depth_to_color is not None,
                    "rgbd_to_end": left_rgbd_to_end is not None,
                } if need_bilateral else None,
                "geometry_startup_error": geometry_startup_error,
                "policy": policy.__dict__,
                "metric_policy": metric_policy.__dict__,
            })
            while (
                (args.max_frames == 0 or sequence < args.max_frames)
                and (args.max_duration_s <= 0.0 or time.monotonic() - started_at < args.max_duration_s)
            ):
                sequence += 1
                frame_record = {
                    "event": "frame", "frame": sequence,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "motion_enabled": args.allow_motion, "p4_verified": p4_verified,
                    "gripper_verified": gripper_verified,
                    "left_start_precondition_enforced": left_start_precondition_enforced,
                    "right_fixed": right_fixed, "sequence": sequence,
                }
                if args.allow_motion:
                    measured_right = controller.checked_arm_state()[RIGHT_ARM_OFFSET:RIGHT_ARM_OFFSET + 7]
                    right_drift = max(abs(actual - expected) for actual, expected in zip(measured_right, right_reference))
                    if right_drift > args.right_drift_tolerance_rad:
                        frame_record.update({"state": "fault_hold", "reason": "right_observer_drift", "right_drift_rad": right_drift})
                        log_record(log, frame_record)
                        terminal_line_active = emit_terminal_record(frame_record, terminal_output_mode) or terminal_line_active
                        fatal_fault = True
                        break
                try:
                    image = camera.get_latest_image(color_type, 1000.0)
                    camera_timestamp = int(getattr(image, "timestamp_ns", 0))
                    if camera_timestamp > 0 and camera_timestamp <= last_camera_timestamp:
                        frame_record.update({"state": "hold", "reason": "duplicate_or_old_camera_frame"})
                        log_record(log, frame_record)
                        terminal_line_active = emit_terminal_record(frame_record, terminal_output_mode) or terminal_line_active
                        continue
                    if camera_timestamp > 0:
                        last_camera_timestamp = camera_timestamp
                    color_rgb = decode_color_rgb(image, agibot_gdk)
                    frame_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                    right_infer_frame_bgr, right_crop_offset = center_crop(frame_bgr, args.right_crop_percent)
                except Exception as exc:
                    frame_record.update({"state": "fault_hold", "reason": "color_perception_failure", "error": str(exc)})
                    log_record(log, frame_record)
                    terminal_line_active = emit_terminal_record(frame_record, terminal_output_mode) or terminal_line_active
                    fatal_fault = True
                    break

                aligned_depth = None
                depth_timestamp = None
                geometry_reason = geometry_startup_error
                right_end_to_base = None
                tf_timestamp = None
                right_tf_info: dict = {}
                left_image = None
                left_color_rgb = None
                left_frame_bgr = None
                left_aligned_depth = None
                left_end_to_base = None
                left_tf_timestamp = None
                if need_rgbd:
                    try:
                        if args.right_depth_registration == "reproject" and depth_to_color is None:
                            raise RuntimeError(geometry_startup_error or "official right depth-to-color calibration is unavailable")
                        depth_image = camera.get_nearest_image(depth_type, camera_timestamp, 1000.0)
                        depth_timestamp = int(getattr(depth_image, "timestamp_ns", 0))
                        if depth_timestamp <= 0:
                            raise RuntimeError("invalid right depth timestamp")
                        frame_record["right_color_timestamp_ns"] = camera_timestamp
                        frame_record["right_depth_timestamp_ns"] = depth_timestamp
                        frame_record["right_rgbd_skew_ms"] = abs(depth_timestamp - camera_timestamp) / 1_000_000.0
                        depth_u16 = decode_depth_u16(depth_image, agibot_gdk)
                        if color_intrinsic is None:
                            color_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(color_type),
                                width=int(image.width),
                                height=int(image.height),
                            )
                        if depth_intrinsic is None:
                            depth_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(depth_type),
                                width=int(depth_image.width),
                                height=int(depth_image.height),
                            )
                        aligned_depth = depth_on_color_grid(
                            depth_u16,
                            registration_mode=args.right_depth_registration,
                            depth_intrinsic=depth_intrinsic,
                            color_intrinsic=color_intrinsic,
                            depth_to_color=depth_to_color,
                        )
                        try:
                            right_end_to_base, tf_timestamp, right_tf_info = transform_to_base_at_with_info(
                                tf,
                                "arm_r_end_link",
                                camera_timestamp,
                                static_fallback_max_gap_ms=args.right_tf_static_fallback_max_gap_ms,
                            )
                        except Exception as exc:
                            # Camera-frame XYZ remains useful for the read-only
                            # hole check even when the optional base TF is late.
                            right_end_to_base = None
                            tf_timestamp = None
                            geometry_reason = f"right_end_tf_unavailable: {exc}"
                    except Exception as exc:
                        geometry_reason = str(exc)

                if need_bilateral:
                    try:
                        left_image = camera.get_latest_image(left_color_type, 1000.0)
                        left_timestamp = int(getattr(left_image, "timestamp_ns", 0))
                        if left_timestamp <= 0:
                            raise RuntimeError("invalid left color timestamp")
                        if left_timestamp <= last_left_camera_timestamp:
                            raise RuntimeError("duplicate_or_old_left_camera_frame")
                        last_left_camera_timestamp = left_timestamp
                        left_color_rgb = decode_color_rgb(left_image, agibot_gdk)
                        left_frame_bgr = cv2.cvtColor(left_color_rgb, cv2.COLOR_RGB2BGR)
                        left_depth_image = camera.get_nearest_image(left_depth_type, left_timestamp, 1000.0)
                        left_depth_timestamp = int(getattr(left_depth_image, "timestamp_ns", 0))
                        if left_depth_timestamp <= 0:
                            raise RuntimeError("invalid left depth timestamp")
                        if abs(left_depth_timestamp - left_timestamp) > args.max_rgbd_skew_ms * 1_000_000.0:
                            raise RuntimeError(
                                f"left RGB-D timestamp skew exceeds limit: {abs(left_depth_timestamp - left_timestamp) / 1_000_000.0:.1f} ms"
                            )
                        left_depth_u16 = decode_depth_u16(left_depth_image, agibot_gdk)
                        if color_intrinsic is None:
                            color_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(color_type), width=int(image.width), height=int(image.height)
                            )
                        if depth_intrinsic is None:
                            depth_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(depth_type), width=int(depth_image.width), height=int(depth_image.height)
                            )
                        if left_color_intrinsic is None:
                            left_color_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(left_color_type), width=int(left_image.width), height=int(left_image.height)
                            )
                        if left_depth_intrinsic is None:
                            left_depth_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(left_depth_type), width=int(left_depth_image.width), height=int(left_depth_image.height)
                            )
                        if left_depth_to_color is None or left_rgbd_to_end is None:
                            raise RuntimeError("official_left_rgbd_calibration_unavailable")
                        left_aligned_depth = align_depth_to_color(
                            left_depth_u16,
                            depth_intrinsic=left_depth_intrinsic,
                            color_intrinsic=left_color_intrinsic,
                            depth_to_color=left_depth_to_color,
                        )
                        left_end_to_base, left_tf_timestamp = transform_to_base_at(tf, LEFT_END_LINK, left_timestamp)
                    except Exception as exc:
                        geometry_reason = f"left_target_perception_unavailable: {exc}"

                try:
                    detections = infer_detection(
                        model,
                        right_infer_frame_bgr,
                        confidence=args.confidence,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        device=args.device,
                    )
                except Exception as exc:
                    frame_record.update({"state": "fault_hold", "reason": "detector_failure", "error": str(exc)})
                    log_record(log, frame_record)
                    terminal_line_active = emit_terminal_record(frame_record, terminal_output_mode) or terminal_line_active
                    fatal_fault = True
                    break
                # Inference runs on the center crop to reject edge-of-workcell
                # false positives. Geometry still uses full-frame pixels, so
                # map boxes back before depth/TF calculations.
                crop_detections = detections
                detections = [translate_detection(item, right_crop_offset) for item in crop_detections]
                all_holes = [item for item in detections if item.class_name == "hole"]
                if args.disable_right_hole_lock:
                    selected_hole = max(all_holes, key=lambda item: float(item.confidence), default=None)
                    right_lock_reason = "lock_disabled_current_frame" if selected_hole is not None else "missing_hole"
                else:
                    selected_hole, right_lock_reason = select_locked_hole(
                        all_holes,
                        previous_right_hole_px,
                        max_jump_px=args.right_hole_track_max_jump_px,
                    )
                holes = [selected_hole] if selected_hole is not None else []
                if selected_hole is not None and not args.disable_right_hole_lock:
                    previous_right_hole_px = selected_hole.center_px
                # Keep a raw detector candidate available for read-only XYZ
                # output when temporal locking rejects it.  This candidate is
                # deliberately not added to ``holes`` and cannot drive motion.
                preview_hole = selected_hole
                preview_xyz_source = "locked_hole_preview"
                if preview_hole is None and all_holes:
                    preview_hole = max(all_holes, key=lambda item: float(item.confidence))
                    preview_xyz_source = "raw_detection_preview"
                elif args.disable_right_hole_lock and preview_hole is not None:
                    preview_xyz_source = "current_frame_detection"
                # Terminal/read-only visibility must not wait for the motion
                # controller's temporal window.  Produce a provisional XYZ
                # from this frame's raw registered depth; the multi-frame
                # window below remains the separate safety gate for motion.
                right_hole_xyz_preview = None
                if preview_hole is not None and aligned_depth is not None and color_intrinsic is not None:
                    try:
                        right_crop_guard = (
                            float(right_crop_offset[0]),
                            float(right_crop_offset[1]),
                            float(right_crop_offset[0] + right_infer_frame_bgr.shape[1] - 1),
                            float(right_crop_offset[1] + right_infer_frame_bgr.shape[0] - 1),
                        )
                        preview_depth, _ = estimate_hole_depth_adaptive(
                            aligned_depth,
                            preview_hole.xyxy,
                            intrinsic=color_intrinsic,
                            scales=(1.0,),
                            guard_xyxy=right_crop_guard,
                            max_padding_px=0.0,
                            min_valid_pixels=1,
                            min_support_fraction=0.0,
                            min_covered_sectors=1,
                            max_mad_m=args.max_depth_mad_mm / 1000.0,
                            sampling_region="box",
                        )
                        right_hole_xyz_preview = preview_depth.point_color_m
                    except Exception:
                        # A provisional display sample is best-effort only;
                        # the normal depth/window diagnostics report the reason.
                        right_hole_xyz_preview = None
                right_hole_xyz_end_preview = None
                right_hole_xyz_base_preview = None
                if right_hole_xyz_preview is not None and rgbd_to_right_end is not None and right_end_to_base is not None:
                    try:
                        right_hole_xyz_end_preview = apply_transform(rgbd_to_right_end, right_hole_xyz_preview)
                        right_hole_xyz_base_preview = right_color_point_to_base(
                            right_hole_xyz_preview, rgbd_to_right_end, right_end_to_base
                        )
                    except Exception:
                        right_hole_xyz_end_preview = None
                        right_hole_xyz_base_preview = None
                hole_depth_window = None
                hole_depth_window_box = None
                hole_depth_window_ready = False
                hole_depth_diagnostics: dict = {}
                if selected_hole is None:
                    hole_depth_accumulator.reset(right_lock_reason)
                    hole_depth_diagnostics = {
                        "state": "hold",
                        "window_frames": 0,
                        "reason": right_lock_reason,
                    }
                else:
                    hole_depth_diagnostics = {
                        "state": "hold",
                        "window_frames": 0,
                        "reason": geometry_reason or "waiting_for_target_pair",
                    }
                left_targets: list[LeftTargetObservation] = []
                left_overlay_detections: list[BoxDetection] = []
                left_association: dict = {}
                had_left_track_reference = previous_left_target_px is not None
                if need_bilateral and left_model is not None and left_color_rgb is not None:
                    try:
                        left_targets = infer_left_target_boxes(
                            left_model,
                            left_frame_bgr,
                            confidence=args.left_confidence,
                            imgsz=args.left_imgsz,
                            device=args.device,
                            selection=args.left_selection,
                            aligned_depth=left_aligned_depth,
                            depth_scale_m=0.001,
                            previous_tip_px=previous_left_target_px,
                            previous_hinge_px=previous_left_hinge_px,
                            max_track_jump_px=args.left_track_max_jump_px,
                            max_shaft_distance_px=args.left_shaft_max_distance_px,
                            max_shaft_horizontal_distance_px=args.left_shaft_max_horizontal_distance_px,
                            shaft_ambiguity_margin_px=args.left_shaft_ambiguity_margin_px,
                            overlay_detections=left_overlay_detections,
                            association_info=left_association,
                        )
                        if left_targets:
                            previous_left_target_px = left_targets[0].tip_px
                            previous_left_hinge_px = left_targets[0].hinge_center_px
                    except Exception as exc:
                        geometry_reason = f"left_target_detector_failure: {exc}"
                # Legacy off/veto modes keep their original single-right-camera
                # semantics.  Correct bilateral direct mode uses left shaft as
                # target and deliberately ignores any right-camera target box.
                targets = left_targets if need_bilateral else [item for item in detections if item.class_name == "target"]
                pair_detections = holes + targets
                # The right-only veto path owns the hole-depth window.  Only
                # bilateral direct geometry needs a left target before it can
                # consume a batch for pair geometry and motion safety.
                if can_accumulate_hole_depth(
                    selected_hole=selected_hole is not None,
                    aligned_depth=aligned_depth is not None,
                    target_count=len(targets),
                    geometry_mode=args.geometry_mode,
                ) and depth_timestamp is not None:
                    window_result = hole_depth_accumulator.add(
                        aligned_depth,
                        depth_timestamp,
                        selected_hole.xyxy,
                    )
                    hole_depth_window = window_result.depth_u16
                    hole_depth_window_box = window_result.box_xyxy
                    hole_depth_diagnostics = window_result.diagnostics
                    hole_depth_window_ready = hole_depth_window is not None
                elif selected_hole is None:
                    hole_depth_accumulator.reset(right_lock_reason)
                elif args.geometry_mode == "direct" and len(targets) != 1:
                    hole_depth_accumulator.reset("target_pair_unavailable")
                    hole_depth_diagnostics = {
                        "state": "hold",
                        "window_frames": 0,
                        "reason": "target_pair_unavailable",
                    }
                elif aligned_depth is None or depth_timestamp is None:
                    hole_depth_diagnostics = {
                        "state": "hold",
                        "window_frames": 0,
                        "reason": geometry_reason or "right_depth_unavailable",
                    }
                frame_record["detections"] = [
                    {"camera": "right", "class": item.class_name, "confidence": round(item.confidence, 4), "center_px": [round(value, 1) for value in item.center_px]}
                    for item in holes
                ]
                frame_record["right_hole_lock"] = {
                    "reason": right_lock_reason,
                    "candidate_count": len(all_holes),
                    "locked": selected_hole is not None and not args.disable_right_hole_lock,
                    "mode": "disabled" if args.disable_right_hole_lock else "temporal",
                    "center_px": None if selected_hole is None else [round(value, 1) for value in selected_hole.center_px],
                }
                if right_hole_xyz_preview is not None:
                    frame_record["right_hole_xyz_color_camera_m"] = [
                        round(float(value), 6) for value in right_hole_xyz_preview
                    ]
                    frame_record["right_hole_xyz_source"] = preview_xyz_source
                if right_hole_xyz_end_preview is not None:
                    frame_record["right_hole_xyz_right_end_link_m"] = [
                        round(float(value), 6) for value in right_hole_xyz_end_preview
                    ]
                if right_hole_xyz_base_preview is not None:
                    frame_record["right_hole_xyz_base_link_m"] = [
                        round(float(value), 6) for value in right_hole_xyz_base_preview
                    ]
                frame_record["hole_depth"] = hole_depth_diagnostics
                # Keep JSONL compact: the web overlay receives all raw
                # hinge/shaft boxes, while the frame timeline records only
                # the selected shaft used as the target.
                frame_record["left_detections"] = [
                    {"camera": "left", "class": item.class_name, "confidence": round(item.confidence, 4), "center_px": [round(value, 1) for value in item.center_px]}
                    for item in left_targets
                ] if need_bilateral else []
                if need_bilateral:
                    frame_record["left_association"] = left_association
                if need_bilateral and len(targets) == 1:
                    frame_record["target_tip_px_left"] = [round(float(value), 1) for value in targets[0].tip_px]
                    frame_record["target_source"] = (
                        "left_locked_hinge_nearest_shaft_temporal"
                        if had_left_track_reference
                        else "left_selected_hinge_nearest_shaft_initial"
                    )
                frame_record["pair_status"] = pair_status(
                    pair_detections,
                    {"hole": holes[0], "target": targets[0]} if len(holes) == 1 and len(targets) == 1 else {},
                )
                frame_record["geometry_mode"] = args.geometry_mode
                frame_record["right_depth_registration"] = args.right_depth_registration
                frame_record["hole_depth_region"] = args.hole_depth_region
                frame_record["right_transform_chain"] = "P_base=T_base_arm_r_end(T_arm_r_end_right_rgbd P_right_color)"
                if geometry_reason:
                    frame_record["geometry_reason"] = geometry_reason
                if tf_timestamp is not None:
                    frame_record["right_tf_timestamp_ns"] = tf_timestamp
                if right_tf_info:
                    frame_record.update({
                        "right_tf_query_timestamp_ns": right_tf_info.get("query_timestamp_ns"),
                        "right_tf_age_ms": right_tf_info.get("age_ms"),
                        "right_tf_lookup_mode": right_tf_info.get("mode"),
                    })
                    if right_tf_info.get("exact_lookup_error"):
                        frame_record["right_tf_lookup_error"] = right_tf_info["exact_lookup_error"]
                if left_tf_timestamp is not None:
                    frame_record["left_tf_timestamp_ns"] = left_tf_timestamp
                if need_rgbd and len(holes) != 1 and not geometry_reason:
                    geometry_reason = "require_exactly_one_hole"
                    frame_record["geometry_reason"] = geometry_reason
                elif need_bilateral and len(targets) != 1 and not geometry_reason:
                    geometry_reason = "require_exactly_one_hole_and_target"
                    frame_record["geometry_reason"] = geometry_reason
                pair_geometry: PairGeometry | None = None
                hole_base_m = None
                target_base_m = None
                hole_depth_support = None
                target_depth_support = None
                # Estimate the right hole independently of the left target.
                # This makes the camera-frame XYZ check observable even when
                # the held shaft is temporarily occluded or not detected.
                if need_rgbd and len(holes) == 1:
                    try:
                        if aligned_depth is None or color_intrinsic is None:
                            raise RuntimeError("right hole RGB-D depth grid unavailable")
                        if hole_depth_window is None or hole_depth_window_box is None:
                            raise RuntimeError(hole_depth_diagnostics.get("reason", "hole_depth_collecting"))
                        right_crop_guard = (
                            float(right_crop_offset[0]),
                            float(right_crop_offset[1]),
                            float(right_crop_offset[0] + right_infer_frame_bgr.shape[1] - 1),
                            float(right_crop_offset[1] + right_infer_frame_bgr.shape[0] - 1),
                        )
                        right_hole_depth, adaptive_diagnostics = estimate_hole_depth_adaptive(
                            hole_depth_window,
                            hole_depth_window_box,
                            intrinsic=color_intrinsic,
                            scales=depth_scales,
                            guard_xyxy=right_crop_guard,
                            max_padding_px=args.hole_depth_max_padding_px,
                            min_valid_pixels=args.min_depth_pixels,
                            min_support_fraction=args.min_depth_support,
                            min_covered_sectors=args.hole_depth_min_covered_sectors,
                            max_mad_m=args.max_depth_mad_mm / 1000.0,
                            sampling_region=args.hole_depth_region,
                        )
                        frame_record["hole_depth"].update(adaptive_diagnostics)
                        selected_depth_stats = adaptive_diagnostics.get("selected")
                        if isinstance(selected_depth_stats, dict):
                            frame_record["hole_depth_stats"] = {
                                "valid_count": selected_depth_stats.get("valid_pixels"),
                                "mean_m": selected_depth_stats.get("mean_m"),
                                "median_m": selected_depth_stats.get("median_m", selected_depth_stats.get("depth_m")),
                                "mad_mm": selected_depth_stats.get("mad_mm"),
                                "robust_mean_m": selected_depth_stats.get("robust_mean_m"),
                            }
                        frame_record["right_hole_xyz_color_camera_m"] = [
                            round(float(value), 6) for value in right_hole_depth.point_color_m
                        ]
                        frame_record["right_hole_xyz_source"] = "temporal_hole_depth_window"
                        hole_depth_support = right_hole_depth.support_fraction
                        if rgbd_to_right_end is not None and right_end_to_base is not None:
                            hole_end = apply_transform(rgbd_to_right_end, right_hole_depth.point_color_m)
                            hole_base_m = right_color_point_to_base(
                                right_hole_depth.point_color_m, rgbd_to_right_end, right_end_to_base
                            )
                            frame_record["right_hole_xyz_right_end_link_m"] = [
                                round(float(value), 6) for value in hole_end
                            ]
                            frame_record["right_hole_xyz_base_link_m"] = [
                                round(float(value), 6) for value in hole_base_m
                            ]
                        if need_bilateral and len(targets) == 1:
                            if left_aligned_depth is None or left_color_intrinsic is None or left_end_to_base is None or left_rgbd_to_end is None:
                                raise RuntimeError("left target RGB-D geometry unavailable")
                            left_target_depth = estimate_point_depth(
                                left_aligned_depth,
                                targets[0].tip_px,
                                intrinsic=left_color_intrinsic,
                                radius_px=4,
                                min_valid_pixels=args.min_depth_pixels,
                                min_support_fraction=args.min_depth_support,
                                max_mad_m=args.max_depth_mad_mm / 1000.0,
                            )
                            target_end = apply_transform(left_rgbd_to_end, left_target_depth.point_color_m)
                            target_base_m = apply_transform(left_end_to_base, target_end)
                            target_depth_support = left_target_depth.support_fraction
                        elif not need_bilateral and len(targets) == 1:
                            old_pair_geometry = estimate_pair_geometry(
                                aligned_depth,
                                hole_xyxy=holes[0].xyxy,
                                target_xyxy=targets[0].xyxy,
                                color_intrinsic=color_intrinsic,
                                rgbd_to_right_end=rgbd_to_right_end,
                                right_end_to_base=right_end_to_base,
                                min_valid_pixels=args.min_depth_pixels,
                                min_support_fraction=args.min_depth_support,
                                max_mad_m=args.max_depth_mad_mm / 1000.0,
                            )
                            pair_geometry = old_pair_geometry
                            target_base_m = old_pair_geometry.target_base_m
                            target_depth_support = old_pair_geometry.target.support_fraction
                        frame_record.update({
                            "hole_depth_support": hole_depth_support,
                        })
                        if hole_base_m is not None:
                            frame_record["hole_base_m"] = list(hole_base_m)
                        if target_base_m is not None:
                            frame_record["target_base_m"] = list(target_base_m)
                        if target_depth_support is not None:
                            frame_record["target_depth_support"] = target_depth_support
                    except AdaptiveDepthError as exc:
                        frame_record["hole_depth"].update(exc.diagnostics)
                        frame_record["geometry_reason"] = str(exc)
                    except Exception as exc:
                        frame_record["geometry_reason"] = str(exc)
                if hole_depth_window_ready:
                    # Consume each non-overlapping temporal batch only after
                    # this frame had a paired geometry attempt.
                    hole_depth_accumulator.reset("window_consumed")

                decision = None
                metric_decision = None
                pixel_decision = None
                if stream is not None and time.monotonic() < step_settle_until:
                    frame_record.update({"state": "step_settling", "reason": "waiting_for_left_target_arrival_before_consuming_new_pair"})
                elif args.geometry_mode == "off":
                    if pixel_gate is None:
                        frame_record.update({"state": "calibration_required", "reason": "provide_right_camera_pixel_jacobian_for_motion"})
                    else:
                        can_move = bool(stream is not None and stream.is_target_reached() and not aligned)
                        pixel_decision = pixel_gate.update(
                            sequence=sequence,
                            hole_center_px=None if len(holes) != 1 else holes[0].center_px,
                            target_center_px=None if len(targets) != 1 else targets[0].center_px,
                            hole_confidence=holes[0].confidence if len(holes) == 1 else 0.0,
                            target_confidence=targets[0].confidence if len(targets) == 1 else 0.0,
                            hole_count=len(holes), target_count=len(targets), can_move=can_move,
                        )
                        decision = pixel_decision
                        frame_record.update({
                            "state": decision.state, "reason": decision.reason,
                            "pixel_error_px": decision.pixel_error_px,
                            "requested_delta_m": list(decision.requested_delta_m),
                            "accepted_delta_m": list(decision.accepted_delta_m),
                            "cumulative_offset_m": list(decision.cumulative_offset_m),
                            "cumulative_distance_m": decision.cumulative_distance_m,
                            "step_index": decision.step_index,
                        })
                elif args.geometry_mode == "direct":
                    if metric_gate is None:
                        frame_record.update({"state": "geometry_hold", "reason": "metric_gate_not_initialized"})
                    elif hole_base_m is None or target_base_m is None:
                        metric_gate.reset_observation()
                        frame_record.update({"state": "geometry_hold", "reason": frame_record.get("geometry_reason", "require_one_pair_with_valid_rgbd_depth")})
                    else:
                        can_move = bool(stream is not None and stream.is_target_reached() and not aligned)
                        metric_decision = metric_gate.update(
                            sequence=sequence,
                            hole_position_m=hole_base_m,
                            target_position_m=target_base_m,
                            hole_confidence=holes[0].confidence,
                            target_confidence=targets[0].confidence,
                            can_move=can_move,
                        )
                        decision = metric_decision
                        frame_record.update({
                            "state": metric_decision.state, "reason": metric_decision.reason,
                            "geometry_error_m": metric_decision.error_m,
                            "requested_delta_m": list(metric_decision.requested_delta_m),
                            "accepted_delta_m": list(metric_decision.accepted_delta_m),
                            "cumulative_offset_m": list(metric_decision.cumulative_offset_m),
                            "cumulative_distance_m": metric_decision.cumulative_distance_m,
                            "step_index": metric_decision.step_index,
                            "hole_base_m": list(hole_base_m),
                            "target_base_m": list(target_base_m),
                        })
                else:  # RGB-D veto around the existing pixel-J controller.
                    if pixel_gate is None:
                        frame_record.update({"state": "calibration_required", "reason": "veto_motion_requires_pixel_jacobian"})
                    elif metric_gate is None or pair_geometry is None:
                        pixel_gate.reset_observation()
                        if metric_gate is not None:
                            metric_gate.reset_observation()
                        frame_record.update({"state": "geometry_hold", "reason": frame_record.get("geometry_reason", "require_one_pair_with_valid_rgbd_depth")})
                    else:
                        can_move = bool(stream is not None and stream.is_target_reached() and not aligned)
                        metric_decision = metric_gate.update(
                            sequence=sequence,
                            hole_position_m=pair_geometry.hole_base_m,
                            target_position_m=pair_geometry.target_base_m,
                            hole_confidence=holes[0].confidence,
                            target_confidence=targets[0].confidence,
                            can_move=False,
                        )
                        pixel_decision = pixel_gate.update(
                            sequence=sequence,
                            hole_center_px=holes[0].center_px,
                            target_center_px=targets[0].center_px,
                            hole_confidence=holes[0].confidence,
                            target_confidence=targets[0].confidence,
                            hole_count=1,
                            target_count=1,
                            can_move=can_move,
                        )
                        decision = pixel_decision
                        frame_record.update({
                            "state": pixel_decision.state, "reason": pixel_decision.reason,
                            "pixel_error_px": pixel_decision.pixel_error_px,
                            "metric_error_m": metric_decision.error_m,
                            "metric_requested_delta_m": list(metric_decision.requested_delta_m),
                            "hole_base_m": list(pair_geometry.hole_base_m),
                            "target_base_m": list(pair_geometry.target_base_m),
                        })
                        if pixel_decision.state == "move":
                            agrees = metric_decision.state in {"ready", "waiting_alignment", "aligned"} and metric_and_pixel_corrections_agree(pixel_decision.requested_delta_m, metric_decision.requested_delta_m)
                            frame_record["geometry_agrees"] = agrees
                            if not agrees:
                                pixel_gate.reset_observation()
                                metric_gate.reset_observation()
                                frame_record.update({"state": "geometry_disagreement_hold", "reason": "pixel_and_official_rgbd_corrections_disagree"})
                                decision = None
                        elif pixel_decision.state == "aligned" and metric_decision.state != "aligned":
                            frame_record.update({"state": "geometry_wait", "reason": "pixel_aligned_but_metric_alignment_not_confirmed"})
                            decision = None

                if decision is not None and decision.state == "move" and stream is not None and initial_pose is not None:
                    target_pose = left_target_from_offset(initial_pose, decision.cumulative_offset_m)
                    stream.update_target(target_pose)
                    if args.geometry_mode == "off":
                        pixel_gate.reset_observation()
                    elif args.geometry_mode == "direct":
                        metric_gate.reset_observation()
                    else:
                        pixel_gate.reset_observation()
                        metric_gate.reset_observation()
                    hole_depth_accumulator.reset("left_step_started")
                    step_settle_until = time.monotonic() + args.step_settle_s
                    frame_record["command_source"] = "left_mode1_end_effector_position_stream"
                elif decision is not None and decision.state == "aligned":
                    aligned = True

                right_overlay = draw_alignment_overlay(
                    frame_bgr,
                    detections,
                    {},
                    frame_record.get("state", last_state),
                    frame_record.get("pixel_error_px"),
                    frame_record.get("geometry_error_m"),
                )
                crop_x0, crop_y0 = right_crop_offset
                crop_x1 = crop_x0 + int(right_infer_frame_bgr.shape[1]) - 1
                crop_y1 = crop_y0 + int(right_infer_frame_bgr.shape[0]) - 1
                cv2.rectangle(right_overlay, (crop_x0, crop_y0), (crop_x1, crop_y1), (0, 165, 255), 3, cv2.LINE_AA)
                cv2.putText(
                    right_overlay,
                    f"INFERENCE ROI {args.right_crop_percent:.0f}%",
                    (crop_x0 + 8, max(24, crop_y0 + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                if selected_hole is not None:
                    lock_x = int(round(selected_hole.center_px[0]))
                    lock_y = int(round(selected_hole.center_px[1]))
                    cv2.circle(right_overlay, (lock_x, lock_y), 10, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(
                        right_overlay,
                        "LOCKED HOLE",
                        (max(8, lock_x - 55), max(24, lock_y - 14)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                if need_bilateral and left_frame_bgr is not None:
                    left_display_detections = list(left_overlay_detections)
                    if left_targets:
                        left_display_detections.append(left_targets[0].box)
                    left_overlay = draw_alignment_overlay(
                        left_frame_bgr,
                        left_display_detections,
                        {},
                        "left target",
                        None,
                        None,
                    )
                    if left_targets and left_targets[0].hinge_center_px is not None:
                        hinge_px = tuple(int(round(value)) for value in left_targets[0].hinge_center_px)
                        tip_px = tuple(int(round(value)) for value in left_targets[0].tip_px)
                        cv2.circle(left_overlay, hinge_px, 9, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.line(left_overlay, hinge_px, tip_px, (0, 255, 0), 2, cv2.LINE_AA)
                        cv2.putText(
                            left_overlay,
                            "LOCKED HINGE -> SHAFT",
                            (max(8, hinge_px[0] - 90), max(24, hinge_px[1] - 14)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.58,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA,
                        )
                    if left_overlay.shape[0] != right_overlay.shape[0]:
                        scale = right_overlay.shape[0] / float(left_overlay.shape[0])
                        left_overlay = cv2.resize(left_overlay, (int(left_overlay.shape[1] * scale), right_overlay.shape[0]))
                    # Display order follows the operator's view: left hand first,
                    # right hand second.  Detection/TF/control frames are unchanged.
                    overlay = np.hstack((left_overlay, right_overlay))
                else:
                    overlay = right_overlay
                ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if shared is not None and ok:
                    shared.update(encoded.tobytes(), {
                        "state": frame_record.get("state", last_state), "sequence": sequence,
                        "camera": "right_hand_color + left_hand_color" if need_bilateral else "right_hand_color",
                        "detections": frame_record.get("detections", []),
                        "right_hole_lock": frame_record.get("right_hole_lock"),
                        "hole_depth": frame_record.get("hole_depth"),
                        "left_detections": [
                            {"camera": "left", "class": item.class_name, "confidence": round(item.confidence, 4), "center_px": [round(value, 1) for value in item.center_px]}
                            for item in left_overlay_detections
                        ] if need_bilateral else [],
                        "left_association": frame_record.get("left_association", {}),
                        "pair_status": frame_record.get("pair_status"), "pixel_error_px": frame_record.get("pixel_error_px"),
                        "geometry_mode": args.geometry_mode,
                        "geometry_reason": frame_record.get("geometry_reason"),
                        "geometry_error_m": frame_record.get("geometry_error_m"),
                        "hole_base_m": frame_record.get("hole_base_m"), "target_base_m": frame_record.get("target_base_m"),
                        "right_hole_xyz_color_camera_m": frame_record.get("right_hole_xyz_color_camera_m"),
                        "right_hole_xyz_right_end_link_m": frame_record.get("right_hole_xyz_right_end_link_m"),
                        "right_hole_xyz_base_link_m": frame_record.get("right_hole_xyz_base_link_m"),
                        "right_hole_xyz_source": frame_record.get("right_hole_xyz_source"),
                        "right_tf_timestamp_ns": frame_record.get("right_tf_timestamp_ns"),
                        "right_tf_lookup_mode": frame_record.get("right_tf_lookup_mode"),
                        "right_tf_age_ms": frame_record.get("right_tf_age_ms"),
                        "right_tf_lookup_error": frame_record.get("right_tf_lookup_error"),
                        "geometry_agrees": frame_record.get("geometry_agrees"),
                        "accepted_delta_m": frame_record.get("accepted_delta_m"),
                        "cumulative_offset_m": frame_record.get("cumulative_offset_m"),
                        "right_fixed": True, "motion": "left_xy_only" if args.allow_motion else "read_only_preview",
                    })
                log_record(log, frame_record)
                terminal_line_active = emit_terminal_record(frame_record, terminal_output_mode) or terminal_line_active
                last_state = frame_record.get("state", last_state)
                if aligned:
                    success_basis = "官方 RGB-D 三维容差" if args.geometry_mode == "direct" else "像素与官方 RGB-D 双重容差" if args.geometry_mode == "veto" else "像素容差"
                    if terminal_line_active:
                        print(flush=True)
                        terminal_line_active = False
                    print(f"双臂定位完成：右臂保持观察位，左臂已满足{success_basis}；未下探、未夹爪、未插入。")
                    break
        return 1 if fatal_fault else 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"双臂定位伺服失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if terminal_line_active:
            print(flush=True)
        if stream is not None:
            stream.stop()
        if server is not None:
            server.shutdown()
            server.server_close()
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
