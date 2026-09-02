"""YOLO-seg helpers for the left-D405 hinge detector.

The module intentionally has no GDK, camera, or robot-control import.  The
caller supplies a BGR image and applies GDK depth/TF geometry to the returned
full-image masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def load_yolo_segmentation_model(model_path: str | Path, *, device: str) -> Any:
    """Load local YOLO-seg weights lazily.

    Keeping the Ultralytics import here lets the established SAM3-only tools
    continue to run without that optional dependency.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "缺少 ultralytics；请在 .venv-sam3 安装后再运行 YOLO 复核网页。"
        ) from exc
    path = Path(model_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到左 D405 YOLO 权重：{path}")
    model = YOLO(str(path))
    model.to(device)
    return model


def infer_yolo_masks(
    model: Any,
    image_bgr: np.ndarray,
    *,
    confidence: float,
    imgsz: int,
    device: str,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Return masks, confidences, and the matching class IDs.

    The old left model had one class, which made it safe for callers to
    ignore ``boxes.cls``.  The current model is two-class (hinge/shaft), so
    every consumer must retain the class ID instead of treating all masks as
    graspable hinges.
    """
    results = model.predict(
        source=image_bgr,
        imgsz=int(imgsz),
        conf=float(confidence),
        device=device,
        verbose=False,
        retina_masks=True,
        max_det=10,
    )
    if not results:
        return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    result = results[0]
    if result.masks is None or result.boxes is None:
        return [], np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)
    masks = [np.asarray(mask, dtype=bool).squeeze() for mask in result.masks.data.detach().cpu().numpy()]
    scores = result.boxes.conf.detach().float().cpu().numpy().reshape(-1)
    class_ids = result.boxes.cls.detach().int().cpu().numpy().reshape(-1)
    if not (len(masks) == len(scores) == len(class_ids)):
        raise RuntimeError(
            "YOLO 掩膜、置信度和类别数量不一致："
            f"{len(masks)} != {len(scores)} != {len(class_ids)}"
        )
    expected_shape = tuple(int(value) for value in image_bgr.shape[:2])
    for index, mask in enumerate(masks):
        if mask.shape != expected_shape:
            raise RuntimeError(f"YOLO 掩膜 #{index} 尺寸 {mask.shape} 不等于输入图像 {expected_shape}")
    return masks, scores, class_ids


def split_yolo_candidates(
    statistics: list[dict],
    geometry_indices: list[int],
    *,
    confidence_threshold: float,
    hinge_class_id: int = 0,
    shaft_class_id: int = 1,
) -> tuple[list[int], list[int]]:
    """Split eligible detections into graspable hinges and auxiliary shafts.

    The returned indices preserve the detector order.  Unknown classes and
    low-confidence detections are deliberately excluded from both lists.
    This keeps class policy in one dependency-free function shared by the
    motion state machine and the read-only review page.
    """
    if not 0.0 <= float(confidence_threshold) <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    hinges: list[int] = []
    shafts: list[int] = []
    for raw_index in geometry_indices:
        index = int(raw_index)
        if index < 0 or index >= len(statistics):
            continue
        item = statistics[index]
        try:
            class_id = int(item.get("yolo_class_id", -1))
            confidence = float(item.get("yolo_confidence", 0.0))
        except (TypeError, ValueError):
            continue
        if not np.isfinite(confidence) or confidence < float(confidence_threshold):
            continue
        if class_id == int(hinge_class_id):
            hinges.append(index)
        elif class_id == int(shaft_class_id):
            shafts.append(index)
    return hinges, shafts


def associate_shaft_to_hinge(
    statistics: list[dict],
    hinge_index: int | None,
    *,
    shaft_indices: list[int] | None = None,
    shaft_class_id: int = 1,
    max_center_distance_px: float = 320.0,
    max_horizontal_distance_px: float = 140.0,
    ambiguity_margin_px: float = 20.0,
) -> dict:
    """Associate one shaft detection with the selected hinge only.

    This is deliberately a conservative image-space gate.  The shaft is
    expected to stay near the selected hinge's image column; the horizontal
    gate prevents a shaft belonging to a neighbouring hinge from winning just
    because it has a smaller Euclidean distance.  It never picks a shaft by
    itself: without a selected hinge, or when two shafts are nearly equally
    close, the result is rejected and the motion state machine can continue
    using the hinge-only grasp target.
    """
    if hinge_index is None:
        return {"shaft_index": None, "reason": "no_selected_hinge"}
    if hinge_index < 0 or hinge_index >= len(statistics):
        return {"shaft_index": None, "reason": "selected_hinge_index_invalid"}
    if not (
        np.isfinite(max_center_distance_px)
        and max_center_distance_px > 0
        and np.isfinite(max_horizontal_distance_px)
        and max_horizontal_distance_px > 0
        and np.isfinite(ambiguity_margin_px)
        and ambiguity_margin_px >= 0
    ):
        return {"shaft_index": None, "reason": "association_parameters_invalid"}
    hinge = statistics[hinge_index]
    hinge_center = np.asarray(hinge.get("center_px", ()), dtype=np.float64)
    if hinge_center.shape != (2,) or not np.all(np.isfinite(hinge_center)):
        return {"shaft_index": None, "reason": "selected_hinge_center_invalid"}
    candidates: list[tuple[float, int]] = []
    allowed_indices = None if shaft_indices is None else {int(index) for index in shaft_indices}
    for index, item in enumerate(statistics):
        try:
            class_id = int(item.get("yolo_class_id", -1))
        except (TypeError, ValueError):
            continue
        if index == hinge_index or class_id != shaft_class_id:
            continue
        if allowed_indices is not None and index not in allowed_indices:
            continue
        center = np.asarray(item.get("center_px", ()), dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            continue
        delta = center - hinge_center
        distance = float(np.linalg.norm(delta))
        if (
            np.isfinite(distance)
            and distance <= max_center_distance_px
            and abs(float(delta[0])) <= max_horizontal_distance_px
        ):
            candidates.append((distance, index))
    if not candidates:
        return {
            "shaft_index": None,
            "reason": "no_nearby_shaft",
            "candidate_count": 0,
            "max_center_distance_px": float(max_center_distance_px),
            "max_horizontal_distance_px": float(max_horizontal_distance_px),
        }
    candidates.sort()
    best_distance, best_index = candidates[0]
    if len(candidates) > 1 and candidates[1][0] - best_distance < ambiguity_margin_px:
        return {
            "shaft_index": None,
            "reason": "ambiguous_nearby_shafts",
            "candidate_count": len(candidates),
            "distances_px": [round(distance, 2) for distance, _ in candidates[:3]],
        }
    best_center = np.asarray(statistics[best_index]["center_px"], dtype=np.float64)
    best_delta = best_center - hinge_center
    return {
        "shaft_index": int(best_index),
        "reason": "matched_nearest_shaft",
        "distance_px": round(best_distance, 2),
        "horizontal_offset_px": round(float(best_delta[0]), 2),
        "vertical_offset_px": round(float(best_delta[1]), 2),
        "candidate_count": len(candidates),
        "max_center_distance_px": float(max_center_distance_px),
        "max_horizontal_distance_px": float(max_horizontal_distance_px),
    }


def resize_masks_to_full_image(
    masks: list[np.ndarray], *, full_shape: tuple[int, int]
) -> list[np.ndarray]:
    """Map masks from a resized YOLO input back to full-image pixel coordinates."""
    import cv2

    height, width = int(full_shape[0]), int(full_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("full_shape dimensions must be positive")
    expanded: list[np.ndarray] = []
    for mask in masks:
        source = np.asarray(mask, dtype=np.uint8).squeeze()
        if source.ndim != 2:
            raise ValueError(f"YOLO mask must be 2D, got shape {source.shape}")
        expanded.append(cv2.resize(source, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool))
    return expanded


def review_matches(
    yolo_selected: dict | None,
    sam_selected: dict | None,
    *,
    max_pixel_error: float,
    max_depth_error_m: float,
    allow_missing_depth: bool = False,
) -> tuple[bool, dict]:
    """Require YOLO and SAM3 to identify the same full-image target."""
    if yolo_selected is None:
        return False, {"reason": "no_eligible_yolo_target"}
    if sam_selected is None:
        return False, {"reason": "no_eligible_sam_target"}
    yolo_center = np.asarray(yolo_selected["center_px"], dtype=np.float64)
    sam_center = np.asarray(sam_selected["center_px"], dtype=np.float64)
    pixel_error = float(np.linalg.norm(yolo_center - sam_center))
    yolo_depth = yolo_selected.get("depth_median_m")
    sam_depth = sam_selected.get("depth_median_m")
    if yolo_depth is None or sam_depth is None:
        if allow_missing_depth:
            accepted = pixel_error <= float(max_pixel_error)
            return accepted, {
                "reason": "pixel_only_match_missing_depth" if accepted else "selected_target_pixel_disagreement",
                "pixel_error": pixel_error,
                "depth_error_m": None,
                "max_pixel_error": float(max_pixel_error),
                "max_depth_error_m": float(max_depth_error_m),
            }
        return False, {
            "reason": "missing_depth_for_review",
            "pixel_error": pixel_error,
            "max_pixel_error": float(max_pixel_error),
            "max_depth_error_m": float(max_depth_error_m),
        }
    depth_error = abs(float(yolo_depth) - float(sam_depth))
    accepted = pixel_error <= float(max_pixel_error) and depth_error <= float(max_depth_error_m)
    return accepted, {
        "reason": "matched" if accepted else "selected_target_disagreement",
        "pixel_error": pixel_error,
        "depth_error_m": depth_error,
        "max_pixel_error": float(max_pixel_error),
        "max_depth_error_m": float(max_depth_error_m),
    }
