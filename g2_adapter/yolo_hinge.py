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
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return full-resolution boolean masks and the matching confidences."""
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
        return [], np.empty((0,), dtype=np.float32)
    result = results[0]
    if result.masks is None or result.boxes is None:
        return [], np.empty((0,), dtype=np.float32)
    masks = [np.asarray(mask, dtype=bool).squeeze() for mask in result.masks.data.detach().cpu().numpy()]
    scores = result.boxes.conf.detach().float().cpu().numpy().reshape(-1)
    if len(masks) != len(scores):
        raise RuntimeError(f"YOLO 掩膜和置信度数量不一致：{len(masks)} != {len(scores)}")
    expected_shape = tuple(int(value) for value in image_bgr.shape[:2])
    for index, mask in enumerate(masks):
        if mask.shape != expected_shape:
            raise RuntimeError(f"YOLO 掩膜 #{index} 尺寸 {mask.shape} 不等于输入图像 {expected_shape}")
    return masks, scores


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
