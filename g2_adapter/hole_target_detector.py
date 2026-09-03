"""Small, dependency-light helpers for the post-P4 hole/target detector.

The trained model is a two-class *box* detector.  This module deliberately
keeps detection separate from robot control: a missing box is never replaced
with a held/old box and no function here produces a motion command.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


CLASS_NAMES = {0: "hole", 1: "target"}


@dataclass(frozen=True)
class BoxDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center_px(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.xyxy
        return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, CLASS_NAMES.get(class_id, class_id)))
    if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
        return str(names[class_id])
    return CLASS_NAMES.get(class_id, str(class_id))


def extract_detections(result: Any) -> list[BoxDetection]:
    """Convert one Ultralytics result into plain immutable detections."""
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.detach().cpu().tolist()
    classes = boxes.cls.detach().cpu().int().tolist()
    confidences = boxes.conf.detach().cpu().tolist()
    names = getattr(result, "names", CLASS_NAMES)
    detections: list[BoxDetection] = []
    for coordinates, class_id, confidence in zip(xyxy, classes, confidences):
        values = tuple(float(value) for value in coordinates)
        if len(values) != 4:
            continue
        detections.append(
            BoxDetection(
                class_id=int(class_id),
                class_name=_class_name(names, int(class_id)),
                confidence=float(confidence),
                xyxy=values,
            )
        )
    return detections


def best_per_class(detections: list[BoxDetection]) -> dict[str, BoxDetection]:
    """Return the highest-confidence box for each known class.

    This is suitable for a preview overlay only.  The future alignment stage
    must add a workcell-specific identity/ROI gate before using a box for
    motion, because a confidence ranking alone cannot resolve multiple holes.
    """
    selected: dict[str, BoxDetection] = {}
    for item in detections:
        if item.class_name not in {"hole", "target"}:
            continue
        previous = selected.get(item.class_name)
        if previous is None or item.confidence > previous.confidence:
            selected[item.class_name] = item
    return selected


def pair_status(detections: list[BoxDetection], selected: dict[str, BoxDetection]) -> str:
    """Describe whether a same-frame hole/target pair is available."""
    counts = {name: sum(item.class_name == name for item in detections) for name in ("hole", "target")}
    if counts["hole"] == 0 or counts["target"] == 0:
        missing = "hole" if counts["hole"] == 0 else "target"
        return f"missing_{missing}"
    if counts["hole"] > 1 or counts["target"] > 1:
        return "multiple_candidates_preview_only"
    if set(selected) == {"hole", "target"}:
        return "paired"
    return "unusable"


def center_crop(frame: Any, percent: float = 60.0) -> tuple[Any, tuple[int, int]]:
    """Return a centered crop and its ``(x, y)`` offset in the source frame."""
    value = float(percent)
    if not math.isfinite(value) or not 0.0 < value <= 100.0:
        raise ValueError("center crop percent must be in (0, 100]")
    shape = getattr(frame, "shape", None)
    if shape is None or len(shape) < 2:
        raise ValueError("frame must have height and width")
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("frame must have positive height and width")
    crop_width = max(1, min(width, int(round(width * value / 100.0))))
    crop_height = max(1, min(height, int(round(height * value / 100.0))))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    return frame[y0 : y0 + crop_height, x0 : x0 + crop_width], (x0, y0)


def translate_detection(item: BoxDetection, offset: tuple[int, int]) -> BoxDetection:
    """Map a detection from a cropped image back to source-frame pixels."""
    x_offset, y_offset = offset
    x0, y0, x1, y1 = item.xyxy
    return BoxDetection(
        class_id=item.class_id,
        class_name=item.class_name,
        confidence=item.confidence,
        xyxy=(x0 + x_offset, y0 + y_offset, x1 + x_offset, y1 + y_offset),
    )


def select_locked_hole(
    holes: list[BoxDetection],
    previous_center_px: tuple[float, float] | None,
    *,
    max_jump_px: float = 120.0,
) -> tuple[BoxDetection | None, str]:
    """Select one hole and keep its identity across frames.

    The first usable frame selects the highest-confidence candidate.  Once a
    lock exists, only a candidate within ``max_jump_px`` of the previous
    center can replace it; a missing or jumping detection returns ``None`` so
    callers hold safely instead of using a stale box.
    """
    if not math.isfinite(float(max_jump_px)) or float(max_jump_px) <= 0.0:
        raise ValueError("max hole lock jump must be positive and finite")
    if not holes:
        return None, "missing_hole"
    if previous_center_px is None:
        return max(holes, key=lambda item: float(item.confidence)), "initial_confidence_lock"
    previous = (float(previous_center_px[0]), float(previous_center_px[1]))
    if not all(math.isfinite(value) for value in previous):
        return max(holes, key=lambda item: float(item.confidence)), "initial_confidence_lock"
    nearby = [
        item
        for item in holes
        if math.hypot(item.center_px[0] - previous[0], item.center_px[1] - previous[1]) <= float(max_jump_px)
    ]
    if not nearby:
        return None, "locked_hole_jump_exceeds_limit"
    selected = min(
        nearby,
        key=lambda item: (
            math.hypot(item.center_px[0] - previous[0], item.center_px[1] - previous[1]),
            -float(item.confidence),
        ),
    )
    return selected, "temporal_lock"


def load_detection_model(path: str, *, device: str) -> Any:
    """Load the external YOLO detector without importing Ultralytics at import time."""
    from ultralytics import YOLO

    model = YOLO(path)
    # Move once at startup when supported; predict() still receives device so
    # the code works with both the upstream and the project's Ultralytics.
    if hasattr(model, "to"):
        model.to(device)
    return model


def infer_detection(model: Any, frame_bgr: Any, *, confidence: float, iou: float, imgsz: int, device: str) -> list[BoxDetection]:
    results = model.predict(
        source=frame_bgr,
        conf=float(confidence),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device,
        verbose=False,
    )
    if not results:
        return []
    return extract_detections(results[0])
