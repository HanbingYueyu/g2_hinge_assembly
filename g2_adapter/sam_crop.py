"""Center-crop helpers for left D405 SAM3 input.

The model sees only the central 70% of the colour image.  Masks are expanded
back into the original colour-image coordinate system before depth geometry or
visual servo code consumes them.
"""

from __future__ import annotations

import numpy as np


SAM_CENTER_CROP_FRACTION = 0.70


def center_crop_bounds(image_shape: tuple[int, ...], *, fraction: float = SAM_CENTER_CROP_FRACTION) -> tuple[int, int, int, int]:
    """Return ``(x0, y0, x1, y1)`` for a centered crop of an image."""
    if len(image_shape) < 2:
        raise ValueError("image_shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("crop fraction must be in (0, 1]")
    crop_width = max(1, min(width, int(round(width * fraction))))
    crop_height = max(1, min(height, int(round(height * fraction))))
    x0 = (width - crop_width) // 2
    y0 = (height - crop_height) // 2
    return x0, y0, x0 + crop_width, y0 + crop_height


def center_crop_image(image: np.ndarray, *, fraction: float = SAM_CENTER_CROP_FRACTION) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return the central SAM input crop and its bounds in the source image."""
    bounds = center_crop_bounds(image.shape, fraction=fraction)
    x0, y0, x1, y1 = bounds
    return np.asarray(image)[y0:y1, x0:x1].copy(), bounds


def expand_crop_masks(
    crop_masks: list[np.ndarray],
    *,
    full_shape: tuple[int, int],
    bounds: tuple[int, int, int, int],
) -> list[np.ndarray]:
    """Map SAM masks from crop pixels back into full-image pixels."""
    height, width = (int(full_shape[0]), int(full_shape[1]))
    x0, y0, x1, y1 = bounds
    crop_shape = (y1 - y0, x1 - x0)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("crop bounds are outside full_shape")
    expanded: list[np.ndarray] = []
    for mask in crop_masks:
        crop_mask = np.asarray(mask, dtype=bool).squeeze()
        if crop_mask.shape != crop_shape:
            raise ValueError(f"crop mask shape {crop_mask.shape} does not match crop shape {crop_shape}")
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y0:y1, x0:x1] = crop_mask
        expanded.append(full_mask)
    return expanded


def draw_crop_box(image_bgr: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Draw the SAM input boundary over a full-frame BGR preview."""
    import cv2

    x0, y0, x1, y1 = bounds
    output = np.asarray(image_bgr).copy()
    cv2.rectangle(output, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
    cv2.putText(output, "SAM center crop 70%", (x0 + 8, max(y0 + 25, 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return output
