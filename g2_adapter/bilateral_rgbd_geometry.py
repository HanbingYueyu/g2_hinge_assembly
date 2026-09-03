"""Pure geometry helpers for the right-camera bilateral workflow.

The functions in this module intentionally do not import GDK or a detector.
They make the frame convention explicit: ``T_A_B`` maps a point expressed in
frame B into frame A.  Keeping this code pure makes the official right RGB-D
calibration chain testable without connecting to a robot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DepthEstimate:
    """Robust depth estimate for one detector box."""

    center_px: tuple[float, float]
    depth_m: float
    valid_pixels: int
    support_fraction: float
    mad_m: float
    point_color_m: tuple[float, float, float]
    mean_m: float
    robust_mean_m: float


@dataclass(frozen=True)
class PairGeometry:
    """The two detected points expressed in ``base_link``."""

    hole: DepthEstimate
    target: DepthEstimate
    hole_base_m: tuple[float, float, float]
    target_base_m: tuple[float, float, float]

    @property
    def error_target_minus_hole_m(self) -> tuple[float, float, float]:
        return tuple(
            float(target - hole)
            for target, hole in zip(self.target_base_m, self.hole_base_m)
        )

    @property
    def correction_hole_minus_target_xy_m(self) -> tuple[float, float, float]:
        error = self.error_target_minus_hole_m
        return (-error[0], -error[1], 0.0)


class AdaptiveDepthError(ValueError):
    """Raised when no safe adaptive hole-depth sample is available."""

    def __init__(self, message: str, diagnostics: dict) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def _as_vector(value: Sequence[float], size: int, name: str) -> np.ndarray:
    try:
        vector = np.asarray(tuple(float(item) for item in value), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {size} finite values") from exc
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _rotation_from_xyzw(quaternion: Sequence[float]) -> np.ndarray:
    x, y, z, w = _as_vector(quaternion, 4, "rotation_xyzw")
    norm = float(np.linalg.norm((x, y, z, w)))
    if norm <= 1e-12:
        raise ValueError("rotation_xyzw must not be zero")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float64,
    )


def _transform_parts(transform: Mapping[str, Sequence[float]]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(transform, Mapping):
        raise ValueError("transform must be a mapping")
    return _rotation_from_xyzw(transform["rotation_xyzw"]), _as_vector(
        transform["translation_m"], 3, "translation_m"
    )


def apply_transform(transform: Mapping[str, Sequence[float]], point_source_m: Sequence[float]) -> tuple[float, float, float]:
    """Apply ``T_A_B`` to a point in frame B and return a point in frame A."""
    rotation, translation = _transform_parts(transform)
    point = _as_vector(point_source_m, 3, "point_source_m")
    result = rotation @ point + translation
    return tuple(float(value) for value in result)


def right_color_point_to_base(
    point_right_color_m: Sequence[float],
    right_end_from_color: Mapping[str, Sequence[float]],
    base_from_right_end: Mapping[str, Sequence[float]],
) -> tuple[float, float, float]:
    """Map a point from the right RGB-D/color optical frame to ``base_link``.

    The argument names deliberately encode the frame direction.  The first
    transform is GDK's ``kRightHandRGBDToArmREndLink`` (camera ->
    ``arm_r_end_link``); the second is the dynamic GDK TF lookup
    ``lookup_transform("base_link", "arm_r_end_link", ...)`` (end link ->
    ``base_link``).  No URDF gripper-camera link belongs in this chain.
    """
    point_right_end_m = apply_transform(right_end_from_color, point_right_color_m)
    return apply_transform(base_from_right_end, point_right_end_m)


def compose_transform(
    t_a_b: Mapping[str, Sequence[float]],
    t_b_c: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    """Compose ``T_A_B @ T_B_C`` into ``T_A_C``."""
    rotation_a_b, translation_a_b = _transform_parts(t_a_b)
    rotation_b_c, translation_b_c = _transform_parts(t_b_c)
    rotation = rotation_a_b @ rotation_b_c
    translation = rotation_a_b @ translation_b_c + translation_a_b
    # Convert the matrix back to a quaternion only for a serialisable result.
    # The point application path below never depends on this conversion.
    quaternion = _quaternion_from_rotation(rotation)
    return {
        "translation_m": [float(value) for value in translation],
        "rotation_xyzw": [float(value) for value in quaternion],
    }


def inverse_transform(transform: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Return the inverse of ``T_A_B`` (that is, ``T_B_A``)."""
    rotation, translation = _transform_parts(transform)
    inverse_rotation = rotation.T
    inverse_translation = -(inverse_rotation @ translation)
    return {
        "translation_m": [float(value) for value in inverse_translation],
        "rotation_xyzw": [float(value) for value in _quaternion_from_rotation(inverse_rotation)],
    }


def _quaternion_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    else:
        diagonal = np.diag(matrix)
        index = int(np.argmax(diagonal))
        if index == 0:
            s = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 1e-15)) * 2.0
            x = 0.25 * s
            y = (matrix[0, 1] + matrix[1, 0]) / s
            z = (matrix[0, 2] + matrix[2, 0]) / s
            w = (matrix[2, 1] - matrix[1, 2]) / s
        elif index == 1:
            s = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 1e-15)) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / s
            y = 0.25 * s
            z = (matrix[1, 2] + matrix[2, 1]) / s
            w = (matrix[0, 2] - matrix[2, 0]) / s
        else:
            s = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 1e-15)) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / s
            y = (matrix[1, 2] + matrix[2, 1]) / s
            z = 0.25 * s
            w = (matrix[1, 0] - matrix[0, 1]) / s
    result = np.asarray((x, y, z, w), dtype=np.float64)
    result /= np.linalg.norm(result)
    return tuple(float(value) for value in result)


def backproject_pixel(
    center_px: Sequence[float],
    depth_m: float,
    intrinsic: Mapping[str, float],
) -> tuple[float, float, float]:
    """Backproject a color optical pixel using its measured Z depth."""
    u, v = _as_vector(center_px, 2, "center_px")
    z = float(depth_m)
    if not math.isfinite(z) or z <= 0.0:
        raise ValueError("depth_m must be positive and finite")
    fx, fy = float(intrinsic["fx"]), float(intrinsic["fy"])
    ppx, ppy = float(intrinsic["ppx"]), float(intrinsic["ppy"])
    if not all(math.isfinite(value) and value > 0.0 for value in (fx, fy)):
        raise ValueError("camera focal lengths must be positive and finite")
    # GDK exposes the OpenCV/RealSense Brown-Conrady coefficients alongside
    # ``fx, fy, cx, cy``.  Detector pixels are in the (distorted) image, so
    # undo that distortion before applying the pinhole back-projection.
    distorted_x = (u - ppx) / fx
    distorted_y = (v - ppy) / fy
    coefficients = intrinsic.get("distortion", ())
    try:
        coefficients = tuple(float(value) for value in coefficients)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera distortion must contain finite values") from exc
    if coefficients and not all(math.isfinite(value) for value in coefficients):
        raise ValueError("camera distortion must contain finite values")
    undistorted_x, undistorted_y = _undistort_normalized_pixel(
        distorted_x, distorted_y, coefficients
    )
    result = (undistorted_x * z, undistorted_y * z, z)
    return tuple(float(value) for value in result)


def _undistort_normalized_pixel(
    distorted_x: float, distorted_y: float, coefficients: Sequence[float]
) -> tuple[float, float]:
    """Invert OpenCV's radial/tangential distortion for one normalized pixel."""
    if len(coefficients) < 4 or not any(abs(value) > 1e-15 for value in coefficients):
        return float(distorted_x), float(distorted_y)
    k1, k2, p1, p2 = (float(value) for value in coefficients[:4])
    k3 = float(coefficients[4]) if len(coefficients) >= 5 else 0.0
    k4 = float(coefficients[5]) if len(coefficients) >= 6 else 0.0
    k5 = float(coefficients[6]) if len(coefficients) >= 7 else 0.0
    k6 = float(coefficients[7]) if len(coefficients) >= 8 else 0.0
    x, y = float(distorted_x), float(distorted_y)
    for _ in range(8):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        if abs(denominator) <= 1e-12:
            break
        radial = numerator / denominator
        if abs(radial) <= 1e-12:
            break
        tangential_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        tangential_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (float(distorted_x) - tangential_x) / radial
        y = (float(distorted_y) - tangential_y) / radial
    return x, y


def _box_mask(
    shape: tuple[int, int],
    xyxy: Sequence[float],
    role: str,
) -> tuple[np.ndarray, tuple[float, float]]:
    if role not in {"hole", "target"}:
        raise ValueError("role must be hole or target")
    x0, y0, x1, y1 = _as_vector(xyxy, 4, "xyxy")
    height, width = shape
    left = max(0, int(math.floor(x0)))
    top = max(0, int(math.floor(y0)))
    right = min(width, int(math.ceil(x1)) + 1)
    bottom = min(height, int(math.ceil(y1)) + 1)
    if right <= left or bottom <= top:
        raise ValueError("xyxy does not intersect the image")
    rows, cols = np.mgrid[top:bottom, left:right]
    cx, cy = (float(x0 + x1) * 0.5, float(y0 + y1) * 0.5)
    rx = max(float(x1 - x0) * 0.5, 1.0)
    ry = max(float(y1 - y0) * 0.5, 1.0)
    normalized_radius = ((cols - cx) / rx) ** 2 + ((rows - cy) / ry) ** 2
    if role == "target":
        selected = normalized_radius <= 0.64
    else:
        # The box center is often the empty hole. Prefer its rim, not the
        # background visible through the opening.
        selected = (normalized_radius >= 0.16) & (normalized_radius <= 1.0)
    return selected, (cx, cy)


def hole_ring_depth_support(
    depth_u16: np.ndarray,
    xyxy: Sequence[float],
    *,
    depth_scale_m: float = 0.001,
    min_depth_m: float = 0.03,
    max_depth_m: float = 2.0,
) -> dict[str, float | int]:
    """Summarize valid depth in the scale-1 ring of one hole box.

    The admission test deliberately uses the same rim semantics as the hole
    estimator: the empty hole center is excluded, and enlarged adaptive
    regions are not allowed to make an otherwise empty frame count as valid.
    """
    depth = np.asarray(depth_u16)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("depth_u16 must be a two-dimensional uint16 image")
    if not math.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be positive and finite")
    if not math.isfinite(min_depth_m) or not math.isfinite(max_depth_m) or min_depth_m <= 0.0 or max_depth_m < min_depth_m:
        raise ValueError("depth range is invalid")
    selected, _ = _box_mask((int(depth.shape[0]), int(depth.shape[1])), xyxy, "hole")
    x0, y0, x1, y1 = _as_vector(xyxy, 4, "xyxy")
    left = max(0, int(math.floor(x0)))
    top = max(0, int(math.floor(y0)))
    right = min(depth.shape[1], int(math.ceil(x1)) + 1)
    bottom = min(depth.shape[0], int(math.ceil(y1)) + 1)
    patch = depth[top:bottom, left:right]
    raw_valid = patch[selected]
    raw_valid = raw_valid[raw_valid > 0]
    values_m = raw_valid.astype(np.float64) * float(depth_scale_m)
    valid = values_m[(values_m >= float(min_depth_m)) & (values_m <= float(max_depth_m))]
    ring_area = max(1, int(np.count_nonzero(selected)))
    return {
        "ring_pixels": ring_area,
        "raw_valid_pixels": int(raw_valid.size),
        "valid_pixels": int(valid.size),
        "support_fraction": float(valid.size / ring_area),
    }


def _robust_depth(values_m: np.ndarray, *, max_mad_m: float) -> tuple[float, float, float]:
    values = np.asarray(values_m, dtype=np.float64)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("depth sample is empty or non-finite")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    threshold = max(0.003, 3.0 * mad)
    inliers = values[np.abs(values - median) <= threshold]
    if inliers.size == 0:
        raise ValueError("depth sample has no robust inliers")
    selected_median = float(np.median(inliers))
    selected_mad = float(np.median(np.abs(inliers - selected_median)))
    if selected_mad > max_mad_m:
        raise ValueError("depth sample is too noisy")
    return selected_median, selected_mad, float(np.mean(inliers))


def estimate_box_depth(
    aligned_depth_u16: np.ndarray,
    xyxy: Sequence[float],
    *,
    intrinsic: Mapping[str, float],
    role: str,
    depth_scale_m: float = 0.001,
    min_valid_pixels: int = 8,
    min_support_fraction: float = 0.20,
    min_depth_m: float = 0.03,
    max_depth_m: float = 2.0,
    max_mad_m: float = 0.015,
) -> DepthEstimate:
    """Estimate a box's color-frame 3D point from aligned depth.

    ``hole`` uses a rim/annulus and ``target`` uses an inner region.  The
    returned point uses the detector center and a robust Z estimate; this
    avoids treating the empty hole center or one noisy depth pixel as truth.
    """
    depth = np.asarray(aligned_depth_u16)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("aligned_depth_u16 must be a uint16 image")
    if not math.isfinite(depth_scale_m) or depth_scale_m <= 0.0:
        raise ValueError("depth_scale_m must be positive and finite")
    selected, center = _box_mask((int(depth.shape[0]), int(depth.shape[1])), xyxy, role)
    x0, y0, x1, y1 = _as_vector(xyxy, 4, "xyxy")
    left = max(0, int(math.floor(x0)))
    top = max(0, int(math.floor(y0)))
    right = min(depth.shape[1], int(math.ceil(x1)) + 1)
    bottom = min(depth.shape[0], int(math.ceil(y1)) + 1)
    patch = depth[top:bottom, left:right]
    values = patch[selected]
    values = values[values > 0].astype(np.float64) * float(depth_scale_m)
    box_area = max(1, int(selected.sum()))
    support_fraction = float(values.size) / float(box_area)
    if values.size < int(min_valid_pixels):
        raise ValueError("insufficient valid depth pixels")
    if support_fraction < float(min_support_fraction):
        raise ValueError("depth support fraction is too small")
    values = values[(values >= float(min_depth_m)) & (values <= float(max_depth_m))]
    if values.size < int(min_valid_pixels):
        raise ValueError("depth values are outside the valid range")
    mean_m = float(np.mean(values))
    depth_m, mad_m, robust_mean_m = _robust_depth(values, max_mad_m=float(max_mad_m))
    point_color = backproject_pixel(center, depth_m, intrinsic)
    return DepthEstimate(
        center_px=(float(center[0]), float(center[1])),
        depth_m=float(depth_m),
        valid_pixels=int(values.size),
        support_fraction=float(support_fraction),
        mad_m=float(mad_m),
        point_color_m=point_color,
        mean_m=mean_m,
        robust_mean_m=robust_mean_m,
    )


def estimate_hole_depth_adaptive(
    aligned_depth_u16: np.ndarray,
    xyxy: Sequence[float],
    *,
    intrinsic: Mapping[str, float],
    scales: Sequence[float] = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0),
    guard_xyxy: Sequence[float] | None = None,
    max_padding_px: float = 300.0,
    min_valid_pixels: int = 8,
    min_support_fraction: float = 0.20,
    min_covered_sectors: int = 3,
    depth_scale_m: float = 0.001,
    min_depth_m: float = 0.03,
    max_depth_m: float = 2.0,
    max_mad_m: float = 0.015,
    sampling_region: str = "ring",
) -> tuple[DepthEstimate, dict]:
    """Estimate hole depth from a box or the smallest safe nested rim region.

    ``sampling_region='ring'`` keeps the original hole center/radii fixed while
    the outer rim expands; ``sampling_region='box'`` uses every pixel in the
    detector box at scale 1.  In both cases invalid values are removed and the
    returned depth is a robust median.  ``guard_xyxy`` limits sampling to the
    central right-camera ROI.
    """
    depth = np.asarray(aligned_depth_u16)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("aligned_depth_u16 must be a uint16 image")
    if not math.isfinite(max_padding_px) or max_padding_px < 0.0:
        raise ValueError("max_padding_px must be finite and non-negative")
    if int(min_covered_sectors) < 1 or int(min_covered_sectors) > 4:
        raise ValueError("min_covered_sectors must be in [1, 4]")
    if sampling_region not in {"ring", "box"}:
        raise ValueError("sampling_region must be 'ring' or 'box'")
    x0, y0, x1, y1 = _as_vector(xyxy, 4, "xyxy")
    cx, cy = (float(x0 + x1) * 0.5, float(y0 + y1) * 0.5)
    rx = max(float(x1 - x0) * 0.5, 1.0)
    ry = max(float(y1 - y0) * 0.5, 1.0)
    try:
        candidates = sorted({float(scale) for scale in scales})
    except (TypeError, ValueError) as exc:
        raise ValueError("scales must contain finite positive values") from exc
    if not candidates or candidates[0] != 1.0 or any(not math.isfinite(scale) or scale < 1.0 for scale in candidates):
        raise ValueError("scales must start at 1.0 and be finite non-decreasing values")
    guard = None if guard_xyxy is None else _as_vector(guard_xyxy, 4, "guard_xyxy")
    diagnostics = {"state": "hold", "scales": []}
    for scale in candidates:
        if sampling_region == "box" and scale != 1.0:
            diagnostics["scales"].append({"scale": scale, "reason": "box_region_does_not_expand"})
            continue
        if (scale - 1.0) * max(rx, ry) > max_padding_px:
            diagnostics["scales"].append({"scale": scale, "reason": "max_padding_exceeded"})
            continue
        ex0, ey0 = cx - rx * scale, cy - ry * scale
        ex1, ey1 = cx + rx * scale, cy + ry * scale
        if guard is not None:
            ex0, ey0 = max(ex0, float(guard[0])), max(ey0, float(guard[1]))
            ex1, ey1 = min(ex1, float(guard[2])), min(ey1, float(guard[3]))
        left = max(0, int(math.floor(ex0)))
        top = max(0, int(math.floor(ey0)))
        right = min(depth.shape[1], int(math.ceil(ex1)) + 1)
        bottom = min(depth.shape[0], int(math.ceil(ey1)) + 1)
        if right <= left or bottom <= top:
            diagnostics["scales"].append({"scale": scale, "reason": "outside_guard"})
            continue
        rows, cols = np.mgrid[top:bottom, left:right]
        normalized_radius = ((cols - cx) / rx) ** 2 + ((rows - cy) / ry) ** 2
        if sampling_region == "box":
            # In registered same-pixel mode the detector box is already in the
            # color grid.  Use every pixel in that rectangle; invalid depth
            # values are removed below.  The center may be zero for a true
            # opening, which is why zero filtering remains mandatory.
            selected = np.ones_like(normalized_radius, dtype=bool)
        else:
            selected = (normalized_radius >= 0.16) & (normalized_radius <= scale * scale)
        if guard is not None:
            selected &= (cols >= float(guard[0])) & (cols <= float(guard[2])) & (rows >= float(guard[1])) & (rows <= float(guard[3]))
        raw_values = depth[top:bottom, left:right]
        valid = selected & (raw_values > 0)
        raw_count = int(np.count_nonzero(valid))
        mask_area = max(1, int(np.count_nonzero(selected)))
        support = float(raw_count) / float(mask_area)
        values = raw_values[valid].astype(np.float64) * float(depth_scale_m)
        values = values[(values >= float(min_depth_m)) & (values <= float(max_depth_m))]
        sectors = 0
        for sector in range(4):
            sector_mask = ((cols >= cx).astype(np.int8) + 2 * (rows >= cy).astype(np.int8)) == sector
            if np.count_nonzero(valid & sector_mask) > 0:
                sectors += 1
        entry = {
            "scale": scale,
            "sampling_region": sampling_region,
            "valid_pixels": int(values.size),
            "raw_valid_pixels": raw_count,
            "support_fraction": round(support, 4),
            "covered_sectors": sectors,
        }
        if values.size < int(min_valid_pixels):
            entry["reason"] = "insufficient_valid_pixels"
            diagnostics["scales"].append(entry)
            continue
        if support < float(min_support_fraction):
            entry["reason"] = "support_fraction_too_small"
            diagnostics["scales"].append(entry)
            continue
        if sectors < int(min_covered_sectors):
            entry["reason"] = "sector_coverage_insufficient"
            diagnostics["scales"].append(entry)
            continue
        try:
            mean_m = float(np.mean(values))
            depth_m, mad_m, robust_mean_m = _robust_depth(values, max_mad_m=float(max_mad_m))
        except ValueError as exc:
            entry["reason"] = str(exc)
            diagnostics["scales"].append(entry)
            continue
        point_color = backproject_pixel((cx, cy), depth_m, intrinsic)
        entry.update({
            "depth_m": round(depth_m, 5),
            "mean_m": round(mean_m, 5),
            "median_m": round(depth_m, 5),
            "robust_mean_m": round(robust_mean_m, 5),
            "mad_mm": round(1000.0 * mad_m, 2),
            "state": "valid",
        })
        diagnostics.update({"state": "valid", "selected": entry})
        return DepthEstimate(
            center_px=(cx, cy),
            depth_m=depth_m,
            valid_pixels=int(values.size),
            support_fraction=support,
            mad_m=mad_m,
            point_color_m=point_color,
            mean_m=mean_m,
            robust_mean_m=robust_mean_m,
        ), diagnostics
    diagnostics["reason"] = "hole_depth_unobservable_at_max_scale"
    raise AdaptiveDepthError("hole depth unavailable at all adaptive scales", diagnostics)


def estimate_point_depth(
    aligned_depth_u16: np.ndarray,
    center_px: Sequence[float],
    *,
    intrinsic: Mapping[str, float],
    depth_scale_m: float = 0.001,
    radius_px: int = 4,
    min_valid_pixels: int = 8,
    min_support_fraction: float = 0.20,
    max_mad_m: float = 0.015,
) -> DepthEstimate:
    """Estimate depth around a specific pixel (e.g. the visible shaft tip)."""
    depth = np.asarray(aligned_depth_u16)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("aligned_depth_u16 must be a uint16 image")
    cx, cy = _as_vector(center_px, 2, "center_px")
    radius = int(radius_px)
    if radius < 1:
        raise ValueError("radius_px must be positive")
    left = max(0, int(math.floor(cx)) - radius)
    top = max(0, int(math.floor(cy)) - radius)
    right = min(depth.shape[1], int(math.floor(cx)) + radius + 1)
    bottom = min(depth.shape[0], int(math.floor(cy)) + radius + 1)
    rows, cols = np.mgrid[top:bottom, left:right]
    selected = (cols - cx) ** 2 + (rows - cy) ** 2 <= float(radius * radius)
    values = depth[top:bottom, left:right][selected]
    values = values[values > 0].astype(np.float64) * float(depth_scale_m)
    support = float(values.size) / float(max(1, int(selected.sum())))
    if values.size < int(min_valid_pixels):
        raise ValueError("insufficient valid tip depth pixels")
    if support < float(min_support_fraction):
        raise ValueError("tip depth support fraction is too small")
    mean_m = float(np.mean(values))
    depth_m, mad_m, robust_mean_m = _robust_depth(values, max_mad_m=float(max_mad_m))
    return DepthEstimate(
        center_px=(float(cx), float(cy)),
        depth_m=float(depth_m),
        valid_pixels=int(values.size),
        support_fraction=support,
        mad_m=float(mad_m),
        point_color_m=backproject_pixel((float(cx), float(cy)), depth_m, intrinsic),
        mean_m=mean_m,
        robust_mean_m=robust_mean_m,
    )


def estimate_pair_geometry(
    aligned_depth_u16: np.ndarray,
    *,
    hole_xyxy: Sequence[float],
    target_xyxy: Sequence[float],
    color_intrinsic: Mapping[str, float],
    rgbd_to_right_end: Mapping[str, Sequence[float]],
    right_end_to_base: Mapping[str, Sequence[float]],
    depth_scale_m: float = 0.001,
    min_valid_pixels: int = 8,
    min_support_fraction: float = 0.20,
    max_mad_m: float = 0.015,
) -> PairGeometry:
    """Estimate hole/target points and transform them into ``base_link``."""
    hole = estimate_box_depth(
        aligned_depth_u16,
        hole_xyxy,
        intrinsic=color_intrinsic,
        role="hole",
        depth_scale_m=depth_scale_m,
        min_valid_pixels=min_valid_pixels,
        min_support_fraction=min_support_fraction,
        max_mad_m=max_mad_m,
    )
    target = estimate_box_depth(
        aligned_depth_u16,
        target_xyxy,
        intrinsic=color_intrinsic,
        role="target",
        depth_scale_m=depth_scale_m,
        min_valid_pixels=min_valid_pixels,
        min_support_fraction=min_support_fraction,
        max_mad_m=max_mad_m,
    )
    hole_end = apply_transform(rgbd_to_right_end, hole.point_color_m)
    target_end = apply_transform(rgbd_to_right_end, target.point_color_m)
    hole_base = apply_transform(right_end_to_base, hole_end)
    target_base = apply_transform(right_end_to_base, target_end)
    return PairGeometry(
        hole=hole,
        target=target,
        hole_base_m=hole_base,
        target_base_m=target_base,
    )
