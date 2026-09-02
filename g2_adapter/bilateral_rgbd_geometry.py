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
    result = ((u - ppx) * z / fx, (v - ppy) * z / fy, z)
    return tuple(float(value) for value in result)


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


def _robust_depth(values_m: np.ndarray, *, max_mad_m: float) -> tuple[float, float]:
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
    return selected_median, selected_mad


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
    depth_m, mad_m = _robust_depth(values, max_mad_m=float(max_mad_m))
    point_color = backproject_pixel(center, depth_m, intrinsic)
    return DepthEstimate(
        center_px=(float(center[0]), float(center[1])),
        depth_m=float(depth_m),
        valid_pixels=int(values.size),
        support_fraction=float(support_fraction),
        mad_m=float(mad_m),
        point_color_m=point_color,
    )


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
    depth_m, mad_m = _robust_depth(values, max_mad_m=float(max_mad_m))
    return DepthEstimate(
        center_px=(float(cx), float(cy)),
        depth_m=float(depth_m),
        valid_pixels=int(values.size),
        support_fraction=support,
        mad_m=float(mad_m),
        point_color_m=backproject_pixel((float(cx), float(cy)), depth_m, intrinsic),
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
