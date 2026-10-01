"""Pure geometry for the head RGB-D two-hole localization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .bilateral_rgbd_geometry import apply_transform, backproject_pixel, inverse_transform
from .camera import transform_to_dict


def box_depth_m(
    aligned_depth_u16: np.ndarray,
    box_xyxy: Sequence[float],
    *,
    depth_scale_m: float = 0.001,
    min_valid_pixels: int = 12,
    depth_range_m: tuple[float, float] = (0.2, 3.0),
    max_mad_m: float = 0.04,
) -> dict:
    """Estimate surface depth from sparse projected samples inside a box.

    A hole center can have no valid depth or show the surface behind it. The
    returned depth is the robust visible-surface depth near the detected hole,
    not a measurement of the empty space inside the opening.
    """
    depth = np.asarray(aligned_depth_u16)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("aligned depth must be a two-dimensional uint16 array")
    if len(box_xyxy) != 4 or not np.all(np.isfinite(box_xyxy)):
        raise ValueError("box must contain four finite xyxy values")
    x0f, y0f, x1f, y1f = map(float, box_xyxy)
    if not x0f < x1f or not y0f < y1f:
        raise ValueError("box must have positive area")
    x0, y0 = max(0, int(np.floor(x0f))), max(0, int(np.floor(y0f)))
    x1, y1 = min(depth.shape[1], int(np.ceil(x1f))), min(depth.shape[0], int(np.ceil(y1f)))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("box lies outside the aligned depth image")
    patch = depth[y0:y1, x0:x1]
    values = patch.astype(np.float64) * depth_scale_m
    low, high = depth_range_m
    values = values[(values >= low) & (values <= high)]
    if values.size < min_valid_pixels:
        raise ValueError(f"insufficient depth samples: {values.size} < {min_valid_pixels}")
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > max_mad_m:
        raise ValueError(f"depth spread too large: MAD={mad:.4f} m")
    inliers = values[np.abs(values - median) <= max(0.015, 3.0 * mad)]
    if inliers.size < min_valid_pixels:
        raise ValueError(f"insufficient consistent depth samples: {inliers.size} < {min_valid_pixels}")
    return {
        "depth_m": float(np.median(inliers)),
        "valid_pixels": int(values.size),
        "inlier_pixels": int(inliers.size),
        "support_fraction": float(values.size / patch.size),
        "mad_m": mad,
    }


def localize_box(
    box_xyxy: Sequence[float],
    aligned_depth_u16: np.ndarray,
    color_intrinsic: Mapping,
    head_from_color: Mapping,
    base_from_head: Mapping,
) -> dict:
    """Return one detected center in head RGB optical, head_link3, and base_link."""
    estimate = box_depth_m(aligned_depth_u16, box_xyxy)
    x0, y0, x1, y1 = map(float, box_xyxy)
    center = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    color_point = backproject_pixel(center, estimate["depth_m"], color_intrinsic)
    head_point = apply_transform(head_from_color, color_point)
    base_point = apply_transform(base_from_head, head_point)
    return {
        "center_head_color_px": list(center),
        "box_head_color_xyxy_px": list(map(float, box_xyxy)),
        "point_head_color_optical_m": list(color_point),
        "point_head_link3_m": list(head_point),
        "point_base_link_m": list(base_point),
        "depth": estimate,
    }


def project_color_pixel_to_base_z(
    center_px: Sequence[float], plane_z_m: float,
    color_intrinsic: Mapping, head_from_color: Mapping, base_from_head: Mapping,
) -> tuple[float, float, float]:
    """Intersect a head-color pixel ray with a horizontal plane in base_link."""
    if not np.isfinite(plane_z_m):
        raise ValueError("plane Z must be finite")
    origin_head = apply_transform(head_from_color, (0.0, 0.0, 0.0))
    ray_head = apply_transform(head_from_color, backproject_pixel(center_px, 1.0, color_intrinsic))
    origin = np.asarray(apply_transform(base_from_head, origin_head), dtype=float)
    through = np.asarray(apply_transform(base_from_head, ray_head), dtype=float)
    direction = through - origin
    if abs(direction[2]) < 1e-6:
        raise ValueError("color ray is nearly parallel to the base_link horizontal plane")
    distance = (float(plane_z_m) - origin[2]) / direction[2]
    if not 0.2 <= distance <= 3.0:
        raise ValueError(f"plane intersection is outside camera working range: {distance:.3f} m")
    point = origin + distance * direction
    return tuple(float(value) for value in point)


def base_from_head_at(tf, image_timestamp_ns: int, *, max_fallback_age_ms: float = 100.0) -> tuple[dict, dict]:
    """Read GDK TF at image time, with a bounded near-time fallback.

    GDK returns T_source_target for lookup_transform(target, source), so its
    raw base_link/head_link3 result must be inverted once at this boundary.
    """
    raw_latest, latest_timestamp = tf.lookup_transform_latest(
        "base_link", "head_link3", return_timestamp=True,
    )
    latest_timestamp = None if latest_timestamp is None else int(latest_timestamp)
    gap_ms = None if latest_timestamp is None else abs(latest_timestamp - image_timestamp_ns) / 1e6
    if gap_ms is not None and gap_ms <= 10.0:
        return inverse_transform(transform_to_dict(raw_latest)), {
            "mode": "near_latest", "timestamp_ns": latest_timestamp, "age_ms": gap_ms,
        }
    try:
        raw = tf.lookup_transform("base_link", "head_link3", int(image_timestamp_ns))
        return inverse_transform(transform_to_dict(raw)), {
            "mode": "exact_image_time", "timestamp_ns": int(image_timestamp_ns), "age_ms": 0.0,
        }
    except Exception as error:
        if gap_ms is None or gap_ms > max_fallback_age_ms:
            raise RuntimeError(
                f"head TF unavailable at image time; latest gap={gap_ms} ms, "
                f"limit={max_fallback_age_ms} ms: {error}"
            ) from error
        return inverse_transform(transform_to_dict(raw_latest)), {
            "mode": "bounded_latest_fallback", "timestamp_ns": latest_timestamp,
            "age_ms": gap_ms, "exact_lookup_error": str(error),
        }
