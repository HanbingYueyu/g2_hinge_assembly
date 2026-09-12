"""Multi-height pixel references for the oblique left hand camera.

The hand camera is oblique, so the pixel at which the real gripper centre
must appear changes as the arm approaches the hinge.  This module keeps the
calibration data independent from GDK and provides a deterministic lookup
from camera-to-target distance.  The integrated workflow can estimate that
distance from the head-camera base-frame target and the measured left
end-link pose when the hand depth image is invalid.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


HEIGHT_REFERENCE_SCHEMA_VERSION = 2
REQUIRED_HEIGHTS_MM = (0.0, 50.0, 100.0)
_HEIGHT_TOLERANCE_MM = 1e-6


def load_height_samples(path: str | Path) -> list[dict]:
    """Load the calibrated 0/50/100 mm samples without STEP geometry."""
    document = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError("height reference must contain a JSON object")
    return validate_height_samples(document.get("height_samples"))


def camera_distance_from_base_point(
    base_point_m: Sequence[float],
    end_link_position_m: Sequence[float],
    end_link_orientation_xyzw: Sequence[float],
    end_link_from_camera: Mapping[str, Sequence[float]],
) -> float:
    """Estimate camera-to-target distance from a base-frame target point.

    ``end_link_from_camera`` is GDK's static camera-to-end-link transform.
    This fallback needs no hand-camera depth: it uses the reliable head-camera
    target in ``base_link`` and the current measured left end-link pose.
    """
    point = np.asarray(tuple(float(value) for value in base_point_m), dtype=np.float64)
    position = np.asarray(tuple(float(value) for value in end_link_position_m), dtype=np.float64)
    quaternion = np.asarray(tuple(float(value) for value in end_link_orientation_xyzw), dtype=np.float64)
    if point.shape != (3,) or position.shape != (3,) or quaternion.shape != (4,):
        raise ValueError("base point, end-link position, and orientation must be finite vectors")
    if not np.isfinite(point).all() or not np.isfinite(position).all() or not np.isfinite(quaternion).all():
        raise ValueError("base point, end-link position, and orientation must be finite vectors")
    qnorm = float(np.linalg.norm(quaternion))
    if qnorm <= 1e-12:
        raise ValueError("end-link orientation must not be zero")
    x, y, z, w = quaternion / qnorm
    base_from_end = np.eye(4, dtype=np.float64)
    base_from_end[:3, :3] = np.asarray((
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    ), dtype=np.float64)
    base_from_end[:3, 3] = position
    translation = np.asarray(tuple(float(value) for value in end_link_from_camera["translation_m"]), dtype=np.float64)
    camera_quaternion = np.asarray(tuple(float(value) for value in end_link_from_camera["rotation_xyzw"]), dtype=np.float64)
    if translation.shape != (3,) or camera_quaternion.shape != (4,) or not np.isfinite(translation).all() or not np.isfinite(camera_quaternion).all():
        raise ValueError("end_link_from_camera must contain finite translation and rotation")
    qnorm = float(np.linalg.norm(camera_quaternion))
    if qnorm <= 1e-12:
        raise ValueError("end_link_from_camera rotation must not be zero")
    x, y, z, w = camera_quaternion / qnorm
    end_from_camera = np.eye(4, dtype=np.float64)
    end_from_camera[:3, :3] = np.asarray((
        (1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)),
        (2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)),
        (2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)),
    ), dtype=np.float64)
    end_from_camera[:3, 3] = translation
    camera_point = np.linalg.inv(base_from_end @ end_from_camera) @ np.r_[point, 1.0]
    distance = float(np.linalg.norm(camera_point[:3]))
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("computed camera distance must be positive and finite")
    return distance


def _finite_vector(value, shape: tuple[int, ...], name: str) -> list[float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite values with shape {shape}")
    return [float(item) for item in array.reshape(-1)]


def aggregate_height_observations(
    height_mm: float,
    *,
    pixels: Sequence[Sequence[float]],
    grasp_xyz: Sequence[Sequence[float]],
) -> dict:
    """Reduce one manually marked height to a robust median calibration anchor."""
    try:
        height = float(height_mm)
    except (TypeError, ValueError) as exc:
        raise ValueError("height_mm must be finite") from exc
    if not math.isfinite(height):
        raise ValueError("height_mm must be finite")
    pixel_array = np.asarray(pixels, dtype=np.float64)
    xyz_array = np.asarray(grasp_xyz, dtype=np.float64)
    if pixel_array.ndim != 2 or pixel_array.shape[1] != 2 or xyz_array.ndim != 2 or xyz_array.shape[1] != 3:
        raise ValueError("pixels must be Nx2 and grasp_xyz must be Nx3")
    if len(pixel_array) == 0 or len(pixel_array) != len(xyz_array) or not np.isfinite(pixel_array).all() or not np.isfinite(xyz_array).all():
        raise ValueError("pixels and grasp_xyz must contain the same non-empty finite samples")
    median_pixel = np.median(pixel_array, axis=0)
    median_xyz = np.median(xyz_array, axis=0)
    distance = float(np.linalg.norm(median_xyz))
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("median grasp point must have a positive camera distance")
    return {
        "height_mm": height,
        "camera_distance_m": distance,
        "target_pixel_px": [float(value) for value in median_pixel],
        "grasp_xyz_camera_m": [float(value) for value in median_xyz],
        "sample_count": int(len(pixel_array)),
    }


def validate_height_samples(samples: Sequence[Mapping[str, object]]) -> list[dict]:
    """Validate and sort the required 0/50/100 mm calibration anchors."""
    if not isinstance(samples, Sequence) or isinstance(samples, (str, bytes)):
        raise ValueError("height_samples must be a sequence")
    canonical: list[dict] = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            raise ValueError("each height sample must be an object")
        try:
            height = float(sample["height_mm"])
            distance = float(sample["camera_distance_m"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("height sample requires height_mm and camera_distance_m") from exc
        if not math.isfinite(height) or not math.isfinite(distance) or distance <= 0.0:
            raise ValueError("height_mm must be finite and camera_distance_m must be positive")
        pixel = _finite_vector(sample.get("target_pixel_px"), (2,), "target_pixel_px")
        grasp_xyz = sample.get("grasp_xyz_camera_m")
        canonical_sample = {
            "height_mm": height,
            "camera_distance_m": distance,
            "target_pixel_px": pixel,
        }
        if grasp_xyz is not None:
            canonical_sample["grasp_xyz_camera_m"] = _finite_vector(
                grasp_xyz, (3,), "grasp_xyz_camera_m"
            )
        if "sample_count" in sample:
            try:
                count = int(sample["sample_count"])
            except (TypeError, ValueError) as exc:
                raise ValueError("sample_count must be an integer") from exc
            if count < 1:
                raise ValueError("sample_count must be positive")
            canonical_sample["sample_count"] = count
        canonical.append(canonical_sample)

    canonical.sort(key=lambda item: item["height_mm"])
    if len(canonical) != len(REQUIRED_HEIGHTS_MM):
        raise ValueError("height_samples must contain exactly 0, 50 and 100 mm anchors")
    for actual, expected in zip(canonical, REQUIRED_HEIGHTS_MM):
        if abs(actual["height_mm"] - expected) > _HEIGHT_TOLERANCE_MM:
            raise ValueError("height_samples must contain exactly 0, 50 and 100 mm anchors")
    distances = [item["camera_distance_m"] for item in canonical]
    if any(next_distance <= distance for distance, next_distance in zip(distances, distances[1:])):
        raise ValueError("camera_distance_m must increase strictly with calibration height")
    return canonical


def interpolate_target_pixel(
    samples: Sequence[Mapping[str, object]], camera_distance_m: float
) -> tuple[np.ndarray, dict]:
    """Interpolate the gripper target pixel from the current PnP distance.

    Distances outside the calibrated interval clamp to the nearest anchor.
    The returned detail is JSON-serialisable and is intended for diagnostics.
    """
    anchors = validate_height_samples(samples)
    distance = float(camera_distance_m)
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("camera_distance_m must be a positive finite value")
    if distance >= anchors[-1]["camera_distance_m"]:
        return np.asarray(anchors[-1]["target_pixel_px"], dtype=np.float64), {
            "mode": "clamped_high",
            "camera_distance_m": distance,
            "lower_height_mm": anchors[-1]["height_mm"],
            "upper_height_mm": anchors[-1]["height_mm"],
            "alpha": 1.0,
        }
    if distance <= anchors[0]["camera_distance_m"]:
        return np.asarray(anchors[0]["target_pixel_px"], dtype=np.float64), {
            "mode": "clamped_low",
            "camera_distance_m": distance,
            "lower_height_mm": anchors[0]["height_mm"],
            "upper_height_mm": anchors[0]["height_mm"],
            "alpha": 0.0,
        }
    for lower, upper in zip(anchors, anchors[1:]):
        low_distance = lower["camera_distance_m"]
        high_distance = upper["camera_distance_m"]
        if low_distance <= distance <= high_distance:
            alpha = (distance - low_distance) / (high_distance - low_distance)
            lower_pixel = np.asarray(lower["target_pixel_px"], dtype=np.float64)
            upper_pixel = np.asarray(upper["target_pixel_px"], dtype=np.float64)
            pixel = lower_pixel + alpha * (upper_pixel - lower_pixel)
            return pixel, {
                "mode": "interpolated",
                "camera_distance_m": distance,
                "lower_height_mm": lower["height_mm"],
                "upper_height_mm": upper["height_mm"],
                "alpha": float(alpha),
            }
    raise RuntimeError("camera distance did not fall into a validated calibration interval")


def target_pixel_for_step_pose(
    samples: Sequence[Mapping[str, object]], step_pose: Mapping[str, object]
) -> tuple[np.ndarray, dict]:
    """Calculate a target pixel from an accepted STEP pose."""
    if not isinstance(step_pose, Mapping) or not step_pose.get("accepted"):
        raise ValueError("accepted STEP pose is required")
    grasp_xyz = _finite_vector(step_pose.get("grasp_xyz_camera_m"), (3,), "grasp_xyz_camera_m")
    distance = float(np.linalg.norm(np.asarray(grasp_xyz, dtype=np.float64)))
    return interpolate_target_pixel(samples, distance)
