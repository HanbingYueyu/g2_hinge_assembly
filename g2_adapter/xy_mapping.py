"""Pure helpers for the guarded one-step left-hand XY mapping probe."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


MAX_VERTICAL_IK_RESIDUAL_M = 0.00005


def pixel_jacobian_from_end_pose_trials(
    *,
    x_actual_base_delta_m: Sequence[float],
    y_actual_base_delta_m: Sequence[float],
    x_pixel_delta_px: Sequence[float],
    y_pixel_delta_px: Sequence[float],
) -> np.ndarray:
    """Fit ``pixel_delta = J @ actual_end_link_xy_delta`` from two trials.

    The two trials must originate from the same GDK end-pose control
    interface used in production (historical probes used mode 1; formal motion
    now uses mode 3). Actual status-pose displacement rather
    than nominal request is used, so small whole-body IK cross-axis coupling
    is included in the calibrated mapping.
    """
    deltas = (
        np.asarray(x_actual_base_delta_m, dtype=np.float64),
        np.asarray(y_actual_base_delta_m, dtype=np.float64),
        np.asarray(x_pixel_delta_px, dtype=np.float64),
        np.asarray(y_pixel_delta_px, dtype=np.float64),
    )
    if any(value.shape != (3,) for value in deltas[:2]) or any(value.shape != (2,) for value in deltas[2:]):
        raise ValueError("end-pose trials require two 3-D actual deltas and two 2-D pixel deltas")
    if not all(np.all(np.isfinite(value)) for value in deltas):
        raise ValueError("end-pose trials must be finite")
    actual_xy = np.column_stack((deltas[0][:2], deltas[1][:2]))
    pixel_delta = np.column_stack((deltas[2], deltas[3]))
    determinant = float(np.linalg.det(actual_xy))
    if abs(determinant) < 1e-8:
        raise ValueError("actual end-link XY trials are nearly parallel; cannot calibrate a 2-D mapping")
    jacobian = pixel_delta @ np.linalg.inv(actual_xy)
    if np.linalg.matrix_rank(jacobian) < 2 or not np.all(np.isfinite(jacobian)):
        raise ValueError("measured pixel Jacobian is singular or non-finite")
    return jacobian


def single_axis_base_displacement(axis: str, direction: str, step_m: float) -> np.ndarray:
    """Return exactly one horizontal base-link displacement for a probe."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be x or y")
    if direction not in {"positive", "negative"}:
        raise ValueError("direction must be positive or negative")
    if not math.isfinite(step_m) or not 0.0 < step_m <= 0.001:
        raise ValueError("step_m must be finite and in (0, 0.001]")
    sign = 1.0 if direction == "positive" else -1.0
    return np.asarray((sign * step_m, 0.0, 0.0) if axis == "x" else (0.0, sign * step_m, 0.0), dtype=np.float64)


def choose_locked_candidate(
    statistics: Sequence[dict],
    candidates: Sequence[int],
    *,
    reference_center_px: Sequence[float],
    max_distance_px: float,
) -> int | None:
    """Keep the original target; never substitute a new one after motion."""
    reference = np.asarray(reference_center_px, dtype=np.float64)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError("reference_center_px must contain two finite values")
    if not math.isfinite(max_distance_px) or max_distance_px <= 0.0:
        raise ValueError("max_distance_px must be positive and finite")
    nearby = []
    for index in candidates:
        center = np.asarray(statistics[index].get("center_px"), dtype=np.float64)
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            continue
        distance = float(np.linalg.norm(center - reference))
        if distance <= max_distance_px:
            nearby.append((distance, int(index)))
    return None if not nearby else min(nearby)[1]


def choose_candidate_by_left_rank(
    statistics: Sequence[dict], candidates: Sequence[int], *, rank: int
) -> int | None:
    """Select the Nth candidate from image left, independent of mask order."""
    if rank < 1:
        raise ValueError("rank must be at least one")
    ordered = []
    for index in candidates:
        center = np.asarray(statistics[index].get("center_px"), dtype=np.float64)
        if center.shape == (2,) and np.all(np.isfinite(center)):
            ordered.append((float(center[0]), int(index)))
    ordered.sort()
    return None if len(ordered) < rank else ordered[rank - 1][1]


def planned_base_displacement_is_safe(
    *,
    requested_base_delta_m: Sequence[float],
    planned_base_delta_m: Sequence[float],
    min_fraction: float,
) -> bool:
    """Accept only a finite, horizontal planned step that retains requested sign."""
    requested = np.asarray(requested_base_delta_m, dtype=np.float64)
    planned = np.asarray(planned_base_delta_m, dtype=np.float64)
    if requested.shape != (3,) or planned.shape != (3,):
        return False
    if not np.all(np.isfinite(requested)) or not np.all(np.isfinite(planned)):
        return False
    if not math.isfinite(min_fraction) or not 0.0 < min_fraction <= 1.0:
        return False
    requested_norm = float(np.linalg.norm(requested[:2]))
    if (
        requested_norm <= 0.0
        or abs(float(requested[2])) > 1e-12
        or abs(float(planned[2])) > MAX_VERTICAL_IK_RESIDUAL_M
    ):
        return False
    return float(np.dot(requested[:2], planned[:2])) >= min_fraction * requested_norm * requested_norm
