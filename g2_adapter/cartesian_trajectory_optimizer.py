"""Offline simplification and retiming for absolute Cartesian teaching paths."""

from __future__ import annotations

import math
import statistics
from typing import Iterable

import numpy as np

try:
    from .cartesian_teach import (
        CartesianSample,
        CartesianTrajectory,
        _quaternion_distance,
        _slerp,
    )
except ImportError:
    from cartesian_teach import CartesianSample, CartesianTrajectory, _quaternion_distance, _slerp


_CONTROL_TICK_NS = 10_000_000


def _validate_parameters(
    *, position_tolerance_m: float, orientation_tolerance_rad: float,
    jitter_position_m: float, jitter_orientation_rad: float, median_window: int,
    pause_position_m: float, pause_orientation_rad: float,
    outlier_position_m: float, outlier_orientation_rad: float,
    smoothing_window_s: float,
) -> None:
    for value, label in (
        (position_tolerance_m, "position_tolerance_m"),
        (orientation_tolerance_rad, "orientation_tolerance_rad"),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{label} must be positive and finite")
    for value, label in (
        (jitter_position_m, "jitter_position_m"),
        (jitter_orientation_rad, "jitter_orientation_rad"),
        (pause_position_m, "pause_position_m"),
        (pause_orientation_rad, "pause_orientation_rad"),
        (outlier_position_m, "outlier_position_m"),
        (outlier_orientation_rad, "outlier_orientation_rad"),
    ):
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{label} must be non-negative and finite")
    if not isinstance(median_window, int) or median_window < 1 or median_window % 2 == 0:
        raise ValueError("median_window must be a positive odd integer")
    if not math.isfinite(smoothing_window_s) or smoothing_window_s < 0.0:
        raise ValueError("smoothing_window_s must be non-negative and finite")


def _aligned_quaternion(quaternion: Iterable[float], reference: Iterable[float]) -> np.ndarray:
    value = np.asarray(tuple(quaternion), dtype=float)
    reference_value = np.asarray(tuple(reference), dtype=float)
    if float(np.dot(value, reference_value)) < 0.0:
        value = -value
    return value


def _median_quaternion(values: Iterable[Iterable[float]], reference: Iterable[float]) -> tuple[float, ...]:
    aligned = np.asarray([_aligned_quaternion(value, reference) for value in values], dtype=float)
    result = np.median(aligned, axis=0)
    norm = float(np.linalg.norm(result))
    if not math.isfinite(norm) or norm <= 1e-12:
        return tuple(float(value) for value in reference)
    result /= norm
    return tuple(float(value) for value in result)


def _position_distance(first: Iterable[float], second: Iterable[float]) -> float:
    return math.sqrt(sum((a-b) ** 2 for a, b in zip(first, second)))


def _interpolate_sample(before: CartesianSample, after: CartesianSample, current: CartesianSample) -> CartesianSample:
    duration = after.t_ns - before.t_ns
    fraction = 0.5 if duration <= 0 else (current.t_ns - before.t_ns) / duration
    fraction = min(1.0, max(0.0, fraction))
    position = tuple(a + fraction * (b - a) for a, b in zip(before.position_m, after.position_m))
    orientation = _slerp(before.orientation_xyzw, after.orientation_xyzw, fraction)
    joints = tuple(
        a + fraction * (b - a)
        for a, b in zip(before.reference_joint_positions_rad, after.reference_joint_positions_rad)
    )
    return CartesianSample(current.t_ns, position, orientation, joints)


def _repair_isolated_outliers(
    trajectory: CartesianTrajectory, *, outlier_position_m: float,
    outlier_orientation_rad: float,
) -> tuple[tuple[CartesianSample, ...], int]:
    """Replace isolated tracking spikes with interpolation between neighbours."""
    samples = list(trajectory.samples)
    repaired = 0
    for index in range(1, len(samples) - 1):
        before, current, after = samples[index - 1:index + 2]
        expected = _interpolate_sample(before, after, current)
        position_error = _position_distance(current.position_m, expected.position_m)
        orientation_error = _quaternion_distance(current.orientation_xyzw, expected.orientation_xyzw)
        bridge_position = _position_distance(before.position_m, after.position_m)
        spike_position = max(
            _position_distance(before.position_m, current.position_m),
            _position_distance(current.position_m, after.position_m),
        )
        bridge_orientation = _quaternion_distance(before.orientation_xyzw, after.orientation_xyzw)
        spike_orientation = max(
            _quaternion_distance(before.orientation_xyzw, current.orientation_xyzw),
            _quaternion_distance(current.orientation_xyzw, after.orientation_xyzw),
        )
        position_spike = (
            position_error > outlier_position_m
            and bridge_position < max(outlier_position_m, spike_position * 0.5)
        )
        orientation_spike = (
            orientation_error > outlier_orientation_rad
            and bridge_orientation < max(outlier_orientation_rad, spike_orientation * 0.5)
        )
        if position_spike or orientation_spike:
            samples[index] = expected
            repaired += 1
    return tuple(samples), repaired


def _median_filter(
    trajectory: CartesianTrajectory, *, median_window: int,
    jitter_position_m: float, jitter_orientation_rad: float,
) -> tuple[CartesianSample, ...]:
    samples = trajectory.samples
    if median_window == 1 or len(samples) < 3:
        return samples
    radius = median_window // 2
    filtered = list(samples)
    for index in range(1, len(samples)-1):
        first = max(0, index-radius)
        last = min(len(samples), index+radius+1)
        neighbours = samples[first:last]
        position = tuple(
            statistics.median(sample.position_m[axis] for sample in neighbours)
            for axis in range(3)
        )
        orientation = _median_quaternion(
            (sample.orientation_xyzw for sample in neighbours),
            samples[index].orientation_xyzw,
        )
        if (
            _position_distance(position, samples[index].position_m) <= jitter_position_m
            and _quaternion_distance(orientation, samples[index].orientation_xyzw) <= jitter_orientation_rad
        ):
            filtered[index] = CartesianSample(
                samples[index].t_ns, position, orientation,
                samples[index].reference_joint_positions_rad,
            )
    return tuple(filtered)


def _smooth_timed_samples(
    samples: tuple[CartesianSample, ...], *, window_s: float,
) -> tuple[CartesianSample, ...]:
    """Suppress short 100 Hz reversals without jumping away from either endpoint."""
    if window_s == 0.0 or len(samples) < 5:
        return samples
    times = np.asarray([sample.t_ns for sample in samples], dtype=np.float64) / 1e9
    positions = np.asarray([sample.position_m for sample in samples], dtype=np.float64)
    joints = np.asarray([sample.reference_joint_positions_rad for sample in samples], dtype=np.float64)
    orientations = np.asarray([sample.orientation_xyzw for sample in samples], dtype=np.float64)
    half_window_s = window_s / 2.0
    sigma_s = window_s / 4.0
    smooth_positions = np.empty_like(positions)
    smooth_joints = np.empty_like(joints)
    smooth_orientations = np.empty_like(orientations)
    for index in range(len(samples)):
        first = int(np.searchsorted(times, times[index] - half_window_s, side="left"))
        last = int(np.searchsorted(times, times[index] + half_window_s, side="right"))
        weights = np.exp(-0.5 * ((times[first:last] - times[index]) / sigma_s) ** 2)
        weights /= weights.sum()
        smooth_positions[index] = np.sum(positions[first:last] * weights[:, None], axis=0)
        smooth_joints[index] = np.sum(joints[first:last] * weights[:, None], axis=0)
        quaternion_window = orientations[first:last].copy()
        quaternion_window[quaternion_window @ orientations[index] < 0.0] *= -1.0
        smooth_orientations[index] = np.sum(quaternion_window * weights[:, None], axis=0)
        smooth_orientations[index] /= np.linalg.norm(smooth_orientations[index])

    def endpoint_weight(distance_s: float) -> float:
        fraction = min(1.0, max(0.0, distance_s / half_window_s))
        return 1.0 - 3.0 * fraction**2 + 2.0 * fraction**3

    start_orientation = _aligned_quaternion(orientations[0], smooth_orientations[0])
    end_orientation = _aligned_quaternion(orientations[-1], smooth_orientations[-1])
    result = [samples[0]]
    for index in range(1, len(samples) - 1):
        start_weight = endpoint_weight(times[index] - times[0])
        end_weight = endpoint_weight(times[-1] - times[index])
        position = (
            smooth_positions[index]
            + start_weight * (positions[0] - smooth_positions[0])
            + end_weight * (positions[-1] - smooth_positions[-1])
        )
        reference_joints = (
            smooth_joints[index]
            + start_weight * (joints[0] - smooth_joints[0])
            + end_weight * (joints[-1] - smooth_joints[-1])
        )
        orientation = (
            smooth_orientations[index]
            + start_weight * (start_orientation - smooth_orientations[0])
            + end_weight * (end_orientation - smooth_orientations[-1])
        )
        orientation /= np.linalg.norm(orientation)
        result.append(CartesianSample(
            samples[index].t_ns,
            tuple(float(value) for value in position),
            tuple(float(value) for value in orientation),
            tuple(float(value) for value in reference_joints),
        ))
    result.append(samples[-1])
    return tuple(result)


def _interpolated_pose_error(
    samples: tuple[CartesianSample, ...], first: int, last: int, index: int,
) -> tuple[float, float]:
    start, end, current = samples[first], samples[last], samples[index]
    duration = end.t_ns-start.t_ns
    fraction = 0.0 if duration <= 0 else (current.t_ns-start.t_ns)/duration
    fraction = min(1.0, max(0.0, fraction))
    position = tuple(a + fraction*(b-a) for a, b in zip(start.position_m, end.position_m))
    orientation = _slerp(start.orientation_xyzw, end.orientation_xyzw, fraction)
    return _position_distance(current.position_m, position), _quaternion_distance(current.orientation_xyzw, orientation)


def _select_keyframes(
    samples: tuple[CartesianSample, ...], *, position_tolerance_m: float,
    orientation_tolerance_rad: float,
) -> tuple[list[int], float, float]:
    if len(samples) <= 2:
        return list(range(len(samples))), 0.0, 0.0
    selected = {0, len(samples)-1}
    stack = [(0, len(samples)-1)]
    max_position_error = 0.0
    max_orientation_error = 0.0
    while stack:
        first, last = stack.pop()
        worst_score = 1.0
        split = None
        for index in range(first+1, last):
            position_error, orientation_error = _interpolated_pose_error(samples, first, last, index)
            max_position_error = max(max_position_error, position_error)
            max_orientation_error = max(max_orientation_error, orientation_error)
            score = max(position_error/position_tolerance_m, orientation_error/orientation_tolerance_rad)
            if score > worst_score:
                worst_score = score
                split = index
        if split is not None:
            selected.add(split)
            stack.extend(((first, split), (split, last)))
    return sorted(selected), max_position_error, max_orientation_error


def _active_interval(
    first: CartesianSample, second: CartesianSample, *,
    pause_position_m: float, pause_orientation_rad: float,
) -> bool:
    return (
        _position_distance(first.position_m, second.position_m) > pause_position_m
        or _quaternion_distance(first.orientation_xyzw, second.orientation_xyzw) > pause_orientation_rad
    )


def _retime_keyframes(
    samples: tuple[CartesianSample, ...], indices: list[int], *,
    pause_position_m: float, pause_orientation_rad: float,
) -> tuple[CartesianSample, ...]:
    if len(indices) <= 1:
        sample = samples[indices[0]]
        return (CartesianSample(
            0, sample.position_m, sample.orientation_xyzw,
            sample.reference_joint_positions_rad,
        ),)
    elapsed = 0
    result = [CartesianSample(0, samples[indices[0]].position_m,
                              samples[indices[0]].orientation_xyzw,
                              samples[indices[0]].reference_joint_positions_rad)]
    for previous_index, current_index in zip(indices, indices[1:]):
        active_duration = sum(
            samples[index+1].t_ns-samples[index].t_ns
            for index in range(previous_index, current_index)
            if _active_interval(
                samples[index], samples[index+1],
                pause_position_m=pause_position_m,
                pause_orientation_rad=pause_orientation_rad,
            )
        )
        elapsed += max(_CONTROL_TICK_NS, active_duration)
        if current_index == indices[-1]:
            # A fractional final tick would create a near-zero 100 Hz replay
            # interval and spuriously inflate the measured acceleration.
            elapsed = ((elapsed + _CONTROL_TICK_NS - 1) // _CONTROL_TICK_NS) * _CONTROL_TICK_NS
        sample = samples[current_index]
        result.append(CartesianSample(
            elapsed, sample.position_m, sample.orientation_xyzw,
            sample.reference_joint_positions_rad,
        ))
    return tuple(result)


def optimize_cartesian_trajectory(
    trajectory: CartesianTrajectory, *, position_tolerance_m: float = 0.001,
    orientation_tolerance_rad: float = math.radians(1.0),
    jitter_position_m: float = 0.002,
    jitter_orientation_rad: float = math.radians(1.0),
    median_window: int = 7,
    pause_position_m: float = 0.00001,
    pause_orientation_rad: float = math.radians(0.01),
    outlier_position_m: float = 0.008,
    outlier_orientation_rad: float = math.radians(5.0),
    smoothing_window_s: float = 1.0,
) -> tuple[CartesianTrajectory, dict]:
    """Simplify a Cartesian path while preserving its absolute endpoints.

    Stationary intervals are removed from the keyframe timing. The returned
    sparse path is intended to be resampled and solved by the IK pipeline.
    """
    _validate_parameters(
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
        jitter_position_m=jitter_position_m,
        jitter_orientation_rad=jitter_orientation_rad,
        median_window=median_window,
        pause_position_m=pause_position_m,
        pause_orientation_rad=pause_orientation_rad,
        outlier_position_m=outlier_position_m,
        outlier_orientation_rad=outlier_orientation_rad,
        smoothing_window_s=smoothing_window_s,
    )
    if not trajectory.samples:
        raise ValueError("Cartesian trajectory is empty")
    repaired, repaired_count = _repair_isolated_outliers(
        trajectory,
        outlier_position_m=outlier_position_m,
        outlier_orientation_rad=outlier_orientation_rad,
    )
    repaired_trajectory = CartesianTrajectory(
        trajectory.side, trajectory.base_frame, trajectory.tip_frame,
        trajectory.joint_names, trajectory.fixed_joint_positions, repaired,
    )
    filtered = _median_filter(
        repaired_trajectory, median_window=median_window,
        jitter_position_m=jitter_position_m,
        jitter_orientation_rad=jitter_orientation_rad,
    )
    smoothed = _smooth_timed_samples(filtered, window_s=smoothing_window_s)
    indices, max_position_error, max_orientation_error = _select_keyframes(
        smoothed,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_rad=orientation_tolerance_rad,
    )
    keyframes = _retime_keyframes(
        smoothed, indices,
        pause_position_m=pause_position_m,
        pause_orientation_rad=pause_orientation_rad,
    )
    result = CartesianTrajectory(
        trajectory.side, trajectory.base_frame, trajectory.tip_frame,
        trajectory.joint_names, trajectory.fixed_joint_positions, keyframes,
    )
    report = {
        "original_samples": len(trajectory.samples),
        "keyframes": len(keyframes),
        "removed_samples": len(trajectory.samples)-len(keyframes),
        "key_indices": indices,
        "original_duration_s": trajectory.samples[-1].t_ns/1e9,
        "optimized_duration_s": keyframes[-1].t_ns/1e9,
        "position_tolerance_mm": position_tolerance_m*1000.0,
        "orientation_tolerance_deg": math.degrees(orientation_tolerance_rad),
        "jitter_position_mm": jitter_position_m*1000.0,
        "jitter_orientation_deg": math.degrees(jitter_orientation_rad),
        "pause_position_mm": pause_position_m*1000.0,
        "pause_orientation_deg": math.degrees(pause_orientation_rad),
        "median_window": median_window,
        "smoothing_window_ms": smoothing_window_s * 1000.0,
        "max_simplification_position_error_mm": max_position_error*1000.0,
        "max_simplification_orientation_error_deg": math.degrees(max_orientation_error),
        "stationary_intervals_removed": True,
        "isolated_outliers_repaired": repaired_count,
        "outlier_position_mm": outlier_position_m*1000.0,
        "outlier_orientation_deg": math.degrees(outlier_orientation_rad),
    }
    return result, report
