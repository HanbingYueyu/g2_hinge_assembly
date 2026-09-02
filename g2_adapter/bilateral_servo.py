"""Pure state machine for post-P4 bilateral XY alignment.

The fixed right-hand camera provides ``target - hole`` pixels.  Only the
left-arm base-link XY position is changed.  There are deliberately no GDK,
camera, or model imports here so every motion decision is testable offline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from .visual_servo import empirical_horizontal_base_delta_from_pixel_error
except ImportError:
    from visual_servo import empirical_horizontal_base_delta_from_pixel_error


@dataclass(frozen=True)
class AlignmentPolicy:
    pixel_tolerance_px: float = 8.0
    stable_frames: int = 4
    aligned_frames: int = 5
    max_pair_jump_px: float = 25.0
    max_error_px: float = 260.0
    max_step_m: float = 0.0015
    max_total_displacement_m: float = 0.010
    max_total_travel_m: float = 0.020
    max_steps: int = 20
    min_confidence: float = 0.25

    def validate(self) -> None:
        values = (
            self.pixel_tolerance_px, self.max_pair_jump_px, self.max_error_px,
            self.max_step_m, self.max_total_displacement_m,
            self.max_total_travel_m, self.min_confidence,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("alignment policy contains non-finite values")
        if not 1.0 <= self.pixel_tolerance_px <= 100.0:
            raise ValueError("pixel_tolerance_px must be in [1, 100]")
        if not 1 <= int(self.stable_frames) <= 30:
            raise ValueError("stable_frames must be in [1, 30]")
        if not 1 <= int(self.aligned_frames) <= 30:
            raise ValueError("aligned_frames must be in [1, 30]")
        if not 1.0 <= self.max_pair_jump_px <= 500.0:
            raise ValueError("max_pair_jump_px must be in [1, 500]")
        if not self.pixel_tolerance_px < self.max_error_px <= 2000.0:
            raise ValueError("max_error_px must be greater than pixel tolerance and <= 2000")
        if not 1e-5 <= self.max_step_m <= 0.010:
            raise ValueError("max_step_m must be in [0.00001, 0.010]")
        if not self.max_step_m <= self.max_total_displacement_m <= 0.20:
            raise ValueError("max_total_displacement_m must be >= max_step_m and <= 0.20")
        if not self.max_step_m <= self.max_total_travel_m <= 0.50:
            raise ValueError("max_total_travel_m must be >= max_step_m and <= 0.50")
        if not 1 <= int(self.max_steps) <= 1000:
            raise ValueError("max_steps must be in [1, 1000]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")


@dataclass(frozen=True)
class AlignmentDecision:
    state: str
    reason: str
    pixel_error_px: tuple[float, float] | None
    requested_delta_m: tuple[float, float, float]
    accepted_delta_m: tuple[float, float, float]
    stable_count: int
    aligned_count: int
    step_index: int
    cumulative_distance_m: float
    cumulative_offset_m: tuple[float, float, float]


def _center(value: Sequence[float] | None) -> np.ndarray | None:
    if value is None:
        return None
    try:
        result = np.asarray(tuple(float(item) for item in value), dtype=np.float64)
    except (TypeError, ValueError):
        return None
    return result if result.shape == (2,) and np.all(np.isfinite(result)) else None


class BilateralAlignmentGate:
    """Convert fresh, unique same-frame detections into bounded XY steps."""

    def __init__(self, pixel_jacobian_px_per_m: Sequence[Sequence[float]], policy: AlignmentPolicy | None = None) -> None:
        matrix = np.asarray(pixel_jacobian_px_per_m, dtype=np.float64)
        if matrix.shape != (2, 2) or not np.all(np.isfinite(matrix)):
            raise ValueError("pixel_jacobian_px_per_m must be a finite 2x2 matrix")
        if np.linalg.matrix_rank(matrix) < 2:
            raise ValueError("pixel_jacobian_px_per_m must be invertible")
        condition = float(np.linalg.cond(matrix))
        if not math.isfinite(condition) or condition > 100.0:
            raise ValueError("pixel_jacobian_px_per_m is too ill-conditioned (condition > 100)")
        self.pixel_jacobian = matrix
        self.policy = policy or AlignmentPolicy()
        self.policy.validate()
        self.reset()

    def reset(self) -> None:
        self._last_sequence: int | None = None
        self._last_pair: np.ndarray | None = None
        self._stable_count = 0
        self._aligned_count = 0
        self._aligned = False
        self._travel_m = 0.0
        self._offset = np.zeros(3, dtype=np.float64)
        self._steps = 0

    def reset_observation(self) -> None:
        """Discard pre-step pixels while retaining the bounded motion budget."""
        self._last_pair = None
        self._stable_count = 0
        self._aligned_count = 0
        self._aligned = False

    @property
    def cumulative_distance_m(self) -> float:
        return self._travel_m

    @property
    def cumulative_offset_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._offset)

    def _decision(self, state: str, reason: str, error: np.ndarray | None = None, requested: np.ndarray | None = None, accepted: np.ndarray | None = None) -> AlignmentDecision:
        return AlignmentDecision(
            state=state,
            reason=reason,
            pixel_error_px=None if error is None else tuple(float(value) for value in error),
            requested_delta_m=tuple(float(value) for value in (requested if requested is not None else np.zeros(3))),
            accepted_delta_m=tuple(float(value) for value in (accepted if accepted is not None else np.zeros(3))),
            stable_count=self._stable_count,
            aligned_count=self._aligned_count,
            step_index=self._steps,
            cumulative_distance_m=float(self._travel_m),
            cumulative_offset_m=self.cumulative_offset_m,
        )

    def update(
        self,
        *,
        sequence: int,
        hole_center_px: Sequence[float] | None,
        target_center_px: Sequence[float] | None,
        hole_confidence: float = 1.0,
        target_confidence: float = 1.0,
        hole_count: int = 1,
        target_count: int = 1,
        can_move: bool = False,
    ) -> AlignmentDecision:
        """Consume exactly one new observation; no old error is ever reused."""
        try:
            sequence = int(sequence)
        except (TypeError, ValueError):
            return self._decision("hold", "invalid_observation_sequence")
        if self._last_sequence is not None and sequence <= self._last_sequence:
            return self._decision("hold", "duplicate_or_old_observation")
        self._last_sequence = sequence
        if hole_count != 1 or target_count != 1:
            self.reset_observation()
            return self._decision("hold", "require_exactly_one_hole_and_target")
        try:
            confidences = (float(hole_confidence), float(target_confidence))
        except (TypeError, ValueError):
            confidences = (math.nan, math.nan)
        if not all(math.isfinite(value) and value >= self.policy.min_confidence for value in confidences):
            self.reset_observation()
            return self._decision("hold", "confidence_below_gate")
        hole = _center(hole_center_px)
        target = _center(target_center_px)
        if hole is None or target is None:
            self.reset_observation()
            return self._decision("hold", "missing_or_invalid_pair")
        pair = np.concatenate((hole, target))
        if self._last_pair is None:
            self._stable_count = 1
        elif max(float(np.linalg.norm(pair[:2] - self._last_pair[:2])), float(np.linalg.norm(pair[2:] - self._last_pair[2:]))) > self.policy.max_pair_jump_px:
            self._stable_count = 1
            self._aligned_count = 0
        else:
            self._stable_count += 1
        self._last_pair = pair
        error = target - hole
        error_norm = float(np.linalg.norm(error))
        if error_norm > self.policy.max_error_px:
            self.reset_observation()
            return self._decision("hold", "pixel_error_exceeds_safety_gate", error=error)
        if error_norm <= self.policy.pixel_tolerance_px:
            self._aligned_count += 1
            if self._aligned_count >= self.policy.aligned_frames and self._stable_count >= self.policy.stable_frames:
                self._aligned = True
                return self._decision("aligned", "pixel_error_within_tolerance", error=error)
            return self._decision("waiting_alignment", "within_tolerance_but_not_yet_confirmed", error=error)
        self._aligned_count = 0
        if self._aligned:
            return self._decision("aligned", "alignment_already_confirmed", error=error)
        if self._stable_count < self.policy.stable_frames:
            return self._decision("waiting_stability", "pair_not_stable_for_required_frames", error=error)
        requested = empirical_horizontal_base_delta_from_pixel_error(error, self.pixel_jacobian, max_step_m=self.policy.max_step_m)
        if not can_move:
            return self._decision("ready", "correction_available_motion_disabled_or_in_flight", error=error, requested=requested)
        if self._steps >= self.policy.max_steps:
            return self._decision("limit_hold", "maximum_step_count_reached", error=error, requested=requested)
        requested_distance = float(np.linalg.norm(requested[:2]))
        if requested_distance <= 1e-12:
            return self._decision("hold", "zero_visual_correction", error=error, requested=requested)
        travel_remaining = self.policy.max_total_travel_m - self._travel_m
        if travel_remaining <= 1e-12:
            return self._decision("limit_hold", "cumulative_travel_limit_reached", error=error, requested=requested)
        scale = min(1.0, travel_remaining / requested_distance)
        radial_remaining = self.policy.max_total_displacement_m - float(np.linalg.norm(self._offset[:2]))
        if radial_remaining <= 1e-12:
            return self._decision("limit_hold", "radial_displacement_limit_reached", error=error, requested=requested)
        scale = min(scale, radial_remaining / (float(np.linalg.norm(self._offset[:2])) + requested_distance))
        accepted = requested * max(0.0, scale)
        accepted_distance = float(np.linalg.norm(accepted[:2]))
        if accepted_distance <= 1e-12:
            return self._decision("limit_hold", "remaining_xy_limit_too_small", error=error, requested=requested)
        self._travel_m += accepted_distance
        self._offset += accepted
        self._steps += 1
        return self._decision("move", "fresh_stable_pair_bounded_xy_correction", error=error, requested=requested, accepted=accepted)


@dataclass(frozen=True)
class MetricAlignmentPolicy:
    """Safety limits for the official RGB-D metric error path."""

    position_tolerance_m: float = 0.002
    stable_frames: int = 4
    aligned_frames: int = 5
    max_pair_jump_m: float = 0.008
    max_error_m: float = 0.060
    max_step_m: float = 0.0015
    max_total_displacement_m: float = 0.010
    max_total_travel_m: float = 0.020
    max_steps: int = 20
    min_confidence: float = 0.25

    def validate(self) -> None:
        values = (
            self.position_tolerance_m, self.max_pair_jump_m, self.max_error_m,
            self.max_step_m, self.max_total_displacement_m,
            self.max_total_travel_m, self.min_confidence,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("metric alignment policy contains non-finite values")
        if not 1e-5 <= self.position_tolerance_m <= 0.10:
            raise ValueError("position_tolerance_m must be in [1e-5, 0.10]")
        if not 1 <= int(self.stable_frames) <= 30:
            raise ValueError("stable_frames must be in [1, 30]")
        if not 1 <= int(self.aligned_frames) <= 30:
            raise ValueError("aligned_frames must be in [1, 30]")
        if not self.position_tolerance_m < self.max_pair_jump_m <= 0.50:
            raise ValueError("max_pair_jump_m must be greater than position tolerance and <= 0.50")
        if not self.position_tolerance_m < self.max_error_m <= 1.0:
            raise ValueError("max_error_m must be greater than position tolerance and <= 1.0")
        if not 1e-5 <= self.max_step_m <= 0.010:
            raise ValueError("max_step_m must be in [1e-5, 0.010]")
        if not self.max_step_m <= self.max_total_displacement_m <= 0.20:
            raise ValueError("max_total_displacement_m must be >= max_step_m and <= 0.20")
        if not self.max_step_m <= self.max_total_travel_m <= 0.50:
            raise ValueError("max_total_travel_m must be >= max_step_m and <= 0.50")
        if not 1 <= int(self.max_steps) <= 1000:
            raise ValueError("max_steps must be in [1, 1000]")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")


@dataclass(frozen=True)
class MetricAlignmentDecision:
    state: str
    reason: str
    error_m: tuple[float, float, float] | None
    requested_delta_m: tuple[float, float, float]
    accepted_delta_m: tuple[float, float, float]
    stable_count: int
    aligned_count: int
    step_index: int
    cumulative_distance_m: float
    cumulative_offset_m: tuple[float, float, float]


class BilateralMetricAlignmentGate:
    """Convert fresh right-camera 3-D pairs into bounded left XY steps.

    The target is the held hinge pin and the hole is fixed.  A positive
    correction is therefore ``hole_base - target_base``; Z is deliberately
    discarded by this gate.
    """

    def __init__(self, policy: MetricAlignmentPolicy | None = None) -> None:
        self.policy = policy or MetricAlignmentPolicy()
        self.policy.validate()
        self.reset()

    def reset(self) -> None:
        self._last_sequence: int | None = None
        self._last_pair: np.ndarray | None = None
        self._stable_count = 0
        self._aligned_count = 0
        self._aligned = False
        self._travel_m = 0.0
        self._offset = np.zeros(3, dtype=np.float64)
        self._steps = 0

    def reset_observation(self) -> None:
        self._last_pair = None
        self._stable_count = 0
        self._aligned_count = 0
        self._aligned = False

    @property
    def cumulative_distance_m(self) -> float:
        return self._travel_m

    @property
    def cumulative_offset_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self._offset)

    def _decision(
        self,
        state: str,
        reason: str,
        error: np.ndarray | None = None,
        requested: np.ndarray | None = None,
        accepted: np.ndarray | None = None,
    ) -> MetricAlignmentDecision:
        return MetricAlignmentDecision(
            state=state,
            reason=reason,
            error_m=None if error is None else tuple(float(value) for value in error),
            requested_delta_m=tuple(float(value) for value in (requested if requested is not None else np.zeros(3))),
            accepted_delta_m=tuple(float(value) for value in (accepted if accepted is not None else np.zeros(3))),
            stable_count=self._stable_count,
            aligned_count=self._aligned_count,
            step_index=self._steps,
            cumulative_distance_m=float(self._travel_m),
            cumulative_offset_m=self.cumulative_offset_m,
        )

    @staticmethod
    def _point(value: Sequence[float] | None) -> np.ndarray | None:
        if value is None:
            return None
        try:
            point = np.asarray(tuple(float(item) for item in value), dtype=np.float64)
        except (TypeError, ValueError):
            return None
        return point if point.shape == (3,) and np.all(np.isfinite(point)) else None

    def update(
        self,
        *,
        sequence: int,
        hole_position_m: Sequence[float] | None,
        target_position_m: Sequence[float] | None,
        hole_confidence: float = 1.0,
        target_confidence: float = 1.0,
        hole_count: int = 1,
        target_count: int = 1,
        can_move: bool = False,
    ) -> MetricAlignmentDecision:
        try:
            sequence = int(sequence)
        except (TypeError, ValueError):
            return self._decision("hold", "invalid_observation_sequence")
        if self._last_sequence is not None and sequence <= self._last_sequence:
            return self._decision("hold", "duplicate_or_old_observation")
        self._last_sequence = sequence
        if hole_count != 1 or target_count != 1:
            self.reset_observation()
            return self._decision("hold", "require_exactly_one_hole_and_target")
        try:
            confidences = (float(hole_confidence), float(target_confidence))
        except (TypeError, ValueError):
            confidences = (math.nan, math.nan)
        if not all(math.isfinite(value) and value >= self.policy.min_confidence for value in confidences):
            self.reset_observation()
            return self._decision("hold", "confidence_below_gate")
        hole = self._point(hole_position_m)
        target = self._point(target_position_m)
        if hole is None or target is None:
            self.reset_observation()
            return self._decision("hold", "missing_or_invalid_metric_pair")
        pair = np.concatenate((hole[:2], target[:2]))
        if self._last_pair is None:
            self._stable_count = 1
        elif max(float(np.linalg.norm(pair[:2] - self._last_pair[:2])), float(np.linalg.norm(pair[2:] - self._last_pair[2:]))) > self.policy.max_pair_jump_m:
            self._stable_count = 1
            self._aligned_count = 0
        else:
            self._stable_count += 1
        self._last_pair = pair
        error = target - hole
        error[2] = 0.0
        error_norm = float(np.linalg.norm(error[:2]))
        if error_norm > self.policy.max_error_m:
            self.reset_observation()
            return self._decision("hold", "metric_error_exceeds_safety_gate", error=error)
        if error_norm <= self.policy.position_tolerance_m:
            self._aligned_count += 1
            if self._aligned_count >= self.policy.aligned_frames and self._stable_count >= self.policy.stable_frames:
                self._aligned = True
                return self._decision("aligned", "metric_error_within_tolerance", error=error)
            return self._decision("waiting_alignment", "within_tolerance_but_not_yet_confirmed", error=error)
        self._aligned_count = 0
        if self._aligned:
            return self._decision("aligned", "alignment_already_confirmed", error=error)
        if self._stable_count < self.policy.stable_frames:
            return self._decision("waiting_stability", "metric_pair_not_stable_for_required_frames", error=error)
        requested = np.asarray((-error[0], -error[1], 0.0), dtype=np.float64)
        requested_distance = float(np.linalg.norm(requested[:2]))
        if requested_distance > self.policy.max_step_m:
            requested *= self.policy.max_step_m / requested_distance
            requested_distance = self.policy.max_step_m
        if not can_move:
            return self._decision("ready", "correction_available_motion_disabled_or_in_flight", error=error, requested=requested)
        if self._steps >= self.policy.max_steps:
            return self._decision("limit_hold", "maximum_step_count_reached", error=error, requested=requested)
        travel_remaining = self.policy.max_total_travel_m - self._travel_m
        radial_remaining = self.policy.max_total_displacement_m - float(np.linalg.norm(self._offset[:2]))
        if travel_remaining <= 1e-12 or radial_remaining <= 1e-12:
            return self._decision("limit_hold", "xy_motion_budget_reached", error=error, requested=requested)
        scale = min(1.0, travel_remaining / requested_distance, radial_remaining / (float(np.linalg.norm(self._offset[:2])) + requested_distance))
        accepted = requested * max(0.0, scale)
        accepted_distance = float(np.linalg.norm(accepted[:2]))
        if accepted_distance <= 1e-12:
            return self._decision("limit_hold", "remaining_xy_limit_too_small", error=error, requested=requested)
        self._travel_m += accepted_distance
        self._offset += accepted
        self._steps += 1
        return self._decision("move", "fresh_stable_metric_pair_bounded_xy_correction", error=error, requested=requested, accepted=accepted)


def metric_and_pixel_corrections_agree(
    pixel_delta_m: Sequence[float],
    metric_delta_m: Sequence[float],
    *,
    min_cosine: float = 0.35,
    min_norm_m: float = 0.00025,
    min_ratio: float = 0.20,
    max_ratio: float = 5.0,
) -> bool:
    """Return whether two non-trivial XY corrections have compatible signs."""
    pixel = np.asarray(tuple(float(value) for value in pixel_delta_m), dtype=np.float64)
    metric = np.asarray(tuple(float(value) for value in metric_delta_m), dtype=np.float64)
    if pixel.shape != (3,) or metric.shape != (3,) or not np.all(np.isfinite(pixel)) or not np.all(np.isfinite(metric)):
        return False
    pixel_norm = float(np.linalg.norm(pixel[:2]))
    metric_norm = float(np.linalg.norm(metric[:2]))
    if pixel_norm < min_norm_m or metric_norm < min_norm_m:
        return True
    cosine = float(np.dot(pixel[:2], metric[:2]) / (pixel_norm * metric_norm))
    ratio = metric_norm / pixel_norm
    return cosine >= float(min_cosine) and float(min_ratio) <= ratio <= float(max_ratio)
