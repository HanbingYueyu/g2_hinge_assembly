"""Pure force-contact detection primitives for the left G2 grasp flow."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ForceContactParameters:
    frame_name: str
    force_delta_threshold_n: float
    consecutive_samples: int
    baseline_samples: int
    sample_period_s: float
    require_for_grasp: bool
    relief_max_lift_m: float
    relief_force_delta_threshold_n: float
    relief_consecutive_samples: int
    # Extra upward clearance commanded after the force has been released.
    # 0 disables it; the amount is applied on top of the relief lift.
    relief_extra_lift_m: float = 0.0

    def validate(self) -> None:
        if not self.frame_name:
            raise ValueError("contact.frame_name must be non-empty")
        if not math.isfinite(self.force_delta_threshold_n) or self.force_delta_threshold_n <= 0.0:
            raise ValueError("contact.force_delta_threshold_n must be positive and finite")
        if self.consecutive_samples < 1:
            raise ValueError("contact.consecutive_samples must be at least 1")
        if self.baseline_samples < 3:
            raise ValueError("contact.baseline_samples must be at least 3")
        if not math.isfinite(self.sample_period_s) or not 0.01 <= self.sample_period_s <= 0.10:
            raise ValueError("contact.sample_period_s must be in [0.01, 0.10]")
        if not math.isfinite(self.relief_max_lift_m) or not 0.0 < self.relief_max_lift_m <= 0.02:
            raise ValueError("contact.relief_max_lift_m must be in (0, 0.02]")
        if not math.isfinite(self.relief_force_delta_threshold_n) or self.relief_force_delta_threshold_n <= 0.0:
            raise ValueError("contact.relief_force_delta_threshold_n must be positive and finite")
        if self.relief_consecutive_samples < 1:
            raise ValueError("contact.relief_consecutive_samples must be at least 1")
        if not math.isfinite(self.relief_extra_lift_m) or not 0.0 <= self.relief_extra_lift_m <= 0.02:
            raise ValueError("contact.relief_extra_lift_m must be in [0, 0.02]")


class ForceContactDetector:
    """Require consecutive force-magnitude excursions from a fresh baseline."""

    def __init__(self, parameters: ForceContactParameters, baseline_force_n) -> None:
        parameters.validate()
        baseline = np.asarray(baseline_force_n, dtype=np.float64)
        if baseline.shape != (3,) or not np.all(np.isfinite(baseline)):
            raise ValueError("force-contact baseline must be three finite values")
        self.parameters = parameters
        self.baseline_force_n = baseline
        self.consecutive_hits = 0

    def update(self, force_n) -> tuple[bool, float, int]:
        force = np.asarray(force_n, dtype=np.float64)
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise ValueError("force-contact sample must be three finite values")
        delta_force_n = float(np.linalg.norm(force - self.baseline_force_n))
        if delta_force_n >= self.parameters.force_delta_threshold_n:
            self.consecutive_hits += 1
        else:
            self.consecutive_hits = 0
        return (
            self.consecutive_hits >= self.parameters.consecutive_samples,
            delta_force_n,
            self.consecutive_hits,
        )
