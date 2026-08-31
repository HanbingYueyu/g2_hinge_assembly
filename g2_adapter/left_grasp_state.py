"""Pure state machine for the final left-hand G2 visual grasp.

The state machine has no GDK, SAM3, camera, or robot imports.  It converts a
fresh, tracked RGB-D observation into a small camera-frame XY correction,
then requests one force-guarded base-frame descent.  Image depth is never a
condition for continuing vertical motion: the runtime's relative-force guard
is the primary stop condition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


def requires_calibrated_config(*, allow_motion: bool, stop_after: str) -> bool:
    """Only a complete sequence may close the gripper and require approval."""
    return bool(allow_motion and stop_after == "none")


class GraspPhase(str, Enum):
    ALIGN_XY = "align_xy"
    APPROACH_DEPTH = "approach_depth"
    FINAL_DESCENT = "final_descent"
    GRASP = "grasp"
    GRASPED = "grasped"


@dataclass(frozen=True)
class GraspParameters:
    pixel_tolerance: float
    center_confirmations: int
    max_xy_step_m: float
    target_u_px: float
    target_v_px: float
    # Retained solely for schema compatibility with earlier captured configs.
    # Do not use image depth to decide whether to move closer to the workpiece.
    pre_descent_depth_m: float
    pre_descent_confirmations: int
    pre_descent_step_m: float
    pre_descent_duration_s: float
    final_descent_speed_mps: float
    final_descent_duration_s: float
    final_descent_completion_tolerance_m: float = 0.0005


@dataclass(frozen=True)
class Observation:
    center_px: tuple[float, float]
    depth_m: float
    fx: float
    fy: float
    ppx: float
    ppy: float


@dataclass(frozen=True)
class Step:
    phase: GraspPhase
    camera_delta_m: tuple[float, float, float] | None
    base_delta_m: tuple[float, float, float] | None = None
    motion_duration_s: float | None = None
    is_final_descent: bool = False
    should_close_gripper: bool = False
    reason: str = ""


class LeftGraspState:
    """Preserve the OpenArm sequence without assuming its hardware values."""

    def __init__(self, parameters: GraspParameters) -> None:
        self.parameters = parameters
        self.phase = GraspPhase.ALIGN_XY
        self.centered_frames = 0

    @staticmethod
    def _bounded(vector: np.ndarray, limit: float) -> np.ndarray:
        magnitude = float(np.linalg.norm(vector))
        if magnitude <= limit or magnitude == 0.0:
            return vector
        return vector * (limit / magnitude)

    def update(self, observation: Observation, *, now: float) -> Step:
        p = self.parameters
        if not all(math.isfinite(value) for value in (*observation.center_px, observation.depth_m, observation.fx, observation.fy, observation.ppx, observation.ppy)):
            return Step(self.phase, None, reason="non_finite_observation")
        if observation.depth_m <= 0.0 or observation.fx <= 0.0 or observation.fy <= 0.0:
            return Step(self.phase, None, reason="invalid_depth_or_intrinsics")
        if self.phase == GraspPhase.GRASPED:
            return Step(self.phase, None, reason="terminal_hold")

        if self.phase == GraspPhase.ALIGN_XY:
            pixel_error = np.asarray(
                (observation.center_px[0] - p.target_u_px, observation.center_px[1] - p.target_v_px),
                dtype=np.float64,
            )
            if float(np.max(np.abs(pixel_error))) <= p.pixel_tolerance:
                self.centered_frames += 1
                if self.centered_frames >= p.center_confirmations:
                    self.phase = GraspPhase.FINAL_DESCENT
                    return Step(self.phase, None, reason="xy_centered_force_primary_descent")
                return Step(self.phase, None, reason="xy_center_confirmation")
            self.centered_frames = 0
            # Moving the wrist camera by this vector makes the observed
            # object move towards its optical centre.  The runtime maps this
            # through the empirically measured pixel-to-base Jacobian.
            delta = np.asarray(
                (
                    pixel_error[0] * observation.depth_m / observation.fx,
                    pixel_error[1] * observation.depth_m / observation.fy,
                    0.0,
                ),
                dtype=np.float64,
            )
            delta = self._bounded(delta, p.max_xy_step_m)
            return Step(self.phase, tuple(float(value) for value in delta), reason="xy_correction")

        if self.phase == GraspPhase.FINAL_DESCENT:
            # G2 base_link uses Z-up, so a negative Z increment is a physical
            # vertical descent.  This is one fixed, timed trajectory rather
            # than a camera-frame-rate-dependent series of steps.
            distance_m = p.final_descent_speed_mps * p.final_descent_duration_s
            return Step(
                self.phase,
                None,
                base_delta_m=(0.0, 0.0, -distance_m),
                motion_duration_s=p.final_descent_duration_s,
                is_final_descent=True,
                reason="force_guarded_final_descent",
            )

        return Step(self.phase, None, reason="unexpected_phase")

    def confirm_final_descent(self) -> Step:
        """Permit gripper closure only after the timed descent command returns."""
        if self.phase != GraspPhase.FINAL_DESCENT:
            return Step(self.phase, None, reason="final_descent_not_pending")
        self.phase = GraspPhase.GRASP
        return Step(self.phase, None, should_close_gripper=True, reason="timed_final_descent_complete")

    def confirm_grasp(self) -> Step:
        if self.phase != GraspPhase.GRASP:
            return Step(self.phase, None, reason="grasp_not_pending")
        self.phase = GraspPhase.GRASPED
        return Step(self.phase, None, reason="gripper_command_sent")

    def reset_alignment(self) -> None:
        """Forget a changed target before any further descent can occur."""
        if self.phase == GraspPhase.GRASPED:
            return
        self.phase = GraspPhase.ALIGN_XY
        self.centered_frames = 0

    def reset_for_regrasp(self) -> None:
        """Start a new visual alignment after a confirmed failed attempt.

        Unlike ``reset_alignment`` this is deliberately valid after GRASPED:
        the caller has already retracted and confirmed that the Omnipicker is
        open, so it is safe to leave the terminal closure phase and reacquire
        the next visible instance.
        """
        self.phase = GraspPhase.ALIGN_XY
        self.centered_frames = 0
