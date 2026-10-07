"""Frame-safe, horizontal visual-servo math for the G2 grasp loop."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class NearContactForceHold:
    """Hysteresis for waiting at a measured hold pose near force contact."""

    def __init__(self, *, pause_delta_n: float, release_delta_n: float):
        if (not math.isfinite(pause_delta_n) or pause_delta_n <= 0
                or not math.isfinite(release_delta_n)
                or not 0 <= release_delta_n < pause_delta_n):
            raise ValueError("near-contact force hold thresholds must be ordered and finite")
        self.pause_delta_n = float(pause_delta_n)
        self.release_delta_n = float(release_delta_n)
        self.holding = False

    def update(self, force_delta_n: float) -> bool:
        if not math.isfinite(force_delta_n):
            self.holding = True
        elif self.holding:
            self.holding = force_delta_n > self.release_delta_n
        else:
            self.holding = force_delta_n >= self.pause_delta_n
        return self.holding


class LargeDescentXYErrorGuard:
    """Pause descent for confirmed large errors until XY is aligned again."""

    def __init__(self, *, threshold_m: float = 0.010, resume_tolerance_m: float = 0.005,
                 consecutive_frames: int = 2):
        if (not math.isfinite(threshold_m) or threshold_m <= 0
                or not math.isfinite(resume_tolerance_m)
                or not 0 < resume_tolerance_m < threshold_m
                or consecutive_frames < 1):
            raise ValueError("large descent XY error guard needs a positive threshold and frame count")
        self.threshold_m = float(threshold_m)
        self.resume_tolerance_m = float(resume_tolerance_m)
        self.consecutive_frames = int(consecutive_frames)
        self.large_frames = 0
        self.xy_only = False

    def update(self, error_m: float | None) -> bool:
        if error_m is None or not math.isfinite(error_m):
            self.large_frames = 0
            return self.xy_only
        if self.xy_only:
            if error_m <= self.resume_tolerance_m:
                self.xy_only = False
                self.large_frames = 0
            return self.xy_only
        self.large_frames = self.large_frames + 1 if error_m > self.threshold_m else 0
        if self.large_frames >= self.consecutive_frames:
            self.xy_only = True
        return self.xy_only


def anchored_descent_xy_target(requested, actual, delta, *, max_lookahead_m=0.005):
    """Advance the existing XY goal without rebasing it onto measured drift."""
    goal = np.asarray(requested, dtype=np.float64).copy()
    actual = np.asarray(actual, dtype=np.float64)
    delta = np.asarray(delta, dtype=np.float64)
    # Impedance tracking leaves a steady residual: do not require sub-mm
    # arrival before advancing. Keep the goal anchored and limit lookahead.
    proposed = goal[:2] + delta[:2]
    if np.linalg.norm(proposed - actual[:2]) <= max_lookahead_m:
        goal[:2] = proposed
    return tuple(goal)


class ReverseXYMotionGuard:
    """Detect measured travel opposite a persistent requested XY direction."""

    def __init__(self):
        self.started = None
        self.origin = None
        self.direction = None

    def update(self, now, actual, requested):
        actual = np.asarray(actual, dtype=np.float64)[:2]
        pending = np.asarray(requested, dtype=np.float64)[:2] - actual
        distance = float(np.linalg.norm(pending))
        if distance < 0.001:
            self.started = None
            return False
        direction = pending / distance
        if self.started is None or float(np.dot(direction, self.direction)) < 0.5:
            self.started, self.origin, self.direction = now, actual.copy(), direction
            return False
        if now - self.started < 0.8:
            return False
        reverse = float(np.dot(actual - self.origin, self.direction)) < -0.003
        self.started, self.origin, self.direction = now, actual.copy(), direction
        return reverse


def relief_translation_speed_mps(automatic_speed_mps: float) -> float:
    """Return the unloading-lift speed used after force contact.

    The automatic visual/force motion speed is the single operator-facing
    setting.  Unloading must be gentler, so it runs at exactly half that
    speed; callers restore the automatic speed after the lift attempt.
    """
    speed = float(automatic_speed_mps)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError("automatic translation speed must be positive and finite")
    return speed * 0.5


def base_position_after_delta(
    end_link_position_m: Sequence[float], base_delta_m: Sequence[float]
) -> tuple[float, float, float]:
    """Return the next ``base_link`` end-link target after a bounded delta."""
    position = np.asarray(end_link_position_m, dtype=np.float64)
    delta = np.asarray(base_delta_m, dtype=np.float64)
    if position.shape != (3,) or delta.shape != (3,):
        raise ValueError("end-link position and base delta must be 3-vectors")
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(delta)):
        raise ValueError("end-link position and base delta must be finite")
    return tuple(float(value) for value in position + delta)


def empirical_horizontal_base_delta_from_pixel_error(
    pixel_error_px: Sequence[float],
    pixel_jacobian_px_per_m: Sequence[Sequence[float]],
    *,
    max_step_m: float,
) -> np.ndarray:
    """Return bounded base XY correction from a measured image Jacobian.

    The Jacobian models ``pixel_delta = J @ base_delta``.  Therefore an
    observed error ``current_pixel - target_pixel`` is corrected by solving
    ``J @ base_delta = -pixel_error``.  The output is always horizontal.
    """
    error = np.asarray(pixel_error_px, dtype=np.float64)
    jacobian = np.asarray(pixel_jacobian_px_per_m, dtype=np.float64)
    if error.shape != (2,) or jacobian.shape != (2, 2):
        raise ValueError("pixel error must be a 2-vector and Jacobian must be 2x2")
    if not np.all(np.isfinite(error)) or not np.all(np.isfinite(jacobian)) or not math.isfinite(max_step_m):
        raise ValueError("empirical visual-servo inputs must be finite")
    if max_step_m <= 0.0:
        raise ValueError("max_step_m must be positive")
    if np.linalg.matrix_rank(jacobian) < 2:
        raise ValueError("empirical pixel Jacobian is singular")
    horizontal = np.linalg.solve(jacobian, -error)
    result = np.asarray((horizontal[0], horizontal[1], 0.0), dtype=np.float64)
    magnitude = float(np.linalg.norm(result[:2]))
    if magnitude > max_step_m:
        result *= max_step_m / magnitude
    return result


def horizontal_base_delta_from_camera_displacement(
    camera_displacement_m: Sequence[float],
    point_camera_m: Sequence[float],
    *,
    fx: float,
    fy: float,
    base_from_camera_rotation: np.ndarray,
    max_step_m: float,
) -> np.ndarray:
    """Approximate an image correction using base-link horizontal translation.

    ``camera_displacement_m`` is the bounded camera translation that would
    correct the image error if arbitrary 3-D motion were allowed.  We convert
    its desired pixel change through the pinhole Jacobian, solve only for the
    base X/Y components, and set base Z exactly to zero.  Thus image alignment
    cannot begin the vertical descent phase by accident.
    """
    camera_displacement = np.asarray(camera_displacement_m, dtype=np.float64)
    point_camera = np.asarray(point_camera_m, dtype=np.float64)
    rotation = np.asarray(base_from_camera_rotation, dtype=np.float64)
    if camera_displacement.shape != (3,) or point_camera.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("visual-servo inputs must be 3-vectors and a 3x3 rotation")
    if not all(math.isfinite(value) for value in (*camera_displacement, *point_camera, float(fx), float(fy), float(max_step_m))):
        raise ValueError("visual-servo inputs must be finite")
    if point_camera[2] <= 0.0 or fx <= 0.0 or fy <= 0.0 or max_step_m <= 0.0:
        raise ValueError("camera depth, focal lengths, and max step must be positive")

    x, y, z = point_camera
    image_from_camera_point = np.asarray(
        ((fx / z, 0.0, -fx * x / (z * z)), (0.0, fy / z, -fy * y / (z * z))),
        dtype=np.float64,
    )
    # A base translation moves the camera by the same base vector; the static
    # object's coordinates in the camera frame move in the opposite direction.
    camera_from_base = rotation.T
    image_from_base_xy = image_from_camera_point @ (-camera_from_base[:, :2])
    desired_image_delta = image_from_camera_point @ (-camera_displacement)
    if np.linalg.matrix_rank(image_from_base_xy) < 2:
        raise ValueError("camera projection is singular for horizontal visual servo")
    horizontal = np.linalg.solve(image_from_base_xy, desired_image_delta)
    result = np.asarray((horizontal[0], horizontal[1], 0.0), dtype=np.float64)
    magnitude = float(np.linalg.norm(result[:2]))
    if magnitude > max_step_m:
        result *= max_step_m / magnitude
    return result
