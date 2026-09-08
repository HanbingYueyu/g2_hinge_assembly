"""Metric XY correction with an explicitly approximate fixed gripper TCP."""
import numpy as np


def xy_error(base_from_end, end_from_camera, point_camera, tcp_end, compensation_xy):
    base_from_end = np.asarray(base_from_end, dtype=float)
    end_from_camera = np.asarray(end_from_camera, dtype=float)
    target = (base_from_end @ end_from_camera @ np.r_[point_camera, 1.0])[:3]
    tcp = (base_from_end @ np.r_[tcp_end, 1.0])[:3]
    error = target[:2] + np.asarray(compensation_xy) - tcp[:2]
    if not np.isfinite(error).all():
        raise ValueError("non-finite metric XY error")
    return error.tolist()


def bounded_xy(error, limit):
    vector = np.r_[error, 0.0].astype(float)
    size = np.linalg.norm(vector)
    return vector * min(1.0, limit / size) if size else vector
