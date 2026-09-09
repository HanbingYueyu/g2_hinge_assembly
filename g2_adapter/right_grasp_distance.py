"""Robust fixed-workpiece pose averaging in the robot base frame."""
import numpy as np


def workpiece_moved(reference, current, translation_m, rotation_deg):
    if not np.isfinite([translation_m, rotation_deg]).all() or min(translation_m, rotation_deg) <= 0:
        raise ValueError("movement thresholds must be finite and positive")
    reference, current = np.asarray(reference), np.asarray(current)
    displacement = float(np.linalg.norm(current[:3, 3] - reference[:3, 3]))
    relative = current[:3, :3] @ reference[:3, :3].T
    angle = float(np.rad2deg(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))
    return displacement > translation_m or angle > rotation_deg, displacement, angle


def robust_pose_mean(poses):
    if len(poses) != 3:
        raise ValueError("exactly 3 fresh poses required")
    matrices = np.asarray(poses, dtype=float)
    if matrices.shape != (3, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError("invalid pose samples")
    centre = np.median(matrices[:, :3, 3], axis=0)
    distances = np.linalg.norm(matrices[:, :3, 3] - centre, axis=1)
    keep = np.argsort(distances)[:3]
    selected = matrices[keep]
    if distances[keep].max() > .005:
        raise ValueError("workpiece position samples not stable within 5 mm")
    u, _, vh = np.linalg.svd(selected[:, :3, :3].sum(axis=0))
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vh)]) @ vh
    angles = np.arccos(np.clip((np.einsum('nij,ij->n', selected[:, :3, :3], rotation) - 1) / 2, -1, 1))
    if angles.max() > np.deg2rad(3):
        raise ValueError("workpiece orientation samples not stable within 3 degrees")
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = selected[:, :3, 3].mean(axis=0)
    return result
