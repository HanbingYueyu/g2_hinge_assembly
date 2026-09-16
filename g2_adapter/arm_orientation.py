"""Lightweight arm-orientation math shared by motion and vision scripts."""

from __future__ import annotations

import math

import numpy as np


def quaternion_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized XYZW quaternion."""
    matrix = np.asarray(rotation, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            x, y, z, w = 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            x, y, z, w = (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            x, y, z, w = (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    return tuple(float(value) for value in quaternion)


def fixed_forward_down_orientation(
    *, forward_yaw_deg: float = 0.0,
    gripper_pitch_deg: float = 30.0,
    gripper_roll_deg: float = 180.0,
) -> tuple[float, float, float, float]:
    """Build the same G2 left-end orientation used by the full workflow."""
    values = (forward_yaw_deg, gripper_pitch_deg, gripper_roll_deg)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("末端固定姿态参数必须是有限数值")
    if not -180.0 <= float(forward_yaw_deg) <= 180.0:
        raise ValueError("forward_yaw_deg must be in [-180, 180]")
    if not -89.0 <= float(gripper_pitch_deg) <= 89.0:
        raise ValueError("gripper_pitch_deg must be in [-89, 89]")
    if not -180.0 <= float(gripper_roll_deg) <= 180.0:
        raise ValueError("gripper_roll_deg must be in [-180, 180]")
    yaw = math.radians(float(forward_yaw_deg))
    pitch = math.radians(float(gripper_pitch_deg))
    roll = math.radians(float(gripper_roll_deg))
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    base_rotation = np.column_stack((
        np.array([0.0, 1.0, 0.0]),
        np.array([sin_pitch, 0.0, cos_pitch]),
        np.array([cos_pitch, 0.0, -sin_pitch]),
    ))
    cy, sy = math.cos(yaw), math.sin(yaw)
    yaw_rotation = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    cr, sr = math.cos(roll), math.sin(roll)
    local_roll = np.array(((cr, -sr, 0.0), (sr, cr, 0.0), (0.0, 0.0, 1.0)))
    return quaternion_from_rotation(yaw_rotation @ base_rotation @ local_roll)
