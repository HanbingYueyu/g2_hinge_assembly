"""URDF forward kinematics and continuous numerical pose IK for G2 arms."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


class KinematicsError(RuntimeError):
    pass


def _vector(text: str | None, default: Sequence[float]) -> np.ndarray:
    values = default if text is None else tuple(float(item) for item in text.split())
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise KinematicsError(f"invalid URDF vector: {text!r}")
    return result


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=float)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=float)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=float)
    return rz @ ry @ rx


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s, d = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        ((c+x*x*d, x*y*d-z*s, x*z*d+y*s),
         (y*x*d+z*s, c+y*y*d, y*z*d-x*s),
         (z*x*d-y*s, z*y*d+x*s, c+z*z*d)), dtype=np.float64,
    )


def _transform(rotation=None, translation=None) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the SO(3) log error in the base/world coordinate frame."""
    relative = target @ current.T
    cosine = min(1.0, max(-1.0, (float(np.trace(relative)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    skew = np.array((relative[2, 1]-relative[1, 2], relative[0, 2]-relative[2, 0], relative[1, 0]-relative[0, 1]))
    if angle < 1e-8:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(relative)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis /= np.linalg.norm(axis)
        return axis * angle
    return skew * (angle / (2.0 * math.sin(angle)))


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class IKDiagnostics:
    iterations: int
    position_error_m: float
    orientation_error_rad: float


class UrdfKinematics:
    def __init__(self, urdf_path: str | Path) -> None:
        self.path = Path(urdf_path).expanduser().resolve()
        root = ET.parse(self.path).getroot()
        self.by_child: dict[str, UrdfJoint] = {}
        for element in root.findall("joint"):
            origin, axis = element.find("origin"), element.find("axis")
            joint = UrdfJoint(
                str(element.attrib["name"]), str(element.attrib["type"]),
                str(element.find("parent").attrib["link"]), str(element.find("child").attrib["link"]),
                _vector(None if origin is None else origin.get("xyz"), (0, 0, 0)),
                _vector(None if origin is None else origin.get("rpy"), (0, 0, 0)),
                _vector(None if axis is None else axis.get("xyz"), (1, 0, 0)),
            )
            self.by_child[joint.child] = joint

    def chain(self, base_link: str, tip_link: str) -> tuple[UrdfJoint, ...]:
        reverse, link = [], tip_link
        while link != base_link:
            try:
                joint = self.by_child[link]
            except KeyError as exc:
                raise KinematicsError(f"{tip_link} is not below {base_link}") from exc
            reverse.append(joint)
            link = joint.parent
        return tuple(reversed(reverse))

    def forward(self, base_link: str, tip_link: str, joint_positions: Mapping[str, float]) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        for joint in self.chain(base_link, tip_link):
            transform = transform @ _transform(_rpy_matrix(joint.rpy), joint.xyz)
            value = float(joint_positions.get(joint.name, 0.0))
            if joint.kind in {"revolute", "continuous"}:
                transform = transform @ _transform(_axis_angle(joint.axis, value))
            elif joint.kind == "prismatic":
                transform = transform @ _transform(translation=joint.axis * value)
        return transform

    def jacobian(self, base_link: str, tip_link: str, joint_positions: Mapping[str, float], active_joint_names: Sequence[str]) -> np.ndarray:
        active = set(active_joint_names)
        transform, columns = np.eye(4, dtype=np.float64), {}
        for joint in self.chain(base_link, tip_link):
            transform = transform @ _transform(_rpy_matrix(joint.rpy), joint.xyz)
            if joint.name in active:
                columns[joint.name] = (joint.kind, transform[:3, 3].copy(), transform[:3, :3] @ joint.axis)
            value = float(joint_positions.get(joint.name, 0.0))
            if joint.kind in {"revolute", "continuous"}:
                transform = transform @ _transform(_axis_angle(joint.axis, value))
            elif joint.kind == "prismatic":
                transform = transform @ _transform(translation=joint.axis * value)
        tip = transform[:3, 3]
        result = np.zeros((6, len(active_joint_names)), dtype=np.float64)
        for index, name in enumerate(active_joint_names):
            kind, origin, axis = columns[name]
            if kind in {"revolute", "continuous"}:
                result[:3, index], result[3:, index] = np.cross(axis, tip-origin), axis
            else:
                result[:3, index] = axis
        return result

    def solve_pose(
        self, base_link: str, tip_link: str, fixed_joint_positions: Mapping[str, float],
        active_joint_names: Sequence[str], target: np.ndarray, initial: Sequence[float], *,
        reference: Sequence[float] | None = None, joint_limits: Sequence[tuple[float, float]],
        damping: float = 0.02, reference_weight: float = 1e-4,
        max_joint_step: float = 0.05, max_iterations: int = 250,
        position_tolerance_m: float = 5e-5, orientation_tolerance_rad: float = 5e-4,
    ) -> tuple[np.ndarray, IKDiagnostics]:
        names = tuple(active_joint_names)
        q = np.asarray(initial, dtype=np.float64).copy()
        ref = q.copy() if reference is None else np.asarray(reference, dtype=np.float64)
        limits = np.asarray(joint_limits, dtype=np.float64)
        if q.shape != (len(names),) or ref.shape != q.shape or limits.shape != (len(names), 2):
            raise KinematicsError("IK dimensions do not match active joints")
        if np.asarray(target).shape != (4, 4):
            raise KinematicsError("target pose must be a 4x4 transform")
        for iteration in range(1, max_iterations + 1):
            state = {**fixed_joint_positions, **dict(zip(names, q))}
            actual = self.forward(base_link, tip_link, state)
            position_error = target[:3, 3] - actual[:3, 3]
            orientation_error = rotation_error(target[:3, :3], actual[:3, :3])
            position_norm, orientation_norm = float(np.linalg.norm(position_error)), float(np.linalg.norm(orientation_error))
            if position_norm <= position_tolerance_m and orientation_norm <= orientation_tolerance_rad:
                return q, IKDiagnostics(iteration-1, position_norm, orientation_norm)
            jacobian = self.jacobian(base_link, tip_link, state, names)
            error = np.concatenate((position_error, orientation_error))
            lhs = jacobian.T @ jacobian + (damping*damping + reference_weight) * np.eye(len(names))
            rhs = jacobian.T @ error + reference_weight * (ref - q)
            try:
                delta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError as exc:
                raise KinematicsError("IK linear solve failed") from exc
            scale = max(1.0, float(np.max(np.abs(delta))) / max_joint_step)
            q = np.clip(q + delta/scale, limits[:, 0], limits[:, 1])
        raise KinematicsError(
            f"pose IK did not converge: position_error={position_norm:.6f}m orientation_error={orientation_norm:.6f}rad"
        )
