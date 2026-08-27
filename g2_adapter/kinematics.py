"""Small URDF FK/Jacobian layer used instead of MoveIt."""

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
    norm = float(np.linalg.norm(axis))
    if norm <= 0.0:
        raise KinematicsError("URDF revolute axis has zero length")
    x, y, z = axis / norm
    c, s, one_c = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array(
        (
            (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
            (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
            (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
        ),
        dtype=float,
    )


def _transform(rotation: np.ndarray | None = None, translation: np.ndarray | None = None):
    result = np.eye(4, dtype=np.float64)
    if rotation is not None:
        result[:3, :3] = rotation
    if translation is not None:
        result[:3, 3] = translation
    return result


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray


class UrdfKinematics:
    def __init__(self, urdf_path: str | Path) -> None:
        self.path = Path(urdf_path).expanduser().resolve()
        root = ET.parse(self.path).getroot()
        self.by_child: dict[str, UrdfJoint] = {}
        for element in root.findall("joint"):
            origin = element.find("origin")
            axis = element.find("axis")
            joint = UrdfJoint(
                name=str(element.attrib["name"]),
                kind=str(element.attrib["type"]),
                parent=str(element.find("parent").attrib["link"]),
                child=str(element.find("child").attrib["link"]),
                xyz=_vector(None if origin is None else origin.get("xyz"), (0, 0, 0)),
                rpy=_vector(None if origin is None else origin.get("rpy"), (0, 0, 0)),
                axis=_vector(None if axis is None else axis.get("xyz"), (1, 0, 0)),
            )
            if joint.child in self.by_child:
                raise KinematicsError(f"multiple URDF parents for {joint.child}")
            self.by_child[joint.child] = joint

    def chain(self, base_link: str, tip_link: str) -> tuple[UrdfJoint, ...]:
        reverse = []
        link = tip_link
        while link != base_link:
            try:
                joint = self.by_child[link]
            except KeyError as exc:
                raise KinematicsError(f"{tip_link} is not below {base_link}") from exc
            reverse.append(joint)
            link = joint.parent
        return tuple(reversed(reverse))

    def forward(
        self,
        base_link: str,
        tip_link: str,
        joint_positions: Mapping[str, float],
    ) -> np.ndarray:
        transform = np.eye(4, dtype=np.float64)
        for joint in self.chain(base_link, tip_link):
            transform = transform @ _transform(_rpy_matrix(joint.rpy), joint.xyz)
            if joint.kind in {"revolute", "continuous"}:
                angle = float(joint_positions.get(joint.name, 0.0))
                if not math.isfinite(angle):
                    raise KinematicsError(f"non-finite joint position: {joint.name}")
                transform = transform @ _transform(_axis_angle(joint.axis, angle))
            elif joint.kind == "prismatic":
                distance = float(joint_positions.get(joint.name, 0.0))
                transform = transform @ _transform(translation=joint.axis * distance)
        return transform

    def jacobian(
        self,
        base_link: str,
        tip_link: str,
        joint_positions: Mapping[str, float],
        active_joint_names: Sequence[str],
    ) -> np.ndarray:
        active = set(active_joint_names)
        transform = np.eye(4, dtype=np.float64)
        columns = {}
        chain = self.chain(base_link, tip_link)
        for joint in chain:
            transform = transform @ _transform(_rpy_matrix(joint.rpy), joint.xyz)
            if joint.name in active:
                axis_world = transform[:3, :3] @ joint.axis
                columns[joint.name] = (joint.kind, transform[:3, 3].copy(), axis_world)
            if joint.kind in {"revolute", "continuous"}:
                transform = transform @ _transform(
                    _axis_angle(joint.axis, float(joint_positions.get(joint.name, 0.0)))
                )
            elif joint.kind == "prismatic":
                transform = transform @ _transform(
                    translation=joint.axis * float(joint_positions.get(joint.name, 0.0))
                )
        tip = transform[:3, 3]
        jacobian = np.zeros((6, len(active_joint_names)), dtype=np.float64)
        for index, name in enumerate(active_joint_names):
            if name not in columns:
                raise KinematicsError(f"active joint {name} is not in requested chain")
            kind, origin, axis = columns[name]
            if kind in {"revolute", "continuous"}:
                jacobian[:3, index] = np.cross(axis, tip - origin)
                jacobian[3:, index] = axis
            elif kind == "prismatic":
                jacobian[:3, index] = axis
        return jacobian

    def damped_joint_delta(
        self,
        base_link: str,
        tip_link: str,
        joint_positions: Mapping[str, float],
        active_joint_names: Sequence[str],
        cartesian_delta: Sequence[float],
        *,
        damping: float = 0.03,
        max_joint_delta: float = 0.01,
    ) -> np.ndarray:
        delta = np.asarray(cartesian_delta, dtype=np.float64)
        if delta.shape not in {(3,), (6,)} or not np.all(np.isfinite(delta)):
            raise KinematicsError("cartesian_delta must contain 3 or 6 finite values")
        jacobian = self.jacobian(
            base_link, tip_link, joint_positions, active_joint_names
        )[: len(delta), :]
        system = jacobian @ jacobian.T + (float(damping) ** 2) * np.eye(len(delta))
        result = jacobian.T @ np.linalg.solve(system, delta)
        return np.clip(result, -abs(max_joint_delta), abs(max_joint_delta))

