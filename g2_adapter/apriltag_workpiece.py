"""ROS-free AprilTag workpiece pose and G2 right-grasp target geometry.

The default CAD geometry is loaded from the electric screwdriver profile.
G2-specific tool geometry is intentionally supplied by a
calibration file; an OpenArm TCP transform must never be assumed for G2.
"""

from __future__ import annotations

import math
import json
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


REQUIRED_TAG_IDS = frozenset((0, 1, 2))


class AprilTagGeometryError(ValueError):
    pass


def _profile_vector(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise AprilTagGeometryError(f"{name} must be a finite 3-vector")
    return vector.copy()


@dataclass(frozen=True)
class TagGeometry:
    """One physical workpiece's AprilTag object-point geometry.

    ``tag_size_m`` is the effective black-corner size used by PnP.  The
    profile defines the physical layout independently of the grasp calibration.
    """

    tag_size_m: float
    centers_m: Mapping[int, np.ndarray]
    u: np.ndarray
    v: Mapping[int, np.ndarray]

    @classmethod
    def from_mapping(cls, value: Mapping[str, object], name: str = "tag_geometry") -> "TagGeometry":
        if not isinstance(value, Mapping):
            raise AprilTagGeometryError(f"{name} must be a JSON object")
        family = value.get("tag_family")
        if family is not None and str(family) != "tagStandard41h12":
            raise AprilTagGeometryError(f"{name}.tag_family must be tagStandard41h12")
        raw_size = value.get("tag_size_m", value.get("tag_corner_size_m"))
        try:
            tag_size_m = float(raw_size)
        except (TypeError, ValueError) as exc:
            raise AprilTagGeometryError(f"{name}.tag_size_m must be positive") from exc
        if not math.isfinite(tag_size_m) or tag_size_m <= 0.0:
            raise AprilTagGeometryError(f"{name}.tag_size_m must be positive")
        raw_centers = value.get("centers_m")
        raw_v = value.get("v")
        if not isinstance(raw_centers, Mapping) or not isinstance(raw_v, Mapping):
            raise AprilTagGeometryError(f"{name} requires centers_m and v mappings")
        try:
            center_ids = {int(tag_id) for tag_id in raw_centers}
            v_ids = {int(tag_id) for tag_id in raw_v}
        except (TypeError, ValueError) as exc:
            raise AprilTagGeometryError(f"{name} tag ids must be integers") from exc
        if center_ids != set(REQUIRED_TAG_IDS) or v_ids != set(REQUIRED_TAG_IDS):
            raise AprilTagGeometryError(f"{name} must define exactly tag ids 0, 1, 2")
        centers: dict[int, np.ndarray] = {}
        verticals: dict[int, np.ndarray] = {}
        for tag_id in sorted(REQUIRED_TAG_IDS):
            center_value = raw_centers.get(str(tag_id), raw_centers.get(tag_id))
            vertical_value = raw_v.get(str(tag_id), raw_v.get(tag_id))
            centers[tag_id] = _profile_vector(center_value, f"{name}.centers_m[{tag_id}]")
            verticals[tag_id] = _profile_vector(vertical_value, f"{name}.v[{tag_id}]")
        horizontal = _profile_vector(value.get("u"), f"{name}.u")
        if not math.isclose(float(np.linalg.norm(horizontal)), 1.0, abs_tol=1e-4):
            raise AprilTagGeometryError(f"{name}.u must be unit length")
        for tag_id, vertical in verticals.items():
            if not math.isclose(float(np.linalg.norm(vertical)), 1.0, abs_tol=1e-4):
                raise AprilTagGeometryError(f"{name}.v[{tag_id}] must be unit length")
            if abs(float(np.dot(horizontal, vertical))) > 1e-4:
                raise AprilTagGeometryError(f"{name}.u and v[{tag_id}] must be orthogonal")
        return cls(tag_size_m, centers, horizontal, verticals)

    def object_corners(self, tag_id: int) -> np.ndarray:
        if tag_id not in REQUIRED_TAG_IDS:
            raise AprilTagGeometryError(f"unsupported workpiece tag id: {tag_id}")
        half = self.tag_size_m / 2.0
        center, horizontal, vertical = self.centers_m[tag_id], self.u, self.v[tag_id]
        return np.asarray((
            center - half * horizontal - half * vertical,
            center + half * horizontal - half * vertical,
            center + half * horizontal + half * vertical,
            center - half * horizontal + half * vertical,
        ), dtype=np.float64)

    @property
    def object_points(self) -> dict[int, np.ndarray]:
        return {tag_id: self.object_corners(tag_id) for tag_id in sorted(REQUIRED_TAG_IDS)}


DEFAULT_TAG_GEOMETRY = TagGeometry.from_mapping(
    json.loads(Path(__file__).with_name("tag_geometry_electric_screwdriver.json").read_text(encoding="utf-8"))
)

# Public aliases use the same electric screwdriver geometry.
TAG_SIZE_M = DEFAULT_TAG_GEOMETRY.tag_size_m
TAG_CENTERS_M = DEFAULT_TAG_GEOMETRY.centers_m
TAG_U = DEFAULT_TAG_GEOMETRY.u
TAG_V = DEFAULT_TAG_GEOMETRY.v


@dataclass(frozen=True)
class WorkpiecePose:
    """``T_camera_workpiece`` estimated with the color camera intrinsics."""

    camera_from_workpiece: np.ndarray
    mean_reprojection_error_px: float
    max_reprojection_error_px: float
    inlier_count: int
    sample_counts: dict[int, int]
    point_diagnostics: tuple = ()
    expected_tag_corners: dict | None = None


@dataclass(frozen=True)
class GraspTargets:
    """Right ``arm_r_end_link`` targets, all in ``base_link``."""

    base_from_workpiece: np.ndarray
    base_from_pregrasp_end: np.ndarray
    base_from_grasp_end: np.ndarray


def tag_object_corners(tag_id: int, geometry: TagGeometry | None = None) -> np.ndarray:
    """Corners ordered BL, BR, TR, TL: the OpenArm detector's convention."""
    return (DEFAULT_TAG_GEOMETRY if geometry is None else geometry).object_corners(tag_id)


TAG_OBJECT_POINTS = {tag_id: tag_object_corners(tag_id) for tag_id in REQUIRED_TAG_IDS}
# AprilTag libraries do not all expose their four corners with the same first
# corner/direction.  Evaluate every cyclic ordering and its mirror globally;
# all three tags must share one convention.  The chosen convention is logged
# and must win decisively, so this is a detector-API compatibility check, not
# an attempt to hide an ambiguous pose.
CORNER_ORDER_CANDIDATES = {
    "forward_0": (0, 1, 2, 3), "forward_1": (1, 2, 3, 0),
    "forward_2": (2, 3, 0, 1), "forward_3": (3, 0, 1, 2),
    "reverse_0": (0, 3, 2, 1), "reverse_1": (1, 0, 3, 2),
    "reverse_2": (2, 1, 0, 3), "reverse_3": (3, 2, 1, 0),
}


def as_transform(value: Mapping[str, Sequence[float]] | Sequence[Sequence[float]], name: str) -> np.ndarray:
    """Parse a finite homogeneous transform from a JSON-friendly input."""
    if isinstance(value, Mapping):
        try:
            matrix = np.asarray(value["matrix"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise AprilTagGeometryError(f"{name} must contain a 4x4 matrix") from exc
    else:
        matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise AprilTagGeometryError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-8):
        raise AprilTagGeometryError(f"{name} last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) <= 0.0:
        raise AprilTagGeometryError(f"{name} rotation must be a proper orthonormal matrix")
    return matrix.copy()


def transform_from_dict(value: Mapping[str, Sequence[float]]) -> np.ndarray:
    """Build a matrix from the GDK ``translation_m``/``rotation_xyzw`` form."""
    try:
        translation = np.asarray(value["translation_m"], dtype=np.float64)
        x, y, z, w = (float(item) for item in value["rotation_xyzw"])
    except (KeyError, TypeError, ValueError) as exc:
        raise AprilTagGeometryError("GDK transform is malformed") from exc
    norm = math.sqrt(x * x + y * y + z * z + w * w)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)) or norm <= 1e-12:
        raise AprilTagGeometryError("GDK transform contains invalid values")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = ((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
                      (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
                      (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)))
    matrix[:3, 3] = translation
    return matrix


def transform_dict_from_matrix(matrix: np.ndarray) -> dict[str, list[float]]:
    """Serialize a validated SE(3) matrix in the GDK transform-dict form."""
    transform = as_transform(matrix, "transform")
    return {
        "translation_m": [float(item) for item in transform[:3, 3]],
        "rotation_xyzw": [float(item) for item in quaternion_from_rotation(transform[:3, :3])],
    }


def inverse_transform_dict(value: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Invert a GDK-shaped transform dictionary without guessing its frames."""
    return transform_dict_from_matrix(np.linalg.inv(transform_from_dict(value)))


def quaternion_from_rotation(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise AprilTagGeometryError("rotation must be 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = ((matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale,
                      (matrix[1, 0] - matrix[0, 1]) / scale, 0.25 * scale)
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = (0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                          (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[2, 1] - matrix[1, 2]) / scale)
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = ((matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale,
                          (matrix[1, 2] + matrix[2, 1]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale)
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = ((matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale,
                          0.25 * scale, (matrix[1, 0] - matrix[0, 1]) / scale)
    vector = np.asarray(quaternion, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    return tuple(float(item) for item in vector)


def pose_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    """Return translation metres and relative rotation degrees."""
    translation = float(np.linalg.norm(first[:3, 3] - second[:3, 3]))
    error = first[:3, :3].T @ second[:3, :3]
    degrees = math.degrees(math.acos(float(np.clip((np.trace(error) - 1.0) / 2.0, -1.0, 1.0))))
    return translation, degrees


class RollingWorkpiecePoseEstimator:
    """Per-tag rolling-average PnP used by the original working package."""

    def __init__(
        self, *, sample_count: int = 20, max_reprojection_error_px: float = 3.0,
        max_sample_age_s: float = 2.5, geometry: TagGeometry | None = None,
    ) -> None:
        if sample_count < 1 or max_reprojection_error_px <= 0.0 or max_sample_age_s <= 0.0:
            raise AprilTagGeometryError("invalid AprilTag pose-estimation parameters")
        self.sample_count = int(sample_count)
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        self.max_sample_age_s = float(max_sample_age_s)
        self.geometry = DEFAULT_TAG_GEOMETRY if geometry is None else geometry
        self._samples = {tag_id: deque(maxlen=self.sample_count) for tag_id in REQUIRED_TAG_IDS}
        self._sample_times = {tag_id: deque(maxlen=self.sample_count) for tag_id in REQUIRED_TAG_IDS}
        self._last_diagnostics: dict[str, object] = {"state": "collecting_samples"}

    @property
    def sample_counts(self) -> dict[int, int]:
        return {tag_id: len(self._samples[tag_id]) for tag_id in sorted(REQUIRED_TAG_IDS)}

    @property
    def last_diagnostics(self) -> dict[str, object]:
        """Latest quality-gate outcome, safe to expose in a live UI/log."""
        return dict(self._last_diagnostics)

    def reset(self) -> None:
        self._visible_ids = set()
        for tag_id in REQUIRED_TAG_IDS:
            self._samples[tag_id].clear()
            self._sample_times[tag_id].clear()
        self._last_diagnostics = {"state": "samples_reset"}

    def add(self, detections: Mapping[int, Sequence[Sequence[float]]], *, timestamp_s: float) -> None:
        if not math.isfinite(timestamp_s) or timestamp_s <= 0.0:
            raise AprilTagGeometryError("AprilTag image timestamp must be positive and finite")
        self._visible_ids = set()
        for tag_id, corners in detections.items():
            if tag_id not in REQUIRED_TAG_IDS:
                continue
            points = np.asarray(corners, dtype=np.float64)
            if points.shape != (4, 2) or not np.all(np.isfinite(points)):
                continue
            self._samples[tag_id].append(points.copy())
            self._visible_ids.add(tag_id)
            self._sample_times[tag_id].append(float(timestamp_s))

    def estimate(self, camera_matrix: np.ndarray, distortion: Sequence[float] = (), *, now_s: float | None = None) -> WorkpiecePose | None:
        available = [tag_id for tag_id in sorted(getattr(self, "_visible_ids", ()))
                     if len(self._samples[tag_id]) >= self.sample_count]
        if len(available) < 2:
            self._last_diagnostics = {"state": "collecting_samples", "sample_counts": self.sample_counts}
            return None
        if now_s is not None:
            if not math.isfinite(now_s) or now_s <= 0.0:
                raise AprilTagGeometryError("current timestamp must be positive and finite")
            available = [tag_id for tag_id in available
                         if now_s - self._sample_times[tag_id][0] <= self.max_sample_age_s]
            if len(available) < 2:
                self._last_diagnostics = {
                    "state": "stale_tag_samples", "sample_counts": self.sample_counts,
                    "max_sample_age_s": self.max_sample_age_s,
                }
                return None
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("OpenCV is required for AprilTag PnP; install the project 'tags' extra") from exc
        intrinsic = np.asarray(camera_matrix, dtype=np.float64)
        coefficients = np.asarray(tuple(distortion), dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
            raise AprilTagGeometryError("camera matrix must be finite 3x3")
        object_points = np.vstack([self.geometry.object_points[tag_id] for tag_id in available])
        raw_points = {tag_id: np.mean(np.stack(self._samples[tag_id]), axis=0) for tag_id in available}
        candidates = []
        diagnostics: list[dict[str, object]] = []
        for name, order in CORNER_ORDER_CANDIDATES.items():
            # The deployed detector/model convention is forward_0 (also
            # verified by the three-tag runs). Coplanar tags 1+2 cannot
            # independently infer mirrored corner conventions from geometry.
            if len(available) == 2 and name != "forward_0":
                continue
            image_points = np.vstack([raw_points[tag_id][list(order)] for tag_id in available])
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                object_points, image_points, intrinsic, coefficients, iterationsCount=500,
                reprojectionError=3.0, confidence=0.999,
                flags=cv2.SOLVEPNP_EPNP if len(available) == 2 else cv2.SOLVEPNP_ITERATIVE,
            )
            if len(available) == 2:
                # Two tags supply only eight points; solve all of them and
                # verify every corner rather than rely on a minimal subset.
                success, rvec, tvec = cv2.solvePnP(
                    object_points, image_points, intrinsic, coefficients,
                    flags=cv2.SOLVEPNP_SQPNP,
                )
                inliers = np.arange(len(object_points)).reshape(-1, 1) if success else None
            if not success or inliers is None or len(inliers) < 6:
                diagnostics.append({"order": name, "state": "ransac_failed" if not success else "insufficient_inliers", "inlier_count": 0 if inliers is None else int(len(inliers))})
                continue
            selected = inliers.reshape(-1)
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(object_points[selected], image_points[selected], intrinsic, coefficients, rvec, tvec)
            projected, _ = cv2.projectPoints(object_points, rvec, tvec, intrinsic, coefficients)
            errors = np.linalg.norm(projected.reshape(-1, 2) - image_points, axis=1)
            if len(available) == 2:
                selected = np.flatnonzero(errors <= 3.0)
                if len(selected) < 8:
                    continue
            mean_error, max_error = float(np.mean(errors)), float(np.max(errors))
            diagnostics.append({"order": name, "state": "solved", "inlier_count": int(len(selected)), "mean_reprojection_error_px": mean_error, "max_reprojection_error_px": max_error})
            rotation, _ = cv2.Rodrigues(rvec)
            if np.any((object_points @ rotation.T + np.asarray(tvec).reshape(3))[:, 2] <= 0):
                continue
            matrix = np.eye(4, dtype=np.float64)
            matrix[:3, :3], matrix[:3, 3] = rotation, np.asarray(tvec, dtype=np.float64).reshape(3)
            expected_tags = {}
            for tag_id in sorted(REQUIRED_TAG_IDS):
                tag_projected, _ = cv2.projectPoints(
                    self.geometry.object_points[tag_id], rvec, tvec, intrinsic, coefficients,
                )
                expected_tags[int(tag_id)] = tag_projected.reshape(-1, 2).tolist()
            points = tuple({
                "tag_id": int(available[i // 4]), "model_corner_index": int(i % 4),
                "detected_corner_index": int(order[i % 4]),
                "image_px": image_points[i].tolist(),
                "projected_px": projected.reshape(-1, 2)[i].tolist(),
                "error_px": float(errors[i]), "inlier": bool(i in selected),
            } for i in range(len(image_points)))
            candidates.append((mean_error, max_error, name, matrix, image_points,
                               int(len(selected)), points, expected_tags))
        if not candidates:
            self._last_diagnostics = {"state": "pnp_ransac_failed", "sample_counts": self.sample_counts, "corner_candidates": diagnostics}
            return None
        candidates.sort(key=lambda item: item[0])
        mean_error, max_error, order, matrix, _image_points, inlier_count, points, expected_tags = candidates[0]
        summary = sorted(diagnostics, key=lambda item: float(item.get("mean_reprojection_error_px", math.inf)))
        if len(candidates) > 1 and candidates[1][0] - mean_error < 0.25:
            self._last_diagnostics = {
                "state": "pnp_corner_order_ambiguous", "best_order": order, "points": points,
                "expected_tag_corners": expected_tags,
                "mean_reprojection_error_px": mean_error, "corner_candidates": summary,
            }
            return None
        if mean_error > self.max_reprojection_error_px:
            self._last_diagnostics = {
                "state": "pnp_reprojection_rejected", "corner_order": order, "inlier_count": inlier_count,
                "mean_reprojection_error_px": mean_error, "max_reprojection_error_px": max_error,
                "gate_px": self.max_reprojection_error_px, "corner_candidates": summary, "points": points,
                "expected_tag_corners": expected_tags,
            }
            return None
        self._last_diagnostics = {
            "state": "pnp_accepted", "corner_order": order, "inlier_count": inlier_count, "points": points,
            "expected_tag_corners": expected_tags,
            "mean_reprojection_error_px": mean_error, "max_reprojection_error_px": max_error, "corner_candidates": summary,
        }
        return WorkpiecePose(matrix, mean_error, max_error, inlier_count, self.sample_counts,
                             points, expected_tags)


def g2_grasp_targets(
    camera_from_workpiece: np.ndarray, base_from_right_end: np.ndarray, right_end_from_camera: np.ndarray,
    right_end_from_grasp_tcp: np.ndarray, *, pregrasp_offset_m: float = 0.030,
    grasp_tcp_z_offset_m: float = 0.020, grasp_tcp_x_offset_m: float = 0.0,
    workpiece_from_pregrasp_end: np.ndarray | None = None,
) -> GraspTargets:
    """Convert the original CAD grasp convention into calibrated G2 end-link poses.

    ``right_end_from_grasp_tcp`` is a required, measured rigid mapping from
    this workpiece's grasp-TCP convention to G2's ``arm_r_end_link``. It is
    not inferred from any URDF or copied from the OpenArm TCP.
    """
    t_camera_workpiece = as_transform(camera_from_workpiece, "camera_from_workpiece")
    t_base_right_end = as_transform(base_from_right_end, "base_from_right_end")
    t_right_end_camera = as_transform(right_end_from_camera, "right_end_from_camera")
    t_right_end_tcp = as_transform(right_end_from_grasp_tcp, "right_end_from_grasp_tcp")
    base_from_workpiece = t_base_right_end @ t_right_end_camera @ t_camera_workpiece
    base_from_pre_tcp, base_from_grasp_tcp = grasp_tcp_targets(
        base_from_workpiece, pregrasp_offset_m=pregrasp_offset_m,
        grasp_tcp_z_offset_m=grasp_tcp_z_offset_m, grasp_tcp_x_offset_m=grasp_tcp_x_offset_m,
    )
    base_from_pregrasp_end = (
        base_from_workpiece @ as_transform(workpiece_from_pregrasp_end, "workpiece_from_pregrasp_end")
        if workpiece_from_pregrasp_end is not None
        else base_from_pre_tcp @ np.linalg.inv(t_right_end_tcp)
    )
    return GraspTargets(
        base_from_workpiece,
        base_from_pregrasp_end,
        base_from_grasp_tcp @ np.linalg.inv(t_right_end_tcp),
    )


def grasp_tcp_targets(
    base_from_workpiece: np.ndarray, *, pregrasp_offset_m: float = 0.030,
    grasp_tcp_z_offset_m: float = 0.020, grasp_tcp_x_offset_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the original package's virtual pregrasp/grasp TCP poses in base.

    These are workpiece geometry targets only.  A caller must use a measured
    G2 ``T_arm_r_end_link_grasp_tcp`` before converting either pose to a G2
    end-link command.
    """
    t_base_workpiece = as_transform(base_from_workpiece, "base_from_workpiece")
    grasp_point = np.array((0.0, -59.1856153247895, -15.679092927351), dtype=np.float64) / 1000.0
    pregrasp_point = grasp_point + np.array((0.0, -float(pregrasp_offset_m), 0.0), dtype=np.float64)
    r_workpiece_grasp = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)), dtype=np.float64)
    r_grasp_tcp = np.asarray(((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)
    workpiece_from_tcp = np.eye(4, dtype=np.float64)
    workpiece_from_tcp[:3, :3], workpiece_from_tcp[:3, 3] = r_workpiece_grasp @ r_grasp_tcp, grasp_point
    workpiece_from_tcp[:3, 3] += workpiece_from_tcp[:3, 2] * float(grasp_tcp_z_offset_m)
    workpiece_from_tcp[:3, 3] += workpiece_from_tcp[:3, 0] * float(grasp_tcp_x_offset_m)
    workpiece_from_pre_tcp = workpiece_from_tcp.copy()
    # The source applies the final TCP-local compensation to the grasp pose.
    # Pregrasp must retain that same compensation, then back off 30 mm in the
    # workpiece approach direction; otherwise a non-zero TCP Z offset changes
    # the pregrasp/grasp separation from 30 mm to 50 mm.
    workpiece_from_pre_tcp[:3, 3] += pregrasp_point - grasp_point
    return t_base_workpiece @ workpiece_from_pre_tcp, t_base_workpiece @ workpiece_from_tcp


def right_end_from_grasp_tcp_candidate(
    base_from_right_end: np.ndarray, base_from_workpiece: np.ndarray, *,
    pregrasp_offset_m: float = 0.030, grasp_tcp_z_offset_m: float = 0.020,
    grasp_tcp_x_offset_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Infer ``T_arm_r_end_link_grasp_tcp`` from one manually placed grasp pose.

    This is geometry only and deliberately emits no robot command.  It is
    valid only when the measured G2 end link is physically holding the exact
    virtual grasp-TCP pose returned from the source workpiece convention.
    """
    t_base_right_end = as_transform(base_from_right_end, "base_from_right_end")
    base_from_pre_tcp, base_from_grasp_tcp = grasp_tcp_targets(
        base_from_workpiece, pregrasp_offset_m=pregrasp_offset_m,
        grasp_tcp_z_offset_m=grasp_tcp_z_offset_m, grasp_tcp_x_offset_m=grasp_tcp_x_offset_m,
    )
    return np.linalg.inv(t_base_right_end) @ base_from_grasp_tcp, base_from_pre_tcp, base_from_grasp_tcp


def workpiece_from_pregrasp_end_candidate(
    base_from_workpiece: np.ndarray, base_from_right_end: np.ndarray,
) -> np.ndarray:
    """Capture a manually verified G2 pregrasp end-link pose relative to workpiece.

    It is intentionally independent of an OpenArm pregrasp offset.  The
    operator chooses the collision-free pose; subsequent observations move it
    rigidly with the AprilTag workpiece frame.
    """
    t_base_workpiece = as_transform(base_from_workpiece, "base_from_workpiece")
    t_base_right_end = as_transform(base_from_right_end, "base_from_right_end")
    return np.linalg.inv(t_base_workpiece) @ t_base_right_end


def constrained_pregrasp_end_candidate(
    base_from_raw_pregrasp_end: np.ndarray, base_from_grasp_end: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Project a manually held pregrasp onto the calibrated straight approach.

    The calibrated grasp end-link ``+Z`` axis is the only permitted approach
    direction.  The returned pose preserves the operator-selected axial
    distance but removes lateral offset and adopts the grasp orientation, so
    the later GDK move is a pure fixed-orientation translation.
    """
    raw = as_transform(base_from_raw_pregrasp_end, "base_from_raw_pregrasp_end")
    grasp = as_transform(base_from_grasp_end, "base_from_grasp_end")
    approach_axis = grasp[:3, 2]
    grasp_to_raw = grasp[:3, 3] - raw[:3, 3]
    axial_distance_m = float(np.dot(grasp_to_raw, approach_axis))
    lateral_error_m = float(np.linalg.norm(grasp_to_raw - axial_distance_m * approach_axis))
    _translation_unused, orientation_error_deg = pose_distance(raw, grasp)
    constrained = grasp.copy()
    constrained[:3, 3] = grasp[:3, 3] - axial_distance_m * approach_axis
    return constrained, axial_distance_m, lateral_error_m, orientation_error_deg
