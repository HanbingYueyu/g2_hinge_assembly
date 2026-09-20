"""Pure data and interpolation helpers for absolute Cartesian teaching."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

try:
    from .g2_kinematics import UrdfKinematics
    from .safety import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
except ImportError:
    from g2_kinematics import UrdfKinematics
    from safety import LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES


BODY_JOINT_NAMES = tuple(f"idx0{i}_body_joint{i}" for i in range(1, 6))


def _fixed_body_positions(document: Mapping) -> dict[str, float]:
    raw = document.get("fixed_joint_positions", {})
    if set(raw) != set(BODY_JOINT_NAMES):
        raise ValueError("trajectory must contain exactly the five fixed body joints")
    values = {str(name): float(value) for name, value in raw.items()}
    if not all(math.isfinite(value) for value in values.values()):
        raise ValueError("trajectory fixed body joints contain NaN or Inf")
    return values


@dataclass(frozen=True)
class CartesianSample:
    t_ns: int
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]
    reference_joint_positions_rad: tuple[float, ...]


@dataclass(frozen=True)
class CartesianTrajectory:
    side: str
    base_frame: str
    tip_frame: str
    joint_names: tuple[str, ...]
    fixed_joint_positions: Mapping[str, float]
    samples: tuple[CartesianSample, ...]


@dataclass(frozen=True)
class JointTrajectoryPoint:
    t_ns: int
    joint_positions_rad: tuple[float, ...]
    position_error_m: float
    orientation_error_rad: float


@dataclass(frozen=True)
class JointTrajectory:
    side: str
    joint_names: tuple[str, ...]
    fixed_joint_positions: Mapping[str, float]
    source_sha256: str
    urdf_sha256: str
    points: tuple[JointTrajectoryPoint, ...]


@dataclass(frozen=True)
class ValidationReport:
    replay_ready: bool
    errors: tuple[str, ...]
    max_velocity_rad_s: float
    max_acceleration_rad_s2: float
    max_position_error_m: float
    max_orientation_error_rad: float


def cartesian_document(trajectory: CartesianTrajectory, *, urdf_sha256: str, **metadata) -> dict:
    return {
        "schema_version": 1,
        "kind": "g2_absolute_cartesian_trajectory",
        "side": trajectory.side,
        "base_frame": trajectory.base_frame,
        "tip_frame": trajectory.tip_frame,
        "joint_names": list(trajectory.joint_names),
        "fixed_joint_positions": dict(trajectory.fixed_joint_positions),
        "urdf_sha256": str(urdf_sha256),
        "samples": [
            {
                "t_ns": sample.t_ns,
                "position_m": list(sample.position_m),
                "orientation_xyzw": list(sample.orientation_xyzw),
                "reference_joint_positions_rad": list(sample.reference_joint_positions_rad),
            }
            for sample in trajectory.samples
        ],
        **metadata,
    }


def load_cartesian_document(document: Mapping) -> CartesianTrajectory:
    if document.get("schema_version") != 1 or document.get("kind") != "g2_absolute_cartesian_trajectory":
        raise ValueError("unsupported Cartesian trajectory document")
    side = str(document.get("side"))
    if side not in {"left", "right"}:
        raise ValueError("trajectory side must be left or right")
    names = tuple(str(value) for value in document.get("joint_names", ()))
    expected_names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
    if names != expected_names:
        raise ValueError(f"Cartesian trajectory joint names do not match {side} arm")
    expected_tip = "arm_l_end_link" if side == "left" else "arm_r_end_link"
    if document.get("base_frame") != "base_link" or document.get("tip_frame") != expected_tip:
        raise ValueError("Cartesian trajectory frames do not match declared side")
    fixed = _fixed_body_positions(document)
    samples = []
    previous = -1
    for record in document.get("samples", ()):
        t_ns = int(record["t_ns"])
        position = tuple(float(value) for value in record["position_m"])
        orientation = tuple(float(value) for value in record["orientation_xyzw"])
        joints = tuple(float(value) for value in record["reference_joint_positions_rad"])
        if t_ns <= previous or len(position) != 3 or len(orientation) != 4 or len(joints) != 7:
            raise ValueError("malformed Cartesian trajectory sample")
        if not all(math.isfinite(value) for value in (*position, *orientation, *joints)):
            raise ValueError("Cartesian trajectory contains NaN or Inf")
        norm = math.sqrt(sum(value*value for value in orientation))
        if not 0.99 <= norm <= 1.01:
            raise ValueError("Cartesian trajectory contains a non-unit quaternion")
        samples.append(CartesianSample(t_ns, position, tuple(value/norm for value in orientation), joints))
        previous = t_ns
    if not samples or samples[0].t_ns != 0:
        raise ValueError("Cartesian trajectory must start at t_ns=0")
    return CartesianTrajectory(
        side, str(document.get("base_frame")), str(document.get("tip_frame")), names,
        fixed,
        tuple(samples),
    )


def joint_document(trajectory: JointTrajectory, report: ValidationReport, **metadata) -> dict:
    return {
        "schema_version": 1,
        "kind": "g2_solved_absolute_joint_trajectory",
        "side": trajectory.side,
        "joint_names": list(trajectory.joint_names),
        "fixed_joint_positions": dict(trajectory.fixed_joint_positions),
        "source_sha256": trajectory.source_sha256,
        "urdf_sha256": trajectory.urdf_sha256,
        "points": [
            {"t_ns": point.t_ns, "joint_positions_rad": list(point.joint_positions_rad),
             "position_error_m": point.position_error_m,
             "orientation_error_rad": point.orientation_error_rad}
            for point in trajectory.points
        ],
        "validation": {
            "replay_ready": report.replay_ready,
            "errors": list(report.errors),
            "max_velocity_rad_s": report.max_velocity_rad_s,
            "max_acceleration_rad_s2": report.max_acceleration_rad_s2,
            "max_position_error_m": report.max_position_error_m,
            "max_orientation_error_rad": report.max_orientation_error_rad,
        },
        **metadata,
    }


def load_joint_document(document: Mapping) -> tuple[JointTrajectory, ValidationReport]:
    if document.get("schema_version") != 1 or document.get("kind") != "g2_solved_absolute_joint_trajectory":
        raise ValueError("unsupported solved trajectory document")
    side = str(document.get("side"))
    if side not in {"left", "right"}:
        raise ValueError("solved trajectory side must be left or right")
    names = tuple(str(value) for value in document.get("joint_names", ()))
    expected_names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
    if names != expected_names:
        raise ValueError(f"solved trajectory joint names do not match {side} arm")
    fixed = _fixed_body_positions(document)
    points = tuple(JointTrajectoryPoint(
        int(record["t_ns"]), tuple(float(value) for value in record["joint_positions_rad"]),
        float(record["position_error_m"]), float(record["orientation_error_rad"]),
    ) for record in document.get("points", ()))
    if not points or points[0].t_ns != 0 or any(len(point.joint_positions_rad) != 7 for point in points):
        raise ValueError("solved trajectory points are malformed")
    if any(after.t_ns <= before.t_ns for before, after in zip(points, points[1:])):
        raise ValueError("solved trajectory timestamps must be strictly increasing")
    if any(not all(math.isfinite(value) for value in (*point.joint_positions_rad,
                                                       point.position_error_m,
                                                       point.orientation_error_rad)) for point in points):
        raise ValueError("solved trajectory contains NaN or Inf")
    trajectory = JointTrajectory(
        side, names,
        fixed,
        str(document["source_sha256"]), str(document["urdf_sha256"]), points,
    )
    raw = document.get("validation", {})
    report = ValidationReport(
        bool(raw.get("replay_ready", False)), tuple(str(value) for value in raw.get("errors", ())),
        float(raw.get("max_velocity_rad_s", math.inf)), float(raw.get("max_acceleration_rad_s2", math.inf)),
        float(raw.get("max_position_error_m", math.inf)), float(raw.get("max_orientation_error_rad", math.inf)),
    )
    return trajectory, report


def matrix_to_quaternion(rotation) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to normalized XYZW quaternion."""
    matrix = np.asarray(rotation, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = ((matrix[2, 1]-matrix[1, 2])/scale, (matrix[0, 2]-matrix[2, 0])/scale,
                      (matrix[1, 0]-matrix[0, 1])/scale, 0.25*scale)
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0+matrix[0, 0]-matrix[1, 1]-matrix[2, 2])*2.0
            quaternion = (0.25*scale, (matrix[0, 1]+matrix[1, 0])/scale,
                          (matrix[0, 2]+matrix[2, 0])/scale, (matrix[2, 1]-matrix[1, 2])/scale)
        elif index == 1:
            scale = math.sqrt(1.0+matrix[1, 1]-matrix[0, 0]-matrix[2, 2])*2.0
            quaternion = ((matrix[0, 1]+matrix[1, 0])/scale, 0.25*scale,
                          (matrix[1, 2]+matrix[2, 1])/scale, (matrix[0, 2]-matrix[2, 0])/scale)
        else:
            scale = math.sqrt(1.0+matrix[2, 2]-matrix[0, 0]-matrix[1, 1])*2.0
            quaternion = ((matrix[0, 2]+matrix[2, 0])/scale, (matrix[1, 2]+matrix[2, 1])/scale,
                          0.25*scale, (matrix[1, 0]-matrix[0, 1])/scale)
    values = np.asarray(quaternion, dtype=float)
    values /= np.linalg.norm(values)
    return tuple(float(value) for value in values)


def quaternion_to_matrix(quaternion) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=float) / np.linalg.norm(quaternion)
    return np.array((
        (1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
        (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
        (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)),
    ), dtype=float)


def sample_transform(sample: CartesianSample) -> np.ndarray:
    result = np.eye(4, dtype=float)
    result[:3, :3] = quaternion_to_matrix(sample.orientation_xyzw)
    result[:3, 3] = sample.position_m
    return result


def _slerp(a, b, fraction: float) -> tuple[float, float, float, float]:
    first, second = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    dot = float(np.dot(first, second))
    if dot < 0.0:
        second, dot = -second, -dot
    dot = min(1.0, max(-1.0, dot))
    if dot > 0.9995:
        result = first + fraction * (second-first)
        result /= np.linalg.norm(result)
    else:
        angle = math.acos(dot)
        result = (math.sin((1-fraction)*angle)*first + math.sin(fraction*angle)*second) / math.sin(angle)
    return tuple(float(value) for value in result)


def _quaternion_distance(a, b) -> float:
    first, second = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    first, second = first/np.linalg.norm(first), second/np.linalg.norm(second)
    return 2.0 * math.acos(min(1.0, abs(float(np.dot(first, second)))))


def trim_leading_stationary(
    trajectory: CartesianTrajectory, *, translation_threshold_m: float = 0.0005,
    rotation_threshold_rad: float = math.radians(0.5),
) -> tuple[CartesianTrajectory, int]:
    initial = trajectory.samples[0]
    for index, sample in enumerate(trajectory.samples):
        translation = math.sqrt(sum((a-b)**2 for a, b in zip(sample.position_m, initial.position_m)))
        rotation = _quaternion_distance(sample.orientation_xyzw, initial.orientation_xyzw)
        if translation > translation_threshold_m or rotation > rotation_threshold_rad:
            start = max(0, index-1)
            origin = trajectory.samples[start].t_ns
            samples = tuple(CartesianSample(
                item.t_ns-origin, item.position_m, item.orientation_xyzw,
                item.reference_joint_positions_rad,
            ) for item in trajectory.samples[start:])
            return CartesianTrajectory(
                trajectory.side, trajectory.base_frame, trajectory.tip_frame,
                trajectory.joint_names, trajectory.fixed_joint_positions, samples,
            ), start
    return trajectory, 0


def _interpolate(trajectory: CartesianTrajectory, t_ns: int) -> CartesianSample:
    if t_ns <= 0:
        return trajectory.samples[0]
    if t_ns >= trajectory.samples[-1].t_ns:
        return trajectory.samples[-1]
    for before, after in zip(trajectory.samples, trajectory.samples[1:]):
        if t_ns <= after.t_ns:
            fraction = (t_ns-before.t_ns)/(after.t_ns-before.t_ns)
            position = tuple(a + fraction*(b-a) for a, b in zip(before.position_m, after.position_m))
            joints = tuple(a + fraction*(b-a) for a, b in zip(before.reference_joint_positions_rad, after.reference_joint_positions_rad))
            return CartesianSample(t_ns, position, _slerp(before.orientation_xyzw, after.orientation_xyzw, fraction), joints)
    raise AssertionError("trajectory endpoint not found")


def resample_cartesian_trajectory(trajectory: CartesianTrajectory, *, rate_hz: float) -> CartesianTrajectory:
    if rate_hz <= 0.0 or not math.isfinite(rate_hz):
        raise ValueError("rate_hz must be positive and finite")
    if not trajectory.samples or trajectory.samples[0].t_ns != 0:
        raise ValueError("trajectory must start at t_ns=0")
    interval = int(round(1e9/rate_hz))
    end = trajectory.samples[-1].t_ns
    times = list(range(0, end, interval))
    if not times or times[-1] != end:
        times.append(end)
    return CartesianTrajectory(
        trajectory.side, trajectory.base_frame, trajectory.tip_frame, trajectory.joint_names,
        trajectory.fixed_joint_positions, tuple(_interpolate(trajectory, value) for value in times),
    )


def validate_joint_trajectory(
    trajectory: JointTrajectory, *, joint_limits, max_velocity_rad_s: float,
    max_acceleration_rad_s2: float, max_position_error_m: float,
    max_orientation_error_rad: float,
) -> ValidationReport:
    errors: list[str] = []
    velocities = []
    for point in trajectory.points:
        for name, value, (lower, upper) in zip(trajectory.joint_names, point.joint_positions_rad, joint_limits):
            if not lower <= value <= upper:
                errors.append(f"joint limit: {name}={value:.6f}")
    for before, after in zip(trajectory.points, trajectory.points[1:]):
        seconds = (after.t_ns-before.t_ns)/1e9
        if seconds <= 0:
            errors.append("timestamps are not strictly increasing")
            continue
        velocities.append(tuple((b-a)/seconds for a, b in zip(before.joint_positions_rad, after.joint_positions_rad)))
    observed_velocity = max((abs(value) for row in velocities for value in row), default=0.0)
    if observed_velocity > max_velocity_rad_s:
        errors.append(f"joint velocity {observed_velocity:.6f} exceeds {max_velocity_rad_s:.6f}")
    accelerations = []
    for index, (before, after) in enumerate(zip(velocities, velocities[1:]), 1):
        seconds = (trajectory.points[index+1].t_ns-trajectory.points[index-1].t_ns)/2e9
        accelerations.extend((b-a)/seconds for a, b in zip(before, after))
    observed_acceleration = max((abs(value) for value in accelerations), default=0.0)
    if observed_acceleration > max_acceleration_rad_s2:
        errors.append(f"joint acceleration {observed_acceleration:.6f} exceeds {max_acceleration_rad_s2:.6f}")
    position_error = max((point.position_error_m for point in trajectory.points), default=math.inf)
    orientation_error = max((point.orientation_error_rad for point in trajectory.points), default=math.inf)
    if position_error > max_position_error_m:
        errors.append(f"FK position error {position_error:.6f} exceeds {max_position_error_m:.6f}")
    if orientation_error > max_orientation_error_rad:
        errors.append(f"FK orientation error {orientation_error:.6f} exceeds {max_orientation_error_rad:.6f}")
    return ValidationReport(not errors, tuple(errors), observed_velocity, observed_acceleration, position_error, orientation_error)


def solve_cartesian_trajectory(
    trajectory: CartesianTrajectory,
    model: UrdfKinematics,
    *,
    joint_limits,
    source_sha256: str,
    urdf_sha256: str,
    max_velocity_rad_s: float,
    max_acceleration_rad_s2: float,
    max_position_error_m: float = 1e-4,
    max_orientation_error_rad: float = 1e-3,
) -> tuple[JointTrajectory, ValidationReport]:
    if not trajectory.samples:
        raise ValueError("Cartesian trajectory is empty")
    points = []
    previous = np.asarray(trajectory.samples[0].reference_joint_positions_rad, dtype=float)
    for sample in trajectory.samples:
        solution, diagnostic = model.solve_pose(
            trajectory.base_frame,
            trajectory.tip_frame,
            trajectory.fixed_joint_positions,
            trajectory.joint_names,
            sample_transform(sample),
            previous,
            reference=sample.reference_joint_positions_rad,
            joint_limits=joint_limits,
            position_tolerance_m=max_position_error_m * 0.5,
            orientation_tolerance_rad=max_orientation_error_rad * 0.5,
        )
        points.append(JointTrajectoryPoint(
            sample.t_ns,
            tuple(float(value) for value in solution),
            diagnostic.position_error_m,
            diagnostic.orientation_error_rad,
        ))
        previous = solution
    solved = JointTrajectory(
        trajectory.side,
        trajectory.joint_names,
        trajectory.fixed_joint_positions,
        source_sha256,
        urdf_sha256,
        tuple(points),
    )
    report = validate_joint_trajectory(
        solved,
        joint_limits=joint_limits,
        max_velocity_rad_s=max_velocity_rad_s,
        max_acceleration_rad_s2=max_acceleration_rad_s2,
        max_position_error_m=max_position_error_m,
        max_orientation_error_rad=max_orientation_error_rad,
    )
    return solved, report
