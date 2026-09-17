"""Pure data model and time-based interpolation for G2 arm teaching."""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence

try:
    from .safety import (
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        SafetyError,
        validate_arm_target,
    )
except ImportError:
    from safety import (
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        SafetyError,
        validate_arm_target,
    )


SCHEMA_VERSION = 1


class TrajectoryError(ValueError):
    """A stored teaching trajectory is malformed or incompatible."""


@dataclass(frozen=True)
class TrajectorySample:
    t_ns: int
    joint_positions_rad: tuple[float, ...]


@dataclass(frozen=True)
class Trajectory:
    side: str
    joint_names: tuple[str, ...]
    samples: tuple[TrajectorySample, ...]

    @property
    def duration_ns(self) -> int:
        return self.samples[-1].t_ns


def arm_joint_names(side: str) -> tuple[str, ...]:
    if side == "left":
        return LEFT_ARM_JOINT_NAMES
    if side == "right":
        return RIGHT_ARM_JOINT_NAMES
    raise TrajectoryError("side must be left or right")


def _validate_selected_positions(side: str, positions: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in positions)
    if len(values) != 7:
        raise TrajectoryError(f"{side} trajectory sample must contain seven joint positions")
    if not all(math.isfinite(value) for value in values):
        raise TrajectoryError(f"{side} trajectory sample contains NaN or Inf")
    complete = values + (0.0,) * 7 if side == "left" else (0.0,) * 7 + values
    validate_arm_target(complete)
    return values


def _sample(side: str, t_ns: int, positions: Sequence[float]) -> TrajectorySample:
    if isinstance(t_ns, bool) or int(t_ns) != t_ns or int(t_ns) < 0:
        raise TrajectoryError("sample t_ns must be a non-negative integer")
    return TrajectorySample(int(t_ns), _validate_selected_positions(side, positions))


def build_trajectory(
    side: str,
    raw_samples: Iterable[tuple[int, Sequence[float]]],
) -> Trajectory:
    """Build a validated single-arm trajectory from relative-time samples."""
    names = arm_joint_names(side)
    samples = tuple(_sample(side, timestamp, positions) for timestamp, positions in raw_samples)
    if not samples:
        raise TrajectoryError("trajectory must contain at least one sample")
    if samples[0].t_ns != 0:
        raise TrajectoryError("first trajectory sample must have t_ns=0")
    for previous, following in zip(samples, samples[1:]):
        if following.t_ns <= previous.t_ns:
            raise TrajectoryError("trajectory sample timestamps must be strictly increasing")
    return Trajectory(side=side, joint_names=names, samples=samples)


def trim_leading_stationary(trajectory: Trajectory) -> tuple[Trajectory, int]:
    """Trim initial idle against the first sample, keeping the onset's preceding frame."""
    initial = trajectory.samples[0].joint_positions_rad
    threshold = math.radians(0.5)
    for index, sample in enumerate(trajectory.samples):
        if any(abs(current - first) > threshold for current, first in zip(sample.joint_positions_rad, initial)):
            start = max(0, index - 1)
            origin_ns = trajectory.samples[start].t_ns
            return build_trajectory(trajectory.side, ((s.t_ns - origin_ns, s.joint_positions_rad) for s in trajectory.samples[start:])), start
    return trajectory, 0


def trajectory_document(trajectory: Trajectory) -> dict:
    """Return the schema-versioned JSON-compatible trajectory document."""
    return {
        "schema_version": SCHEMA_VERSION,
        "side": trajectory.side,
        "joint_names": list(trajectory.joint_names),
        "samples": [
            {"t_ns": sample.t_ns, "joint_positions_rad": list(sample.joint_positions_rad)}
            for sample in trajectory.samples
        ],
    }


def load_trajectory(document: Mapping) -> Trajectory:
    """Validate and parse a JSON-compatible trajectory document."""
    if not isinstance(document, Mapping):
        raise TrajectoryError("trajectory document must be an object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise TrajectoryError(f"unsupported trajectory schema_version: {document.get('schema_version')!r}")
    side = document.get("side")
    names = arm_joint_names(side)
    if tuple(document.get("joint_names", ())) != names:
        raise TrajectoryError(f"trajectory joint_names do not match {side} arm")
    records = document.get("samples")
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise TrajectoryError("trajectory samples must be an array")
    raw_samples = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TrajectoryError("trajectory sample must be an object")
        raw_samples.append((record.get("t_ns"), record.get("joint_positions_rad")))
    return build_trajectory(side, raw_samples)


def _smoothstep5(fraction: float) -> float:
    return 6.0 * fraction**5 - 15.0 * fraction**4 + 10.0 * fraction**3


def _interpolate(trajectory: Trajectory, original_t_ns: int) -> tuple[float, ...]:
    if original_t_ns <= 0:
        return trajectory.samples[0].joint_positions_rad
    if original_t_ns >= trajectory.duration_ns:
        return trajectory.samples[-1].joint_positions_rad
    for before, after in zip(trajectory.samples, trajectory.samples[1:]):
        if original_t_ns <= after.t_ns:
            linear = (original_t_ns - before.t_ns) / (after.t_ns - before.t_ns)
            fraction = _smoothstep5(linear)
            return tuple(
                initial + fraction * (final - initial)
                for initial, final in zip(before.joint_positions_rad, after.joint_positions_rad)
            )
    raise AssertionError("trajectory endpoint was not found")


def iter_smooth_targets(
    trajectory: Trajectory,
    *,
    rate_hz: float,
    speed_scale: float,
) -> Iterator[TrajectorySample]:
    """Yield uniform, quintic-smoothed targets over scaled replay time."""
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise TrajectoryError("rate_hz must be positive and finite")
    if not math.isfinite(speed_scale) or speed_scale <= 0.0:
        raise TrajectoryError("speed_scale must be positive and finite")
    if trajectory.duration_ns == 0:
        yield TrajectorySample(0, trajectory.samples[0].joint_positions_rad)
        return
    interval_ns = int(round(1_000_000_000 / rate_hz))
    if interval_ns <= 0:
        raise TrajectoryError("rate_hz is too high for nanosecond replay timing")
    replay_duration_ns = int(round(trajectory.duration_ns / speed_scale))
    replay_times = list(range(0, replay_duration_ns, interval_ns))
    replay_times.append(replay_duration_ns)
    for replay_t_ns in replay_times:
        original_t_ns = min(trajectory.duration_ns, int(round(replay_t_ns * speed_scale)))
        yield TrajectorySample(replay_t_ns, _interpolate(trajectory, original_t_ns))
