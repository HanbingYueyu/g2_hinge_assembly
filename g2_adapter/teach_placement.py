"""Common loading and replay helpers for main-flow teaching placements."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

try:
    from .cartesian_teach import load_joint_document
    from .cartesian_teach_trajectory_gdk import (
        DEFAULT_URDF,
        _sha256,
        replay_trajectory,
        verify_live_model,
    )
    from .g2_kinematics import UrdfKinematics
    from .teach_trajectory import load_trajectory
except ImportError:
    from cartesian_teach import load_joint_document
    from cartesian_teach_trajectory_gdk import DEFAULT_URDF, _sha256, replay_trajectory, verify_live_model
    from g2_kinematics import UrdfKinematics
    from teach_trajectory import load_trajectory


@dataclass(frozen=True)
class TeachSpec:
    """A validated main-flow teaching document and its replay representation."""

    path: Path
    side: str
    kind: str
    trajectory: object
    report: object | None = None


def _side_label(side: str) -> str:
    return "左臂" if side == "left" else "右臂"


def load_teach_spec(path: str | Path, side: str) -> TeachSpec:
    """Load either a legacy joint-teach file or a solved Cartesian-teach file.

    Raw absolute Cartesian recordings are intentionally rejected here: the main
    flow must never start a robot before the complete trajectory has passed the
    offline IK and motion-limit checks.
    """
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    resolved = Path(path).expanduser().resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    kind = document.get("kind")
    if kind == "g2_solved_absolute_joint_trajectory":
        trajectory, report = load_joint_document(document)
        if trajectory.side != side:
            raise ValueError(f"示教轨迹必须是{_side_label(side)}：{resolved}")
        if not report.replay_ready:
            detail = "; ".join(report.errors) or "离线验证未通过"
            raise ValueError(f"{_side_label(side)}末端轨迹未通过离线 solve 验证：{detail}")
        return TeachSpec(resolved, side, "cartesian_solved", trajectory, report)
    if kind == "g2_absolute_cartesian_trajectory":
        raise ValueError(f"{resolved} 是末端 raw 轨迹，请先执行 solve 后再用于主流程")

    trajectory = load_trajectory(document)
    if trajectory.side != side:
        raise ValueError(f"示教轨迹必须是{_side_label(side)}：{resolved}")
    return TeachSpec(resolved, side, "joint", trajectory)


def replay_solved_trajectory(
    gdk,
    robot,
    spec: TeachSpec,
    *,
    speed_scale: float,
    start_velocity_rad_s: float,
    max_velocity_rad_s: float = 1.0,
    max_acceleration_rad_s2: float = 15.0,
    move_to_start: bool = True,
    start_only: bool = False,
) -> int:
    """Replay a solved absolute Cartesian-teach document in position mode."""
    if spec.kind != "cartesian_solved" or spec.report is None:
        raise ValueError("replay_solved_trajectory 需要已 solve 的末端轨迹")
    trajectory = spec.trajectory
    urdf = DEFAULT_URDF.resolve()
    if trajectory.urdf_sha256 != _sha256(urdf):
        raise ValueError("当前 URDF 与末端轨迹离线 IK 使用的 URDF 不一致")
    model_error = verify_live_model(UrdfKinematics(urdf), robot, spec.side)
    if not start_only:
        print("运动检查通过，开始移动。", flush=True)
    return replay_trajectory(
        gdk,
        robot,
        trajectory,
        spec.report,
        speed_scale=speed_scale,
        start_velocity_rad_s=start_velocity_rad_s,
        max_velocity_rad_s=max_velocity_rad_s,
        max_acceleration_rad_s2=max_acceleration_rad_s2,
        move_to_start=move_to_start,
        start_only=start_only,
    )
