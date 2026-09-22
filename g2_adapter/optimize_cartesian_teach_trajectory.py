#!/usr/bin/env python3
"""Simplify an absolute Cartesian teaching path and solve it to replayable IK."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

try:
    from .cartesian_teach import (
        joint_document,
        load_cartesian_document,
        resample_cartesian_trajectory,
        solve_cartesian_trajectory,
    )
    from .cartesian_teach_trajectory_gdk import DEFAULT_URDF
    from .cartesian_trajectory_optimizer import optimize_cartesian_trajectory
    from .g2_kinematics import UrdfKinematics
    from .safety import ARM_POSITION_LIMITS
except ImportError:
    from cartesian_teach import joint_document, load_cartesian_document, resample_cartesian_trajectory, solve_cartesian_trajectory
    from cartesian_teach_trajectory_gdk import DEFAULT_URDF
    from cartesian_trajectory_optimizer import optimize_cartesian_trajectory
    from g2_kinematics import UrdfKinematics
    from safety import ARM_POSITION_LIMITS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, document: dict) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="优化 G2 绝对末端轨迹并一次性完成整轨 IK，输出可回放关节轨迹",
    )
    parser.add_argument("--input", required=True, type=Path, help="*_cartesian_raw.json")
    parser.add_argument("--output", required=True, type=Path, help="*_cartesian_solved.json")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--rate-hz", type=float, default=100.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--jitter-position-mm", type=float, default=2.0,
                        help="中值平滑允许修正的局部位置抖动，默认 2 mm")
    parser.add_argument("--jitter-orientation-deg", type=float, default=1.0,
                        help="中值平滑允许修正的局部姿态抖动，默认 1 deg")
    parser.add_argument("--outlier-position-mm", type=float, default=8.0,
                        help="孤立大位移尖峰的修复阈值，默认 8 mm")
    parser.add_argument("--outlier-orientation-deg", type=float, default=5.0,
                        help="孤立大姿态尖峰的修复阈值，默认 5 deg")
    parser.add_argument("--pause-position-mm", type=float, default=0.01,
                        help="小于此位移的采样间隔视为停顿并压缩")
    parser.add_argument("--pause-orientation-deg", type=float, default=0.01,
                        help="小于此旋转的采样间隔视为停顿并压缩")
    parser.add_argument("--median-window", type=int, default=7,
                        help="中值平滑窗口，必须为奇数，默认 7")
    parser.add_argument("--max-velocity-rad-s", type=float, default=1.0)
    parser.add_argument("--max-acceleration-rad-s2", type=float, default=15.0)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    source = args.input.expanduser().resolve()
    output = args.output.expanduser().resolve()
    urdf = args.urdf.expanduser().resolve()
    try:
        raw = load_cartesian_document(json.loads(source.read_text(encoding="utf-8")))
        optimized, optimization = optimize_cartesian_trajectory(
            raw,
            position_tolerance_m=args.position_tolerance_mm/1000.0,
            orientation_tolerance_rad=math.radians(args.orientation_tolerance_deg),
            jitter_position_m=args.jitter_position_mm/1000.0,
            jitter_orientation_rad=math.radians(args.jitter_orientation_deg),
            median_window=args.median_window,
            pause_position_m=args.pause_position_mm/1000.0,
            pause_orientation_rad=math.radians(args.pause_orientation_deg),
            outlier_position_m=args.outlier_position_mm/1000.0,
            outlier_orientation_rad=math.radians(args.outlier_orientation_deg),
        )
        resampled = resample_cartesian_trajectory(optimized, rate_hz=args.rate_hz)
        limits = ARM_POSITION_LIMITS[:7] if raw.side == "left" else ARM_POSITION_LIMITS[7:]
        solved, report = solve_cartesian_trajectory(
            resampled, UrdfKinematics(urdf),
            joint_limits=limits,
            source_sha256=_sha256(source),
            urdf_sha256=_sha256(urdf),
            max_velocity_rad_s=args.max_velocity_rad_s,
            max_acceleration_rad_s2=args.max_acceleration_rad_s2,
        )
        document = joint_document(
            solved, report,
            solved_at_utc=datetime.now(timezone.utc).isoformat(),
            solve_rate_hz=args.rate_hz,
            optimization=optimization,
        )
        _atomic_write_json(output, document)
        print(
            f"末端轨迹优化：{optimization['original_samples']} 点 → "
            f"{optimization['keyframes']} 个关键点；"
            f"{optimization['original_duration_s']:.2f}s → "
            f"{optimization['optimized_duration_s']:.2f}s；"
            f"修复孤立尖峰 {optimization['isolated_outliers_repaired']} 个。",
            flush=True,
        )
        print(f"整轨 IK：{len(solved.points)} 点；replay_ready={report.replay_ready}")
        print(
            f"最大 FK 误差：{report.max_position_error_m*1000:.3f} mm，"
            f"{math.degrees(report.max_orientation_error_rad):.3f}°",
        )
        if report.errors:
            print("验证失败：" + "; ".join(report.errors))
        print(f"已保存：{output}")
        return 0 if report.replay_ready else 1
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"末端轨迹优化/IK失败：{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
