#!/usr/bin/env python3
"""Record, solve and replay an independent absolute G2 end-link trajectory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import time
from collections.abc import Callable, Iterator
from datetime import datetime, timezone
from pathlib import Path

try:
    from .cartesian_teach import (
        CartesianSample, CartesianTrajectory, JointTrajectory, ValidationReport,
        cartesian_document, joint_document, load_cartesian_document, load_joint_document,
        quaternion_to_matrix, resample_cartesian_trajectory, solve_cartesian_trajectory,
        trim_leading_stationary, validate_joint_trajectory,
    )
    from .g2_kinematics import UrdfKinematics, rotation_error
    from .safety import ARM_POSITION_LIMITS, LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
except ImportError:
    from cartesian_teach import (
        CartesianSample, CartesianTrajectory, JointTrajectory, ValidationReport,
        cartesian_document, joint_document, load_cartesian_document, load_joint_document,
        quaternion_to_matrix, resample_cartesian_trajectory, solve_cartesian_trajectory,
        trim_leading_stationary, validate_joint_trajectory,
    )
    from g2_kinematics import UrdfKinematics, rotation_error
    from safety import ARM_POSITION_LIMITS, LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES


BODY_JOINT_NAMES = tuple(f"idx0{i}_body_joint{i}" for i in range(1, 6))
DEFAULT_URDF = Path(__file__).resolve().parents[1] / "robot_description/G2_t2v2_crsB_omnipicker.urdf"


def enter_position_control(robot, *, monotonic_fn=time.monotonic, sleep_fn=time.sleep) -> None:
    before = robot.get_motion_control_status()
    if int(before.error_code) != 0:
        raise RuntimeError(f"cannot switch control mode while error={before.error_code}: {before.error_msg}")
    if int(before.control_mode) == 1:
        return
    result = robot.set_control_mode(1)
    if result not in (None, 0):
        raise RuntimeError(f"set_control_mode(1) returned {result}")
    deadline = monotonic_fn() + 2.0
    while True:
        status = robot.get_motion_control_status()
        if int(status.control_mode) == 1 and int(status.error_code) == 0:
            return
        if monotonic_fn() >= deadline:
            raise RuntimeError(
                f"failed to enter healthy control_mode=1: mode={status.control_mode} "
                f"error={status.error_code}: {status.error_msg}"
            )
        sleep_fn(0.02)


def _joint_map(robot) -> dict[str, float]:
    result = {}
    for state in robot.get_joint_states().get("states", ()):
        name, value = str(state.get("name", "")), float(state.get("motor_position", math.nan))
        if name and math.isfinite(value):
            result[name] = value
        if int(state.get("error_code", 0)) != 0:
            raise RuntimeError(f"joint state error: {name}={state.get('error_code')}")
    return result


def _end_pose(status, frame_name: str):
    names = [str(value) for value in status.frame_names]
    try:
        pose = status.frame_poses[names.index(frame_name)]
    except (ValueError, IndexError) as exc:
        raise RuntimeError(f"GDK motion status does not contain {frame_name}") from exc
    position = (float(pose.position.x), float(pose.position.y), float(pose.position.z))
    quaternion = (float(pose.orientation.x), float(pose.orientation.y), float(pose.orientation.z), float(pose.orientation.w))
    norm = math.sqrt(sum(value*value for value in quaternion))
    if not all(math.isfinite(value) for value in (*position, *quaternion)) or not 0.99 <= norm <= 1.01:
        raise RuntimeError("GDK returned an invalid end-link pose")
    return position, tuple(value/norm for value in quaternion)


def verify_live_model(
    model: UrdfKinematics, robot, side: str, *, max_position_error_m: float = 0.005,
    max_orientation_error_rad: float = math.radians(2.0),
) -> tuple[float, float]:
    names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
    tip = "arm_l_end_link" if side == "left" else "arm_r_end_link"
    joints = _joint_map(robot)
    missing = [name for name in (*BODY_JOINT_NAMES, *names) if name not in joints]
    if missing:
        raise RuntimeError(f"GDK joint state is missing for model check: {missing}")
    predicted = model.forward("base_link", tip, joints)
    status = robot.get_motion_control_status()
    if int(status.error_code) != 0:
        raise RuntimeError(f"cannot verify live model while motion error={status.error_code}")
    position, quaternion = _end_pose(status, tip)
    position_error = float(sum((a-b)**2 for a, b in zip(predicted[:3, 3], position)) ** 0.5)
    orientation_error = float(math.sqrt(sum(value*value for value in rotation_error(
        quaternion_to_matrix(quaternion), predicted[:3, :3],
    ))))
    if position_error > max_position_error_m or orientation_error > max_orientation_error_rad:
        raise RuntimeError(
            f"URDF/GDK end-link mismatch: {position_error*1000:.3f}mm, "
            f"{math.degrees(orientation_error):.3f}deg"
        )
    return position_error, orientation_error


def record_trajectory(
    robot, side: str, *, rate_hz: float = 100.0, duration_s: float | None = None,
    monotonic_ns_fn=time.monotonic_ns, sleep_fn=time.sleep,
) -> CartesianTrajectory:
    if side not in {"left", "right"}:
        raise ValueError("side must be left or right")
    if not math.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError("rate_hz must be positive and finite")
    if duration_s is not None and (not math.isfinite(duration_s) or duration_s < 0):
        raise ValueError("duration_s must be non-negative and finite")
    enter_position_control(robot, monotonic_fn=lambda: monotonic_ns_fn()/1e9, sleep_fn=sleep_fn)
    names = LEFT_ARM_JOINT_NAMES if side == "left" else RIGHT_ARM_JOINT_NAMES
    tip = "arm_l_end_link" if side == "left" else "arm_r_end_link"
    initial_joints = _joint_map(robot)
    missing = [name for name in (*BODY_JOINT_NAMES, *names) if name not in initial_joints]
    if missing:
        raise RuntimeError(f"GDK joint state is missing: {missing}")
    fixed = {name: initial_joints[name] for name in BODY_JOINT_NAMES}
    period_ns = int(round(1e9/rate_hz))
    duration_ns = None if duration_s is None else int(round(duration_s*1e9))
    started = monotonic_ns_fn()
    deadline = started
    samples = []
    try:
        while True:
            now = monotonic_ns_fn()
            status = robot.get_motion_control_status()
            if int(status.error_code) != 0:
                raise RuntimeError(f"motion-control error while recording: {status.error_code} {status.error_msg}")
            joints = _joint_map(robot)
            position, orientation = _end_pose(status, tip)
            samples.append(CartesianSample(
                0 if not samples else now-started, position, orientation,
                tuple(joints[name] for name in names),
            ))
            if duration_ns is not None and now-started >= duration_ns:
                break
            deadline += period_ns
            remaining = deadline-monotonic_ns_fn()
            if remaining > 0:
                sleep_fn(remaining/1e9)
    except KeyboardInterrupt:
        pass
    return CartesianTrajectory(side, "base_link", tip, tuple(names), fixed, tuple(samples))


def ensure_replayable(trajectory: JointTrajectory, report: ValidationReport) -> None:
    if not report.replay_ready:
        detail = "; ".join(report.errors) or "offline validation did not pass"
        raise ValueError(f"trajectory is not replay-ready: {detail}")
    if not trajectory.points:
        raise ValueError("trajectory is not replay-ready: no points")


def _joint_at(trajectory: JointTrajectory, source_t_ns: int) -> tuple[float, ...]:
    if source_t_ns <= 0:
        return trajectory.points[0].joint_positions_rad
    if source_t_ns >= trajectory.points[-1].t_ns:
        return trajectory.points[-1].joint_positions_rad
    for before, after in zip(trajectory.points, trajectory.points[1:]):
        if source_t_ns <= after.t_ns:
            fraction = (source_t_ns-before.t_ns)/(after.t_ns-before.t_ns)
            return tuple(a + fraction*(b-a) for a, b in zip(before.joint_positions_rad, after.joint_positions_rad))
    raise AssertionError("joint trajectory endpoint not found")


def iter_scaled_joint_targets(
    trajectory: JointTrajectory, *, speed_scale: float, rate_hz: float = 100.0,
) -> Iterator[tuple[float, ...]]:
    if not math.isfinite(speed_scale) or speed_scale <= 0.0:
        raise ValueError("speed_scale must be positive and finite")
    if not math.isfinite(rate_hz) or rate_hz <= 0.0:
        raise ValueError("rate_hz must be positive and finite")
    duration = int(round(trajectory.points[-1].t_ns/speed_scale))
    interval = int(round(1e9/rate_hz))
    times = list(range(0, duration, interval))
    times.append(duration)
    for replay_t_ns in times:
        yield _joint_at(trajectory, min(trajectory.points[-1].t_ns, int(round(replay_t_ns*speed_scale))))


def _scaled_target_count(trajectory: JointTrajectory, *, speed_scale: float, rate_hz: float = 100.0) -> int:
    interval = int(round(1e9/rate_hz))
    duration = int(round(trajectory.points[-1].t_ns/speed_scale))
    return len(range(0, duration, interval)) + 1


def replay_trajectory(
    gdk, robot, trajectory: JointTrajectory, report: ValidationReport, *, speed_scale: float,
    start_velocity_rad_s: float = 0.05, fixed_joint_tolerance_rad: float = 0.01,
    max_velocity_rad_s: float = 0.5, max_acceleration_rad_s2: float = 1.0,
    sleep_fn=time.sleep, monotonic_fn=time.monotonic,
    progress_sink: Callable[[str], None] | None = None,
) -> int:
    ensure_replayable(trajectory, report)
    if not math.isfinite(speed_scale) or speed_scale <= 0.0:
        raise ValueError("speed_scale must be positive and finite")
    if not math.isfinite(start_velocity_rad_s) or start_velocity_rad_s <= 0.0:
        raise ValueError("start_velocity_rad_s must be positive and finite")
    if progress_sink is None:
        progress_sink = lambda message: print(message, flush=True)
    limits = ARM_POSITION_LIMITS[:7] if trajectory.side == "left" else ARM_POSITION_LIMITS[7:]
    local_report = validate_joint_trajectory(
        trajectory,
        joint_limits=limits,
        max_velocity_rad_s=max_velocity_rad_s/speed_scale,
        max_acceleration_rad_s2=max_acceleration_rad_s2/(speed_scale*speed_scale),
        max_position_error_m=1e-4,
        max_orientation_error_rad=1e-3,
    )
    if not local_report.replay_ready:
        raise ValueError("trajectory failed local replay validation: " + "; ".join(local_report.errors))
    current = _joint_map(robot)
    for name, recorded in trajectory.fixed_joint_positions.items():
        if name not in current or abs(current[name]-recorded) > fixed_joint_tolerance_rad:
            raise RuntimeError(f"fixed joint {name} differs from recording")
    enter_position_control(robot, monotonic_fn=monotonic_fn, sleep_fn=sleep_fn)
    request = gdk.JointControlReq()
    request.detail = f"absolute Cartesian teaching {trajectory.side} start"
    request.life_time = 30.0
    request.joint_names = list(trajectory.joint_names)
    request.joint_positions = list(trajectory.points[0].joint_positions_rad)
    request.joint_velocities = [float(start_velocity_rad_s)] * 7
    robot.joint_control_request(request)
    deadline = monotonic_fn() + 30.0
    while True:
        state = _joint_map(robot)
        error = max(abs(state[name]-target) for name, target in zip(trajectory.joint_names, request.joint_positions))
        if error <= 0.03:
            break
        if monotonic_fn() >= deadline:
            raise RuntimeError("trajectory start was not reached")
        sleep_fn(0.05)
    group = 0 if trajectory.side == "left" else 1
    send_deadline = monotonic_fn()
    sent = 0
    total_targets = _scaled_target_count(trajectory, speed_scale=speed_scale, rate_hz=100.0)
    next_progress_percent = 0
    for target in iter_scaled_joint_targets(trajectory, speed_scale=speed_scale, rate_hz=100.0):
        status = robot.get_motion_control_status()
        if int(status.control_mode) != 1 or int(status.error_code) != 0:
            raise RuntimeError("position control became unhealthy during replay")
        robot.move_arm_joint_servo(list(target), 0.01, group, False)
        sent += 1
        completed = min(sent-1, max(0, total_targets-1))
        progress_percent = 100 if total_targets <= 1 else int(100*completed/(total_targets-1))
        if progress_percent >= next_progress_percent:
            progress_sink(f"回放点进度：{min(progress_percent, 100)}%")
            next_progress_percent = (progress_percent//10+1)*10
        send_deadline += 0.01
        remaining = send_deadline-monotonic_fn()
        if remaining > 0:
            sleep_fn(remaining)
    deadline = monotonic_fn() + 5.0
    final = trajectory.points[-1].joint_positions_rad
    while True:
        state = _joint_map(robot)
        if max(abs(state[name]-target) for name, target in zip(trajectory.joint_names, final)) <= 0.03:
            break
        if monotonic_fn() >= deadline:
            raise RuntimeError("trajectory endpoint was not reached")
        sleep_fn(0.05)
    return sent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="G2 独立绝对末端位姿示教、离线 IK 与位控回放")
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="用机械臂拖动按钮示教，记录 end_link 绝对位姿")
    record.add_argument("--side", choices=("left", "right"), required=True)
    record.add_argument("--output", type=Path, required=True)
    record.add_argument("--rate-hz", type=float, default=100.0)
    record.add_argument("--duration-s", type=float)
    record.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    record.add_argument("--leading-motion-mm", type=float, default=0.5,
                        help="保留首个明显移动点之前的姿态；默认移动超过0.5mm")
    record.add_argument("--leading-motion-deg", type=float, default=0.5,
                        help="保留首个明显移动点之前的姿态；默认旋转超过0.5度")
    record.add_argument("--allow-motion", action="store_true", help="确认允许切换到 control_mode=1；录制不发送运动目标")
    solve = commands.add_parser("solve", help="离线把整条绝对末端轨迹解算成连续关节轨迹")
    solve.add_argument("--input", type=Path, required=True)
    solve.add_argument("--output", type=Path, required=True)
    solve.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    solve.add_argument("--rate-hz", type=float, default=100.0)
    solve.add_argument("--max-velocity-rad-s", type=float, default=0.5)
    solve.add_argument("--max-acceleration-rad-s2", type=float, default=2.0)
    inspect = commands.add_parser("inspect", help="离线查看轨迹和验证结果")
    inspect.add_argument("--input", type=Path, required=True)
    replay = commands.add_parser("replay", help="位控复现已完成整轨 IK 验证的关节轨迹")
    replay.add_argument("--input", type=Path, required=True)
    replay.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    replay.add_argument("--speed-scale", type=float, default=0.2)
    replay.add_argument("--start-velocity-rad-s", type=float, default=0.05)
    replay.add_argument("--max-velocity-rad-s", type=float, default=0.5)
    replay.add_argument("--max-acceleration-rad-s2", type=float, default=2.0)
    replay.add_argument("--allow-motion", action="store_true")
    return parser


def _open_gdk():
    import agibot_gdk
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK 初始化失败")
    return agibot_gdk, agibot_gdk.Robot()


def _solve(args) -> int:
    source = args.input.expanduser().resolve()
    urdf = args.urdf.expanduser().resolve()
    trajectory = load_cartesian_document(json.loads(source.read_text(encoding="utf-8")))
    resampled = resample_cartesian_trajectory(trajectory, rate_hz=args.rate_hz)
    limits = ARM_POSITION_LIMITS[:7] if trajectory.side == "left" else ARM_POSITION_LIMITS[7:]
    solved, report = solve_cartesian_trajectory(
        resampled, UrdfKinematics(urdf), joint_limits=limits,
        source_sha256=_sha256(source), urdf_sha256=_sha256(urdf),
        max_velocity_rad_s=args.max_velocity_rad_s,
        max_acceleration_rad_s2=args.max_acceleration_rad_s2,
    )
    document = joint_document(
        solved, report, solved_at_utc=datetime.now(timezone.utc).isoformat(),
        solve_rate_hz=args.rate_hz,
    )
    _atomic_json(args.output.expanduser().resolve(), document)
    print(f"整轨 IK：{len(solved.points)} 点；replay_ready={report.replay_ready}")
    print(f"最大 FK 误差：{report.max_position_error_m*1000:.3f} mm，{math.degrees(report.max_orientation_error_rad):.3f}°")
    if report.errors:
        print("验证失败：" + "; ".join(report.errors), file=sys.stderr)
    return 0 if report.replay_ready else 1


def _inspect(args) -> int:
    document = json.loads(args.input.expanduser().read_text(encoding="utf-8"))
    if document.get("kind") == "g2_absolute_cartesian_trajectory":
        trajectory = load_cartesian_document(document)
        print(json.dumps({"kind": document["kind"], "side": trajectory.side,
                          "samples": len(trajectory.samples), "duration_s": trajectory.samples[-1].t_ns/1e9,
                          "base_frame": trajectory.base_frame, "tip_frame": trajectory.tip_frame}, ensure_ascii=False, indent=2))
    else:
        trajectory, report = load_joint_document(document)
        print(json.dumps({"kind": document.get("kind"), "side": trajectory.side,
                          "points": len(trajectory.points), "duration_s": trajectory.points[-1].t_ns/1e9,
                          "validation": report.__dict__}, ensure_ascii=False, indent=2))
    return 0


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    gdk = None
    try:
        if args.command == "solve":
            return _solve(args)
        if args.command == "inspect":
            return _inspect(args)
        if not args.allow_motion:
            print("该命令会切换或驱动机器人位控；请显式传入 --allow-motion。", file=sys.stderr)
            return 2
        gdk, robot = _open_gdk()
        if args.command == "record":
            urdf = args.urdf.expanduser().resolve()
            if (
                not math.isfinite(args.leading_motion_mm) or args.leading_motion_mm < 0.0
                or not math.isfinite(args.leading_motion_deg) or args.leading_motion_deg < 0.0
            ):
                raise ValueError("leading motion thresholds must be non-negative and finite")
            print("已切换到位控录制；请按住机械臂自身拖动按钮进行示教，Ctrl+C 结束。", flush=True)
            trajectory = record_trajectory(robot, args.side, rate_hz=args.rate_hz, duration_s=args.duration_s)
            original_samples = len(trajectory.samples)
            trajectory, removed = trim_leading_stationary(
                trajectory,
                translation_threshold_m=args.leading_motion_mm/1000.0,
                rotation_threshold_rad=math.radians(args.leading_motion_deg),
            )
            _atomic_json(args.output.expanduser().resolve(), cartesian_document(
                trajectory, urdf_sha256=_sha256(urdf), recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                sample_rate_hz=args.rate_hz,
                leading_stationary_trim={"removed_samples": removed, "original_samples": original_samples,
                                         "translation_threshold_m": args.leading_motion_mm/1000.0,
                                         "rotation_threshold_deg": args.leading_motion_deg},
            ))
            print(f"已删除开头静止 {removed} 点；保存 {len(trajectory.samples)} 个绝对末端位姿点：{args.output}")
            return 0
        source = args.input.expanduser().resolve()
        trajectory, report = load_joint_document(json.loads(source.read_text(encoding="utf-8")))
        urdf = args.urdf.expanduser().resolve()
        if trajectory.urdf_sha256 != _sha256(urdf):
            raise ValueError("当前 URDF 与离线 IK 使用的 URDF 不一致")
        model_error = verify_live_model(UrdfKinematics(urdf), robot, trajectory.side)
        print(f"模型核对通过：{model_error[0]*1000:.3f} mm，{math.degrees(model_error[1]):.3f}°")
        print(f"轨迹已通过离线验证；即将以 {args.speed_scale:g} 倍速位控回放。", flush=True)
        sent = replay_trajectory(
            gdk, robot, trajectory, report, speed_scale=args.speed_scale,
            start_velocity_rad_s=args.start_velocity_rad_s,
            max_velocity_rad_s=args.max_velocity_rad_s,
            max_acceleration_rad_s2=args.max_acceleration_rad_s2,
        )
        print(f"回放完成：发送 {sent} 个 100 Hz 单臂关节目标。")
        return 0
    except KeyboardInterrupt:
        print("已由操作者中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"绝对末端示教操作失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if gdk is not None:
            try:
                gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
