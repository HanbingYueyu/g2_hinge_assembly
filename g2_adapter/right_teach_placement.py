"""Right-arm taught continuation after the original hole-offset transport."""

import math
from copy import copy
from pathlib import Path
from types import SimpleNamespace

try:
    from .left_teach_placement import _GatedRobot, end_on_takeover
    from .teach_placement import load_teach_spec, replay_solved_trajectory
    from .teach_trajectory_gdk import G2TrajectorySession
except ImportError:
    from left_teach_placement import _GatedRobot, end_on_takeover
    from teach_placement import load_teach_spec, replay_solved_trajectory
    from teach_trajectory_gdk import G2TrajectorySession


def right_trajectory_paths(args):
    paths = getattr(args, "right_teach_trajectories", None)
    if paths is not None:
        if len(paths) != 4:
            raise ValueError("--right-teach-trajectories 需要按顺序指定4个轨迹文件")
        return list(paths)
    if args.right_teach_trajectory is None:
        raise ValueError("teach 模式需要 --right-teach-trajectories 或 --right-teach-trajectory")
    return [args.right_teach_trajectory]


def validate_right_teach_args(args):
    trajectories = []
    for path in right_trajectory_paths(args):
        trajectories.append(load_teach_spec(path, "right"))
    for value in (args.right_teach_speed_scale, args.right_teach_start_velocity_rad_s):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("示教倍率和起点速度必须为正有限值")
    return trajectories


def run_right_teach_sequence(services, args, runtime):
    paths = right_trajectory_paths(args)
    results = []
    for index, path in enumerate(paths, 1):
        if index > 1:
            print(f"[右手示教] 等待右键长按2秒开始第 {index}/{len(paths)} 段。", flush=True)
            runtime.wait_for_advance(SimpleNamespace(stage=f"right_teach_gate_{index}"))
        print(f"[右手示教] 开始第 {index}/{len(paths)} 段：{path}", flush=True)
        segment_args = copy(args)
        segment_args.right_teach_trajectory = path
        segment_args.right_teach_trajectories = None
        runtime.emit("right_teach_segment_start", segment=index, trajectory=str(path))
        result = replay_right_placement(services, segment_args)
        results.append(result)
        runtime.emit("right_teach_segment_complete", segment=index, **result)
    return results[0] if len(results) == 1 else {"method": "teach", "segments": results}


def replay_right_placement(services, args):
    specs = validate_right_teach_args(args)
    spec = specs[0]
    robot = _GatedRobot(
        services.robot,
        services.gate,
        stop_on_takeover=True,
        automatic_guard=getattr(getattr(services, "controller", None), "check_automatic_guard", None),
    )
    if spec.kind == "cartesian_solved":
        result = {
            "method": "teach",
            "trajectory_type": "cartesian_solved",
            "trajectory": str(spec.path),
            "speed_scale": args.right_teach_speed_scale,
        }
        with end_on_takeover(result, robot, "右手", getattr(services, "worker", None)):
            result["sent"] = replay_solved_trajectory(
                services.gdk,
                robot,
                spec,
                speed_scale=args.right_teach_speed_scale,
                start_velocity_rad_s=args.right_teach_start_velocity_rad_s,
                max_velocity_rad_s=getattr(args, "teach_max_velocity_rad_s", 1.0),
                max_acceleration_rad_s2=getattr(args, "teach_max_acceleration_rad_s2", 15.0),
            )
            print(f"右手末端示教回放完成：发送 {result['sent']} 个关节目标。", flush=True)
        return result
    trajectory = spec.trajectory
    session = G2TrajectorySession(
        services.gdk,
        robot,
        "right",
        controller=getattr(services, "controller", None),
        monotonic_fn=robot.clock,
        event_sink=print,
    )
    result = {"method": "teach", "trajectory": str(spec.path), "speed_scale": args.right_teach_speed_scale}
    with end_on_takeover(result, robot, "右手", getattr(services, "worker", None)):
        result["sent"] = session.replay(trajectory, speed_scale=args.right_teach_speed_scale, start_velocity_rad_s=args.right_teach_start_velocity_rad_s)
        session._wait_for_start(trajectory.samples[-1].joint_positions_rad)
        print(f"右手示教回放完成：发送 {result['sent']} 个关节目标。", flush=True)
    return result
