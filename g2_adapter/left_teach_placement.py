"""Borrow the flow's robot session to replay a taught left-arm placement."""

import math
import time
from contextlib import contextmanager
from pathlib import Path

try:
    from .teach_trajectory_gdk import G2TrajectorySession
    from .teach_placement import load_teach_spec, replay_solved_trajectory
    from .spacemouse_takeover import TeachReplayTakenOver
    from .safety import AdvanceRightRequested
except ImportError:
    from teach_trajectory_gdk import G2TrajectorySession
    from teach_placement import load_teach_spec, replay_solved_trajectory
    from spacemouse_takeover import TeachReplayTakenOver
    from safety import AdvanceRightRequested


def read_left_trajectory(path):
    if path is None:
        raise ValueError("teach 模式需要 --left-teach-trajectory")
    return load_teach_spec(path, "left")


def validate_teach_args(args):
    trajectory = read_left_trajectory(args.left_teach_trajectory)
    for value in (args.left_teach_speed_scale, args.left_teach_start_velocity_rad_s):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("示教倍率和起点速度必须为正有限值")
    return trajectory


class _GatedRobot:
    def __init__(self, robot, gate, *, stop_on_takeover=False, automatic_guard=None):
        self.robot = robot
        self.automatic_guard = automatic_guard
        self.gate = gate
        self.paused_s = 0.0
        self.stop_on_takeover = stop_on_takeover
        self.generation = getattr(gate, "pause_generation", 0)
        self.sent = 0

    def __getattr__(self, name):
        function = getattr(self.robot, name)
        if name not in {
            "get_joint_states",
            "set_control_mode",
            "set_joint_impedance",
            "joint_control_request",
            "move_arm_joint_servo",
        } or self.gate is None:
            return function

        def write(*args, **kwargs):
            if self.automatic_guard is not None:
                self.automatic_guard()
            started = time.monotonic()
            reserve = self.gate.automatic_teach_motion(self.generation) if self.stop_on_takeover and hasattr(self.gate, "automatic_teach_motion") else self.gate.automatic_motion()
            with reserve:
                self.paused_s += time.monotonic() - started
                result = function(*args, **kwargs)
                if name == "move_arm_joint_servo":
                    self.sent += 1
                return result
        return write

    def clock(self):
        return time.monotonic() - self.paused_s


@contextmanager
def end_on_takeover(result, robot, label, worker=None):
    previous = getattr(worker, "allow_advance_right", False)
    if worker is not None:
        worker.allow_advance_right = True
    try:
        yield
    except AdvanceRightRequested:
        result.update(interrupted_by_advance=True, sent=robot.sent)
        print(f"{label}当前移动已结束；进入下一阶段。", flush=True)
    except TeachReplayTakenOver:
        result.update(interrupted_by_takeover=True, sent=robot.sent)
        print(f"{label}已切换为 SpaceMouse 手动控制；自动移动已停止。", flush=True)
    finally:
        if worker is not None:
            worker.allow_advance_right = previous


def move_left_to_teach_start(services, args):
    spec = validate_teach_args(args)
    robot = _GatedRobot(services.robot, services.gate,
        automatic_guard=getattr(getattr(services, 'controller', None), 'check_automatic_guard', None))
    print("正在移动到左手装配准备位置…", flush=True)
    if spec.kind == "cartesian_solved":
        replay_solved_trajectory(services.gdk, robot, spec,
            speed_scale=args.left_teach_speed_scale,
            start_velocity_rad_s=args.left_teach_start_velocity_rad_s,
            max_velocity_rad_s=getattr(args, "teach_max_velocity_rad_s", 1.0),
            max_acceleration_rad_s2=getattr(args, "teach_max_acceleration_rad_s2", 15.0),
            start_only=True)
    else:
        session = G2TrajectorySession(services.gdk, robot, "left",
            controller=getattr(services, "controller", None), monotonic_fn=robot.clock, event_sink=print)
        session.controller.prepare_joint_impedance_control(("left",))
        session.controller._require_joint_impedance_mode("teach trajectory start")
        initial = spec.trajectory.samples[0].joint_positions_rad
        session._plan_to_start(initial, velocity_rad_s=args.left_teach_start_velocity_rad_s)
        session._wait_for_start(initial)
    print("左手装配准备位置已到达。", flush=True)
    return {"method": "teach_start", "trajectory": str(spec.path)}


def replay_left_placement(services, args):
    spec = validate_teach_args(args)
    robot = _GatedRobot(services.robot, services.gate, stop_on_takeover=True,
                        automatic_guard=getattr(getattr(services, 'controller', None), 'check_automatic_guard', None))
    if spec.kind == "cartesian_solved":
        result = {
            "method": "teach",
            "trajectory_type": "cartesian_solved",
            "trajectory": str(spec.path),
            "speed_scale": args.left_teach_speed_scale,
        }
        with end_on_takeover(result, robot, "左手", getattr(services, "worker", None)):
            result["sent"] = replay_solved_trajectory(
                services.gdk,
                robot,
                spec,
                speed_scale=args.left_teach_speed_scale,
                start_velocity_rad_s=args.left_teach_start_velocity_rad_s,
                max_velocity_rad_s=getattr(args, "teach_max_velocity_rad_s", 1.0),
                max_acceleration_rad_s2=getattr(args, "teach_max_acceleration_rad_s2", 15.0),
                move_to_start=False,
            )
            print("左手铰链放置动作已完成。", flush=True)
        return result
    trajectory = spec.trajectory
    session = G2TrajectorySession(
        services.gdk,
        robot,
        "left",
        controller=getattr(services, "controller", None),
        monotonic_fn=robot.clock,
        event_sink=lambda message: None if message.startswith("移动进度：") else print(message),
    )
    result = {"method": "teach", "trajectory": str(spec.path), "speed_scale": args.left_teach_speed_scale}
    with end_on_takeover(result, robot, "左手", getattr(services, "worker", None)):
        result["sent"] = session.replay(trajectory, speed_scale=args.left_teach_speed_scale, start_velocity_rad_s=args.left_teach_start_velocity_rad_s, move_to_start=False)
        session._wait_for_start(trajectory.samples[-1].joint_positions_rad)
        print("左手铰链放置动作已完成。", flush=True)
    return result
