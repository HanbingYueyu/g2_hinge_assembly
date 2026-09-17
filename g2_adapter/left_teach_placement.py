"""Borrow the flow's robot session to replay a taught left-arm placement."""

import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

try:
    from .teach_trajectory import load_trajectory
    from .teach_trajectory_gdk import G2TrajectorySession
    from .spacemouse_takeover import TeachReplayTakenOver
    from .safety import AdvanceRightRequested
except ImportError:
    from teach_trajectory import load_trajectory
    from teach_trajectory_gdk import G2TrajectorySession
    from spacemouse_takeover import TeachReplayTakenOver
    from safety import AdvanceRightRequested


def read_left_trajectory(path):
    if path is None:
        raise ValueError("teach 模式需要 --left-teach-trajectory")
    trajectory = load_trajectory(json.loads(Path(path).expanduser().read_text(encoding="utf-8")))
    if trajectory.side != "left":
        raise ValueError("放置示教轨迹必须是左臂")
    return trajectory


def validate_teach_args(args):
    trajectory = read_left_trajectory(args.left_teach_trajectory)
    for value in (args.left_teach_speed_scale, args.left_teach_start_velocity_rad_s):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("示教倍率和起点速度必须为正有限值")
    return trajectory


class _GatedRobot:
    def __init__(self, robot, gate, *, stop_on_takeover=False):
        self.robot = robot
        self.gate = gate
        self.paused_s = 0.0
        self.stop_on_takeover = stop_on_takeover
        self.generation = getattr(gate, "pause_generation", 0)
        self.sent = 0

    def __getattr__(self, name):
        function = getattr(self.robot, name)
        if name not in {"get_joint_states", "set_control_mode", "joint_control_request", "move_arm_joint_servo"} or self.gate is None:
            return function

        def write(*args, **kwargs):
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
        print(f"{label}示教已结束；进入下一阶段。", flush=True)
    except TeachReplayTakenOver:
        result.update(interrupted_by_takeover=True, sent=robot.sent)
        print(f"{label}示教已因 SpaceMouse 接管结束；不再回放或等待终点。", flush=True)
    finally:
        if worker is not None:
            worker.allow_advance_right = previous


def replay_left_placement(services, args):
    trajectory = validate_teach_args(args)
    robot = _GatedRobot(services.robot, services.gate, stop_on_takeover=True)
    session = G2TrajectorySession(services.gdk, robot, "left", monotonic_fn=robot.clock, event_sink=print)
    result = {"method": "teach", "trajectory": str(Path(args.left_teach_trajectory).expanduser().resolve()), "speed_scale": args.left_teach_speed_scale}
    with end_on_takeover(result, robot, "左手", getattr(services, "worker", None)):
        result["sent"] = session.replay(trajectory, speed_scale=args.left_teach_speed_scale, start_velocity_rad_s=args.left_teach_start_velocity_rad_s)
        session._wait_for_start(trajectory.samples[-1].joint_positions_rad)
        print(f"左手示教放置完成：发送 {result['sent']} 个关节目标。", flush=True)
    return result
