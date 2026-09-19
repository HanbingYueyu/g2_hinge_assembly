#!/usr/bin/env python3
"""Standalone single-arm hand-guided trajectory teaching for G2."""

from __future__ import annotations

import argparse
from bisect import bisect_right
import json
import math
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from collections.abc import Callable, Sequence
from pathlib import Path

try:
    from .control import G2Controller, STANDARD_JOINT_IMPEDANCE_KP
    from .safety import (
        ARM_JOINT_NAMES,
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        reorder_arm_positions,
        validate_arm_target,
    )
    from .teach_trajectory import (
        Trajectory,
        build_trajectory,
        iter_smooth_targets,
        load_trajectory,
        trajectory_document,
        trim_leading_stationary,
    )
except ImportError:
    from control import G2Controller, STANDARD_JOINT_IMPEDANCE_KP
    from safety import (
        ARM_JOINT_NAMES,
        LEFT_ARM_JOINT_NAMES,
        RIGHT_ARM_JOINT_NAMES,
        reorder_arm_positions,
        validate_arm_target,
    )
    from teach_trajectory import (
        Trajectory,
        build_trajectory,
        iter_smooth_targets,
        load_trajectory,
        trajectory_document,
        trim_leading_stationary,
    )


LOW_TEACH_KP = (50.0, 50.0, 50.0, 60.0, 30.0, 30.0, 30.0)
REPLAY_RATE_HZ = 100.0
REPLAY_CONTROL_PERIOD_S = 1.0 / REPLAY_RATE_HZ


class G2TrajectorySession:
    """One selected G2 arm's teach and replay lifecycle."""

    def __init__(
        self,
        gdk,
        robot,
        side: str,
        *,
        controller: G2Controller | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        monotonic_ns_fn: Callable[[], int] = time.monotonic_ns,
        monotonic_fn: Callable[[], float] = time.monotonic,
        event_sink: Callable[[str], None] | None = None,
    ) -> None:
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        self.gdk = gdk
        self.robot = robot
        self.side = side
        # All mode and impedance writes go through the single verified
        # controller boundary.  Production callers pass the orchestrator's
        # controller; a standalone caller gets one built over the same robot.
        self.controller = (
            controller if controller is not None else G2Controller(gdk, robot, allow_motion=True)
        )
        self._sleep = sleep_fn
        self._monotonic_ns = monotonic_ns_fn
        self._monotonic = monotonic_fn
        self._event_sink = event_sink if event_sink is not None else lambda _: None

    @property
    def joint_names(self) -> tuple[str, ...]:
        return LEFT_ARM_JOINT_NAMES if self.side == "left" else RIGHT_ARM_JOINT_NAMES

    @property
    def _offset(self) -> int:
        return 0 if self.side == "left" else 7

    @property
    def _impedance_group(self) -> int:
        return 0 if self.side == "left" else 1

    def _arm_state(self) -> tuple[float, ...]:
        return reorder_arm_positions(self.robot.get_joint_states())

    def _selected_state(self) -> tuple[float, ...]:
        return self._arm_state()[self._offset : self._offset + 7]

    def record(
        self,
        *,
        duration_s: float | None = None,
        rate_hz: float = 100.0,
        kp: Sequence[float] = LOW_TEACH_KP,
        stop_requested: Callable[[], bool] | None = None,
    ) -> Trajectory:
        """Record actual selected-arm positions while joint impedance is active."""
        if duration_s is not None and (not math.isfinite(duration_s) or duration_s < 0.0):
            raise ValueError("duration_s must be non-negative and finite")
        if not math.isfinite(rate_hz) or rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive and finite")
        values = tuple(float(value) for value in kp)
        if len(values) != 7 or not all(math.isfinite(value) for value in values):
            raise ValueError("kp must contain seven finite values")
        period_ns = int(round(1_000_000_000 / rate_hz))
        duration_ns = None if duration_s is None else int(round(duration_s * 1_000_000_000))
        raw_samples: list[tuple[int, tuple[float, ...]]] = []
        self.controller.set_joint_impedance(self.side, values)
        try:
            self._plan_to_start(self._selected_state(), velocity_rad_s=0.05)
            started_ns = self._monotonic_ns()
            next_sample_ns = started_ns
            while True:
                if raw_samples and stop_requested is not None and stop_requested():
                    break
                now_ns = self._monotonic_ns()
                elapsed_ns = now_ns - started_ns
                raw_samples.append((0 if not raw_samples else elapsed_ns, self._selected_state()))
                if duration_ns is not None and elapsed_ns >= duration_ns:
                    break
                next_sample_ns += period_ns
                remaining_ns = next_sample_ns - self._monotonic_ns()
                if remaining_ns > 0:
                    self._sleep(remaining_ns / 1_000_000_000)
        except KeyboardInterrupt:
            pass
        finally:
            self.controller.set_joint_impedance(self.side, STANDARD_JOINT_IMPEDANCE_KP)
        return build_trajectory(self.side, raw_samples)

    def _plan_to_start(self, target: Sequence[float], *, velocity_rad_s: float) -> None:
        request = self.gdk.JointControlReq()
        request.detail = f"teach trajectory {self.side} start"
        request.life_time = 30.0
        request.joint_names = list(self.joint_names)
        request.joint_positions = [float(value) for value in target]
        request.joint_velocities = [float(velocity_rad_s)] * 7
        self.robot.joint_control_request(request)

    def _wait_for_start(self, target: Sequence[float], *, tolerance_rad: float = 0.03, timeout_s: float = 30.0) -> None:
        deadline = self._monotonic() + timeout_s
        while True:
            actual = self._selected_state()
            if max(abs(current - desired) for current, desired in zip(actual, target)) <= tolerance_rad:
                return
            if self._monotonic() >= deadline:
                raise RuntimeError("recorded trajectory start was not reached before replay")
            self._sleep(0.05)

    def replay(
        self,
        trajectory: Trajectory,
        *,
        speed_scale: float = 0.2,
        start_velocity_rad_s: float = 0.05,
    ) -> int:
        """Move to the recorded start, then slowly replay a selected-arm path."""
        if trajectory.side != self.side or trajectory.joint_names != self.joint_names:
            raise ValueError("trajectory does not match this session arm")
        if not math.isfinite(speed_scale) or speed_scale <= 0.0:
            raise ValueError("speed_scale must be positive and finite")
        if not math.isfinite(start_velocity_rad_s) or start_velocity_rad_s <= 0.0:
            raise ValueError("start_velocity_rad_s must be positive and finite")
        self.controller.prepare_joint_impedance_control((self.side,))
        self.controller._require_joint_impedance_mode("teach trajectory replay start")
        initial = trajectory.samples[0].joint_positions_rad
        self._event_sink(f"正在移动到录制起点（{self.side} 臂）…")
        self._plan_to_start(initial, velocity_rad_s=start_velocity_rad_s)
        self._wait_for_start(initial)
        stationary_other_arm = self._arm_state()[7:] if self.side == "left" else self._arm_state()[:7]
        targets = iter_smooth_targets(trajectory, rate_hz=REPLAY_RATE_HZ, speed_scale=speed_scale)
        replay_duration_ns = int(round(trajectory.duration_ns / speed_scale))
        replay_duration_s = replay_duration_ns / 1_000_000_000
        recorded_sample_times_ns = tuple(sample.t_ns for sample in trajectory.samples)
        self._event_sink(
            f"起点已到达；开始以 {REPLAY_RATE_HZ:.0f} Hz 慢放，预计 {replay_duration_s:.1f} 秒。"
        )
        deadline = self._monotonic()
        last_health_check = deadline
        sent = 0
        next_progress_percent = 0
        for target in targets:
            complete = (
                target.joint_positions_rad + stationary_other_arm
                if self.side == "left"
                else stationary_other_arm + target.joint_positions_rad
            )
            validated = validate_arm_target(complete)
            now = self._monotonic()
            if now - last_health_check >= 0.1:
                self.controller._require_joint_impedance_mode("teach trajectory replay")
                last_health_check = now
            self.robot.move_arm_joint_servo(
                list(validated[self._offset : self._offset + 7]),
                REPLAY_CONTROL_PERIOD_S,
                self._impedance_group,
                False,
            )
            sent += 1
            source_t_ns = min(trajectory.duration_ns, int(round(target.t_ns * speed_scale)))
            completed_sample_index = bisect_right(recorded_sample_times_ns, source_t_ns) - 1
            progress_percent = (
                int(100 * completed_sample_index / (len(trajectory.samples) - 1))
                if len(trajectory.samples) > 1
                else 100
            )
            if progress_percent >= next_progress_percent:
                self._event_sink(f"回放点进度：{min(progress_percent, 100)}%")
                next_progress_percent = (progress_percent // 10 + 1) * 10
            deadline += REPLAY_CONTROL_PERIOD_S
            remaining_s = deadline - self._monotonic()
            if remaining_s > 0.0:
                self._sleep(remaining_s)
        return sent


def _atomic_write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="独立的 G2 单臂拖动示教和慢速关节轨迹复现")
    commands = parser.add_subparsers(dest="command", required=True)
    teach = commands.add_parser("teach", help="进入关节阻抗模式并记录实际关节角")
    teach.add_argument("--side", required=True, choices=("left", "right"))
    teach.add_argument("--output", required=True, type=Path)
    teach.add_argument("--rate-hz", type=float, default=100.0)
    teach.add_argument("--duration-s", type=float, default=None, help="省略时按 Ctrl+C 停止记录")
    teach.add_argument("--allow-motion", action="store_true", help="确认切换机器人控制模式")
    inspect = commands.add_parser("inspect", help="离线检查轨迹文件，不连接机器人")
    inspect.add_argument("--input", required=True, type=Path)
    replay = commands.add_parser("replay", help="低速复现已记录的单臂关节轨迹")
    replay.add_argument("--input", required=True, type=Path)
    replay.add_argument("--speed-scale", type=float, default=0.2)
    replay.add_argument("--start-velocity-rad-s", type=float, default=0.05)
    replay.add_argument("--allow-motion", action="store_true", help="确认机器人将移动")
    return parser.parse_args(argv)


def _inspect(path: Path) -> dict:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    trajectory = load_trajectory(document)
    maximum_rate = 0.0
    for before, after in zip(trajectory.samples, trajectory.samples[1:]):
        interval_s = (after.t_ns - before.t_ns) / 1_000_000_000
        maximum_rate = max(
            maximum_rate,
            max(abs(end - start) / interval_s for start, end in zip(before.joint_positions_rad, after.joint_positions_rad)),
        )
    return {
        "side": trajectory.side,
        "samples": len(trajectory.samples),
        "duration_s": trajectory.duration_ns / 1_000_000_000,
        "max_sample_delta_rad_s": maximum_rate,
    }


def _open_gdk():
    try:
        import agibot_gdk
    except ImportError as exc:
        raise RuntimeError("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行") from exc
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK 初始化失败")
    return agibot_gdk, agibot_gdk.Robot()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "inspect":
        try:
            print(json.dumps(_inspect(args.input), ensure_ascii=False, indent=2))
            return 0
        except Exception as exc:
            print(f"轨迹检查失败：{exc}", file=sys.stderr)
            return 1
    if not args.allow_motion:
        print(f"{args.command} 会改变机器人控制状态；请显式传入 --allow-motion。", file=sys.stderr)
        return 2
    gdk = None
    try:
        gdk, robot = _open_gdk()
        controller = G2Controller(gdk, robot, allow_motion=True)
        if args.command == "teach":
            session = G2TrajectorySession(gdk, robot, args.side, controller=controller)
            print("已进入示教准备：确认末端负载参数正确后可拖动手臂；按 Ctrl+C 结束记录。", flush=True)
            trajectory = session.record(duration_s=args.duration_s, rate_hz=args.rate_hz)
            raw_samples = len(trajectory.samples)
            original_duration_ns = trajectory.duration_ns
            trajectory, removed = trim_leading_stationary(trajectory)
            document = trajectory_document(trajectory)
            document["recorded_at_utc"] = datetime.now(timezone.utc).isoformat()
            document["sample_rate_hz"] = args.rate_hz
            document["leading_stationary_trim"] = {"threshold_deg": 0.5, "removed_samples": removed, "removed_duration_ns": original_duration_ns - trajectory.duration_ns, "original_samples": raw_samples}
            _atomic_write_json(args.output.expanduser().resolve(), document)
            print(f"已删除开头静止 {removed} 个样本；已保存 {len(trajectory.samples)} 个样本：{args.output}")
            return 0
        trajectory = load_trajectory(json.loads(args.input.expanduser().read_text(encoding="utf-8")))
        session = G2TrajectorySession(
            gdk,
            robot,
            trajectory.side,
            controller=controller,
            event_sink=lambda message: print(message, flush=True),
        )
        sent = session.replay(
            trajectory,
            speed_scale=args.speed_scale,
            start_velocity_rad_s=args.start_velocity_rad_s,
        )
        print(f"回放完成：发送 {sent} 个 {REPLAY_RATE_HZ:.0f} Hz 关节目标。")
        return 0
    except KeyboardInterrupt:
        print("已由操作者中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"示教轨迹操作失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if gdk is not None:
            try:
                gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
