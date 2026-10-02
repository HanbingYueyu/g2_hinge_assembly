"""Terminal-selected, repeatable grasp demonstrations using the staged GDK context."""

from __future__ import annotations

import json
import os
import select
import sys
import termios
import threading
import tty
from pathlib import Path

import numpy as np

try:
    from .control import CartesianPoseTarget
    from .safety import SafetyError
except ImportError:
    from control import CartesianPoseTarget
    from safety import SafetyError


MODE_LABELS = {
    '1': '抓取铰链',
    '2': '抓取铰链并移动',
    '3': '抓取电批',
}


class DemoModeSelection:
    """The input thread only queues a request; robot code commits it between rounds."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self.active = None
        self.pending = None

    def choose(self, key: str) -> str:
        if key not in (*MODE_LABELS, 'q'):
            raise ValueError('请选择 1、2、3 或 q')
        with self._lock:
            self.pending = key
            self._ready.set()
        return '完成当前轮后停止' if key == 'q' else MODE_LABELS[key]

    def wait_first(self) -> str | None:
        while True:
            self._ready.wait()
            with self._lock:
                if self.pending is not None:
                    self.active = self.pending
                    self.pending = None
                    self._ready.clear()
                    return None if self.active == 'q' else self.active

    def next_round(self) -> str | None:
        with self._lock:
            if self.pending is not None:
                self.active = self.pending
                self.pending = None
                self._ready.clear()
            return None if self.active == 'q' else self.active


class DemoKeyboard:
    """Read 1/2/3/q without sending robot motion from the input thread."""

    def __init__(self, selection: DemoModeSelection, stream=None):
        self.selection = selection
        self.stream = sys.stdin if stream is None else stream
        self._stop = threading.Event()
        self._thread = None
        self._fd = None
        self._saved = None
        self._cbreak = False

    def start(self):
        try:
            self._fd = self.stream.fileno()
            self._saved = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._cbreak = True
        except (AttributeError, OSError, ValueError, termios.error):
            self._saved = None
        self._thread = threading.Thread(target=self._run, name='g2-demo-keyboard', daemon=True)
        self._thread.start()

    def _run(self):
        pending = b''
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._fd], [], [], .1)
                if not readable:
                    continue
                data = os.read(self._fd, 1024)
                if not data:
                    self.selection.choose('q')
                    return
                keys = data if self._cbreak else ()
                if not self._cbreak:
                    pending += data
                    lines = pending.split(b'\n')
                    pending = lines.pop()
                    keys = [line.strip()[:1][0] for line in lines if line.strip()]
                for byte in keys:
                    key = chr(byte).lower()
                    if key in MODE_LABELS or key == 'q':
                        label = self.selection.choose(key)
                        print(f'\n[演示选择] {label}；当前轮完成后生效。', flush=True)
            except (OSError, ValueError, TypeError):
                self.selection.choose('q')
                return

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._saved is not None and self._fd is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)
            except (termios.error, OSError, ValueError):
                pass


def run_demo_round(mode: str, actions: dict[str, callable]) -> None:
    """Complete a whole pick-and-return transaction before the mode can change."""
    if mode not in MODE_LABELS:
        raise ValueError(f'未知演示模式：{mode}')
    if mode in {'1', '2'}:
        actions['head']()
        if mode == '2':
            actions['require_hole']()
        actions['left_pick']()
        if mode == '2':
            actions['to_hole']()
        actions['left_return']()
    else:
        actions['right_pick']()
        actions['right_vlm']()
        actions['right_lift']()
        actions['right_return']()


def check_left_return_arrival(*, saved_position, actual_position, descent: dict) -> None:
    """A premature force stop keeps the hinge closed instead of dropping it."""
    error_m = float(np.linalg.norm(
        np.asarray(actual_position, dtype=float) - np.asarray(saved_position, dtype=float)
    ))
    if not np.isfinite(error_m) or error_m > 0.010:
        raise SafetyError(
            f"铰链放回提前接触或未到原抓取点：误差 {error_m * 1000:.1f} mm，"
            f"下探停止原因 {descent.get('stop_reason')}；保持夹持并停止"
        )


def _move_left_to_saved_pose(ctx) -> None:
    """Return the held hinge from clearance height to its measured pickup pose."""
    try:
        from .left_hole_joint_move import move_left_hole_joint_target
        from .left_seat_motion import force_guarded_left_descent
    except ImportError:
        from left_hole_joint_move import move_left_hole_joint_target
        from left_seat_motion import force_guarded_left_descent
    saved = ctx.services.get('left_demo_grasp_pose')
    if saved is None:
        raise SafetyError('本轮没有保存左手抓取位姿；拒绝放回并松爪')
    controller = ctx.services.controller
    current = controller.read_end_effector_pose('arm_l_end_link')
    upper = CartesianPoseTarget(
        (saved.position_m[0], saved.position_m[1], saved.position_m[2] + 0.10),
        saved.orientation_xyzw,
    )
    travel = float(np.linalg.norm(np.asarray(current.position_m) - np.asarray(upper.position_m)))
    if travel > 0.60:
        raise SafetyError(f'铰链返回上方点距离 {travel * 1000:.1f} mm 超过 600 mm；保持夹持并停止')
    move_left_hole_joint_target(
        controller, upper, velocity_rad_s=ctx.args.auto_preset_velocity_rad_s,
    )
    measured = controller.read_end_effector_pose('arm_l_end_link')
    if float(np.linalg.norm(np.asarray(measured.position_m) - np.asarray(upper.position_m))) > 0.005:
        raise SafetyError('左手未到放回点上方；保持夹持并停止')
    contact_config = json.loads(Path(ctx.args.left_config).expanduser().read_text(encoding='utf-8'))['contact']
    descent = force_guarded_left_descent(
        controller, ctx.services.robot, max_distance_mm=100.0,
        force_threshold_n=float(contact_config['force_delta_threshold_n']),
        speed_mm_s=min(ctx.args.auto_speed_mm_s, 10.0),
        completion_tolerance_mm=5.0, timeout_s=30.0,
    )
    ctx.runtime.emit_event(descent)
    actual = controller.read_end_effector_pose('arm_l_end_link')
    check_left_return_arrival(
        saved_position=saved.position_m, actual_position=actual.position_m, descent=descent,
    )
    monitor = ctx.services.get('vlm_gripper_monitor')
    if monitor is not None:
        monitor.finish_before_release()
    ctx.runtime.open_left_gripper()
    try:
        from .left_seat_motion import move_left_vertical
    except ImportError:
        from left_seat_motion import move_left_vertical
    ctx.runtime.emit_event(move_left_vertical(
        controller, distance_mm=100.0, speed_mm_s=ctx.args.auto_speed_mm_s,
    ))


def _run_right_pick(ctx) -> None:
    try:
        from .right_apriltag_grasp_gdk import run_right_grasp_stage
    except ImportError:
        from right_apriltag_grasp_gdk import run_right_grasp_stage
    try:
        from .run_bilateral_flow_gdk import concise_child_stdout
    except ImportError:
        from run_bilateral_flow_gdk import concise_child_stdout
    # The stage owns the right camera while AprilTag tracking is active.
    feed = ctx.feeds.get('right')
    if feed is not None:
        feed.pause()
    try:
        with concise_child_stdout(event_sink=ctx.runtime.emit_event):
            code = run_right_grasp_stage(services=ctx.services, argv=ctx.settings.right_argv())
    finally:
        if feed is not None:
            feed.resume()
    if code != 0 or ctx.services.get('right_saved_grasp_pose') is None:
        raise SafetyError(f'电批抓取未完成（返回码 {code}）；保持现状并停止演示')


def _check_right_vlm(ctx) -> dict:
    try:
        from .run_bilateral_flow_gdk import _one_shot_right_fixture_check
    except ImportError:
        from run_bilateral_flow_gdk import _one_shot_right_fixture_check
    return _one_shot_right_fixture_check(ctx, phase='demo_right_grasp')


def _lift_right(ctx) -> None:
    try:
        from .right_hole_motion import lift_after_grasp
    except ImportError:
        from right_hole_motion import lift_after_grasp
    ctx.runtime.emit_event(lift_after_grasp(
        ctx.services.controller, ctx.services.get('right_saved_grasp_pose'),
        speed_mps=min(ctx.args.auto_speed_mm_s, 30.0) / 1000.0,
    ))


def _return_right(ctx) -> None:
    try:
        from .right_apriltag_grasp_gdk import return_screwdriver
    except ImportError:
        from right_apriltag_grasp_gdk import return_screwdriver
    saved = ctx.services.get('right_saved_grasp_pose')
    config = ctx.services.get('right_config')
    if saved is None or config is None:
        raise SafetyError('缺少电批抓取位姿或配置；拒绝放回')
    return_screwdriver(ctx.services.controller, saved, config, ctx.runtime.emit_event, release=True)
    try:
        from .right_hole_motion import lift_right_before_return
    except ImportError:
        from right_hole_motion import lift_right_before_return
    ctx.runtime.emit_event(lift_right_before_return(
        ctx.services.controller, speed_mm_s=ctx.args.auto_speed_mm_s, distance_mm=100.0,
    ))
    ctx.services.controller.set_position_control_phase(False)
    ctx.services.controller.prepare_joint_impedance_control(('right',))


def run_repeated_demo(ctx, selection: DemoModeSelection) -> int:
    """Reuse the full-flow GDK session, models, cameras, motion gate and stages."""
    try:
        from .run_bilateral_flow_gdk import (
            stage_home, stage_head_localize, stage_left_grasp, stage_left_to_hole_offset,
        )
    except ImportError:
        from run_bilateral_flow_gdk import (
            stage_home, stage_head_localize, stage_left_grasp, stage_left_to_hole_offset,
        )
    print('演示选项：1.抓取铰链  2.抓取铰链并移动  3.抓取电批；按 q 完成本轮后结束。', flush=True)
    mode = selection.wait_first()
    if mode is None:
        return 0
    previous = None
    round_number = 0
    while mode is not None:
        try:
            if mode != previous:
                ctx.runtime.emit('demo_home_start', mode=mode)
                stage_home(ctx)
            round_number += 1
            ctx.runtime.emit('demo_round_start', mode=mode, label=MODE_LABELS[mode], round=round_number)
            ctx.services.put('left_demo_grasp_pose', None)
            ctx.services.put('right_saved_grasp_pose', None)
            ctx.services.put('right_saved_workpiece_pose', None)
            actions = {
                'head': lambda: _head_localize(ctx, stage_head_localize),
                'require_hole': lambda: _require_hole(ctx),
                'left_pick': lambda: stage_left_grasp(ctx),
                'to_hole': lambda: _require_hole_move(ctx, stage_left_to_hole_offset),
                'left_return': lambda: _move_left_to_saved_pose(ctx),
                'right_pick': lambda: _run_right_pick(ctx),
                'right_vlm': lambda: _check_right_vlm(ctx),
                'right_lift': lambda: _lift_right(ctx),
                'right_return': lambda: _return_right(ctx),
            }
            run_demo_round(mode, actions)
            ctx.runtime.emit('demo_round_complete', mode=mode, round=round_number)
        except Exception as exc:
            ctx.runtime.emit('demo_round_failed', mode=mode, round=round_number, error=str(exc))
            print(f'[演示停止] {exc}', file=sys.stderr, flush=True)
            return 1
        previous = mode
        mode = selection.next_round()
        if mode != previous and mode is not None:
            print(f'[演示切换] 下一轮：{MODE_LABELS[mode]}', flush=True)
    # A completed cycle has already returned the part and raised the jaw.
    try:
        stage_home(ctx)
    except Exception as exc:
        ctx.runtime.emit('demo_finish_failed', error=str(exc))
        return 1
    return 0


def _require_hole_move(ctx, stage):
    result = stage(ctx)
    if result.get('skipped'):
        raise SafetyError('本轮未定位到 hole；无法演示铰链移动，保持夹持并停止')
    return result


def _head_localize(ctx, stage):
    feed = ctx.feeds.get('head')
    if feed is not None:
        feed.pause()
    try:
        return stage(ctx)
    finally:
        if feed is not None:
            feed.resume()


def _require_hole(ctx):
    if (ctx.services.get('head_targets') or {}).get('hole') is None:
        raise SafetyError('头部未定位到 hole；本轮不抓取铰链')
