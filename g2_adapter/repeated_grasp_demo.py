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
from copy import copy

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
    '4': '抓取电批并移动',
    '5': '全流程单次（不启动电批）',
    '6': '全流程单次（启动电批）',
}


class DemoCancelled(SafetyError):
    """Esc cancelled this round; the existing GDK session remains reusable."""


class DemoModeSelection:
    """The input thread only queues a request; robot code commits it between rounds."""

    def __init__(self):
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self.active = None
        self.pending = None
        self._cancel = threading.Event()

    def choose(self, key: str) -> str:
        if key in {'esc', '\x1b'}:
            with self._lock:
                if self.active in MODE_LABELS:
                    self._cancel.set()
            return '停止当前流程并返回选择界面'
        if key not in (*MODE_LABELS, '0', 'q'):
            raise ValueError('请选择 0、1、2、3、4、5、6 或 q')
        with self._lock:
            self.pending = key
            self._ready.set()
        if key == '0':
            return '切换左右夹爪张开/闭合'
        return ('完成当前轮后返回选择界面' if self.active in MODE_LABELS else '退出演示') if key == 'q' else MODE_LABELS[key]

    def check_cancelled(self):
        if self._cancel.is_set():
            raise DemoCancelled('Esc 取消当前演示')

    def return_to_menu(self):
        with self._lock:
            self.active = self.pending = None
            self._ready.clear()
            self._cancel = threading.Event()

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
            if self.active in {'5', '6'} and self.pending is None:
                self.active = None
            if self.pending is not None:
                self.active = None if self.pending == 'q' else self.pending
                self.pending = None
                self._ready.clear()
            return self.active


class DemoKeyboard:
    """Read 1/2/3/4/5/6/q without sending robot motion from the input thread."""

    def __init__(self, selection: DemoModeSelection, stream=None, *, language_switch=None):
        self.selection = selection
        self.language_switch = language_switch
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
                    if key == '9':
                        if self.language_switch is None:
                            print('\n[语音] 语音播报未启用。', flush=True)
                        else:
                            language = self.language_switch()
                            print('\n[语音] 已切换为' + ('英文' if language == 'en' else '中文'), flush=True)
                        continue
                    if key in MODE_LABELS or key in {'0', 'q', '\x1b'}:
                        label = self.selection.choose(key)
                        print(f'\n[演示选择] {label}' + ('。' if key in {'q', '\x1b'} else '；当前轮完成后生效。'), flush=True)
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
    if mode in {'5', '6'}:
        actions['full_flow'](mode)
    elif mode in {'1', '2'}:
        actions['head']()
        if mode == '2':
            actions['require_hole']()
        actions['left_pick']()
        if mode == '2':
            actions['to_hole']()
        actions['left_return']()
    else:
        if mode == '4':
            actions['head']()
            actions['require_hole']()
        actions['right_pick']()
        actions['right_vlm']()
        if mode == '4':
            actions['right_lift']()
            actions['right_to_hole']()
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


try:
    from .return_timing import timed_return_step
except ImportError:
    from return_timing import timed_return_step


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
    return_position = (saved.position_m[0] + 0.008, saved.position_m[1], saved.position_m[2])
    controller = ctx.services.controller
    current = controller.read_end_effector_pose('arm_l_end_link')
    upper = CartesianPoseTarget(
        (return_position[0], return_position[1], return_position[2] + 0.10),
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
        speed_mm_s=ctx.args.auto_speed_mm_s,
        completion_tolerance_mm=5.0, timeout_s=30.0, diagnose_return=True,
    )
    with timed_return_step('放回事件处理'):
        ctx.runtime.emit_event(descent)
    with timed_return_step('放回位置核对'):
        actual = controller.read_end_effector_pose('arm_l_end_link')
        check_left_return_arrival(
            saved_position=return_position, actual_position=actual.position_m, descent=descent,
        )
    monitor = ctx.services.get('vlm_gripper_monitor')
    if monitor is not None:
        with timed_return_step('结束夹持监测'):
            monitor.finish_before_release()
    with timed_return_step('夹爪张开指令及反馈'):
        ctx.runtime.open_left_gripper(hold_pose_stream=True)
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


def _move_right_to_hole(ctx) -> None:
    try:
        from .right_hole_motion import move_right_tool_above_hole
    except ImportError:
        from right_hole_motion import move_right_tool_above_hole
    saved = ctx.services.get('right_saved_grasp_pose')
    workpiece = ctx.services.get('right_saved_workpiece_pose')
    hole = (ctx.services.get('head_targets') or {}).get('hole')
    if saved is None or workpiece is None or hole is None:
        raise SafetyError('缺少电批抓取位姿、工件位姿或 hole 位置；保持夹持并停止演示')
    transport = move_right_tool_above_hole(
        ctx.services.controller, saved, workpiece, hole['position_base_link_m'],
        x_mm=ctx.args.right_hole_x_mm, y_mm=ctx.args.right_hole_y_mm,
        z_mm=ctx.args.right_hole_z_mm,
        yaw_deg=ctx.args.right_hole_yaw_deg, pitch_deg=ctx.args.right_hole_pitch_deg,
        roll_deg=ctx.args.right_hole_roll_deg,
        speed_mm_s=ctx.args.auto_speed_mm_s, lift_before_move=False,
        joint_velocity_rad_s=ctx.args.auto_preset_velocity_rad_s / 2.0,
    )
    ctx.runtime.emit_event(transport)


def _release_right_in_place(ctx):
    try:
        from .right_apriltag_grasp_gdk import move_right_gripper_with_feedback
    except ImportError:
        from right_apriltag_grasp_gdk import move_right_gripper_with_feedback
    config = ctx.services.get('right_config')
    if config is None or ctx.services.get('right_saved_grasp_pose') is None:
        raise SafetyError('缺少电批抓取位姿或配置；拒绝松爪')
    opened = move_right_gripper_with_feedback(
        ctx.services.controller, float(config['gripper'].get('open_rad', -0.785)),
    )
    ctx.runtime.emit_event({"event": "right_return_released", **opened})
    ctx.services.controller.set_position_control_phase(False)
    ctx.services.controller.prepare_joint_impedance_control(('right',))


def _return_right(ctx) -> None:
    try:
        from .right_apriltag_grasp_gdk import return_screwdriver
    except ImportError:
        from right_apriltag_grasp_gdk import return_screwdriver
    saved = ctx.services.get('right_saved_grasp_pose')
    config = ctx.services.get('right_config')
    if saved is None or config is None:
        raise SafetyError('缺少电批抓取位姿或配置；拒绝放回')
    return_screwdriver(
        ctx.services.controller, saved, config, ctx.runtime.emit_event, release=True,
        joint_velocity_rad_s=ctx.args.auto_preset_velocity_rad_s / 2.0,
    )
    try:
        from .right_hole_motion import lift_right_before_return
    except ImportError:
        from right_hole_motion import lift_right_before_return
    ctx.runtime.emit_event(lift_right_before_return(
        ctx.services.controller, speed_mm_s=ctx.args.auto_speed_mm_s, distance_mm=100.0,
    ))
    ctx.services.controller.set_position_control_phase(False)
    ctx.services.controller.prepare_joint_impedance_control(('right',))


class AutomaticDemoRuntime:
    """Skip advance gates while retaining guards and point-arrival waits."""

    def __init__(self, runtime, mode, runner):
        self._runtime = runtime
        self.automatic_demo_mode = mode
        self.runner = runner
        self._point_generation = None

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def wait_for_advance(self, spec, **_kwargs):
        self.controller.check_automatic_guard()
        self.emit("demo_automatic_advance", stage=str(getattr(spec, "stage", spec)))

    def on_right_teach_point(self, index, result):
        try:
            from .dianpi_runner import DianpiTriggerResult
        except ImportError:
            from dianpi_runner import DianpiTriggerResult
        gate = getattr(self._runtime, 'gate', None)
        self._point_generation = getattr(gate, 'pause_generation', None)
        delay_s = 1.0 if self.automatic_demo_mode == '6' else 3.0
        self.emit('demo_screw_point_hold_start', point=index, mode=self.automatic_demo_mode, dwell_s=delay_s)
        deadline = self.clock()+delay_s
        while self.clock() < deadline:
            self._check_guard()
            self.sleep(min(.02, max(0.,deadline-self.clock())))
        if self.automatic_demo_mode == '6':
            self._check_guard()
            if gate is not None and isinstance(self._point_generation, int):
                with gate.automatic_teach_motion(self._point_generation):
                    outcome = self.runner.trigger(automatic=True)
            else:
                outcome = self.runner.trigger(automatic=True)
            if outcome != DianpiTriggerResult.STARTED:
                raise SafetyError(f"螺丝点位 {index} 电批未启动：{outcome}")
            timeout = self.clock()+120.0
            while True:
                self._check_guard()
                running = self.runner.running
                error = self.runner.last_error
                if error is not None:
                    raise SafetyError(f"电批运行失败：{error}")
                if not running:
                    break
                if self.clock() >= timeout:
                    raise SafetyError("等待电批运行结束超过 120 秒；停止自动演示")
                self.sleep(.02)
            self._check_guard()
            self._runtime.lift_after_screw(distance_mm=20.0)
            self._check_guard()
        self.emit('demo_screw_point_hold_complete', point=index, mode=self.automatic_demo_mode)

    def _check_guard(self):
        gate = getattr(self._runtime, 'gate', None)
        if gate is not None and (
            gate.is_paused() is True
            or (self._point_generation is not None and gate.pause_generation != self._point_generation)
        ):
            raise SafetyError('螺丝点位等待期间发生人工接管；停止自动演示')
        self.controller.check_automatic_guard()
        if self.worker is not None:
            self.worker.raise_if_failed()


def _run_full_flow_demo(ctx, mode):
    try:
        from .run_bilateral_flow_gdk import _drive
    except ImportError:
        from run_bilateral_flow_gdk import _drive
    args = copy(ctx.args)
    if not args.teach_enable or len(args.right_teach_trajectories or []) != 4:
        raise SafetyError('模式 5/6 需要启用示教并指定四段右手轨迹')
    args.repeat_grasp_demo = False
    try:
        from .left_teach_placement import validate_teach_args
        from .right_teach_placement import validate_right_teach_args
    except ImportError:
        from left_teach_placement import validate_teach_args
        from right_teach_placement import validate_right_teach_args
    validate_teach_args(args)
    validate_right_teach_args(args)
    args.from_stage = 'home'
    args.through_stage = 'finish'
    runner = ctx.services.dianpi_runner
    policy = runner.set_trigger_policy(enabled=mode == '6', automatic_only=True)
    previous_runtime, previous_args = ctx.runtime, ctx.settings.args
    ctx.runtime = AutomaticDemoRuntime(previous_runtime, mode, runner)
    ctx.settings.args = args
    try:
        if _drive(ctx, args) != 0:
            raise SafetyError('自动全流程未完成；停止演示并保持当前状态')
    finally:
        ctx.runtime = previous_runtime
        ctx.settings.args = previous_args
        runner.set_trigger_policy(enabled=policy[0], automatic_only=policy[1])


def toggle_demo_grippers(ctx):
    controller = ctx.services.controller
    actual = [controller.read_gripper_position(side) for side in ('left', 'right')]
    if any(value is None or not np.isfinite(value) for value in actual):
        raise SafetyError('无法读取左右夹爪状态；不执行切换')
    target = 0. if all(abs(value + np.pi / 4.) <= .035 for value in actual) else -np.pi / 4.
    result = ctx.runtime._open_grippers(
        ('left', 'right'), open_rad=target, timeout_s=6., resend_period_s=.1,
        event='demo_grippers_toggled',
    )
    print('[夹爪] 左右夹爪已' + ('闭合' if target == 0. else '张开'), flush=True)
    return result


def print_demo_menu():
    print(
        '\n演示选项：\n'
        '0.左右夹爪同时张开/闭合（运行中按下，本轮完成后执行）\n'
        '1.抓取铰链\n'
        '2.抓取铰链并移动\n'
        '3.抓取电批\n'
        '4.抓取电批并移动\n'
        '5.全流程不启动电批（单次）\n'
        '6.全流程启动电批（单次）\n'
        '9.切换中文/英文语音\n'
        'Esc.停止当前流程并回到选择界面\n'
        'q.完成本轮后回到选择界面（选择界面按 q 结束）',
        flush=True,
    )


def _recover_hinge_grasp(ctx, reason):
    monitor = ctx.services.get('vlm_gripper_monitor')
    if monitor is None or not (ctx.services.get('head_targets') or {}).get('hinge'):
        raise SafetyError('VLM 重抓缺少本次头部铰链定位或夹持监测')
    monitor.set_phase('idle')
    ctx.runtime.emit('vlm_regrasp_requested', reason=str(reason))
    controller = ctx.services.controller
    controller.prepare_joint_impedance_control(('left',))
    current = list(controller.checked_arm_state()[:7])
    for _ in range(20):
        controller.hold_joint_positions('left', current)
        time.sleep(.01)
    try:
        from .left_seat_motion import move_left_vertical
    except ImportError:
        from left_seat_motion import move_left_vertical
    ctx.runtime.emit_event(move_left_vertical(
        controller, distance_mm=50., speed_mm_s=ctx.args.auto_speed_mm_s))
    print('[VLM 重抓] 已停止当前动作并上抬 50 mm，重新定位抓取铰链。', flush=True)


def _stop_cancelled_round(ctx):
    monitor = ctx.services.get('vlm_gripper_monitor')
    if monitor is not None:
        monitor.set_phase('idle')
    job = ctx.services.get('insertion_review_job')
    if job is not None:
        job.stop.set()
        ctx.services.put('insertion_review_job', None)
    ctx.services.dianpi_runner.stop_current_cycle()
    controller = ctx.services.controller
    # All owning stage streams have unwound. Replace remaining planned goals
    # with measured joints, and retain both gripper states.
    guard = controller.demo_cancel_guard
    controller.demo_cancel_guard = None
    try:
        with controller.manual_motion():
            controller.prepare_joint_impedance_control(('left', 'right'))
            current = tuple(controller.checked_arm_state())
            controller.hold_joint_positions('left', current[:7])
            controller.hold_joint_positions('right', current[7:])
    finally:
        controller.demo_cancel_guard = guard


class _CancelableRuntime:
    def __init__(self, runtime, selection):
        self._runtime, self.selection = runtime, selection

    def __getattr__(self, name):
        return getattr(self._runtime, name)

    def emit(self, *args, **kwargs):
        self.selection.check_cancelled()
        return self._runtime.emit(*args, **kwargs)

    def emit_event(self, record):
        self.selection.check_cancelled()
        return self._runtime.emit_event(record)


class _CancelableVlmClient:
    """Unwind the motion owner without waiting for a slow read-only HTTP call."""
    def __init__(self, client, selection):
        self.client, self.selection = client, selection
        self.lock = threading.Lock()

    def __getattr__(self, name):
        return getattr(self.client, name)

    def run_bytes(self, *args, **kwargs):
        done = threading.Event()
        result, errors = [], []
        cancel = self.selection._cancel
        def request():
            try:
                with self.lock:
                    if cancel.is_set():
                        return
                    result.append(self.client.run_bytes(*args, **kwargs))
            except Exception as exc:
                errors.append(exc)
            finally:
                done.set()
        threading.Thread(target=request, name='demo-vlm-request', daemon=True).start()
        while not done.wait(.02):
            self.selection.check_cancelled()
        self.selection.check_cancelled()
        if errors:
            raise errors[0]
        if not result:
            raise DemoCancelled('VLM 请求已取消')
        return result[0]


def run_repeated_demo(ctx, selection: DemoModeSelection) -> int:
    controller = ctx.services.controller
    gate = controller.motion_gate
    old_guard = getattr(controller, 'demo_cancel_guard', None)
    old_gate_guard = getattr(gate, 'cancel_guard', None) if gate is not None else None
    previous_runtime = ctx.runtime
    previous_client = ctx.services.get('vlm_client')
    runner = getattr(ctx.services, "dianpi_runner", None)
    previous_gpio_policy = runner.set_gpio_trigger_override(True) if runner is not None else None
    ctx.runtime = _CancelableRuntime(previous_runtime, selection)
    if previous_client is not None:
        ctx.services.put('vlm_client', _CancelableVlmClient(previous_client, selection))
    controller.demo_cancel_guard = selection.check_cancelled
    if gate is not None:
        gate.cancel_guard = selection.check_cancelled
    try:
        return _run_repeated_demo(ctx, selection)
    finally:
        if runner is not None:
            runner.set_gpio_trigger_override(previous_gpio_policy)
        ctx.runtime = previous_runtime
        ctx.services.put('vlm_client', previous_client)
        controller.demo_cancel_guard = old_guard
        if gate is not None:
            gate.cancel_guard = old_gate_guard


def _run_repeated_demo(ctx, selection: DemoModeSelection) -> int:
    """Reuse the full-flow GDK session, models, cameras, motion gate and stages."""
    try:
        from .run_bilateral_flow_gdk import (
            stage_home, stage_head_localize, stage_left_grasp, stage_left_to_hole_offset,
        )
    except ImportError:
        from run_bilateral_flow_gdk import (
            stage_home, stage_head_localize, stage_left_grasp, stage_left_to_hole_offset,
        )
    print_demo_menu()
    mode = selection.wait_first()
    if mode is None:
        return 0
    previous = None
    round_number = 0
    while mode is not None:
        if mode == '0':
            try:
                toggle_demo_grippers(ctx)
            except Exception as exc:
                print(f'[夹爪切换失败] {exc}', file=sys.stderr, flush=True)
            # Complete this menu action without discarding a newly queued key.
            with selection._lock:
                selection.active = None
            previous = None
            print_demo_menu()
            mode = selection.wait_first()
            continue
        try:
            round_number += 1
            ctx.runtime.emit('demo_round_start', mode=mode, label=MODE_LABELS[mode], round=round_number)
            if mode != previous and mode not in {'5', '6'}:
                ctx.runtime.emit('demo_home_start', mode=mode)
                stage_home(ctx)
            ctx.services.put('left_demo_grasp_pose', None)
            ctx.services.put('right_saved_grasp_pose', None)
            ctx.services.put('right_saved_workpiece_pose', None)
            actions = {
                'full_flow': lambda selected_mode: _run_full_flow_demo(ctx, selected_mode),
                'head': lambda: _head_localize(ctx, stage_head_localize),
                'require_hole': lambda: _require_hole(ctx),
                'left_pick': lambda: _pick_left_with_confirmation(ctx, stage_left_grasp),
                'to_hole': lambda: _require_hole_move(ctx, stage_left_to_hole_offset),
                'left_return': lambda: _move_left_to_saved_pose(ctx),
                'right_pick': lambda: _run_right_pick(ctx),
                'right_vlm': lambda: _check_right_vlm(ctx),
                'right_lift': lambda: _lift_right(ctx),
                'right_to_hole': lambda: _move_right_to_hole(ctx),
                'right_return': lambda: _release_right_in_place(ctx) if mode == '3' else _return_right(ctx),
            }
            run_demo_round(mode, actions)
            if mode in {'1', '2'}:
                ctx.runtime.emit('demo_home_start', mode=mode)
                stage_home(ctx)
            selection.check_cancelled()
            ctx.runtime.emit('demo_round_complete', mode=mode, round=round_number)
        except Exception as exc:
            if selection._cancel.is_set():
                try:
                    _stop_cancelled_round(ctx)
                except Exception as stop_error:
                    ctx.runtime._runtime.emit('demo_stop_failed', error=str(stop_error))
                    print(f'[停止失败] {stop_error}', file=sys.stderr, flush=True)
                    return 1
                selection.return_to_menu()
                ctx.runtime.emit('demo_round_cancelled', mode=mode, round=round_number)
                print('[演示已停止] 保持当前位置和夹爪状态，请重新选择。', flush=True)
                previous = None
                print_demo_menu()
                mode = selection.wait_first()
                if mode is None:
                    return 0
                continue
            try:
                from .vlm_gripper_monitor import VlmRegraspRequested
            except ImportError:
                from vlm_gripper_monitor import VlmRegraspRequested
            if mode in {'1', '2'} and isinstance(exc, VlmRegraspRequested):
                try:
                    _recover_hinge_grasp(ctx, exc)
                except Exception as recovery_error:
                    if selection._cancel.is_set():
                        _stop_cancelled_round(ctx)
                        selection.return_to_menu()
                        ctx.runtime.emit('demo_round_cancelled', mode=mode, round=round_number)
                        print_demo_menu()
                        mode = selection.wait_first()
                        previous = None
                        continue
                    ctx.runtime.emit('demo_round_failed', mode=mode, error=str(recovery_error))
                    print(f'[重抓恢复失败] {recovery_error}', file=sys.stderr, flush=True)
                    return 1
                previous = mode
                continue
            ctx.runtime.emit('demo_round_failed', mode=mode, round=round_number, error=str(exc))
            print(f'[演示停止] {exc}', file=sys.stderr, flush=True)
            return 1
        previous = mode
        mode = selection.next_round()
        if mode is None and selection.active is None:
            if previous not in {'1', '2', '5', '6'}:
                selection.active = previous
                try:
                    stage_home(ctx)
                    selection.check_cancelled()
                except Exception as exc:
                    if not selection._cancel.is_set():
                        ctx.runtime.emit('demo_finish_failed', error=str(exc))
                        return 1
                    try:
                        _stop_cancelled_round(ctx)
                    except Exception as stop_error:
                        ctx.runtime._runtime.emit('demo_stop_failed', error=str(stop_error))
                        return 1
                    selection.return_to_menu()
                    ctx.runtime.emit('demo_round_cancelled', mode=previous, round=round_number)
                    print('[演示已停止] 保持当前位置和夹爪状态，请重新选择。', flush=True)
                selection.active = None
            previous = None
            print_demo_menu()
            mode = selection.wait_first()
            if mode is None:
                return 0
        else:
            print_demo_menu()
        if mode != previous and mode in MODE_LABELS:
            print(f'[演示切换] 下一轮：{MODE_LABELS[mode]}', flush=True)
    # A completed cycle has already returned the part and raised the jaw.
    try:
        stage_home(ctx)
    except Exception as exc:
        ctx.runtime.emit('demo_finish_failed', error=str(exc))
        return 1
    return 0


def _pick_left_with_confirmation(ctx, stage):
    try:
        from .run_bilateral_flow_gdk import _confirm_left_grasp
    except ImportError:
        from run_bilateral_flow_gdk import _confirm_left_grasp
    result = stage(ctx)
    # Short demo rounds must not release the hinge and invalidate the VLM
    # epoch before its post-grasp vote/success announcement has completed.
    _confirm_left_grasp(ctx)
    return result


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
        raise SafetyError('头部未定位到 hole；本轮不开始抓取')
