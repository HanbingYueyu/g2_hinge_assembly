#!/usr/bin/env python3
"""Single-process runtime for the staged bilateral workflow.

One GDK session, one ``G2Controller``, one ``MotionPauseGate`` and one
SpaceMouse worker live here; stage functions receive this object instead of
creating their own GDK.  The module imports no GDK/vision library at import
time, so every helper can be unit tested with fakes.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from .right_post_grasp_sequence import load_right_pose, move_right_preset
    from .safety import SafetyError
except ImportError:
    from right_post_grasp_sequence import load_right_pose, move_right_preset
    from safety import SafetyError


OPEN_GRIPPER_RAD = -0.785
GRIPPER_OPEN_TOLERANCE_RAD = 0.035


def console_event_summary(record: dict) -> str | None:
    """Return one short operator-facing line; full records stay in JSONL."""
    event = record.get("event")
    if event == "stage_start":
        return f"[阶段开始] {record.get('title', record.get('stage'))} ({record.get('stage')})"
    if event == "stage_complete":
        return f"[阶段完成] {record.get('stage')}"
    if event == "stage_failed":
        return None  # The sequencer writes the same concise failure to stderr.
    if event == "left_preset_arrived":
        return f"[到位] {record.get('preset')}"
    if event == "right_preset_arrived":
        return f"[到位] {record.get('preset')}"
    if event == "left_seat_descent_complete":
        reason = {"force_threshold": "力阈值", "max_distance": "最大距离"}.get(
            record.get("stop_reason"), record.get("stop_reason"),
        )
        return (
            f"[下探完成] 原因={reason} 距离={record.get('descent_mm')} mm "
            f"力变化={record.get('delta_force_n')} N"
        )
    if event == "left_release_lift_complete":
        return f"[上抬完成] {record.get('distance_mm')} mm"
    if event == "stage_waiting_for_advance":
        return f"[等待确认] {record.get('stage')}：SpaceMouse 右按钮长按2秒或键盘 Enter"
    if event == "stage_advance_received":
        return f"[确认收到] {record.get('stage')}"
    if event == "bilateral_flow_stopped_at_stage":
        return f"[调试结束] 已完成 {record.get('stage')}"
    if event == "bilateral_flow_complete":
        return "[流程完成] 双臂流程全部完成"
    if event == "bilateral_models_preload_start":
        return "[初始化] 正在加载视觉模型…"
    if event == "bilateral_models_preloaded":
        return "[初始化] 视觉模型加载完成"
    if event in {
        "vlm_gripper_result", "vlm_pin_result", "vlm_c_hole_result",
        "vlm_convex_inserted_result", "vlm_four_screws_result",
        "vlm_right_fixture_grasp_result",
    }:
        result = record.get("result") or {}
        label = {
            "vlm_gripper_result": "夹持",
            "vlm_pin_result": "凸轴",
            "vlm_c_hole_result": "C孔",
            "vlm_convex_inserted_result": "凸轴严格",
            "vlm_four_screws_result": "四颗螺丝",
            "vlm_right_fixture_grasp_result": "右手夹具",
        }[event]
        value = result.get("state", result.get("obvious_c_leak"))
        if event == "vlm_convex_inserted_result":
            value = result.get("decision")
        elif event == "vlm_four_screws_result":
            value = result.get("latest_decision")
        elif event == "vlm_right_fixture_grasp_result":
            value = result.get("decision", result.get("latest_decision"))
        stable = f" stable={record['stable_state']}" if 'stable_state' in record else ''
        review = f" round={record['review_round']}/3" if 'review_round' in record else ''
        timing = ''
        if event == 'vlm_gripper_result':
            timing = f" image_time={record.get('input_status', {}).get('image_time', 'unknown')} elapsed={record.get('elapsed_s', 'unknown')}s"
        reason = result.get("reason", result.get("evidence"))
        return f"[VLM {label}] result={value}{stable}{review} confidence={result.get('confidence')} reason={reason}{timing}"
    if event == "vlm_check_warning":
        return f"[VLM 警告] task={record.get('task')} error={record.get('error')}"
    if record.get("level") == "warning":
        return f"[警告] {record.get('reason', event)}"
    return None


def append_jsonl_record(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")

# Presets owned by the staged flow, with the arm each one must be recorded for.
STAGED_PRESETS: dict[str, str] = {
    "LR1": "right",
    "RR1": "right",
    "LL1": "left",
    "LL2": "left",
}


@dataclass
class PreloadedModels:
    """Vision assets loaded once and kept resident for the whole workflow."""

    head_yolo: Any = None
    left_yolo: Any = None
    apriltag_detector: Any = None
    loaded: dict[str, Any] = field(default_factory=dict)


class BilateralRuntime:
    """Own the single robot writer and the shared left/right stage helpers."""

    def __init__(
        self,
        *,
        args,
        controller,
        gate=None,
        worker=None,
        models: PreloadedModels | None = None,
        journal_path: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.args = args
        self.controller = controller
        self.gate = gate
        self.worker = worker
        self.models = models or PreloadedModels()
        self.journal_path = Path(journal_path) if journal_path is not None else None
        self.clock = clock
        self.sleep = sleep
        self.events: list[dict] = []

    # ------------------------------------------------------------------ log
    @property
    def poses_path(self) -> Path:
        return Path(self.args.poses)

    def emit_event(self, event: dict) -> dict:
        record = {"utc": datetime.now(timezone.utc).isoformat(), **event}
        self.events.append(record)
        if self.journal_path is not None:
            append_jsonl_record(self.journal_path, record)
        summary = console_event_summary(record)
        if summary is not None:
            print(summary, flush=True)
        return record

    def emit(self, event: str, **fields: Any) -> dict:
        return self.emit_event({"event": event, **fields})

    # --------------------------------------------------------------- assets
    def load_left_preset(self, name: str) -> list[float]:
        try:
            from .run_left_premove_sequence import load_presets
        except ImportError:
            from run_left_premove_sequence import load_presets
        [(_, values)] = load_presets(self.poses_path, (str(name),))
        return list(values)

    def load_preset(self, name: str) -> tuple[str, list[float]]:
        """Load one staged preset and confirm it was recorded for its own arm."""
        expected = STAGED_PRESETS.get(str(name))
        if expected is None:
            raise ValueError(f"{name} 不是分阶段流程使用的预设名")
        if expected == "left":
            values = self.load_left_preset(name)
        else:
            values = list(load_right_pose(self.poses_path, str(name)))
        return expected, values

    def validate_plan_assets(self) -> dict:
        """Check every staged preset exists for the correct arm (read-only)."""
        report: dict[str, Any] = {}
        missing: list[str] = []
        for name, side in STAGED_PRESETS.items():
            if name == "LL1" and getattr(self.args, "teach_enable", False):
                continue
            try:
                self.load_preset(name)
                report[name] = {"side": side, "ok": True}
            except Exception as exc:  # noqa: BLE001 - reported to the operator
                report[name] = {"side": side, "ok": False, "error": str(exc)}
                missing.append(name)
        return {"presets": report, "missing": missing}

    # -------------------------------------------------------------- grippers
    def _open_grippers(
        self,
        sides: tuple[str, ...],
        *,
        open_rad: float,
        timeout_s: float,
        resend_period_s: float,
        event: str,
    ) -> dict:
        target = float(open_rad)
        if not math.isfinite(target) or not -math.pi / 4.0 <= target <= 0.0:
            raise ValueError("夹爪张开角度必须在 [-π/4, 0] rad")
        if not (math.isfinite(float(timeout_s)) and float(timeout_s) > 0.0):
            raise ValueError("夹爪反馈超时必须为正的有限值")
        if not (math.isfinite(float(resend_period_s)) and 0.0 < float(resend_period_s) <= 1.0):
            raise ValueError("夹爪重发周期必须在 (0, 1] 秒")
        deadline = self.clock() + float(timeout_s)
        attempts = {side: 0 for side in sides}
        next_send_at = 0.0
        actual: dict[str, float | None] = {}
        while True:
            now = self.clock()
            if now >= next_send_at:
                for side in sides:
                    self.controller.move_gripper(side, target)
                    attempts[side] += 1
                next_send_at = now + float(resend_period_s)
            actual = {side: self.controller.read_gripper_position(side) for side in sides}
            if all(
                value is not None and abs(float(value) - target) <= GRIPPER_OPEN_TOLERANCE_RAD
                for value in actual.values()
            ):
                return self.emit(
                    event, target_rad=target, actual_position_rad=actual,
                    command_attempts=attempts, sides=list(sides),
                )
            if now >= deadline:
                raise SafetyError(
                    f"夹爪张开反馈超时：sides={list(sides)}, actual={actual}，attempts={attempts}；"
                    "单发 move_end_effector_joint 不会被重复，已尝试周期重发；不继续后续自动动作"
                )
            self.sleep(0.05)

    def open_all_grippers(
        self,
        *,
        open_rad: float = OPEN_GRIPPER_RAD,
        timeout_s: float = 6.0,
        resend_period_s: float = 0.1,
    ) -> dict:
        """Open both Omnipickers and gate on real feedback before continuing.

        ``move_gripper`` only submits a non-blocking ``move_end_effector_joint``
        request, so one call is not enough: the proven ``set_omnipicker.py``
        tool re-sends the same target every ~100 ms until the measured jaw angle
        matches.  This does exactly that for both sides and never switches the
        GDK control mode.
        """
        return self._open_grippers(
            ("left", "right"), open_rad=open_rad, timeout_s=timeout_s,
            resend_period_s=resend_period_s, event="all_grippers_opened",
        )

    def open_left_gripper(
        self,
        *,
        open_rad: float = OPEN_GRIPPER_RAD,
        timeout_s: float = 6.0,
        resend_period_s: float = 0.1,
    ) -> dict:
        """Release the left jaw only, with the same repeated-request feedback."""
        return self._open_grippers(
            ("left",), open_rad=open_rad, timeout_s=timeout_s,
            resend_period_s=resend_period_s, event="left_gripper_opened",
        )

    # ------------------------------------------------------------------ home
    def move_both_to_home(
        self, *, velocity_rad_s: float = 0.08, arrival_tolerance_rad: float = 0.03, timeout_s: float = 25.0,
    ) -> dict:
        try:
            from .workflow_home import WORKFLOW_HOME_POSE_NAME, load_workflow_home, move_workflow_home
        except ImportError:
            from workflow_home import WORKFLOW_HOME_POSE_NAME, load_workflow_home, move_workflow_home
        pose_name = getattr(self.args, "home_pose_name", WORKFLOW_HOME_POSE_NAME)
        target = load_workflow_home(self.poses_path, pose_name)
        record = move_workflow_home(
            self.controller,
            target,
            velocity_rad_s=velocity_rad_s,
            arrival_tolerance_rad=arrival_tolerance_rad,
            timeout_s=timeout_s,
        )
        record["preset"] = pose_name
        return self.emit_event(record)

    # --------------------------------------------------------------- presets
    def move_left_presets(
        self,
        names: Sequence[str],
        *,
        velocity_rad_s: float = 0.08,
        arrival_tolerance_rad: float = 0.03,
        timeout_s: float = 25.0,
    ) -> list[dict]:
        records: list[dict] = []
        for name in names:
            target = self.load_left_preset(name)
            self.controller.move_preset(
                "left", target,
                velocity_rad_s=velocity_rad_s, life_time_s=timeout_s,
                detail=f"staged bilateral {name}",
            )
            deadline = self.clock() + float(timeout_s)
            while True:
                measured = list(self.controller.checked_arm_state())[:7]
                error = max(abs(actual - expected) for actual, expected in zip(measured, target))
                if error <= float(arrival_tolerance_rad):
                    records.append(self.emit(
                        "left_preset_arrived", preset=str(name),
                        max_joint_error_rad=float(error),
                    ))
                    break
                if self.clock() >= deadline:
                    raise SafetyError(f"左手 {name} 在 {timeout_s:.1f} s 内未到位；停止后续动作")
                self.sleep(0.1)
        return records

    def move_right_preset(
        self,
        name: str,
        *,
        velocity_rad_s: float = 0.08,
        arrival_tolerance_rad: float = 0.03,
        timeout_s: float = 25.0,
    ) -> dict:
        target = list(load_right_pose(self.poses_path, str(name)))
        record = move_right_preset(
            self.controller, target,
            preset_name=str(name),
            velocity_rad_s=velocity_rad_s,
            arrival_tolerance_rad=arrival_tolerance_rad,
            timeout_s=timeout_s,
        )
        return self.emit_event(record)

    # ----------------------------------------------------------------- gates
    def wait_for_advance(self, spec, *, poll_s: float = 0.01) -> None:
        """Block at one SpaceMouse gate, consuming exactly one advance request."""
        if self.worker is None:
            raise SafetyError("等待下一阶段需要已启动的 SpaceMouse worker")
        stage = getattr(spec, "stage", spec)
        stage_name = getattr(stage, "value", str(stage))
        self.emit(
            "stage_waiting_for_advance", stage=stage_name,
            source="spacemouse_right_long_press_2s_or_enter", takeover_available=True,
        )
        previous = self.worker.allow_advance_right
        self.worker.allow_advance_right = True
        try:
            while True:
                guard = getattr(self.controller, 'check_automatic_guard', None)
                if guard is not None:
                    guard()
                self.worker.raise_if_failed()
                if self.worker.advance_requested:
                    self.worker.consume_advance()
                    self.emit("stage_advance_received", stage=stage_name)
                    return
                self.sleep(poll_s)
        finally:
            self.worker.allow_advance_right = previous

    def release_gate(self) -> None:
        """Drop every gate reference before a final, non-preemptible move."""
        if self.worker is not None:
            self.worker.stop_for_final_motion()
        self.controller.motion_gate = None
