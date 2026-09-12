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

# Presets owned by the staged flow, with the arm each one must be recorded for.
STAGED_PRESETS: dict[str, str] = {
    "LR1": "right",
    "RR1": "right",
    "RR2": "right",
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
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if self.journal_path is not None:
            self.journal_path.parent.mkdir(parents=True, exist_ok=True)
            with self.journal_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
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
            source="spacemouse_right_long_press_2s", takeover_available=True,
        )
        previous = self.worker.allow_advance_right
        self.worker.allow_advance_right = True
        try:
            while True:
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
