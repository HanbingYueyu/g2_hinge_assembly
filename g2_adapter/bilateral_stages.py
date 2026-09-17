#!/usr/bin/env python3
"""Pure stage plan for the single-process bilateral workflow.

The sequencer owns the ordered 9-step plan, which stages wait for a SpaceMouse
"next stage" gesture, and the per-stage result bookkeeping.  It deliberately has
no GDK, camera, SpaceMouse, or robot imports so the whole sequence can be unit
tested without hardware.

Step mapping (operator spec):

1. open both grippers, move both arms to ``workflow_home``
2. head YOLO + depth: leftmost hinge (class 0) and highest-confidence hole (class 1)
3. left arm above hinge (yaw 0 / pitch 42 / roll 180) -> visual grasp -> lift 150 mm
   -> above hole (yaw 20 / pitch 42 / roll 180)
4. right arm to ``LR1``
5. left arm to ``LL1`` -> force-guarded descent -> SpaceMouse gate 1 -> release
   -> lift 100 mm -> ``LL2``
6. right arm to ``RR1`` -> AprilTag grasp of the electric screwdriver
7. lift 100 mm -> fixed downward orientation above detected ``hole``
8. SpaceMouse gate 2 -> right arm returns to the saved grasp pose
9. open all grippers -> both arms to ``workflow_home``
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Stage(str, Enum):
    HOME = "home"
    HEAD_LOCALIZE = "head_localize"
    LEFT_GRASP = "left_grasp"
    VLM_GRIPPER_CHECK = "vlm_gripper_check"
    RIGHT_TO_LR1 = "right_to_lr1"
    GATE_LEFT = "gate_left"
    LEFT_TO_LL1_AND_SEAT = "left_to_ll1_and_seat"
    LEFT_RELEASE_LIFT_TO_LL2 = "left_release_lift_to_ll2"
    VLM_ASSEMBLY_CHECK = "vlm_assembly_check"
    RIGHT_GRASP = "right_grasp"
    GATE_RIGHT = "gate_right"
    RIGHT_RETURN = "right_return"
    FINISH = "finish"


@dataclass(frozen=True)
class StageSpec:
    stage: Stage
    title: str
    user_step: str
    waits_for_advance: bool = False
    camera_pane: str | None = None


DEFAULT_PLAN: tuple[StageSpec, ...] = (
    StageSpec(Stage.HOME, "张开左右夹爪并回到 workflow_home", "1"),
    StageSpec(Stage.HEAD_LOCALIZE, "头部定位最左 hinge 与最高置信度 hole", "2", camera_pane="head"),
    StageSpec(Stage.LEFT_GRASP, "左臂到铰链上方→视觉伺服抓取→上抬 150 mm→到 hole XYZ 偏移位置", "3", camera_pane="left"),
    StageSpec(Stage.VLM_GRIPPER_CHECK, "VLM 只读检查左夹爪是否仍夹持铰链", "3"),
    StageSpec(Stage.RIGHT_TO_LR1, "右臂移动到 LR1", "4", camera_pane="right"),
    # Motion-only handlers do not publish frames: keep the raw camera feed live.
    StageSpec(Stage.LEFT_TO_LL1_AND_SEAT, "左臂移动到 LL1 并力保护下探", "5"),
    StageSpec(Stage.GATE_LEFT, "等待 SpaceMouse 下一阶段指令（门1）", "5", waits_for_advance=True),
    StageSpec(Stage.LEFT_RELEASE_LIFT_TO_LL2, "张开左爪、上抬 100 mm 并移动到 LL2", "5"),
    StageSpec(Stage.VLM_ASSEMBLY_CHECK, "VLM 只读检查凸轴 B 与明显 C 孔外漏", "5"),
    StageSpec(
        Stage.RIGHT_GRASP,
        "右臂移动到 RR1、AprilTag 抓取电批、上抬 100 mm 并移动到 hole XYZ 偏移位置",
        "6-7", camera_pane="right",
    ),
    StageSpec(Stage.GATE_RIGHT, "等待 SpaceMouse 下一阶段指令（门2）", "8", waits_for_advance=True),
    StageSpec(Stage.RIGHT_RETURN, "右臂回到抓取成功时保存的位姿", "8", camera_pane="right"),
    StageSpec(Stage.FINISH, "张开全部夹爪并双臂回到 workflow_home", "9"),
)


def stage_names() -> tuple[str, ...]:
    """Ordered stage identifiers, for CLI choices and logging."""
    return tuple(spec.stage.value for spec in DEFAULT_PLAN)


class StageSequencer:
    """Walk a fixed stage plan once, recording each stage's outcome."""

    def __init__(self, plan: tuple[StageSpec, ...] = DEFAULT_PLAN, *, start_at: str | None = None) -> None:
        self.plan = tuple(plan)
        if not self.plan:
            raise ValueError("阶段计划不能为空")
        self._index = 0
        self.results: list[dict] = []
        self.failure: dict | None = None
        if start_at is not None:
            self.skip_to(start_at)

    @property
    def index(self) -> int:
        return self._index

    @property
    def is_done(self) -> bool:
        return self._index >= len(self.plan)

    @property
    def current(self) -> StageSpec | None:
        return None if self.is_done else self.plan[self._index]

    @property
    def remaining(self) -> tuple[StageSpec, ...]:
        return self.plan[self._index:]

    def skip_to(self, stage: str) -> None:
        """Start the plan at ``stage``, skipping everything before it."""
        target = Stage(stage)
        for position, spec in enumerate(self.plan):
            if spec.stage is target:
                self._index = position
                return
        raise ValueError(f"阶段计划里没有 {target.value}")

    def complete_current(self, payload: dict | None = None) -> StageSpec | None:
        """Mark the current stage complete and return the next stage, if any."""
        if self.is_done:
            return None
        spec = self.plan[self._index]
        self.results.append({"stage": spec.stage.value, "user_step": spec.user_step, "ok": True, **(payload or {})})
        self._index += 1
        return self.current

    def fail_current(self, error: str) -> None:
        """Record the current stage as failed; the sequence stops here."""
        if self.is_done:
            return
        spec = self.plan[self._index]
        self.failure = {"stage": spec.stage.value, "user_step": spec.user_step, "ok": False, "error": str(error)}
        self.results.append(self.failure)

    def summary(self) -> dict:
        return {
            "completed": [result for result in self.results if result.get("ok")],
            "failure": self.failure,
            "finished": self.is_done and self.failure is None,
        }
