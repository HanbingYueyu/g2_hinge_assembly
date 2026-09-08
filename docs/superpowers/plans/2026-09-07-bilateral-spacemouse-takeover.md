# Bilateral SpaceMouse Takeover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 串联现有左手和右手视觉 demo，并支持 SpaceMouse 左右手临时接管、3 秒无操作恢复，以及左阶段右按钮长按 2 秒推进到右手。

**Architecture:** 在现有 GDK 单写入边界增加可选 `MotionPauseGate`。每个 demo 内由唯一 SpaceMouse worker 读取 HID；接管时 gate 暂停自动运动，worker 通过同一 `G2Controller` 的 bypass 上下文发送手动末端目标，3 秒无操作后解除 gate。统一入口顺序调用左/右 demo，左阶段收到推进事件时返回专用状态码后启动右阶段。

**Tech Stack:** Python 3.10, `unittest`, NumPy, existing GDK bindings, Linux hidraw.

**Spec:** `docs/superpowers/specs/2026-09-07-bilateral-spacemouse-takeover-design.md`

## Global Constraints

- 默认不发送运动；只有 `--allow-motion` 才允许 GDK 命令。
- 任一时刻只能有一个 GDK 运动写入者。
- SpaceMouse 断连、输入过期、GDK 错误或接管恢复失败时停止后续自动动作。
- 不改变现有视觉检测、抓取标定、力控阈值和姿态点位。
- 真机动作不纳入自动测试；测试使用 FakeGdk/FakeRobot。

---

### Task 1: Add pure gesture and pause state machines

**Files:**
- Create: `g2_adapter/spacemouse_takeover.py`
- Test: `g2_adapter/test_spacemouse_takeover.py`

**Interfaces:**
- `TakeoverEvent` enum values: `NONE`, `TAKEOVER_LEFT`, `TAKEOVER_RIGHT`, `ADVANCE_RIGHT`, `TIMEOUT`.
- `SpaceMouseGestureState.update(frame: InputFrame, now: float) -> tuple[TakeoverEvent, ...]`.
- `MotionPauseGate.pause()`, `.resume()`, `.abort(error)`, `.wait_for_auto()`, `.is_paused()`, and `manual_motion()` context manager.

- [ ] **Step 1: Write failing tests** for left/right double-click, right 2-second hold, click-window expiry, long-hold gripper suppression, and gate blocking/resume/abort.
- [ ] **Step 2: Run `PYTHONPATH=g2_adapter ./scripts/run_gdk_python.sh -m unittest test_spacemouse_takeover` and confirm the new tests fail because the module is absent.
- [ ] **Step 3: Implement the pure gesture state machine.** Use the existing 0.55 s double-click window, 0.40 s short-click limit, 2.0 s right long-press threshold, and report-batched `InputFrame.pressed` edges. Emit a consumed-button set so double/long gestures cannot toggle the gripper.
- [ ] **Step 4: Implement `MotionPauseGate`.** Automatic callers block while paused; the `manual_motion()` context bypasses the wait for the single manual writer; `abort()` wakes waiters by raising the stored `SafetyError`.
- [ ] **Step 5: Run the focused tests and confirm they pass.**

### Task 2: Make GDK motion calls obey the pause gate

**Files:**
- Modify: `g2_adapter/control.py` around `G2Controller.__init__`, `_require_motion`, and motion command boundaries.
- Test: `g2_adapter/test_control.py`

**Interfaces:**
- `G2Controller(..., motion_gate: MotionPauseGate | None = None)`.
- `G2Controller.manual_motion()` returns the gate bypass context when configured.

- [ ] **Step 1: Add a failing test** proving an automatic `_send_right_cartesian_pose` blocks while the gate is paused, while the same call inside `manual_motion()` proceeds.
- [ ] **Step 2: Run the focused control test and confirm the failure.**
- [ ] **Step 3: Add the optional gate without changing default callers.** `_require_motion()` waits on the gate for automatic callers; all existing direct/stream/preset/gripper sends continue using this common boundary.
- [ ] **Step 4: Add abort propagation and verify no command is sent after an abort.**
- [ ] **Step 5: Run `python3 -m unittest discover -s g2_adapter -p 'test_control.py'`.**

### Task 3: Add the in-process SpaceMouse takeover worker

**Files:**
- Modify: `g2_adapter/spacemouse_takeover.py`
- Modify: `g2_adapter/spacemouse_teleop.py` to support fixed-side external gestures and the requested released/held 3D/6D mapping.
- Test: `g2_adapter/test_spacemouse_takeover.py`
- Test: `g2_adapter/test_spacemouse_teleop.py`

**Interfaces:**
- `SpaceMouseTakeoverWorker(controller, gate, device, config, *, on_advance_right, clock=time.monotonic)`.
- `.start()`, `.stop()`, `.raise_if_failed()`, `.advance_requested`.

- [ ] **Step 1: Add failing worker tests** with a fake input source: a double-click activates the requested side, a held left button allows all six axes, released left suppresses rotation, and 3 seconds without nonzero axes/buttons resumes automatic mode.
- [ ] **Step 2: Run the worker tests and confirm they fail.**
- [ ] **Step 3: Implement the worker.** It owns `CompactHID`, pauses the gate before activation, switches/validates `control_mode=1` in the manual bypass context, runs `TeleopSession` at 50 Hz, strips consumed gesture edges, and sends one measured hold before timeout/resume.
- [ ] **Step 4: On a right 2-second hold, send a measured hold, set `advance_requested`, keep the gate paused, and let the current demo return its dedicated advance status.
- [ ] **Step 5: Surface worker failures through `MotionPauseGate.abort()` and verify no automatic command resumes.
- [ ] **Step 6: Run all SpaceMouse unit tests.**

### Task 4: Wire optional takeover into the left and right demos

**Files:**
- Modify: `g2_adapter/left_grasp_gdk.py` argument parsing, controller setup, main loop entry, and cleanup.
- Modify: `g2_adapter/right_apriltag_grasp_gdk.py` argument parsing, controller setup, main loop entry, and cleanup.
- Test: `g2_adapter/test_left_grasp_gdk.py` if present, otherwise add `g2_adapter/test_bilateral_workflow.py` fakes for the wiring boundary.

**Interfaces:**
- New optional flags on both demos: `--spacemouse-device`, `--spacemouse-speed-mm-s`, `--spacemouse-rotation-deg-s`, and `--spacemouse-deadzone`.
- Left demo returns `EXIT_ADVANCE_RIGHT = 20` when the right long-press event is received.

- [ ] **Step 1: Add failing CLI tests** for accepting the takeover options, leaving the default path unchanged, and returning the advance status from the left phase.
- [ ] **Step 2: Run those tests and confirm failure.**
- [ ] **Step 3: Construct one `MotionPauseGate` and one worker after each demo creates its controller; pass the gate to `G2Controller` before any motion call.
- [ ] **Step 4: At the top of each main loop, call `worker.raise_if_failed()` and, for the left demo, return `EXIT_ADVANCE_RIGHT` when requested. While paused, suspend visual state-machine ticks without creating new targets; camera/model workers may remain alive.
- [ ] **Step 5: Stop the worker before closing the camera/GDK and preserve existing cleanup ordering.
- [ ] **Step 6: Run the left/right demo tests plus existing control/safety tests.**

### Task 5: Add the bilateral sequential entry point

**Files:**
- Create: `g2_adapter/run_bilateral_workflow_gdk.py`
- Create: `g2_adapter/test_bilateral_workflow.py`
- Modify: `常用命令.md`
- Modify: `g2_adapter/SPACEMOUSE.md`

**Interfaces:**
- `run_bilateral_workflow_gdk.py` forwards configured left/right demo arguments and requires `--allow-motion` for an executing workflow.
- Stage return codes: `0` left completed normally, `20` left advanced by right long press, nonzero failure stops the sequence.

- [ ] **Step 1: Add failing tests** for left-normal→right, left-advance→right, right failure stopping, and `--allow-motion` validation.
- [ ] **Step 2: Run the tests and confirm failure.**
- [ ] **Step 3: Implement the runner.** Start the left demo with `--full-cycle` and SpaceMouse options; after return code `0` or `20`, start the right demo with `--execute --premove-pr1 --post-grasp-pr2` and the same SpaceMouse options. Do not create a second HID/GDK owner in the runner.
- [ ] **Step 4: Emit JSON stage events and preserve child stdout/stderr for the operator.
- [ ] **Step 5: Add the complete command and button semantics to both docs.
- [ ] **Step 6: Run the full relevant test matrix and `git diff --check`.**

### Task 6: Final verification

- [ ] **Step 1:** Run `./scripts/run_gdk_python.sh -m unittest discover -s g2_adapter -p 'test_spacemouse*.py'`.
- [ ] **Step 2:** Run `python3 -m unittest discover -s g2_adapter -p 'test_control.py'` and `python3 -m unittest discover -s g2_adapter -p 'test_safety.py'`.
- [ ] **Step 3:** Run `python3 -m compileall -q` on all changed Python files.
- [ ] **Step 4:** Run `git diff --check`.
- [ ] **Step 5:** Verify preview/CLI paths do not initialize GDK or send motion without `--allow-motion`; do not execute the real bilateral command.
