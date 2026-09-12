# Dianpi SpaceMouse Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `dianpi/dp/Control.py` 的 RM1010 电批周期接入双臂 SpaceMouse 流程，用双键组合异步触发，并保留 P10 下降沿触发。

**Architecture:** 把电批 GPIO 时序封装为可注入 IO 的 `DianpiController`，独立保留 `Control.py` 的 P10 主循环；新增懒加载、单周期互斥的 `DianpiCycleRunner`，由 SpaceMouse worker 非阻塞触发。手势状态机在确认双键组合后屏蔽全部单键语义，避免夹爪、切手和阶段推进冲突。

**Tech Stack:** Python 3.12, `ctypes` RM1010 public API, `threading`, pytest, 现有 `SpaceMouseTakeoverWorker`/GDK 控制器。

**Spec:** `docs/superpowers/specs/2026-09-12-dianpi-spacemouse-integration-design.md`

## Global Constraints

- 不修改 RM1010 引脚编号、模式、上下拉配置，也不增加硬件连线。
- P10 保持上拉输入、低电平下降沿触发；P0/P1/P2/P8/P9 输出初始 HIGH，P11 输出初始 LOW。
- 电批周期不能调用或暂停 G2 运动，也不能阻塞 SpaceMouse 50 Hz 读写。
- 双键组合必须在两键按下时间差不超过 150 ms、持续至少 300 ms、六轴回中时触发一次；两键释放前不得再次触发。
- 双键组合从两键重叠开始屏蔽单击、双击、左键长按和右键长按；组合期间手臂只保持当前实测位姿。
- 周期运行中再次组合只返回 `DIANPI_BUSY`，不得并发执行第二个周期。
- 集成失败只记录 `dianpi_error`，不得让 G2 视觉阶段退出；独立运行 `Control.py` 的 P10 兼容行为保持不变。
- 只提交本功能相关文件，保留工作区已有的其他修改。

---

### Task 1: 提取可复用的 RM1010 控制器

**Files:**
- Modify: `dianpi/dp/Control.py`
- Create: `dianpi/dp/test_control.py`

**Interfaces:**
- `DianpiController(*, scan_fn=Device_Scan, open_fn=Device_Open, close_fn=Device_Close, init_fn=IO_InitMultiPin, read_fn=IO_ReadMultiPin, write_fn=IO_WriteMultiPin, sleep_fn=time.sleep, monotonic_fn=time.monotonic, output_fn=print)`；所有参数均可替换为测试替身。
- `DianpiController.open() -> None` 扫描第一台设备、初始化既有引脚并写入安全初始电平；失败抛出 `DianpiError` 子类并尽量关闭已打开设备。
- `DianpiController.read_inputs() -> dict[int, int] | None` 返回 P3/P4/P10 状态。
- `DianpiController.run_cycle() -> None` 在锁内执行 P9 低 50 ms、等待 2 s、P8 低 10 s 且第 2 s 将 P0 低、到第 10 s 同时恢复 P0/P8 高；周期中持续读取并输出 P3/P4，第二次调用抛出 `DianpiBusyError`。
- `DianpiController.close() -> None` 幂等地恢复 P0/P1/P2/P8/P9 高、P11 低并关闭设备。
- `main()` 仅负责 P10 下降沿轮询和生命周期调用 `open/read_inputs/run_cycle/close`，不再直接编排 GPIO 时序。

- [ ] **Step 1: Write the failing pure-logic tests**

  在 `dianpi/dp/test_control.py` 建立 FakeClock/FakeIO：记录每次写 pin、初始化配置、读状态和 sleep 时间；覆盖以下断言：

  ```python
  def test_open_initializes_existing_pins_and_safe_levels():
      controller, io = make_controller()
      controller.open()
      assert io.initialized_pins == EXPECTED_INIT_ORDER
      assert io.writes[-6:] == [(0, 1), (1, 1), (2, 1), (8, 1), (9, 1), (11, 0)]

  def test_run_cycle_preserves_pulse_and_delayed_forward_timing():
      controller, io = opened_controller()
      controller.run_cycle()
      assert io.low_high_interval(9) == pytest.approx(0.05)
      assert io.write_time(8, 0) == pytest.approx(2.05)
      assert io.write_time(0, 0) == pytest.approx(4.05)
      assert io.write_time(0, 1) == pytest.approx(12.05)
      assert io.write_time(8, 1) == pytest.approx(12.05)

  def test_second_cycle_is_rejected_while_first_cycle_is_running():
      controller, io = opened_controller(block_cycle=True)
      with pytest.raises(DianpiBusyError):
          controller.run_cycle()

  def test_close_restores_safe_levels_and_is_idempotent():
      controller, io = opened_controller()
      controller.close()
      controller.close()
      assert io.cleanup_writes == [(0, 1), (1, 1), (2, 1), (8, 1), (9, 1), (11, 0)]
  ```

- [ ] **Step 2: Run the focused tests and verify they fail**

  Run: `pytest -q dianpi/dp/test_control.py`

  Expected: FAIL because `DianpiController`, `DianpiBusyError`, and the injectable lifecycle do not yet exist.

- [ ] **Step 3: Implement the controller and preserve standalone P10 behavior**

  Move the existing scan/open/init/read/write/timing code into the stated class methods. Use `threading.Lock` plus an `_cycle_running` flag around `run_cycle`; use the injected monotonic clock and sleep function for elapsed-time checks so production keeps the same 50 ms polling while tests run without wall-clock delays. Ensure init/read/write return codes become descriptive `DianpiError` messages. Rewrite the module entry point to retain the existing P10 falling-edge loop and Ctrl+C cleanup through the class.

- [ ] **Step 4: Run the focused tests and compile check**

  Run: `pytest -q dianpi/dp/test_control.py && python -m py_compile dianpi/dp/Control.py`

  Expected: all controller tests PASS and compilation exits 0.

- [ ] **Step 5: Commit the controller extraction**

  ```bash
  git add dianpi/dp/Control.py dianpi/dp/test_control.py
  git commit -m "refactor: expose dianpi cycle controller"
  ```

### Task 2: Add conflict-safe dual-button gesture recognition

**Files:**
- Modify: `g2_adapter/spacemouse_takeover.py`
- Test: `g2_adapter/test_spacemouse_takeover.py`

**Interfaces:**
- Extend `TakeoverEvent` with `START_DIANPI = "start_dianpi"`.
- Add `DIANPI_CHORD_WINDOW_S = 0.15` and `DIANPI_CHORD_HOLD_S = 0.30`.
- `SpaceMouseGestureState.update(frame, now)` returns one `START_DIANPI` at most per two-button hold, with `consumed_buttons == (0, 1)` for every frame while the chord is blocked.

- [ ] **Step 1: Write failing gesture tests**

  Add tests for a valid chord, delayed second button, too-short hold, non-neutral axes, release/re-arm, and conflict suppression:

  ```python
  def test_dual_button_chord_emits_once_after_neutral_hold():
      state = SpaceMouseGestureState()
      assert state.update(frame(0.00, buttons=(True, False)), 0.00).events == ()
      assert state.update(frame(0.10, buttons=(True, True)), 0.10).events == ()
      result = state.update(frame(0.41, buttons=(True, True)), 0.41)
      assert result.events == (TakeoverEvent.START_DIANPI,)
      assert state.update(frame(0.50, buttons=(True, True)), 0.50).events == ()

  def test_chord_suppresses_takeover_gripper_and_advance_events():
      state = SpaceMouseGestureState()
      state.update(frame(0.00, buttons=(True, True)), 0.00)
      result = state.update(frame(2.00, buttons=(True, True)), 2.00)
      assert TakeoverEvent.START_DIANPI in result.events
      assert TakeoverEvent.ADVANCE_RIGHT not in result.events
      state.update(frame(2.10, buttons=(False, False)), 2.10)
      assert state.update(frame(2.20, pressed=(1,)), 2.20).events == ()

  def test_chord_requires_window_hold_and_neutral_axes():
      state = SpaceMouseGestureState()
      state.update(frame(0.00, buttons=(True, False)), 0.00)
      state.update(frame(0.20, buttons=(True, True)), 0.20)
      assert state.update(frame(0.60, buttons=(True, True)), 0.60).events == ()
      state.update(frame(1.00, buttons=(False, False)), 1.00)
      state.update(frame(1.10, buttons=(True, True), axes=(0.5, 0, 0, 0, 0, 0)), 1.10)
      assert state.update(frame(1.50, buttons=(True, True)), 1.50).events == (TakeoverEvent.START_DIANPI,)
  ```

- [ ] **Step 2: Run the new tests and verify they fail**

  Run: `pytest -q g2_adapter/test_spacemouse_takeover.py -k dianpi`

  Expected: FAIL because the event and chord state do not yet exist.

- [ ] **Step 3: Implement the chord state machine**

  Track the time each button went down, a chord candidate/blocked flag, and whether the start event was emitted. On first overlap, clear pending single-click/long-press state and consume both buttons. Emit only after the press-time difference is at most 150 ms, the hold duration is at least 300 ms, and all axes are within `deadzone`; keep consuming until both buttons are released. Do not pass chord frames to any individual gesture path.

- [ ] **Step 4: Run gesture and existing SpaceMouse tests**

  Run: `pytest -q g2_adapter/test_spacemouse_takeover.py g2_adapter/test_spacemouse_teleop.py`

  Expected: new chord tests and all existing takeover/teleop tests PASS.

- [ ] **Step 5: Commit the gesture state machine**

  ```bash
  git add g2_adapter/spacemouse_takeover.py g2_adapter/test_spacemouse_takeover.py
  git commit -m "feat: detect dianpi dual-button chord"
  ```

### Task 3: Build the asynchronous Dianpi cycle runner

**Files:**
- Create: `g2_adapter/dianpi_runner.py`
- Create: `g2_adapter/test_dianpi_runner.py`

**Interfaces:**
- `DianpiTriggerResult` enum values: `STARTED`, `BUSY`, `ERROR`.
- `DianpiCycleRunner(*, controller_factory=None, event_sink=print, thread_factory=threading.Thread)`; default factory imports `dianpi.dp.Control.DianpiController` lazily.
- `trigger() -> DianpiTriggerResult` never raises hardware exceptions; it starts a daemon worker once, returns `BUSY` while a cycle is active, and returns `ERROR` only when a prior/open/run failure prevents starting.
- `close(timeout_s=1.0) -> None` rejects new triggers, joins the worker, calls controller `close()` once, and is idempotent.
- `running -> bool` and `last_error -> BaseException | None` expose state for logs/tests.

- [ ] **Step 1: Write failing runner tests**

  Use a fake controller with events to prove lazy construction, one active cycle, busy handling, and swallowed errors:

  ```python
  def test_trigger_lazily_opens_and_runs_one_cycle():
      runner, controller = make_runner()
      assert runner.trigger() is DianpiTriggerResult.STARTED
      controller.cycle_finished.wait(1)
      assert controller.open_calls == 1 and controller.cycle_calls == 1

  def test_trigger_while_running_returns_busy_without_second_cycle():
      runner, controller = make_blocking_runner()
      assert runner.trigger() is DianpiTriggerResult.STARTED
      assert runner.trigger() is DianpiTriggerResult.BUSY
      assert controller.cycle_calls == 1

  def test_hardware_error_is_reported_without_raising_to_worker():
      events = []
      runner = make_failing_runner(event_sink=events.append)
      assert runner.trigger() is DianpiTriggerResult.STARTED
      wait_until(lambda: runner.last_error is not None)
      assert events[-1]["event"] == "dianpi_error"
      runner.close()
  ```

- [ ] **Step 2: Run runner tests and verify they fail**

  Run: `pytest -q g2_adapter/test_dianpi_runner.py`

  Expected: FAIL because the runner module and result enum do not yet exist.

- [ ] **Step 3: Implement lazy threaded lifecycle**

  Guard state with a lock; set the active flag before starting the daemon thread so a second chord cannot race the first. The thread constructs/imports the controller on its first run, calls `open()` once, calls `run_cycle()`, emits `{"event": "dianpi_started"}`/`dianpi_finished`/`dianpi_busy`/`dianpi_error` records, and stores exceptions instead of re-raising. Keep a successfully opened controller for later cycles and close it only from `close()`.

- [ ] **Step 4: Run runner tests and compile check**

  Run: `pytest -q g2_adapter/test_dianpi_runner.py && python -m py_compile g2_adapter/dianpi_runner.py`

  Expected: all runner tests PASS and compilation exits 0.

- [ ] **Step 5: Commit the runner**

  ```bash
  git add g2_adapter/dianpi_runner.py g2_adapter/test_dianpi_runner.py
  git commit -m "feat: run dianpi cycles asynchronously"
  ```

### Task 4: Connect the runner to both G2 SpaceMouse workers

**Files:**
- Modify: `g2_adapter/spacemouse_takeover.py`
- Modify: `g2_adapter/left_grasp_gdk.py`
- Modify: `g2_adapter/right_apriltag_grasp_gdk.py`
- Test: `g2_adapter/test_spacemouse_takeover.py`

**Interfaces:**
- Add optional `dianpi_runner=None` to `SpaceMouseTakeoverWorker.__init__`.
- On `START_DIANPI`, call `runner.trigger()` and return without executing takeover, gripper, side-switch, or advance handling; log the returned result. If a manual session is active, send one measured-pose hold through that session and leave no stale axis target.
- The left and right GDK entry points each construct one runner and pass it to their worker; cleanup order is `takeover_worker.stop()` then `dianpi_runner.close()`.

- [ ] **Step 1: Write failing worker integration tests**

  Add a fake runner to assert a chord starts it exactly once, suppresses manual actions, and reports busy without affecting worker failure state:

  ```python
  def test_worker_routes_start_dianpi_without_g2_gesture_action():
      runner = FakeDianpiRunner()
      worker = make_worker(dianpi_runner=runner)
      worker.step(frame(0.00, buttons=(True, True)), 0.00)
      worker.step(frame(0.31, buttons=(True, True)), 0.31)
      assert runner.calls == 1
      assert worker.active_side is None

  def test_busy_dianpi_trigger_does_not_raise_worker_failure():
      runner = FakeDianpiRunner(result=DianpiTriggerResult.BUSY)
      worker = make_worker(dianpi_runner=runner)
      emit_chord(worker)
      worker.raise_if_failed()
  ```

- [ ] **Step 2: Run integration tests and verify they fail**

  Run: `pytest -q g2_adapter/test_spacemouse_takeover.py -k dianpi`

  Expected: FAIL because the worker does not accept or route a Dianpi runner.

- [ ] **Step 3: Implement worker routing and lifecycle wiring**

  Handle `START_DIANPI` before `ADVANCE_RIGHT`; sanitize the frame with both buttons consumed. When a manual session exists, tick it with the sanitized neutral frame under `manual_motion()` so the command is based on current feedback; never call the normal click/side-switch path for the chord. Instantiate the runner at the existing worker construction points in the left/right scripts and close it in every normal/error/final-motion cleanup path.

- [ ] **Step 4: Run focused and regression tests**

  Run: `pytest -q g2_adapter/test_spacemouse_takeover.py g2_adapter/test_spacemouse_teleop.py g2_adapter/test_bilateral_workflow.py && python -m py_compile g2_adapter/spacemouse_takeover.py g2_adapter/left_grasp_gdk.py g2_adapter/right_apriltag_grasp_gdk.py`

  Expected: all tests PASS and compilation exits 0.

- [ ] **Step 5: Commit worker integration**

  ```bash
  git add g2_adapter/spacemouse_takeover.py g2_adapter/left_grasp_gdk.py g2_adapter/right_apriltag_grasp_gdk.py g2_adapter/test_spacemouse_takeover.py
  git commit -m "feat: integrate dianpi trigger with g2 spacemouse"
  ```

### Task 5: Document the gesture and verify the final surface

**Files:**
- Modify: `常用命令.md`
- Modify: `g2_adapter/SPACEMOUSE.md`
- Test: existing focused suites and Python compilation

**Interfaces:**
- Documentation must state the exact dual-button chord: press both within 150 ms, hold 300 ms with axes centered, release both before retry; chord suppresses other button actions and reports busy when a cycle is already running.
- Documentation must state that P10 remains available for standalone `dianpi/dp/Control.py`, and the integrated process must not run concurrently with another RM1010 owner.

- [ ] **Step 1: Add concise user-facing instructions**

  Update the existing SpaceMouse command notes and add one short section showing that the same bilateral workflow command starts the integration; do not add new hardware or a separate process command.

- [ ] **Step 2: Run the complete relevant verification**

  Run: `pytest -q g2_adapter/test_spacemouse_takeover.py g2_adapter/test_spacemouse_teleop.py g2_adapter/test_bilateral_workflow.py dianpi/dp/test_control.py g2_adapter/test_dianpi_runner.py && python -m compileall -q dianpi/dp g2_adapter`

  Expected: all tests PASS and compileall exits 0.

- [ ] **Step 3: Inspect the diff for scope and safety**

  Run: `git diff --check && git diff --stat -- dianpi/dp/Control.py dianpi/dp/test_control.py g2_adapter/dianpi_runner.py g2_adapter/test_dianpi_runner.py g2_adapter/spacemouse_takeover.py g2_adapter/test_spacemouse_takeover.py g2_adapter/left_grasp_gdk.py g2_adapter/right_apriltag_grasp_gdk.py 常用命令.md g2_adapter/SPACEMOUSE.md`

  Expected: no whitespace errors; only the listed integration files are part of this feature.

- [ ] **Step 4: Commit documentation and final verification**

  ```bash
  git add 常用命令.md g2_adapter/SPACEMOUSE.md
  git commit -m "docs: describe dianpi spacemouse trigger"
  ```
