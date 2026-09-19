# Default Joint Impedance Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GDK `control_mode=3` joint impedance with the measured seven-axis stiffness the default for every formal robot-motion path, while preserving low-stiffness drag teaching and explicit diagnostic modes.

**Architecture:** Centralize mode-3 preparation and standard stiffness in `G2Controller`, then make each public motion boundary require that prepared state before it creates a planner request, servo stream, or end-effector stream. Full-flow and standalone motion entry points prepare the required arm groups explicitly; teaching temporarily overrides one group and always restores the standard stiffness without leaving mode 3.

**Tech Stack:** Python 3.12, AgiBot GDK 4.1.5 APIs, `unittest`, NumPy, existing G2 fake-robot test harnesses

**Spec:** `docs/superpowers/specs/2026-09-19-default-joint-impedance-control-design.md`

## Global Constraints

- “Joint force control” means GDK `control_mode=3` joint impedance, not raw torque or constant-force control.
- Standard left/right stiffness is exactly `kp = (400, 400, 400, 500, 300, 200, 200)`.
- Drag teaching stiffness is exactly `kp = (50, 50, 50, 60, 30, 30, 30)` for only the selected arm.
- Formal motion must never silently fall back to `control_mode=1` when mode-3 preparation fails.
- `joint_control_request()`, `move_arm_joint_servo()`, and `end_effector_pose_control()` remain the three motion interfaces.
- Read-only camera, annotation, training, and status paths must not change the robot mode.
- The explicit position-mode recovery tool remains mode 1; the Cartesian-impedance demo remains mode 2.
- Arm ordering remains left joints 0–6 followed by right joints 7–13; control groups remain 0=left, 1=right, 2=both.
- Production code changes follow test-first red/green cycles; no live robot movement is part of automated tests.

---

### Task 1: Central joint-impedance preparation API

**Files:**
- Modify: `g2_adapter/control.py`
- Modify: `g2_adapter/test_control.py`

**Interfaces:**
- Produces: `STANDARD_JOINT_IMPEDANCE_KP: tuple[float, ...]`
- Produces: `G2Controller.prepare_joint_impedance_control(sides: Sequence[str] = ("left", "right"), kp: Sequence[float] = STANDARD_JOINT_IMPEDANCE_KP) -> None`
- Produces: `G2Controller.set_joint_impedance(side: str, kp: Sequence[float]) -> None`
- Consumes: `robot.set_control_mode(3)`, `robot.set_joint_impedance(list[gdk.Impedance], group)` and live motion status

- [ ] **Step 1: Extend the fake GDK and robot, then write failing preparation tests**

Add `Impedance` support and recorded impedance calls to the existing fakes in `g2_adapter/test_control.py`. Add tests equivalent to:

```python
def test_prepare_joint_impedance_enters_mode_3_and_configures_both_arms(self):
    robot = FakeRobot()
    controller = G2Controller(FakeGdk, robot, allow_motion=True)

    controller.prepare_joint_impedance_control()

    self.assertEqual(robot.control_mode, 3)
    self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 1])
    self.assertEqual(
        [[item.kp for item in values] for values, _ in robot.joint_impedance_calls],
        [[400, 400, 400, 500, 300, 200, 200]] * 2,
    )

def test_prepare_joint_impedance_is_idempotent_while_mode_3_stays_healthy(self):
    robot = FakeRobot()
    controller = G2Controller(FakeGdk, robot, allow_motion=True)
    controller.prepare_joint_impedance_control()
    controller.prepare_joint_impedance_control()
    self.assertEqual(len(robot.joint_impedance_calls), 2)

def test_read_only_controller_construction_does_not_change_mode(self):
    robot = FakeRobot()
    G2Controller(FakeGdk, robot, allow_motion=False)
    self.assertEqual(robot.control_mode, 1)
    self.assertEqual(robot.joint_impedance_calls, [])
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run:

```bash
cd /home/flyfuture/g2_hinge_assembly
.venv-sam3/bin/python -m unittest \
  g2_adapter.test_control.ControlTests.test_prepare_joint_impedance_enters_mode_3_and_configures_both_arms \
  g2_adapter.test_control.ControlTests.test_prepare_joint_impedance_is_idempotent_while_mode_3_stays_healthy \
  g2_adapter.test_control.ControlTests.test_read_only_controller_construction_does_not_change_mode -v
```

Expected: FAIL because `prepare_joint_impedance_control` and the fake impedance recording do not exist.

- [ ] **Step 3: Implement validated, idempotent preparation**

In `g2_adapter/control.py`, define the constant near the other control constants and implement the two methods. The implementation must validate seven finite positive values, map `left/right` to groups `0/1`, enter mode 3 before setting impedance, cache the `(side, kp)` configuration, invalidate the cache whenever `set_control_mode()` changes away from mode 3, and finish by verifying `error_code == 0` and `control_mode == 3`.

Use this public shape:

```python
STANDARD_JOINT_IMPEDANCE_KP = (400.0, 400.0, 400.0, 500.0, 300.0, 200.0, 200.0)

def set_joint_impedance(self, side: str, kp: Sequence[float]) -> None:
    self._require_motion()
    # validate side and seven kp values, enter mode 3, set group 0 or 1,
    # verify live status, and update the configured-group cache

def prepare_joint_impedance_control(
    self,
    sides: Sequence[str] = ("left", "right"),
    kp: Sequence[float] = STANDARD_JOINT_IMPEDANCE_KP,
) -> None:
    # normalize unique sides in left/right order and configure only stale groups
```

Do not call either method from `G2Controller.__init__`.

- [ ] **Step 4: Run the complete controller test module**

Run:

```bash
.venv-sam3/bin/python -m unittest g2_adapter.test_control -v
```

Expected: the new tests PASS; pre-existing mode-1 expectation tests may still fail and are intentionally migrated in Task 2.

- [ ] **Step 5: Commit the central API**

```bash
git add g2_adapter/control.py g2_adapter/test_control.py
git commit -m "feat: add standard joint impedance preparation"
```

---

### Task 2: Run all formal controller motion interfaces in mode 3

**Files:**
- Modify: `g2_adapter/control.py`
- Modify: `g2_adapter/test_control.py`

**Interfaces:**
- Consumes: `G2Controller.prepare_joint_impedance_control()` from Task 1
- Produces: planner, arm-servo, and end-effector public entry points that require healthy mode 3
- Preserves: `enable_left_cartesian_impedance()` and `start_left_cartesian_impedance_stream()` as explicit mode-2 diagnostics

- [ ] **Step 1: Change controller expectations to mode 3 and add failing coverage for all three interfaces**

Update existing tests named around “position stream uses mode 1” and “setpoint requires mode 1” to assert mode 3. Add a test that starts from mode 1 and exercises one method from each interface family:

```python
def test_formal_motion_interfaces_prepare_joint_impedance(self):
    robot = FakeRobot()
    controller = G2Controller(FakeGdk, robot, allow_motion=True)
    controller.move_preset("left", [0.0] * 7)
    self.assertEqual(robot.control_mode, 3)

    controller.servo_step([0.0] * 14)
    stream = controller.start_left_end_effector_position_stream(
        CartesianPoseTarget((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
    )
    stream.stop()

    self.assertEqual(robot.control_mode, 3)
    self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 1])
```

Also update fake motion status defaults to mode 3 only in tests that represent prepared formal motion; keep explicit wrong-mode tests setting mode 1 or 2 themselves.

- [ ] **Step 2: Run the controller tests and verify failures mention mode 1 or missing preparation**

Run:

```bash
.venv-sam3/bin/python -m unittest g2_adapter.test_control -v
```

Expected: FAIL in end-effector mode checks and old mode-1 assertions.

- [ ] **Step 3: Prepare mode 3 at public motion boundaries**

In `g2_adapter/control.py`:

- Call `prepare_joint_impedance_control((side,))` before a single-arm `move_preset()`.
- Call `prepare_joint_impedance_control()` once before bilateral planner/servo loops.
- Call the appropriate single-arm preparation before starting left/right end-effector streams and before one-shot left setpoints.
- Change live status checks inside stream startup, send loops, and actual-arrival waits from `control_mode == 1` to `control_mode == 3`.
- Update docstrings and exceptions from “position mode 1” to “joint impedance mode 3”.
- Do not call preparation on every 50/100 Hz tick; prepare once before each stream or servo loop and retain periodic health checks inside the loop.
- Leave the explicit Cartesian-impedance path requiring mode 2. When it starts from mode 3, allow `enable_left_cartesian_impedance()` to switch explicitly to mode 2 instead of requiring mode 1 first.

- [ ] **Step 4: Run controller and Cartesian-impedance demo tests**

Run:

```bash
.venv-sam3/bin/python -m unittest \
  g2_adapter.test_control \
  g2_adapter.test_demo_left_impedance_move \
  g2_adapter.test_demo_joint_impedance_move -v
```

Expected: PASS, including the preserved explicit mode-2 demo behavior.

- [ ] **Step 5: Commit the formal interface migration**

```bash
git add g2_adapter/control.py g2_adapter/test_control.py g2_adapter/test_demo_left_impedance_move.py
git commit -m "feat: run formal motion interfaces in joint impedance"
```

---

### Task 3: Keep drag teaching low-stiffness and restore standard mode 3

**Files:**
- Modify: `g2_adapter/teach_trajectory_gdk.py`
- Modify: `g2_adapter/test_teach_trajectory_gdk.py`
- Modify: `g2_adapter/left_teach_placement.py`
- Modify: `g2_adapter/right_teach_placement.py`
- Modify: `g2_adapter/test_left_teach_placement.py`
- Modify: `g2_adapter/test_right_teach_placement.py`

**Interfaces:**
- Consumes: `STANDARD_JOINT_IMPEDANCE_KP` from Task 1
- Preserves: `LOW_TEACH_KP = (50, 50, 50, 60, 30, 30, 30)`
- Produces: record cleanup that restores standard `kp` on the selected arm and leaves `control_mode=3`
- Produces: replay that prepares standard stiffness before planner/servo commands

- [ ] **Step 1: Rewrite teaching tests to state the desired restoration behavior**

Update/add tests equivalent to:

```python
def test_record_restores_standard_impedance_without_returning_to_position_mode(self):
    robot = FakeRobot()
    session = G2TrajectorySession(FakeGdk(), robot, "left")
    session.record(duration_s=0.0, rate_hz=100.0)
    self.assertEqual(robot.control_modes, [3])
    self.assertEqual(
        [tuple(item.kp for item in values) for values, group in robot.impedance_calls if group == 0],
        [(50, 50, 50, 60, 30, 30, 30), (400, 400, 400, 500, 300, 200, 200)],
    )

def test_record_interruption_also_restores_standard_impedance(self):
    # make the sampling callback raise KeyboardInterrupt after the first sample
    # and assert the same final standard-kp call and no mode-1 transition
```

Update replay assertions so `control_modes == [3]` and single-arm servo group remains `0` for left or `1` for right.

- [ ] **Step 2: Run teaching tests and verify the old mode-1 cleanup fails**

Run:

```bash
.venv-sam3/bin/python -m unittest \
  g2_adapter.test_teach_trajectory_gdk \
  g2_adapter.test_left_teach_placement \
  g2_adapter.test_right_teach_placement -v
```

Expected: FAIL where current code calls `set_control_mode(1)` after record/replay.

- [ ] **Step 3: Implement low-kp override with guaranteed standard-kp restoration**

In `G2TrajectorySession.record()`:

```python
self.robot.set_control_mode(3)
self.robot.set_joint_impedance(low_values, group)
try:
    # hold current selected-arm angles and sample feedback
finally:
    self.robot.set_joint_impedance(standard_values, group)
```

Remove every record/replay cleanup call that switches to mode 1. Before replay’s start-point planner request, enter mode 3 and apply standard stiffness to the selected group. Update placement wrappers so SpaceMouse takeover or early replay termination leaves standard mode 3 rather than restoring mode 1.

- [ ] **Step 4: Run all teaching and takeover-end tests**

Run:

```bash
.venv-sam3/bin/python -m unittest \
  g2_adapter.test_teach_trajectory_gdk \
  g2_adapter.test_left_teach_placement \
  g2_adapter.test_right_teach_placement \
  g2_adapter.test_teach_takeover_end -v
```

Expected: PASS.

- [ ] **Step 5: Commit teaching behavior**

```bash
git add \
  g2_adapter/teach_trajectory_gdk.py \
  g2_adapter/left_teach_placement.py \
  g2_adapter/right_teach_placement.py \
  g2_adapter/test_teach_trajectory_gdk.py \
  g2_adapter/test_left_teach_placement.py \
  g2_adapter/test_right_teach_placement.py
git commit -m "feat: restore standard impedance after drag teaching"
```

---

### Task 4: Move SpaceMouse and takeover sessions to mode 3

**Files:**
- Modify: `g2_adapter/run_spacemouse_gdk.py`
- Modify: `g2_adapter/spacemouse_teleop.py`
- Modify: `g2_adapter/spacemouse_takeover.py`
- Modify: `g2_adapter/test_spacemouse_cli.py`
- Modify: `g2_adapter/test_spacemouse_teleop.py`
- Modify: `g2_adapter/test_spacemouse_takeover.py`

**Interfaces:**
- Consumes: `G2Controller.prepare_joint_impedance_control()`
- Produces: `ensure_joint_impedance_mode(controller: G2Controller) -> None` in `run_spacemouse_gdk.py`
- Preserves: 50 Hz teleoperation, motion mode 0/1 requirement, end-effector target integration, and takeover pause ownership

- [ ] **Step 1: Add failing CLI, teleop-health, and takeover tests**

Rename the CLI helper test and assert:

```python
def test_robot_start_switches_position_mode_to_joint_impedance(self):
    robot = test_control.FakeRobot()
    robot.control_mode = 1
    controller = G2Controller(test_control.FakeGdk, robot, allow_motion=True)
    ensure_joint_impedance_mode(controller)
    self.assertEqual(robot.control_mode, 3)
    self.assertEqual([group for _, group in robot.joint_impedance_calls], [0, 1])
```

Add one `TeleopSession._healthy()` test accepting mode 3 and rejecting mode 1. Update takeover activation tests to expect `prepare_joint_impedance_control((side,))` rather than `set_control_mode(1)`.

- [ ] **Step 2: Run SpaceMouse tests and verify they fail on existing mode-1 assumptions**

Run:

```bash
.venv-sam3/bin/python -m unittest \
  g2_adapter.test_spacemouse_cli \
  g2_adapter.test_spacemouse_teleop \
  g2_adapter.test_spacemouse_takeover -v
```

Expected: FAIL with mode-1 expectations.

- [ ] **Step 3: Implement mode-3 startup and takeover**

- Replace `ensure_position_mode()` with `ensure_joint_impedance_mode()` and call `controller.prepare_joint_impedance_control()` after validating motion mode 0/1.
- Change `TeleopSession._healthy()` to require `control_mode == 3`.
- In `SpaceMouseTakeoverWorker._activate()`, while holding `manual_motion()`, call `prepare_joint_impedance_control((side,))` before the first tick; never call `set_control_mode(1)`.
- Keep read-only preview and `--list-devices` paths free of GDK initialization and mode writes.

- [ ] **Step 4: Run the SpaceMouse suite**

Run:

```bash
.venv-sam3/bin/python -m unittest \
  g2_adapter.test_spacemouse_cli \
  g2_adapter.test_spacemouse_teleop \
  g2_adapter.test_spacemouse_takeover \
  g2_adapter.test_stage_enter_retry -v
```

Expected: PASS.

- [ ] **Step 5: Commit SpaceMouse migration**

```bash
git add \
  g2_adapter/run_spacemouse_gdk.py \
  g2_adapter/spacemouse_teleop.py \
  g2_adapter/spacemouse_takeover.py \
  g2_adapter/test_spacemouse_cli.py \
  g2_adapter/test_spacemouse_teleop.py \
  g2_adapter/test_spacemouse_takeover.py
git commit -m "feat: use joint impedance for SpaceMouse control"
```

---

### Task 5: Prepare the full flow and standalone formal motion entry points

**Files:**
- Modify: `g2_adapter/run_bilateral_flow_gdk.py`
- Modify: `g2_adapter/move_left_above_head_hinge_gdk.py`
- Modify: `g2_adapter/right_apriltag_grasp_gdk.py`
- Modify: `g2_adapter/run_preset.py`
- Modify: `g2_adapter/run_left_premove_sequence.py`
- Modify: `g2_adapter/test_bilateral_runtime.py`
- Modify: `g2_adapter/test_move_left_above_head_hinge_gdk.py`
- Modify: `g2_adapter/test_right_apriltag_grasp_gdk.py`

**Interfaces:**
- Consumes: `G2Controller.prepare_joint_impedance_control()`
- Produces: every motion-authorized production CLI starts formal motion in mode 3
- Preserves: all stage ordering, visual localization, planner targets, velocities, timeouts, and `--allow-motion` gates

- [ ] **Step 1: Add failing entry-point tests**

For the main flow, assert that preparation occurs after `G2Controller` construction and before `SpaceMouseTakeoverWorker.start()` or the first stage. For left/right standalone grasp tests, replace expected direct `set_control_mode(1)` calls with `prepare_joint_impedance_control()` and assert no motion is sent if preparation raises.

Use call-order assertions of this form:

```python
events = []
controller.prepare_joint_impedance_control.side_effect = lambda: events.append("impedance_ready")
worker.start.side_effect = lambda: events.append("worker_start")
# invoke the entry point
self.assertLess(events.index("impedance_ready"), events.index("worker_start"))
```

- [ ] **Step 2: Run the focused flow tests and verify failures**

Run:

```bash
.venv-sam3/bin/python -m unittest \
  g2_adapter.test_bilateral_runtime \
  g2_adapter.test_move_left_above_head_hinge_gdk \
  g2_adapter.test_right_apriltag_grasp_gdk -v
```

Expected: FAIL because entry points either do not prepare a mode or explicitly request mode 1.

- [ ] **Step 3: Replace direct mode-1 setup with standard preparation**

- In `run_bilateral_flow_gdk.py`, call `controller.prepare_joint_impedance_control()` immediately after the initial state check and before starting any motion-capable worker.
- In `move_left_above_head_hinge_gdk.py`, replace both direct mode-1 switches with `prepare_joint_impedance_control(("left",))`.
- In `right_apriltag_grasp_gdk.py`, replace the standalone action path’s direct mode-1 switch and mode-1 verification with right-arm standard preparation and mode-3 verification.
- In `run_preset.py` and `run_left_premove_sequence.py`, prepare only the arm groups the command will move before the first planner request.
- Do not add mode changes to any branch that lacks `--allow-motion`.

- [ ] **Step 4: Run flow, grasp, preset, and sequence tests**

Run:

```bash
.venv-sam3/bin/python -m unittest \
  g2_adapter.test_bilateral_runtime \
  g2_adapter.test_move_left_above_head_hinge_gdk \
  g2_adapter.test_right_apriltag_grasp_gdk \
  g2_adapter.test_right_post_grasp_sequence \
  g2_adapter.test_left_above_hole \
  g2_adapter.test_left_seat_motion -v
```

Expected: PASS.

- [ ] **Step 5: Commit formal entry-point preparation**

```bash
git add \
  g2_adapter/run_bilateral_flow_gdk.py \
  g2_adapter/move_left_above_head_hinge_gdk.py \
  g2_adapter/right_apriltag_grasp_gdk.py \
  g2_adapter/run_preset.py \
  g2_adapter/run_left_premove_sequence.py \
  g2_adapter/test_bilateral_runtime.py \
  g2_adapter/test_move_left_above_head_hinge_gdk.py \
  g2_adapter/test_right_apriltag_grasp_gdk.py
git commit -m "feat: prepare production flows for joint impedance"
```

---

### Task 6: Remove stale mode-1 assumptions and update operator documentation

**Files:**
- Modify: `G2机器人控制注意事项与依赖路径.md`
- Modify: `常用命令.md`
- Modify: any production/test Python file still identified by the audit below

**Interfaces:**
- Consumes: all behavior implemented in Tasks 1–5
- Produces: no unapproved mode-1 fallback in formal motion; documentation that matches runtime behavior

- [ ] **Step 1: Audit remaining mode-1 references**

Run:

```bash
rg -n "set_control_mode\(1\)|control_mode.?[!=]=.?1|mode-1|模式.?1|位控" \
  g2_adapter G2机器人控制注意事项与依赖路径.md 常用命令.md
```

Classify every hit as one of:

1. Explicit position-mode recovery tool — keep.
2. Historical/documented contrast — rewrite to state the new default accurately.
3. Cartesian mode-2 diagnostic cleanup — keep only where the explicit demo intentionally returns to its documented recovery state.
4. Formal production motion — replace with mode-3 preparation/check and add or update the nearest test before implementation.

- [ ] **Step 2: Add a source audit test for forbidden production fallbacks**

Create a focused unittest in `g2_adapter/test_default_joint_impedance_policy.py` that scans the known formal-motion files from Tasks 2–5 and fails if they contain direct `set_control_mode(1)` calls. Exclude only `set_left_position_control_mode.py` and `demo_left_impedance_move.py` by not placing them in the scanned allowlist.

Run:

```bash
.venv-sam3/bin/python -m unittest g2_adapter.test_default_joint_impedance_policy -v
```

Expected before final cleanup: FAIL if any production fallback remains.

- [ ] **Step 3: Update documentation and remove remaining formal-motion fallbacks**

Update the control guide to say:

```text
正式运动默认 control_mode=3；标准关节 kp 为
400 400 400 500 300 200 200。
拖拽示教期间所选手臂临时使用
50 50 50 60 30 30 30，结束后恢复标准 kp 并保持模式 3。
```

Update `常用命令.md` so the full-flow, SpaceMouse, preset, and teaching sections no longer describe mode 1 as their default. Keep explicit recovery and Cartesian-impedance commands under a clearly marked diagnostic heading.

- [ ] **Step 4: Run the policy test, core regression suite, syntax check, and diff check**

Run:

```bash
.venv-sam3/bin/python -m unittest \
  g2_adapter.test_default_joint_impedance_policy \
  g2_adapter.test_control \
  g2_adapter.test_teach_trajectory_gdk \
  g2_adapter.test_left_teach_placement \
  g2_adapter.test_right_teach_placement \
  g2_adapter.test_spacemouse_cli \
  g2_adapter.test_spacemouse_teleop \
  g2_adapter.test_spacemouse_takeover \
  g2_adapter.test_bilateral_runtime \
  g2_adapter.test_move_left_above_head_hinge_gdk \
  g2_adapter.test_right_apriltag_grasp_gdk -v

.venv-sam3/bin/python -m compileall -q g2_adapter
git diff --check
```

Expected: all tests PASS, `compileall` exits 0, and `git diff --check` emits no output.

- [ ] **Step 5: Commit policy enforcement and documentation**

```bash
git add \
  g2_adapter/test_default_joint_impedance_policy.py \
  G2机器人控制注意事项与依赖路径.md \
  常用命令.md
git add -u g2_adapter
git commit -m "docs: make joint impedance the default control mode"
```

---

### Task 7: Perform staged read-only and supervised hardware verification

**Files:**
- Modify only if a verified discrepancy requires a separately reviewed fix

**Interfaces:**
- Consumes: completed automated implementation from Tasks 1–6
- Produces: recorded evidence that the robot reports mode 3 and follows existing targets with standard stiffness

- [ ] **Step 1: Run a no-motion status check**

Use the existing impedance/status query command documented in `常用命令.md`. Confirm both arms still report the intended stored joint parameters and there is no motion error before enabling movement.

- [ ] **Step 2: Run one supervised single-arm preset with the lowest existing test velocity**

Use `run_preset.py` with an already verified nearby preset and `--allow-motion`. Confirm terminal/log status reports `control_mode=3`, actual arrival succeeds, and the arm holds without unexpected motion.

- [ ] **Step 3: Run SpaceMouse at the existing low default speed**

Confirm mode 3 remains active before, during, and after control; verify translation/rotation direction, release-to-hold behavior, and takeover exit.

- [ ] **Step 4: Record and replay a short single-arm teaching trajectory**

Confirm the arm is manually draggable only during low-kp recording, restores standard stiffness after stopping, and replays the seven-axis trajectory in mode 3.

- [ ] **Step 5: Run the full flow incrementally**

Use `--through-stage` to validate `home`, then left grasp/placement, then right grasp/placement, before running through `finish`. At each boundary inspect the JSONL motion status and actual robot behavior. Stop and open a new debugging task if any mode transition, tracking error, or unexpected compliance appears; do not compensate by switching back to mode 1.
