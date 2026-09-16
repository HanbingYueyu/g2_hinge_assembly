# G2 VLM Checks Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the useful VLM three-check package into `vlm_checks/` and add non-blocking gripper, pin, and C-hole semantic checks to the staged bilateral workflow using the existing Web JPEG cache.

**Architecture:** Keep the portable DashScope client and production assets in a self-contained `vlm_checks` package, with the real key in a Git-ignored mode-600 file. Add a G2 adapter that consumes thread-safe `BilateralWeb.snapshot_pane()` JPEG bytes, runs gripper checking after left transport, and runs pin/C-hole checks concurrently after LL2; results are terminal/log observations only and never affect motion control.

**Tech Stack:** Python 3.12, `requests`, OpenCV already present in `.venv-sam3`, `concurrent.futures.ThreadPoolExecutor`, `unittest`, GDK camera Web cache.

**Spec:** `docs/superpowers/specs/2026-09-16-vlm-checks-integration-design.md`

## Global Constraints

- Runtime directory name is exactly `vlm_checks`.
- Do not call or import files from `/home/flyfuture/下载/VLM三项判别迁移包_20260916` after migration.
- Migrate only the runtime client, three prompts, four production reference images, env example, verified-results manifest, package test, requirements, and concise README.
- Do not migrate `.pytest_cache`, `__pycache__`, `references/review_all`, `original_gripper_ros`, or historical shell wrappers.
- Copy the real key to `vlm_checks/config/vlm.env`, preserve mode `600`, ignore it explicitly in Git, and never print or log it.
- VLM reads only the existing `BilateralWeb` JPEG cache; it must not pause feeds or call GDK cameras.
- All VLM results and failures are observational. They must never stop, retry, regrasp, or alter robot motion.
- `KeyboardInterrupt` and `SystemExit` must propagate; only ordinary `Exception` values become non-blocking warning records.
- `pin` and `c-hole` execute concurrently with at most two worker threads.
- No real DashScope request is made by automated tests.

---

### Task 1: Migrate the minimal package and protect the real key

**Files:**
- Create: `vlm_checks/__init__.py`
- Create: `vlm_checks/src/__init__.py`
- Create: `vlm_checks/src/vlm_three_checks.py`
- Create: `vlm_checks/prompts/pin_inserted.txt`
- Create: `vlm_checks/prompts/obvious_c_leak.txt`
- Create: `vlm_checks/prompts/gripper_drop.txt`
- Create: `vlm_checks/references/pin/pin_target_B.jpg`
- Create: `vlm_checks/references/pin/pin_state_examples.jpg`
- Create: `vlm_checks/references/c_hole/hole_groups_C_D.jpg`
- Create: `vlm_checks/references/c_hole/hole_state_examples.jpg`
- Create: `vlm_checks/config/vlm.env.example`
- Create locally, ignored: `vlm_checks/config/vlm.env`
- Create: `vlm_checks/docs/verified_results.json`
- Create: `vlm_checks/tests/test_package.py`
- Create: `vlm_checks/requirements.txt`
- Create: `vlm_checks/README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: source migration package `/home/flyfuture/下载/VLM三项判别迁移包_20260916`.
- Produces: importable package `vlm_checks.src.vlm_three_checks`; local secret `vlm_checks/config/vlm.env` unavailable to Git.

- [ ] **Step 1: Add the failing repository-safety test**

Create `vlm_checks/tests/test_package.py` with a test that enumerates only required production files and verifies the env example contains placeholder text rather than a real key:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = (
    "src/vlm_three_checks.py",
    "prompts/pin_inserted.txt",
    "prompts/obvious_c_leak.txt",
    "prompts/gripper_drop.txt",
    "references/pin/pin_target_B.jpg",
    "references/pin/pin_state_examples.jpg",
    "references/c_hole/hole_groups_C_D.jpg",
    "references/c_hole/hole_state_examples.jpg",
    "config/vlm.env.example",
    "docs/verified_results.json",
)

def test_required_runtime_files_exist():
    assert not [name for name in REQUIRED if not (ROOT / name).is_file()]

def test_example_does_not_contain_a_real_key():
    text = (ROOT / "config/vlm.env.example").read_text(encoding="utf-8")
    assert "sk-your-beijing-dashscope-key" in text
```

- [ ] **Step 2: Run the test and verify the migration is incomplete**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-sam3/bin/python -m pytest -q vlm_checks/tests/test_package.py
```

Expected: FAIL because the required files have not been copied.

- [ ] **Step 3: Copy only the approved files and create package markers**

Use explicit source and destination paths; do not recursively copy the whole migration package:

```bash
mkdir -p vlm_checks/{src,prompts,references/pin,references/c_hole,config,docs,tests}
cp '/home/flyfuture/下载/VLM三项判别迁移包_20260916/src/vlm_three_checks.py' vlm_checks/src/
cp '/home/flyfuture/下载/VLM三项判别迁移包_20260916/prompts/'{pin_inserted,obvious_c_leak,gripper_drop}.txt vlm_checks/prompts/
cp '/home/flyfuture/下载/VLM三项判别迁移包_20260916/references/pin/'{pin_target_B,pin_state_examples}.jpg vlm_checks/references/pin/
cp '/home/flyfuture/下载/VLM三项判别迁移包_20260916/references/c_hole/'{hole_groups_C_D,hole_state_examples}.jpg vlm_checks/references/c_hole/
cp '/home/flyfuture/下载/VLM三项判别迁移包_20260916/config/vlm.env.example' vlm_checks/config/
cp '/home/flyfuture/下载/VLM三项判别迁移包_20260916/docs/verified_results.json' vlm_checks/docs/
cp '/home/flyfuture/下载/VLM三项判别迁移包_20260916/requirements.txt' vlm_checks/
install -m 600 '/home/flyfuture/下载/VLM三项判别迁移包_20260916/config/vlm.env' vlm_checks/config/vlm.env
```

Create empty `vlm_checks/__init__.py` and `vlm_checks/src/__init__.py`. Write `vlm_checks/README.md` with the three task definitions, default endpoint/model, local key location, and the warning that checks never control motion.

- [ ] **Step 4: Ignore and verify the secret**

Add this exact line to `.gitignore`:

```gitignore
/vlm_checks/config/vlm.env
```

Run:

```bash
test "$(stat -c '%a' vlm_checks/config/vlm.env)" = 600
git check-ignore -q vlm_checks/config/vlm.env
! git status --short -- vlm_checks/config/vlm.env | grep .
```

Expected: all commands exit 0 and no key content is printed.

- [ ] **Step 5: Run the package test**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-sam3/bin/python -m pytest -q vlm_checks/tests/test_package.py
```

Expected: PASS.

- [ ] **Step 6: Commit only non-secret migrated files**

```bash
git add .gitignore vlm_checks ':!vlm_checks/config/vlm.env'
git commit -m "feat: migrate portable VLM check assets"
```

Before committing, inspect `git diff --cached --name-only` and confirm `vlm_checks/config/vlm.env` is absent.

---

### Task 2: Add an in-memory JPEG API to the portable client

**Files:**
- Modify: `vlm_checks/src/vlm_three_checks.py`
- Modify: `vlm_checks/tests/test_package.py`

**Interfaces:**
- Consumes: prompt/reference paths and env loading from Task 1.
- Produces:
  - `image_bytes_data_url(data: bytes, mime: str = "image/jpeg") -> str`
  - `VlmClient(env_file: Path, model: str, api_url: str, timeout_s: float)`
  - `VlmClient.run_bytes(task: str, images: list[bytes]) -> dict[str, Any]`

- [ ] **Step 1: Write failing byte-client tests**

Add tests using `unittest.mock.patch` on `requests.Session.post`:

```python
def test_image_bytes_data_url_uses_jpeg_base64():
    module = load_module()
    assert module.image_bytes_data_url(b"abc") == "data:image/jpeg;base64,YWJj"

def test_run_bytes_orders_references_before_live_images(monkeypatch, tmp_path):
    module = load_module()
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": '{"pin_inserted":true}'}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(module.requests.Session, "post", post)
    client = module.VlmClient(tmp_path / "missing.env", "qwen3.8-flash", module.DEFAULT_API_URL, 10.0)
    result = client.run_bytes("pin", [b"right", b"left", b"head"])
    assert result["pin_inserted"] is True
    urls = [item for item in post.call_args.kwargs["json"]["messages"][0]["content"] if item["type"] == "image_url"]
    assert len(urls) == 5  # two references followed by three live views
```

- [ ] **Step 2: Run tests and verify missing interfaces**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-sam3/bin/python -m pytest -q vlm_checks/tests/test_package.py
```

Expected: FAIL for missing `image_bytes_data_url` and `VlmClient`.

- [ ] **Step 3: Implement the in-memory client**

Keep the existing CLI and file-based `run_check()` working. Add a persistent `requests.Session`, load the env file in `VlmClient.__init__`, prepend task reference files, and send live images without writing files:

```python
def image_bytes_data_url(data: bytes, mime: str = "image/jpeg") -> str:
    if not data:
        raise ValueError("image bytes are empty")
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

class VlmClient:
    def __init__(self, env_file: Path, model: str, api_url: str, timeout_s: float):
        load_env_file(Path(env_file))
        self.model = str(model)
        self.api_url = str(api_url)
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()

    def run_bytes(self, task: str, images: list[bytes]) -> dict[str, Any]:
        cfg = TASKS[task]
        urls = [image_data_url(path) for path in cfg["references"]]
        urls.extend(image_bytes_data_url(image) for image in images)
        payload = build_payload(self.model, cfg["prompt"].read_text(encoding="utf-8"), urls)
        key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is missing; check config/vlm.env")
        response = self.session.post(
            self.api_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return parse_json_object(response_text(response.json()))
```

- [ ] **Step 4: Run package tests**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-sam3/bin/python -m pytest -q vlm_checks/tests/test_package.py
```

Expected: PASS without a network request.

- [ ] **Step 5: Commit**

```bash
git add vlm_checks/src/vlm_three_checks.py vlm_checks/tests/test_package.py
git commit -m "feat: support in-memory VLM image checks"
```

---

### Task 3: Build the Web-cache flow adapter

**Files:**
- Create: `g2_adapter/vlm_flow_checks.py`
- Create: `g2_adapter/test_vlm_flow_checks.py`

**Interfaces:**
- Consumes:
  - `BilateralWeb.snapshot_pane(name) -> tuple[bytes | None, dict]`
  - `VlmClient.run_bytes(task, images) -> dict`
- Produces:
  - `parse_crop_xyxy(value: str) -> tuple[int, int, int, int]`
  - `crop_jpeg(jpeg: bytes, crop_xyxy: tuple[int, int, int, int]) -> bytes`
  - `run_gripper_check(web, client, crop_xyxy, clock=time.monotonic) -> dict`
  - `run_assembly_checks(web, client, clock=time.monotonic) -> list[dict]`

- [ ] **Step 1: Write failing adapter tests**

Use real JPEG bytes generated in the test with `cv2.imencode`, a fake Web object returning deterministic bytes/status, and a fake client that records task/image order. Start with this ordering test, then add equivalent named tests for concurrency, missing frames, ordinary exceptions, and `KeyboardInterrupt` propagation:

```python
def test_assembly_orders_pin_right_left_head_and_c_hole_left_right(self):
    frames = {"head": b"head", "left": b"left", "right": b"right"}
    web = FakeWeb(frames)
    client = RecordingClient()
    records = run_assembly_checks(web, client)
    self.assertEqual(client.calls, [
        ("pin", [b"right", b"left", b"head"]),
        ("c-hole", [b"left", b"right"]),
    ])
    self.assertEqual([record["task"] for record in records], ["pin", "c-hole"])
```

Implement these additional tests with literal expectations:

- `test_gripper_uses_left_full_then_crop`: assert the first input equals the full JPEG and decode the second input to assert its dimensions match the clipped crop.
- `test_assembly_starts_pin_and_c_hole_concurrently`: use two `threading.Event` barriers; each fake request sets its started event and waits up to one second for the other before returning. Assert neither timed out.
- `test_missing_web_frame_returns_warning_without_client_call`: return `None` for `left`, assert event `vlm_check_warning`, and assert zero client calls.
- `test_http_exception_becomes_warning_but_keyboard_interrupt_propagates`: raise `RuntimeError("timeout")` and assert a warning record, then raise `KeyboardInterrupt` and assert it escapes.

- [ ] **Step 2: Run tests and verify the adapter is absent**

Run:

```bash
PYTHONPATH=g2_adapter .venv-sam3/bin/python -m unittest test_vlm_flow_checks
```

Expected: import failure for `vlm_flow_checks`.

- [ ] **Step 3: Implement snapshot, crop, validation, and records**

Implement `snapshot_jpegs(web, names)` by calling only `web.snapshot_pane(name)`. Do not access `web._frames`, raw feeds, cameras, or GDK. Each result record has this stable shape:

```python
{
    "event": "vlm_gripper_result|vlm_pin_result|vlm_c_hole_result|vlm_check_warning",
    "task": "gripper|pin|c-hole",
    "ok": True,
    "elapsed_s": 1.23,
    "input_status": {"left": {"state": "left_raw_camera"}},
    "result": {"state": "attached", "confidence": 0.94},
}
```

Validate only the task-defining fields:

- gripper: `state` in `attached|not_attached|dropped|unknown`;
- pin: `state` in `inserted|not_inserted|unknown`;
- c-hole: `obvious_c_leak` is `True`, `False`, or `None`.

Convert ordinary exceptions to `vlm_check_warning`; do not catch `BaseException`.

- [ ] **Step 4: Implement concurrent assembly checks**

Use:

```python
with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-assembly") as pool:
    futures = {
        "pin": pool.submit(_safe_check, client, "pin", [right, left, head], input_status, clock),
        "c-hole": pool.submit(_safe_check, client, "c-hole", [left, right], input_status, clock),
    }
```

Return records in deterministic order `[pin_record, c_hole_record]` even if C-hole completes first.

- [ ] **Step 5: Run adapter tests**

Run:

```bash
PYTHONPATH=g2_adapter .venv-sam3/bin/python -m unittest test_vlm_flow_checks
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add g2_adapter/vlm_flow_checks.py g2_adapter/test_vlm_flow_checks.py
git commit -m "feat: adapt Web snapshots for VLM checks"
```

---

### Task 4: Add explicit VLM stages to the bilateral stage plan

**Files:**
- Modify: `g2_adapter/bilateral_stages.py`
- Modify: `g2_adapter/run_bilateral_flow_gdk.py`
- Modify: `g2_adapter/bilateral_runtime.py`
- Modify: `g2_adapter/test_bilateral_stages.py`
- Modify: `g2_adapter/test_bilateral_flow_cli.py`
- Modify: `g2_adapter/test_bilateral_runtime.py`

**Interfaces:**
- Consumes: `run_gripper_check()` and `run_assembly_checks()` from Task 3.
- Produces:
  - stages `Stage.VLM_GRIPPER_CHECK` and `Stage.VLM_ASSEMBLY_CHECK`;
  - handlers `stage_vlm_gripper_check(ctx)` and `stage_vlm_assembly_check(ctx)`;
  - CLI configuration stored in `FlowContext.args`.

- [ ] **Step 1: Write failing stage-order and CLI tests**

Assert the exact stage subsequences:

```python
names = list(stage_names())
assert names[names.index("left_grasp") + 1] == "vlm_gripper_check"
assert names[names.index("left_release_lift_to_ll2") + 1] == "vlm_assembly_check"
assert names[names.index("vlm_assembly_check") + 1] == "right_grasp"
```

Assert defaults:

```python
args = flow.parse_args([])
assert args.disable_vlm_checks is False
assert args.vlm_env_file == PROJECT_ROOT / "vlm_checks/config/vlm.env"
assert args.vlm_model == "qwen3.8-flash"
assert args.vlm_timeout_s == 60.0
assert args.vlm_gripper_crop == "80,120,560,480"
```

Add dispatch tests with patched adapter calls showing every returned record is emitted but a warning still completes the stage.

- [ ] **Step 2: Run tests and verify new stages/flags are absent**

Run:

```bash
PYTHONPATH=g2_adapter .venv-sam3/bin/python -m unittest test_bilateral_stages test_bilateral_flow_cli test_bilateral_runtime
```

Expected: FAIL for missing stage enum values and CLI attributes.

- [ ] **Step 3: Add stage enum entries and plan positions**

Add:

```python
VLM_GRIPPER_CHECK = "vlm_gripper_check"
VLM_ASSEMBLY_CHECK = "vlm_assembly_check"
```

Place the first immediately after `LEFT_GRASP` and the second immediately after `LEFT_RELEASE_LIFT_TO_LL2`. Titles must state that these are read-only semantic checks.

- [ ] **Step 4: Add CLI configuration and validation**

Add exact flags:

```python
--disable-vlm-checks
--vlm-env-file Path(PROJECT_ROOT / "vlm_checks/config/vlm.env")
--vlm-model qwen3.8-flash
--vlm-api-url https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
--vlm-timeout-s 60.0
--vlm-gripper-crop 80,120,560,480
```

Validate timeout in `[1, 300]` seconds and validate crop via `parse_crop_xyxy()` before GDK initialization. Invalid syntax returns exit code 2.

- [ ] **Step 5: Construct one VLM client and implement handlers**

After `BilateralWeb` setup, create one `VlmClient` and store it with `services.put("vlm_client", client)`. If disabled, key missing, or construction fails, store `None` plus a warning string; do not fail `run_flow()`.

Handlers follow this contract:

```python
def stage_vlm_gripper_check(ctx):
    if ctx.args.disable_vlm_checks:
        return {"skipped": True, "reason": "disabled"}
    crop = parse_crop_xyxy(ctx.args.vlm_gripper_crop)
    record = run_gripper_check(ctx.services.web, ctx.services.get("vlm_client"), crop)
    ctx.runtime.emit_event(record)
    return {"task": "gripper", "ok": record["ok"]}
```

Assembly emits both records returned by `run_assembly_checks`. Neither handler raises because of a VLM result or ordinary VLM exception.

- [ ] **Step 6: Add concise terminal summaries**

Extend `console_event_summary()` with:

```text
[VLM 夹持] state=<state> confidence=<confidence> reason=<reason>
[VLM 凸轴] state=<state> confidence=<confidence> reason=<reason>
[VLM C孔] obvious_c_leak=<value> confidence=<confidence> reason=<reason>
[VLM 警告] task=<task> error=<error>
```

Never include request headers, environment values, Base64 strings, or full response bodies in terminal summaries.

- [ ] **Step 7: Run stage tests**

Run:

```bash
PYTHONPATH=g2_adapter .venv-sam3/bin/python -m unittest test_bilateral_stages test_bilateral_flow_cli test_bilateral_runtime test_vlm_flow_checks
```

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add g2_adapter/bilateral_stages.py g2_adapter/run_bilateral_flow_gdk.py g2_adapter/bilateral_runtime.py g2_adapter/test_bilateral_stages.py g2_adapter/test_bilateral_flow_cli.py g2_adapter/test_bilateral_runtime.py
git commit -m "feat: add observational VLM stages to bilateral flow"
```

---

### Task 5: Document operation and perform end-to-end offline verification

**Files:**
- Modify: `常用命令.md`
- Modify: `vlm_checks/README.md`
- Test: all files changed in Tasks 1–4.

**Interfaces:**
- Consumes: final CLI from Task 4.
- Produces: operator command containing no secret and reproducible verification evidence.

- [ ] **Step 1: Update the full-flow command**

Add these flags to `常用命令.md`:

```bash
--vlm-env-file vlm_checks/config/vlm.env \
--vlm-model qwen3.8-flash \
--vlm-timeout-s 60 \
```

Document `--disable-vlm-checks` directly below the command. Do not include the key value.

- [ ] **Step 2: Document observable stage behavior**

In `vlm_checks/README.md`, list the two stage names, image order for each task, Web-cache-only data source, parallel assembly requests, and the explicit statement that results do not affect robot actions.

- [ ] **Step 3: Run package and flow test suites**

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv-sam3/bin/python -m pytest -q vlm_checks/tests/test_package.py
PYTHONPATH=g2_adapter .venv-sam3/bin/python -m unittest \
  test_vlm_flow_checks \
  test_bilateral_stages \
  test_bilateral_flow_cli \
  test_bilateral_runtime
```

Expected: all tests PASS without network access.

- [ ] **Step 4: Verify syntax, secret isolation, and migrated scope**

Run:

```bash
.venv-sam3/bin/python -m py_compile \
  vlm_checks/src/vlm_three_checks.py \
  g2_adapter/vlm_flow_checks.py \
  g2_adapter/run_bilateral_flow_gdk.py
test "$(stat -c '%a' vlm_checks/config/vlm.env)" = 600
git check-ignore -q vlm_checks/config/vlm.env
! git ls-files --error-unmatch vlm_checks/config/vlm.env >/dev/null 2>&1
test ! -e vlm_checks/references/review_all
test ! -e vlm_checks/original_gripper_ros
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 5: Run a read-only CLI preflight**

Run:

```bash
./scripts/run_gdk_sam3_python.sh g2_adapter/run_bilateral_flow_gdk.py --list-stages
```

Expected: output contains `vlm_gripper_check` after `left_grasp` and `vlm_assembly_check` after `left_release_lift_to_ll2`; no GDK motion starts.

- [ ] **Step 6: Commit documentation**

```bash
git add 常用命令.md vlm_checks/README.md
git commit -m "docs: add VLM check flow operation"
```

- [ ] **Step 7: Review final diff without exposing the key**

Run:

```bash
git status --short
git log --oneline -6
git diff HEAD~5 --stat
```

Inspect filenames and diff statistics only. Do not run `git diff --no-index` or any command that prints `vlm_checks/config/vlm.env`.
