#!/usr/bin/env python3
"""One read-only web page for all three G2 cameras of the staged flow.

The staged workflow owns three cameras at once (head RGB-D, left D405, right
hand).  Each stage publishes into its own named pane through the same
``update(jpeg, status)`` API the single-camera viewers already use.

The page always shows all three cameras: the head camera on the first row and
the left/right hand cameras side by side on the second row.  There is no
hide/show control; the fallback feed keeps every pane live with the plain
camera image whenever no stage is annotating it.

The module imports no GDK, torch or vision library, so the server can be tested
with plain HTTP calls.
"""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PANE_NAMES = ("head", "left", "right")

PAGE = """<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>__TITLE__</title>
<style>
body{margin:0;background:#161616;color:#eee;font:15px sans-serif}
main{max-width:1600px;margin:auto;padding:14px}
.pane{border:1px solid #333;border-radius:8px;background:#1c1c1c;min-width:0}
.pane-head{padding:8px 12px;background:#222;border-radius:8px 8px 0 0;font-weight:600}
.pane-body{padding:10px}
.pane-body img{width:100%;background:#111;border:1px solid #333;min-height:60px;display:block}
.pane-body pre{white-space:pre-wrap;background:#242424;padding:8px;border-radius:6px;max-height:200px;overflow:auto;margin:8px 0 0}
.row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-top:14px}
@media (max-width:1100px){.row{grid-template-columns:1fr}}
</style>
<main>
<h2>__TITLE__</h2>
<section class='pane' id='pane-head'>
  <div class='pane-head'>HEAD 头部 RGB（hinge / hole）</div>
  <div class='pane-body'><img id='img-head' alt='head 等待相机帧'><pre id='st-head'>waiting</pre></div>
</section>
<div class='row'>
  <section class='pane' id='pane-left'>
    <div class='pane-head'>LEFT 左手 D405</div>
    <div class='pane-body'><img id='img-left' alt='left 等待相机帧'><pre id='st-left'>waiting</pre></div>
  </section>
  <section class='pane' id='pane-right'>
    <div class='pane-head'>RIGHT 右手 AprilTag</div>
    <div class='pane-body'><img id='img-right' alt='right 等待相机帧'><pre id='st-right'>waiting</pre></div>
  </section>
</div>
</main>
<script>
const NAMES = __NAMES__;
const state = {};
async function tick() {
  try {
    const response = await fetch('/status.json', {cache: 'no-store'});
    const payload = await response.json();
    for (const name of NAMES) {
      const pane = payload.panes ? payload.panes[name] : null;
      const status = pane && pane.status ? pane.status : {};
      const sequence = pane ? pane.sequence : 0;
      document.querySelector('#st-' + name).textContent = JSON.stringify(status, null, 2);
      if (sequence !== state[name]) {
        state[name] = sequence;
        document.querySelector('#img-' + name).src = '/frame/' + name + '.jpg?seq=' + sequence;
      }
    }
  } catch (error) {
    for (const name of NAMES) {
      document.querySelector('#st-' + name).textContent = String(error);
    }
  }
}
setInterval(tick, 150); tick();
</script></html>"""


class _Pane:
    """Publisher facade for one named camera pane."""

    def __init__(self, owner: "BilateralWeb", name: str) -> None:
        self._owner = owner
        self._name = name

    def update(self, overlay_jpeg: bytes | None, status: dict) -> None:
        self._owner.update_pane(self._name, overlay_jpeg, status)

    def update_status(self, status: dict) -> None:
        """Publish a status heartbeat while keeping the last image.

        The right-hand demo's own SharedView exposes this, so an injected pane
        must duck-type it or the stage aborts with AttributeError.
        """
        self._owner.update_pane_status(self._name, status)

    def snapshot(self) -> tuple[bytes | None, dict]:
        return self._owner.snapshot_pane(self._name)


class BilateralWeb:
    """Serve the head, left and right camera panes from one HTTP endpoint."""

    def __init__(self, *, host: str = "127.0.0.1", port: int = 8767,
                 title: str = "G2 分阶段双臂流程（只读）") -> None:
        self.host = str(host)
        self.port = int(port)
        self.title = str(title)
        self._lock = threading.Lock()
        self._frames: dict[str, bytes | None] = {name: None for name in PANE_NAMES}
        self._raw_frames = {name: None for name in PANE_NAMES}
        self._raw_status = {name: {} for name in PANE_NAMES}
        self._raw_at = {name: float('-inf') for name in PANE_NAMES}
        self._status: dict[str, dict] = {name: {"state": f"{name}_waiting"} for name in PANE_NAMES}
        self._sequence: dict[str, int] = {name: 0 for name in PANE_NAMES}
        # Last time a stage pushed an annotated frame into this pane.  The raw
        # fallback feed backs off while an overlay is still fresh, so the page
        # shows the YOLO view when one exists and the plain camera otherwise.
        self._overlay_at: dict[str, float] = {name: float("-inf") for name in PANE_NAMES}
        self._hidden: set[str] = set()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # -------------------------------------------------------------- publishing
    def pane(self, name: str) -> _Pane:
        if name not in self._frames:
            raise ValueError(f"unknown camera pane: {name}")
        return _Pane(self, name)

    def update_pane(self, name: str, overlay_jpeg: bytes | None, status: dict) -> None:
        if name not in self._frames:
            raise ValueError(f"unknown camera pane: {name}")
        with self._lock:
            self._frames[name] = overlay_jpeg
            self._status[name] = dict(status)
            self._sequence[name] += 1
            self._overlay_at[name] = time.monotonic()

    def overlay_age(self, name: str, *, now: float | None = None) -> float:
        """Seconds since the last stage overlay, or ``inf`` if there was none."""
        if name not in self._frames:
            raise ValueError(f"unknown camera pane: {name}")
        current = time.monotonic() if now is None else float(now)
        with self._lock:
            published_at = self._overlay_at[name]
        if published_at == float("-inf"):
            return float("inf")
        return max(0.0, current - published_at)

    def publish_raw_pane(
        self, name: str, jpeg: bytes | None, status: dict, *, hold_s: float = 1.0,
    ) -> bool:
        """Publish a raw camera frame unless a stage overlay is still fresh.

        Returns ``True`` when the raw frame was shown, ``False`` when the pane
        is currently owned by an active stage overlay.
        """
        if name not in self._frames:
            raise ValueError(f"unknown camera pane: {name}")
        if jpeg is None:
            return False
        with self._lock:
            self._raw_frames[name] = bytes(jpeg)
            self._raw_status[name] = dict(status)
            self._raw_at[name] = time.monotonic()
            if time.monotonic() - self._overlay_at[name] < float(hold_s):
                return False
            self._frames[name] = jpeg
            self._status[name] = dict(status)
            self._sequence[name] += 1
        return True

    def update(self, overlay_jpeg: bytes | None, status: dict) -> None:
        """Unqualified updates belong to the left D405 pane."""
        self.update_pane("left", overlay_jpeg, status)

    def update_pane_status(self, name: str, status: dict) -> None:
        """Merge a status heartbeat without changing the image or its freshness.

        A status-only update must not suppress the raw fallback feed: that pane
        is not being used for YOLO right now, so the plain camera image is the
        correct thing to show.
        """
        if name not in self._frames:
            raise ValueError(f"unknown camera pane: {name}")
        with self._lock:
            merged = dict(self._status[name])
            merged.update(status)
            merged["sequence"] = self._sequence[name]
            self._status[name] = merged

    def hide_pane(self, name: str) -> None:
        if name not in self._frames:
            raise ValueError(f"unknown camera pane: {name}")
        with self._lock:
            self._hidden.add(name)

    def show_pane(self, name: str) -> None:
        with self._lock:
            self._hidden.discard(name)

    def snapshot_pane(self, name: str) -> tuple[bytes | None, dict]:
        with self._lock:
            return self._frames[name], dict(self._status[name])

    def snapshot_raw_pane(self, name: str) -> tuple[bytes | None, dict]:
        """Original camera JPEG only; never fall back to an annotated frame."""
        with self._lock:
            return self._raw_frames[name], {
                **self._raw_status[name],
                'raw_age_s': max(0.0, time.monotonic() - self._raw_at[name]),
            }

    def snapshot(self) -> tuple[bytes | None, dict]:
        return self.snapshot_pane("left")

    def status_payload(self) -> dict:
        with self._lock:
            return {
                "title": self.title,
                "hidden": sorted(self._hidden),
                "panes": {
                    name: {"sequence": self._sequence[name], "status": dict(self._status[name])}
                    for name in PANE_NAMES
                },
            }

    # ------------------------------------------------------------------ server
    def handler(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:  # noqa: A002
                return

            def _bytes(self, payload: bytes, content_type: str) -> None:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path in {"/", "/index.html"}:
                    page = (
                        PAGE.replace("__TITLE__", owner.title)
                        .replace("__NAMES__", json.dumps(list(PANE_NAMES)))
                    )
                    self._bytes(page.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if path == "/status.json":
                    self._bytes(
                        json.dumps(owner.status_payload(), ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8",
                    )
                    return
                if path.startswith(("/frame/", "/raw/")):
                    raw = path.startswith("/raw/")
                    name = path[len("/raw/") if raw else len("/frame/"):]
                    if name.endswith(".jpg"):
                        name = name[: -len(".jpg")]
                    if name not in PANE_NAMES:
                        self.send_error(HTTPStatus.NOT_FOUND, "unknown pane")
                        return
                    image, _status = owner.snapshot_raw_pane(name) if raw else owner.snapshot_pane(name)
                    if image is None or (raw and _status.get('raw_age_s', float('inf')) > 1.0):
                        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "waiting for first frame")
                        return
                    self._bytes(image, "image/jpeg")
                    return
                if path == "/overlay.jpg":
                    image, _status = owner.snapshot()
                    if image is None:
                        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "waiting for first frame")
                        return
                    self._bytes(image, "image/jpeg")
                    return
                self.send_error(HTTPStatus.NOT_FOUND)

        return Handler

    def start(self) -> str:
        if self._server is not None:
            raise RuntimeError("BilateralWeb 已经在运行")
        self._server = ThreadingHTTPServer((self.host, self.port), self.handler())
        self.port = int(self._server.server_address[1])
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        server, thread = self._server, self._thread
        self._server = self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2.0)

    @property
    def running(self) -> bool:
        return self._server is not None


class RawPaneFeed:
    """Keep one camera pane live with the plain image when no stage overlay exists.

    The staged flow only annotates one camera at a time.  This feed publishes
    the raw frame for its pane whenever the stages are not publishing an
    overlay there. Camera ownership is handed off by pause/resume; the
    original-image cache also refreshes while a recent overlay remains visible.
    """

    def __init__(self, pane_name: str, read_jpeg, web: BilateralWeb, *, hold_s: float = 1.0,
                 fps: float = 6.0, clock=time.monotonic) -> None:
        if pane_name not in PANE_NAMES:
            raise ValueError(f"unknown camera pane: {pane_name}")
        if not (isinstance(fps, (int, float)) and fps > 0.0):
            raise ValueError("fps must be positive")
        if hold_s < 0.0:
            raise ValueError("hold_s must be non-negative")
        self.pane_name = pane_name
        self.read_jpeg = read_jpeg
        self.web = web
        self.hold_s = float(hold_s)
        self.period_s = 1.0 / float(fps)
        self.clock = clock
        self.published = 0
        self.skipped = 0
        self.last_error: BaseException | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._read_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def pause(self) -> None:
        """Stop touching this camera while a stage owns it."""
        self._paused.set()
        with self._read_lock:
            pass  # Drain a read already in flight before a stage takes ownership.

    def resume(self) -> None:
        self._paused.clear()

    def step(self) -> str:
        """Run one iteration; returns paused/overlay/published/empty/error."""
        with self._read_lock:
            return self._step()

    def _step(self) -> str:
        if self._paused.is_set():
            return "paused"
        # When no stage owns the camera, keep the original-image cache fresh
        # even while an overlay remains on the page. Paused feeds still never
        # compete with a stage's camera reader.
        try:
            jpeg = self.read_jpeg()
        except BaseException as exc:  # noqa: BLE001 - reported, never raised
            self.last_error = exc
            # Surface the failure on the page instead of leaving a blank pane.
            try:
                self.web.update_pane_status(self.pane_name, {
                    "state": f"{self.pane_name}_raw_error",
                    "source": "raw_camera",
                    "error": str(exc)[:200],
                    "note": "原画面读取失败，正在按帧率重试",
                })
            except Exception:
                pass
            return "error"
        if not jpeg:
            return "empty"
        status = {
            "state": f"{self.pane_name}_raw_camera",
            "source": "raw_camera",
            "note": "当前没有阶段使用该相机做 YOLO，显示原画面",
        }
        if self.web.publish_raw_pane(self.pane_name, jpeg, status, hold_s=self.hold_s):
            self.published += 1
            return "published"
        self.skipped += 1
        return "overlay"

    def _run(self) -> None:
        while not self._stop.is_set():
            self.step()
            self._stop.wait(self.period_s)

    def start(self) -> "RawPaneFeed":
        if self._thread is not None:
            raise RuntimeError(f"raw pane feed {self.pane_name} is already running")
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name=f"raw-pane-{self.pane_name}", daemon=True,
        )
        self._thread.start()
        return self

    def stop(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, float(timeout_s)))

    @property
    def running(self) -> bool:
        return self._thread is not None
