"""Asynchronous, deduplicated speech for semantic VLM results."""

from __future__ import annotations

import queue
import threading
from typing import Callable


_RESULT_FIELDS = {
    "vlm_gripper_result": "state",
    "vlm_pin_result": "state",
    "vlm_c_hole_result": "obvious_c_leak",
    "vlm_convex_inserted_result": "decision",
    "vlm_four_screws_result": "latest_decision",
    "vlm_right_fixture_grasp_result": "decision",
}

_RESULT_PHRASES = {
    "vlm_gripper_result": {
        "attached": "铰链已经夹稳了。",
        "not_attached": "还没有夹住铰链。",
        "dropped": "铰链掉了，请检查。",
        "unknown": "暂时无法判断铰链是否夹稳。",
    },
    "vlm_pin_result": {
        "inserted": "凸轴已经插入到位。",
        "not_inserted": "凸轴还没有插到位。",
        "unknown": "暂时无法判断凸轴是否插到位。",
    },
    "vlm_c_hole_result": {
        "true": "安装位置的 C 孔还有明显外露。",
        "false": "安装位置没有看到明显的 C 孔外露。",
        "unknown": "暂时无法判断 C 孔是否外露。",
    },
    "vlm_convex_inserted_result": {
        "已插入": "凸轴已经插入到位。",
        "未插入": "凸轴还没有插到位。",
        "unknown": "暂时无法判断凸轴是否插到位。",
    },
    "vlm_four_screws_result": {
        "yes": "四颗螺丝看起来都已打好。",
        "no": "还有螺丝没有打好。",
        "unknown": "暂时无法判断螺丝是否都已打好。",
    },
    "vlm_right_fixture_grasp_result": {
        "yes": "右手已经夹住电批了。",
        "no": "右手还没有夹住电批。",
        "unknown": "暂时无法判断右手是否夹住电批。",
    },
}


def _result_message(event: dict) -> tuple[tuple[str, str], str] | None:
    event_name = event.get("event")
    spec = _RESULT_FIELDS.get(event_name)
    result = event.get("result")
    if spec is None or not isinstance(result, dict) or event.get("ok") is False:
        return None
    field = spec
    if event_name == "vlm_right_fixture_grasp_result" and field not in result:
        field = "latest_decision"
    if field not in result:
        return None
    raw_value = result[field]
    normalized = str(raw_value).strip().lower() if isinstance(raw_value, str) else str(raw_value).lower()
    if normalized in {"unknown", "none", "不确定", ""}:
        normalized = "unknown"
    message = _RESULT_PHRASES[event_name].get(normalized)
    if message is None:
        return None
    return (str(event_name), normalized), message


class VlmTtsAnnouncer:
    """Queue VLM result speech so TTS latency never blocks robot stages."""

    def __init__(
        self,
        interaction,
        warning_sink: Callable[[str], None] | None = None,
    ) -> None:
        self._interaction = interaction
        self._warning_sink = warning_sink
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._lock = threading.Lock()
        self._spoken_or_queued: set[tuple[str, str]] = set()
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="vlm-tts", daemon=True)
        self._thread.start()

    def handle_event(self, event: dict) -> bool:
        """Queue a supported semantic result once; warnings are never spoken."""
        parsed = _result_message(event)
        if parsed is None:
            return False
        key, message = parsed
        with self._lock:
            if self._closed or key in self._spoken_or_queued:
                return False
            self._spoken_or_queued.add(key)
            self._queue.put(message)
        return True

    def _run(self) -> None:
        while True:
            message = self._queue.get()
            try:
                if message is None:
                    return
                try:
                    self._interaction.play_tts(message)
                except Exception as exc:  # noqa: BLE001 - speech must not affect robot flow
                    if self._warning_sink is not None:
                        try:
                            self._warning_sink(str(exc))
                        except Exception:
                            pass
            finally:
                self._queue.task_done()

    def close(self, timeout_s: float = 3.0) -> bool:
        """Drain queued announcements and stop the worker before GDK release."""
        with self._lock:
            if not self._closed:
                self._closed = True
                self._queue.put(None)
        self._thread.join(timeout=max(0.0, float(timeout_s)))
        return not self._thread.is_alive()

    @property
    def is_alive(self) -> bool:
        """Whether the worker may still be inside a GDK Interaction call."""
        return self._thread.is_alive()
