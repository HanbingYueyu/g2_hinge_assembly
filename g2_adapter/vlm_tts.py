"""Asynchronous, deduplicated speech for semantic VLM results."""

from __future__ import annotations

import queue
import threading
from typing import Callable


_FLOW_PHRASES = {
    ("stage_start", "home"): "现在开始演示冰箱铰链装配锁附作业流程",
    ("left_above_hinge_arrived", None): "现在开始抓取铰链",
    ("left_insertion_start", None): "现在插入铰链",
    ("right_lr2_clearance_arrived", None): "开始对齐铰链螺丝孔",
    ("right_pr1_premove_arrived", "RR1"): "开始抓取电批",
    ("dianpi_started", None): "开始打螺丝",
    ("stage_start", "finish"): "演示完成，回到初始位置",
}


def _result_message(event: dict) -> tuple[tuple[str, str], str] | None:
    event_name = event.get("event")
    flow_key = (event_name, event.get("stage") if event_name == "stage_start" else
                event.get("preset") if event_name == "right_pr1_premove_arrived" else None)
    if flow_key in _FLOW_PHRASES:
        return ("flow", str(flow_key)), _FLOW_PHRASES[flow_key]
    if event_name == "vlm_check_start":
        task = event.get("task")
        phrases = {
            "convex-inserted": "正在检查铰链是否插入",
            "hole-alignment": "正在检查螺丝孔是否对齐",
            "right-fixture-grasp": "正在检查电批是否夹住",
            "four-screws": "正在检查四颗螺丝是否打好",
        }
        if task in phrases:
            return ("check_start", task), phrases[task]
        return None
    if event_name == "right_gripper_closed":
        return ("fixture", "True"), "电批抓取成功"
    result = event.get("result") or {}
    if event_name == "vlm_regrasp_requested":
        return ("gripper", "lost"), "铰链脱落，重新抓取"
    if event_name == "left_visual_regrasp_requested":
        reason = event.get("regrasp_reason")
        if reason is not None and reason != "post_grasp_target_lost_timeout":
            return None
        return ("visual_regrasp", str(event.get("regrasp_attempt"))), "铰链脱落，重新抓取"
    if event_name == "vlm_gripper_result":
        if event.get("workflow_phase") != "post_grasp" or event.get("ok") is False:
            return None
        state = event.get("stable_state", result.get("state"))
        if state == "attached":
            return ("gripper", "attached"), "铰链抓取成功"
        if state in {"not_attached", "dropped"}:
            return ("gripper", "lost"), "铰链脱落，重新抓取"
        return None
    if event.get("ok") is False:
        return None
    if event_name == "vlm_convex_inserted_result" and event.get("aggregate"):
        decision = result.get("decision")
        if decision not in {"已插入", "未插入", "不确定"}:
            return None
        success = decision == "已插入"
        return ("pin", str(success)), "铰链成功插入" if success else "铰链疑似未插入"
    if event_name == "vlm_hole_alignment_result" and event.get("aggregate"):
        decision = result.get("decision")
        if decision not in {"无明显大偏差", "明显大偏差", "不确定"}:
            return None
        success = decision == "无明显大偏差"
        return ("hole", str(success)), "铰链螺丝孔已对齐" if success else "铰链螺丝孔疑似未对齐"
    if event_name == "vlm_right_fixture_grasp_result":
        decision = result.get("decision", result.get("latest_decision"))
        if decision not in {"抓住", "未抓住", "不确定", "yes", "no", "unknown"}:
            return None
        return ("fixture", "True"), "电批抓取成功"
    if event_name == "vlm_four_screws_result":
        decision = result.get("latest_decision")
        if decision not in {"yes", "no", "unknown"}:
            return None
        success = decision == "yes"
        return ("screws", str(success)), "识别到4颗螺丝成功打入" if success else "识别到有螺丝未成功打入"
    return None


class VlmTtsAnnouncer:
    """Queue VLM result speech so TTS latency never blocks robot stages."""

    def __init__(
        self,
        interaction,
        warning_sink: Callable[[str], None] | None = None,
        event_sink: Callable[[dict], None] | None = None,
    ) -> None:
        self._interaction = interaction
        self._warning_sink = warning_sink
        self._event_sink = event_sink
        self._queue: queue.Queue[tuple[int, tuple[str, str], str] | None] = queue.Queue()
        self._generation = 0
        self._lock = threading.Lock()
        self._spoken_or_queued: set[tuple[str, str]] = set()
        self._demo_mode: str | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="vlm-tts", daemon=True)
        self._thread.start()

    def handle_event(self, event: dict) -> bool:
        """Queue a supported semantic result once per run or demo round."""
        if event.get("event") == "demo_round_cancelled":
            with self._lock:
                self._generation += 1
                self._spoken_or_queued.clear()
            return False
        if event.get("event") == "demo_round_start":
            with self._lock:
                self._demo_mode = str(event.get("mode"))
                self._generation += 1
                self._spoken_or_queued.clear()
        if event.get("event") in {"vlm_regrasp_requested", "left_visual_regrasp_requested"}:
            with self._lock:
                self._spoken_or_queued.discard(("flow", str(("left_above_hinge_arrived", None))))
        if event.get("event") == "vlm_regrasp_requested":
            with self._lock:
                self._spoken_or_queued.discard(_result_message({"event": "left_insertion_start"})[0])
                loss_already_queued = ("gripper", "lost") in self._spoken_or_queued
                self._spoken_or_queued = {
                    key for key in self._spoken_or_queued if key[0] != "gripper"
                }
                if loss_already_queued:
                    self._spoken_or_queued.add(("gripper", "lost"))
                    return False
        elif event.get("event") == "left_visual_regrasp_requested":
            with self._lock:
                self._spoken_or_queued.discard(_result_message({"event": "left_insertion_start"})[0])
                self._spoken_or_queued.discard(("gripper", "attached"))
        if event.get("event") == "left_insertion_start":
            with self._lock:
                if self._demo_mode == "2":
                    return False
        parsed = _result_message(event)
        if parsed is None:
            return False
        key, message = parsed
        with self._lock:
            if key == ("gripper", "attached"):
                self._spoken_or_queued.discard(("gripper", "lost"))
            if self._closed or key in self._spoken_or_queued:
                return False
            self._spoken_or_queued.add(key)
            generation = self._generation
            self._queue.put((generation, key, message))
        self._audit('tts_queued', message, generation)
        return True

    def _audit(self, event, message, generation, **fields):
        if self._event_sink is not None:
            try:
                self._event_sink({'event': event, 'text': message, 'generation': generation, **fields})
            except Exception:
                pass

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                generation, key, message = item
                with self._lock:
                    if generation != self._generation:
                        continue
                try:
                    self._audit('tts_send_start', message, generation)
                    self._interaction.play_tts(message)
                    self._audit('tts_send_returned', message, generation)
                except Exception as exc:  # noqa: BLE001 - speech must not affect robot flow
                    self._audit('tts_send_failed', message, generation, error=str(exc))
                    with self._lock:
                        if generation == self._generation:
                            self._spoken_or_queued.discard(key)
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
