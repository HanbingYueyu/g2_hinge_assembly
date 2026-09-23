"""Tests for deduplicated, non-blocking VLM speech announcements."""

import importlib.util
import threading
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("vlm_tts.py")


def load_module():
    if not MODULE_PATH.is_file():
        raise AssertionError("VLM speech announcer has not been implemented")
    spec = importlib.util.spec_from_file_location("vlm_tts", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeInteraction:
    def __init__(self, error=None):
        self.messages = []
        self.error = error
        self.called = threading.Event()

    def play_tts(self, message):
        if self.error:
            raise self.error
        self.messages.append(message)
        self.called.set()


class BlockingInteraction:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def play_tts(self, _message):
        self.started.set()
        self.release.wait()


class VlmTtsTests(unittest.TestCase):
    def test_identical_task_result_is_spoken_once_and_changed_result_is_spoken(self):
        module = load_module()
        interaction = FakeInteraction()
        announcer = module.VlmTtsAnnouncer(interaction)
        event = {
            "event": "vlm_gripper_result",
            "task": "gripper",
            "ok": True,
            "result": {"state": "attached", "reason": "铰链位于两指之间"},
        }
        self.assertTrue(announcer.handle_event(event))
        self.assertFalse(announcer.handle_event({
            **event, "result": {"state": "attached", "reason": "铰链被夹爪牢固夹持"},
        }))
        self.assertTrue(announcer.handle_event({
            **event, "result": {"state": "dropped", "reason": "目标已掉落"},
        }))
        announcer.close()
        self.assertEqual(len(interaction.messages), 2)
        self.assertEqual(interaction.messages[0], "铰链已经夹稳了。")
        self.assertEqual(interaction.messages[1], "铰链掉了，请检查。")
        self.assertNotIn("铰链位于两指之间", interaction.messages[0])
        self.assertNotIn("夹持：", interaction.messages[0])

    def test_results_are_announced_as_natural_task_specific_sentences(self):
        module = load_module()
        interaction = FakeInteraction()
        announcer = module.VlmTtsAnnouncer(interaction)
        events = [
            ("vlm_pin_result", {"state": "not_inserted"}, "凸轴还没有插到位。"),
            ("vlm_c_hole_result", {"obvious_c_leak": False}, "安装位置没有看到明显的 C 孔外露。"),
            ("vlm_convex_inserted_result", {"decision": "已插入"}, "凸轴已经插入到位。"),
            ("vlm_four_screws_result", {"latest_decision": "yes"}, "四颗螺丝看起来都已打好。"),
        ]
        for event_name, result, _ in events:
            self.assertTrue(announcer.handle_event({"event": event_name, "ok": True, "result": result}))
        announcer.close()
        self.assertEqual(interaction.messages, [sentence for _, _, sentence in events])

    def test_warnings_are_not_spoken_but_unknown_decision_is_spoken_once(self):
        module = load_module()
        interaction = FakeInteraction()
        announcer = module.VlmTtsAnnouncer(interaction)
        self.assertFalse(announcer.handle_event({
            "event": "vlm_check_warning", "task": "gripper", "error": "no image",
        }))
        unknown = {
            "event": "vlm_gripper_result", "ok": True, "result": {"state": "unknown"},
        }
        self.assertTrue(announcer.handle_event(unknown))
        self.assertFalse(announcer.handle_event(unknown))
        announcer.close()
        self.assertEqual(len(interaction.messages), 1)
        self.assertEqual(interaction.messages, ["暂时无法判断铰链是否夹稳。"])

    def test_playback_failure_is_reported_without_escaping_event_handler(self):
        module = load_module()
        warnings = []
        interaction = FakeInteraction(RuntimeError("voice service unavailable"))
        announcer = module.VlmTtsAnnouncer(interaction, warning_sink=warnings.append)
        self.assertTrue(announcer.handle_event({
            "event": "vlm_pin_result", "task": "pin", "ok": True,
            "result": {"state": "inserted", "reason": "凸轴已插入"},
        }))
        announcer.close()
        self.assertEqual(len(warnings), 1)
        self.assertIn("voice service unavailable", warnings[0])

    def test_timed_out_close_reports_worker_still_using_gdk(self):
        module = load_module()
        interaction = BlockingInteraction()
        announcer = module.VlmTtsAnnouncer(interaction)
        announcer.handle_event({
            "event": "vlm_pin_result", "task": "pin", "ok": True,
            "result": {"state": "inserted"},
        })
        self.assertTrue(interaction.started.wait(1.0))
        self.assertFalse(announcer.close(timeout_s=0.01))
        self.assertTrue(getattr(announcer, "is_alive", False))
        interaction.release.set()
        self.assertTrue(announcer.close(timeout_s=1.0))
        self.assertFalse(getattr(announcer, "is_alive", True))


if __name__ == "__main__":
    unittest.main()
