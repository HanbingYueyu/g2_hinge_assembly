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
    def test_aligned_hole_vote_uses_revised_success_phrase(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        self.assertTrue(announcer.handle_event({
            'event': 'vlm_hole_alignment_result', 'ok': True, 'aggregate': True,
            'result': {'decision': '无明显大偏差'},
        }))
        announcer.close()
        self.assertEqual(interaction.messages, ['铰链螺丝孔已对齐'])

    def test_workflow_speech_uses_motion_events_and_screw_cycle_start(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        events = [
            ({'event': 'stage_start', 'stage': 'home'}, '现在开始演示冰箱铰链装配锁附作业流程'),
            ({'event': 'left_above_hinge_arrived'}, '现在开始抓取铰链'),
            ({'event': 'left_stage_complete_above_hole'}, '现在插入铰链'),
            ({'event': 'right_lr2_clearance_arrived', 'preset': 'LR2'}, '开始对齐铰链螺丝孔'),
            ({'event': 'right_pr1_premove_arrived', 'preset': 'RR1'}, '开始抓取电批'),
            ({'event': 'dianpi_started'}, '开始打螺丝'),
            ({'event': 'stage_start', 'stage': 'finish'}, '演示完成，回到初始位置'),
        ]
        for event, _ in events:
            self.assertTrue(announcer.handle_event(event))
            self.assertFalse(announcer.handle_event(event))
        self.assertFalse(announcer.handle_event({'event': 'stage_complete', 'stage': 'right_grasp'}))
        announcer.close()
        self.assertEqual(interaction.messages, [message for _, message in events])

    def test_gripper_success_and_loss_repeat_after_regrasp(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        attached = {'event': 'vlm_gripper_result', 'ok': True, 'workflow_phase': 'post_grasp',
                    'stable_state': 'attached', 'result': {'state': 'attached'}}
        lost = {'event': 'vlm_gripper_result', 'ok': True, 'workflow_phase': 'post_grasp',
                'stable_state': 'not_attached', 'result': {'state': 'not_attached'}}
        self.assertTrue(announcer.handle_event(attached))
        self.assertTrue(announcer.handle_event(lost))
        announcer.handle_event({'event': 'vlm_regrasp_requested'})
        self.assertTrue(announcer.handle_event(attached))
        self.assertFalse(announcer.handle_event({'event': 'vlm_gripper_result', 'ok': True,
            'workflow_phase': 'pre_release', 'result': {'state': 'not_attached'}}))
        announcer.close()
        self.assertEqual(interaction.messages, ['铰链抓取成功', '铰链脱落，重新抓取', '铰链抓取成功'])

    def test_visual_regrasp_announces_each_new_attempt_once(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        attached = {'event': 'vlm_gripper_result', 'workflow_phase': 'post_grasp',
                    'stable_state': 'attached', 'result': {'state': 'attached'}}
        self.assertTrue(announcer.handle_event(attached))
        event = {'event': 'left_visual_regrasp_requested', 'regrasp_attempt': 1}
        self.assertTrue(announcer.handle_event(event))
        self.assertFalse(announcer.handle_event(event))
        self.assertTrue(announcer.handle_event(attached))
        self.assertTrue(announcer.handle_event({**event, 'regrasp_attempt': 2}))
        announcer.close()
        self.assertEqual(interaction.messages, ['铰链抓取成功', '铰链脱落，重新抓取',
                                                '铰链抓取成功', '铰链脱落，重新抓取'])

    def test_pre_grasp_visual_retry_does_not_claim_hinge_dropped(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        try:
            self.assertFalse(announcer.handle_event({
                'event': 'left_visual_regrasp_requested', 'regrasp_attempt': 1,
                'regrasp_reason': 'xy_misaligned_near_contact',
            }))
            self.assertTrue(announcer.handle_event({
                'event': 'left_visual_regrasp_requested', 'regrasp_attempt': 2,
                'regrasp_reason': 'post_grasp_target_lost_timeout',
            }))
        finally:
            announcer.close()
        self.assertEqual(interaction.messages, ['铰链脱落，重新抓取'])

    def test_mode_two_keeps_hole_arrival_silent_without_muting_normal_flow(self):
        arrival = {'event': 'left_stage_complete_above_hole'}
        demo = load_module().VlmTtsAnnouncer(demo_interaction := FakeInteraction())
        normal = load_module().VlmTtsAnnouncer(normal_interaction := FakeInteraction())
        try:
            demo.handle_event({'event': 'demo_round_start', 'mode': '2', 'round': 1})
            self.assertTrue(demo.handle_event({'event': 'left_above_hinge_arrived'}))
            self.assertFalse(demo.handle_event(arrival))
            self.assertTrue(normal.handle_event(arrival))
        finally:
            demo.close()
            normal.close()
        self.assertEqual(demo_interaction.messages, ['现在开始抓取铰链'])
        self.assertEqual(normal_interaction.messages, ['现在插入铰链'])

    def test_regrasp_allows_insertion_announcement_again_at_next_hole_arrival(self):
        for regrasp in (
            {'event': 'vlm_regrasp_requested'},
            {'event': 'left_visual_regrasp_requested', 'regrasp_attempt': 1},
        ):
            with self.subTest(regrasp=regrasp['event']):
                interaction = FakeInteraction()
                announcer = load_module().VlmTtsAnnouncer(interaction)
                arrival = {'event': 'left_stage_complete_above_hole'}
                try:
                    self.assertTrue(announcer.handle_event(arrival))
                    self.assertFalse(announcer.handle_event(arrival))
                    announcer.handle_event(regrasp)
                    self.assertTrue(announcer.handle_event(arrival))
                    self.assertFalse(announcer.handle_event(arrival))
                finally:
                    announcer.close()
                self.assertEqual(interaction.messages.count('现在插入铰链'), 2)

    def test_new_screwdriver_demo_round_can_announce_fixture_result_again(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        result = {'event': 'vlm_right_fixture_grasp_result', 'ok': True,
                  'result': {'decision': '抓住'}}
        self.assertTrue(announcer.handle_event(result))
        self.assertFalse(announcer.handle_event(result))
        self.assertFalse(announcer.handle_event({'event': 'demo_round_start', 'mode': '3'}))
        self.assertTrue(announcer.handle_event(result))
        announcer.close()
        self.assertEqual(interaction.messages, ['电批抓取成功', '电批抓取成功'])

    def test_semantic_checks_speak_only_final_decisions(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        self.assertFalse(announcer.handle_event({'event': 'vlm_convex_inserted_result', 'ok': True,
            'review_round': 1, 'result': {'decision': '已插入'}}))
        final = [
            ({'event': 'vlm_convex_inserted_result', 'ok': True, 'aggregate': True,
              'result': {'decision': '已插入'}}, '铰链成功插入'),
            ({'event': 'vlm_hole_alignment_result', 'ok': True, 'aggregate': True,
              'result': {'decision': '明显大偏差'}}, '铰链螺丝孔疑似未对齐'),
            ({'event': 'vlm_right_fixture_grasp_result', 'ok': True,
              'result': {'decision': '抓住'}}, '电批抓取成功'),
            ({'event': 'vlm_four_screws_result', 'ok': True,
              'result': {'latest_decision': 'unknown'}}, '识别到有螺丝未成功打入'),
        ]
        for event, _ in final:
            self.assertTrue(announcer.handle_event(event))
        announcer.close()
        self.assertEqual(interaction.messages, [message for _, message in final])

    def test_failed_or_malformed_vlm_calls_are_not_spoken_as_negative_results(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        for event_name in ('vlm_convex_inserted_result', 'vlm_hole_alignment_result',
                           'vlm_right_fixture_grasp_result', 'vlm_four_screws_result'):
            self.assertFalse(announcer.handle_event({'event': event_name, 'ok': False,
                'aggregate': True, 'result': {'decision': '不确定', 'latest_decision': 'unknown'}}))
            self.assertFalse(announcer.handle_event({'event': event_name, 'ok': True,
                'aggregate': True, 'result': {}}))
        announcer.close()
        self.assertEqual(interaction.messages, [])
    def test_regrasp_resets_only_gripper_speech(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        grip = {'event': 'vlm_gripper_result', 'workflow_phase': 'post_grasp',
                'result': {'state': 'attached'}}
        pin = {'event': 'vlm_convex_inserted_result', 'aggregate': True,
               'result': {'decision': '已插入'}}
        try:
            self.assertTrue(announcer.handle_event(grip))
            self.assertTrue(announcer.handle_event(pin))
            self.assertFalse(announcer.handle_event(grip))
            announcer.handle_event({'event': 'vlm_regrasp_requested'})
            self.assertTrue(announcer.handle_event(grip))
            self.assertFalse(announcer.handle_event(grip))
            self.assertFalse(announcer.handle_event(pin))
        finally:
            announcer.close()
        self.assertEqual(interaction.messages.count('铰链抓取成功'), 2)

    def test_alignment_speaks_only_the_final_vote_and_v47_speaks_chinese_decision(self):
        announcer = load_module().VlmTtsAnnouncer(interaction := FakeInteraction())
        event = {'event': 'vlm_hole_alignment_result', 'ok': True, 'result': {'decision': '明显大偏差'}}
        self.assertFalse(announcer.handle_event(event))
        self.assertTrue(announcer.handle_event({**event, 'aggregate': True}))
        self.assertTrue(announcer.handle_event({'event': 'vlm_right_fixture_grasp_result', 'ok': True, 'result': {'decision': '未抓住'}}))
        announcer.close()
        self.assertEqual(interaction.messages, ['铰链螺丝孔疑似未对齐', '电批疑似抓取失败'])

    def test_identical_task_result_is_spoken_once_and_changed_result_is_spoken(self):
        module = load_module()
        interaction = FakeInteraction()
        announcer = module.VlmTtsAnnouncer(interaction)
        event = {
            "event": "vlm_gripper_result",
            "task": "gripper",
            "ok": True,
            "workflow_phase": "post_grasp",
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
        self.assertEqual(interaction.messages[0], "铰链抓取成功")
        self.assertEqual(interaction.messages[1], "铰链脱落，重新抓取")
        self.assertNotIn("铰链位于两指之间", interaction.messages[0])
        self.assertNotIn("夹持：", interaction.messages[0])

    def test_results_are_announced_as_natural_task_specific_sentences(self):
        module = load_module()
        interaction = FakeInteraction()
        announcer = module.VlmTtsAnnouncer(interaction)
        events = [
            ("vlm_convex_inserted_result", {"decision": "已插入"}, "铰链成功插入"),
            ("vlm_four_screws_result", {"latest_decision": "yes"}, "识别到4颗螺丝成功打入"),
        ]
        for event_name, result, _ in events:
            self.assertTrue(announcer.handle_event({"event": event_name, "ok": True, "result": result,
                                                    "aggregate": event_name == "vlm_convex_inserted_result"}))
        announcer.close()
        self.assertEqual(interaction.messages, [sentence for _, _, sentence in events])

    def test_warnings_are_not_spoken_but_unknown_decision_is_spoken_once(self):
        module = load_module()
        interaction = FakeInteraction()
        announcer = module.VlmTtsAnnouncer(interaction)
        self.assertFalse(announcer.handle_event({
            "event": "vlm_check_warning", "task": "gripper", "error": "no image",
        }))
        unknown = {"event": "vlm_hole_alignment_result", "ok": True, "aggregate": True,
                   "result": {"decision": "不确定"}}
        self.assertTrue(announcer.handle_event(unknown))
        self.assertFalse(announcer.handle_event(unknown))
        announcer.close()
        self.assertEqual(len(interaction.messages), 1)
        self.assertEqual(interaction.messages, ["铰链螺丝孔疑似未对齐"])

    def test_playback_failure_is_reported_without_escaping_event_handler(self):
        module = load_module()
        warnings = []
        interaction = FakeInteraction(RuntimeError("voice service unavailable"))
        announcer = module.VlmTtsAnnouncer(interaction, warning_sink=warnings.append)
        self.assertTrue(announcer.handle_event({
            "event": "vlm_convex_inserted_result", "task": "convex-inserted", "ok": True,
            "aggregate": True, "result": {"decision": "已插入"},
        }))
        announcer.close()
        self.assertEqual(len(warnings), 1)
        self.assertIn("voice service unavailable", warnings[0])

    def test_timed_out_close_reports_worker_still_using_gdk(self):
        module = load_module()
        interaction = BlockingInteraction()
        announcer = module.VlmTtsAnnouncer(interaction)
        announcer.handle_event({
            "event": "vlm_convex_inserted_result", "task": "convex-inserted", "ok": True,
            "aggregate": True, "result": {"decision": "已插入"},
        })
        self.assertTrue(interaction.started.wait(1.0))
        self.assertFalse(announcer.close(timeout_s=0.01))
        self.assertTrue(getattr(announcer, "is_alive", False))
        interaction.release.set()
        self.assertTrue(announcer.close(timeout_s=1.0))
        self.assertFalse(getattr(announcer, "is_alive", True))


if __name__ == "__main__":
    unittest.main()
