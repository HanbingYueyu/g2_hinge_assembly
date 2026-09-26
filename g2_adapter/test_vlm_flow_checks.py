import threading
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vlm_flow_checks import (
    crop_jpeg,
    parse_crop_xyxy,
    run_assembly_checks,
    run_assembly_review,
    run_c_hole_check,
    run_c_hole_review,
    run_hole_alignment_review,
    run_convex_insert_check,
    run_four_screws_check,
    run_gripper_check,
    run_right_fixture_grasp_check,
    start_vlm_run,
    _save_inputs,
)


def jpeg(width=640, height=480):
    ok, encoded = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class FakeWeb:
    def __init__(self, frames):
        self.frames = frames

    def snapshot_raw_pane(self, name):
        return self.frames.get(name), {"state": f"{name}_raw_camera"}


class VlmFlowChecksTests(unittest.TestCase):
    def test_snapshot_rejects_cached_frame_from_before_arrival(self):
        from vlm_flow_checks import _snapshots
        from unittest.mock import Mock
        web = Mock()
        web.snapshot_raw_pane.side_effect = [
            (b'before movement', {'raw_age_s': .5}),
            (b'after arrival', {'raw_age_s': 0.}),
        ]
        with patch('vlm_flow_checks.time.monotonic', return_value=10.), \
             patch('vlm_flow_checks.time.sleep'):
            frames, _ = _snapshots(web, ('right',), not_before=9.9)
        self.assertEqual(frames['right'], b'after arrival')

    def test_hole_alignment_uses_five_right_images_and_majority(self):
        calls, emitted = [], []
        class Client:
            def run_bytes(self, task, images):
                calls.append((task, images))
                return {'decision': '无明显大偏差' if len(calls) <= 3 else '明显大偏差'}
        raw = jpeg(1280, 1056)
        records = run_hole_alignment_review(FakeWeb({'right': raw, 'left': b'wrong camera'}), Client(), emitted.append, sleep=lambda _: None)
        self.assertEqual(calls, [('hole-alignment', [raw])] * 5)
        self.assertEqual(len(emitted), 6)
        self.assertTrue(records[-1]['aggregate'])
        self.assertEqual(records[-1]['result']['decision'], '无明显大偏差')
        self.assertEqual(len(list(self.output.rglob('*_crop.jpg'))), 5)
        self.assertEqual(len(list(self.output.rglob('*_right_vlm_hole_alignment_check_crop.jpg'))), 5)

    def test_strict_only_review_does_not_send_legacy_pin(self):
        calls = []
        class Client:
            def run_bytes(self, task, images):
                calls.append((task, images))
                return {'decision': '已插入'}
        run_assembly_review(FakeWeb({'right': b'right'}), Client(), lambda _: None,
                            sleep=lambda _: None, include_convex=True,
                            include_c_hole=False, include_pin=False)
        self.assertEqual(calls, [('convex-inserted', [b'right'])] * 3)

    def test_assembly_review_refreshes_images_three_rounds_without_motion(self):
        from unittest.mock import Mock
        emit = Mock()
        with patch('vlm_flow_checks.run_assembly_checks', return_value=[{'task': 'pin', 'ok': True}, {'task': 'c-hole', 'ok': True}]) as check:
            records = run_assembly_review(None, None, emit, sleep=lambda _: None)
        self.assertEqual(check.call_count, 3)
        self.assertEqual(emit.call_count, 6)
        self.assertEqual([r['review_round'] for r in records], [1,1,2,2,3,3])

    def test_assembly_review_can_include_strict_convex_check_without_motion(self):
        from unittest.mock import Mock
        emit = Mock()
        with patch('vlm_flow_checks.run_assembly_checks', return_value=[{'task': 'pin', 'ok': True}, {'task': 'c-hole', 'ok': True}]), \
             patch('vlm_flow_checks.run_convex_insert_check', return_value={'task': 'convex-inserted', 'ok': True}) as convex:
            records = run_assembly_review(None, None, emit, sleep=lambda _: None, include_convex=True)
        self.assertEqual(convex.call_count, 3)
        self.assertEqual([r['task'] for r in records], ['pin', 'c-hole', 'convex-inserted'] * 3)

    def test_assembly_review_can_defer_c_hole_until_after_right_gripper_cycle(self):
        class Client:
            def run_bytes(self, task, _images):
                return {"state": "inserted"} if task == "pin" else {"decision": "已插入"}

        web = FakeWeb({"head": b"head", "left": b"left", "right": b"right"})
        with patch("vlm_flow_checks._save_inputs"):
            records = run_assembly_review(
                web, Client(), lambda _record: None, sleep=lambda _: None,
                include_convex=True, include_c_hole=False,
            )
        self.assertEqual([record["task"] for record in records], ["pin", "convex-inserted"] * 3)

    def test_c_hole_check_uses_fresh_left_and_right_original_images(self):
        class Client:
            def __init__(self):
                self.calls = []

            def run_bytes(self, task, images):
                self.calls.append((task, images))
                return {"obvious_c_leak": False}

        client = Client()
        with patch("vlm_flow_checks._save_inputs"):
            record = run_c_hole_check(FakeWeb({"left": b"left-original", "right": b"right-original"}), client)
        self.assertEqual(record["task"], "c-hole")
        self.assertEqual(client.calls, [("c-hole", [b"left-original", b"right-original"])])

    def test_c_hole_review_runs_three_checks_and_emits_each_result(self):
        from unittest.mock import Mock

        emit = Mock()
        with patch("vlm_flow_checks.run_c_hole_check", side_effect=[
            {"task": "c-hole", "ok": True, "result": {"obvious_c_leak": False}},
            {"task": "c-hole", "ok": True, "result": {"obvious_c_leak": False}},
            {"task": "c-hole", "ok": True, "result": {"obvious_c_leak": False}},
        ]) as check:
            records = run_c_hole_review(None, None, emit, sleep=lambda _: None)
        self.assertEqual(check.call_count, 3)
        self.assertEqual([record["review_round"] for record in records], [1, 2, 3])
        self.assertEqual(emit.call_count, 3)
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.directory_patch = patch("vlm_flow_checks.OUTPUT_DIR", self.output, create=True)
        self.directory_patch.start()
        self.addCleanup(self.directory_patch.stop)
        self.run_directory = start_vlm_run()

    def test_images_share_run_directory_and_new_run_gets_new_directory(self):
        first, second = {}, {}
        _save_inputs('stage_a', {'left': b'a'}, first)
        _save_inputs('stage_b', {'right': b'b'}, second)
        first_dir = Path(first['saved_images'][0]).parent
        self.assertEqual(first_dir, Path(second['saved_images'][0]).parent)
        self.assertEqual(first_dir.parent, self.output)
        self.assertRegex(first_dir.name, r'^\d{8}_\d{6}_\d{6}$')
        start_vlm_run()
        third = {}
        _save_inputs('stage_c', {'left': b'c'}, third)
        self.assertNotEqual(first_dir, Path(third['saved_images'][0]).parent)

    def test_gripper_saves_exact_sent_bytes_with_stage_and_camera(self):
        sent = []
        class Client:
            def run_bytes(self, task, images):
                sent.extend(images)
                return {"state": "attached"}
        run_gripper_check(FakeWeb({"left": jpeg()}), Client(), (80, 120, 560, 480))
        files = sorted(self.output.rglob("*.jpg"))
        self.assertEqual(len(files), 2)
        for file in files:
            self.assertRegex(file.name, r"^\d{8}_\d{6}_\d{6}_left_vlm_gripper_check(?:_crop)?\.jpg$")
        full = next(file for file in files if not file.stem.endswith("_crop"))
        crop = next(file for file in files if file.stem.endswith('_crop'))
        self.assertEqual([full.read_bytes(), crop.read_bytes()], sent)

    def test_assembly_saves_shared_frames_once_even_on_api_failure(self):
        class Client:
            def run_bytes(self, task, images):
                raise RuntimeError("timeout")
        frames = {"head": b"head", "left": b"left", "right": b"right"}
        run_assembly_checks(FakeWeb(frames), Client())
        files = list(self.output.rglob("*.jpg"))
        self.assertEqual(len(files), 3)
        for camera, data in frames.items():
            file = next(self.output.rglob(f"*_{camera}_vlm_assembly_check.jpg"))
            self.assertEqual(file.read_bytes(), data)

    def test_save_failure_warns_but_still_calls_vlm(self):
        blocker = self.output / "not_a_directory"
        blocker.touch()
        class Client:
            def run_bytes(self, task, images):
                return {"state": "attached"}
        with patch("vlm_flow_checks.OUTPUT_DIR", blocker):
            with self.assertWarnsRegex(RuntimeWarning, "VLM.*保存"):
                record = run_gripper_check(FakeWeb({"left": jpeg()}), Client(), (80, 120, 560, 480))
        self.assertTrue(record["ok"])

    def test_crop_parser_and_jpeg_crop(self):
        crop = parse_crop_xyxy("80,120,560,480")
        output = cv2.imdecode(np.frombuffer(crop_jpeg(jpeg(), crop), np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(output.shape[:2], (360, 480))

    def test_default_crop_is_center_60_percent_height_40_percent_width(self):
        from vlm_flow_checks import central_gripper_crop
        for width, height, expected in ((1280,1056,(634,512)), (640,480,(288,256))):
            with self.subTest(width=width, height=height):
                result = central_gripper_crop(jpeg(width,height))
                decoded = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape[:2], expected)

    def test_default_crop_samples_center_not_top_left_background(self):
        image = np.zeros((480,640,3), dtype=np.uint8)
        image[96:384,192:448] = 255
        ok, encoded = cv2.imencode('.jpg', image)
        self.assertTrue(ok)
        result = cv2.imdecode(np.frombuffer(crop_jpeg(encoded.tobytes()), np.uint8), cv2.IMREAD_COLOR)
        self.assertGreater(float(result[10:-10,10:-10].mean()), 250.)

    def test_gripper_uses_original_full_then_crop_then_historical_reference(self):
        calls = []
        client = type("Client", (), {"run_bytes": lambda _s, task, images: calls.append((task, images)) or {"state": "attached", "confidence": .9}})()
        record = run_gripper_check(FakeWeb({"left": jpeg()}), client, (80, 120, 560, 480), reference=b'historical')
        self.assertTrue(record["ok"])
        self.assertEqual(calls[0][0], "gripper")
        self.assertEqual(calls[0][1], [jpeg(), crop_jpeg(jpeg(), (80,120,560,480)), b'historical'])

    def test_overlay_never_replaces_original_vlm_input(self):
        from bilateral_web import BilateralWeb
        web = BilateralWeb()
        web.publish_raw_pane('left', jpeg(), {})
        web.update_pane('left', b'annotated', {})
        sent = []
        class Client:
            def run_bytes(self, task, images):
                sent.extend(images)
                return {'state': 'attached'}
        record = run_gripper_check(web, Client(), None)
        self.assertTrue(record['ok'])
        self.assertEqual(sent, [jpeg(), crop_jpeg(jpeg(), None)])
        self.assertEqual(web.snapshot_pane('left')[0], b'annotated')
        self.assertEqual(next(self.output.rglob('*_check.jpg')).read_bytes(), jpeg())

    def test_raw_http_handler_serves_original_and_rejects_stale_frame(self):
        from bilateral_web import BilateralWeb
        web = BilateralWeb()
        web.publish_raw_pane('left', b'original', {})
        web.update_pane('left', b'annotated', {})
        handler_type = web.handler()
        handler = handler_type.__new__(handler_type)
        handler.path = '/raw/left.jpg'
        replies = []
        handler._bytes = lambda data, kind: replies.append(data)
        handler.send_error = lambda code, msg: replies.append(int(code))
        handler.do_GET()
        self.assertEqual(replies, [b'original'])
        web._raw_at['left'] -= 10
        handler.do_GET()
        self.assertEqual(replies, [b'original', 503])

    def test_assembly_refreshes_whole_batch_when_one_camera_is_stale(self):
        calls = {}
        class Web:
            left_reads = 0
            def snapshot_raw_pane(self, name):
                if name == 'left':
                    self.left_reads += 1
                    return b'left', {'raw_age_s': 10 if self.left_reads == 1 else 0}
                if name == 'head':
                    return (b'head-new' if self.left_reads else b'head-old'), {'raw_age_s': 0}
                return b'right', {'raw_age_s': 0}
        class Client:
            def run_bytes(self, task, images):
                calls[task] = images
                return {'state': 'inserted'} if task == 'pin' else {'obvious_c_leak': False}
        with patch('vlm_flow_checks.time.sleep'):
            run_assembly_checks(Web(), Client())
        self.assertEqual(calls['pin'], [b'right', b'left', b'head-new'])

    def test_assembly_orders_images_and_runs_concurrently(self):
        started = {name: threading.Event() for name in ("pin", "c-hole")}
        calls = {}

        class Client:
            def run_bytes(self, task, images):
                calls[task] = images
                started[task].set()
                other = "c-hole" if task == "pin" else "pin"
                if not started[other].wait(1.0):
                    raise RuntimeError("requests were sequential")
                return {"state": "inserted"} if task == "pin" else {"obvious_c_leak": False}

        frames = {"right": b"right", "left": b"left", "head": b"head"}
        records = run_assembly_checks(FakeWeb(frames), Client())
        self.assertEqual(calls["pin"], [b"right", b"left", b"head"])
        self.assertEqual(calls["c-hole"], [b"left", b"right"])
        self.assertEqual([record["task"] for record in records], ["pin", "c-hole"])

    def test_missing_frame_and_exception_are_warnings(self):
        class Client:
            def run_bytes(self, task, images):
                raise RuntimeError("timeout")
        missing = run_gripper_check(FakeWeb({"left": None}), Client(), (80, 120, 560, 480))
        failed = run_gripper_check(FakeWeb({"left": jpeg()}), Client(), (80, 120, 560, 480))
        self.assertEqual(missing["event"], "vlm_check_warning")
        self.assertEqual(failed["event"], "vlm_check_warning")

    def test_convex_check_sends_one_fresh_original_right_frame(self):
        calls = []
        class Client:
            def run_bytes(self, task, images):
                calls.append((task, images))
                return {"decision": "已插入"}
        record = run_convex_insert_check(FakeWeb({"right": b"right-original"}), Client())
        self.assertTrue(record["ok"])
        self.assertEqual(calls, [("convex-inserted", [b"right-original"])])

    def test_screws_and_fixture_checks_send_four_original_frames(self):
        calls = []
        class Client:
            def run_bytes(self, task, images):
                calls.append((task, images))
                return {"latest_decision": "unknown"} if task == "four-screws" else {"decision": "unknown"}
        frames = [jpeg(1280, 1056) for _ in range(4)]
        screws = run_four_screws_check(FakeWeb({"right": frames[-1]}), Client(), frames_override=frames)
        fixture = run_right_fixture_grasp_check(FakeWeb({"right": frames[-1]}), Client(), frames_override=frames)
        self.assertTrue(screws["ok"])
        self.assertTrue(fixture["ok"])
        self.assertEqual(calls[0], ("four-screws", frames))
        self.assertEqual(calls[1][0], "right-fixture-grasp")
        self.assertEqual(calls[1][1], frames)

    def test_keyboard_interrupt_propagates(self):
        class Client:
            def run_bytes(self, task, images):
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run_gripper_check(FakeWeb({"left": jpeg()}), Client(), (80, 120, 560, 480))


if __name__ == "__main__":
    unittest.main()
