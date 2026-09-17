import unittest
from unittest.mock import patch
from urllib.error import URLError, HTTPError

from test_vlm_live_gdk import acquire_frames, run_task


class LiveVlmTests(unittest.TestCase):
    def test_existing_web_avoids_gdk_camera(self):
        reads = []
        def read(url):
            reads.append(url)
            return b"frame"
        with patch("test_vlm_live_gdk.read_http_jpeg", read), patch("test_vlm_live_gdk.capture_gdk", side_effect=AssertionError("camera opened")):
            frames, source = acquire_frames(("left",), "http://localhost:8767")
        self.assertEqual(frames, {"left": b"frame"})
        self.assertEqual(source, "web")
        self.assertEqual(reads, ["http://localhost:8767/raw/left.jpg"])

    def test_gripper_live_sends_one_original_without_crop(self):
        sent = []
        class Client:
            def run_bytes(self, task, images):
                sent.extend(images)
                return {'state': 'attached'}
        with patch('test_vlm_live_gdk._save_inputs'):
            record = run_task('gripper', {'left': b'original'}, 'web', Client(), None)
        self.assertTrue(record['ok'])
        self.assertEqual(sent, [b'original'])

    def test_unavailable_web_falls_back_to_required_cameras(self):
        with patch("test_vlm_live_gdk.read_http_jpeg", side_effect=URLError("refused")), patch("test_vlm_live_gdk.capture_gdk", return_value={"left": b"live"}) as capture:
            frames, source = acquire_frames(("left",), "http://localhost:8767")
        self.assertEqual(frames, {"left": b"live"})
        self.assertEqual(source, "gdk")
        capture.assert_called_once_with(("left",))

    def test_running_web_without_frame_does_not_open_camera(self):
        with patch("test_vlm_live_gdk.read_http_jpeg", side_effect=HTTPError("url", 503, "waiting", {}, None)), patch("test_vlm_live_gdk.capture_gdk", side_effect=AssertionError("camera opened")):
            with self.assertRaises(HTTPError):
                acquire_frames(("left",), "http://localhost:8767")

    def test_single_assembly_task_sends_only_requested_check(self):
        class Client:
            def run_bytes(self, task, images):
                if task != "pin" or images != [b"right", b"left", b"head"]:
                    raise AssertionError("wrong task or image order")
                return {"state": "inserted"}
        with patch("test_vlm_live_gdk._save_inputs"):
            record = run_task("pin", {"head": b"head", "left": b"left", "right": b"right"}, "web", Client(), (80, 120, 560, 480))
        self.assertTrue(record["ok"])


if __name__ == "__main__":
    unittest.main()
