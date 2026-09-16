import threading
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vlm_flow_checks import crop_jpeg, parse_crop_xyxy, run_assembly_checks, run_gripper_check


def jpeg(width=640, height=480):
    ok, encoded = cv2.imencode(".jpg", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class FakeWeb:
    def __init__(self, frames):
        self.frames = frames

    def snapshot_pane(self, name):
        return self.frames.get(name), {"state": f"{name}_raw_camera"}


class VlmFlowChecksTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.directory_patch = patch("vlm_flow_checks.OUTPUT_DIR", self.output, create=True)
        self.directory_patch.start()
        self.addCleanup(self.directory_patch.stop)

    def test_gripper_saves_exact_sent_bytes_with_stage_and_camera(self):
        sent = []
        class Client:
            def run_bytes(self, task, images):
                sent.extend(images)
                return {"state": "attached"}
        run_gripper_check(FakeWeb({"left": jpeg()}), Client(), (80, 120, 560, 480))
        files = sorted(self.output.glob("*.jpg"))
        self.assertEqual(len(files), 2)
        for file in files:
            self.assertRegex(file.name, r"^\d{8}_\d{6}_\d{6}_left_vlm_gripper_check(?:_crop)?\.jpg$")
        full = next(file for file in files if not file.stem.endswith("_crop"))
        crop = next(file for file in files if file.stem.endswith("_crop"))
        self.assertEqual([full.read_bytes(), crop.read_bytes()], sent)

    def test_assembly_saves_shared_frames_once_even_on_api_failure(self):
        class Client:
            def run_bytes(self, task, images):
                raise RuntimeError("timeout")
        frames = {"head": b"head", "left": b"left", "right": b"right"}
        run_assembly_checks(FakeWeb(frames), Client())
        files = list(self.output.glob("*.jpg"))
        self.assertEqual(len(files), 3)
        for camera, data in frames.items():
            file = next(self.output.glob(f"*_{camera}_vlm_assembly_check.jpg"))
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

    def test_gripper_uses_left_full_then_crop(self):
        calls = []
        client = type("Client", (), {"run_bytes": lambda _s, task, images: calls.append((task, images)) or {"state": "attached", "confidence": .9}})()
        record = run_gripper_check(FakeWeb({"left": jpeg()}), client, (80, 120, 560, 480))
        self.assertTrue(record["ok"])
        self.assertEqual(calls[0][0], "gripper")
        self.assertEqual(len(calls[0][1]), 2)

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

    def test_keyboard_interrupt_propagates(self):
        class Client:
            def run_bytes(self, task, images):
                raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            run_gripper_check(FakeWeb({"left": jpeg()}), Client(), (80, 120, 560, 480))


if __name__ == "__main__":
    unittest.main()
