import threading
import unittest

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
