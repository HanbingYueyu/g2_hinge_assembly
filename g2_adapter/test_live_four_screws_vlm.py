import unittest
import cv2
import numpy as np
from live_four_screws_vlm import crop_frame, parse_crop, check_result


class LiveScrewsTests(unittest.TestCase):
    def test_crop_jpeg_has_requested_dimensions(self):
        data = crop_frame(np.zeros((1056, 1280, 3), dtype=np.uint8), (430, 400, 870, 830))
        self.assertEqual(cv2.imdecode(np.frombuffer(data, np.uint8), 1).shape[:2], (430, 440))

    def test_invalid_crop_is_rejected(self):
        for value in ('0,0,0,10', '-1,0,10,10', '1,2,3'):
            with self.assertRaises(ValueError):
                parse_crop(value)
        with self.assertRaises(ValueError):
            crop_frame(np.zeros((10, 10, 3), np.uint8), (0, 0, 20, 20))

    def test_malformed_decision_is_not_accepted_as_success(self):
        self.assertEqual(check_result({'latest_decision': 'yes'})['latest_decision'], 'yes')
        with self.assertRaises(ValueError):
            check_result({'latest_decision': '成功'})


class InputPreviewTests(unittest.TestCase):
    def test_browser_serves_exact_bytes_submitted_to_vlm(self):
        import json
        import threading
        import time
        from http.server import ThreadingHTTPServer
        from urllib.request import urlopen
        from live_four_screws_vlm import LiveReview
        submitted = []
        ready = threading.Event()
        class Client:
            def run_bytes(self, task, images):
                self_task = task
                submitted.append((self_task, images))
                ready.set()
                return {'latest_decision': 'yes', 'reason': '四颗可见'}
        review = LiveReview(lambda: np.zeros((20, 20, 3), np.uint8), Client(),
                            (0, 0, 10, 10), sample_interval=.01, request_interval=10.)
        server = ThreadingHTTPServer(('127.0.0.1', 0), review.handler())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        review.start()
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            self.assertTrue(ready.wait(2.))
            with urlopen(base+'/status', timeout=2) as response:
                state = json.load(response)
            self.assertEqual(submitted[0][0], 'four-screws')
            self.assertEqual(state['input_count'], 4)
            for index in range(4):
                with urlopen(base+f'/input/{index}?id='+str(state['request_id']), timeout=2) as response:
                    self.assertEqual(response.read(), submitted[0][1][index])
            with urlopen(base, timeout=2) as response:
                self.assertIn('实际提交'.encode(), response.read())
        finally:
            review.stop.set()
            for worker in review.threads:
                worker.join(timeout=2)
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
