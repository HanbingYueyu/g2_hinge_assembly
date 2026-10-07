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


class BoxedScrewTests(unittest.TestCase):
    def test_only_boxes_are_drawn_and_crop_size_is_preserved(self):
        from unittest.mock import patch
        from live_four_screws_vlm import boxed_screw_frame
        frame = np.zeros((1056, 1280, 3), np.uint8)
        details = {'detections': [{'bbox_xyxy': [550, 460, 575, 485], 'confidence': .91}]}
        model = object()
        with patch('right_plate_screw_detection.detect', return_value=(None, None, details)) as detect, \
             patch('live_four_screws_vlm.cv2.putText') as text:
            annotated = boxed_screw_frame(frame, model)
        detect.assert_called_once_with(model, frame, confidence=.5)
        text.assert_not_called()
        self.assertFalse(frame.any())
        self.assertEqual(annotated[460, 550].tolist(), [255, 0, 0])
        data = crop_frame(annotated, (430, 400, 870, 830))
        self.assertEqual(cv2.imdecode(np.frombuffer(data, np.uint8), 1).shape[:2], (430, 440))


class AnnotationSubmissionTests(unittest.TestCase):
    def test_submitted_current_is_annotated_not_raw(self):
        import threading
        from live_four_screws_vlm import LiveReview
        done = threading.Event()
        received = []
        class Client:
            def prepare_images(self, frames):
                return (b'ref0', b'ref4', frames[-1])
            def run_bytes(self, task, images):
                received.append(images)
                done.set()
                return {'latest_decision': 'no'}
        def annotate(frame):
            result = frame.copy()
            cv2.rectangle(result, (3, 3), (16, 16), (255, 0, 0), 2)
            return result
        review = LiveReview(lambda: np.zeros((20, 20, 3), np.uint8), Client(),
                            (0, 0, 20, 20), sample_interval=.01,
                            request_interval=10, annotate_frame=annotate)
        review.start()
        try:
            self.assertTrue(done.wait(2))
            current = cv2.imdecode(np.frombuffer(received[0][-1], np.uint8), 1)
            self.assertGreater(int(current[3, 8, 0]), 150)
            self.assertLess(int(current[3, 8, 1]), 100)
            self.assertEqual(received[0][:2], [b'ref0', b'ref4'])
            self.assertEqual(review.inputs[-1], received[0][-1])
        finally:
            review.stop.set()
            for thread in review.threads:
                thread.join(2)

    def test_annotation_failure_never_submits_raw_frame(self):
        import threading
        from unittest.mock import Mock
        from live_four_screws_vlm import LiveReview
        attempted = threading.Event()
        def fail(frame):
            attempted.set()
            raise RuntimeError('YOLO failed')
        client = Mock()
        review = LiveReview(lambda: np.zeros((20, 20, 3), np.uint8), client,
                            (0, 0, 20, 20), annotate_frame=fail)
        review.start()
        try:
            self.assertTrue(attempted.wait(2))
        finally:
            review.stop.set()
            for thread in review.threads:
                thread.join(2)
        client.run_bytes.assert_not_called()
        self.assertEqual(len(review.frames), 0)
        self.assertIsNone(review.latest)
        self.assertEqual(review.state['camera_error'], 'YOLO failed')
