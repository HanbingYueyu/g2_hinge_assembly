import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
import cv2
import numpy as np
from g2_adapter import flow_four_screws as adapter
from vlm_checks.src.vlm_three_checks import VlmClient


class FlowScrewsTests(unittest.TestCase):
    def test_full_flow_task_dispatches_to_new_logic(self):
        client = VlmClient.__new__(VlmClient)
        with patch('g2_adapter.flow_four_screws.run_check', return_value={'latest_decision': 'no'}) as check:
            self.assertEqual(client.run_bytes('four-screws', [b'raw'])['latest_decision'], 'no')
        check.assert_called_once_with(client, [b'raw'])

    def test_latest_frame_count_references_and_crop_are_submitted(self):
        frame = np.zeros((1056, 1280, 3), np.uint8)
        _, jpeg = cv2.imencode('.jpg', frame)
        base = SimpleNamespace(mode='local', model='vlm', api_url='http://example', timeout_s=60)
        with patch.object(adapter, '_MODEL', object()), \
             patch.object(adapter, 'boxed_screw_frame', return_value=(frame, 3)) as annotate, \
             patch('g2_adapter.vlm_flow_checks._save_inputs'), \
             patch.object(adapter.FourScrews20261006Client, 'run_bytes', return_value={'latest_decision': 'no'}) as send:
            result = adapter.run_check(base, [b'old invalid', jpeg.tobytes()])
        self.assertEqual(result['yolo_count'], 3)
        self.assertTrue(annotate.call_args.kwargs['with_count'])
        inputs = send.call_args.args[1]
        self.assertEqual(inputs[:2], (adapter.ASSETS.joinpath('reference_zero.jpg').read_bytes(),
                                    adapter.ASSETS.joinpath('reference_four.jpg').read_bytes()))
        self.assertEqual(cv2.imdecode(np.frombuffer(inputs[-1], np.uint8), 1).shape[:2], (430, 440))
        self.assertEqual(send.call_args.kwargs, {'yolo_count': 3})

    def test_yolo_failure_cannot_submit_raw_images(self):
        base = SimpleNamespace()
        with patch.object(adapter, '_MODEL', object()), \
             patch.object(adapter, 'boxed_screw_frame', side_effect=ValueError('YOLO failed')), \
             patch.object(adapter.FourScrews20261006Client, 'run_bytes') as send:
            with self.assertRaises(ValueError):
                adapter.run_check(base, [b'invalid'])
        send.assert_not_called()
