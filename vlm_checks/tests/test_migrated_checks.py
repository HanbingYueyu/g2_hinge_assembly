import itertools
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from vlm_checks.src.migrated_checks import alignment_roi, alignment_vote, fixture_board, fuse_grasp
from vlm_checks.src.vlm_three_checks import VlmClient


class MigratedChecksTests(unittest.TestCase):
    def test_left_v2_crop_and_request(self):
        from vlm_checks.src.migrated_checks import gripper_roi
        frame = np.zeros((1056,1280,3), np.uint8)
        frame[40:800,260:1020] = 255
        raw = cv2.imencode('.png', frame)[1].tobytes()
        crop = cv2.imdecode(np.frombuffer(gripper_roi(raw), np.uint8), 1)
        self.assertEqual(crop.shape, (900,900,3))
        self.assertGreater(crop.mean(),254)
        for mode in ('local','online'):
            with patch.dict('os.environ', {'VLM_LOCAL_API_KEY':'test','VLM_ONLINE_API_KEY':'test'}):
                client = VlmClient(Path('/nonexistent'), 'test', 'http://test/v1', 1, mode)
                response = Mock()
                response.json.return_value = {'choices':[{'message':{'content':'{"decision":"no","reason":"背景"}'}}]}
                client.session.post = Mock(return_value=response)
                result = client.run_bytes('gripper',[raw])
                self.assertEqual(result['state'],'not_attached')
                self.assertNotIn('confidence',result)
                payload = client.session.post.call_args.kwargs['json']
                self.assertEqual(payload['max_tokens'],210)
                self.assertEqual(len([c for c in payload['messages'][0]['content'] if c['type']=='image_url']),3)

    def test_all_fusion_combinations(self):
        expected = ['抓住', '抓住', '抓住', '抓住', '未抓住', '不确定', '抓住', '不确定', '不确定']
        for pair, want in zip(itertools.product(('yes', 'no', 'unknown'), repeat=2), expected):
            self.assertEqual(fuse_grasp({'decision': pair[0]}, {'decision': pair[1]})['decision'], want)

    def test_majority_requires_three_of_five_and_retains_unknowns(self):
        for values, want in [(['无明显大偏差'] * 3 + ['明显大偏差'] * 2, '无明显大偏差'),
                             (['无明显大偏差'] * 2 + ['明显大偏差'] * 2 + ['不确定'], '不确定')]:
            self.assertEqual(alignment_vote([{'decision': x} for x in values])['decision'], want)

    def test_fixed_image_preparation(self):
        frame = np.zeros((1056, 1280, 3), np.uint8)
        frame[200:600, 400:900] = 255
        raw = cv2.imencode('.png', frame)[1].tobytes()
        roi = cv2.imdecode(np.frombuffer(alignment_roi(raw), np.uint8), 1)
        self.assertEqual(roi.shape, (713, 900, 3))
        self.assertGreater(roi.mean(), 254)
        board = cv2.imdecode(np.frombuffer(fixture_board([raw] * 4), np.uint8), 1)
        self.assertEqual(board.shape, (1710, 2040, 3))
        with self.assertRaises(ValueError):
            alignment_roi(b'bad')

    def test_backends_use_identical_requests_and_v47_conditional_verifier(self):
        raw = cv2.imencode('.jpg', np.zeros((1056, 1280, 3), np.uint8))[1].tobytes()
        payloads = []
        for mode in ('online', 'local'):
            with patch.dict('os.environ', {'VLM_ONLINE_API_KEY': 'test', 'VLM_LOCAL_API_KEY': 'test'}):
                client = VlmClient(Path('/nonexistent'), 'same-test-model', 'http://test/v1', 1, mode)
                responses = []
                for decision in ('no', 'yes', 'yes'):
                    response = Mock()
                    response.json.return_value = {'choices': [{'message': {'content': '{"decision":"'+decision+'"}'}}]}
                    responses.append(response)
                client.session.post = Mock(side_effect=responses)
                result = client.run_bytes('right-fixture-grasp', [raw] * 4)
                self.assertEqual(result['decision'], '抓住')
                self.assertEqual(client.session.post.call_count, 2)
                client.run_bytes('right-fixture-grasp', [raw] * 4)
                self.assertEqual(client.session.post.call_count, 3)
                payloads.append([c.kwargs['json'] for c in client.session.post.call_args_list])
        self.assertEqual(payloads[0], payloads[1])
        self.assertEqual([p['max_tokens'] for p in payloads[0]], [240, 260, 240])
