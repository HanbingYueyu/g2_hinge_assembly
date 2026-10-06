import base64
import unittest
from unittest.mock import Mock, patch
from four_screws_v7_client import ASSETS, LABELS, FourScrewsV7Client, validate_v7


class V7Tests(unittest.TestCase):
    def test_payload_exact_prompt_references_then_latest_two_frames(self):
        client = FourScrewsV7Client.__new__(FourScrewsV7Client)
        client.prompt = (ASSETS/'prompt.txt').read_text(encoding='utf-8')
        client.references = ((ASSETS/'reference_a.jpg').read_bytes(), (ASSETS/'reference_b.jpg').read_bytes())
        client.session = Mock()
        client.mode, client.model, client.api_url, client.timeout_s = 'local', 'test', 'http://example', 60
        result = {'closest_reference': '4', 'count': '4', 'complete': 'yes', 'reason': '四颗可见'}
        client.session.post.return_value.json.return_value = {'choices': [{'message': {'content': __import__('json').dumps(result)}}]}
        images = client.prepare_images([b'old1', b'old2', b'previous', b'current'])
        with patch('four_screws_v7_client.api_key_from_environment', return_value='test'):
            self.assertEqual(client.run_bytes('four-screws', images)['latest_decision'], 'yes')
        payload = client.session.post.call_args.kwargs['json']
        content = payload['messages'][0]['content']
        self.assertEqual(content[0]['text'], client.prompt)
        self.assertEqual([item['text'] for item in content[1::2]], LABELS)
        actual = [base64.b64decode(item['image_url']['url'].split(',', 1)[1]) for item in content[2::2]]
        self.assertEqual(actual, [*client.references, b'previous', b'current'])
        self.assertEqual(payload['max_tokens'], 400)
        self.assertEqual(payload['temperature'], 0)

    def test_v7_count_controls_completion(self):
        for count in ('0', '1', '2', '3', '4', 'unknown'):
            expected = 'unknown' if count == 'unknown' else 'yes' if count == '4' else 'no'
            result = validate_v7({'closest_reference': count, 'count': count, 'complete': expected})
            self.assertEqual(result['latest_decision'], expected)
        with self.assertRaises(ValueError):
            validate_v7({'closest_reference': '3', 'count': '3', 'complete': 'yes'})
