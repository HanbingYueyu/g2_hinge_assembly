import base64
import unittest
from unittest.mock import Mock, patch
from four_screws_20261006_client import ASSETS, LABELS, FourScrews20261006Client, decode_result


class NewCandidateTests(unittest.TestCase):
    def test_count_mapping_and_parse_failures(self):
        for count in range(5):
            self.assertEqual(decode_result('{"count":'+str(count)+'}')['complete'], 'yes' if count == 4 else 'no')
        self.assertEqual(decode_result('```json\n{"count":"4"}\n```')['complete'], 'yes')
        for raw in ('invalid', '[]', '{}', '{"count":5}', '{"count":"unknown"}'):
            self.assertEqual(decode_result(raw)['complete'], 'unknown')

    def test_original_references_and_only_current_crop_are_sent(self):
        client = FourScrews20261006Client.__new__(FourScrews20261006Client)
        client.prompt = (ASSETS/'prompt.txt').read_text().strip()
        client.references = ((ASSETS/'reference_zero.jpg').read_bytes(), (ASSETS/'reference_four.jpg').read_bytes())
        client.session = Mock()
        client.mode, client.model, client.api_url, client.timeout_s = 'local', 'model', 'http://example', 60
        client.session.post.return_value.json.return_value = {'choices': [{'message': {'content': '{"count":4}'}}]}
        inputs = client.prepare_images([b'old1', b'old2', b'previous', b'current'])
        with patch('four_screws_20261006_client.api_key_from_environment', return_value='test'):
            self.assertEqual(client.run_bytes('four-screws', inputs)['complete'], 'yes')
        payload = client.session.post.call_args.kwargs['json']
        content = payload['messages'][0]['content']
        self.assertEqual(content[0]['text'], client.prompt)
        self.assertEqual([item['text'] for item in content[1::2]], LABELS)
        actual = [base64.b64decode(item['image_url']['url'].split(',', 1)[1]) for item in content[2::2]]
        self.assertEqual(actual, [*client.references, b'current'])
        self.assertEqual(payload['max_tokens'], 100)
        self.assertEqual(payload['temperature'], 0)
        client.session.post.return_value.raise_for_status.side_effect = RuntimeError('network failed')
        with patch('four_screws_20261006_client.api_key_from_environment', return_value='test'):
            with self.assertRaises(RuntimeError):
                client.run_bytes('four-screws', inputs)
