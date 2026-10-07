import unittest
from unittest.mock import Mock, patch
from g2_adapter.check_four_screws_gdk import check_once


class StandaloneScrewsTests(unittest.TestCase):
    def test_uses_full_flow_entry_with_exactly_one_frame(self):
        read = Mock(return_value=b'current raw JPEG')
        client = object()
        expected = {'ok': True, 'result': {'latest_decision': 'no'}}
        with patch('g2_adapter.check_four_screws_gdk.run_four_screws_check', return_value=expected) as check:
            self.assertIs(check_once(read, client), expected)
        read.assert_called_once_with()
        check.assert_called_once_with(None, client, frames_override=[b'current raw JPEG'])

    def test_summary_reports_yolo_and_vlm_separately(self):
        import io
        from contextlib import redirect_stdout
        from g2_adapter.check_four_screws_gdk import print_summary
        output = io.StringIO()
        with redirect_stdout(output):
            print_summary({'ok': True, 'result': {'yolo_count': 3, 'model_count': 4, 'latest_decision': 'yes'}})
        self.assertIn('[YOLO] 螺丝数量：3；判别：不齐全', output.getvalue())
        self.assertIn('[VLM] 螺丝数量：4；判别：齐全', output.getvalue())
