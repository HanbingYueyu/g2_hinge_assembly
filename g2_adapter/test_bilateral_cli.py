import contextlib
import io
import unittest

import run_bilateral_flow_gdk as flow


class BilateralCliTests(unittest.TestCase):
    def test_help_renders_percentages_and_lan_vlm_url(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(output):
            flow.parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        text = output.getvalue()
        self.assertIn("60%", text)
        args = flow.parse_args([])
        self.assertEqual(args.vlm_api_url, "http://192.168.30.12:8000/v1")
        self.assertEqual(args.vlm_model, "qwen3-vl-8b-local")


if __name__ == "__main__":
    unittest.main()
