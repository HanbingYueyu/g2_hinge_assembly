import contextlib
import io
import unittest

import run_bilateral_flow_gdk as flow


class BilateralCliTests(unittest.TestCase):
    def test_help_renders_vlm_mode_and_tts_options(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit) as raised, contextlib.redirect_stdout(output):
            flow.parse_args(["--help"])
        self.assertEqual(raised.exception.code, 0)
        text = output.getvalue()
        self.assertIn("60%", text)
        self.assertIn("--vlm-mode", text)
        self.assertIn("--tts-enable", text)
        args = flow.parse_args([])
        self.assertEqual(args.vlm_mode, "local")
        self.assertIsNone(args.vlm_api_url)
        self.assertIsNone(args.vlm_model)
        self.assertFalse(args.tts_enable)


if __name__ == "__main__":
    unittest.main()
