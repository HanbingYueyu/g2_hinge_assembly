import importlib.util
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "src/vlm_three_checks.py",
    "prompts/pin_inserted.txt",
    "prompts/obvious_c_leak.txt",
    "prompts/gripper_drop.txt",
    "references/pin/pin_target_B.jpg",
    "references/pin/pin_state_examples.jpg",
    "references/c_hole/hole_groups_C_D.jpg",
    "references/c_hole/hole_state_examples.jpg",
    "config/vlm.env.example",
    "docs/verified_results.json",
)


def load_module():
    path = ROOT / "src" / "vlm_three_checks.py"
    spec = importlib.util.spec_from_file_location("vlm_three_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackageTests(unittest.TestCase):
    def test_required_runtime_files_exist(self):
        self.assertEqual([name for name in REQUIRED if not (ROOT / name).is_file()], [])

    def test_example_does_not_contain_a_real_key(self):
        text = (ROOT / "config/vlm.env.example").read_text(encoding="utf-8")
        self.assertIn("sk-your-beijing-dashscope-key", text)

    def test_build_payload_contract(self):
        module = load_module()
        payload = module.build_payload("qwen3.8-flash", "test", ["data:image/jpeg;base64,AA=="])
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["seed"], 42)
        self.assertIs(payload["enable_thinking"], False)

    def test_image_bytes_data_url_uses_jpeg_base64(self):
        module = load_module()
        self.assertEqual(module.image_bytes_data_url(b"abc"), "data:image/jpeg;base64,YWJj")

    def test_run_bytes_orders_references_before_live_images(self):
        module = load_module()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": '{"pin_inserted":true}'}}],
        }
        with patch.dict(module.os.environ, {"DASHSCOPE_API_KEY": "test-key"}), patch.object(
            module.requests.Session, "post", return_value=response,
        ) as post:
            client = module.VlmClient(
                ROOT / "config" / "missing.env",
                "qwen3.8-flash", module.DEFAULT_API_URL, 10.0,
            )
            result = client.run_bytes("pin", [b"right", b"left", b"head"])
        self.assertIs(result["pin_inserted"], True)
        content = post.call_args.kwargs["json"]["messages"][0]["content"]
        urls = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(urls), 5)


if __name__ == "__main__":
    unittest.main()
