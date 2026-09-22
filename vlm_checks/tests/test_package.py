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
    "prompts/convex_inserted.txt",
    "prompts/four_screws.txt",
    "prompts/right_fixture_grasp.txt",
    "references/convex_inserted/ref_inserted.jpg",
    "references/convex_inserted/ref_not_inserted_side.jpg",
    "references/convex_inserted/ref_not_inserted_front.jpg",
    "references/right_fixture/positive_board.jpg",
    "references/right_fixture/negative_board.jpg",
)


def load_module():
    path = ROOT / "src" / "vlm_three_checks.py"
    spec = importlib.util.spec_from_file_location("vlm_three_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PackageTests(unittest.TestCase):
    def test_lan_v1_base_url_is_normalized_to_chat_completions(self):
        module = load_module()
        self.assertEqual(
            module.normalize_api_url("http://192.168.30.12:8000/v1"),
            "http://192.168.30.12:8000/v1/chat/completions",
        )
        full = "http://192.168.30.12:8000/v1/chat/completions"
        self.assertEqual(module.normalize_api_url(full), full)

    def test_required_runtime_files_exist(self):
        self.assertEqual([name for name in REQUIRED if not (ROOT / name).is_file()], [])

    def test_example_does_not_contain_a_real_key(self):
        text = (ROOT / "config/vlm.env.example").read_text(encoding="utf-8")
        self.assertIn("replace-with-the-same-random-key-as-the-LAN-server", text)

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

    def test_lan_api_key_is_used_without_exposing_it_in_the_payload(self):
        module = load_module()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": '{"decision":"unknown"}'}}],
        }
        with patch.dict(
            module.os.environ,
            {"VLM_LAN_API_KEY": "lan-secret", "DASHSCOPE_API_KEY": ""},
            clear=False,
        ), patch.object(module.requests.Session, "post", return_value=response) as post:
            client = module.VlmClient(
                ROOT / "config" / "missing.env",
                "qwen3.8-flash", "http://192.168.30.12:8000/v1", 10.0,
            )
            client.run_bytes("four-screws", [b"f1", b"f2", b"f3", b"f4"])
        self.assertEqual(post.call_args.args[0], "http://192.168.30.12:8000/v1/chat/completions")
        self.assertEqual(post.call_args.kwargs["headers"]["Authorization"], "Bearer lan-secret")
        self.assertNotIn("lan-secret", repr(post.call_args.kwargs["json"]))

    def test_standalone_fixture_task_collapses_four_live_images_to_one_board(self):
        module = load_module()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": '{"decision":"unknown"}'}}],
        }
        images = [
            ROOT / "references" / "right_fixture" / "positive_board.jpg",
            ROOT / "references" / "right_fixture" / "negative_board.jpg",
            ROOT / "references" / "convex_inserted" / "ref_inserted.jpg",
            ROOT / "references" / "convex_inserted" / "ref_not_inserted_side.jpg",
        ]
        with patch.dict(module.os.environ, {"VLM_LAN_API_KEY": "lan-secret"}), patch.object(
            module.requests, "post", return_value=response,
        ) as post:
            module.run_check(
                "right-fixture-grasp", images, "qwen3-vl-8b-local", "http://192.168.30.12:8000/v1", 10.0,
            )
        content = post.call_args.kwargs["json"]["messages"][0]["content"]
        urls = [item for item in content if item["type"] == "image_url"]
        self.assertEqual(len(urls), 3)

    def test_migrated_api_tasks_are_registered_with_expected_reference_counts(self):
        module = load_module()
        self.assertTrue({"convex-inserted", "four-screws", "right-fixture-grasp"}.issubset(module.TASKS))
        self.assertEqual(len(module.TASKS["convex-inserted"]["references"]), 3)
        self.assertEqual(len(module.TASKS["four-screws"]["references"]), 0)
        self.assertEqual(len(module.TASKS["right-fixture-grasp"]["references"]), 2)

    def test_migrated_tasks_send_api_images_in_reference_then_live_order(self):
        module = load_module()
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [{"message": {"content": '{"decision":"unknown"}'}}],
        }
        with patch.dict(module.os.environ, {"DASHSCOPE_API_KEY": "test-key"}), patch.object(
            module.requests.Session, "post", return_value=response,
        ) as post:
            client = module.VlmClient(
                ROOT / "config" / "missing.env",
                "qwen3.8-flash", module.DEFAULT_API_URL, 10.0,
            )
            client.run_bytes("convex-inserted", [b"current"])
            convex_content = post.call_args.kwargs["json"]["messages"][0]["content"]
            convex_urls = [item for item in convex_content if item["type"] == "image_url"]
            self.assertEqual(len(convex_urls), 4)
            client.run_bytes("four-screws", [b"f1", b"f2", b"f3", b"f4"])
            screws_content = post.call_args.kwargs["json"]["messages"][0]["content"]
            screws_urls = [item for item in screws_content if item["type"] == "image_url"]
            self.assertEqual(len(screws_urls), 4)
            client.run_bytes("right-fixture-grasp", [b"f1", b"f2", b"f3", b"f4"])
            fixture_content = post.call_args.kwargs["json"]["messages"][0]["content"]
            fixture_urls = [item for item in fixture_content if item["type"] == "image_url"]
            self.assertEqual(len(fixture_urls), 6)


if __name__ == "__main__":
    unittest.main()
