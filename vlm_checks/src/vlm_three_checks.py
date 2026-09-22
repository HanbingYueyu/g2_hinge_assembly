#!/usr/bin/env python3
"""Three portable DashScope VLM checks for hinge assembly.

This is an offline/API client. It never controls a robot.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import requests


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_MODEL = "qwen3.8-flash"

TASKS = {
    "pin": {
        "prompt": ROOT / "prompts" / "pin_inserted.txt",
        "references": [
            ROOT / "references" / "pin" / "pin_target_B.jpg",
            ROOT / "references" / "pin" / "pin_state_examples.jpg",
        ],
    },
    "c-hole": {
        "prompt": ROOT / "prompts" / "obvious_c_leak.txt",
        "references": [
            ROOT / "references" / "c_hole" / "hole_groups_C_D.jpg",
            ROOT / "references" / "c_hole" / "hole_state_examples.jpg",
        ],
    },
    "gripper": {
        "prompt": ROOT / "prompts" / "gripper_drop.txt",
        "references": [],
    },
    "convex-inserted": {
        "prompt": ROOT / "prompts" / "convex_inserted.txt",
        "references": [
            ROOT / "references" / "convex_inserted" / "ref_inserted.jpg",
            ROOT / "references" / "convex_inserted" / "ref_not_inserted_side.jpg",
            ROOT / "references" / "convex_inserted" / "ref_not_inserted_front.jpg",
        ],
    },
    "four-screws": {
        "prompt": ROOT / "prompts" / "four_screws.txt",
        "references": [],
    },
    "right-fixture-grasp": {
        "prompt": ROOT / "prompts" / "right_fixture_grasp.txt",
        "references": [
            ROOT / "references" / "right_fixture" / "positive_board.jpg",
            ROOT / "references" / "right_fixture" / "negative_board.jpg",
        ],
    },
}


def load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        os.environ.setdefault(name.strip(), value.strip().strip("'\""))


def image_data_url(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def image_bytes_data_url(data: bytes, mime: str = "image/jpeg") -> str:
    if not data:
        raise ValueError("image bytes are empty")
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def build_payload(model: str, prompt: str, image_data_urls: Iterable[str]) -> dict[str, Any]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(
        {"type": "image_url", "image_url": {"url": url}}
        for url in image_data_urls
    )
    return {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
        "seed": 42,
        "enable_thinking": False,
        "max_tokens": 256,
    }


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise ValueError(f"VLM response is not JSON: {text[:200]!r}")
        value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("VLM response is not a JSON object")
    return value


def response_text(data: dict[str, Any]) -> str:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError(f"API response has no choices: {data}")
    content = choices[0].get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict))
    raise ValueError("Unsupported API response content")


class VlmClient:
    """Reusable DashScope client accepting in-memory live JPEG frames."""

    def __init__(self, env_file: Path, model: str, api_url: str, timeout_s: float) -> None:
        load_env_file(Path(env_file))
        self.model = str(model)
        self.api_url = str(api_url)
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()

    def run_bytes(self, task: str, images: list[bytes]) -> dict[str, Any]:
        if task not in TASKS:
            raise ValueError(f"unsupported VLM task: {task}")
        cfg = TASKS[task]
        prompt = cfg["prompt"].read_text(encoding="utf-8")
        urls = [image_data_url(Path(path)) for path in cfg["references"]]
        urls.extend(image_bytes_data_url(image) for image in images)
        key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not key:
            raise RuntimeError("DASHSCOPE_API_KEY is missing; check config/vlm.env")
        response = self.session.post(
            self.api_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=build_payload(self.model, prompt, urls),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return parse_json_object(response_text(response.json()))


def run_check(task: str, images: list[Path], model: str, api_url: str, timeout: float) -> dict[str, Any]:
    cfg = TASKS[task]
    prompt = cfg["prompt"].read_text(encoding="utf-8")
    ordered = [*cfg["references"], *images]
    urls = [image_data_url(Path(path)) for path in ordered]
    payload = build_payload(model, prompt, urls)
    key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DASHSCOPE_API_KEY is missing; check config/vlm.env")
    response = requests.post(
        api_url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_json_object(response_text(response.json()))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Portable API VLM checks: pin, c-hole, gripper, convex-inserted, four-screws, right-fixture-grasp",
    )
    parser.add_argument("task", choices=sorted(TASKS))
    parser.add_argument("--image", action="append", required=True, help="Input JPEG/PNG; repeat in chronological/order sequence")
    parser.add_argument("--model", default=os.environ.get("VLM_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-url", default=os.environ.get("VLM_API_URL", DEFAULT_API_URL))
    parser.add_argument("--env-file", default=str(ROOT / "config" / "vlm.env"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    load_env_file(Path(args.env_file))
    try:
        result = run_check(args.task, [Path(x) for x in args.image], args.model, args.api_url, args.timeout)
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
