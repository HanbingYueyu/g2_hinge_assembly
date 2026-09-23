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
DEFAULT_API_URL = os.environ.get("VLM_API_URL", "http://192.168.30.12:8000/v1")
DEFAULT_MODEL = os.environ.get("VLM_MODEL", "qwen3-vl-8b-local")
DEFAULT_ONLINE_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ONLINE_MODEL = "qwen3.8-flash"

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
        "max_tokens": 1024,
    },
    "right-fixture-grasp": {
        "prompt": ROOT / "prompts" / "right_fixture_grasp.txt",
        "references": [
            ROOT / "references" / "right_fixture" / "positive_board.jpg",
            ROOT / "references" / "right_fixture" / "negative_board.jpg",
        ],
        "max_tokens": 1024,
    },
}


def normalize_api_url(url: str) -> str:
    """Accept either an OpenAI-compatible ``/v1`` base or full endpoint."""
    value = str(url).strip().rstrip("/")
    if value.endswith("/chat/completions"):
        return value
    if value.endswith("/v1"):
        return value + "/chat/completions"
    return value + "/chat/completions"


def api_key_from_environment(mode: str = "local") -> str:
    """Resolve credentials for one endpoint without crossing profiles."""
    profile = str(mode).strip().lower()
    if profile == "online":
        return (
            os.environ.get("VLM_ONLINE_API_KEY", "").strip()
            or os.environ.get("DASHSCOPE_API_KEY", "").strip()
        )
    if profile == "local":
        return (
            os.environ.get("VLM_LOCAL_API_KEY", "").strip()
            or os.environ.get("VLM_LAN_API_KEY", "").strip()
            or os.environ.get("DASHSCOPE_API_KEY", "").strip()
        )
    raise ValueError("VLM mode must be 'online' or 'local'")


def resolve_vlm_profile(
    mode: str, model: str | None = None, api_url: str | None = None,
) -> tuple[str, str]:
    """Return selected profile model and base URL; explicit CLI values win."""
    profile = str(mode).strip().lower()
    if profile == "online":
        selected_model = os.environ.get("VLM_ONLINE_MODEL", "").strip() or DEFAULT_ONLINE_MODEL
        selected_url = os.environ.get("VLM_ONLINE_API_URL", "").strip() or DEFAULT_ONLINE_API_URL
    elif profile == "local":
        selected_model = (
            os.environ.get("VLM_LOCAL_MODEL", "").strip()
            or os.environ.get("VLM_MODEL", "").strip()
            or DEFAULT_MODEL
        )
        selected_url = (
            os.environ.get("VLM_LOCAL_API_URL", "").strip()
            or os.environ.get("VLM_API_URL", "").strip()
            or DEFAULT_API_URL
        )
    else:
        raise ValueError("VLM mode must be 'online' or 'local'")
    return str(model).strip() if model else selected_model, str(api_url).strip() if api_url else selected_url


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


def fixture_sequence_board_bytes(images: list[Path]) -> bytes:
    """Encode four source images as the V32 2x2 temporal board."""
    if len(images) != 4:
        raise ValueError("right-fixture-grasp requires four live images")
    import cv2
    import numpy as np

    tiles = []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unable to decode fixture image: {path}")
        tiles.append(cv2.resize(image, (510, 428), interpolation=cv2.INTER_AREA))
    board = np.zeros((856, 1020, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        x = (index % 2) * 510
        y = (index // 2) * 428
        board[y:y + 428, x:x + 510] = tile
        cv2.putText(board, f"frame {index + 1}", (x + 12, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                    .7, (0, 255, 0), 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", board, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("failed to encode right-fixture sequence board")
    return encoded.tobytes()


def build_payload(
    model: str,
    prompt: str,
    image_data_urls: Iterable[str],
    *,
    max_tokens: int = 256,
) -> dict[str, Any]:
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
        "max_tokens": int(max_tokens),
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
    choice = choices[0]
    if choice.get("finish_reason") == "length":
        raise ValueError("VLM response truncated: output token limit reached")
    content = choice.get("message", {}).get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(x.get("text", "") for x in content if isinstance(x, dict))
    raise ValueError("Unsupported API response content")


class VlmClient:
    """Reusable OpenAI-compatible VLM client accepting in-memory JPEG frames."""

    def __init__(
        self, env_file: Path, model: str | None, api_url: str | None,
        timeout_s: float, mode: str = "local",
    ) -> None:
        load_env_file(Path(env_file))
        self.mode = str(mode).strip().lower()
        self.model, profile_url = resolve_vlm_profile(self.mode, model, api_url)
        self.api_url = normalize_api_url(profile_url)
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()

    def run_bytes(self, task: str, images: list[bytes]) -> dict[str, Any]:
        if task not in TASKS:
            raise ValueError(f"unsupported VLM task: {task}")
        cfg = TASKS[task]
        prompt = cfg["prompt"].read_text(encoding="utf-8")
        urls = [image_data_url(Path(path)) for path in cfg["references"]]
        urls.extend(image_bytes_data_url(image) for image in images)
        key = api_key_from_environment(self.mode)
        if not key:
            if self.mode == "online":
                raise RuntimeError("VLM_ONLINE_API_KEY or DASHSCOPE_API_KEY is missing; check config/vlm.env")
            raise RuntimeError("VLM_LOCAL_API_KEY or VLM_LAN_API_KEY is missing; check config/vlm.env")
        response = self.session.post(
            self.api_url,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=build_payload(self.model, prompt, urls, max_tokens=cfg.get("max_tokens", 256)),
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return parse_json_object(response_text(response.json()))


def run_check(
    task: str, images: list[Path], model: str | None, api_url: str | None,
    timeout: float, mode: str = "local",
) -> dict[str, Any]:
    model, api_url = resolve_vlm_profile(mode, model, api_url)
    cfg = TASKS[task]
    prompt = cfg["prompt"].read_text(encoding="utf-8")
    if task == "right-fixture-grasp" and len(images) == 4:
        urls = [image_data_url(Path(path)) for path in cfg["references"]]
        urls.append(image_bytes_data_url(fixture_sequence_board_bytes(images)))
    else:
        ordered = [*cfg["references"], *images]
        urls = [image_data_url(Path(path)) for path in ordered]
    payload = build_payload(model, prompt, urls, max_tokens=cfg.get("max_tokens", 256))
    key = api_key_from_environment(mode)
    if not key:
        if mode == "online":
            raise RuntimeError("VLM_ONLINE_API_KEY or DASHSCOPE_API_KEY is missing; check config/vlm.env")
        raise RuntimeError("VLM_LOCAL_API_KEY or VLM_LAN_API_KEY is missing; check config/vlm.env")
    response = requests.post(
        normalize_api_url(api_url),
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
    parser.add_argument("--vlm-mode", choices=("online", "local"), default="local")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-url", default=None)
    parser.add_argument("--env-file", default=str(ROOT / "config" / "vlm.env"))
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    load_env_file(Path(args.env_file))
    try:
        result = run_check(
            args.task, [Path(x) for x in args.image], args.model, args.api_url,
            args.timeout, args.vlm_mode,
        )
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
