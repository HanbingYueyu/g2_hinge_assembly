#!/usr/bin/env python3
"""Read-only SAM3 prompt test for the right-hand camera.

The image is encoded once per cycle and queried with two independent text
prompts.  No Robot object, arm command, gripper command, or force command is
created here.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

try:
    from .camera import decode_color_rgb
except ImportError:
    from camera import decode_color_rgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SharedView:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.image: bytes | None = None
        self.status: dict = {"state": "starting", "sequence": 0}

    def update(self, image: bytes | None, status: dict) -> None:
        with self.lock:
            self.image, self.status = image, dict(status)

    def snapshot(self) -> tuple[bytes | None, dict]:
        with self.lock:
            return self.image, dict(self.status)


def handler_class(shared: SharedView):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            return

        def send_bytes(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            image, status = shared.snapshot()
            if path == "/status.json":
                self.send_bytes(json.dumps(status, ensure_ascii=False).encode(), "application/json; charset=utf-8")
                return
            if path == "/overlay.jpg":
                if image is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE)
                else:
                    self.send_bytes(image, "image/jpeg")
                return
            if path not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            page = """<!doctype html><meta charset='utf-8'><title>右手 SAM3 双提示词测试</title>
<style>body{margin:0;background:#161616;color:#eee;font:15px sans-serif}main{max-width:1280px;margin:auto;padding:14px}img{width:100%;background:#222}pre{white-space:pre-wrap;background:#242424;padding:10px;border-radius:6px}</style>
<main><h2>右手相机 / SAM3 双提示词（只读）</h2><img id='overlay' alt='等待相机帧'><pre id='status'>starting</pre></main>
<script>let s=0;async function tick(){try{let r=await fetch('/status.json',{cache:'no-store'}),j=await r.json();document.querySelector('#status').textContent=JSON.stringify(j,null,2);if(j.sequence!==s){s=j.sequence;document.querySelector('#overlay').src='/overlay.jpg?seq='+s}}catch(e){document.querySelector('#status').textContent=String(e)}}setInterval(tick,350);tick();</script>"""
            self.send_bytes(page.encode(), "text/html; charset=utf-8")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="右手相机 SAM3 双提示词只读测试")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt")
    parser.add_argument("--shaft-prompt", default="the exposed cylindrical metal pin at the end of the hinge")
    parser.add_argument("--hole-prompt", default="the white circular opening in the fixture")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--interval-s", type=float, default=0.75)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8768)
    return parser.parse_args()


def query_prompt(processor: Sam3Processor, image_state, prompt: str) -> list[dict]:
    result = processor.set_text_prompt(prompt, image_state)
    masks = result["masks"].detach().cpu().numpy()
    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
    output: list[dict] = []
    for mask, score in zip(masks, scores):
        binary = np.asarray(mask, dtype=bool).squeeze()
        ys, xs = np.where(binary)
        if xs.size == 0:
            continue
        output.append({
            "prompt": prompt,
            "confidence": round(float(score), 4),
            "center_px": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
            "xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "area_px": int(binary.sum()),
            "mask": binary,
        })
    return output


def draw_overlay(frame_bgr: np.ndarray, detections: list[dict]) -> np.ndarray:
    image = frame_bgr.copy()
    colors = [(0, 128, 255), (0, 255, 0)]
    for index, item in enumerate(detections):
        color = colors[index % len(colors)]
        x0, y0, x1, y1 = item["xyxy"]
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cx, cy = item["center_px"]
        cv2.circle(image, (int(cx), int(cy)), 5, color, -1)
        label = f"{item['prompt'][:24]} {item['confidence']:.2f}"
        cv2.putText(image, label, (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return image


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.threshold <= 1.0 and args.interval_s > 0.0 and args.max_frames >= 0 and 0 <= args.web_port <= 65535):
        print("参数错误：threshold、interval、max-frames 或端口无效。", file=sys.stderr)
        return 2
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        print(f"找不到 SAM3 权重：{checkpoint}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("未检测到 CUDA；请在 G2 的 GPU 环境运行此预览。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2
    shared = SharedView()
    server = ThreadingHTTPServer((args.web_host, args.web_port), handler_class(shared))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"实时页面：http://{args.web_host}:{args.web_port}")
    initialized = False
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=args.threshold)
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        color_type = agibot_gdk.CameraType.kHandRightColor
        camera = agibot_gdk.Camera([color_type])
        next_inference = 0.0
        sequence = 0
        while args.max_frames == 0 or sequence < args.max_frames:
            image = camera.get_latest_image(color_type, 1000.0)
            color_rgb = decode_color_rgb(image, agibot_gdk)
            now = time.monotonic()
            if now < next_inference:
                time.sleep(min(0.01, next_inference - now))
                continue
            started = time.monotonic()
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                image_state = processor.set_image(Image.fromarray(color_rgb))
                detections = query_prompt(processor, image_state, args.shaft_prompt)
                detections += query_prompt(processor, image_state, args.hole_prompt)
            sequence += 1
            status = {
                "event": "frame",
                "sequence": sequence,
                "utc": datetime.now(timezone.utc).isoformat(),
                "state": "detected" if detections else "no_detection",
                "inference_ms": round((time.monotonic() - started) * 1000.0, 1),
                "prompts": {"shaft": args.shaft_prompt, "hole": args.hole_prompt},
                "detections": [{key: value for key, value in item.items() if key != "mask"} for item in detections],
                "motion": "none; read-only preview",
            }
            overlay = draw_overlay(cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR), detections)
            ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
            shared.update(encoded.tobytes() if ok else None, status)
            print(json.dumps(status, ensure_ascii=False), flush=True)
            next_inference = time.monotonic() + args.interval_s
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"右手 SAM3 预览失败：{exc}", file=sys.stderr)
        return 1
    finally:
        server.shutdown()
        server.server_close()
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

