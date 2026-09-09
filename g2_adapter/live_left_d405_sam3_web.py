#!/usr/bin/env python3
"""Read left G2 RGB-D and display SAM3 results in a local browser.

This is intentionally a perception-only program.  It imports no Robot or
gripper API, creates only a GDK Camera, and never saves camera frames.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import torch

try:
    from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from .localize_left_d405_sam3 import draw_overlay, eligible_candidates, parse_roi, select_candidate, stats_for_mask
    from .sam_crop import center_crop_image, draw_crop_box, expand_crop_masks
except ImportError:
    from camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from localize_left_d405_sam3 import draw_overlay, eligible_candidates, parse_roi, select_candidate, stats_for_mask
    from sam_crop import center_crop_image, draw_crop_box, expand_crop_masks


DEPTH_SCALE_M = 0.001


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="单进程实时查看左手 GDK RGB-D 的 SAM3 检测结果；不含任何机器人控制。"
    )
    parser.add_argument("--host", default="127.0.0.1", help="网页监听地址，默认仅本机。")
    parser.add_argument("--port", type=int, default=8766, help="网页端口，默认 8766。")
    parser.add_argument("--inference-fps", type=float, default=1.0, help="最大 SAM3 推理频率，默认 1 FPS；0 表示不额外限速。")
    parser.add_argument("--warmup", type=int, default=5, help="开始前丢弃的左手彩色帧数。")
    parser.add_argument("--prompt", default="silver triangular hinge", help="SAM3 英文文本提示。")
    parser.add_argument("--threshold", type=float, default=0.10, help="SAM3 置信度阈值。")
    parser.add_argument("--min-area", type=int, default=1, help="候选最小掩膜面积。")
    parser.add_argument("--min-valid-depth-pixels", type=int, default=50, help="候选最少有效对齐深度像素。")
    parser.add_argument("--roi", default="none", metavar="X0,Y0,X1,Y1", help="检测区域，彩色图像像素坐标；默认不限制。")
    parser.add_argument("--selection", choices=("largest", "leftmost", "rightmost"), default="leftmost", help="合格候选选择策略。")
    parser.add_argument("--track-max-distance-px", type=float, default=120.0, help="锁定目标在相邻帧允许的最大中心移动像素，默认 120。")
    parser.add_argument("--track-hold-frames", type=int, default=12, help="锁定目标丢失后拒绝切换的帧数，默认 12。")
    parser.add_argument("--checkpoint", default="checkpoints/sam3.1_multiplex.pt", help="SAM3 权重相对项目根目录的路径。")
    return parser.parse_args()


class SharedView:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.overlay_jpeg: bytes | None = None
        self.status: dict = {"state": "starting", "sequence": 0}

    def update(self, overlay_jpeg: bytes | None, status: dict) -> None:
        with self.lock:
            self.overlay_jpeg = overlay_jpeg
            self.status = status

    def snapshot(self) -> tuple[bytes | None, dict]:
        with self.lock:
            return self.overlay_jpeg, dict(self.status)


def handler_class(shared: SharedView, *, title: str = "G2 左手 D405 / SAM3（只读）"):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
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
                self.send_bytes(json.dumps(status, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")
                return
            if path == "/overlay.jpg":
                if image is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "waiting for first frame")
                else:
                    self.send_bytes(image, "image/jpeg")
                return
            if path not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            page = """<!doctype html><meta charset='utf-8'><title>__TITLE__</title>
<style>body{margin:0;background:#161616;color:#eee;font:15px sans-serif}main{max-width:1280px;margin:auto;padding:14px}img{width:100%;background:#222}pre{white-space:pre-wrap;background:#242424;padding:10px;border-radius:6px}</style>
<main><h2>__TITLE__</h2><img id='overlay' alt='等待相机帧'><pre id='status'>starting</pre></main>
<script>let s=0;async function tick(){try{let r=await fetch('/status.json',{cache:'no-store'});let j=await r.json();document.querySelector('#status').textContent=JSON.stringify(j,null,2);if(j.sequence!==s){s=j.sequence;document.querySelector('#overlay').src='/overlay.jpg?seq='+s}}catch(e){document.querySelector('#status').textContent=String(e)}}setInterval(tick,100);tick();</script>""".replace("__TITLE__", title)
            self.send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")

    return Handler


def run() -> int:
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    args = parse_args()
    if (
        args.inference_fps < 0
        or args.warmup < 0
        or args.track_max_distance_px <= 0
        or args.track_hold_frames < 0
        or not 0 <= args.threshold <= 1
    ):
        print("参数错误：FPS/预热/跟踪参数无效，--threshold 必须在 [0, 1]。", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("未检测到 CUDA，不能运行实时 SAM3。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    shared = SharedView()
    server = ThreadingHTTPServer((args.host, args.port), handler_class(shared))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"实时页面：http://{args.host}:{args.port}")
    print("只读模式：仅访问左手 Camera、TF 和本地 SAM3；按 Ctrl-C 停止。")

    initialized = False
    camera = None
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        color_type = agibot_gdk.CameraType.kHandLeftColor
        depth_type = agibot_gdk.CameraType.kHandLeftDepth
        camera = agibot_gdk.Camera([color_type, depth_type])
        for _ in range(args.warmup):
            camera.get_latest_image(color_type, 1000.0)

        color_intrinsic = None
        depth_intrinsic = None
        depth_to_color = None
        end_link_transform = None
        checkpoint = args.checkpoint
        print(f"loading_checkpoint={checkpoint}", flush=True)
        model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=args.threshold)
        interval = 0.0 if args.inference_fps == 0 else 1.0 / args.inference_fps
        sequence = 0
        previous_completed_at: float | None = None
        tracked_center: np.ndarray | None = None
        missed_track_frames = 0
        while True:
            started = time.monotonic()
            try:
                color = camera.get_latest_image(color_type, 1000.0)
                depth = camera.get_nearest_image(depth_type, int(color.timestamp_ns), 1000.0)
                color_rgb = decode_color_rgb(color, agibot_gdk)
                color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                depth_u16 = decode_depth_u16(depth, agibot_gdk)
                if color_intrinsic is None:
                    color_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height))
                    depth_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height))
                    tf = agibot_gdk.TF()
                    depth_to_color = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor))
                    end_link_transform = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink))
                aligned_depth = align_depth_to_color(
                    depth_u16,
                    depth_intrinsic=depth_intrinsic,
                    color_intrinsic=color_intrinsic,
                    depth_to_color=depth_to_color,
                    depth_scale_m=DEPTH_SCALE_M,
                )
                sam_input, crop_bounds = center_crop_image(color_rgb)
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    state = processor.set_image(Image.fromarray(sam_input))
                    output = processor.set_text_prompt(args.prompt, state)
                crop_masks = [np.asarray(mask, dtype=bool).squeeze() for mask in output["masks"].detach().cpu().numpy()]
                masks = expand_crop_masks(crop_masks, full_shape=color_rgb.shape[:2], bounds=crop_bounds)
                scores = output["scores"].detach().float().cpu().numpy().reshape(-1)
                if len(masks) != len(scores):
                    raise RuntimeError(f"SAM3 掩膜和置信度数量不一致：{len(masks)} != {len(scores)}")
                statistics = [
                    stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform)
                    for mask in masks
                ]
                for item, score in zip(statistics, scores):
                    item["sam3_confidence"] = float(score)
                roi = parse_roi(args.roi, color_bgr.shape[:2])
                eligible = eligible_candidates(
                    statistics,
                    min_area=args.min_area,
                    min_valid_depth_pixels=args.min_valid_depth_pixels,
                    roi=roi,
                    require_depth=False,
                )
                def choose(candidates: list[int]) -> int | None:
                    if not candidates:
                        return None
                    if args.selection == "largest":
                        return max(candidates, key=lambda index: statistics[index]["area_px"])
                    if args.selection == "leftmost":
                        return min(candidates, key=lambda index: statistics[index]["center_px"][0])
                    return max(candidates, key=lambda index: statistics[index]["center_px"][0])
                tracking_state = "acquired"
                if tracked_center is not None:
                    nearby = [
                        index for index in eligible
                        if float(np.linalg.norm(np.asarray(statistics[index]["center_px"]) - tracked_center))
                        <= args.track_max_distance_px
                    ]
                    if nearby:
                        selected = min(
                            nearby,
                            key=lambda index: float(np.linalg.norm(np.asarray(statistics[index]["center_px"]) - tracked_center)),
                        )
                        tracked_center = np.asarray(statistics[selected]["center_px"], dtype=np.float64)
                        missed_track_frames = 0
                        tracking_state = "locked"
                    else:
                        missed_track_frames += 1
                        if missed_track_frames <= args.track_hold_frames:
                            selected = None
                            tracking_state = "target_lost_holding_lock"
                        else:
                            tracked_center = None
                            missed_track_frames = 0
                            selected = choose(eligible)
                            tracking_state = "reacquired" if selected is not None else "no_usable_target"
                else:
                    selected = choose(eligible)
                if selected is not None and tracked_center is None:
                    tracked_center = np.asarray(statistics[selected]["center_px"], dtype=np.float64)
                    missed_track_frames = 0
                depth_ready = (
                    selected is not None
                    and statistics[selected]["valid_depth_px"] >= args.min_valid_depth_pixels
                    and statistics[selected]["xyz_color_camera_m"] is not None
                )
                overlay = draw_crop_box(draw_overlay(color_bgr, masks, statistics, selected), crop_bounds)
                completed_at = time.monotonic()
                elapsed_ms = (completed_at - started) * 1000.0
                processing_fps = None if previous_completed_at is None else 1.0 / max(completed_at - previous_completed_at, 1e-9)
                previous_completed_at = completed_at
                status = {
                    "state": (
                        "ok" if depth_ready
                        else "target_depth_insufficient" if selected is not None
                        else tracking_state
                    ),
                    "sequence": sequence,
                    "prompt": args.prompt,
                    "mask_count": len(statistics),
                    "selected_mask_index": selected,
                    "selection": args.selection,
                    "tracking": {
                        "state": tracking_state,
                        "max_distance_px": args.track_max_distance_px,
                        "hold_frames": args.track_hold_frames,
                        "missed_frames": missed_track_frames,
                    },
                    "roi_xyxy": roi,
                    "sam_center_crop_xyxy": list(crop_bounds),
                    "sam_center_crop_fraction": 0.70,
                    "inference_ms": round(elapsed_ms, 1),
                    "processing_fps": None if processing_fps is None else round(processing_fps, 2),
                    "color_depth_delta_ms": round(abs(int(color.timestamp_ns) - int(depth.timestamp_ns)) / 1_000_000.0, 3),
                    "selected": None if selected is None else statistics[selected],
                    "depth_ready": depth_ready,
                    "selected_confidence": None if selected is None else statistics[selected]["sam3_confidence"],
                    "safety": "read-only camera/TF/SAM3; no robot or gripper command",
                }
                ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise RuntimeError("无法编码浏览器叠加图")
                sequence += 1
                status["sequence"] = sequence
                shared.update(encoded.tobytes(), status)
            except Exception as exc:
                shared.update(None, {"state": "error", "sequence": sequence, "error": str(exc), "safety": "no robot or gripper command"})
                print(f"实时检测帧失败：{exc}", file=sys.stderr)
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("收到 Ctrl-C，正在停止只读实时检测。")
        return 0
    except Exception as exc:
        print(f"实时检测启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        server.shutdown()
        server.server_close()
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(run())
