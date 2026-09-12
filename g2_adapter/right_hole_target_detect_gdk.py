#!/usr/bin/env python3
"""Post-P4 right-hand camera detection for the hole/target model.

This is intentionally a detection-only stage.  It assumes the left-hand
grasp/transport flow has already finished, optionally moves the right arm to
the recorded ``right_post_grasp`` pose, and then reads only the right-hand
color stream.  No gripper, insertion, visual-servo, or force command is sent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2

try:
    from .control import G2Controller
    from .hole_target_detector import (
        best_per_class,
        center_crop,
        infer_detection,
        load_detection_model,
        pair_status,
        select_locked_hole,
        translate_detection,
    )
    from .record_pose import DEFAULT_OUTPUT
    from .camera import decode_color_rgb
except ImportError:
    from control import G2Controller
    from hole_target_detector import (
        best_per_class,
        center_crop,
        infer_detection,
        load_detection_model,
        pair_status,
        select_locked_hole,
        translate_detection,
    )
    from record_pose import DEFAULT_OUTPUT
    from camera import decode_color_rgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hole_target_det_v2" / "best.pt"
RIGHT_ARM_OFFSET = 7


class SharedView:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.overlay_jpeg: bytes | None = None
        self.status: dict = {"state": "starting", "sequence": 0}

    def update(self, image: bytes | None, status: dict) -> None:
        with self.lock:
            self.overlay_jpeg = image
            self.status = dict(status)

    def snapshot(self) -> tuple[bytes | None, dict]:
        with self.lock:
            return self.overlay_jpeg, dict(self.status)


def handler_class(shared: SharedView):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            return

        def _send(self, payload: bytes, content_type: str) -> None:
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
                self._send(json.dumps(status, ensure_ascii=False).encode(), "application/json; charset=utf-8")
                return
            if path == "/overlay.jpg":
                if image is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "waiting for first frame")
                else:
                    self._send(image, "image/jpeg")
                return
            if path not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            page = """<!doctype html><meta charset='utf-8'><title>G2 右手对孔检测</title>
<style>body{margin:0;background:#161616;color:#eee;font:15px sans-serif}main{max-width:1280px;margin:auto;padding:14px}img{width:100%;background:#222}pre{white-space:pre-wrap;background:#242424;padding:10px;border-radius:6px}</style>
<main><h2>G2 右手相机 / hole-target YOLO（只读检测）</h2><img id='overlay' alt='等待相机帧'><pre id='status'>starting</pre></main>
<script>let s=0;async function tick(){try{let r=await fetch('/status.json',{cache:'no-store'});let j=await r.json();document.querySelector('#status').textContent=JSON.stringify(j,null,2);if(j.sequence!==s){s=j.sequence;document.querySelector('#overlay').src='/overlay.jpg?seq='+s}}catch(e){document.querySelector('#status').textContent=String(e)}}setInterval(tick,350);tick();</script>"""
            self._send(page.encode(), "text/html; charset=utf-8")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P4 后右手相机 hole/target YOLO 只读检测")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="hole/target YOLO 权重 best.pt")
    parser.add_argument("--poses", type=Path, default=DEFAULT_OUTPUT, help="包含 right_post_grasp 的 recorded_poses.json")
    parser.add_argument("--right-preset", default="right_post_grasp", help="右臂预移动姿态名，默认 right_post_grasp")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--right-crop-percent", type=float, default=60.0, help="右手图像中心裁剪比例，默认 60%%")
    parser.add_argument("--right-hole-track-max-jump-px", type=float, default=120.0, help="右手孔锁定跨帧最大跳变")
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--velocity-rad-s", type=float, default=0.08)
    parser.add_argument("--arrival-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--premove-timeout-s", type=float, default=25.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示持续检测")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8767, help="0 表示关闭网页")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--allow-motion", action="store_true", help="允许右臂预移动；不指定则只读检测")
    parser.add_argument("--skip-premove", action="store_true", help="测试用：不执行右臂预移动")
    return parser.parse_args()


def load_right_preset(path: Path, name: str) -> list[float]:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    item = document.get("poses", {}).get(name)
    if not isinstance(item, dict) or item.get("side") != "right":
        raise RuntimeError(f"姿态文件缺少右臂 {name}；请先记录：record_pose.py --name {name} --save")
    values = [float(value) for value in item.get("joint_positions_rad", ())]
    if len(values) != 7:
        raise RuntimeError(f"{name} 必须包含 7 个右臂关节值")
    return values


def move_right_preset(controller: G2Controller, target: list[float], args: argparse.Namespace) -> None:
    controller.move_preset("right", target, velocity_rad_s=args.velocity_rad_s, life_time_s=args.premove_timeout_s, detail="post-P4 right hole-target premove")
    deadline = time.monotonic() + args.premove_timeout_s
    while time.monotonic() < deadline:
        measured = controller.checked_arm_state()[RIGHT_ARM_OFFSET : RIGHT_ARM_OFFSET + 7]
        if max(abs(actual - expected) for actual, expected in zip(measured, target)) <= args.arrival_tolerance_rad:
            return
        time.sleep(0.10)
    raise RuntimeError(f"右臂预移动 {args.premove_timeout_s:.1f} s 内未到位；未开始检测")


def draw_overlay(frame_bgr, detections, selected, status: str):
    image = frame_bgr.copy()
    colors = {"hole": (0, 255, 0), "target": (0, 128, 255)}
    for item in detections:
        x0, y0, x1, y1 = [int(value) for value in item.xyxy]
        color = colors.get(item.class_name, (255, 0, 0))
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cx, cy = item.center_px
        cv2.circle(image, (int(cx), int(cy)), 4, color, -1)
        cv2.putText(image, f"{item.class_name} {item.confidence:.2f}", (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(image, f"pair={status}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return image


def main() -> int:
    args = parse_args()
    if not (
        0.0 <= args.confidence <= 1.0
        and 0.0 <= args.iou <= 1.0
        and args.imgsz > 0
        and args.max_frames >= 0
        and 0 <= args.web_port <= 65535
        and math.isfinite(args.velocity_rad_s)
        and args.velocity_rad_s > 0.0
        and 0.001 <= args.arrival_tolerance_rad <= 0.10
        and 1.0 <= args.premove_timeout_s <= 30.0
        and 0.0 < args.right_crop_percent <= 100.0
        and 1.0 <= args.right_hole_track_max_jump_px <= 1000.0
    ):
        print("参数错误：检测、预移动或网页参数无效。", file=sys.stderr)
        return 2
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"找不到 hole/target 权重：{model_path}", file=sys.stderr)
        return 2
    target = None
    if not args.skip_premove:
        try:
            target = load_right_preset(args.poses, args.right_preset)
        except Exception as exc:
            print(f"右臂预移动姿态无效：{exc}", file=sys.stderr)
            return 2
    log_path = (args.log or PROJECT_ROOT / "logs" / f"right_hole_target_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl").expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"mode": "motion" if args.allow_motion else "read_only_preview", "stage": "post_p4_detection", "camera": "right_hand_color", "model": str(model_path), "right_preset": None if args.skip_premove else args.right_preset, "right_crop_percent": args.right_crop_percent, "right_hole_track_max_jump_px": args.right_hole_track_max_jump_px, "log": str(log_path)}, ensure_ascii=False))

    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_python.sh 运行。", file=sys.stderr)
        return 2
    initialized = False
    server = None
    shared = None
    camera = None
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        controller.checked_arm_state()
        if not args.skip_premove:
            if not args.allow_motion:
                print("只读预检：已读取 right_post_grasp，但未发送右臂运动；检测从当前姿态开始。")
            else:
                move_right_preset(controller, target, args)
                print(f"右臂已到位：{args.right_preset}；开始检测。")
        model = load_detection_model(str(model_path), device=args.device)
        color_type = agibot_gdk.CameraType.kHandRightColor
        camera = agibot_gdk.Camera([color_type])
        if args.web_port:
            shared = SharedView()
            server = ThreadingHTTPServer((args.web_host, args.web_port), handler_class(shared))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"实时页面：http://{args.web_host}:{args.web_port}")
        sequence = 0
        last_print = 0.0
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({"event": "start", "utc": datetime.now(timezone.utc).isoformat(), "stage": "post_p4_detection", "camera": "right_hand_color", "model": str(model_path), "premove": None if args.skip_premove else args.right_preset, "right_crop_percent": args.right_crop_percent, "right_hole_track_max_jump_px": args.right_hole_track_max_jump_px}, ensure_ascii=False) + "\n")
            log.flush()
            previous_right_hole_px: tuple[float, float] | None = None
            while args.max_frames == 0 or sequence < args.max_frames:
                started = time.monotonic()
                image = camera.get_latest_image(color_type, 1000.0)
                color_rgb = decode_color_rgb(image, agibot_gdk)
                frame_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                infer_frame_bgr, crop_offset = center_crop(frame_bgr, args.right_crop_percent)
                crop_detections = infer_detection(model, infer_frame_bgr, confidence=args.confidence, iou=args.iou, imgsz=args.imgsz, device=args.device)
                detections = [translate_detection(item, crop_offset) for item in crop_detections]
                all_holes = [item for item in detections if item.class_name == "hole"]
                selected_hole, lock_reason = select_locked_hole(
                    all_holes,
                    previous_right_hole_px,
                    max_jump_px=args.right_hole_track_max_jump_px,
                )
                if selected_hole is not None:
                    previous_right_hole_px = selected_hole.center_px
                selected = best_per_class(detections)
                if selected_hole is None:
                    selected.pop("hole", None)
                else:
                    selected["hole"] = selected_hole
                status_detections = ([selected_hole] if selected_hole is not None else []) + [
                    item for item in detections if item.class_name == "target"
                ]
                status = pair_status(status_detections, selected)
                sequence += 1
                record = {
                    "event": "frame",
                    "frame": sequence,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "state": status,
                    "inference_ms": round((time.monotonic() - started) * 1000.0, 1),
                    "detections": [{"class": item.class_name, "confidence": round(item.confidence, 4), "center_px": [round(value, 1) for value in item.center_px], "xyxy": [round(value, 1) for value in item.xyxy]} for item in detections],
                    "right_hole_lock": {"reason": lock_reason, "candidate_count": len(all_holes), "locked": selected_hole is not None, "center_px": None if selected_hole is None else [round(value, 1) for value in selected_hole.center_px]},
                }
                if set(selected) == {"hole", "target"}:
                    record["center_error_target_minus_hole_px"] = [round(selected["target"].center_px[i] - selected["hole"].center_px[i], 1) for i in range(2)]
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                overlay = draw_overlay(frame_bgr, detections, selected, status)
                crop_x0, crop_y0 = crop_offset
                crop_x1 = crop_x0 + int(infer_frame_bgr.shape[1]) - 1
                crop_y1 = crop_y0 + int(infer_frame_bgr.shape[0]) - 1
                cv2.rectangle(overlay, (crop_x0, crop_y0), (crop_x1, crop_y1), (0, 165, 255), 3, cv2.LINE_AA)
                cv2.putText(
                    overlay,
                    f"INFERENCE ROI {args.right_crop_percent:.0f}%",
                    (crop_x0 + 8, max(24, crop_y0 + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60,
                    (0, 165, 255),
                    2,
                    cv2.LINE_AA,
                )
                if selected_hole is not None:
                    lock_x = int(round(selected_hole.center_px[0]))
                    lock_y = int(round(selected_hole.center_px[1]))
                    cv2.circle(overlay, (lock_x, lock_y), 10, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(overlay, "LOCKED HOLE", (max(8, lock_x - 55), max(24, lock_y - 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
                ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if shared is not None and ok:
                    shared.update(encoded.tobytes(), {"state": status, "sequence": sequence, "camera": "right_hand_color", "detections": record["detections"], "right_hole_lock": record["right_hole_lock"], "center_error_target_minus_hole_px": record.get("center_error_target_minus_hole_px"), "inference_ms": record["inference_ms"], "motion": "none; detection-only"})
                now = time.monotonic()
                if now - last_print >= 1.0:
                    print(f"frame={sequence} pair={status} detections={len(detections)} inference_ms={record['inference_ms']:.1f}", flush=True)
                    last_print = now
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"右手 hole/target 检测失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
