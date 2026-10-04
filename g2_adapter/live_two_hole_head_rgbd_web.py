#!/usr/bin/env python3
"""Read-only live preview of the production head RGB-D two-hole detector."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from .build_head_rgbd_two_hole_dataset import CLASS_NAMES
from .build_head_rgbd_two_hole_v2 import ROI
from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
from .localize_two_holes_head_rgbd_gdk import select_two_boxes
from .two_hole_rgbd import base_from_head_at, localize_box, project_color_pixel_to_base_z


PAGE = r"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>头部 RGB-D 双孔实时识别</title><style>
body{margin:0;background:#171b20;color:#eef2f6;font:15px sans-serif}
main{max-width:1250px;margin:auto;padding:18px}h1{font-size:21px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.card{background:#252c34;padding:12px;border-radius:8px}
img{display:block;width:100%;image-rendering:auto;background:#111}pre{white-space:pre-wrap;line-height:1.5}
.note{color:#adbac7}@media(max-width:800px){.grid{grid-template-columns:1fr}}
</style><main><h1>头部 RGB-D 双孔实时识别</h1>
<p class="note">当前在线模型和固定裁剪框；绿色 0 为台面孔，橙色 1 为铰链端圆孔。只读，不保存图像或控制机器人。</p>
<div class="grid"><div class="card"><h3>检测裁剪区 380×300</h3><img id="crop"></div>
<div class="card"><h3>头部完整彩色图 640×400</h3><img id="full"></div></div>
<div class="card"><h3>本帧定位</h3><pre id="status">等待相机…</pre></div>
<script>
let sequence=-1;
async function refresh(){try{const r=await fetch('/status',{cache:'no-store'});const s=await r.json();
if(s.sequence!==sequence&&s.sequence>0){sequence=s.sequence;document.getElementById('crop').src='/crop.jpg?v='+sequence;document.getElementById('full').src='/full.jpg?v='+sequence;}
let lines=[`帧号 ${s.sequence||0}；RGB/深度时间差 ${s.rgb_depth_delta_ms??'—'} ms`];
if(s.tf)lines.push(`TF: ${s.tf.mode}，时间差 ${s.tf.age_ms} ms`);
for(const name of ['irregular_table_hole','round_hinge_tip_hole']){const p=s.targets?.[name];lines.push(name+': '+(p?`置信度 ${p.confidence.toFixed(3)}，base_link = [${p.point_base_link_m.map(x=>x.toFixed(4)).join(', ')}] m`:'未得到有效 3D 位置'));}
if(s.warning)lines.push('提示：'+s.warning);document.getElementById('status').textContent=lines.join('\n');
}catch(e){document.getElementById('status').textContent='预览连接失败：'+e.message}}
setInterval(refresh,500);refresh();
</script></main></html>"""


class Latest:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.crop_jpeg: bytes | None = None
        self.full_jpeg: bytes | None = None
        self.status: dict = {"sequence": 0, "targets": {}, "warning": "等待头部 RGB-D"}

    def put(self, crop: np.ndarray, full: np.ndarray, status: dict) -> None:
        ok_crop, crop_jpeg = cv2.imencode(".jpg", crop)
        ok_full, full_jpeg = cv2.imencode(".jpg", full)
        if not ok_crop or not ok_full:
            raise RuntimeError("无法编码预览图像")
        with self.lock:
            self.crop_jpeg = crop_jpeg.tobytes()
            self.full_jpeg = full_jpeg.tobytes()
            self.status = status


def make_handler(latest: Latest):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            pass

        def reply(self, data: bytes, mime: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                self.reply(PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            with latest.lock:
                if path == "/status":
                    data = json.dumps(latest.status, ensure_ascii=False).encode("utf-8")
                    mime = "application/json; charset=utf-8"
                elif path == "/crop.jpg" and latest.crop_jpeg is not None:
                    data, mime = latest.crop_jpeg, "image/jpeg"
                elif path == "/full.jpg" and latest.full_jpeg is not None:
                    data, mime = latest.full_jpeg, "image/jpeg"
                else:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
            self.reply(data, mime)

    return Handler


def draw_boxes(full: np.ndarray, crop: np.ndarray, result) -> None:
    x0, y0, _, _ = ROI
    for box in result.boxes:
        class_id = int(box.cls.item())
        if class_id not in (0, 1):
            continue
        left, top, right, bottom = (int(round(float(value))) for value in box.xyxy[0].tolist())
        color = (0, 240, 60) if class_id == 0 else (0, 165, 255)
        label = f"{class_id} {float(box.conf.item()):.2f}"
        cv2.rectangle(crop, (left, top), (right, bottom), color, 2)
        cv2.putText(crop, label, (left, max(15, top - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
        cv2.rectangle(full, (left + x0, top + y0), (right + x0, bottom + y0), color, 2)
        cv2.putText(full, label, (left + x0, max(15, top + y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/two_hole_head_rgbd_yolo_v2/best.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8778)
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument("--interval", type=float, default=0.25)
    args = parser.parse_args()
    if not args.model.is_file() or not 0 < args.confidence < 1 or args.interval < 0:
        parser.error("模型不存在或参数不正确")
    import agibot_gdk as gdk
    from ultralytics import YOLO

    model = YOLO(str(args.model.resolve()))
    if [model.names.get(i) for i in range(2)] != list(CLASS_NAMES):
        raise RuntimeError(f"模型类别不匹配：{model.names}")
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK 初始化失败")
    camera = server = thread = None
    try:
        color_type, depth_type = gdk.CameraType.kHeadColor, gdk.CameraType.kHeadDepth
        camera = gdk.Camera([color_type, depth_type])
        tf = gdk.TF()
        color_from_depth = transform_to_dict(tf.get_tf_from_sensor(gdk.SensorExtrinsicType.kHeadDepthToHeadColor))
        head_from_color = transform_to_dict(tf.get_tf_from_sensor(gdk.SensorExtrinsicType.kHeadRGBDToHeadLink3))
        color_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(color_type), width=640, height=400)
        depth_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(depth_type), width=640, height=400)
        latest = Latest()
        server = ThreadingHTTPServer((args.host, args.port), make_handler(latest))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"头部双孔实时预览：http://127.0.0.1:{args.port}", flush=True)
        last_stamp = None
        sequence = 0
        while True:
            started = time.monotonic()
            color = camera.get_latest_image(color_type, 1000.0)
            stamp = int(color.timestamp_ns)
            if stamp == last_stamp:
                time.sleep(.01)
                continue
            last_stamp = stamp
            depth = camera.get_nearest_image(depth_type, stamp, 1000.0)
            delta_ms = abs(int(depth.timestamp_ns) - stamp) / 1e6
            full = cv2.cvtColor(decode_color_rgb(color, gdk), cv2.COLOR_RGB2BGR)
            depth_u16 = decode_depth_u16(depth, gdk)
            if full.shape[:2] != (400, 640) or depth_u16.shape != (400, 640):
                raise RuntimeError("头部 RGB-D 分辨率不是预期的 640×400")
            x0, y0, x1, y1 = ROI
            crop = full[y0:y1, x0:x1].copy()
            result = model.predict(crop, imgsz=960, conf=args.confidence, iou=0.5,
                                   device=0, verbose=False)[0]
            status = {"sequence": sequence + 1, "rgb_depth_delta_ms": round(delta_ms, 2),
                      "targets": {}, "tf": None, "warning": None}
            if delta_ms <= 50:
                try:
                    boxes = select_two_boxes(result, crop_origin=(x0, y0))
                    aligned = align_depth_to_color(
                        depth_u16, depth_intrinsic=depth_intrinsic,
                        color_intrinsic=color_intrinsic, depth_to_color=color_from_depth,
                        depth_scale_m=0.001,
                    )
                    base_from_head, tf_status = base_from_head_at(tf, stamp, max_fallback_age_ms=100)
                    targets = {}
                    for class_id, box in boxes.items():
                        target = localize_box(box["box_full_xyxy_px"], aligned,
                                              color_intrinsic, head_from_color, base_from_head)
                        target["confidence"] = box["confidence"]
                        targets[CLASS_NAMES[class_id]] = target
                    table_z = targets[CLASS_NAMES[0]]["point_base_link_m"][2]
                    tip = targets[CLASS_NAMES[1]]
                    tip["point_base_link_m"] = list(project_color_pixel_to_base_z(
                        tip["center_head_color_px"], table_z, color_intrinsic,
                        head_from_color, base_from_head,
                    ))
                    status["targets"], status["tf"] = targets, tf_status
                except (RuntimeError, ValueError) as error:
                    status["warning"] = str(error)
            else:
                status["warning"] = "彩色与深度时间差超过 50 ms，本帧不定位"
            draw_boxes(full, crop, result)
            cv2.rectangle(full, (x0, y0), (x1, y1), (255, 255, 255), 1)
            sequence += 1
            latest.put(crop, full, status)
            time.sleep(max(0.0, args.interval - (time.monotonic() - started)))
    except KeyboardInterrupt:
        return 0
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)
        if camera is not None:
            camera.close_camera()
        gdk.gdk_release()


if __name__ == "__main__":
    raise SystemExit(main())
