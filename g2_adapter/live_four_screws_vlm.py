#!/usr/bin/env python3
"""Independent, read-only right-camera screw review with exact VLM input preview."""
from __future__ import annotations

import argparse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import sys
import threading
import time
from urllib.request import urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def parse_crop(value):
    try:
        values = tuple(int(part) for part in value.split(','))
    except ValueError as exc:
        raise ValueError('裁剪格式为 x0,y0,x1,y1') from exc
    if len(values) != 4 or min(values) < 0 or values[2] <= values[0] or values[3] <= values[1]:
        raise ValueError('裁剪坐标必须非负且有正面积')
    return values


def crop_frame(frame, roi):
    x0, y0, x1, y1 = roi
    if frame is None or not (0 <= x0 < x1 <= frame.shape[1] and 0 <= y0 < y1 <= frame.shape[0]):
        raise ValueError('裁剪范围超出相机图像，请调整 --crop')
    ok, data = cv2.imencode('.jpg', frame[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 96])
    if not ok:
        raise RuntimeError('裁剪图编码失败')
    return data.tobytes()


def boxed_screw_frame(frame, model, *, with_count=False):
    """Detect in the trained ROI, then draw only boxes on the full image."""
    from right_plate_screw_detection import detect
    _, _, details = detect(model, frame, confidence=.5)
    annotated = frame.copy()
    for detection in details['detections']:
        x0, y0, x1, y1 = (int(round(v)) for v in detection['bbox_xyxy'])
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 0, 0), 2)
    return (annotated, len(details['detections'])) if with_count else annotated


def check_result(result):
    if not isinstance(result, dict) or result.get('latest_decision') not in {'yes', 'no', 'unknown'}:
        raise ValueError('VLM 未返回有效 latest_decision（yes/no/unknown）')
    return result


PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>四颗螺丝实时判别</title>
<style>body{font-family:sans-serif;margin:24px;background:#f5f6f8}img{max-width:100%;background:#ddd}
.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}pre{white-space:pre-wrap}#live{max-height:420px}</style>
<h2>四颗螺丝实时判别</h2><p>实时裁剪预览；下面四张为最近一次实际提交给 VLM 的三张图：零颗参考图、四颗参考图、当前裁剪图。</p>
<img id="live" src="/crop.jpg"><h3 id="status">等待采集</h3><div class="grid" id="inputs"></div><pre id="detail"></pre>
<script>let current=-1;async function tick(){try{let s=await(await fetch('/status')).json();
document.getElementById('live').src='/crop.jpg?t='+Date.now();
let labels={yes:'四颗螺丝已打好',no:'四颗螺丝尚未全部打好',unknown:'无法确定'};
document.getElementById('status').textContent=(s.running?'正在判别 · ':'')+(s.error||labels[s.result?.latest_decision]||'等待四张图像');
document.getElementById('detail').textContent=JSON.stringify(s,null,2);
if(s.request_id!==current){current=s.request_id;let box=document.getElementById('inputs');box.replaceChildren();
for(let i=0;i<s.input_count;i++){let d=document.createElement('div');let img=document.createElement('img');
img.src='/input/'+i+'?id='+current;d.append(s.input_labels?.[i]||('第 '+(i+1)+' 张'),img);box.append(d)}}
}catch(e){document.getElementById('status').textContent='连接失败：'+e}setTimeout(tick,500)}tick();</script></html>'''


class LiveReview:
    def __init__(self, read_frame, client, roi, *, sample_interval=.25, request_interval=1., annotate_frame=None):
        self.read_frame, self.client, self.roi = read_frame, client, roi
        self.annotate_frame = annotate_frame
        self.sample_interval, self.request_interval = sample_interval, request_interval
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.frames = deque(maxlen=4)
        self.latest = None
        self.inputs = ()
        self.state = {'request_id': 0, 'running': False, 'result': None, 'error': None,
                      'crop_xyxy': list(roi), 'input_count': 0,
                      'input_labels': getattr(client, 'input_labels', [])}
        self.threads = []

    def start(self):
        for task in (self.capture, self.review):
            thread = threading.Thread(target=task, daemon=True)
            thread.start()
            self.threads.append(thread)

    def capture(self):
        last_sample = float('-inf')
        while not self.stop.is_set():
            try:
                frame = self.read_frame()
                yolo_count = None
                if self.annotate_frame is not None:
                    annotated = self.annotate_frame(frame)
                    if isinstance(annotated, tuple):
                        frame, yolo_count = annotated
                    else:
                        frame = annotated
                image = crop_frame(frame, self.roi)
                now = time.monotonic()
                with self.lock:
                    self.latest = image
                    self.state['camera_error'] = None
                    self.state['preview_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                    if now-last_sample >= self.sample_interval:
                        self.frames.append((now, image, yolo_count))
                        last_sample = now
            except Exception as exc:
                with self.lock:
                    self.frames.clear()
                    self.state['camera_error'] = str(exc)
                self.stop.wait(.3)
            self.stop.wait(.03)

    def review(self):
        last_input_time = float('-inf')
        while not self.stop.is_set():
            with self.lock:
                ready = (len(self.frames) == 4 and self.frames[-1][0] > last_input_time
                         and time.monotonic()-self.frames[-1][0] < 1.)
                if ready:
                    frames = tuple(self.frames)
                    live_images = tuple(frame[1] for frame in frames)
                    prepare = getattr(self.client, 'prepare_images', None)
                    self.inputs = tuple(prepare(live_images)) if prepare else live_images
                    self.state.update(request_id=self.state['request_id']+1, running=True,
                                      error=None, result=None, input_count=len(self.inputs),
                                      submitted_time=time.strftime('%Y-%m-%d %H:%M:%S'),
                                      sample_span_s=round(frames[-1][0]-frames[0][0], 3))
            if not ready:
                self.stop.wait(.05)
                continue
            started = time.monotonic()
            try:
                kwargs = {'yolo_count': frames[-1][2]} if frames[-1][2] is not None else {}
                result = check_result(self.client.run_bytes('four-screws', list(self.inputs), **kwargs))
                with self.lock:
                    self.state['result'] = result
            except Exception as exc:
                with self.lock:
                    self.state['error'] = str(exc)
            finally:
                last_input_time = frames[-1][0]
                with self.lock:
                    self.state.update(running=False, elapsed_s=round(time.monotonic()-started, 3),
                                      completed_time=time.strftime('%Y-%m-%d %H:%M:%S'))
            self.stop.wait(self.request_interval)

    def handler(self):
        owner = self
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                path, _, query = self.path.partition('?')
                code, mime = 200, 'image/jpeg'
                with owner.lock:
                    if path == '/':
                        data, mime = PAGE.encode(), 'text/html; charset=utf-8'
                    elif path == '/status':
                        data, mime = json.dumps(owner.state, ensure_ascii=False).encode(), 'application/json'
                    elif path == '/crop.jpg':
                        data = owner.latest
                    elif path.startswith('/input/'):
                        try:
                            index = int(path.removeprefix('/input/'))
                            request_id = int(query.removeprefix('id='))
                            data = owner.inputs[index] if 0 <= index < len(owner.inputs) and request_id == owner.state['request_id'] else None
                        except ValueError:
                            data = None
                    else:
                        data, code = b'Not found', 404
                if data is None:
                    data, code = b'Image unavailable', 404
                self.send_response(code)
                self.send_header('Content-Type', mime)
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)
        return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--crop', default='430,400,870,830', help='原图像素 x0,y0,x1,y1；须包含完整铰链和定位标记')
    parser.add_argument('--source-url', help='读取已有全流程原图，例如 http://127.0.0.1:8767/raw/right；省略则直连右手相机')
    parser.add_argument('--web-host', default='0.0.0.0')
    parser.add_argument('--web-port', type=int, default=8770)
    parser.add_argument('--vlm-env-file', type=Path, default=ROOT/'vlm_checks/config/vlm.env')
    parser.add_argument('--vlm-mode', choices=['local', 'online'], default='local')
    parser.add_argument('--vlm-timeout-s', type=float, default=60.)
    parser.add_argument('--sample-interval-s', type=float, default=.25)
    parser.add_argument('--request-interval-s', type=float, default=1.)
    args = parser.parse_args(argv)
    try:
        roi = parse_crop(args.crop)
    except ValueError as exc:
        parser.error(str(exc))
    for value in (args.vlm_timeout_s, args.sample_interval_s, args.request_interval_s):
        if not math.isfinite(value) or value <= 0:
            parser.error('时间参数必须为正且有限')
    sys.path.insert(0, str(ROOT))
    from four_screws_20261006_client import FourScrews20261006Client
    client = FourScrews20261006Client(args.vlm_env_file, None, None, args.vlm_timeout_s, args.vlm_mode)
    from ultralytics import YOLO
    from right_plate_screw_detection import MODEL_PATH
    screw_model = YOLO(str(ROOT / MODEL_PATH))
    gdk = None
    server = None
    review = None
    camera = None
    try:
        if args.source_url:
            def read_frame():
                with urlopen(args.source_url, timeout=2.) as response:
                    data = response.read()
                frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError('原图地址未返回有效图像')
                return frame
        else:
            import agibot_gdk as gdk
            from camera import decode_color_rgb
            if gdk.gdk_init() != gdk.GDKRes.kSuccess:
                raise RuntimeError('GDK 初始化失败')
            camera = gdk.Camera([gdk.CameraType.kHandRightColor])
            def read_frame():
                frame = camera.get_latest_image(gdk.CameraType.kHandRightColor, 300.)
                return cv2.cvtColor(decode_color_rgb(frame, gdk), cv2.COLOR_RGB2BGR)
        review = LiveReview(read_frame, client, roi, sample_interval=args.sample_interval_s,
                            request_interval=args.request_interval_s,
                            annotate_frame=lambda frame: boxed_screw_frame(frame, screw_model, with_count=True))
        server = ThreadingHTTPServer((args.web_host, args.web_port), review.handler())
        review.start()
        print(f'浏览器预览：http://127.0.0.1:{args.web_port}；Ctrl+C 停止。裁剪区域：{roi}', flush=True)
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        return 0
    finally:
        if review is not None:
            review.stop.set()
            review.threads[0].join(timeout=3.)
        if server is not None:
            server.server_close()
        if gdk is not None and (review is None or not review.threads[0].is_alive()):
            camera = None
            gdk.gdk_release()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
