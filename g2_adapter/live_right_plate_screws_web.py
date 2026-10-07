"""Browser preview of fixed-ROI screw detection from an existing DDS camera service."""
from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import time

import cv2

from .right_plate_screw_detection import CAMERA_URL, MODEL_PATH, ROI, detect, read_camera

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>螺丝实时检测</title><style>
body{background:#171b20;color:#eef2f6;font:16px sans-serif;margin:24px}
main{max-width:750px;margin:auto}img{width:510px;max-width:100%;height:auto}
pre{white-space:pre-wrap}strong{font-size:24px}</style>
<main><h1>螺丝实时检测</h1>
<p>裁剪范围：原图 (525,435) → (780,815)，255×380 像素</p>
<p><strong id="count">等待画面…</strong></p><img id="view" hidden>
<pre id="status"></pre></main><script>
async function refresh(){try{
 const response=await fetch('/frame',{cache:'no-store'});
 if(!response.ok)throw Error('HTTP '+response.status);
 const s=await response.json();
 const valid=s.ok && (Date.now()/1000-s.updated_at)<6;
 document.getElementById('count').textContent=valid?'螺丝：'+s.count+' 颗':'画面暂不可用';
 const view=document.getElementById('view');view.hidden=!valid;
 if(valid && view.dataset.sequence!==String(s.sequence)){
  view.src=s.image;view.dataset.sequence=String(s.sequence);
 }
 document.getElementById('status').textContent=valid?
  '帧号 '+s.sequence+'，检测耗时 '+s.inference_ms+' ms\n'+
  s.detections.map((b,i)=>'螺丝 '+(i+1)+'，置信度 '+b.confidence.toFixed(3)).join('\n'):
  (s.error||'等待新的相机画面');
 }catch(e){document.getElementById('count').textContent='预览连接失败';
 document.getElementById('view').hidden=true;document.getElementById('status').textContent=e.message;
 }finally{setTimeout(refresh,250)}}refresh();</script></html>'''


class Latest:
    def __init__(self):
        self.lock = threading.Lock()
        self.payload = dict(ok=False, error='等待相机画面', sequence=0)

    def put(self, payload):
        with self.lock:
            self.payload = payload

    def get(self):
        with self.lock:
            return dict(self.payload)


def make_handler(latest):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            path = self.path.split('?', 1)[0]
            if path == '/':
                data, mime = PAGE.encode(), 'text/html; charset=utf-8'
            elif path in ('/frame', '/status'):
                payload = latest.get()
                if path == '/status':
                    payload.pop('image', None)
                data, mime = json.dumps(payload, ensure_ascii=False).encode(), 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass
    return Handler


def run_detector(model, args, latest, stop):
    sequence, last_stamp = 0, None
    while not stop.is_set():
        started = time.monotonic()
        try:
            image, metadata = read_camera(args.camera_url)
            stamp = metadata['source_timestamp']
            if stamp is not None and stamp == last_stamp:
                stop.wait(max(.02, args.interval))
                continue
            began = time.monotonic()
            _, preview, data = detect(model, image, args.confidence)
            elapsed = (time.monotonic()-began)*1000
            encoded, jpeg = cv2.imencode('.jpg', preview)
            if not encoded:
                raise RuntimeError('Failed to encode ROI preview')
            sequence += 1
            latest.put(dict(data, ok=True, sequence=sequence, updated_at=time.time(),
                            inference_ms=round(elapsed, 1), camera=metadata,
                            image='data:image/jpeg;base64,'+base64.b64encode(jpeg).decode()))
            last_stamp = stamp
        except Exception as exc:
            latest.put(dict(ok=False, error=str(exc), sequence=sequence,
                            count=None, crop_xyxy=list(ROI)))
        stop.wait(max(.02, args.interval-(time.monotonic()-started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path(MODEL_PATH))
    parser.add_argument('--camera-url', default=CAMERA_URL)
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--interval', type=float, default=.25)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8793)
    args = parser.parse_args()
    if not args.model.is_file() or not 0 < args.confidence < 1 or args.interval < 0:
        parser.error('模型路径、置信度或检测间隔不正确')
    from ultralytics import YOLO
    model = YOLO(str(args.model.resolve()))
    if model.names != {0: 'plate_screw'}:
        parser.error(f'模型类别不正确：{model.names}')
    latest, stop = Latest(), threading.Event()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(latest))
    worker = threading.Thread(target=run_detector, args=(model, args, latest, stop), daemon=True)
    worker.start()
    print(f'螺丝实时预览：http://{args.host}:{args.port}；Ctrl+C 退出', flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        server.server_close()
        worker.join(timeout=6)


if __name__ == '__main__':
    main()
