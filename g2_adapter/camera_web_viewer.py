#!/usr/bin/env python3
"""Read-only local web viewer for the G2 head and hand RGB-D streams."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional
from urllib.parse import urlparse


STREAM_TYPES = {
    "head_color": "kHeadColor",
    "head_depth": "kHeadDepth",
    "left_color": "kHandLeftColor",
    "left_depth": "kHandLeftDepth",
    "right_color": "kHandRightColor",
    "right_depth": "kHandRightDepth",
}
COLOR_STREAMS = {"head_color", "left_color", "right_color"}
DEPTH_STREAMS = set(STREAM_TYPES) - COLOR_STREAMS


@dataclass(frozen=True)
class Frame:
    data: bytes
    width: int
    height: int
    timestamp_ns: int
    encoding: str
    color_format: str
    bit_depth: int
    sequence: int


class FrameStore:
    def __init__(self, stream_names: tuple[str, ...]) -> None:
        self._frames: dict[str, Frame] = {}
        self._errors: dict[str, str] = {}
        self._stream_names = stream_names
        self._lock = threading.Lock()

    def put(self, stream_name: str, image, sequence: int) -> None:
        frame = Frame(
            data=bytes(image.data),
            width=int(image.width),
            height=int(image.height),
            timestamp_ns=int(image.timestamp_ns),
            encoding=str(image.encoding),
            color_format=str(image.color_format),
            bit_depth=int(image.bit_depth),
            sequence=sequence,
        )
        if not frame.data or frame.width <= 0 or frame.height <= 0:
            raise ValueError("GDK returned an empty or invalid image")
        with self._lock:
            self._frames[stream_name] = frame
            self._errors.pop(stream_name, None)

    def set_error(self, stream_name: str, error: Exception) -> None:
        with self._lock:
            self._errors[stream_name] = str(error)

    def frame(self, stream_name: str) -> Optional[Frame]:
        with self._lock:
            return self._frames.get(stream_name)

    def status(self) -> dict:
        with self._lock:
            streams = {}
            for name in self._stream_names:
                frame = self._frames.get(name)
                streams[name] = (
                    {
                        "available": True,
                        "width": frame.width,
                        "height": frame.height,
                        "timestamp_ns": frame.timestamp_ns,
                        "encoding": frame.encoding,
                        "color_format": frame.color_format,
                        "bit_depth": frame.bit_depth,
                        "sequence": frame.sequence,
                    }
                    if frame is not None
                    else {"available": False, "error": self._errors.get(name, "waiting for frame")}
                )
            return {"streams": streams}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在本机浏览器中只读查看 G2 六路 RGB-D。")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址；默认仅本机可访问。")
    parser.add_argument("--port", type=int, default=8765, help="监听端口，默认 8765。")
    parser.add_argument(
        "--browser-fps", type=float, default=4.0,
        help="浏览器刷新率上限，默认 4 FPS；更高会增加 CPU 与网络负载。",
    )
    parser.add_argument(
        "--capture-fps", type=float, default=15.0,
        help="GDK 最新帧采样频率上限，默认 15 FPS。",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not (0 < args.port < 65536):
        raise ValueError("--port 必须在 1 到 65535 之间")
    if not (0 < args.browser_fps <= 30):
        raise ValueError("--browser-fps 必须在 0 到 30 之间")
    if not (0 < args.capture_fps <= 30):
        raise ValueError("--capture-fps 必须在 0 到 30 之间")


def capture_loop(gdk, camera, store: FrameStore, stop: threading.Event, capture_fps: float) -> None:
    types = {
        name: getattr(gdk.CameraType, type_name)
        for name, type_name in STREAM_TYPES.items()
    }
    sequence = 0
    interval = 1.0 / capture_fps
    while not stop.is_set():
        started = time.monotonic()
        sequence += 1
        for name, camera_type in types.items():
            if stop.is_set():
                break
            try:
                image = camera.get_latest_image(camera_type, 100.0)
                store.put(name, image, sequence)
            except Exception as exc:
                store.set_error(name, exc)
        stop.wait(max(0.0, interval - (time.monotonic() - started)))


PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>G2 RGB-D Viewer</title><style>
body { background:#111827; color:#e5e7eb; font:14px system-ui,sans-serif; margin:0; }
header { padding:14px 18px; position:sticky; top:0; background:#111827; border-bottom:1px solid #374151; z-index:2; }
h1 { margin:0; font-size:18px; } #notice { color:#93c5fd; margin-top:5px; }
#grid { display:grid; grid-template-columns:repeat(3,minmax(250px,1fr)); gap:12px; padding:12px; }
.card { background:#1f2937; border:1px solid #374151; border-radius:8px; overflow:hidden; }
.title { display:flex; justify-content:space-between; padding:8px 10px; font-weight:600; }
.meta { font-size:12px; color:#9ca3af; font-weight:400; } img,canvas { width:100%; display:block; background:#030712; aspect-ratio:4/3; object-fit:contain; }
.error { color:#fca5a5; padding:14px; min-height:36px; } @media(max-width:900px){#grid{grid-template-columns:repeat(2,minmax(220px,1fr));}} @media(max-width:560px){#grid{grid-template-columns:1fr;}}
</style></head><body><header><h1>G2 六路 RGB-D（只读）</h1><div id="notice">正在连接……</div></header><main id="grid"></main>
<script>
const order = ['head_color','head_depth','left_color','left_depth','right_color','right_depth'];
const labels = {head_color:'胸口彩色',head_depth:'胸口深度',left_color:'左手彩色',left_depth:'左手深度',right_color:'右手彩色',right_depth:'右手深度'};
const depth = new Set(['head_depth','left_depth','right_depth']);
const seen = {}; const depthFrames = {}; const depthPointers = {}; const grid = document.getElementById('grid');
for (const name of order) { const card=document.createElement('section'); card.className='card'; card.innerHTML=`<div class="title">${labels[name]}<span class="meta" id="m-${name}"></span></div><div id="v-${name}" class="error">等待图像……</div>`; grid.append(card); }
function updateDepthReadout(name, x, y) { const frame=depthFrames[name]; if(!frame)return; x=Math.max(0,Math.min(frame.width-1,Math.floor(x))); y=Math.max(0,Math.min(frame.height-1,Math.floor(y))); const z=frame.values[y*frame.width+x]; document.getElementById(`m-${name}`).textContent=`${frame.width}×${frame.height} · ${z ? (z/1000).toFixed(3)+' m' : '无深度'} @ (${x},${y})`; }
function depthPixelFromPointer(name, canvas, event) { const frame=depthFrames[name]; if(!frame)return null; const rect=canvas.getBoundingClientRect(); const scale=Math.min(rect.width/frame.width,rect.height/frame.height); const drawWidth=frame.width*scale, drawHeight=frame.height*scale; const offsetX=(rect.width-drawWidth)/2, offsetY=(rect.height-drawHeight)/2; const x=(event.clientX-rect.left-offsetX)/scale, y=(event.clientY-rect.top-offsetY)/scale; if(x<0 || y<0 || x>=frame.width || y>=frame.height)return null; return {x,y}; }
function renderDepth(name, info) { const url=`/frame/${name}?s=${info.sequence}`; fetch(url).then(r=>{if(!r.ok)throw new Error('读取深度帧失败'); return r.arrayBuffer();}).then(buffer=>{
  const values=new Uint16Array(buffer); const pixels=info.width*info.height; if(values.length<pixels)throw new Error('深度帧大小异常');
  let canvas=document.getElementById(`i-${name}`); if(!canvas){canvas=document.createElement('canvas');canvas.id=`i-${name}`;canvas.addEventListener('mousemove', event => {const pointer=depthPixelFromPointer(name,canvas,event);if(!pointer)return;depthPointers[name]=pointer;updateDepthReadout(name,pointer.x,pointer.y);});document.getElementById(`v-${name}`).replaceWith(canvas);}
  canvas.width=info.width; canvas.height=info.height; const ctx=canvas.getContext('2d'); const out=ctx.createImageData(info.width,info.height); const near=200, far=2000;
  depthFrames[name]={values,width:info.width,height:info.height};
  for(let i=0;i<pixels;i++){const z=values[i]; const p=i*4; if(z===0){out.data[p+3]=255;continue;} const t=Math.max(0,Math.min(1,(z-near)/(far-near))); const hue=(1-t)*240; const x=1-Math.abs((hue/60)%2-1); let r=0,g=0,b=0; if(hue<60){r=1;g=x;}else if(hue<120){r=x;g=1;}else if(hue<180){g=1;b=x;}else{g=x;b=1;} out.data[p]=r*255;out.data[p+1]=g*255;out.data[p+2]=b*255;out.data[p+3]=255;}
  ctx.putImageData(out,0,0);
  const centerX=Math.floor(info.width/2), centerY=Math.floor(info.height/2);
  ctx.strokeStyle='#ffffff'; ctx.lineWidth=Math.max(2,Math.round(info.width/640)); ctx.beginPath(); ctx.moveTo(centerX-12,centerY); ctx.lineTo(centerX+12,centerY); ctx.moveTo(centerX,centerY-12); ctx.lineTo(centerX,centerY+12); ctx.stroke();
  const pointer=depthPointers[name];
  updateDepthReadout(name,pointer ? pointer.x : centerX,pointer ? pointer.y : centerY);
}).catch(err=>showError(name,err.message)); }
function renderColor(name, info) { if(!/JPEG|PNG/.test(info.encoding)){showError(name,`不支持的彩色编码：${info.encoding}`);return;} let image=document.getElementById(`i-${name}`); if(!image){image=document.createElement('img');image.id=`i-${name}`;document.getElementById(`v-${name}`).replaceWith(image);} image.src=`/frame/${name}?s=${info.sequence}`; }
function showError(name, message){const old=document.getElementById(`i-${name}`); if(old){const d=document.createElement('div');d.id=`v-${name}`;d.className='error';old.replaceWith(d);} const holder=document.getElementById(`v-${name}`);if(holder)holder.textContent=message;}
async function refresh(){try{const state=await (await fetch('/status',{cache:'no-store'})).json(); document.getElementById('notice').textContent='深度伪彩范围：0.2–2.0 m；鼠标悬停深度图可读毫米深度。'; for(const name of order){const info=state.streams[name];const meta=document.getElementById(`m-${name}`);if(!info.available){meta.textContent='无帧';showError(name,info.error);continue;}meta.textContent=`${info.width}×${info.height} · ${info.bit_depth} bit`;if(seen[name]===info.sequence)continue;seen[name]=info.sequence;depth.has(name)?renderDepth(name,info):renderColor(name,info);}}catch(err){document.getElementById('notice').textContent=`连接失败：${err.message}`;}}
setInterval(refresh, 1000 / __BROWSER_FPS__); refresh();
</script></body></html>"""


def make_handler(store: FrameStore, browser_fps: float):
    class ViewerHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args) -> None:
            return

        def send_json(self, document: dict) -> None:
            body = json.dumps(document, ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path == "/":
                body = PAGE.replace("__BROWSER_FPS__", str(browser_fps)).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if path == "/status":
                self.send_json(store.status())
                return
            if path.startswith("/frame/"):
                name = path.removeprefix("/frame/")
                if name not in STREAM_TYPES:
                    self.send_error(HTTPStatus.NOT_FOUND, "unknown stream")
                    return
                frame = store.frame(name)
                if frame is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "frame unavailable")
                    return
                content_type = "application/octet-stream" if name in DEPTH_STREAMS else "image/jpeg"
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(frame.data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(frame.data)
                return
            self.send_error(HTTPStatus.NOT_FOUND, "not found")

    return ViewerHandler


def main() -> int:
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_python.sh 运行。", file=sys.stderr)
        return 2
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败。", file=sys.stderr)
        return 2
    camera = None
    stop = threading.Event()
    try:
        camera_types = [getattr(agibot_gdk.CameraType, item) for item in STREAM_TYPES.values()]
        camera = agibot_gdk.Camera(camera_types)
        store = FrameStore(tuple(STREAM_TYPES))
        worker = threading.Thread(
            target=capture_loop,
            args=(agibot_gdk, camera, store, stop, args.capture_fps),
            daemon=True,
        )
        worker.start()
        server = ThreadingHTTPServer((args.host, args.port), make_handler(store, args.browser_fps))
        server.daemon_threads = True
        print(f"浏览器打开：http://{args.host}:{args.port}")
        print("只读查看器已启动；按 Ctrl-C 停止。未保存图像，未发送任何机器人控制命令。")
        try:
            server.serve_forever(poll_interval=0.2)
        except KeyboardInterrupt:
            print("正在关闭查看器。")
        finally:
            server.shutdown()
            server.server_close()
        return 0
    except OSError as exc:
        print(f"无法启动网页查看器：{exc}", file=sys.stderr)
        return 1
    finally:
        stop.set()
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        try:
            agibot_gdk.gdk_release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
