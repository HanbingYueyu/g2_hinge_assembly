#!/usr/bin/env python3
"""Live right-camera capture and browser box annotation for hole/target."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2

try:
    from .camera import decode_color_rgb
    from .hole_target_detector import infer_detection, load_detection_model
except ImportError:
    from camera import decode_color_rgb
    from hole_target_detector import infer_detection, load_detection_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hole_target_det_v2" / "best.pt"


PAGE = r"""<!doctype html><meta charset='utf-8'><title>右手 hole/target 实时标注</title>
<style>body{margin:0;background:#171717;color:#eee;font:15px sans-serif}main{max-width:1300px;margin:auto;padding:12px}#bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;background:#252525;padding:10px;border-radius:6px}button{padding:7px 11px;background:#444;color:#fff;border:1px solid #666;border-radius:4px;cursor:pointer}button.active{background:#1976d2}.danger{background:#7b2222}#view{display:block;max-width:100%;margin-top:10px;background:#222}.hint{color:#bbb}pre{white-space:pre-wrap;background:#252525;padding:10px;border-radius:6px}</style>
<main><div id='bar'><button onclick='freeze()'>冻结当前帧</button><button id='hole' class='active' onclick='cls=0'>hole (H)</button><button id='target' onclick='cls=1'>target (T)</button><button onclick='clearBoxes()' class='danger'>清空框</button><button onclick='save()'>保存标注</button><button onclick='discard()'>丢弃并回实时</button><span id='state'></span></div><p class='hint'>实时画面自动刷新。冻结后可拖动新增框；点击框后按 Delete 删除；H/T 切换类别。空画面清空后保存为负样本。</p><canvas id='canvas'></canvas><pre id='info'></pre></main>
<script>
let frozen=false,cls=0,boxes=[],selected=-1,drag=null,w=1,h=1,seq=0;const c=document.querySelector('#canvas'),x=c.getContext('2d'),img=new Image();
function setClass(v){cls=v;document.querySelector('#hole').classList.toggle('active',v===0);document.querySelector('#target').classList.toggle('active',v===1)}
document.querySelector('#hole').onclick=()=>setClass(0);document.querySelector('#target').onclick=()=>setClass(1);
function pos(e){let r=c.getBoundingClientRect();return{x:(e.clientX-r.left)*w/c.width,y:(e.clientY-r.top)*h/c.height}}
function draw(){x.clearRect(0,0,c.width,c.height);if(img.complete)x.drawImage(img,0,0,c.width,c.height);let sx=c.width/w,sy=c.height/h;boxes.forEach((b,i)=>{let x0=Math.min(b.x0,b.x1)*sx,y0=Math.min(b.y0,b.y1)*sy,bw=Math.abs(b.x1-b.x0)*sx,bh=Math.abs(b.y1-b.y0)*sy;x.strokeStyle=b.class_id===0?'#00ff66':'#ff9d00';x.lineWidth=i===selected?4:2;x.strokeRect(x0,y0,bw,bh);x.fillStyle=x.strokeStyle;x.font='16px sans-serif';x.fillText(b.class_id===0?'hole':'target',x0+3,Math.max(16,y0-4))})}
function resize(){let s=Math.min(1,1200/w);c.width=Math.round(w*s);c.height=Math.round(h*s);draw()}
function refresh(){if(frozen)return;fetch('/api/live').then(r=>r.json()).then(s=>{seq=s.sequence;w=s.width;h=s.height;img.onload=resize;img.src='/live.jpg?seq='+seq;document.querySelector('#state').textContent='实时采集中 '+seq})}
function freeze(){fetch('/api/freeze',{method:'POST'}).then(r=>r.json()).then(s=>{if(!s.ok){alert(s.error);return}frozen=true;w=s.width;h=s.height;boxes=s.boxes||[];img.onload=resize;img.src='/frozen.jpg?v='+Date.now();document.querySelector('#state').textContent='已冻结：请检查预标并修正';document.querySelector('#info').textContent=JSON.stringify(s,null,2)})}
function clearBoxes(){boxes=[];selected=-1;draw()}
function save(){if(!frozen){alert('请先冻结一帧');return}fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({boxes})}).then(r=>r.json()).then(s=>{if(!s.ok){alert(s.error);return}document.querySelector('#state').textContent='已保存 '+s.file;frozen=false;boxes=[];selected=-1;setTimeout(refresh,250)})}
function discard(){frozen=false;boxes=[];selected=-1;refresh()}
c.addEventListener('mousedown',e=>{if(!frozen)return;let p=pos(e);selected=-1;for(let i=boxes.length-1;i>=0;i--){let b=boxes[i];if(p.x>=Math.min(b.x0,b.x1)&&p.x<=Math.max(b.x0,b.x1)&&p.y>=Math.min(b.y0,b.y1)&&p.y<=Math.max(b.y0,b.y1)){selected=i;break}}if(selected<0)drag={x0:p.x,y0:p.y,x1:p.x,y1:p.y,class_id:cls};draw()});
c.addEventListener('mousemove',e=>{if(!drag)return;let p=pos(e);drag.x1=p.x;drag.y1=p.y;draw();let sx=c.width/w,sy=c.height/h;x.strokeStyle=cls===0?'#00ff66':'#ff9d00';x.strokeRect(Math.min(drag.x0,drag.x1)*sx,Math.min(drag.y0,drag.y1)*sy,Math.abs(drag.x1-drag.x0)*sx,Math.abs(drag.y1-drag.y0)*sy)});c.addEventListener('mouseup',()=>{if(!drag)return;if(Math.abs(drag.x1-drag.x0)>=4&&Math.abs(drag.y1-drag.y0)>=4)boxes.push(drag);drag=null;draw()});
document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='h')setClass(0);if(e.key.toLowerCase()==='t')setClass(1);if(e.key==='Delete'&&selected>=0){boxes.splice(selected,1);selected=-1;draw()}if(e.key===' '){e.preventDefault();if(frozen)save();else freeze()}});setInterval(refresh,250);refresh();
</script>"""


class CaptureState:
    def __init__(self, out_dir: Path, model, gdk, camera, color_type, confidence: float, device: str) -> None:
        self.out_dir, self.model, self.gdk, self.camera, self.color_type = out_dir, model, gdk, camera, color_type
        self.confidence, self.device = confidence, device
        self.lock = threading.Lock()
        self.latest_bgr = None
        self.latest_sequence = 0
        self.frozen_bgr = None
        self.frozen_detections = []
        self.stop = threading.Event()
        self.worker: threading.Thread | None = None
        self.saved_count = 0

    def start(self) -> None:
        self.worker = threading.Thread(target=self._run, name="right-hole-target-camera", daemon=True)
        self.worker.start()

    def close(self) -> None:
        self.stop.set()
        if self.worker is not None:
            self.worker.join(timeout=2.0)

    def _run(self) -> None:
        while not self.stop.is_set():
            try:
                image = self.camera.get_latest_image(self.color_type, 1000.0)
                rgb = decode_color_rgb(image, self.gdk)
                with self.lock:
                    self.latest_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    self.latest_sequence += 1
            except Exception as exc:
                print(f"读取右手相机失败：{exc}", file=sys.stderr)
                self.stop.wait(0.2)

    def snapshot(self) -> tuple[Any, int]:
        with self.lock:
            if self.latest_bgr is None:
                raise RuntimeError("尚未收到右手相机帧")
            return self.latest_bgr.copy(), self.latest_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="右手相机实时采集并快速标注 hole/target")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/hole_target_box_raw_live"))
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.confidence <= 1.0 and 0 <= args.port <= 65535):
        print("参数错误：confidence 或端口无效。", file=sys.stderr)
        return 2
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"找不到模型：{model_path}", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2
    initialized = False
    state = None
    try:
        model = load_detection_model(str(model_path), device=args.device)
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        color_type = agibot_gdk.CameraType.kHandRightColor
        camera = agibot_gdk.Camera([color_type])
        out_dir = args.out_dir.expanduser().resolve()
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)
        (out_dir / "classes.txt").write_text("hole\ntarget\n", encoding="utf-8")
        state = CaptureState(out_dir, model, agibot_gdk, camera, color_type, args.confidence, args.device)
        state.start()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *_args) -> None:
                return

            def send_json(self, payload: dict, status=HTTPStatus.OK) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def send_jpeg(self, image) -> None:
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    self.send_error(500)
                    return
                data = encoded.tobytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path in {"/", "/index.html"}:
                    data = PAGE.encode()
                    self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
                if path == "/api/live":
                    try:
                        image, sequence = state.snapshot(); h, w = image.shape[:2]
                        self.send_json({"sequence": sequence, "width": w, "height": h})
                    except Exception as exc:
                        self.send_json({"error": str(exc)}, HTTPStatus.SERVICE_UNAVAILABLE)
                    return
                if path == "/live.jpg":
                    try: self.send_jpeg(state.snapshot()[0])
                    except Exception as exc: self.send_error(503, str(exc))
                    return
                if path == "/frozen.jpg":
                    with state.lock: image = None if state.frozen_bgr is None else state.frozen_bgr.copy()
                    if image is None: self.send_error(404)
                    else: self.send_jpeg(image)
                    return
                self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                path = self.path.split("?", 1)[0]
                if path == "/api/freeze":
                    try:
                        image, sequence = state.snapshot()
                        detections = infer_detection(state.model, image, confidence=state.confidence, iou=0.45, imgsz=640, device=state.device)
                        with state.lock:
                            state.frozen_bgr, state.frozen_detections = image, detections
                        h, w = image.shape[:2]
                        boxes = [{"class_id": 0 if item.class_name == "hole" else 1, "x0": item.xyxy[0], "y0": item.xyxy[1], "x1": item.xyxy[2], "y1": item.xyxy[3], "confidence": item.confidence} for item in detections if item.class_name in ("hole", "target")]
                        self.send_json({"ok": True, "sequence": sequence, "width": w, "height": h, "boxes": boxes})
                    except Exception as exc:
                        self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                if path == "/api/save":
                    try:
                        length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length)); boxes = payload.get("boxes", [])
                        with state.lock:
                            if state.frozen_bgr is None: raise RuntimeError("没有冻结帧")
                            image = state.frozen_bgr.copy(); number = state.saved_count; state.saved_count += 1
                        stem = f"frame_{number:06d}"; image_path = state.out_dir / "images" / f"{stem}.jpg"; label_path = state.out_dir / "labels" / f"{stem}.txt"
                        if not cv2.imwrite(str(image_path), image): raise RuntimeError("保存图像失败")
                        h, w = image.shape[:2]; lines = []
                        for box in boxes:
                            if int(box.get("class_id", -1)) not in (0, 1): continue
                            xs = sorted((float(box["x0"]), float(box["x1"])))
                            ys = sorted((float(box["y0"]), float(box["y1"])))
                            bw, bh = xs[1] - xs[0], ys[1] - ys[0]
                            if bw >= 4 and bh >= 4: lines.append(f"{int(box['class_id'])} {(xs[0]+xs[1])/2/w:.6f} {(ys[0]+ys[1])/2/h:.6f} {bw/w:.6f} {bh/h:.6f}")
                        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                        self.send_json({"ok": True, "file": stem, "labels": len(lines)})
                    except Exception as exc:
                        self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self.send_error(404)

        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"实时标注页：http://{args.host}:{args.port}")
        print(f"输出目录：{out_dir}")
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"右手实时标注失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if state is not None:
            state.close()
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
