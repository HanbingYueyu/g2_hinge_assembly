#!/usr/bin/env python3
"""Live left-D405 SAM/YOLO assisted collector for hinge/shaft data.

The camera is read only.  On freeze, SAM3 provides the ``hinge`` polygon and
the existing two-class YOLO-seg model provides only the ``shaft`` polygon.
The operator reviews, deletes, or draws polygons before saving a standard raw
capture under ``images_all``/``labels_all``.
"""

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
import numpy as np

try:
    from .capture_left_d405_hinge_yolo_dataset import SamAssistant
    from .camera import decode_color_rgb
    from .yolo_hinge import infer_yolo_masks, load_yolo_segmentation_model
except ImportError:
    from capture_left_d405_hinge_yolo_dataset import SamAssistant
    from camera import decode_color_rgb
    from yolo_hinge import infer_yolo_masks, load_yolo_segmentation_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "left_d405_hinge_shaft_yolo" / "best.pt"
DEFAULT_SAM_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt"
CLASS_NAMES = {0: "hinge", 1: "shaft"}


PAGE = r"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>左 D405 hinge/shaft YOLO 采集</title>
<style>body{margin:0;background:#171717;color:#eee;font:15px sans-serif}main{max-width:1400px;margin:auto;padding:12px}
#bar{display:flex;gap:6px;align-items:center;flex-wrap:wrap;background:#252525;padding:10px;border-radius:6px}
button{padding:7px 10px;background:#444;color:#fff;border:1px solid #666;border-radius:4px;cursor:pointer}
button.active{background:#1976d2}.danger{background:#7b2222}.hint{color:#bbb}.status{white-space:pre-wrap;background:#252525;padding:8px;margin-top:8px}
canvas{display:block;max-width:100%;max-height:78vh;margin-top:10px;border:1px solid #777;cursor:crosshair;touch-action:none}</style>
<main><div id='bar'>
<button onclick='freeze()'>冻结当前帧并 SAM/YOLO 预标</button>
<button id='hinge' class='active' onclick='setClass(0)'>hinge (H)</button>
<button id='shaft' onclick='setClass(1)'>shaft (S)</button>
<button onclick='toggleDraw()' id='draw'>手动画轮廓</button>
<button onclick='finish()'>完成当前轮廓</button><button onclick='undo()'>撤销点</button>
<button onclick='removeSelected()'>删除选中</button><button onclick='clearClass()' class='danger'>清空当前类</button><button onclick='saveNegative()' class='danger'>保存负样本</button>
<button onclick='save()'>保存这一帧</button><button onclick='discard()'>丢弃并回实时</button>
</div><p class='hint'>冻结后 SAM 自动分割 hinge、YOLO 自动分割 shaft；绿色=hinge，橙色=shaft。点击轮廓可选中，拖动顶点可修正；确认无目标时直接点“保存负样本”。Enter 保存，右键取消当前手画。</p>
<canvas id='canvas'></canvas><pre id='status' class='status'>连接实时画面中…</pre></main>
<script>
let frozen=false,cls=0,polys=[],current=[],selected=-1,drag=null,drawMode=false,w=1,h=1;
const c=document.querySelector('#canvas'),x=c.getContext('2d'),img=new Image();
function setClass(v){cls=v;document.querySelector('#hinge').classList.toggle('active',v===0);document.querySelector('#shaft').classList.toggle('active',v===1)}
function pos(e){const r=c.getBoundingClientRect();return{x:(e.clientX-r.left)*w/r.width,y:(e.clientY-r.top)*h/r.height}}
function color(id){return id===0?'#00ff59':'#ff8a00'}
function draw(){if(!img.complete)return;x.clearRect(0,0,c.width,c.height);x.drawImage(img,0,0,c.width,c.height);const sx=c.width/w,sy=c.height/h;
  polys.forEach((q,i)=>{if(!q.points||q.points.length<2)return;const col=color(q.class_id);x.strokeStyle=col;x.fillStyle=col;x.lineWidth=i===selected?7:4;x.beginPath();x.moveTo(q.points[0][0]*sx,q.points[0][1]*sy);q.points.slice(1).forEach(p=>x.lineTo(p[0]*sx,p[1]*sy));if(q.points.length>2)x.closePath();x.stroke();q.points.forEach(p=>{x.beginPath();x.arc(p[0]*sx,p[1]*sy,i===selected?7:4,0,Math.PI*2);x.fill()});const p=q.points[0];x.font='16px sans-serif';x.fillText(`${q.class_id===0?'hinge':'shaft'}${q.confidence==null?'':` ${Number(q.confidence).toFixed(2)}`}`,p[0]*sx+5,Math.max(18,p[1]*sy-5))});
  if(current.length){x.strokeStyle='#ffbd22';x.fillStyle='#ffbd22';x.lineWidth=6;x.beginPath();x.moveTo(current[0][0]*sx,current[0][1]*sy);current.slice(1).forEach(p=>x.lineTo(p[0]*sx,p[1]*sy));x.stroke();current.forEach(p=>{x.beginPath();x.arc(p[0]*sx,p[1]*sy,6,0,Math.PI*2);x.fill()})}}
function resize(){const s=Math.min(1,1280/w);c.width=Math.round(w*s);c.height=Math.round(h*s);draw()}
function refresh(){if(frozen)return;fetch('/api/live').then(r=>r.json()).then(v=>{w=v.width;h=v.height;img.onload=resize;img.src='/live.jpg?seq='+v.sequence;document.querySelector('#status').textContent=`实时采集中 ${v.sequence}`}).catch(e=>document.querySelector('#status').textContent='实时画面失败：'+e.message)}
function freeze(){fetch('/api/freeze',{method:'POST'}).then(r=>r.json()).then(v=>{if(!v.ok){throw Error(v.error)}frozen=true;w=v.width;h=v.height;polys=v.polygons||[];selected=-1;current=[];img.onload=resize;img.src='/frozen.jpg?t='+Date.now();document.querySelector('#status').textContent=`已冻结：YOLO ${v.detection_count} 个候选；请检查后保存`}).catch(e=>document.querySelector('#status').textContent='冻结失败：'+e.message)}
function toggleDraw(){drawMode=!drawMode;current=[];document.querySelector('#draw').classList.toggle('active',drawMode);draw()}
function finish(){if(current.length<3){return}polys.push({class_id:cls,points:current.map(p=>[p[0],p[1]])});current=[];drawMode=false;document.querySelector('#draw').classList.remove('active');draw()}
function undo(){current.pop();draw()}
function clearClass(){polys=polys.filter(q=>q.class_id!==cls);selected=-1;draw()}
function removeSelected(){if(selected>=0){polys.splice(selected,1);selected=-1;draw()}}
function hitVertex(q,z){for(let i=0;i<q.points.length;i++)if(Math.hypot(q.points[i][0]-z.x,q.points[i][1]-z.y)<22)return i;return -1}
function hitBox(q,z){const xs=q.points.map(p=>p[0]),ys=q.points.map(p=>p[1]);return z.x>=Math.min(...xs)-8&&z.x<=Math.max(...xs)+8&&z.y>=Math.min(...ys)-8&&z.y<=Math.max(...ys)+8}
c.addEventListener('pointerdown',e=>{if(!frozen||e.button!==0)return;e.preventDefault();c.setPointerCapture(e.pointerId);const z=pos(e);if(drawMode){current.push([z.x,z.y]);draw();return}selected=-1;for(let i=polys.length-1;i>=0;i--){const v=hitVertex(polys[i],z);if(v>=0){selected=i;drag={index:i,vertex:v};break}if(hitBox(polys[i],z)){selected=i;break}}draw()});
c.addEventListener('pointermove',e=>{if(!drag)return;const z=pos(e);polys[drag.index].points[drag.vertex]=[z.x,z.y];draw()});
c.addEventListener('pointerup',e=>{if(c.hasPointerCapture(e.pointerId))c.releasePointerCapture(e.pointerId);drag=null});
c.addEventListener('contextmenu',e=>{e.preventDefault();current=[];drawMode=false;document.querySelector('#draw').classList.remove('active');draw()});
function save(){if(!frozen){return}if(current.length){document.querySelector('#status').textContent='请先完成当前轮廓或右键取消';return}fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({polygons:polys})}).then(r=>r.json()).then(v=>{if(!v.ok)throw Error(v.error);frozen=false;polys=[];selected=-1;document.querySelector('#status').textContent=`已保存 ${v.file}，实例=${v.instances}，正样本=${v.positive_frames}，负样本=${v.negative_frames}`;refresh()}).catch(e=>document.querySelector('#status').textContent='保存失败：'+e.message)}
function saveNegative(){if(!frozen){document.querySelector('#status').textContent='请先冻结一帧';return}polys=[];current=[];selected=-1;drawMode=false;document.querySelector('#draw').classList.remove('active');draw();save()}
function discard(){frozen=false;polys=[];current=[];selected=-1;refresh()}
document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='h')setClass(0);if(e.key.toLowerCase()==='s')setClass(1);if(e.key==='Delete')removeSelected();if(e.key==='Enter'){e.preventDefault();save()}});setInterval(refresh,300);refresh();
</script></html>"""


def mask_to_polygon(mask: np.ndarray, *, min_area: float = 20.0) -> list[list[float]] | None:
    binary = np.asarray(mask, dtype=np.uint8).squeeze()
    if binary.ndim != 2:
        return None
    found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < float(min_area):
        return None
    epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
    points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
    if len(points) < 3:
        return None
    return [[float(x), float(y)] for x, y in points]


class YoloAssistant:
    def __init__(self, model_path: Path, *, confidence: float, imgsz: int, device: str) -> None:
        self.model = load_yolo_segmentation_model(model_path, device=device)
        self.confidence = float(confidence)
        self.imgsz = int(imgsz)
        self.device = device
        self.lock = threading.Lock()

    def propose(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        with self.lock:
            masks, scores, class_ids = infer_yolo_masks(
                self.model, image_bgr, confidence=self.confidence, imgsz=self.imgsz, device=self.device
            )
        proposals: list[dict[str, Any]] = []
        for mask, score, class_id in zip(masks, scores, class_ids):
            class_value = int(class_id)
            # The current YOLO hinge result is deliberately ignored: SAM3 is
            # the source of hinge polygons for this collector.
            if class_value != 1:
                continue
            points = mask_to_polygon(mask)
            if points is None:
                continue
            proposals.append({"class_id": class_value, "class_name": CLASS_NAMES[class_value], "confidence": round(float(score), 4), "points": points})
        return proposals


class HybridAssistant:
    def __init__(
        self,
        yolo: YoloAssistant,
        sam_checkpoint: Path,
        *,
        sam_prompt: str,
        sam_threshold: float,
        max_hinge_proposals: int,
    ) -> None:
        self.yolo = yolo
        self.sam = SamAssistant(sam_checkpoint, threshold=sam_threshold, default_prompt=sam_prompt)
        self.sam_prompt = sam_prompt
        self.max_hinge_proposals = int(max_hinge_proposals)

    def propose(self, image_bgr: np.ndarray) -> list[dict[str, Any]]:
        ok, encoded = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError("无法编码冻结图像供 SAM3 使用")
        hinge_polygons, hinge_scores = self.sam.propose(
            encoded.tobytes(), prompt=self.sam_prompt
        )
        proposals: list[dict[str, Any]] = []
        for points, score in zip(hinge_polygons[: self.max_hinge_proposals], hinge_scores[: self.max_hinge_proposals]):
            proposals.append({"class_id": 0, "class_name": "hinge", "confidence": float(score), "source": "sam3", "points": points})
        proposals.extend(self.yolo.propose(image_bgr))
        return proposals


class CaptureStore:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.image_dir = out_dir / "images_all"
        self.label_dir = out_dir / "labels_all"
        self.manifest_path = out_dir / "manifest.jsonl"
        self.lock = threading.RLock()
        self.latest_jpeg: bytes | None = None
        self.latest_shape: tuple[int, int] | None = None
        self.latest_sequence = 0
        self.frozen_jpeg: bytes | None = None
        self.frozen_shape: tuple[int, int] | None = None
        self.next_index = self._next_index()
        self.positive, self.negative = self._counts()
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "classes.txt").write_text("hinge\nshaft\n", encoding="utf-8")
        (out_dir / "dataset_info.json").write_text(json.dumps({"class_names": ["hinge", "shaft"], "source": "G2 left D405 native colour frames"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _next_index(self) -> int:
        values = []
        for path in (self.out_dir / "images_all").glob("frame_*.jpg"):
            try:
                values.append(int(path.stem.rsplit("_", 1)[1]))
            except ValueError:
                pass
        return max(values, default=0) + 1

    def _counts(self) -> tuple[int, int]:
        positive = negative = 0
        for path in (self.out_dir / "labels_all").glob("frame_*.txt"):
            if path.read_text(encoding="utf-8").strip():
                positive += 1
            else:
                negative += 1
        return positive, negative

    def update_latest(self, jpeg: bytes, shape: tuple[int, int]) -> None:
        with self.lock:
            self.latest_jpeg, self.latest_shape = jpeg, shape
            self.latest_sequence += 1

    def freeze(self) -> tuple[bytes, tuple[int, int]]:
        with self.lock:
            if self.latest_jpeg is None or self.latest_shape is None:
                raise RuntimeError("尚未收到左 D405 图像")
            self.frozen_jpeg, self.frozen_shape = self.latest_jpeg, self.latest_shape
            return self.frozen_jpeg, self.frozen_shape

    def discard(self) -> None:
        with self.lock:
            self.frozen_jpeg = None
            self.frozen_shape = None

    def save(self, polygons: Any) -> dict[str, Any]:
        with self.lock:
            if self.frozen_jpeg is None or self.frozen_shape is None:
                raise RuntimeError("请先冻结一帧")
            if not isinstance(polygons, list):
                raise ValueError("polygons 必须是数组")
            height, width = self.frozen_shape
            lines: list[str] = []
            clean: list[dict[str, Any]] = []
            for item in polygons:
                if not isinstance(item, dict):
                    continue
                class_id = int(item.get("class_id", -1))
                points = item.get("points", [])
                if class_id not in CLASS_NAMES or not isinstance(points, list) or len(points) < 3:
                    continue
                coords: list[str] = []
                saved_points: list[list[float]] = []
                for point in points:
                    if not isinstance(point, list) or len(point) != 2:
                        raise ValueError("顶点格式必须是 [x, y]")
                    x, y = float(point[0]), float(point[1])
                    if not np.isfinite(x) or not np.isfinite(y):
                        raise ValueError("顶点坐标无效")
                    x, y = max(0.0, min(width - 1.0, x)), max(0.0, min(height - 1.0, y))
                    saved_points.append([x, y])
                    coords.extend((f"{x / width:.8f}", f"{y / height:.8f}"))
                lines.append(f"{class_id} " + " ".join(coords))
                clean.append({"class_id": class_id, "class_name": CLASS_NAMES[class_id], "points": saved_points})
            stem = f"frame_{self.next_index:06d}"
            (self.image_dir / f"{stem}.jpg").write_bytes(self.frozen_jpeg)
            (self.label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            record = {"index": self.next_index, "timestamp_unix": time.time(), "annotation": "positive" if lines else "negative", "instances": len(lines), "image_path": str(self.image_dir / f"{stem}.jpg"), "label_path": str(self.label_dir / f"{stem}.txt")}
            with self.manifest_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.next_index += 1
            self.positive += int(bool(lines))
            self.negative += int(not lines)
            self.discard()
            return {"ok": True, "file": f"{stem}.jpg", "instances": len(lines), "positive_frames": self.positive, "negative_frames": self.negative}


def handler_class(store: CaptureStore, assistant: HybridAssistant):
    class Handler(BaseHTTPRequestHandler):
        def json(self, value: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode("utf-8")
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def jpeg(self, value: bytes | None) -> None:
            if value is None:
                self.send_error(503, "image is not ready"); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(value))); self.end_headers(); self.wfile.write(value)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/":
                body = PAGE.encode("utf-8"); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            with store.lock:
                if path == "/live.jpg": self.jpeg(store.latest_jpeg); return
                if path == "/frozen.jpg": self.jpeg(store.frozen_jpeg); return
                if path == "/api/live": self.json({"sequence": store.latest_sequence, "width": (store.latest_shape or (0, 0))[1], "height": (store.latest_shape or (0, 0))[0]}); return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/freeze":
                    jpeg, shape = store.freeze(); image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if image is None: raise RuntimeError("无法解码冻结图像")
                    polygons = assistant.propose(image)
                    self.json({"ok": True, "width": shape[1], "height": shape[0], "polygons": polygons, "detection_count": len(polygons)}); return
                if self.path == "/api/save": self.json(store.save(payload.get("polygons", []))); return
                if self.path == "/api/discard": store.discard(); self.json({"ok": True}); return
                self.json({"ok": False, "error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            except Exception as exc:
                self.json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="左 D405 使用 SAM3 分割 hinge、YOLO-seg 分割 shaft")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--sam-prompt", default="silver triangular hinge")
    parser.add_argument("--sam-threshold", type=float, default=0.10)
    parser.add_argument("--max-hinge-proposals", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "datasets" / "left_d405_hinge_shaft_yolo_raw_g2_v1")
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--warmup", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 <= args.confidence <= 1.0 and args.imgsz >= 32 and 0 < args.port < 65536 and args.warmup >= 0 and 0.0 <= args.sam_threshold <= 1.0 and args.max_hinge_proposals >= 1):
        print("参数错误：confidence、imgsz、port 或 warmup 无效。", file=sys.stderr); return 2
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        print(f"找不到 YOLO 权重：{model_path}", file=sys.stderr); return 2
    sam_path = args.sam_checkpoint.expanduser().resolve()
    if not sam_path.is_file():
        print(f"找不到 SAM3 权重：{sam_path}", file=sys.stderr); return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr); return 2
    try:
        assistant = HybridAssistant(
            YoloAssistant(model_path, confidence=args.confidence, imgsz=args.imgsz, device=args.device),
            sam_path,
            sam_prompt=args.sam_prompt,
            sam_threshold=args.sam_threshold,
            max_hinge_proposals=args.max_hinge_proposals,
        )
        store = CaptureStore(args.out_dir.expanduser().resolve())
        server = ThreadingHTTPServer((args.host, args.port), handler_class(store, assistant))
    except Exception as exc:
        print(f"YOLO 采集器初始化失败：{exc}", file=sys.stderr); return 2
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"实时标注页：http://{args.host}:{args.port}")
    print(f"SAM3 权重：{sam_path}")
    print(f"YOLO shaft 权重：{model_path}")
    print(f"输出目录：{store.out_dir}")
    print("只读模式：SAM3 分割 hinge、YOLO 分割 shaft；不发送机器人、运动或夹爪命令。")
    initialized = False; camera = None
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess: raise RuntimeError("GDK 初始化失败")
        initialized = True; color_type = agibot_gdk.CameraType.kHandLeftColor; camera = agibot_gdk.Camera([color_type])
        for _ in range(args.warmup): camera.get_latest_image(color_type, 1000.0)
        while True:
            image = camera.get_latest_image(color_type, 1000.0)
            rgb = decode_color_rgb(image, agibot_gdk); bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok: raise RuntimeError("无法编码左 D405 图像")
            store.update_latest(encoded.tobytes(), (int(bgr.shape[0]), int(bgr.shape[1])))
    except KeyboardInterrupt:
        print("收到 Ctrl-C，停止只读 YOLO 采集。")
        return 0
    except Exception as exc:
        print(f"左 D405 YOLO 采集失败：{exc}", file=sys.stderr); return 1
    finally:
        server.shutdown(); server.server_close()
        if camera is not None:
            try: camera.close_camera()
            except Exception: pass
        if initialized:
            try: agibot_gdk.gdk_release()
            except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
