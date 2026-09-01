#!/usr/bin/env python3
"""Fast browser annotator for ``hole``/``target`` YOLO boxes.

The tool samples a video (or reads an image directory), optionally creates
low-threshold YOLO prelabels, and lets an operator correct boxes in a browser.
It never opens GDK and never sends robot commands.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2

try:
    from .hole_target_detector import infer_detection, load_detection_model
except ImportError:
    from hole_target_detector import infer_detection, load_detection_model


CLASS_NAMES = ("hole", "target")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>hole/target 快速框标注</title>
<style>
body{margin:0;background:#171717;color:#eee;font:15px sans-serif}main{max-width:1300px;margin:auto;padding:12px}
#toolbar{display:flex;gap:8px;align-items:center;flex-wrap:wrap;background:#252525;padding:10px;border-radius:6px}
button{padding:7px 11px;background:#444;color:#fff;border:1px solid #666;border-radius:4px;cursor:pointer}button.active{background:#1976d2}
#canvas{display:block;max-width:100%;margin-top:10px;background:#222;cursor:crosshair}#status{white-space:pre-wrap;background:#252525;padding:10px;border-radius:6px}
.hint{color:#bbb}.danger{background:#7b2222}
</style>
<main><div id="toolbar">
<button id="hole" class="active" onclick="setClass(0)">hole (H)</button><button id="target" onclick="setClass(1)">target (T)</button>
<button onclick="previous()">上一帧</button><button onclick="next(true)">保存并下一帧</button><button onclick="save(false)">保存</button><button class="danger" onclick="clearBoxes()">清空框</button><button onclick="skip()">跳过</button>
<label>跳转帧号 <input id="jump" type="number" min="1" value="1" style="width:76px;padding:6px;background:#333;color:#fff;border:1px solid #666;border-radius:4px" onkeydown="if(event.key==='Enter'){event.preventDefault();event.stopPropagation();jumpTo()}"></label><button onclick="jumpTo()">跳转</button>
<span id="counter"></span></div>
<p class="hint">帧号按抽帧后的标注序号计（从 1 开始）；初始为 hole，类别选择会保持；按 T 切换 target；右键复制上一帧位置；Enter 保存并下一帧。</p>
<canvas id="canvas"></canvas><pre id="status">加载中…</pre></main>
<script>
let index=0, total=0, width=1, height=1, boxes=[], currentClass=0, selected=-1, dragging=null;
const canvas=document.querySelector('#canvas'), ctx=canvas.getContext('2d'), image=new Image();
function setClass(c){currentClass=c;document.querySelector('#hole').classList.toggle('active',c===0);document.querySelector('#target').classList.toggle('active',c===1)}
function fit(){const scale=Math.min(1,1200/width);canvas.width=Math.round(width*scale);canvas.height=Math.round(height*scale);draw();return scale}
function toImage(e){const r=canvas.getBoundingClientRect();return {x:(e.clientX-r.left)*width/canvas.width,y:(e.clientY-r.top)*height/canvas.height}}
function norm(b){return {x0:Math.max(0,Math.min(width,b.x0)),y0:Math.max(0,Math.min(height,b.y0)),x1:Math.max(0,Math.min(width,b.x1)),y1:Math.max(0,Math.min(height,b.y1)),class_id:b.class_id}}
function draw(){ctx.clearRect(0,0,canvas.width,canvas.height);if(image.complete)ctx.drawImage(image,0,0,canvas.width,canvas.height);const sx=canvas.width/width,sy=canvas.height/height;boxes.forEach((b,i)=>{const x=Math.min(b.x0,b.x1)*sx,y=Math.min(b.y0,b.y1)*sy,w=Math.abs(b.x1-b.x0)*sx,h=Math.abs(b.y1-b.y0)*sy;ctx.strokeStyle=b.class_id===0?'#00ff66':'#ff9d00';ctx.lineWidth=i===selected?4:2;ctx.strokeRect(x,y,w,h);ctx.fillStyle=ctx.strokeStyle;ctx.font='16px sans-serif';ctx.fillText((b.class_id===0?'hole':'target')+' '+(i+1),x+3,Math.max(16,y-4))})}
function load(){fetch('/api/frame/'+index).then(r=>r.json()).then(s=>{total=s.total;index=s.index;width=s.width;height=s.height;boxes=s.boxes||[];selected=-1;document.querySelector('#jump').max=total;document.querySelector('#jump').value=index+1;image.onload=()=>{fit()};image.src='/image/'+s.file+'?v='+Date.now();document.querySelector('#counter').textContent=`${index+1}/${total}  ${s.name}`;document.querySelector('#status').textContent=JSON.stringify({frame:index+1,boxes:boxes,prelabel:s.prelabel},null,2)})}
function save(goNext){fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:index,boxes:boxes})}).then(r=>r.json()).then(s=>{if(s.ok&&goNext)next(false);else document.querySelector('#status').textContent='已保存\n'+JSON.stringify(s,null,2)})}
function next(saveFirst){if(saveFirst){save(true)}else if(index+1<total){index++;load()}}
function previous(){if(index>0){index--;load()}}
function jumpTo(){const value=Number(document.querySelector('#jump').value);if(!Number.isInteger(value)||value<1||value>total){document.querySelector('#status').textContent=`帧号范围：1-${total}`;return}const target=value-1;if(target===index)return;index=target;load()}
function clearBoxes(){boxes=[];selected=-1;draw()}
function skip(){if(index+1<total){index++;load()}}
canvas.addEventListener('mousedown',e=>{if(e.button!==0)return;const p=toImage(e);selected=-1;for(let i=boxes.length-1;i>=0;i--){const b=boxes[i];if(p.x>=Math.min(b.x0,b.x1)&&p.x<=Math.max(b.x0,b.x1)&&p.y>=Math.min(b.y0,b.y1)&&p.y<=Math.max(b.y0,b.y1)){selected=i;break}}if(selected<0)dragging={x0:p.x,y0:p.y,x1:p.x,y1:p.y,class_id:currentClass};draw()});
canvas.addEventListener('mousemove',e=>{if(!dragging)return;const p=toImage(e);dragging.x1=p.x;dragging.y1=p.y;draw();const b=norm(dragging),sx=canvas.width/width,sy=canvas.height/height;ctx.strokeStyle=currentClass===0?'#00ff66':'#ff9d00';ctx.lineWidth=2;ctx.strokeRect(Math.min(b.x0,b.x1)*sx,Math.min(b.y0,b.y1)*sy,Math.abs(b.x1-b.x0)*sx,Math.abs(b.y1-b.y0)*sy)});
canvas.addEventListener('mouseup',e=>{if(!dragging)return;const b=norm(dragging);dragging=null;if(Math.abs(b.x1-b.x0)>=4&&Math.abs(b.y1-b.y0)>=4)boxes.push(b);draw()});
canvas.addEventListener('contextmenu',e=>{e.preventDefault();if(index<=0){document.querySelector('#status').textContent='第一帧没有上一帧';return}fetch('/api/frame/'+(index-1)).then(r=>r.json()).then(s=>{boxes=(s.boxes||[]).map(b=>({...b}));selected=-1;draw();document.querySelector('#status').textContent=`已复制上一帧位置（${boxes.length} 个框），未保存`})});
document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='h')setClass(0);if(e.key.toLowerCase()==='t')setClass(1);if(e.key==='Delete'&&selected>=0){boxes.splice(selected,1);selected=-1;draw()}if(e.key==='Enter'){e.preventDefault();save(true)}if(e.key===' '){e.preventDefault();save(true)}});load();
</script>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="视频抽帧并在浏览器快速标注 hole/target YOLO 框")
    parser.add_argument("--source", type=Path, required=True, help="视频文件或图片目录")
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/hole_target_box_raw"))
    parser.add_argument("--model", type=Path, default=Path("models/hole_target_det_v2/best.pt"))
    parser.add_argument("--stride", type=int, default=6, help="视频抽帧间隔；30 FPS 视频默认每秒约 5 帧")
    parser.add_argument("--confidence", type=float, default=0.10, help="自动预标置信度，建议 0.05～0.20")
    parser.add_argument("--max-frames", type=int, default=0, help="最多抽取帧数，0 表示不限制")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-prelabel", action="store_true", help="不加载 YOLO，全部手工画框")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    return parser.parse_args()


def image_sources(source: Path, stride: int) -> list[tuple[str, Any]]:
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        return [(p.name, p) for p in files]
    if not source.is_file():
        raise RuntimeError(f"找不到输入：{source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频：{source}")
    result: list[tuple[str, Any]] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % stride == 0:
            result.append((f"{source.stem}_{frame_index:07d}.jpg", frame))
        frame_index += 1
    capture.release()
    return result


def load_image(item: Any):
    if isinstance(item, Path):
        image = cv2.imread(str(item), cv2.IMREAD_COLOR)
    else:
        image = item
    if image is None:
        raise RuntimeError("读取图像失败")
    return image


def to_yolo(box: dict, width: int, height: int) -> str:
    x0, y0, x1, y1 = float(box["x0"]), float(box["y0"]), float(box["x1"]), float(box["y1"])
    left, right = sorted((max(0.0, min(width, x0)), max(0.0, min(width, x1))))
    top, bottom = sorted((max(0.0, min(height, y0)), max(0.0, min(height, y1))))
    bw, bh = right - left, bottom - top
    return f"{int(box['class_id'])} {(left + right) / 2 / width:.6f} {(top + bottom) / 2 / height:.6f} {bw / width:.6f} {bh / height:.6f}"


def read_yolo(path: Path, width: int, height: int) -> list[dict]:
    if not path.is_file():
        return []
    boxes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 5:
            continue
        class_id, cx, cy, bw, bh = int(values[0]), *(float(v) for v in values[1:5])
        boxes.append({"class_id": class_id, "x0": (cx - bw / 2) * width, "y0": (cy - bh / 2) * height, "x1": (cx + bw / 2) * width, "y1": (cy + bh / 2) * height})
    return boxes


class State:
    def __init__(self, out_dir: Path, items: list[tuple[str, Path]], metadata: list[dict]) -> None:
        self.out_dir, self.items, self.metadata = out_dir, items, metadata
        self.lock = threading.Lock()


def main() -> int:
    args = parse_args()
    if args.stride < 1 or args.max_frames < 0 or not 0.0 <= args.confidence <= 1.0 or not 0 <= args.port <= 65535:
        print("参数错误：stride、confidence、max-frames 或端口无效。", file=sys.stderr)
        return 2
    source = args.source.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    image_dir, label_dir = out_dir / "images", out_dir / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    try:
        raw_items = image_sources(source, args.stride)
        if args.max_frames:
            raw_items = raw_items[: args.max_frames]
        if not raw_items:
            raise RuntimeError("输入中没有可用帧")
        model = None
        if not args.no_prelabel:
            model_path = args.model.expanduser().resolve()
            if model_path.is_file():
                print(f"加载 YOLO 预标模型：{model_path}")
                model = load_detection_model(str(model_path), device=args.device)
            else:
                print(f"未找到预标模型，改为纯手工：{model_path}", file=sys.stderr)
        items: list[tuple[str, Path]] = []
        metadata: list[dict] = []
        for frame_number, (name, raw) in enumerate(raw_items):
            frame = load_image(raw)
            image_path = image_dir / name
            if not image_path.is_file():
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError(f"保存图像失败：{image_path}")
            height, width = frame.shape[:2]
            label_path = label_dir / f"{image_path.stem}.txt"
            prelabel = False
            if not label_path.exists() and model is not None:
                detections = infer_detection(model, frame, confidence=args.confidence, iou=0.45, imgsz=640, device=args.device)
                lines = []
                for item in detections:
                    if item.class_name not in CLASS_NAMES:
                        continue
                    lines.append(to_yolo({"class_id": CLASS_NAMES.index(item.class_name), "x0": item.xyxy[0], "y0": item.xyxy[1], "x1": item.xyxy[2], "y1": item.xyxy[3]}, width, height))
                label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                prelabel = True
            items.append((image_path.name, image_path))
            metadata.append({"name": image_path.name, "width": width, "height": height, "prelabel": prelabel})
        (out_dir / "classes.txt").write_text("hole\ntarget\n", encoding="utf-8")
        (out_dir / "index.json").write_text(json.dumps({"classes": list(CLASS_NAMES), "source": str(source), "stride": args.stride, "items": metadata}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        print(f"准备标注数据失败：{exc}", file=sys.stderr)
        return 1

    state = State(out_dir, items, metadata)

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

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                data = INDEX_HTML.encode()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/image/"):
                name = Path(path.removeprefix("/image/")).name
                image_path = image_dir / name
                if not image_path.is_file():
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                data = image_path.read_bytes()
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(data)
                return
            if path.startswith("/api/frame/"):
                try:
                    index = max(0, min(len(state.items) - 1, int(path.rsplit("/", 1)[1])))
                except ValueError:
                    self.send_error(HTTPStatus.BAD_REQUEST)
                    return
                name, image_path = state.items[index]
                meta = state.metadata[index]
                boxes = read_yolo(label_dir / f"{image_path.stem}.txt", int(meta["width"]), int(meta["height"]))
                self.send_json({"index": index, "total": len(state.items), "file": name, "name": name, "width": meta["width"], "height": meta["height"], "boxes": boxes, "prelabel": meta["prelabel"]})
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] != "/api/save":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                index = int(payload["index"])
                boxes = payload.get("boxes", [])
                name, image_path = state.items[index]
                meta = state.metadata[index]
                lines = [to_yolo(box, int(meta["width"]), int(meta["height"])) for box in boxes if int(box.get("class_id", -1)) in (0, 1)]
                (label_dir / f"{image_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
                self.send_json({"ok": True, "index": index, "labels": len(lines), "file": name})
            except Exception as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"标注页：http://{args.host}:{args.port}")
    print(f"输出目录：{out_dir}")
    print("Ctrl-C 停止；图像写入 images/，YOLO 标签写入 labels/。")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
