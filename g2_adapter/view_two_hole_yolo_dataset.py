#!/usr/bin/env python3
"""Read-only browser viewer for the head RGB-D two-hole YOLO labels.

The viewer reads existing images and YOLO txt files. It never opens GDK,
loads the detection model, or writes into the dataset.
"""

from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2


DEFAULT_DATASET = Path(__file__).resolve().parents[1] / "datasets/two_hole_head_rgbd_yolo_v2"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

PAGE = r"""<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><title>头部 RGB-D 双孔 YOLO 标注查看</title>
<style>
body{margin:0;background:#171b20;color:#eef2f6;font:15px sans-serif}
main{max-width:1250px;margin:auto;padding:18px}h1{font-size:21px;margin:0 0 12px}
.bar{display:flex;gap:9px;align-items:center;flex-wrap:wrap;background:#252c34;padding:12px;border-radius:8px}
button,select,input{background:#38424d;color:#fff;border:1px solid #637282;border-radius:5px;padding:7px}
button{cursor:pointer}button:hover{background:#4d5d6c}input[type=number]{width:85px}
#stage{margin-top:15px;overflow:auto;max-height:72vh;background:#0d1115;border-radius:8px;padding:12px}
canvas{display:block;image-rendering:auto}#meta{white-space:pre-wrap;line-height:1.5;background:#252c34;padding:12px;border-radius:8px}
.note{color:#adbac7}.c0{color:#51ef96}.c1{color:#ffbf69}
</style><main>
<h1>头部 RGB-D 双孔 YOLO 标注查看</h1>
<div class="bar">
<label>数据分组 <select id="split"><option value="val">验证集</option><option value="train">训练集</option></select></label>
<button id="prev">← 上一张</button><button id="next">下一张 →</button>
<label>第 <input id="index" type="number" min="1" value="1"> 张</label><button id="jump">跳转</button>
<label>放大 <select id="scale"><option value="1">1×</option><option value="2" selected>2×</option><option value="3">3×</option></select></label>
<label><input id="show0" type="checkbox" checked> <span class="c0">0 台面不规则孔</span></label>
<label><input id="show1" type="checkbox" checked> <span class="c1">1 铰链端圆孔</span></label>
<strong id="counter"></strong>
</div>
<p class="note">只读查看原始图片和对应 YOLO 标签；←/→ 翻页。画框颜色表示类别，红字表示标签问题。此页不会保存或修改数据。</p>
<div id="stage"><canvas id="canvas"></canvas></div><pre id="meta">正在加载…</pre>
<script>
const $=id=>document.getElementById(id), canvas=$('canvas'),ctx=canvas.getContext('2d');
const image=new Image();let state=null,serial=0;
function redraw(){if(!state||!image.complete)return;const k=Number($('scale').value);canvas.width=state.width*k;canvas.height=state.height*k;ctx.drawImage(image,0,0,canvas.width,canvas.height);ctx.lineWidth=Math.max(2,1.5*k);ctx.font=`bold ${Math.max(13,10*k)}px sans-serif`;
 for(const b of state.boxes){if(!$(b.class_id===0?'show0':'show1').checked)continue;const color=b.class_id===0?'#38ef83':'#ffad43';ctx.strokeStyle=color;ctx.fillStyle=color;ctx.strokeRect(b.x0*k,b.y0*k,(b.x1-b.x0)*k,(b.y1-b.y0)*k);ctx.fillText(`${b.class_id} ${state.classes[b.class_id]}`,b.x0*k+2,Math.max(14,b.y0*k-4));}}
async function load(){const id=++serial,split=$('split').value,index=Math.max(0,Number($('index').value)-1);try{const r=await fetch(`/api/sample?split=${split}&index=${index}`);const d=await r.json();if(!r.ok)throw Error(d.error||r.status);if(id!==serial)return;state=d;$('index').value=d.index+1;$('index').max=d.total;$('counter').textContent=`${d.index+1} / ${d.total}`;image.onload=redraw;image.src=`/image?split=${split}&index=${d.index}&v=${id}`;$('meta').textContent=`文件：${d.name}\n尺寸：${d.width} × ${d.height}\n标签：${d.boxes.length} 个\n`+d.boxes.map(b=>`类别 ${b.class_id} ${d.classes[b.class_id]}：(${b.x0.toFixed(1)}, ${b.y0.toFixed(1)}) → (${b.x1.toFixed(1)}, ${b.y1.toFixed(1)})`).join('\n')+(d.warnings.length?'\n标签问题：'+d.warnings.join('; '):'');}catch(e){$('meta').textContent='加载失败：'+e.message;}}
function step(n){if(!state)return;$('index').value=Math.min(state.total,Math.max(1,state.index+1+n));load()}
$('prev').onclick=()=>step(-1);$('next').onclick=()=>step(1);$('jump').onclick=load;
$('split').onchange=()=>{$('index').value=1;load()};$('scale').onchange=redraw;
$('show0').onchange=redraw;$('show1').onchange=redraw;
$('index').onkeydown=e=>{if(e.key==='Enter')load()};document.onkeydown=e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft')step(-1);if(e.key==='ArrowRight')step(1)};
load();
</script></main></html>"""


def read_boxes(path: Path, width: int, height: int) -> tuple[list[dict], list[str]]:
    boxes: list[dict] = []
    warnings: list[str] = []
    if not path.is_file():
        return boxes, ["缺少同名标签文件"]
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("不是 5 列 YOLO 检测框")
            class_id = int(parts[0])
            cx, cy, bw, bh = map(float, parts[1:])
            if class_id not in (0, 1) or not all(0 <= value <= 1 for value in (cx, cy, bw, bh)) or bw <= 0 or bh <= 0:
                raise ValueError("类别或归一化坐标超出范围")
            x0, y0 = (cx - bw / 2) * width, (cy - bh / 2) * height
            x1, y1 = (cx + bw / 2) * width, (cy + bh / 2) * height
            if x0 < -0.01 or y0 < -0.01 or x1 > width + 0.01 or y1 > height + 0.01:
                raise ValueError("框超出图片")
            boxes.append({"class_id": class_id, "x0": x0, "y0": y0, "x1": x1, "y1": y1})
        except ValueError as error:
            warnings.append(f"第 {line_number} 行：{error}")
    return boxes, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    args = parser.parse_args()
    root = args.data.expanduser().resolve()
    info = json.loads((root / "dataset_info.json").read_text(encoding="utf-8"))
    classes = {int(key): value for key, value in info["classes"].items()}
    if classes != {0: "irregular_table_hole", 1: "round_hinge_tip_hole"}:
        raise ValueError("数据集类别与头部双孔模型不匹配")
    images = {
        split: sorted(path for path in (root / "images" / split).iterdir()
                      if path.suffix.lower() in IMAGE_SUFFIXES)
        for split in ("train", "val")
    }
    if not all(images.values()):
        raise ValueError("训练集或验证集为空")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            pass

        def send_data(self, data: bytes, content_type: str, status=HTTPStatus.OK) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            route = urlsplit(self.path)
            if route.path in ("/", "/index.html"):
                self.send_data(PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            if route.path not in ("/api/sample", "/image"):
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            query = parse_qs(route.query)
            split = query.get("split", ["val"])[0]
            try:
                index = int(query.get("index", ["0"])[0])
            except ValueError:
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            if split not in images or not 0 <= index < len(images[split]):
                self.send_error(HTTPStatus.BAD_REQUEST)
                return
            image = images[split][index]
            if route.path == "/image":
                content_type = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
                self.send_data(image.read_bytes(), content_type)
                return
            frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if frame is None:
                self.send_error(HTTPStatus.INTERNAL_SERVER_ERROR, "Cannot read image")
                return
            height, width = frame.shape[:2]
            boxes, warnings = read_boxes(root / "labels" / split / f"{image.stem}.txt", width, height)
            payload = {"split": split, "index": index, "total": len(images[split]),
                       "name": image.name, "width": width, "height": height,
                       "boxes": boxes, "warnings": warnings, "classes": classes}
            self.send_data(json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                           "application/json; charset=utf-8")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"头部双孔 YOLO 标注查看：http://{args.host}:{server.server_port}", flush=True)
    print(f"训练图 {len(images['train'])} 张，验证图 {len(images['val'])} 张；只读，不连接机器人。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
