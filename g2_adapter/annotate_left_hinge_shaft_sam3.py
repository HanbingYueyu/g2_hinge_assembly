#!/usr/bin/env python3
"""Offline SAM-assisted annotation for existing left-camera hinge images.

Existing class-0 hinge polygons are reused from ``labels_all`` when supplied.
SAM3 proposes class-1 polygons on demand; every proposal remains editable and
must be confirmed by the operator before saving.  The second class defaults to
``shaft`` and can be renamed (for example, to ``hole``) for another dataset.
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
import numpy as np

try:
    from .capture_hinge_yolo_dataset_gdk import SamAssistant
except ImportError:
    from capture_hinge_yolo_dataset_gdk import SamAssistant


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("hinge", "shaft")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


PAGE = r"""<!doctype html><html lang='zh-CN'><meta charset='utf-8'>
<title>左相机 hinge/__SECONDARY_NAME__ 离线标注</title>
<style>body{background:#171717;color:#eee;font:15px sans-serif;margin:12px}
button{margin:2px;padding:7px 10px;background:#444;color:#fff;border:1px solid #777;border-radius:4px}
button.active{background:#1976d2}.danger{background:#7b2222}.hint{color:#bbb}
#cwrap{overflow:auto;max-width:96vw;max-height:78vh;border:1px solid #444;background:#101010}
canvas{display:block;border:1px solid #777;cursor:crosshair;touch-action:none}
#status{white-space:pre-wrap;background:#252525;padding:8px;margin-top:8px}</style>
<main><div>
<button onclick='propose()'>SAM 生成 __SECONDARY_NAME__ 建议</button>
<button id='hinge' class='active' onclick='setClass(0)'>hinge (H)</button>
<button id='shaft' onclick='setClass(1)'>__SECONDARY_NAME__ (S)</button>
<button id='draw' onclick='toggleDraw()'>手动画轮廓</button>
<button id='ellipse' onclick='toggleEllipse()'>画椭圆（新建）</button>
<button onclick='copyPrevHoles()'>复制上一帧 __SECONDARY_NAME__（右键）</button>
<button onclick='finish()'>完成当前轮廓</button><button onclick='undo()'>撤销点</button>
<button onclick='removeSelected()'>删除选中</button><button class='danger' onclick='clearShaft()'>清空 __SECONDARY_NAME__</button>
<button onclick='previous()'>上一帧</button><button onclick='saveNext()'>保存并下一帧</button>
<button onclick='saveOnly()'>保存</button><button onclick='skip()'>跳过</button>
<label>帧号 <input id='jump' type='number' min='1' style='width:70px' onkeydown="if(event.key==='Enter')jumpTo()"></label>
<button onclick='jumpTo()'>跳转</button>
<button onclick='zoomBy(-1)'>缩小 −</button><span id='zoomLabel' style='display:inline-block;min-width:52px;text-align:center'>100%</span><button onclick='zoomBy(1)'>放大 +</button><button onclick='zoomFit()'>适应</button><button onclick='zoomReset()'>1:1</button></div>
<p class='hint'>旧 hinge 掩膜自动保留；SAM 只保留靠近 hinge 的少量 __SECONDARY_NAME__ 建议。新建椭圆：点击“画椭圆（新建）”后按住左键拖出，松开后自动进入编辑；编辑时点击橙色椭圆，再拖中心移动或拖四周橙色手柄调整大小。右键：绘制或选中时取消当前操作，空闲时复制上一帧的 __SECONDARY_NAME__ 位置。滚轮或“放大 + / 缩小 −”缩放视图（50%–400%，大图自动限制上限），1:1 复位；线宽按原图尺寸固定，放大时不会变粗。</p>
<div id='cwrap'><canvas id='canvas'></canvas></div><pre id='status'>加载中…</pre></main>
<script>
let index=0,total=0,width=1,height=1,polys=[],current=[],currentClass=0,selected=-1,drawMode=false,ellipseMode=false,ellipseStart=null,ellipsePreview=null,dragState=null,zoom=(function(){try{var z=Number(localStorage.getItem('dsh-anno-zoom'));return z>0?z:null}catch(e){return null}})();
const c=document.querySelector('#canvas'),x=c.getContext('2d'),img=new Image();
function setClass(v){currentClass=v;document.querySelector('#hinge').classList.toggle('active',v===0);document.querySelector('#shaft').classList.toggle('active',v===1)}
function clearModes(){drawMode=false;ellipseMode=false;current=[];ellipseStart=null;ellipsePreview=null;dragState=null;document.querySelector('#draw').classList.remove('active');document.querySelector('#ellipse').classList.remove('active')}
function toggleDraw(){const on=!drawMode;clearModes();if(on){drawMode=true;setClass(1);document.querySelector('#draw').classList.add('active')}draw()}
function toggleEllipse(){const on=!ellipseMode;clearModes();if(on){ellipseMode=true;setClass(1);document.querySelector('#ellipse').classList.add('active')}draw()}
function p(e){const r=c.getBoundingClientRect();return{x:(e.clientX-r.left)*width/r.width,y:(e.clientY-r.top)*height/r.height}}
function draw(){if(!img.complete)return;x.clearRect(0,0,c.width,c.height);x.drawImage(img,0,0,c.width,c.height);const sx=c.width/width,sy=c.height/height,k=Math.max(1,width/640);
  polys.forEach((q,i)=>{const col=q.class_id===0?'#00ff59':'#ff7a00';x.strokeStyle=col;x.fillStyle=col;x.lineWidth=q.class_id===1?(i===selected?2.5:2):(i===selected?3:2)*k;if(q.shape==='ellipse'){x.beginPath();x.ellipse(q.cx*sx,q.cy*sy,q.rx*sx,q.ry*sy,0,0,Math.PI*2);x.stroke();if(i===selected){[[q.cx+q.rx,q.cy],[q.cx,q.cy+q.ry],[q.cx-q.rx,q.cy],[q.cx,q.cy-q.ry]].forEach(z=>{x.beginPath();x.arc(z[0]*sx,z[1]*sy,4*k,0,Math.PI*2);x.fill()})}return}if(!q.points||q.points.length<2)return;x.beginPath();x.moveTo(q.points[0][0]*sx,q.points[0][1]*sy);q.points.slice(1).forEach(z=>x.lineTo(z[0]*sx,z[1]*sy));if(q.points.length>2)x.closePath();x.stroke();if(q.class_id===0)q.points.forEach(z=>{x.beginPath();x.arc(z[0]*sx,z[1]*sy,3*k,0,Math.PI*2);x.fill()})});
  const e=ellipsePreview;if(e){x.strokeStyle='#ff7a00';x.lineWidth=2;x.beginPath();x.ellipse(e.cx*sx,e.cy*sy,e.rx*sx,e.ry*sy,0,0,Math.PI*2);x.stroke()}if(current.length){x.strokeStyle='#ff7a00';x.fillStyle='#ff7a00';x.lineWidth=2;x.beginPath();x.moveTo(current[0][0]*sx,current[0][1]*sy);current.slice(1).forEach(z=>x.lineTo(z[0]*sx,z[1]*sy));x.stroke();current.forEach(z=>{x.beginPath();x.arc(z[0]*sx,z[1]*sy,3*k,0,Math.PI*2);x.fill()})}
}
function maxZoomFor(){const base=Math.min(1,1280/width);return Math.max(1,Math.min(4,2560/(Math.max(1,width,height)*base)))}
function applyZoom(redraw=true){if(!width||!height)return;const base=Math.min(1,1280/width);zoom=Math.max(0.5,Math.min(maxZoomFor(),zoom));const scale=base*zoom;c.width=Math.max(1,Math.round(width*scale));c.height=Math.max(1,Math.round(height*scale));c.style.width=c.width+'px';c.style.height=c.height+'px';const el=document.querySelector('#zoomLabel');if(el)el.textContent=Math.round(zoom*100)+'%';try{localStorage.setItem('dsh-anno-zoom',String(zoom))}catch(e){}if(redraw)draw()}
function zoomBy(dir){zoom*=dir>0?1.25:0.8;applyZoom()}
function fitZoom(){const base=Math.min(1,1280/width);const vw=window.innerWidth*0.96,vh=window.innerHeight*0.78;const fit=Math.min(vw/(width*base),vh/(height*base));return Math.max(1,Math.min(maxZoomFor(),fit))}
function zoomFit(){zoom=fitZoom();applyZoom()}
function zoomReset(){zoom=1;applyZoom()}
c.addEventListener('wheel',e=>{e.preventDefault();zoomBy(e.deltaY<0?1:-1)},{passive:false});
function load(){clearModes();fetch('/api/frame/'+index).then(r=>r.json()).then(v=>{index=v.index;total=v.total;width=v.width;height=v.height;polys=v.polygons||[];selected=-1;document.querySelector('#jump').max=total;document.querySelector('#jump').value=index+1;img.onload=()=>{zoom=zoom||fitZoom();applyZoom(false);draw()};img.src='/image/'+v.file+'?t='+Date.now();document.querySelector('#status').textContent=`${index+1}/${total}  ${v.file}\n轮廓=${polys.length}，其中 __SECONDARY_NAME__=${polys.filter(q=>q.class_id===1).length}`})}
function post(url,data={}){return fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(r=>r.json().then(v=>{if(!r.ok)throw Error(v.error||'请求失败');return v}))}
function ellipsePoints(q){const out=[];for(let i=0;i<48;i++){const a=2*Math.PI*i/48;out.push([q.cx+q.rx*Math.cos(a),q.cy+q.ry*Math.sin(a)])}return out}
function serializable(){return polys.map(q=>q.shape==='ellipse'?{class_id:q.class_id,points:ellipsePoints(q),shape:'ellipse',cx:q.cx,cy:q.cy,rx:q.rx,ry:q.ry}:q)}
async function copyPrevHoles(){if(index<=0){document.querySelector('#status').textContent='已是第一帧，没有上一帧可复制。';return}try{const r=await fetch('/api/frame/'+(index-1),{cache:'no-store'});const v=await r.json();if(!r.ok)throw Error(v.error||'读取上一帧失败');const prev=(v.polygons||[]).filter(q=>q.class_id===1).map(q=>JSON.parse(JSON.stringify(q)));const before=polys.filter(q=>q.class_id===1).length;if(!prev.length){document.querySelector('#status').textContent='上一帧（'+(index)+'/'+total+'）没有 __SECONDARY_NAME__，未改动本帧。';return}polys=polys.filter(q=>q.class_id!==1).concat(prev);selected=-1;draw();document.querySelector('#status').textContent='已复制上一帧 '+prev.length+' 个 __SECONDARY_NAME__'+(before?'，覆盖本帧原有 '+before+' 个':'')+'；请检查后“保存”或按 Enter。'}catch(e){document.querySelector('#status').textContent='复制上一帧失败：'+e.message}}
function propose(){document.querySelector('#status').textContent='SAM3 处理中…';post('/api/propose',{index,prompt:document.querySelector('#prompt')?.value||''}).then(v=>{polys=polys.filter(q=>q.class_id!==1).concat((v.polygons||[]).map(points=>({class_id:1,points})));draw();document.querySelector('#status').textContent=`SAM 生成 ${v.polygons.length} 个 __SECONDARY_NAME__ 建议，请检查后保存`}).catch(e=>document.querySelector('#status').textContent='SAM 失败：'+e.message)}
function finish(){if(current.length<3){return}polys.push({class_id:currentClass,points:current});current=[];drawMode=false;document.querySelector('#draw').classList.remove('active');draw()}
function undo(){current.pop();draw()}
function removeSelected(){if(selected>=0){polys.splice(selected,1);selected=-1;document.querySelector('#status').textContent='已删除选中轮廓';draw()}}
function clearShaft(){polys=polys.filter(q=>q.class_id!==1);selected=-1;draw()}
function saveOnly(){post('/api/save',{index,polygons:serializable()}).then(v=>document.querySelector('#status').textContent='已保存：'+v.file).catch(e=>document.querySelector('#status').textContent='保存失败：'+e.message)}
function saveNext(){post('/api/save',{index,polygons:serializable()}).then(()=>{if(index+1<total){index++;load()}}).catch(e=>document.querySelector('#status').textContent='保存失败：'+e.message)}
function previous(){if(index>0){index--;load()}}
function skip(){if(index+1<total){index++;load()}}
function jumpTo(){const v=Number(document.querySelector('#jump').value);if(Number.isInteger(v)&&v>=1&&v<=total){index=v-1;load()}}
function hit(q,z){if(q.shape==='ellipse'){const dx=(z.x-q.cx)/Math.max(1,q.rx),dy=(z.y-q.cy)/Math.max(1,q.ry);return dx*dx+dy*dy<=1.15}if(!q.points||q.points.length<3)return false;const xs=q.points.map(a=>a[0]),ys=q.points.map(a=>a[1]);return z.x>=Math.min(...xs)&&z.x<=Math.max(...xs)&&z.y>=Math.min(...ys)&&z.y<=Math.max(...ys)}
function ellipseHandle(q,z){if(q.shape!=='ellipse')return '';const handles={right:[q.cx+q.rx,q.cy],bottom:[q.cx,q.cy+q.ry],left:[q.cx-q.rx,q.cy],top:[q.cx,q.cy-q.ry]};for(const [name,h] of Object.entries(handles)){if(Math.hypot(z.x-h[0],z.y-h[1])<30)return name}return ''}
function selectAt(z){selected=-1;for(let i=polys.length-1;i>=0;i--){if(hit(polys[i],z)){selected=i;break}}return selected}
c.addEventListener('pointerdown',e=>{if(e.button!==0)return;e.preventDefault();c.setPointerCapture(e.pointerId);const z=p(e);if(ellipseMode){ellipseStart=z;ellipsePreview={class_id:currentClass,shape:'ellipse',cx:z.x,cy:z.y,rx:0,ry:0};dragState={type:'create'};draw();return}if(drawMode){current.push([z.x,z.y]);draw();return}if(selectAt(z)>=0&&polys[selected].shape==='ellipse'){const q=polys[selected],handle=ellipseHandle(q,z);dragState=handle?{type:handle,index:selected}:{type:'move',index:selected,dx:z.x-q.cx,dy:z.y-q.cy};document.querySelector('#status').textContent=`已选中椭圆 ${selected+1}，可拖动中心或四周手柄编辑`;}else{dragState=null}draw()});
c.addEventListener('pointermove',e=>{const z=p(e);if(dragState?.type==='create'&&ellipseStart){ellipsePreview={class_id:currentClass,shape:'ellipse',cx:(ellipseStart.x+z.x)/2,cy:(ellipseStart.y+z.y)/2,rx:Math.abs(z.x-ellipseStart.x)/2,ry:Math.abs(z.y-ellipseStart.y)/2};draw();return}if(!dragState||dragState.index===undefined)return;const q=polys[dragState.index];if(dragState.type==='move'){q.cx=z.x-dragState.dx;q.cy=z.y-dragState.dy}else if(dragState.type==='right'||dragState.type==='left'){q.rx=Math.max(3,Math.abs(z.x-q.cx))}else if(dragState.type==='top'||dragState.type==='bottom'){q.ry=Math.max(3,Math.abs(z.y-q.cy))}draw()});
c.addEventListener('pointerup',e=>{if(e.button!==0)return;if(c.hasPointerCapture(e.pointerId))c.releasePointerCapture(e.pointerId);if(dragState?.type==='create'&&ellipsePreview){if(ellipsePreview.rx>=3&&ellipsePreview.ry>=3){polys.push({...ellipsePreview});selected=polys.length-1;ellipseMode=false;document.querySelector('#ellipse').classList.remove('active');document.querySelector('#status').textContent=`已创建椭圆 ${selected+1}；可拖动中心或四周手柄编辑`;}else{document.querySelector('#status').textContent='椭圆太小，未创建'}ellipseStart=null;ellipsePreview=null;draw()}dragState=null});
c.addEventListener('pointercancel',e=>{if(c.hasPointerCapture(e.pointerId))c.releasePointerCapture(e.pointerId);ellipseStart=null;ellipsePreview=null;dragState=null;draw()});
c.addEventListener('contextmenu',e=>{e.preventDefault();if(current.length||ellipsePreview||ellipseStart||selected>=0||drawMode||ellipseMode){current=[];ellipseStart=null;ellipsePreview=null;dragState=null;selected=-1;clearModes();draw();return}copyPrevHoles()});
document.addEventListener('keydown',e=>{if(e.key.toLowerCase()==='h')setClass(0);if(e.key.toLowerCase()==='s')setClass(1);if(e.key==='Enter'){e.preventDefault();saveNext()}if(e.key==='Delete')removeSelected()});load();
</script></html>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="复用旧 hinge 掩膜，使用 SAM3 辅助标注第二类目标")
    parser.add_argument("--source", type=Path, required=True, help="已有图片目录，或包含 images/train、images/val、images/test 的 YOLO 数据集根目录")
    parser.add_argument("--hinge-label-dir", type=Path, default=None, help="原始 images_all 对应的旧 hinge 标签目录；传标准 YOLO 数据集根目录时无需指定")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--secondary-class-name", choices=("shaft", "hole"), default="shaft", help="第二类轮廓的名称；默认 shaft，头部数据标 hole 时设为 hole")
    parser.add_argument("--sam-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt")
    parser.add_argument("--shaft-prompt", default="the exposed cylindrical metal pin at the bottom tip of the hinge")
    parser.add_argument("--secondary-prompt", default=None, help="SAM 第二类提示词；不指定时按类别使用默认提示词")
    parser.add_argument("--sam-threshold", type=float, default=0.10)
    parser.add_argument("--max-shaft-proposals", type=int, default=3, help="每帧最多保留的 SAM shaft 建议数")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8774)
    return parser.parse_args()


def read_polygon_labels(path: Path, width: int, height: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) < 7 or len(values[1:]) % 2:
            continue
        try:
            class_id = int(values[0])
            coords = [float(v) for v in values[1:]]
        except ValueError:
            continue
        points = [[coords[i] * width, coords[i + 1] * height] for i in range(0, len(coords), 2)]
        result.append({"class_id": class_id, "points": points})
    return result


def filter_shaft_proposals(
    polygons: list[list[list[float]]],
    scores: list[float],
    hinge_polygons: list[dict[str, Any]],
    *,
    width: int,
    height: int,
    maximum: int,
) -> tuple[list[list[list[float]]], list[float]]:
    """Reject text-prompt false positives outside/over the visible hinges."""
    if not hinge_polygons:
        return [], []
    hinge_mask = np.zeros((height, width), dtype=np.uint8)
    hinge_areas: list[float] = []
    for item in hinge_polygons:
        if int(item.get("class_id", -1)) != 0:
            continue
        points = np.asarray(item.get("points", []), dtype=np.float32).reshape(-1, 1, 2)
        if len(points) < 3:
            continue
        mask = np.zeros_like(hinge_mask)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        hinge_mask |= mask
        hinge_areas.append(float(mask.sum()))
    if not hinge_areas:
        return [], []
    # A pin is at the hinge tip; allow a small amount outside the old mask.
    near_hinge = cv2.dilate(hinge_mask, np.ones((61, 61), dtype=np.uint8), iterations=1)
    candidates: list[tuple[float, list[list[float]]]] = []
    for polygon, score in zip(polygons, scores):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        if len(points) < 3:
            continue
        mask = np.zeros_like(hinge_mask)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 1)
        area = float(mask.sum())
        if area < 20:
            continue
        overlap = float((mask & hinge_mask).sum()) / area
        moments = cv2.moments(mask, binaryImage=True)
        if moments["m00"] <= 0:
            continue
        cx = int(round(moments["m10"] / moments["m00"]))
        cy = int(round(moments["m01"] / moments["m00"]))
        if not (0 <= cx < width and 0 <= cy < height) or not near_hinge[cy, cx]:
            continue
        # Whole-hinge masks are not useful shaft suggestions.
        if any(area > hinge_area * 0.45 for hinge_area in hinge_areas):
            continue
        candidates.append((float(score), polygon))
    candidates.sort(key=lambda item: item[0], reverse=True)
    candidates = candidates[: max(1, int(maximum))]
    return [item[1] for item in candidates], [round(item[0], 4) for item in candidates]


class State:
    def __init__(self, items: list[tuple[Path, Path | None, str]], out_dir: Path, secondary_class_name: str) -> None:
        self.items = items
        self.out_dir = out_dir
        self.secondary_class_name = secondary_class_name
        self.lock = threading.RLock()
        self.polygons: dict[int, list[dict[str, Any]]] = {}
        self.meta: list[dict[str, int | str]] = []
        (out_dir / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / "labels").mkdir(parents=True, exist_ok=True)
        (out_dir / "classes.txt").write_text(f"hinge\n{secondary_class_name}\n", encoding="utf-8")
        (out_dir / "dataset_info.json").write_text(
            json.dumps({"class_names": ["hinge", secondary_class_name], "source": f"existing left-camera images with SAM-assisted {secondary_class_name} annotation"}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if any(split in {"train", "val", "test"} for _, _, split in items):
            split_yaml = "train: images/train\nval: images/val\ntest: images/test"
        else:
            split_yaml = "train: images/all\nval: images/all\ntest: images/all"
        (out_dir / "data.yaml").write_text(
            f"path: {out_dir}\n{split_yaml}\nnames:\n  0: hinge\n  1: {secondary_class_name}\n",
            encoding="utf-8",
        )
        for i, (image, hinge_label, split) in enumerate(items):
            frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if frame is None:
                raise RuntimeError(f"无法读取图像：{image}")
            h, w = frame.shape[:2]
            name = f"{i:06d}_{image.name}"
            destination = out_dir / "images" / split / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                cv2.imwrite(str(destination), frame)
            relative_name = f"{split}/{name}"
            self.meta.append({"width": w, "height": h, "name": relative_name, "split": split, "source": str(image)})
            label_path = out_dir / "labels" / split / f"{Path(name).stem}.txt"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            shape_path = label_path.with_suffix(".json")
            if shape_path.is_file():
                try:
                    saved = json.loads(shape_path.read_text(encoding="utf-8"))
                    self.polygons[i] = saved if isinstance(saved, list) else read_polygon_labels(label_path, w, h)
                except (OSError, json.JSONDecodeError):
                    self.polygons[i] = read_polygon_labels(label_path, w, h)
            elif label_path.is_file():
                self.polygons[i] = read_polygon_labels(label_path, w, h)
            elif hinge_label is not None:
                self.polygons[i] = read_polygon_labels(hinge_label, w, h)
            else:
                self.polygons[i] = []

    def frame(self, index: int) -> dict[str, Any]:
        index = max(0, min(len(self.items) - 1, index))
        with self.lock:
            meta = self.meta[index]
            return {"index": index, "total": len(self.items), "file": meta["name"], "width": meta["width"], "height": meta["height"], "polygons": self.polygons.get(index, [])}

    def save(self, index: int, polygons: list[dict[str, Any]]) -> dict[str, Any]:
        with self.lock:
            if index < 0 or index >= len(self.items):
                raise ValueError("帧号无效")
            meta = self.meta[index]
            width, height = int(meta["width"]), int(meta["height"])
            lines = []
            clean: list[dict[str, Any]] = []
            for item in polygons:
                class_id = int(item.get("class_id", -1))
                points = item.get("points", [])
                if class_id not in (0, 1) or not isinstance(points, list) or len(points) < 3:
                    continue
                coords = []
                normalized = []
                for point in points:
                    x, y = float(point[0]), float(point[1])
                    if not np.isfinite(x) or not np.isfinite(y):
                        raise ValueError("轮廓坐标无效")
                    x, y = max(0.0, min(width - 1.0, x)), max(0.0, min(height - 1.0, y))
                    normalized.append([x, y])
                    coords.extend((f"{x / width:.8f}", f"{y / height:.8f}"))
                saved_item = {"class_id": class_id, "points": normalized}
                if item.get("shape") == "ellipse":
                    for key in ("cx", "cy", "rx", "ry"):
                        value = float(item.get(key, 0.0))
                        if not np.isfinite(value):
                            raise ValueError("椭圆参数无效")
                        saved_item[key] = value
                    saved_item["shape"] = "ellipse"
                clean.append(saved_item)
                lines.append(str(class_id) + " " + " ".join(coords))
            path = self.out_dir / "labels" / str(meta["split"]) / f"{Path(str(meta['name'])).stem}.txt"
            path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            path.with_suffix(".json").write_text(json.dumps(clean, ensure_ascii=False), encoding="utf-8")
            self.polygons[index] = clean
            return {"ok": True, "file": str(meta["name"]), "labels": len(lines)}


def run() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    if not source.is_dir():
        print(f"数据目录不存在：{source}", file=sys.stderr)
        return 2
    if not 0 < args.port < 65536 or not 0 <= args.sam_threshold <= 1 or args.max_shaft_proposals < 1:
        print("端口或 SAM 阈值无效", file=sys.stderr)
        return 2
    items: list[tuple[Path, Path | None, str]] = []
    if (source / "images").is_dir():
        # Standard YOLO dataset: preserve its train/val/test split.
        for split in ("train", "val", "test"):
            image_dir = source / "images" / split
            if not image_dir.is_dir():
                continue
            label_dir = source / "labels" / split
            for image in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
                label = label_dir / f"{image.stem}.txt"
                if not label.is_file():
                    print(f"缺少对应 hinge 标签：{label}", file=sys.stderr)
                    return 2
                items.append((image, label, split))
    else:
        images = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        hinge_dir = args.hinge_label_dir.expanduser().resolve() if args.hinge_label_dir else None
        if hinge_dir is None or not hinge_dir.is_dir():
            print("原始图片目录需要 --hinge-label-dir；标准 YOLO 数据集可直接传根目录", file=sys.stderr)
            return 2
        items = [(image, hinge_dir / f"{image.stem}.txt", "all") for image in images]
    if not items:
        print(f"数据目录中没有图片：{source}", file=sys.stderr)
        return 2
    try:
        default_prompt = args.secondary_prompt or (
            "the visible circular hole in the hinge" if args.secondary_class_name == "hole" else args.shaft_prompt
        )
        sam = SamAssistant(args.sam_checkpoint.expanduser().resolve(), threshold=args.sam_threshold, default_prompt=default_prompt)
        state = State(items, args.out_dir.expanduser().resolve(), args.secondary_class_name)
    except Exception as exc:
        print(f"初始化失败：{exc}", file=sys.stderr)
        return 2

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, value: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status); self.send_header("Content-Type", "application/json; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in {"/", "/index.html"}:
                body = PAGE.replace("__SECONDARY_NAME__", args.secondary_class_name).encode(); self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            if path.startswith("/api/frame/"):
                try: self.send_json(state.frame(int(path.rsplit("/", 1)[1])))
                except Exception as exc: self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            if path.startswith("/image/"):
                name = Path(path.removeprefix("/image/")); image = state.out_dir / "images" / name
                if not image.is_file(): self.send_error(404); return
                body = image.read_bytes(); self.send_response(200); self.send_header("Content-Type", "image/jpeg"); self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0")); payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/api/save": self.send_json(state.save(int(payload["index"]), payload.get("polygons", []))); return
                if self.path == "/api/propose":
                    index = int(payload["index"]); meta = state.meta[index]; jpeg = cv2.imencode(".jpg", cv2.imread(str(state.out_dir / "images" / str(meta["name"])), cv2.IMREAD_COLOR))[1].tobytes()
                    polygons, scores = sam.propose(jpeg, prompt=str(payload.get("prompt") or default_prompt))
                    polygons, scores = filter_shaft_proposals(polygons, scores, state.polygons.get(index, []), width=int(meta["width"]), height=int(meta["height"]), maximum=args.max_shaft_proposals)
                    self.send_json({"polygons": polygons, "scores": scores}); return
                self.send_json({"error": "unknown endpoint"}, HTTPStatus.NOT_FOUND)
            except Exception as exc: self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, _format: str, *_args: object) -> None: return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"标注页：http://{args.host}:{args.port}"); print(f"输出目录：{state.out_dir}"); print(f"图片数：{len(items)}；Ctrl-C 停止")
    try: server.serve_forever()
    except KeyboardInterrupt: return 0
    finally: server.server_close()


if __name__ == "__main__":
    raise SystemExit(run())
