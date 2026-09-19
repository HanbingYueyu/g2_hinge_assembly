#!/usr/bin/env python3
"""Read-only browser collector for a native G2 hinge YOLO-seg dataset.

The operator freezes a current GDK colour frame in a browser and draws a
polygon around *every* visible hinge.  Saved labels are standard one-class
YOLO segmentation labels.  No SAM proposal is used as ground truth and this
program deliberately has no Robot, motion, mode, or gripper API.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

try:
    from .camera import decode_color_rgb
except ImportError:
    from camera import decode_color_rgb


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PAGE = """<!doctype html>
<html lang='zh-CN'><meta charset='utf-8'><title>G2 左 D405 铰链数据标注（只读）</title>
<style>
body { background:#171717; color:#eee; font:16px sans-serif; margin:16px; }
button { margin:2px; padding:8px 11px; } #status { margin:10px 0; color:#9ef; }
canvas { max-width:96vw; max-height:78vh; border:1px solid #777; cursor:crosshair; display:block; }
</style><body>
<h2>G2 左 D405 铰链数据标注（只读）</h2>
<p>点击“冻结当前帧”后会自动运行一次 SAM；请检查并修正建议，再保存。一帧中所有铰链都要标。无铰链/干扰画面可直接保存为空标签。</p>
<button onclick='freezeFrame()'>冻结当前帧并自动 SAM</button>
<input id='prompt' size='28' value='silver triangular hinge'><button onclick='samPropose()'>SAM 生成建议</button>
<button onclick='finishPolygon()'>完成当前实例</button><button onclick='deleteLastPolygon()'>删除最后实例</button>
<button onclick='undoPoint()'>撤销当前点</button><button onclick='clearAnnotations()'>清空标注</button><button onclick='discardFrame()'>丢弃，回实时</button>
<button onclick='saveSample()'>保存这一帧</button>
<div id='status'>连接实时画面中…</div><canvas id='canvas'></canvas>
<script>
const canvas=document.getElementById('canvas'), ctx=canvas.getContext('2d'), status=document.getElementById('status');
let image=new Image(), frozen=false, polygons=[], current=[], notice='';
function text() { const state = frozen ? `已冻结：完成实例=${polygons.length}，当前点=${current.length}。保存前请确认每个铰链均已标注。` : '实时画面：点击“冻结当前帧”后才能标注。'; status.textContent = notice ? `${notice}　${state}` : state; }
function redraw() {
  if (!image.naturalWidth) return;
  canvas.width=image.naturalWidth; canvas.height=image.naturalHeight; ctx.drawImage(image,0,0);
  const draw=(points, color, close) => { if (!points.length) return; ctx.strokeStyle=color; ctx.fillStyle=color; ctx.lineWidth=3; ctx.beginPath(); ctx.moveTo(points[0][0],points[0][1]); for(const p of points.slice(1))ctx.lineTo(p[0],p[1]); if(close&&points.length>=3)ctx.closePath(); ctx.stroke(); for(const p of points){ctx.beginPath();ctx.arc(p[0],p[1],4,0,Math.PI*2);ctx.fill();} };
  polygons.forEach(p=>draw(p,'#00ff59',true)); draw(current,'#ffbd22',false); text();
}
function load(url) { image.onload=redraw; image.src=url+'?t='+Date.now(); }
async function post(url, value={}) { const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)}); const v=await r.json(); if(!r.ok)throw Error(v.error||'request failed'); return v; }
async function freezeFrame(){ try { status.textContent='冻结并运行 SAM 推理中，请等待…'; const out=await post('/freeze',{prompt:document.getElementById('prompt').value}); frozen=true; polygons=out.polygons||[]; current=[]; notice=out.sam_error ? `已冻结，但 SAM 建议失败：${out.sam_error}，请手工标注。` : `已冻结并生成 ${polygons.length} 个 SAM 建议；请人工检查后保存。`; load('/frozen.jpg'); } catch(e){status.textContent='冻结失败：'+e.message;} }
function finishPolygon(){ if(!frozen)return; if(current.length<3){status.textContent='当前实例至少需要 3 个点。';return;} polygons.push(current); current=[]; notice='已完成一个手工实例。'; redraw(); }
function deleteLastPolygon(){ if(!frozen)return; polygons.pop(); redraw(); }
function undoPoint(){ if(!frozen)return; current.pop(); redraw(); }
function clearAnnotations(){ if(!frozen)return; polygons=[];current=[];redraw(); }
async function samPropose(){ if(!frozen){status.textContent='请先冻结一帧。';return;} try { status.textContent='SAM 推理中，请等待…'; const out=await post('/sam-propose',{prompt:document.getElementById('prompt').value}); polygons=out.polygons; current=[]; notice=`SAM 给出 ${out.polygons.length} 个已完成建议（${out.scores.join(', ')}）。确认无误后可直接点“保存这一帧”，无需再点“完成当前实例”。`; redraw(); } catch(e){status.textContent='SAM 建议失败：'+e.message;} }
async function discardFrame(){ try{await post('/discard'); frozen=false; polygons=[];current=[];notice='已丢弃冻结帧，未保存任何数据。';load('/frame.jpg');}catch(e){status.textContent='丢弃失败：'+e.message;} }
async function saveSample(){ if(!frozen){status.textContent='请先冻结一帧。';return;} if(current.length){status.textContent='请先完成当前实例，或撤销其点。';return;} try { const out=await post('/save',{polygons}); frozen=false; polygons=[]; notice=`保存成功：${out.stem}，${out.instances} 个铰链；累计正样本帧=${out.positive_frames}，负样本帧=${out.negative_frames}。`; load('/frame.jpg'); } catch(e){status.textContent='保存失败：'+e.message;} }
canvas.addEventListener('click', e=>{ if(!frozen||!image.naturalWidth)return; const r=canvas.getBoundingClientRect(); current.push([(e.clientX-r.left)*canvas.width/r.width,(e.clientY-r.top)*canvas.height/r.height]); redraw(); });
setInterval(()=>{if(!frozen)load('/frame.jpg');},350); load('/frame.jpg');
</script></body></html>"""


class CaptureStore:
    def __init__(self, out_dir: Path, *, target_name: str, source: str = "G2 left D405 native colour frames") -> None:
        self.out_dir = out_dir
        self.image_dir = out_dir / "images_all"
        self.label_dir = out_dir / "labels_all"
        self.manifest_path = out_dir / "manifest.jsonl"
        self.target_name = target_name
        self.lock = threading.RLock()
        self.latest_jpeg: bytes | None = None
        self.latest_timestamp_ns: int | None = None
        self.frozen_jpeg: bytes | None = None
        self.frozen_timestamp_ns: int | None = None
        self.frozen_shape: tuple[int, int] | None = None
        self.next_index = self._next_index()
        self.positive, self.negative = self._counts()
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.label_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "dataset_info.json").write_text(
            json.dumps({"class_names": [target_name], "source": source}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _next_index(self) -> int:
        indices: list[int] = []
        for path in self.out_dir.glob("images_all/frame_*.jpg"):
            try:
                indices.append(int(path.stem.rsplit("_", 1)[1]))
            except ValueError:
                continue
        return max(indices, default=0) + 1

    def _counts(self) -> tuple[int, int]:
        positive = negative = 0
        for label in self.out_dir.glob("labels_all/frame_*.txt"):
            if label.read_text(encoding="utf-8").strip():
                positive += 1
            else:
                negative += 1
        return positive, negative

    def update_latest(self, jpeg: bytes, timestamp_ns: int) -> None:
        with self.lock:
            self.latest_jpeg = jpeg
            self.latest_timestamp_ns = timestamp_ns

    def freeze(self) -> None:
        with self.lock:
            if self.latest_jpeg is None:
                raise RuntimeError("尚未收到彩色相机图像")
            image = cv2.imdecode(np.frombuffer(self.latest_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("无法解码当前相机图像")
            self.frozen_jpeg = self.latest_jpeg
            self.frozen_timestamp_ns = self.latest_timestamp_ns
            self.frozen_shape = image.shape[:2]

    def discard(self) -> None:
        with self.lock:
            self.frozen_jpeg = None
            self.frozen_timestamp_ns = None
            self.frozen_shape = None

    def _labels_for(self, polygons: Any) -> tuple[str, int]:
        if not isinstance(polygons, list):
            raise ValueError("polygons 必须是数组")
        if self.frozen_shape is None:
            raise RuntimeError("没有已冻结的图像")
        height, width = self.frozen_shape
        lines: list[str] = []
        for polygon in polygons:
            if not isinstance(polygon, list) or len(polygon) < 3:
                raise ValueError("每个铰链实例至少需要 3 个顶点")
            points: list[str] = []
            for point in polygon:
                if not isinstance(point, list) or len(point) != 2:
                    raise ValueError("顶点格式必须是 [x, y]")
                x, y = float(point[0]), float(point[1])
                if not np.isfinite(x) or not np.isfinite(y) or not (0 <= x < width and 0 <= y < height):
                    raise ValueError("顶点在图像范围外")
                points.extend((f"{x / width:.8f}", f"{y / height:.8f}"))
            lines.append("0 " + " ".join(points))
        return ("\n".join(lines) + ("\n" if lines else ""), len(lines))

    def save(self, polygons: Any) -> dict[str, Any]:
        with self.lock:
            if self.frozen_jpeg is None:
                raise RuntimeError("请先冻结一帧")
            labels, count = self._labels_for(polygons)
            stem = f"frame_{self.next_index:06d}"
            image_path = self.image_dir / f"{stem}.jpg"
            label_path = self.label_dir / f"{stem}.txt"
            image_path.write_bytes(self.frozen_jpeg)
            label_path.write_text(labels, encoding="utf-8")
            record = {
                "index": self.next_index,
                "timestamp_unix": time.time(),
                "gdk_color_timestamp_ns": self.frozen_timestamp_ns,
                "annotation": "positive" if count else "negative",
                "instances": count,
                "image_path": str(image_path),
                "yolo_seg_label_path": str(label_path),
                "image_shape_hw": list(self.frozen_shape or ()),
            }
            with self.manifest_path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.next_index += 1
            self.positive += int(count > 0)
            self.negative += int(count == 0)
            self.discard()
            return {"stem": stem, "instances": count, "positive_frames": self.positive, "negative_frames": self.negative}


class SamAssistant:
    """Generate editable polygon proposals for a frozen frame only."""

    def __init__(self, checkpoint: Path, *, threshold: float, default_prompt: str) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SAM 标注需要 CUDA")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"找不到 SAM3 权重：{checkpoint}")
        self.lock = threading.Lock()
        self.default_prompt = default_prompt
        self.processor = Sam3Processor(
            build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False),
            confidence_threshold=float(threshold),
        )

    @staticmethod
    def _polygon(mask: np.ndarray) -> list[list[float]] | None:
        binary = np.asarray(mask, dtype=np.uint8).squeeze()
        found = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = found[0] if len(found) == 2 else found[1]
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 200.0:
            return None
        epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(points) < 3:
            return None
        return [[float(x), float(y)] for x, y in points]

    def propose(self, jpeg: bytes, *, prompt: str) -> tuple[list[list[list[float]]], list[float]]:
        bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError("无法解码冻结图像")
        prompt = str(prompt).strip() or self.default_prompt
        if not prompt:
            raise ValueError("SAM 提示词不能为空")
        with self.lock, torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            state = self.processor.set_image(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            output = self.processor.set_text_prompt(prompt, state)
        masks = [np.asarray(mask, dtype=bool).squeeze() for mask in output["masks"].detach().cpu().numpy()]
        scores = output["scores"].detach().float().cpu().numpy().reshape(-1)
        if len(masks) != len(scores):
            raise RuntimeError("SAM 返回的掩膜和置信度数量不一致")
        proposals: list[tuple[list[list[float]], float]] = []
        for mask, score in zip(masks, scores):
            polygon = self._polygon(mask)
            if polygon is not None:
                proposals.append((polygon, float(score)))
        proposals.sort(key=lambda item: item[1], reverse=True)
        return [item[0] for item in proposals], [round(item[1], 4) for item in proposals]


def handler_class(store: CaptureStore, sam_assistant: SamAssistant, *, camera_label: str = "左 D405"):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, data: dict[str, Any]) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _jpeg(self, value: bytes | None) -> None:
            if value is None:
                self.send_error(503, "image is not ready")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.split("?", 1)[0] == "/":
                body = PAGE.replace("G2 左 D405", f"G2 {camera_label}").encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            with store.lock:
                if self.path.startswith("/frame.jpg"):
                    self._jpeg(store.latest_jpeg)
                    return
                if self.path.startswith("/frozen.jpg"):
                    self._jpeg(store.frozen_jpeg)
                    return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/freeze":
                    store.freeze()
                    try:
                        with store.lock:
                            jpeg = store.frozen_jpeg
                        polygons, scores = sam_assistant.propose(
                            jpeg,
                            prompt=payload.get("prompt", sam_assistant.default_prompt),
                        )
                        self._json(200, {"ok": True, "polygons": polygons, "scores": scores})
                    except (ValueError, RuntimeError) as exc:
                        # A frozen frame is still useful for manual annotation
                        # when SAM is temporarily unable to produce proposals.
                        self._json(200, {"ok": True, "polygons": [], "scores": [], "sam_error": str(exc)})
                elif self.path == "/discard":
                    store.discard()
                    self._json(200, {"ok": True})
                elif self.path == "/save":
                    self._json(200, store.save(payload.get("polygons")))
                elif self.path == "/sam-propose":
                    with store.lock:
                        if store.frozen_jpeg is None:
                            raise RuntimeError("请先冻结一帧")
                        jpeg = store.frozen_jpeg
                    polygons, scores = sam_assistant.propose(jpeg, prompt=payload.get("prompt", sam_assistant.default_prompt))
                    self._json(200, {"polygons": polygons, "scores": scores})
                else:
                    self._json(404, {"error": "unknown endpoint"})
            except (ValueError, RuntimeError, json.JSONDecodeError) as exc:
                self._json(400, {"error": str(exc)})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2 彩色相机的只读 YOLO-seg 数据采集与 SAM 浏览器辅助标注。")
    parser.add_argument("--camera", choices=("left", "head"), default="left", help="采集左手 D405 或头部彩色相机")
    parser.add_argument("--out-dir", type=Path, default=None, help="默认按 --camera 使用独立数据目录")
    parser.add_argument("--target-name", default="hinge")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--sam-checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt")
    parser.add_argument("--sam-prompt", default="silver triangular hinge")
    parser.add_argument("--sam-threshold", type=float, default=0.10)
    return parser.parse_args()


def run() -> int:
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = PROJECT_ROOT / "datasets" / (
            "head_hinge_yolo_raw_g2_v1" if args.camera == "head"
            else "left_d405_hinge_yolo_raw_g2_v1"
        )
    if args.warmup < 0 or not 0 < args.port < 65536 or not 0 <= args.sam_threshold <= 1:
        print("参数错误：warmup、port 或 SAM 阈值无效。", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2
    print(f"加载 SAM3 标注助手：{args.sam_checkpoint.resolve()}", flush=True)
    try:
        sam_assistant = SamAssistant(args.sam_checkpoint.resolve(), threshold=args.sam_threshold, default_prompt=args.sam_prompt)
    except Exception as exc:
        print(f"SAM 标注助手启动失败：{exc}", file=sys.stderr)
        return 2
    camera_label = "头部彩色相机" if args.camera == "head" else "左 D405"
    store = CaptureStore(
        args.out_dir.resolve(), target_name=args.target_name,
        source=f"G2 {camera_label} native colour frames",
    )
    server = ThreadingHTTPServer(
        (args.host, args.port), handler_class(store, sam_assistant, camera_label=camera_label),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"浏览器标注页：http://{args.host}:{args.port}")
    print(f"输出目录：{args.out_dir.resolve()}")
    print("SAM 仅生成可编辑建议，绝不自动保存为训练真值。")
    print("只读采集：只创建 GDK Camera，不创建 Robot、不发送运动、模式或夹爪命令。")
    initialized = False
    camera = None
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        color_type = (
            agibot_gdk.CameraType.kHeadColor
            if args.camera == "head"
            else agibot_gdk.CameraType.kHandLeftColor
        )
        camera = agibot_gdk.Camera([color_type])
        for _ in range(args.warmup):
            camera.get_latest_image(color_type, 1000.0)
        while True:
            image = camera.get_latest_image(color_type, 1000.0)
            rgb = decode_color_rgb(image, agibot_gdk)
            ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise RuntimeError(f"无法编码{camera_label}图像")
            store.update_latest(jpeg.tobytes(), int(image.timestamp_ns))
    except KeyboardInterrupt:
        print("收到 Ctrl-C，停止只读数据采集。")
        return 0
    except Exception as exc:
        print(f"{camera_label}数据采集失败：{exc}", file=sys.stderr)
        return 1
    finally:
        server.shutdown()
        server.server_close()
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(run())
