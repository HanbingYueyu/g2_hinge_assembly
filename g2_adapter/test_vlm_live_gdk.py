"""One-shot live VLM checks, without any robot motion commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import cv2

try:
    from .camera import decode_color_rgb
    from .vlm_flow_checks import _safe_check, _save_inputs, crop_jpeg, parse_crop_xyxy, start_vlm_run
except ImportError:
    from camera import decode_color_rgb
    from vlm_flow_checks import _safe_check, _save_inputs, crop_jpeg, parse_crop_xyxy, start_vlm_run

ROOT = Path(__file__).resolve().parents[1]


def read_http_jpeg(url):
    with urlopen(url, timeout=3) as response:
        image = response.read()
    if not image:
        raise RuntimeError("Web has no JPEG frame")
    return image


def capture_gdk(names):
    import agibot_gdk as gdk
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK 初始化失败")
    camera = None
    try:
        types = {"head": gdk.CameraType.kHeadColor, "left": gdk.CameraType.kHandLeftColor, "right": gdk.CameraType.kHandRightColor}
        camera = gdk.Camera([types[name] for name in names])
        frames = {}
        for name in names:
            for attempt in range(3):
                try:
                    frame = camera.get_latest_image(types[name], 1000.0)
                    bgr = cv2.cvtColor(decode_color_rgb(frame, gdk), cv2.COLOR_RGB2BGR)
                    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise RuntimeError(f"{name} JPEG 编码失败")
                    frames[name] = encoded.tobytes()
                    break
                except Exception:
                    if attempt == 2:
                        raise
        return frames
    finally:
        try:
            if camera is not None:
                camera.close_camera()
        finally:
            gdk.gdk_release()


def acquire_frames(names, web_url):
    if web_url:
        try:
            frames = {name: read_http_jpeg(f"{web_url.rstrip('/')}/raw/{name}.jpg") for name in names}
            return frames, "web"
        except HTTPError:
            # A running server with unavailable images must not trigger competing camera reads.
            raise
        except URLError:
            print("[取图] Web 不可连接，使用 GDK 实时相机")
    return capture_gdk(names), "gdk"


def run_task(task, frames, source, client, crop):
    statuses = {name: {"state": f"live_{source}"} for name in frames}
    if task == "gripper":
        crop_box = parse_crop_xyxy(crop) if isinstance(crop, str) else crop
        images = [frames['left'], crop_jpeg(frames['left'], crop_box)]
        _save_inputs("vlm_gripper_check", {'left': images[0], 'left_crop': images[1]}, statuses)
    else:
        _save_inputs("vlm_assembly_check", frames, statuses)
        order = ("right", "left", "head") if task == "pin" else ("left", "right")
        images = [frames[name] for name in order]
    return _safe_check(client, task, images, statuses)


def main():
    parser = argparse.ArgumentParser(description="实时 VLM 单次测试，不控制机械臂")
    parser.add_argument("task", choices=("gripper", "pin", "c-hole"))
    parser.add_argument("--web-url", default="http://127.0.0.1:8767", help="优先复用 Web；空字符串强制独立相机")
    parser.add_argument("--env-file", type=Path, default=ROOT / "vlm_checks/config/vlm.env")
    parser.add_argument("--model", default="qwen3.8-flash")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--gripper-crop", default=None, help="可选像素框；默认居中保留高度60%、宽度40%")
    args = parser.parse_args()
    start_vlm_run()
    sys.path.insert(0, str(ROOT))
    from vlm_checks.src.vlm_three_checks import VlmClient, DEFAULT_API_URL
    try:
        client = VlmClient(args.env_file, args.model, DEFAULT_API_URL, args.timeout)
        names = ("left",) if args.task == "gripper" else (("right", "left", "head") if args.task == "pin" else ("left", "right"))
        frames, source = acquire_frames(names, args.web_url)
        print(f"[取图] 来源={source}，相机={','.join(names)}；开始 {args.task} 判别")
        record = run_task(args.task, frames, source, client, args.gripper_crop)
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return 0 if record["ok"] else 1
    except KeyboardInterrupt:
        print("已中断。")
        return 130
    except Exception as exc:
        print(f"实时 VLM 测试失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
