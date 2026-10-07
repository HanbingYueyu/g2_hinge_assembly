#!/usr/bin/env python3
"""Standalone single-frame screw review using the full-flow check unchanged."""
import argparse
import json
from pathlib import Path
import sys
import time
from urllib.request import urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from g2_adapter.vlm_flow_checks import run_four_screws_check, start_vlm_run
from vlm_checks.src.vlm_three_checks import VlmClient


def check_once(read_jpeg, client):
    # Use the same check entry point as the full flow, with one captured frame.
    return run_four_screws_check(None, client, frames_override=[read_jpeg()])


def print_summary(record):
    result = record.get('result', {})
    count = result.get('yolo_count')
    yolo = '齐全' if count == 4 else '不齐全' if isinstance(count, int) and 0 <= count < 4 else '不确定'
    labels = {'yes': '齐全', 'no': '不齐全', 'unknown': '不确定'}
    vlm = labels.get(result.get('latest_decision'), '不确定')
    print(f"[YOLO] 螺丝数量：{count if count is not None else '未知'}；判别：{yolo}", flush=True)
    print(f"[VLM] 螺丝数量：{result.get('model_count', '未知')}；判别：{vlm}", flush=True)
    if not record.get('ok'):
        print(f"[判别失败] {record.get('error', '未完成')}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vlm-mode', choices=('local', 'online'), default='local')
    parser.add_argument('--vlm-env-file', type=Path, default=ROOT/'vlm_checks/config/vlm.env')
    parser.add_argument('--vlm-timeout-s', type=float, default=60)
    parser.add_argument('--source-url', help='复用全流程原图，例如 http://127.0.0.1:8767/raw/right')
    parser.add_argument('--repeat', action='store_true', help='循环判别；默认只判别一次')
    args = parser.parse_args(argv)
    if not np.isfinite(args.vlm_timeout_s) or args.vlm_timeout_s <= 0:
        parser.error('--vlm-timeout-s 必须为正且有限')
    client = VlmClient(args.vlm_env_file, None, None, args.vlm_timeout_s, args.vlm_mode)
    directory = start_vlm_run()
    print(f'判别输入图片保存目录：{directory}', flush=True)
    gdk = None
    camera = None
    try:
        if args.source_url:
            def read_jpeg():
                with urlopen(args.source_url, timeout=2) as response:
                    return response.read()
        else:
            import agibot_gdk as gdk_module
            from g2_adapter.camera import decode_color_rgb
            if gdk_module.gdk_init() != gdk_module.GDKRes.kSuccess:
                raise RuntimeError('GDK 初始化失败')
            gdk = gdk_module
            camera = gdk.Camera([gdk.CameraType.kHandRightColor])
            def read_jpeg():
                frame = camera.get_latest_image(gdk.CameraType.kHandRightColor, 300.)
                bgr = cv2.cvtColor(decode_color_rgb(frame, gdk), cv2.COLOR_RGB2BGR)
                ok, encoded = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 96])
                if not ok:
                    raise RuntimeError('右手原图编码失败')
                return encoded.tobytes()
        while True:
            record = check_once(read_jpeg, client)
            print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
            print_summary(record)
            if not args.repeat:
                return 0 if record.get('ok') else 1
            time.sleep(1)
    except KeyboardInterrupt:
        return 0
    finally:
        camera = None
        if gdk is not None:
            gdk.gdk_release()


if __name__ == '__main__':
    raise SystemExit(main())
