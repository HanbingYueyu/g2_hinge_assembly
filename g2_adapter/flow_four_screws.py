"""Full-flow adapter for the standalone YOLO + VLM screw review."""
import threading
from pathlib import Path
import cv2
import numpy as np
from .four_screws_20261006_client import ASSETS, FourScrews20261006Client
from .live_four_screws_vlm import boxed_screw_frame, crop_frame

ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()
_MODEL = None


def run_check(base_client, frames):
    if not frames:
        raise ValueError('需要当前右手图像')
    frame = cv2.imdecode(np.frombuffer(frames[-1], np.uint8), cv2.IMREAD_COLOR)
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from ultralytics import YOLO
            _MODEL = YOLO(str(ROOT / 'models/right_plate_screw_yolo_20261007/best.pt'))
        annotated, count = boxed_screw_frame(frame, _MODEL, with_count=True)
    crop = crop_frame(annotated, (430, 400, 870, 830))
    client = FourScrews20261006Client.__new__(FourScrews20261006Client)
    for name in ('mode', 'model', 'api_url', 'timeout_s'):
        setattr(client, name, getattr(base_client, name))
    client.prompt = (ASSETS / 'prompt.txt').read_text(encoding='utf-8').strip()
    client.references = ((ASSETS / 'reference_zero.jpg').read_bytes(),
                         (ASSETS / 'reference_four.jpg').read_bytes())
    inputs = client.prepare_images([crop])
    from .vlm_flow_checks import _save_inputs
    statuses = {'yolo_count': count, 'crop_xyxy': [430, 400, 870, 830]}
    _save_inputs('vlm_four_screws_yolo', dict(zip(
        ('right_reference_zero', 'right_reference_four', 'right_yolo_crop'), inputs)), statuses)
    import requests
    with requests.Session() as session:
        session.trust_env = False
        client.session = session
        result = client.run_bytes('four-screws', inputs, yolo_count=count)
    return {**result, 'yolo_count': count, 'input_status': statuses}
