"""Shared fixed-ROI screw detection; reads images only."""
from __future__ import annotations

import base64
import json
from urllib.request import urlopen

import cv2
import numpy as np

ROI = (525, 435, 780, 815)  # left, top, right-exclusive, bottom-exclusive
IMAGE_SHAPE = (1056, 1280)
INFERENCE_SIZE = 384
CAMERA_URL = 'http://127.0.0.1:5001/get_all_images?indices=18'
MODEL_PATH = 'models/right_plate_screw_yolo_20261007/best.pt'


def read_camera(url):
    with urlopen(url, timeout=5) as response:
        payload = json.load(response)
    entries = [item for item in payload['images']
               if item.get('topic') == '/camera/hand_right_color']
    if (len(entries) != 1 or not entries[0].get('available')
            or entries[0].get('age_ms', 1e9) > 500):
        raise RuntimeError('No fresh right-hand color frame')
    entry = entries[0]
    encoded = base64.b64decode(entry['image'].split(',', 1)[1])
    image = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    return image, {key: entry.get(key) for key in ('source_timestamp', 'age_ms')}


def detect(model, image, confidence=.5):
    if image is None or image.shape[:2] != IMAGE_SHAPE:
        raise ValueError('Expected a full 1280x1056 right-hand image before ROI cropping')
    if model.names != {0: 'plate_screw'}:
        raise ValueError(f'Expected the plate_screw model, got {model.names}')
    x0, y0, x1, y1 = ROI
    crop = image[y0:y1, x0:x1].copy()
    result = model.predict(crop, imgsz=INFERENCE_SIZE, conf=confidence,
                           iou=.5, max_det=100, verbose=False)[0]
    detections = []
    for box in result.boxes:
        local = box.xyxy[0].tolist()
        full = [local[0]+x0, local[1]+y0, local[2]+x0, local[3]+y0]
        detections.append(dict(class_id=int(box.cls.item()),
                               confidence=float(box.conf.item()),
                               bbox_xyxy=full, bbox_crop_xyxy=local))
    return crop, result.plot(labels=False, line_width=2), dict(
        count=len(detections), detections=detections, crop_xyxy=list(ROI),
        crop_width=x1-x0, crop_height=y1-y0, imgsz=INFERENCE_SIZE,
        bbox_coordinate_system='bbox_xyxy: full image; bbox_crop_xyxy: ROI image')
