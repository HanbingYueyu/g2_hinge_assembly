"""Crop a full right-hand camera image and detect each screw in the fixed ROI."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import cv2
from .right_plate_screw_detection import MODEL_PATH, detect, read_camera


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=MODEL_PATH)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', type=Path)
    source.add_argument('--camera-url', help='Existing DDS endpoint; does not start a service')
    parser.add_argument('--confidence', type=float, default=.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.confidence < 1:
        parser.error('--confidence must be between 0 and 1')
    if args.image:
        image = cv2.imread(str(args.image))
    else:
        image, _ = read_camera(args.camera_url)
    from ultralytics import YOLO
    crop, preview, data = detect(YOLO(args.model), image, args.confidence)
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.imwrite(str(args.output / 'source.jpg'), image)
    cv2.imwrite(str(args.output / 'crop.jpg'), crop)
    cv2.imwrite(str(args.output / 'detections.jpg'), preview)
    data['model'] = args.model
    (args.output / 'detections.json').write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data, ensure_ascii=False))


if __name__ == '__main__':
    main()
