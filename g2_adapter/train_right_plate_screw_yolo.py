"""Train an independent single-class plate-screw detector; no camera or robot access."""
from __future__ import annotations
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--model', default='models/two_hole_head_rgbd_yolo_v2/best.pt')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--imgsz', type=int, default=1280)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--project', default='runs/right_plate_screw_yolo')
    parser.add_argument('--name', default='20261007_v1')
    args = parser.parse_args()
    from ultralytics import YOLO
    model = YOLO(args.model)
    model.train(data=str(args.data.resolve()), task='detect', epochs=args.epochs,
                imgsz=args.imgsz, batch=args.batch, device=0, workers=4,
                project=str(Path(args.project).resolve()), name=args.name, exist_ok=False,
                optimizer='AdamW', lr0=.001, lrf=.01, weight_decay=.0005,
                cos_lr=True, patience=25, seed=42, deterministic=True, amp=True,
                rect=False, mosaic=0., mixup=0., copy_paste=0.,
                degrees=0., translate=0., scale=0., fliplr=0., flipud=0.,
                hsv_h=.015, hsv_s=.2, hsv_v=.15)
    print('BEST WEIGHTS:', Path(model.trainer.save_dir) / 'weights/best.pt')


if __name__ == '__main__':
    main()
