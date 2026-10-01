#!/usr/bin/env python3
"""Train two-hole detection using only GDK head RGB-D color frames."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("datasets/two_hole_head_rgbd_yolo_v1_trainval/data.yaml"))
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--project", type=Path, default=Path("runs/two_hole_head_rgbd_yolo"))
    parser.add_argument("--name", default="v1_960")
    args = parser.parse_args()
    if not args.data.is_file():
        parser.error(f"Dataset config does not exist: {args.data}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(
        task="detect", data=str(args.data.resolve()), epochs=args.epochs,
        imgsz=args.imgsz, batch=args.batch, device=0, workers=4,
        project=str(args.project.resolve()), name=args.name,
        pretrained=True, patience=25, seed=42, deterministic=True,
        optimizer="AdamW", lr0=0.001, lrf=0.01, weight_decay=0.0005,
        cos_lr=True, amp=True, rect=True,
        mosaic=0.0, mixup=0.0, copy_paste=0.0,
        degrees=0.0, translate=0.0, scale=0.0,
        fliplr=0.0, flipud=0.0,
        hsv_h=0.015, hsv_s=0.2, hsv_v=0.15,
    )
    print(f"Best weights: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
