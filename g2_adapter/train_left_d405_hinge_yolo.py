#!/usr/bin/env python3
"""Train a G2-native one-class hinge YOLO11 segmentation model."""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 G2 左 D405 原生画面铰链 YOLO11-seg。")
    parser.add_argument("--data", type=Path, default=Path("datasets/left_d405_hinge_yolo_g2_v1/data.yaml"))
    parser.add_argument("--model", default="yolo11n-seg.pt", help="初始权重；也可指定已验证的本地 .pt。")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO 内部 letterbox 尺寸；保留 G2 原图比例。")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--project", type=Path, default=Path("runs/left_d405_hinge_yolo_seg"))
    parser.add_argument("--name", default="g2_native_v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise SystemExit(f"找不到数据配置：{args.data.resolve()}")
    if args.epochs < 1 or args.imgsz < 32 or args.batch == 0 or args.workers < 0:
        raise SystemExit("epochs/imgsz/batch/workers 参数无效。")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("缺少 ultralytics；请先安装 ultralytics==8.4.80。") from exc
    model = YOLO(args.model)
    result = model.train(
        task="segment", data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, workers=args.workers, project=str(args.project.resolve()),
        name=args.name, pretrained=True, patience=30, seed=42, deterministic=True,
        degrees=5.0, translate=0.08, scale=0.20, fliplr=0.0, flipud=0.0,
    )
    save_dir = Path(model.trainer.save_dir).resolve()
    print(f"训练完成：{result}")
    print(f"推荐权重：{save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
