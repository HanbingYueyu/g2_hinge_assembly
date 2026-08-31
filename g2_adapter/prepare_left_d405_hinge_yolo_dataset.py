#!/usr/bin/env python3
"""Build a deterministic YOLO-seg train/val/test dataset from reviewed G2 captures."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从已人工审核的左 D405 铰链帧构建 YOLO-seg 数据集。")
    parser.add_argument("--raw-dir", action="append", default=None, help="采集目录；可重复指定以合并多批数据。")
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/left_d405_hinge_yolo_g2_v1"))
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-positive", type=int, default=80)
    parser.add_argument("--min-negative", type=int, default=25)
    parser.add_argument("--target-name", default="hinge")
    return parser.parse_args()


def split(records: list[dict], train_ratio: float, val_ratio: float) -> dict[str, list[dict]]:
    end_train = round(len(records) * train_ratio)
    end_val = end_train + round(len(records) * val_ratio)
    return {"train": records[:end_train], "val": records[end_train:end_val], "test": records[end_val:]}


def main() -> None:
    args = parse_args()
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio + args.val_ratio < 1):
        raise SystemExit("train/val 比例必须为正，且必须保留 test 集。")
    if args.min_positive < 1 or args.min_negative < 0:
        raise SystemExit("最小样本数无效。")
    raw_dirs = [Path(value).resolve() for value in (args.raw_dir or ["datasets/left_d405_hinge_yolo_raw_g2_v1"])]
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"拒绝覆盖非空目录：{out_dir}")
    samples: list[dict] = []
    for source_index, raw_dir in enumerate(raw_dirs):
        images, labels = raw_dir / "images_all", raw_dir / "labels_all"
        if not images.is_dir() or not labels.is_dir():
            raise SystemExit(f"缺少 images_all/labels_all：{raw_dir}")
        for image in sorted(images.glob("frame_*.jpg")):
            label = labels / f"{image.stem}.txt"
            if not label.is_file():
                raise SystemExit(f"缺少标注：{label}")
            samples.append({"source": source_index, "image": image, "label": label, "positive": bool(label.read_text(encoding="utf-8").strip())})
    positives = [item for item in samples if item["positive"]]
    negatives = [item for item in samples if not item["positive"]]
    if len(positives) < args.min_positive or len(negatives) < args.min_negative:
        raise SystemExit(f"样本不足：positive={len(positives)}/{args.min_positive}, negative={len(negatives)}/{args.min_negative}")
    random.seed(args.seed)
    random.shuffle(positives)
    random.shuffle(negatives)
    groups = {part: [] for part in ("train", "val", "test")}
    for records in (positives, negatives):
        for part, portion in split(records, args.train_ratio, args.val_ratio).items():
            groups[part].extend(portion)
    for part, records in groups.items():
        for item in records:
            name = f"s{item['source']:02d}_{item['image'].name}"
            image_out = out_dir / "images" / part / name
            label_out = out_dir / "labels" / part / f"{Path(name).stem}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["image"], image_out)
            shutil.copy2(item["label"], label_out)
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: {args.target_name}\n",
        encoding="utf-8",
    )
    summary = {"raw_dirs": [str(path) for path in raw_dirs], "positive": len(positives), "negative": len(negatives), "splits": {part: len(records) for part, records in groups.items()}, "data_yaml": str(data_yaml)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
