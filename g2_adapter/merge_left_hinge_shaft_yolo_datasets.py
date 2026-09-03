#!/usr/bin/env python3
"""Merge the established left hinge/shaft dataset with new raw captures."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并旧的标准 hinge/shaft 数据和新的 images_all/labels_all 采集")
    parser.add_argument("--existing-dir", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_label(path: Path) -> None:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.split()
        if not values:
            continue
        if len(values) < 7 or (len(values) - 1) % 2:
            raise SystemExit(f"标签格式无效：{path}:{line_number}")
        try:
            class_id = int(values[0])
            coords = [float(value) for value in values[1:]]
        except ValueError as exc:
            raise SystemExit(f"标签包含非数字：{path}:{line_number}") from exc
        if class_id not in (0, 1) or any(not 0.0 <= value <= 1.0 for value in coords):
            raise SystemExit(f"标签类别或坐标无效：{path}:{line_number}")


def standard_items(root: Path) -> list[tuple[Path, Path, str]]:
    items: list[tuple[Path, Path, str]] = []
    for split in SPLITS:
        image_dir, label_dir = root / "images" / split, root / "labels" / split
        if not image_dir.is_dir():
            continue
        if not label_dir.is_dir():
            raise SystemExit(f"缺少标签目录：{label_dir}")
        for image in sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise SystemExit(f"缺少对应标签：{label}")
            validate_label(label)
            items.append((image, label, split))
    if not items:
        raise SystemExit(f"标准数据集没有图片：{root}")
    return items


def raw_items(root: Path, *, train_ratio: float, val_ratio: float, seed: int) -> list[tuple[Path, Path, str]]:
    image_dir, label_dir = root / "images_all", root / "labels_all"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise SystemExit(f"新的采集目录缺少 images_all/labels_all：{root}")
    all_items: list[tuple[Path, Path]] = []
    for image in sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
        label = label_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"缺少对应标签：{label}")
        validate_label(label)
        all_items.append((image, label))
    if not all_items:
        raise SystemExit(f"新的采集目录没有图片：{root}")
    # Keep positive/negative proportions in each split so empty labels do not
    # accidentally all land in validation or test.
    positives = [item for item in all_items if item[1].read_text(encoding="utf-8").strip()]
    negatives = [item for item in all_items if not item[1].read_text(encoding="utf-8").strip()]
    rng = random.Random(seed)
    rng.shuffle(positives); rng.shuffle(negatives)
    result: list[tuple[Path, Path, str]] = []
    for group in (positives, negatives):
        train_end = round(len(group) * train_ratio)
        val_end = train_end + round(len(group) * val_ratio)
        for image, label in group[:train_end]: result.append((image, label, "train"))
        for image, label in group[train_end:val_end]: result.append((image, label, "val"))
        for image, label in group[val_end:]: result.append((image, label, "test"))
    return result


def copy_items(items: list[tuple[Path, Path, str]], out_dir: Path, prefix: str) -> dict[str, int]:
    counts = {split: 0 for split in SPLITS}
    for index, (image, label, split) in enumerate(items):
        stem = f"{prefix}_{index:06d}_{image.stem}"
        image_out = out_dir / "images" / split / f"{stem}{image.suffix.lower()}"
        label_out = out_dir / "labels" / split / f"{stem}.txt"
        image_out.parent.mkdir(parents=True, exist_ok=True); label_out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image, image_out); shutil.copy2(label, label_out)
        counts[split] += 1
    return counts


def main() -> int:
    args = parse_args()
    if not (0.0 < args.train_ratio < 1.0 and 0.0 < args.val_ratio < 1.0 and args.train_ratio + args.val_ratio < 1.0):
        raise SystemExit("train/val 比例无效")
    existing, raw, out = (path.expanduser().resolve() for path in (args.existing_dir, args.raw_dir, args.out_dir))
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"拒绝覆盖非空目录：{out}")
    old_items = standard_items(existing)
    new_items = raw_items(raw, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed)
    old_counts = copy_items(old_items, out, "old")
    new_counts = copy_items(new_items, out, "new")
    (out / "classes.txt").write_text("hinge\nshaft\n", encoding="utf-8")
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: hinge\n  1: shaft\n",
        encoding="utf-8",
    )
    summary = {"existing_dir": str(existing), "raw_dir": str(raw), "splits": {split: old_counts[split] + new_counts[split] for split in SPLITS}, "existing_splits": old_counts, "new_splits": new_counts, "data_yaml": str(out / "data.yaml")}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
