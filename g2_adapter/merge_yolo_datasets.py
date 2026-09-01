#!/usr/bin/env python3
"""Merge standard YOLO detection datasets while preserving their splits."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并多个标准 YOLO 数据集，保留原有 train/val/test 划分")
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_dirs = [path.expanduser().resolve() for path in args.input_dir]
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录：{out_dir}")
    counts = {split: 0 for split in SPLITS}
    for source_index, input_dir in enumerate(input_dirs):
        if not input_dir.is_dir():
            raise SystemExit(f"数据集目录不存在：{input_dir}")
        found_split = False
        for split in SPLITS:
            image_dir, label_dir = input_dir / "images" / split, input_dir / "labels" / split
            if not image_dir.is_dir():
                continue
            found_split = True
            if not label_dir.is_dir():
                raise SystemExit(f"缺少标注目录：{label_dir}")
            for image in sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
                label = label_dir / f"{image.stem}.txt"
                if not label.is_file():
                    raise SystemExit(f"缺少对应标注：{label}")
                name = f"s{source_index:02d}_{image.name}"
                image_out = out_dir / "images" / split / name
                label_out = out_dir / "labels" / split / f"{Path(name).stem}.txt"
                image_out.parent.mkdir(parents=True, exist_ok=True)
                label_out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, image_out)
                shutil.copy2(label, label_out)
                counts[split] += 1
        if not found_split:
            raise SystemExit(f"数据集没有 images/train、images/val 或 images/test：{input_dir}")
    if counts["train"] == 0 or counts["val"] == 0:
        raise SystemExit(f"合并结果缺少 train 或 val：{counts}")
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: hole\n  1: target\n",
        encoding="utf-8",
    )
    summary = {"input_dirs": [str(path) for path in input_dirs], "splits": counts, "data_yaml": str(data_yaml)}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
