#!/usr/bin/env python3
"""Merge reviewed hole/target YOLO annotation directories into one dataset."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PARTS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并多台电脑导出的 hole/target 标注并划分训练集")
    parser.add_argument("--input-dir", type=Path, action="append", required=True, help="一个标注目录，可重复指定；目录内需有 images/ 和 labels/")
    parser.add_argument("--out-dir", type=Path, required=True, help="输出 YOLO 数据集目录，必须为空或不存在")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing-labels", action="store_true", help="跳过没有对应 txt 的图像；默认直接报错")
    parser.add_argument("--require-one-each", action="store_true", help="只保留恰好两个框且 hole/target 各一个的样本")
    return parser.parse_args()


def split_records(records: list[dict], train_ratio: float, val_ratio: float, rng: random.Random) -> dict[str, list[dict]]:
    rng.shuffle(records)
    train_end = round(len(records) * train_ratio)
    val_end = train_end + round(len(records) * val_ratio)
    return {"train": records[:train_end], "val": records[train_end:val_end], "test": records[val_end:]}


def discover_input_dirs(paths: list[Path]) -> list[Path]:
    """Accept either a direct annotation dir or a root containing scene dirs."""
    discovered: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        candidates = [path] if (path / "images").is_dir() and (path / "labels").is_dir() else sorted(
            parent for parent in path.rglob("images") if parent.is_dir() and (parent.parent / "labels").is_dir()
        )
        for images in candidates:
            directory = images.parent.resolve()
            if directory not in seen:
                seen.add(directory)
                discovered.append(directory)
    return discovered


def has_one_each(label: Path) -> bool:
    class_ids: list[int] = []
    for line in label.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        try:
            class_ids.append(int(fields[0]))
        except ValueError:
            return False
    return len(class_ids) == 2 and sorted(class_ids) == [0, 1]


def main() -> int:
    args = parse_args()
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio + args.val_ratio < 1):
        raise SystemExit("train/val 比例必须为正，且总和小于 1。")
    input_roots = [path.expanduser().resolve() for path in args.input_dir]
    input_dirs = discover_input_dirs(input_roots)
    if not input_dirs:
        raise SystemExit("没有找到包含 images/ 和 labels/ 的标注目录。")
    out_dir = args.out_dir.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录：{out_dir}")

    records: list[dict] = []
    missing: list[Path] = []
    filtered = 0
    for source_index, input_dir in enumerate(input_dirs):
        image_dir, label_dir = input_dir / "images", input_dir / "labels"
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise SystemExit(f"标注目录缺少 images/labels：{input_dir}")
        for image in sorted(path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS):
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                missing.append(label)
                continue
            if args.require_one_each and not has_one_each(label):
                filtered += 1
                continue
            records.append({"source": source_index, "image": image, "label": label, "positive": bool(label.read_text(encoding="utf-8").strip())})
    if missing and not args.skip_missing_labels:
        sample = "\n".join(str(path) for path in missing[:10])
        more = f"\n…另有 {len(missing) - 10} 个" if len(missing) > 10 else ""
        raise SystemExit(f"有图像缺少对应标注（共 {len(missing)} 个）：\n{sample}{more}\n如需跳过请加 --skip-missing-labels")
    if not records:
        raise SystemExit("没有找到可合并的图像和标注。")

    positive = [record for record in records if record["positive"]]
    negative = [record for record in records if not record["positive"]]
    rng = random.Random(args.seed)
    groups = {part: [] for part in PARTS}
    for subset in (positive, negative):
        split = split_records(subset, args.train_ratio, args.val_ratio, rng)
        for part in PARTS:
            groups[part].extend(split[part])
    for part in PARTS:
        for record in groups[part]:
            name = f"s{record['source']:02d}_{record['image'].name}"
            image_out = out_dir / "images" / part / name
            label_out = out_dir / "labels" / part / f"{Path(name).stem}.txt"
            image_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["image"], image_out)
            shutil.copy2(record["label"], label_out)
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: hole\n  1: target\n",
        encoding="utf-8",
    )
    summary = {
        "input_dirs": [str(path) for path in input_dirs],
        "total": len(records),
        "positive": len(positive),
        "negative": len(negative),
        "missing_labels_skipped": len(missing) if args.skip_missing_labels else 0,
        "filtered_out": filtered,
        "splits": {part: len(groups[part]) for part in PARTS},
        "data_yaml": str(data_yaml),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
