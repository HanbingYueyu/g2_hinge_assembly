#!/usr/bin/env python3
"""Build a deterministic YOLO-seg train/val/test split from reviewed annotations.

Accepted source layouts (first match wins):

* ``<source>/images/all`` + ``<source>/labels/all``
  output of ``annotate_left_hinge_shaft_sam3.py`` (single "all" split)
* ``<source>/images_all`` + ``<source>/labels_all``
  raw capture layout from ``capture_left_d405_hinge_yolo_dataset.py``

Class names come from ``<source>/classes.txt`` or ``<source>/dataset_info.json``,
falling back to ``--class-names``.  Frames are shuffled with a fixed seed and
stratified by the set of class ids they contain, so negative frames and rare
classes still land in every split.  Nothing is written back to the source.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, required=True, help="已审核的数据集目录（images/all 或 images_all 布局）")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class-names", default=None, help="逗号分隔的类别名；仅当源目录没有 classes.txt/dataset_info.json 时需要")
    return parser.parse_args()


def resolve_layout(source: Path) -> tuple[Path, Path]:
    for images, labels in ((source / "images" / "all", source / "labels" / "all"), (source / "images_all", source / "labels_all")):
        if images.is_dir() and labels.is_dir():
            return images, labels
    raise SystemExit(f"源目录缺少 images/all+labels/all 或 images_all+labels_all：{source}")


def resolve_class_names(source: Path, fallback: str | None) -> list[str]:
    classes_path = source / "classes.txt"
    if classes_path.is_file():
        names = [line.strip() for line in classes_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if names:
            return names
    info_path = source / "dataset_info.json"
    if info_path.is_file():
        try:
            names = json.loads(info_path.read_text(encoding="utf-8")).get("class_names") or []
            if names:
                return [str(name) for name in names]
        except (OSError, json.JSONDecodeError):
            pass
    if fallback:
        names = [name.strip() for name in fallback.split(",") if name.strip()]
        if names:
            return names
    raise SystemExit("无法确定类别名：请在源目录提供 classes.txt/dataset_info.json，或传入 --class-names")


def label_class_ids(label_path: Path) -> set[int]:
    ids: set[int] = set()
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if not values:
            continue
        try:
            ids.add(int(values[0]))
        except ValueError as exc:
            raise SystemExit(f"标签类别号非法：{label_path}: {line!r}") from exc
    return ids


def split_group(items: list[dict], train_ratio: float, val_ratio: float) -> dict[str, list[dict]]:
    train_end = round(len(items) * train_ratio)
    val_end = train_end + round(len(items) * val_ratio)
    return {"train": items[:train_end], "val": items[train_end:val_end], "test": items[val_end:]}


def main() -> None:
    args = parse_args()
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio + args.val_ratio < 1):
        raise SystemExit("train/val 比例必须为正，且必须为非空的 test 集留出比例")
    source = args.source.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"源目录不存在：{source}")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"拒绝覆盖非空目录：{out_dir}")
    images_dir, labels_dir = resolve_layout(source)
    class_names = resolve_class_names(source, args.class_names)

    samples: list[dict] = []
    for image in sorted(p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES):
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            raise SystemExit(f"缺少对应标签：{label}")
        ids = label_class_ids(label)
        unknown = {value for value in ids if value < 0 or value >= len(class_names)}
        if unknown:
            raise SystemExit(f"标签类别号 {sorted(unknown)} 超出类别表 {class_names}：{label}")
        samples.append({"image": image, "label": label, "ids": ids})
    if not samples:
        raise SystemExit(f"源目录没有可用图片：{images_dir}")

    groups: dict[tuple[int, ...], list[dict]] = {}
    for sample in samples:
        groups.setdefault(tuple(sorted(sample["ids"])), []).append(sample)

    random.seed(args.seed)
    split_samples: dict[str, list[dict]] = {kind: [] for kind in ("train", "val", "test")}
    for signature in sorted(groups):
        bucket = groups[signature]
        random.shuffle(bucket)
        for kind, portion in split_group(bucket, args.train_ratio, args.val_ratio).items():
            split_samples[kind].extend(portion)

    for kind, records in split_samples.items():
        (out_dir / "images" / kind).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / kind).mkdir(parents=True, exist_ok=True)
        for record in sorted(records, key=lambda item: item["image"].name):
            shutil.copy2(record["image"], out_dir / "images" / kind / record["image"].name)
            shutil.copy2(record["label"], out_dir / "labels" / kind / record["label"].name)

    (out_dir / "classes.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")
    names_yaml = "\n".join(f"  {index}: {name}" for index, name in enumerate(class_names))
    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )
    summary = {
        "source": str(source),
        "out_dir": str(out_dir),
        "class_names": class_names,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "splits": {kind: len(records) for kind, records in split_samples.items()},
        "frames_with_class": {
            name: sum(1 for sample in samples if index in sample["ids"]) for index, name in enumerate(class_names)
        },
        "data_yaml": str(out_dir / "data.yaml"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
