#!/usr/bin/env python3
from __future__ import annotations
"""Create a non-destructive train/val/test YOLO-seg split from reviewed masks."""

import argparse
import json
import random
import shutil
from pathlib import Path


def split(items: list[dict], train_ratio: float, val_ratio: float) -> dict[str, list[dict]]:
    random.shuffle(items)
    count = len(items)
    train_end = round(count * train_ratio)
    val_end = train_end + round(count * val_ratio)
    return {"train": items[:train_end], "val": items[train_end:val_end], "test": items[val_end:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        action="append",
        default=None,
        help="reviewed raw capture directory; repeat to merge multiple directories",
    )
    parser.add_argument("--out-dir", default="datasets/right_d405_slotted_shaft_seg_yolo")
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-negatives",
        type=int,
        default=10,
        help="minimum reviewed no-target samples required unless --allow-no-negatives is supplied",
    )
    parser.add_argument("--allow-no-negatives", action="store_true", help="permit an experimental positive-only split")
    parser.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        metavar="SOURCE_INDEX:FRAME_STEM",
        help="exclude one reviewed sample, e.g. 1:frame_000004; source index follows --raw-dir order",
    )
    parser.add_argument("--target-name", default=None, help="override the class name stored in data.yaml")
    args = parser.parse_args()
    if not (0 < args.train_ratio < 1 and 0 < args.val_ratio < 1 and args.train_ratio + args.val_ratio < 1):
        parser.error("train/val ratios must be positive and leave a non-empty test ratio")
    raw_dirs = [Path(value).resolve() for value in (args.raw_dir or ["datasets/right_d405_slotted_shaft_seg_raw"])]
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite existing output: {out_dir}")
    if args.min_negatives < 0:
        parser.error("--min-negatives cannot be negative")

    requested_exclusions: set[tuple[int, str]] = set()
    for value in args.exclude_sample:
        try:
            source_text, stem = value.split(":", 1)
            source_index = int(source_text)
        except ValueError as exc:
            parser.error(f"invalid --exclude-sample {value!r}; expected SOURCE_INDEX:FRAME_STEM")
            raise AssertionError from exc
        if source_index < 0 or source_index >= len(raw_dirs) or not stem.startswith("frame_"):
            parser.error(f"invalid --exclude-sample {value!r}")
        requested_exclusions.add((source_index, stem))

    target_name = args.target_name
    samples: list[dict] = []
    applied_exclusions: set[tuple[int, str]] = set()
    for source_index, raw_dir in enumerate(raw_dirs):
        image_dir, label_dir, mask_dir = raw_dir / "images_all", raw_dir / "labels_all", raw_dir / "masks_all"
        if not image_dir.is_dir() or not label_dir.is_dir() or not mask_dir.is_dir():
            raise SystemExit(f"reviewed raw directories missing under: {raw_dir}")
        intrinsics_path = raw_dir / "intrinsics.json"
        if target_name is None and intrinsics_path.is_file():
            names = json.loads(intrinsics_path.read_text(encoding="utf-8")).get("class_names", [])
            if names:
                target_name = str(names[0])
        for image_path in sorted(image_dir.glob("frame_*.jpg")):
            label_path, mask_path = label_dir / f"{image_path.stem}.txt", mask_dir / f"{image_path.stem}.png"
            if not label_path.exists() or not mask_path.exists():
                continue
            source_key = (source_index, image_path.stem)
            if source_key in requested_exclusions:
                applied_exclusions.add(source_key)
                continue
            # Source prefixes prevent frame_000001 collisions when raw captures are merged.
            output_stem = f"frame_s{source_index:02d}_{image_path.stem.removeprefix('frame_')}"
            samples.append(
                {
                    "source_index": source_index,
                    "source_dir": raw_dir,
                    "image": image_path,
                    "label": label_path,
                    "mask": mask_path,
                    "output_stem": output_stem,
                }
            )
    missing_exclusions = requested_exclusions - applied_exclusions
    if missing_exclusions:
        raise SystemExit(f"requested exclusions were not found: {sorted(missing_exclusions)}")
    target_name = target_name or "target"
    positive = [record for record in samples if record["label"].read_text(encoding="utf-8").strip()]
    negative = [path for path in samples if path not in positive]
    if len(positive) < 20:
        raise SystemExit(f"need at least 20 positive reviewed masks; found positive={len(positive)}")
    if len(negative) < args.min_negatives and not args.allow_no_negatives:
        raise SystemExit(
            f"need at least {args.min_negatives} negative reviewed masks; "
            f"found positive={len(positive)}, negative={len(negative)}"
        )
    random.seed(args.seed)
    grouped = {kind: [] for kind in ("train", "val", "test")}
    for group in (positive, negative):
        for kind, paths in split(group, args.train_ratio, args.val_ratio).items():
            grouped[kind].extend(paths)
    for kind, paths in grouped.items():
        for subdir in (out_dir / "images" / kind, out_dir / "labels" / kind, out_dir / "masks" / kind):
            subdir.mkdir(parents=True, exist_ok=True)
        for record in paths:
            output_stem = record["output_stem"]
            shutil.copy2(record["image"], out_dir / "images" / kind / f"{output_stem}.jpg")
            shutil.copy2(record["label"], out_dir / "labels" / kind / f"{output_stem}.txt")
            shutil.copy2(record["mask"], out_dir / "masks" / kind / f"{output_stem}.png")
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: {target_name}\n", encoding="utf-8")
    summary = {
        "raw_dirs": [str(path) for path in raw_dirs],
        "out_dir": str(out_dir),
        "target_name": target_name,
        "positive": len(positive),
        "negative": len(negative),
        "min_negatives": args.min_negatives,
        "excluded_samples": [f"{index}:{stem}" for index, stem in sorted(applied_exclusions)],
        "positive_only_experimental": not negative,
        "splits": {kind: len(paths) for kind, paths in grouped.items()},
        "data_yaml": str(yaml_path),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
