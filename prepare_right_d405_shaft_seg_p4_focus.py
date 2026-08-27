#!/usr/bin/env python3
"""Build a reviewed P4-focused shaft-segmentation dataset without mutating raw data.

The old right-D405 set contains several observation poses.  The automatic flow
needs the much narrower P4 view, so this tool keeps every reviewed automatic-P4
sample and only the old positives whose annotated C-shaft has compatible image
position, scale, and area.  Every copied item carries a manifest entry, making
the selection auditable and reversible.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/home/jpb/桌面/st3/traditional")


def load_sample(raw_dir: Path, stem: str) -> dict | None:
    image = raw_dir / "images_all" / f"{stem}.jpg"
    label = raw_dir / "labels_all" / f"{stem}.txt"
    mask = raw_dir / "masks_all" / f"{stem}.png"
    if not (image.is_file() and label.is_file() and mask.is_file()):
        return None
    image_mask = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
    if image_mask is None:
        return None
    return {"image": image, "label": label, "mask": mask, "mask_array": image_mask > 127}


def p4_like_positive(mask: np.ndarray) -> bool:
    """Conservative image-space envelope measured from reviewed P4 captures."""
    rows, cols = np.where(mask)
    if len(cols) == 0:
        return False
    center_x = float(cols.mean())
    center_y = float(rows.mean())
    area = int(len(cols))
    # Deliberately broader than the automatic-P4 examples: it retains useful
    # near-P4 pose/illumination changes but excludes the markedly different
    # old right-arm observation pose.
    return 280.0 <= center_x <= 405.0 and 0.0 <= center_y <= 190.0 and 450 <= area <= 1400


def copy_record(record: dict, split: str, out_dir: Path) -> None:
    for kind in ("images", "labels", "masks"):
        (out_dir / kind / split).mkdir(parents=True, exist_ok=True)
    stem = record["output_stem"]
    shutil.copy2(record["image"], out_dir / "images" / split / f"{stem}.jpg")
    shutil.copy2(record["label"], out_dir / "labels" / split / f"{stem}.txt")
    shutil.copy2(record["mask"], out_dir / "masks" / split / f"{stem}.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-raw-dir",
        default=str(ROOT / "datasets/right_d405_slotted_shaft_seg_raw"),
    )
    parser.add_argument(
        "--auto-p4-raw-dir",
        default=str(ROOT / "datasets/right_d405_slotted_shaft_seg_auto_p4_raw"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "datasets/right_d405_slotted_shaft_seg_p4_focus_v2"),
    )
    args = parser.parse_args()

    old_dir = Path(args.old_raw_dir).expanduser().resolve()
    auto_dir = Path(args.auto_p4_raw_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {out_dir}")

    records: list[dict] = []
    # Old data: retain only reviewed positives matching the P4 visual envelope.
    for image_path in sorted((old_dir / "images_all").glob("frame_*.jpg")):
        sample = load_sample(old_dir, image_path.stem)
        if sample is None or not sample["label"].read_text(encoding="utf-8").strip():
            continue
        if not p4_like_positive(sample["mask_array"]):
            continue
        records.append({
            **sample,
            "source": "old_p4_like",
            "kind": "positive",
            "output_stem": f"old_{image_path.stem}",
        })

    # Automatic P4 data: preserve all reviewed samples, including empty-mask
    # true negatives.  These are the most important examples for this task.
    for image_path in sorted((auto_dir / "images_all").glob("frame_*.jpg")):
        sample = load_sample(auto_dir, image_path.stem)
        if sample is None:
            continue
        kind = "positive" if sample["label"].read_text(encoding="utf-8").strip() else "negative"
        records.append({
            **sample,
            "source": "automatic_p4",
            "kind": kind,
            "output_stem": f"auto_{image_path.stem}",
        })

    positives = [record for record in records if record["kind"] == "positive"]
    negatives = [record for record in records if record["kind"] == "negative"]
    if len(positives) < 30 or len(negatives) < 6:
        raise SystemExit(f"need at least 30 positives and 6 negatives; found {len(positives)} / {len(negatives)}")

    # Keep late automatic samples out of training where possible.  This avoids
    # reporting an optimistic validation score from immediately adjacent frames.
    auto_positive = [record for record in positives if record["source"] == "automatic_p4"]
    old_positive = [record for record in positives if record["source"] == "old_p4_like"]
    train, val, test = [], [], []
    test.extend(auto_positive[-3:])
    train.extend(auto_positive[:-3])
    test.extend(negatives[-2:])
    val.extend(negatives[-4:-2])
    train.extend(negatives[:-4])
    # The old compatible view has two distinct capture sessions.  Reserve the
    # tail of it for validation rather than randomising adjacent image frames.
    val.extend(old_positive[-8:-2])
    test.extend(old_positive[-2:])
    train.extend(old_positive[:-8])
    if not val:
        raise SystemExit("no validation records selected")
    splits = {"train": train, "val": val, "test": test}
    for split, split_records in splits.items():
        for record in split_records:
            copy_record(record, split, out_dir)

    (out_dir / "data.yaml").write_text(
        f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: slotted_hinge_shaft\n",
        encoding="utf-8",
    )
    summary = {
        "selection": "all reviewed automatic-P4 samples + old positives inside P4 image envelope",
        "old_raw_dir": str(old_dir),
        "automatic_p4_raw_dir": str(auto_dir),
        "positive": len(positives),
        "negative": len(negatives),
        "splits": {name: len(values) for name, values in splits.items()},
        "records": {
            name: [
                {"source": record["source"], "kind": record["kind"], "stem": record["output_stem"]}
                for record in values
            ]
            for name, values in splits.items()
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
