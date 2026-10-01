#!/usr/bin/env python3
"""Split two-hole crops by source frame, never by augmented image."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


VALIDATION_FRAMES = set(range(12, 18)) | set(range(42, 48))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("datasets/two_hole_yolo_lit_v3"))
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/two_hole_yolo_lit_v3_trainval"))
    args = parser.parse_args()
    source = args.source.resolve()
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"Refusing to overwrite nonempty directory: {out}")
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text().splitlines()]
    classes = json.loads((source / "dataset_info.json").read_text())["classes"]
    counts = {"train": 0, "val": 0}
    frame_sets = {"train": set(), "val": set()}
    for row in rows:
        index = int(row["source_index"])
        split = "val" if index in VALIDATION_FRAMES else "train"
        if split == "val" and row["variant"] != 0:
            continue
        stem = Path(row["image"]).stem
        for kind, suffix in (("images", ".png"), ("labels", ".txt")):
            destination = out / kind / split / f"{stem}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            target = (source / kind.replace("images", "images_all").replace("labels", "labels_all") /
                      f"{stem}{suffix}")
            if not target.is_file():
                raise SystemExit(f"Missing source: {target}")
            os.symlink(target, destination)
        counts[split] += 1
        frame_sets[split].add(index)
    if frame_sets["train"] & frame_sets["val"]:
        raise AssertionError("Source frames leaked across splits")
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {class_id}: {classes[str(class_id)]}\n" for class_id in range(len(classes))),
        encoding="utf-8",
    )
    summary = {"source": str(source), "counts": counts,
               "source_frames": {key: sorted(value) for key, value in frame_sets.items()},
               "note": "Validation uses only unaugmented frames from the same static lighting session; collect a separate session for an independent test."}
    (out / "split.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"data_yaml": str(out / "data.yaml"), "counts": counts,
                      "source_frame_counts": {key: len(value) for key, value in frame_sets.items()}},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
