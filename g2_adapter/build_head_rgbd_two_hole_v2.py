#!/usr/bin/env python3
"""Build a larger fixed-crop two-hole detector dataset from head RGB-D records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .build_head_rgbd_two_hole_dataset import BOXES, CLASS_NAMES

ROI = (70, 90, 450, 390)  # Native head-color pixels; 380x300.
OLD_RAW = Path("datasets/two_hole_head_rgbd_raw/20260930_132601")
LOCALIZATION_ROOT = Path("datasets/two_hole_right_teach")
NEW_RAW = Path("datasets/two_hole_head_rgbd_raw_v2/20261001_113416")
REFERENCE = Path(
    "datasets/two_hole_right_teach/session_20261001_112319_4mKg7I/"
    "reference/20261001_112321/sample_0000.json"
)
TARGET_KEYS = CLASS_NAMES


def _valid_boxes(boxes: list[tuple[int, tuple[float, float, float, float]]]) -> None:
    if sorted(class_id for class_id, _ in boxes) != [0, 1]:
        raise ValueError("Expected exactly one box for each target class")
    for _, (x0, y0, x1, y1) in boxes:
        if not (0 <= x0 < x1 <= 640 and 0 <= y0 < y1 <= 400):
            raise ValueError(f"Invalid full-image box: {(x0, y0, x1, y1)}")
        if not (ROI[0] < x0 < x1 < ROI[2] and ROI[1] < y0 < y1 < ROI[3]):
            raise ValueError(f"Target outside fixed inference crop: {(x0, y0, x1, y1)}")
    centers = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for _, b in boxes]
    if not (30 < centers[1][0] - centers[0][0] < 65
            and abs(centers[1][1] - centers[0][1]) < 22):
        raise ValueError(f"Unexpected target class geometry: {boxes}")


def _localized_samples(root: Path):
    for path in sorted(root.rglob("sample_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        targets = record["targets"]
        boxes = []
        for class_id, key in enumerate(TARGET_KEYS):
            target = targets[key]
            if target["class_id"] != class_id or target["confidence"] < 0.6:
                raise ValueError(f"Unreliable class label in {path}")
            boxes.append((class_id, tuple(map(float, target["box_head_color_xyxy_px"]))))
        _valid_boxes(boxes)
        image = path.with_name(path.stem + "_color.png")
        if not image.is_file():
            raise ValueError(f"Missing head-color image: {image}")
        yield image, boxes, "localized_rgbd", path.parent.as_posix()


def _old_raw_samples(root: Path):
    for image in sorted((root / "color").glob("frame_*.png")):
        if not (root / "depth_u16" / image.name).is_file():
            raise ValueError(f"Missing paired depth: {image}")
        boxes = [(class_id, tuple(map(float, box))) for class_id, box in BOXES]
        _valid_boxes(boxes)
        yield image, boxes, "original_manual_boxes", root.as_posix()


def _new_raw_samples(root: Path, reference_path: Path):
    reference_record = json.loads(reference_path.read_text(encoding="utf-8"))
    reference_image = cv2.imread(str(reference_path.with_name(reference_path.stem + "_color.png")))
    if reference_image is None:
        raise ValueError("Missing manually reviewed template image")
    templates = []
    for class_id, key in enumerate(TARGET_KEYS):
        x0, y0, x1, y1 = map(round, reference_record["targets"][key]["box_head_color_xyxy_px"])
        template = cv2.cvtColor(reference_image[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        templates.append((class_id, (x0, y0, x1, y1), template))
    seen_positions: dict[int, set[tuple[int, int]]] = {0: set(), 1: set()}
    count = 0
    for image in sorted((root / "color").glob("frame_*.png")):
        if not (root / "depth_u16" / image.name).is_file():
            raise ValueError(f"Missing paired depth: {image}")
        frame = cv2.imread(str(image))
        if frame is None or frame.shape[:2] != (400, 640):
            raise ValueError(f"Invalid head-color frame: {image}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = []
        for class_id, (x0, y0, x1, y1), template in templates:
            left, top = max(0, x0 - 35), max(0, y0 - 35)
            right, bottom = min(640, x1 + 35), min(400, y1 + 35)
            scores = cv2.matchTemplate(
                gray[top:bottom, left:right], template, cv2.TM_CCOEFF_NORMED
            )
            _, score, _, offset = cv2.minMaxLoc(scores)
            minimum = 0.82 if class_id == 0 else 0.88
            if score < minimum:
                raise ValueError(f"Template label below {minimum}: {image}, class {class_id}, score {score:.3f}")
            bx0, by0 = left + offset[0], top + offset[1]
            seen_positions[class_id].add((bx0, by0))
            boxes.append((class_id, (float(bx0), float(by0),
                                     float(bx0 + template.shape[1]),
                                     float(by0 + template.shape[0]))))
        _valid_boxes(boxes)
        count += 1
        yield image, boxes, "reviewed_template_match", root.as_posix()
    if count < 40:
        raise ValueError(f"Expected at least 40 newly captured head RGB-D pairs, found {count}")
    for class_id, positions in seen_positions.items():
        if len(positions) > 5:
            raise ValueError(f"Template labels moved unexpectedly for class {class_id}: {positions}")


def _transform_boxes(boxes, matrix):
    transformed = []
    for class_id, (x0, y0, x1, y1) in boxes:
        corners = np.array(
            [[x0 - ROI[0], y0 - ROI[1], 1], [x1 - ROI[0], y0 - ROI[1], 1],
             [x1 - ROI[0], y1 - ROI[1], 1], [x0 - ROI[0], y1 - ROI[1], 1]],
            dtype=np.float64,
        )
        xy = corners @ matrix.T
        left, top = xy.min(axis=0)
        right, bottom = xy.max(axis=0)
        if not (1 <= left < right <= ROI[2] - ROI[0] - 1
                and 1 <= top < bottom <= ROI[3] - ROI[1] - 1):
            return None
        transformed.append((class_id, (left, top, right, bottom)))
    return transformed


def _write_example(out: Path, split: str, stem: str, image: np.ndarray,
                   boxes, provenance: dict):
    height, width = image.shape[:2]
    assert (width, height) == (ROI[2] - ROI[0], ROI[3] - ROI[1])
    labels = []
    boxed = image.copy()
    for class_id, (x0, y0, x1, y1) in boxes:
        labels.append(
            f"{class_id} {(x0+x1)/(2*width):.8f} {(y0+y1)/(2*height):.8f} "
            f"{(x1-x0)/width:.8f} {(y1-y0)/height:.8f}\n"
        )
        color = (0, 255, 0) if class_id == 0 else (255, 180, 0)
        cv2.rectangle(boxed, (round(x0), round(y0)), (round(x1), round(y1)), color, 2)
        cv2.putText(boxed, str(class_id), (round(x0), max(11, round(y0) - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    paths = {
        "image": out / "images" / split / f"{stem}.png",
        "label": out / "labels" / split / f"{stem}.txt",
        "boxed": out / "boxed" / split / f"{stem}.png",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(paths["image"]), image) or not cv2.imwrite(str(paths["boxed"]), boxed):
        raise OSError(f"Could not save image or boxed image for {stem}")
    paths["label"].write_text("".join(labels), encoding="utf-8")
    return {"split": split, "image": str(paths["image"]), "label": str(paths["label"]),
            "boxed": str(paths["boxed"]), **provenance}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/two_hole_head_rgbd_yolo_v2"))
    parser.add_argument("--old-raw", type=Path, default=OLD_RAW)
    parser.add_argument("--new-raw", type=Path, default=NEW_RAW)
    parser.add_argument("--localization-root", type=Path, default=LOCALIZATION_ROOT)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--augments", type=int, default=3)
    args = parser.parse_args()
    if args.augments < 0:
        parser.error("--augments must be nonnegative")
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error(f"Refusing to overwrite nonempty dataset: {out}")
    sources = (
        list(_old_raw_samples(args.old_raw))
        + list(_localized_samples(args.localization_root))
        + list(_new_raw_samples(args.new_raw, args.reference))
    )
    if len(sources) < 200:
        parser.error(f"Too few distinct RGB-D source frames: {len(sources)}")
    rng = np.random.default_rng(20261001)
    width, height = ROI[2] - ROI[0], ROI[3] - ROI[1]
    rows = []
    for source_index, (path, full_boxes, label_method, source_group) in enumerate(sources):
        frame = cv2.imread(str(path))
        if frame is None or frame.shape[:2] != (400, 640):
            raise ValueError(f"Expected 640x400 head-color frame: {path}")
        crop = frame[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        # Hold out distinct new-capture frames and one complete shifted localization run.
        is_new = label_method == "reviewed_template_match"
        split = "val" if (
            (is_new and int(path.stem.split("_")[-1]) >= 36)
            or ("run_20260930_164" in source_group)
        ) else "train"
        variants = 0 if split == "val" else args.augments
        for variant in range(variants + 1):
            if variant == 0:
                output_image = crop.copy()
                matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 0, 1)
                shifted_boxes = _transform_boxes(full_boxes, matrix)
                angle = tx = ty = 0.0
            else:
                for _ in range(100):
                    angle = float(rng.uniform(-18, 18))
                    tx = float(rng.uniform(-65, 65))
                    ty = float(rng.uniform(-42, 42))
                    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
                    matrix[:, 2] += [tx, ty]
                    shifted_boxes = _transform_boxes(full_boxes, matrix)
                    if shifted_boxes is not None:
                        break
                else:
                    raise ValueError(f"Could not generate in-bounds augmentation for {path}")
                output_image = cv2.warpAffine(
                    crop, matrix, (width, height), flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101,
                )
            if shifted_boxes is None:
                raise ValueError(f"Original target clipped in {path}")
            stem = f"s{source_index:04d}_v{variant:02d}"
            rows.append(_write_example(
                out, split, stem, output_image, shifted_boxes,
                {"source_color": str(path), "source_group": source_group,
                 "label_method": label_method, "variant": variant,
                 "angle_degrees": angle, "translation_xy": [tx, ty],
                 "boxes_crop_xyxy": [{"class_id": cid, "xyxy": list(box)}
                                     for cid, box in shifted_boxes]},
            ))
    (out / "data.yaml").write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {name}\n" for i, name in enumerate(CLASS_NAMES)), encoding="utf-8",
    )
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    counts = {split: sum(row["split"] == split for row in rows) for split in ("train", "val")}
    summary = {"camera": "GDK kHeadColor + kHeadDepth", "fixed_crop_xyxy": ROI,
               "classes": dict(enumerate(CLASS_NAMES)), "source_frames": len(sources),
               "counts": counts, "augmentation": {"degrees": 18, "translate_x_px": 65,
                                                   "translate_y_px": 42},
               "boxed_images": len(rows)}
    (out / "dataset_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
