#!/usr/bin/env python3
"""Make labelled, fixed-size crops of the two visually distinct white holes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Coordinates are in native 1920x1536 kHeadStereoRight frames.
ROI = (280, 575, 1180, 1215)  # x0, y0, x1, y1; 900x640 at inference too.
BOXES = (
    (0, (753, 1073, 797, 1107)),  # irregular white ring in the tabletop
    (1, (907, 1101, 939, 1133)),  # round hole at the silver hinge tip
)
CLASS_NAMES = ("irregular_table_hole", "round_hinge_tip_hole")


def transformed_boxes(matrix: np.ndarray, crop_origin: tuple[int, int]) -> list[tuple[int, tuple[float, ...]]]:
    results = []
    for class_id, (x0, y0, x1, y1) in BOXES:
        corners = np.array([
            [x0 - crop_origin[0], y0 - crop_origin[1], 1],
            [x1 - crop_origin[0], y0 - crop_origin[1], 1],
            [x1 - crop_origin[0], y1 - crop_origin[1], 1],
            [x0 - crop_origin[0], y1 - crop_origin[1], 1],
        ], dtype=np.float64)
        moved = corners @ matrix.T
        left, top = moved.min(axis=0)
        right, bottom = moved.max(axis=0)
        if not (0 <= left < right < ROI[2] - ROI[0] and 0 <= top < bottom < ROI[3] - ROI[1]):
            raise ValueError("Augmentation moved a box outside the crop")
        results.append((class_id, (float(left), float(top), float(right), float(bottom))))
    return results


def yolo_labels(boxes: list[tuple[int, tuple[float, ...]]]) -> str:
    width, height = ROI[2] - ROI[0], ROI[3] - ROI[1]
    lines = []
    for class_id, (x0, y0, x1, y1) in boxes:
        lines.append(f"{class_id} {(x0+x1)/2/width:.8f} {(y0+y1)/2/height:.8f} "
                     f"{(x1-x0)/width:.8f} {(y1-y0)/height:.8f}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Directory of native lossless GDK frames")
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/two_hole_yolo_lit_v3"))
    parser.add_argument("--augment-per-frame", type=int, default=4)
    args = parser.parse_args()
    if args.augment_per_frame < 0:
        parser.error("--augment-per-frame must be nonnegative")
    files = sorted(args.source.glob("frame_*.png"))
    if not files:
        parser.error("No frame_*.png files in source directory")
    image_dir = args.out_dir / "images_all"
    label_dir = args.out_dir / "labels_all"
    boxed_dir = args.out_dir / "boxed_all"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    boxed_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260930)
    width, height = ROI[2] - ROI[0], ROI[3] - ROI[1]
    records = []
    for index, file in enumerate(files):
        frame = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (1536, 1920):
            raise ValueError(f"Expected native 1920x1536 frame: {file}")
        for variant in range(args.augment_per_frame + 1):
            dx = int(rng.integers(-10, 11)) if variant else 0
            dy = int(rng.integers(-8, 9)) if variant else 0
            x0, y0 = ROI[0] + dx, ROI[1] + dy
            crop = frame[y0:y0+height, x0:x0+width].copy()
            angle = float(rng.uniform(-3.0, 3.0)) if variant else 0.0
            tx = float(rng.uniform(-4.0, 4.0)) if variant else 0.0
            ty = float(rng.uniform(-3.0, 3.0)) if variant else 0.0
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            matrix[:, 2] += [tx, ty]
            if variant:
                crop = cv2.warpAffine(crop, matrix, (width, height),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            boxes = transformed_boxes(matrix, (x0, y0))
            stem = f"f{index:04d}_v{variant:02d}"
            image_path = image_dir / f"{stem}.png"
            label_path = label_dir / f"{stem}.txt"
            boxed_path = boxed_dir / f"{stem}.png"
            if not cv2.imwrite(str(image_path), crop):
                raise RuntimeError(f"Could not save {image_path}")
            label_path.write_text(yolo_labels(boxes), encoding="utf-8")
            boxed = crop.copy()
            for class_id, (bx0, by0, bx1, by1) in boxes:
                color = (0, 255, 0) if class_id == 0 else (255, 180, 0)
                cv2.rectangle(boxed, (round(bx0), round(by0)),
                              (round(bx1), round(by1)), color, 2)
                cv2.putText(boxed, str(class_id), (round(bx0), round(by0) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if not cv2.imwrite(str(boxed_path), boxed):
                raise RuntimeError(f"Could not save {boxed_path}")
            records.append({"image": str(image_path), "label": str(label_path),
                            "boxed_image": str(boxed_path),
                            "source": str(file), "source_index": index, "variant": variant,
                            "crop_xyxy": [x0, y0, x0+width, y0+height],
                            "angle_degrees": angle, "translate_xy": [tx, ty],
                            "boxes_crop_xyxy": [{"class_id": c, "xyxy": list(b)} for c, b in boxes]})
    (args.out_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    (args.out_dir / "dataset_info.json").write_text(json.dumps({
        "source_camera": "GDK kHeadStereoRight", "source_shape_wh": [1920, 1536],
        "inference_crop_xyxy": list(ROI), "crop_shape_wh": [width, height],
        "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "class_reference_boxes_full_xyxy": {str(i): list(box) for i, box in BOXES},
        "raw_frame_count": len(files), "augmentation_variants_per_frame": args.augment_per_frame,
        "validation_note": "All frames come from one static session; obtain an independent session for validation.",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} labelled {width}x{height} crops from {len(files)} native frames to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
