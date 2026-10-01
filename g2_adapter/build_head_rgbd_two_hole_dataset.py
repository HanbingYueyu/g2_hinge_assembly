#!/usr/bin/env python3
"""Label large crops from the head RGB-D color stream for two-hole detection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Native kHeadColor image is 640x400. This crop matches the operator example.
ROI = (105, 125, 345, 315)  # 240x190, x0/y0/x1/y1.
BOXES = (
    (0, (279, 287, 296, 303)),  # irregular white tabletop hole
    (1, (323, 293, 338, 308)),  # round hole at the silver hinge tip
)
CLASS_NAMES = ("irregular_table_hole", "round_hinge_tip_hole")


def transformed_boxes(matrix: np.ndarray, origin: tuple[int, int]) -> list[tuple[int, tuple[float, ...]]]:
    results = []
    width, height = ROI[2] - ROI[0], ROI[3] - ROI[1]
    for class_id, (x0, y0, x1, y1) in BOXES:
        corners = np.array([[x0-origin[0], y0-origin[1], 1],
                            [x1-origin[0], y0-origin[1], 1],
                            [x1-origin[0], y1-origin[1], 1],
                            [x0-origin[0], y1-origin[1], 1]], dtype=np.float64)
        xy = corners @ matrix.T
        left, top = xy.min(axis=0)
        right, bottom = xy.max(axis=0)
        if not (0 <= left < right < width and 0 <= top < bottom < height):
            raise ValueError("Augmentation placed a target outside the crop")
        results.append((class_id, (float(left), float(top), float(right), float(bottom))))
    return results


def yolo_labels(boxes: list[tuple[int, tuple[float, ...]]]) -> str:
    width, height = ROI[2] - ROI[0], ROI[3] - ROI[1]
    return "".join(
        f"{class_id} {(x0+x1)/2/width:.8f} {(y0+y1)/2/height:.8f} "
        f"{(x1-x0)/width:.8f} {(y1-y0)/height:.8f}\n"
        for class_id, (x0, y0, x1, y1) in boxes
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="One head RGB-D capture session")
    parser.add_argument("--out-dir", type=Path, default=Path("datasets/two_hole_head_rgbd_yolo_v1"))
    parser.add_argument("--augment-per-frame", type=int, default=4)
    args = parser.parse_args()
    if args.augment_per_frame < 0:
        parser.error("--augment-per-frame must be nonnegative")
    source = args.source.resolve()
    files = sorted((source / "color").glob("frame_*.png"))
    if not files:
        parser.error("No head RGB-D color frames found")
    out = args.out_dir
    for folder in ("images_all", "labels_all", "boxed_all"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260930)
    width, height = ROI[2] - ROI[0], ROI[3] - ROI[1]
    records = []
    for index, source_image in enumerate(files):
        frame = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
        if frame is None or frame.shape[:2] != (400, 640):
            raise ValueError(f"Expected native 640x400 head color image: {source_image}")
        depth_path = source / "depth_u16" / source_image.name
        if not depth_path.is_file():
            raise ValueError(f"Missing paired head depth frame: {depth_path}")
        for variant in range(args.augment_per_frame + 1):
            dx = int(rng.integers(-3, 4)) if variant else 0
            dy = int(rng.integers(-2, 3)) if variant else 0
            x0, y0 = ROI[0] + dx, ROI[1] + dy
            crop = frame[y0:y0+height, x0:x0+width].copy()
            angle = float(rng.uniform(-0.7, 0.7)) if variant else 0.0
            tx = float(rng.uniform(-1.0, 1.0)) if variant else 0.0
            ty = float(rng.uniform(-0.5, 0.5)) if variant else 0.0
            matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            matrix[:, 2] += [tx, ty]
            if variant:
                crop = cv2.warpAffine(crop, matrix, (width, height),
                                      flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            boxes = transformed_boxes(matrix, (x0, y0))
            stem = f"f{index:04d}_v{variant:02d}"
            image_path = out / "images_all" / f"{stem}.png"
            label_path = out / "labels_all" / f"{stem}.txt"
            boxed_path = out / "boxed_all" / f"{stem}.png"
            if not cv2.imwrite(str(image_path), crop):
                raise RuntimeError(f"Could not write {image_path}")
            label_path.write_text(yolo_labels(boxes), encoding="utf-8")
            boxed = crop.copy()
            for class_id, (bx0, by0, bx1, by1) in boxes:
                color = (0, 255, 0) if class_id == 0 else (255, 180, 0)
                cv2.rectangle(boxed, (round(bx0), round(by0)),
                              (round(bx1), round(by1)), color, 1)
                cv2.putText(boxed, str(class_id), (round(bx0), round(by0) - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            if not cv2.imwrite(str(boxed_path), boxed):
                raise RuntimeError(f"Could not write {boxed_path}")
            records.append({"image": str(image_path), "label": str(label_path),
                            "boxed_image": str(boxed_path), "source_color": str(source_image),
                            "source_depth": str(depth_path), "source_index": index,
                            "variant": variant, "crop_xyxy": [x0, y0, x0+width, y0+height],
                            "angle_degrees": angle, "translation_xy": [tx, ty],
                            "boxes_crop_xyxy": [{"class_id": class_id, "xyxy": list(box)}
                                                for class_id, box in boxes]})
    (out / "manifest.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (out / "dataset_info.json").write_text(json.dumps({
        "camera": "GDK kHeadColor + kHeadDepth RGB-D pair",
        "color_shape_wh": [640, 400], "inference_crop_xyxy": list(ROI),
        "crop_shape_wh": [width, height],
        "classes": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "class_reference_boxes_full_xyxy": {str(i): list(box) for i, box in BOXES},
        "raw_frame_count": len(files), "augmentation_variants_per_frame": args.augment_per_frame,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(records)} labelled {width}x{height} head-color crops from {len(files)} RGB-D pairs to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
