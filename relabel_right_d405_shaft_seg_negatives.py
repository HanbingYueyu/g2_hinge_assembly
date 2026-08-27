#!/usr/bin/env python3
"""Offline correction tool for reviewed right-D405 shaft annotations.

This deliberately never opens a RealSense stream and never controls ROS or a
robot.  It replays frames which were previously marked negative, lets the
operator add the actual C-shaped slotted shaft with a SAM3.1 box prompt, and
atomically replaces that frame's mask/YOLO label only after ``s`` is pressed.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


SAM3_SOURCE = Path(__file__).resolve().parent / "sam3"
if str(SAM3_SOURCE) not in sys.path:
    sys.path.insert(0, str(SAM3_SOURCE))
from sam3.model_builder import build_sam3_predictor


WINDOW_NAME = "Right D405 Shaft Negative Review (offline)"


def box_from_roi(roi, width: int, height: int) -> tuple[int, int, int, int] | None:
    x, y, box_width, box_height = (int(value) for value in roi)
    if box_width < 6 or box_height < 6:
        return None
    x0, y0 = max(0, min(width - 1, x)), max(0, min(height - 1, y))
    x1, y1 = min(width, x0 + box_width), min(height, y0 + box_height)
    return (x0, y0, x1, y1) if x1 - x0 >= 6 and y1 - y0 >= 6 else None


def to_numpy(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def sam_box_mask(predictor, bgr: np.ndarray, box: tuple[int, int, int, int], threshold: float) -> np.ndarray:
    height, width = bgr.shape[:2]
    x0, y0, x1, y1 = box
    normalized_box = [[x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height]]
    image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    state = predictor.model.init_state(resource_path=[image], offload_video_to_cpu=True, async_loading_frames=False)
    try:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, output = predictor.model.add_prompt(
                inference_state=state,
                frame_idx=0,
                text_str="visual",
                boxes_xywh=normalized_box,
                box_labels=[1],
                output_prob_thresh=threshold,
            )
        raw_masks = to_numpy(output.get("out_binary_masks"))
        if raw_masks is None:
            return np.zeros((height, width), dtype=bool)
        if raw_masks.ndim == 4:
            raw_masks = raw_masks[:, 0]
        if raw_masks.ndim == 2:
            raw_masks = raw_masks[None]
        guard = np.zeros((height, width), dtype=bool)
        guard[y0:y1, x0:x1] = True
        candidates = [mask.astype(bool) & guard for mask in raw_masks]
        return max(candidates, key=lambda mask: int(mask.sum()), default=np.zeros_like(guard))
    finally:
        state.clear()


def mask_to_yolo_segmentation(mask: np.ndarray) -> str:
    height, width = mask.shape
    found = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]
    lines = []
    for contour in contours:
        if cv2.contourArea(contour) < 8.0:
            continue
        simplified = cv2.approxPolyDP(contour, epsilon=0.8, closed=True).reshape(-1, 2)
        if len(simplified) >= 3:
            points = " ".join(f"{x / width:.8f} {y / height:.8f}" for x, y in simplified)
            lines.append(f"0 {points}")
    return "\n".join(lines) + ("\n" if lines else "")


class MaskEditor:
    def __init__(self, brush_radius: int) -> None:
        self.mask: np.ndarray | None = None
        self.brush_radius = brush_radius

    def set_mask(self, mask: np.ndarray | None) -> None:
        self.mask = None if mask is None else mask.astype(bool).copy()

    def _paint(self, x: int, y: int, value: bool) -> None:
        if self.mask is None:
            return
        brush = np.zeros(self.mask.shape, dtype=np.uint8)
        cv2.circle(brush, (x, y), self.brush_radius, 255, -1, cv2.LINE_AA)
        if value:
            self.mask |= brush.astype(bool)
        else:
            self.mask &= ~brush.astype(bool)

    def mouse(self, event, x, y, flags, _userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON):
            self._paint(x, y, True)
        if event == cv2.EVENT_RBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON):
            self._paint(x, y, False)


def draw(color: np.ndarray, editor: MaskEditor, prompt_box, index: int, total: int, status: str) -> np.ndarray:
    view = color.copy()
    if editor.mask is not None:
        overlay = np.zeros_like(view)
        overlay[editor.mask] = (0, 255, 0)
        view = cv2.addWeighted(view, 1.0, overlay, 0.38, 0.0)
    if prompt_box is not None:
        x0, y0, x1, y1 = prompt_box
        cv2.rectangle(view, (x0, y0), (x1, y1), (255, 0, 255), 2, cv2.LINE_AA)
    pixels = 0 if editor.mask is None else int(editor.mask.sum())
    lines = [
        f"Offline negative review {index + 1}/{total}; this frame was previously saved as NEGATIVE",
        f"mask_pixels={pixels}; b: tight C-shaft box  left:add  right:erase  [ / ] brush",
        "s: replace this frame as POSITIVE and next   n: keep true negative and next",
        "a/left: previous  d/right: next without changes  c: clear  q/esc: quit",
        status,
    ]
    for row, text in enumerate(lines):
        y = 24 + 22 * row
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (70, 255, 70), 1, cv2.LINE_AA)
    return view


def append_review_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sam3.1_multiplex.pt")
    parser.add_argument("--raw-dir", default="datasets/right_d405_slotted_shaft_seg_auto_p4_raw")
    parser.add_argument("--sam-threshold", type=float, default=0.35)
    parser.add_argument("--brush-radius", type=int, default=5)
    parser.add_argument("--min-mask-pixels", type=int, default=30)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    raw_dir = Path(args.raw_dir).expanduser().resolve()
    image_dir, mask_dir, label_dir = raw_dir / "images_all", raw_dir / "masks_all", raw_dir / "labels_all"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SAM3.1 checkpoint not found: {checkpoint}")
    if not all(path.is_dir() for path in (image_dir, mask_dir, label_dir)):
        raise FileNotFoundError(f"reviewed raw directories missing under: {raw_dir}")
    frames = [
        path for path in sorted(image_dir.glob("frame_*.jpg"))
        if not (label_dir / f"{path.stem}.txt").read_text(encoding="utf-8").strip()
    ]
    if not frames:
        print("no currently-negative frames found; nothing to correct", flush=True)
        return

    print(f"loading SAM3.1 for offline review: {checkpoint}", flush=True)
    muted = io.StringIO()
    with contextlib.redirect_stdout(muted):
        predictor = build_sam3_predictor(
            version="sam3.1", checkpoint_path=str(checkpoint), use_fa3=False,
            use_rope_real=False, async_loading_frames=False,
            default_output_prob_thresh=args.sam_threshold,
        )
    print(f"loaded {len(frames)} originally-negative frames. No file changes occur until s is pressed.", flush=True)

    review_log = raw_dir / "negative_reannotation.jsonl"
    editor = MaskEditor(args.brush_radius)
    current, prompt_box = 0, None
    status = "Draw a tight box around the real C-shaped shaft."
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, editor.mouse)
    try:
        while True:
            image_path = frames[current]
            color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if color is None:
                raise RuntimeError(f"failed to read {image_path}")
            cv2.imshow(WINDOW_NAME, draw(color, editor, prompt_box, current, len(frames), status))
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("c"):
                editor.set_mask(None)
                prompt_box = None
                status = "Cleared current unsaved mask."
            elif key == ord("["):
                editor.brush_radius = max(1, editor.brush_radius - 1)
                status = f"Brush radius={editor.brush_radius}px"
            elif key == ord("]"):
                editor.brush_radius = min(50, editor.brush_radius + 1)
                status = f"Brush radius={editor.brush_radius}px"
            elif key == ord("b"):
                roi = cv2.selectROI(WINDOW_NAME, color, showCrosshair=True, fromCenter=False)
                prompt_box = box_from_roi(roi, color.shape[1], color.shape[0])
                if prompt_box is None:
                    status = "Box cancelled/too small."
                else:
                    start = time.monotonic()
                    editor.set_mask(sam_box_mask(predictor, color, prompt_box, args.sam_threshold))
                    status = f"SAM proposal in {(time.monotonic() - start) * 1000:.0f} ms. Repair then press s."
            elif key == ord("s"):
                if editor.mask is None or int(editor.mask.sum()) < args.min_mask_pixels:
                    status = "Refusing empty positive. Use b and repair the green mask first."
                    continue
                mask_path = mask_dir / f"{image_path.stem}.png"
                label_path = label_dir / f"{image_path.stem}.txt"
                if not cv2.imwrite(str(mask_path), editor.mask.astype(np.uint8) * 255):
                    raise RuntimeError(f"failed to write {mask_path}")
                label_path.write_text(mask_to_yolo_segmentation(editor.mask), encoding="utf-8")
                append_review_log(review_log, {
                    "timestamp_unix": time.time(), "frame": image_path.stem,
                    "old_annotation": "negative", "new_annotation": "positive",
                    "prompt_box_xyxy": prompt_box, "mask_pixels": int(editor.mask.sum()),
                })
                status = f"Saved {image_path.stem} as POSITIVE."
                print(status, flush=True)
                if current == len(frames) - 1:
                    break
                current += 1
                editor.set_mask(None)
                prompt_box = None
            elif key == ord("n"):
                append_review_log(review_log, {
                    "timestamp_unix": time.time(), "frame": image_path.stem,
                    "old_annotation": "negative", "new_annotation": "negative",
                    "reason": "operator confirmed true no-target frame",
                })
                status = f"Kept {image_path.stem} as true NEGATIVE."
                print(status, flush=True)
                if current == len(frames) - 1:
                    break
                current += 1
                editor.set_mask(None)
                prompt_box = None
            elif key in (ord("a"), 81):
                current = max(0, current - 1)
                editor.set_mask(None)
                prompt_box = None
                status = "Previous frame; no changes made."
            elif key in (ord("d"), 83):
                current = min(len(frames) - 1, current + 1)
                editor.set_mask(None)
                prompt_box = None
                status = "Next frame; no changes made."
    finally:
        cv2.destroyAllWindows()
        del predictor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
