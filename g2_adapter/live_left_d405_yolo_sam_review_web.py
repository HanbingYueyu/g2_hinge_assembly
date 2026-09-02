#!/usr/bin/env python3
"""Read-only GDK left-D405 YOLO primary detection with SAM3 review.

YOLO11-seg runs on every scheduled frame.  SAM3 runs at a lower frequency on
the same RGB-D frame; a target is published only while both models agree.
This program never imports a Robot or gripper API and never saves frames.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

try:
    from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from .live_left_d405_sam3_web import SharedView, handler_class
    from .localize_left_d405_sam3 import eligible_candidates, parse_roi, stats_for_mask
    from .sam_crop import center_crop_bounds, center_crop_image, center_crop_image_aspect, draw_crop_box, expand_crop_masks
    from .yolo_hinge import associate_shaft_to_hinge, infer_yolo_masks, load_yolo_segmentation_model, resize_masks_to_full_image, review_matches, split_yolo_candidates
except ImportError:
    from camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from live_left_d405_sam3_web import SharedView, handler_class
    from localize_left_d405_sam3 import eligible_candidates, parse_roi, stats_for_mask
    from sam_crop import center_crop_bounds, center_crop_image, center_crop_image_aspect, draw_crop_box, expand_crop_masks
    from yolo_hinge import associate_shaft_to_hinge, infer_yolo_masks, load_yolo_segmentation_model, resize_masks_to_full_image, review_matches, split_yolo_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_SCALE_M = 0.001
HINGE_CLASS_ID = 0
SHAFT_CLASS_ID = 1
CLASS_NAMES = {HINGE_CLASS_ID: "hinge", SHAFT_CLASS_ID: "shaft"}
_PREFERRED_G2_HINGE_SHAFT_YOLO = PROJECT_ROOT / "models" / "left_d405_hinge_shaft_yolo" / "best.pt"
_ROLLBACK_G2_HINGE_YOLO = PROJECT_ROOT / "models" / "left_d405_hinge_yolo" / "best.pt"
DEFAULT_G2_NATIVE_YOLO = (
    _PREFERRED_G2_HINGE_SHAFT_YOLO
    if _PREFERRED_G2_HINGE_SHAFT_YOLO.is_file()
    else _ROLLBACK_G2_HINGE_YOLO
)
DEFAULT_YOLO_INPUT_MODE = "native" if _PREFERRED_G2_HINGE_SHAFT_YOLO.is_file() else "aspect-crop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="左 D405：YOLO 主检测、SAM3 低频复核的只读网页。")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--yolo-fps", type=float, default=0.0, help="YOLO 最大调度频率；默认 0，即不额外限速。")
    parser.add_argument("--sam-review-interval-s", type=float, default=1.0, help="SAM3 复核周期，默认 1 秒。")
    parser.add_argument("--sam-review-ttl-s", type=float, default=2.5, help="复核结果最大有效期，默认 2.5 秒。")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--prompt", default="silver triangular hinge")
    parser.add_argument("--selection", choices=("largest", "leftmost", "rightmost"), default="leftmost")
    parser.add_argument("--yolo-confidence", type=float, default=0.01)
    parser.add_argument("--yolo-target-confidence", type=float, default=0.01, help="允许显示及作为抓取候选的 YOLO 最低置信度；默认 0.01。")
    parser.add_argument("--sam-threshold", type=float, default=0.10)
    parser.add_argument("--min-area", type=int, default=1500)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=50)
    parser.add_argument("--roi", default="none", metavar="X0,Y0,X1,Y1", help="可选的额外 ROI；始终与中心 70%% 相交。")
    parser.add_argument("--review-max-pixel-error", type=float, default=80.0)
    parser.add_argument("--review-max-depth-error-m", type=float, default=0.05)
    parser.add_argument("--shaft-association-max-distance-px", type=float, default=320.0)
    parser.add_argument("--shaft-association-max-horizontal-distance-px", type=float, default=140.0)
    parser.add_argument("--shaft-association-ambiguity-margin-px", type=float, default=20.0)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument(
        "--yolo-input-mode",
        choices=("aspect-crop", "native"),
        default=DEFAULT_YOLO_INPUT_MODE,
        help="重新训练的 G2 原生模型用 native；旧 680x420 模型用 aspect-crop。",
    )
    parser.add_argument("--yolo-input-width", type=int, default=680, help="YOLO 训练相机输入宽度；最大同宽高比中心区域会缩放到该宽度。")
    parser.add_argument("--yolo-input-height", type=int, default=420, help="YOLO 训练相机输入高度；最大同宽高比中心区域会缩放到该高度。")
    parser.add_argument("--yolo-device", default="cuda:0")
    parser.add_argument("--yolo-model", type=Path, default=DEFAULT_G2_NATIVE_YOLO)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt")
    return parser.parse_args()


def choose_candidate(statistics: list[dict], candidates: list[int], selection: str) -> int | None:
    if not candidates:
        return None
    if selection == "largest":
        return max(candidates, key=lambda index: statistics[index]["area_px"])
    if selection == "leftmost":
        return min(candidates, key=lambda index: statistics[index]["center_px"][0])
    return max(candidates, key=lambda index: statistics[index]["center_px"][0])


def intersect_roi(first: tuple[int, int, int, int] | None, second: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    if first is None:
        return second
    x0 = max(first[0], second[0])
    y0 = max(first[1], second[1])
    x1 = min(first[2], second[2])
    y1 = min(first[3], second[3])
    return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)


def sam_review(
    processor: Sam3Processor,
    color_rgb: np.ndarray,
    aligned_depth: np.ndarray,
    *,
    color_intrinsic: dict,
    end_link_transform: dict,
    prompt: str,
    min_area: int,
    min_valid_depth_pixels: int,
    roi: tuple[int, int, int, int] | None,
    selection: str,
    reference_center_px: list[float] | None,
) -> tuple[list[np.ndarray], list[dict], int | None]:
    sam_input, crop_bounds = center_crop_image(color_rgb)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_state = processor.set_image(Image.fromarray(sam_input))
        output = processor.set_text_prompt(prompt, image_state)
    crop_masks = [np.asarray(mask, dtype=bool).squeeze() for mask in output["masks"].detach().cpu().numpy()]
    masks = expand_crop_masks(crop_masks, full_shape=color_rgb.shape[:2], bounds=crop_bounds)
    scores = output["scores"].detach().float().cpu().numpy().reshape(-1)
    if len(masks) != len(scores):
        raise RuntimeError(f"SAM3 掩膜和置信度数量不一致：{len(masks)} != {len(scores)}")
    statistics = [stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform) for mask in masks]
    for item, score in zip(statistics, scores):
        item["sam3_confidence"] = float(score)
    candidates = eligible_candidates(
        statistics,
        min_area=min_area,
        min_valid_depth_pixels=min_valid_depth_pixels,
        roi=roi,
        require_depth=True,
    )
    if reference_center_px is None:
        selected = choose_candidate(statistics, candidates, selection)
    else:
        reference = np.asarray(reference_center_px, dtype=np.float64)
        selected = min(
            candidates,
            key=lambda index: float(np.linalg.norm(np.asarray(statistics[index]["center_px"], dtype=np.float64) - reference)),
            default=None,
        )
    return masks, statistics, selected


def draw_yolo_overlay(
    color_bgr: np.ndarray,
    masks: list[np.ndarray],
    statistics: list[dict],
    selected: int | None,
    visible_indices: list[int],
    sam_crop_bounds: tuple[int, int, int, int],
    yolo_source_bounds: tuple[int, int, int, int],
    yolo_input_size: tuple[int, int],
    yolo_input_mode: str,
    review: dict,
    sam_selected: dict | None,
) -> np.ndarray:
    overlay = color_bgr.copy()
    for index in visible_indices:
        item = statistics[index]
        mask = np.asarray(masks[index], dtype=bool)
        class_id = int(item.get("yolo_class_id", HINGE_CLASS_ID))
        base_color = (0, 220, 0) if class_id == HINGE_CLASS_ID else (0, 165, 255)
        color = (0, 255, 0) if index == selected else base_color
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.asarray(color)).astype(np.uint8)
        x0, y0, x1, y1 = item["bbox_xyxy"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), color, 2)
        cv2.putText(
            overlay,
            f"{CLASS_NAMES.get(class_id, f'class{class_id}')} #{index} conf={float(item['yolo_confidence']):.2f} depth={int(item['valid_depth_px'])}",
            (x0, max(y0 - 5, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2,
            cv2.LINE_AA,
        )
    overlay = draw_crop_box(overlay, sam_crop_bounds)
    x0, y0, x1, y1 = yolo_source_bounds
    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 0, 255), 2)
    yolo_text = (
        f"YOLO aspect crop -> {yolo_input_size[0]}x{yolo_input_size[1]}"
        if yolo_input_mode == "aspect-crop"
        else f"YOLO native image, model letterbox={yolo_input_size[0]}"
    )
    cv2.putText(overlay, yolo_text, (x0 + 8, min(y1 - 10, overlay.shape[0] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
    if sam_selected is not None:
        x0, y0, x1, y1 = sam_selected["bbox_xyxy"]
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 0), 2)
        cv2.putText(overlay, "SAM review", (x0, min(y1 + 20, overlay.shape[0] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
    accepted = bool(review.get("accepted")) and bool(review.get("fresh"))
    color = (0, 220, 0) if accepted else (0, 100, 255)
    message = "YOLO target accepted" if accepted else f"YOLO withheld: {review.get('reason', 'awaiting_sam_review')}"
    selected_text = "YOLO selected: none" if selected is None else f"YOLO selected: #{selected}"
    cv2.putText(overlay, selected_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)
    cv2.putText(overlay, message[:110], (12, overlay.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
    return overlay


def valid_args(args: argparse.Namespace) -> bool:
    return (
        args.yolo_fps >= 0
        and args.sam_review_interval_s > 0
        and args.sam_review_ttl_s > 0
        and args.warmup >= 0
        and 0 <= args.yolo_confidence <= 1
        and 0 <= args.yolo_target_confidence <= 1
        and 0 <= args.sam_threshold <= 1
        and args.min_area >= 1
        and args.min_valid_depth_pixels >= 1
        and args.review_max_pixel_error > 0
        and args.review_max_depth_error_m > 0
        and args.shaft_association_max_distance_px > 0
        and args.shaft_association_max_horizontal_distance_px > 0
        and args.shaft_association_ambiguity_margin_px >= 0
        and args.yolo_imgsz > 0
        and args.yolo_input_width > 0
        and args.yolo_input_height > 0
    )


def run() -> int:
    args = parse_args()
    if not valid_args(args):
        print("参数错误：频率、阈值、面积、深度或复核容差无效。", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("未检测到 CUDA，不能运行 YOLO/SAM3 复核。", file=sys.stderr)
        return 2
    if not args.checkpoint.is_file() or not args.yolo_model.is_file():
        print(f"找不到权重：sam={args.checkpoint} yolo={args.yolo_model}", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    shared = SharedView()
    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer((args.host, args.port), handler_class(shared, title="G2 左手 D405 / YOLO 主检测 + SAM3 复核（只读）"))
    import threading
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"实时页面：http://{args.host}:{args.port}")
    print("只读模式：YOLO 主检测、SAM3 低频复核；不创建 Robot、不发送运动或夹爪命令。")

    initialized = False
    camera = None
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        color_type = agibot_gdk.CameraType.kHandLeftColor
        depth_type = agibot_gdk.CameraType.kHandLeftDepth
        camera = agibot_gdk.Camera([color_type, depth_type])
        for _ in range(args.warmup):
            camera.get_latest_image(color_type, 1000.0)

        print(f"loading_yolo={args.yolo_model}", flush=True)
        yolo_model = load_yolo_segmentation_model(args.yolo_model, device=args.yolo_device)
        print(f"loading_sam3={args.checkpoint}", flush=True)
        sam_model = build_sam3_image_model(checkpoint_path=str(args.checkpoint), load_from_HF=False)
        sam_processor = Sam3Processor(sam_model, confidence_threshold=args.sam_threshold)

        color_intrinsic = depth_intrinsic = depth_to_color = end_link_transform = None
        sequence = 0
        last_completed_at: float | None = None
        last_review_at = float("-inf")
        review: dict = {"accepted": False, "fresh": False, "reason": "awaiting_first_sam_review"}
        interval = 0.0 if args.yolo_fps == 0 else 1.0 / args.yolo_fps
        while True:
            started = time.monotonic()
            try:
                color = camera.get_latest_image(color_type, 1000.0)
                depth = camera.get_nearest_image(depth_type, int(color.timestamp_ns), 1000.0)
                color_rgb = decode_color_rgb(color, agibot_gdk)
                color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                depth_u16 = decode_depth_u16(depth, agibot_gdk)
                if color_intrinsic is None:
                    color_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height))
                    depth_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height))
                    tf = agibot_gdk.TF()
                    depth_to_color = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor))
                    end_link_transform = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink))
                aligned_depth = align_depth_to_color(depth_u16, depth_intrinsic=depth_intrinsic, color_intrinsic=color_intrinsic, depth_to_color=depth_to_color, depth_scale_m=DEPTH_SCALE_M)
                crop_bounds = center_crop_bounds(color_rgb.shape)
                requested_roi = parse_roi(args.roi, color_rgb.shape[:2])
                effective_roi = intersect_roi(requested_roi, crop_bounds)
                if args.yolo_input_mode == "native":
                    yolo_source = color_bgr
                    yolo_source_bounds = (0, 0, color_bgr.shape[1], color_bgr.shape[0])
                    yolo_input = yolo_source
                else:
                    yolo_source, yolo_source_bounds = center_crop_image_aspect(
                        color_bgr,
                        aspect_width=args.yolo_input_width,
                        aspect_height=args.yolo_input_height,
                    )
                    yolo_input = cv2.resize(
                        yolo_source,
                        (args.yolo_input_width, args.yolo_input_height),
                        interpolation=cv2.INTER_LINEAR,
                    )
                yolo_input_masks, yolo_scores, yolo_class_ids = infer_yolo_masks(
                    yolo_model, yolo_input, confidence=args.yolo_confidence, imgsz=args.yolo_imgsz, device=args.yolo_device
                )
                yolo_source_masks = resize_masks_to_full_image(
                    yolo_input_masks, full_shape=yolo_source.shape[:2]
                )
                yolo_masks = expand_crop_masks(
                    yolo_source_masks,
                    full_shape=color_rgb.shape[:2],
                    bounds=yolo_source_bounds,
                )
                yolo_stats = [stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform) for mask in yolo_masks]
                for item, score, class_id in zip(yolo_stats, yolo_scores, yolo_class_ids):
                    item["yolo_confidence"] = float(score)
                    item["yolo_class_id"] = int(class_id)
                    item["yolo_class_name"] = CLASS_NAMES.get(int(class_id), f"class{int(class_id)}")
                yolo_visual_candidates = [
                    index
                    for index, item in enumerate(yolo_stats)
                    if (
                        item["area_px"] >= args.min_area
                        and effective_roi is not None
                        and effective_roi[0] <= item["center_px"][0] <= effective_roi[2]
                        and effective_roi[1] <= item["center_px"][1] <= effective_roi[3]
                        and float(item["yolo_confidence"]) >= args.yolo_target_confidence
                    )
                ]
                yolo_geometry_candidates = eligible_candidates(
                    yolo_stats,
                    min_area=args.min_area,
                    min_valid_depth_pixels=args.min_valid_depth_pixels,
                    roi=effective_roi,
                    require_depth=True,
                ) if effective_roi is not None else []
                hinge_geometry_candidates, shaft_geometry_candidates = split_yolo_candidates(
                    yolo_stats,
                    yolo_geometry_candidates,
                    confidence_threshold=args.yolo_target_confidence,
                    hinge_class_id=HINGE_CLASS_ID,
                    shaft_class_id=SHAFT_CLASS_ID,
                )
                yolo_candidates = hinge_geometry_candidates
                yolo_diagnostics = [
                    {
                        "mask_index": index,
                        "class_id": int(item.get("yolo_class_id", -1)),
                        "class_name": item.get("yolo_class_name"),
                        "confidence": float(item["yolo_confidence"]),
                        "area_px": int(item["area_px"]),
                        "center_px": item.get("center_px"),
                        "valid_depth_px": int(item["valid_depth_px"]),
                        "depth_median_m": item.get("depth_median_m"),
                        "visual_eligible": index in yolo_visual_candidates,
                        "geometry_eligible": index in yolo_geometry_candidates,
                        "hinge_geometry_eligible": index in hinge_geometry_candidates,
                        "shaft_geometry_eligible": index in shaft_geometry_candidates,
                        "target_eligible": index in yolo_candidates,
                    }
                    for index, item in enumerate(yolo_stats)
                ]
                yolo_index = choose_candidate(yolo_stats, yolo_candidates, args.selection)
                yolo_selected = None if yolo_index is None else yolo_stats[yolo_index]
                shaft_association = associate_shaft_to_hinge(
                    yolo_stats,
                    yolo_index,
                    shaft_indices=shaft_geometry_candidates,
                    max_center_distance_px=args.shaft_association_max_distance_px,
                    max_horizontal_distance_px=args.shaft_association_max_horizontal_distance_px,
                    ambiguity_margin_px=args.shaft_association_ambiguity_margin_px,
                )

                now = time.monotonic()
                sam_selected = review.get("sam_selected")
                if yolo_selected is not None and now - last_review_at >= args.sam_review_interval_s:
                    _, sam_stats, sam_index = sam_review(
                        sam_processor, color_rgb, aligned_depth,
                        color_intrinsic=color_intrinsic, end_link_transform=end_link_transform,
                        prompt=args.prompt, min_area=args.min_area,
                        min_valid_depth_pixels=args.min_valid_depth_pixels,
                        roi=effective_roi, selection=args.selection,
                        reference_center_px=yolo_selected["center_px"],
                    )
                    sam_selected = None if sam_index is None else sam_stats[sam_index]
                    accepted, detail = review_matches(
                        yolo_selected, sam_selected,
                        max_pixel_error=args.review_max_pixel_error,
                        max_depth_error_m=args.review_max_depth_error_m,
                    )
                    review = {
                        **detail,
                        "accepted": accepted,
                        "timestamp_monotonic": now,
                        "sam_mask_count": len(sam_stats),
                        "sam_selected": sam_selected,
                    }
                    last_review_at = now
                elif yolo_selected is None:
                    review = {"accepted": False, "fresh": False, "reason": "no_eligible_yolo_target", "sam_selected": None}
                age = None if "timestamp_monotonic" not in review else now - float(review["timestamp_monotonic"])
                review = dict(review)
                review["age_s"] = None if age is None else round(max(age, 0.0), 3)
                review["fresh"] = age is not None and age <= args.sam_review_ttl_s
                published_index = yolo_index if review.get("accepted") and review.get("fresh") else None
                published = None if published_index is None else yolo_stats[published_index]
                overlay = draw_yolo_overlay(
                    color_bgr, yolo_masks, yolo_stats, yolo_index, yolo_visual_candidates,
                    crop_bounds, yolo_source_bounds,
                    (args.yolo_imgsz, args.yolo_imgsz) if args.yolo_input_mode == "native" else (args.yolo_input_width, args.yolo_input_height),
                    args.yolo_input_mode, review, sam_selected,
                )
                completed_at = time.monotonic()
                processing_fps = None if last_completed_at is None else 1.0 / max(completed_at - last_completed_at, 1e-9)
                last_completed_at = completed_at
                status = {
                    "state": "accepted" if published is not None else "withheld_pending_or_failed_sam_review",
                    "sequence": sequence + 1,
                    "selection": args.selection,
                    "yolo": {
                        "model": str(args.yolo_model), "confidence_threshold": args.yolo_confidence,
                        "mask_count": len(yolo_stats),
                        "hinge_class_id": HINGE_CLASS_ID,
                        "shaft_class_id": SHAFT_CLASS_ID,
                        "visual_candidate_indices": yolo_visual_candidates,
                        "geometry_candidate_indices": hinge_geometry_candidates,
                        "shaft_geometry_candidate_indices": shaft_geometry_candidates,
                        "shaft_indices": [
                            int(index) for index, item in enumerate(yolo_stats)
                            if int(item.get("yolo_class_id", -1)) == SHAFT_CLASS_ID
                        ],
                        "shaft_candidate_indices": [int(index) for index in shaft_geometry_candidates],
                        "shaft_association": shaft_association,
                        "shaft_association_max_distance_px": args.shaft_association_max_distance_px,
                        "shaft_association_max_horizontal_distance_px": args.shaft_association_max_horizontal_distance_px,
                        "shaft_association_ambiguity_margin_px": args.shaft_association_ambiguity_margin_px,
                        "eligible_candidate_indices": yolo_candidates,
                        "target_confidence_threshold": args.yolo_target_confidence,
                        "candidate_diagnostics": yolo_diagnostics,
                        "selected_mask_index": yolo_index, "selected": yolo_selected,
                    },
                    "sam_review": {key: value for key, value in review.items() if key != "timestamp_monotonic"},
                    "published_target": published,
                    "shaft_association": shaft_association,
                    "shaft": (
                        None
                        if shaft_association.get("shaft_index") is None
                        else yolo_stats[int(shaft_association["shaft_index"])]
                    ),
                    "effective_roi_xyxy": effective_roi,
                    "sam_center_crop_xyxy": list(crop_bounds),
                    "sam_center_crop_fraction": 0.70,
                    "yolo_input": (
                        "native_left_d405_color_image; YOLO performs its own letterbox"
                        if args.yolo_input_mode == "native"
                        else "largest_centered_training_aspect_crop_resized_to_training_resolution; masks expanded to full-image coordinates"
                    ),
                    "yolo_input_mode": args.yolo_input_mode,
                    "yolo_input_size": [int(yolo_input.shape[1]), int(yolo_input.shape[0])],
                    "yolo_imgsz": args.yolo_imgsz,
                    "yolo_source_crop_xyxy": list(yolo_source_bounds),
                    "inference_ms": round((completed_at - started) * 1000.0, 1),
                    "processing_fps": None if processing_fps is None else round(processing_fps, 2),
                    "color_depth_delta_ms": round(abs(int(color.timestamp_ns) - int(depth.timestamp_ns)) / 1_000_000.0, 3),
                    "safety": "read-only Camera/TF/YOLO/SAM3; no Robot, motion, mode, or gripper command",
                }
                ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 88])
                if not ok:
                    raise RuntimeError("无法编码浏览器叠加图")
                sequence += 1
                shared.update(encoded.tobytes(), status)
            except Exception as exc:
                shared.update(None, {"state": "error", "sequence": sequence, "error": str(exc), "safety": "no robot or gripper command"})
                print(f"YOLO/SAM3 检测帧失败：{exc}", file=sys.stderr)
            remaining = interval - (time.monotonic() - started)
            if remaining > 0:
                time.sleep(remaining)
    except KeyboardInterrupt:
        print("收到 Ctrl-C，正在停止只读 YOLO/SAM3 检测。")
        return 0
    except Exception as exc:
        print(f"YOLO/SAM3 实时检测启动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        server.shutdown()
        server.server_close()
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(run())
