#!/usr/bin/env python3
"""Localize a text-prompted target from one captured left G2 RGB-D sample.

This program deliberately has no GDK, robot, or gripper imports.  It only
reads an existing capture made by ``capture_left_d405_gdk.py`` and writes
diagnostic artifacts alongside it.  Run it with the separate Python 3.12
SAM3 environment via ``scripts/run_sam3_python.sh``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

try:
    from .sam_crop import center_crop_image, draw_crop_box, expand_crop_masks
except ImportError:
    from sam_crop import center_crop_image, draw_crop_box, expand_crop_masks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_ROOT = PROJECT_ROOT / "outputs" / "left_d405_gdk"
# A perforated metal hinge can expose the distant tray/background through its
# holes.  A whole-mask median would then jump to that background plane.  The
# foreground layer is deliberately quantised coarsely enough to retain a
# slightly tilted hinge, while still separating it from a distant background.
FOREGROUND_DEPTH_BIN_M = 0.005
FOREGROUND_DEPTH_HALF_WIDTH_M = 0.010
FOREGROUND_DEPTH_MIN_FRACTION = 0.05


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对一个已采集的左手 GDK RGB-D 样本执行 SAM3 分割和三维定位（不控制机器人）。"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--scene", help="使用该最终检测场景最新的一次采集。")
    source.add_argument("--sample", type=Path, help="采集目录，目录内应含 metadata.json。")
    parser.add_argument(
        "--prompt",
        default="triangular perforated metal plate",
        help="SAM3 英文文本提示；默认使用较稳健的三角多孔金属板类别短语。",
    )
    parser.add_argument("--threshold", type=float, default=0.15, help="SAM3 置信度阈值。")
    parser.add_argument("--min-area", type=int, default=1500, help="选中目标的最小掩膜面积（像素）。")
    parser.add_argument(
        "--min-valid-depth-pixels",
        type=int,
        default=500,
        help="选中目标所需的最少有效对齐深度像素，默认 500。",
    )
    parser.add_argument(
        "--roi",
        default=None,
        metavar="X0,Y0,X1,Y1",
        help="仅从该彩色图像像素区域内选择候选，用于排除夹爪等非托盘区域。",
    )
    parser.add_argument(
        "--selection",
        choices=("largest", "leftmost", "rightmost"),
        default="largest",
        help="合格候选的选择策略，默认 largest。",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt",
        help="SAM 3.1 multiplex 权重文件。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录；默认写入采集目录的 sam3_localization/。",
    )
    return parser.parse_args()


def latest_capture_for_scene(scene: str) -> Path:
    candidates = sorted(
        (path.parent for path in (CAPTURE_ROOT / scene).glob("*/metadata.json")),
        key=lambda path: path.name,
    )
    if not candidates:
        raise RuntimeError(
            f"没有找到 {scene} 的采集数据；请先在最终检测位运行 "
            f"capture_left_d405_gdk.py --scene {scene}"
        )
    return candidates[-1]


def parse_roi(value: str | None, image_shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if value is None or value.strip().lower() in {"", "none", "off"}:
        return None
    try:
        x0, y0, x1, y1 = (int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise RuntimeError("--roi 格式应为 X0,Y0,X1,Y1（像素）。") from exc
    height, width = image_shape
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise RuntimeError(f"--roi 必须位于图像范围内 0..{width}、0..{height}。")
    return x0, y0, x1, y1


def load_capture(sample: Path) -> tuple[Path, np.ndarray, np.ndarray, dict]:
    sample = sample.expanduser().resolve()
    metadata_path = sample / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"采集目录缺少 metadata.json：{sample}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    color_path = sample / metadata["color"]["file"]
    depth_path = sample / metadata["depth"]["aligned_to_color_file"]
    color_bgr = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise RuntimeError(f"无法读取彩色图像：{color_path}")
    depth_u16 = np.load(depth_path, allow_pickle=False)
    if depth_u16.dtype != np.uint16 or depth_u16.ndim != 2:
        raise RuntimeError(f"投影深度必须是二维 uint16：{depth_path}")
    if color_bgr.shape[:2] != depth_u16.shape:
        raise RuntimeError(
            f"彩色/投影深度尺寸不一致：color={color_bgr.shape[:2]} depth={depth_u16.shape}"
        )
    return sample, color_bgr, depth_u16, metadata


def depth_points(mask: np.ndarray, depth_u16: np.ndarray, scale_m: float, intrinsic: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the nearest supported depth layer and its color-camera XYZ points.

    SAM masks for a perforated hinge contain both the metal surface and the
    distant scene visible through the holes.  Selecting the nearest depth
    layer with meaningful support prevents the background layer from becoming
    the reported object depth merely because it occupies more mask pixels.
    """
    valid = np.asarray(mask, dtype=bool) & (depth_u16 > 0)
    if int(valid.sum()) < 20:
        valid = cv2.dilate(np.asarray(mask, dtype=np.uint8), np.ones((5, 5), np.uint8), iterations=1).astype(bool)
        valid &= depth_u16 > 0
    rows, cols = np.where(valid)
    if rows.size == 0:
        return valid, np.empty((0, 3), dtype=np.float64)
    z_m = depth_u16[rows, cols].astype(np.float64) * scale_m
    bin_ids = np.floor(z_m / FOREGROUND_DEPTH_BIN_M).astype(np.int64)
    unique_bins, counts = np.unique(bin_ids, return_counts=True)
    minimum_support = max(20, int(math.ceil(z_m.size * FOREGROUND_DEPTH_MIN_FRACTION)))
    supported = unique_bins[counts >= minimum_support]
    if supported.size == 0:
        # Keep the largest layer rather than allowing an unsupported nearest
        # noise pixel to masquerade as the foreground object.
        selected_bin = int(unique_bins[int(np.argmax(counts))])
    else:
        selected_bin = int(supported.min())
    layer_center_m = float(np.median(z_m[bin_ids == selected_bin]))
    keep = np.abs(z_m - layer_center_m) <= FOREGROUND_DEPTH_HALF_WIDTH_M
    if int(keep.sum()) < minimum_support:
        keep = bin_ids == selected_bin
    valid.fill(False)
    valid[rows[keep], cols[keep]] = True
    rows, cols, z_m = rows[keep], cols[keep], z_m[keep]
    points = np.column_stack((
        (cols - float(intrinsic["ppx"])) * z_m / float(intrinsic["fx"]),
        (rows - float(intrinsic["ppy"])) * z_m / float(intrinsic["fy"]),
        z_m,
    ))
    return valid, points


def rotation_matrix_xyzw(quaternion: list[float]) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = float(np.linalg.norm((x, y, z, w)))
    if not np.isfinite(norm) or norm < 1e-12:
        raise RuntimeError("GDK 外参四元数无效")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def point_to_left_end_link(point_color_m: np.ndarray, rgbd_to_end_link: dict) -> np.ndarray:
    """Apply GDK's kLeftHandRGBDToArmLEndLink transform to a color-camera point."""
    translation = np.asarray(rgbd_to_end_link["translation_m"], dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise RuntimeError("GDK 外参平移无效")
    return rotation_matrix_xyzw(rgbd_to_end_link["rotation_xyzw"]) @ point_color_m + translation


def stats_for_mask(mask: np.ndarray, depth_u16: np.ndarray, scale_m: float, intrinsic: dict, transform: dict) -> dict:
    mask = np.asarray(mask, dtype=bool)
    rows, cols = np.where(mask)
    if rows.size == 0:
        return {"area_px": 0, "valid_depth_px": 0}
    inliers, xyz = depth_points(mask, depth_u16, scale_m, intrinsic)
    center_px = [float(cols.mean()), float(rows.mean())]
    result = {
        "area_px": int(rows.size),
        "bbox_xyxy": [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())],
        "center_px": center_px,
        "valid_depth_px": int(inliers.sum()),
    }
    if xyz.size == 0:
        result.update({
            "xyz_color_camera_m": None,
            "xyz_left_arm_end_link_m": None,
            "depth_median_m": None,
        })
        return result
    median = np.median(xyz, axis=0)
    result.update({
        "xyz_color_camera_m": [float(value) for value in median],
        "xyz_left_arm_end_link_m": [float(value) for value in point_to_left_end_link(median, transform)],
        "depth_median_m": float(np.median(xyz[:, 2])),
    })
    return result


def draw_overlay(image_bgr: np.ndarray, masks: list[np.ndarray], stats: list[dict], selected: int | None) -> np.ndarray:
    output = image_bgr.copy()
    for index, (mask, item) in enumerate(zip(masks, stats)):
        if item.get("area_px", 0) == 0:
            continue
        class_name = item.get("yolo_class_name")
        if class_name == "shaft":
            color = (0, 165, 255)
        else:
            color = (0, 220, 0) if index == selected else (0, 80, 255)
        output[np.asarray(mask, dtype=bool)] = (
            0.55 * output[np.asarray(mask, dtype=bool)] + 0.45 * np.asarray(color)
        ).astype(np.uint8)
        x0, y0, x1, y1 = item["bbox_xyxy"]
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 2)
        confidence = item.get("sam3_confidence", item.get("yolo_confidence"))
        label = f"{class_name} #{index} area={item['area_px']}" if class_name else f"#{index} area={item['area_px']}"
        if confidence is not None:
            label += f" conf={float(confidence):.2f}"
        cv2.putText(output, label, (x0, max(y0 - 5, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)
    header = "selected: none" if selected is None else f"selected: #{selected}"
    cv2.putText(output, header, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return output


def select_candidate(
    statistics: list[dict],
    *,
    min_area: int,
    min_valid_depth_pixels: int,
    roi: tuple[int, int, int, int] | None,
    selection: str,
) -> int | None:
    """Select one usable target from SAM masks without any robot-side action."""
    eligible = eligible_candidates(
        statistics,
        min_area=min_area,
        min_valid_depth_pixels=min_valid_depth_pixels,
        roi=roi,
    )
    if not eligible:
        return None
    if selection == "largest":
        return max(eligible, key=lambda index: statistics[index]["area_px"])
    if selection == "leftmost":
        return min(eligible, key=lambda index: statistics[index]["center_px"][0])
    if selection == "rightmost":
        return max(eligible, key=lambda index: statistics[index]["center_px"][0])
    raise ValueError(f"unknown selection strategy: {selection}")


def eligible_candidates(
    statistics: list[dict],
    *,
    min_area: int,
    min_valid_depth_pixels: int,
    roi: tuple[int, int, int, int] | None,
    require_depth: bool = True,
) -> list[int]:
    """Return usable candidates, retaining the same rules as ``select_candidate``."""
    eligible: list[int] = []
    for index, item in enumerate(statistics):
        if (
            item["area_px"] < min_area
        ):
            continue
        if require_depth and (
            item["valid_depth_px"] < min_valid_depth_pixels
            or item["xyz_color_camera_m"] is None
        ):
            continue
        if roi is not None:
            x0, y0, x1, y1 = roi
            center_x, center_y = item["center_px"]
            if not (x0 <= center_x <= x1 and y0 <= center_y <= y1):
                continue
        eligible.append(index)
    return eligible


def main() -> int:
    args = parse_args()
    if (
        not 0.0 <= args.threshold <= 1.0
        or args.min_area < 1
        or args.min_valid_depth_pixels < 1
    ):
        print(
            "参数错误：--threshold 必须在 [0, 1]，面积和有效深度像素下限必须为正数。",
            file=sys.stderr,
        )
        return 2
    try:
        sample = latest_capture_for_scene(args.scene) if args.scene else args.sample
        sample, color_bgr, depth_u16, metadata = load_capture(sample)
        checkpoint = args.checkpoint.expanduser().resolve()
        if not checkpoint.is_file():
            raise RuntimeError(f"找不到 SAM3 权重：{checkpoint}")
        if not torch.cuda.is_available():
            raise RuntimeError("SAM3 定位要求 CUDA GPU，但当前 PyTorch 未检测到 CUDA。")

        output_dir = (args.output_dir or sample / "sam3_localization").expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"sample={sample}")
        print(f"loading_checkpoint={checkpoint}", flush=True)
        model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=args.threshold)
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        sam_input, crop_bounds = center_crop_image(color_rgb)
        image = Image.fromarray(sam_input)
        # SAM3's CUDA path mixes bfloat16 activations with float32 checkpoint
        # weights.  The upstream video inference path and the prior
        # traditional script both run this section under CUDA AMP; without it
        # PyTorch rejects the first BF16/FP32 matrix multiply on RTX 3090.
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            state = processor.set_image(image)
            result = processor.set_text_prompt(args.prompt, state)
        crop_masks = result["masks"].detach().cpu().numpy()
        crop_masks = [np.asarray(mask, dtype=bool).squeeze() for mask in crop_masks]
        masks = expand_crop_masks(crop_masks, full_shape=color_bgr.shape[:2], bounds=crop_bounds)

        intrinsic = metadata["color"]["intrinsic"]
        scale_m = float(metadata["depth"]["scale_m_per_unit"])
        transform = metadata["gdk_extrinsics"]["rgbd_to_left_arm_end_link"]
        statistics = [stats_for_mask(mask, depth_u16, scale_m, intrinsic, transform) for mask in masks]
        roi = parse_roi(args.roi, color_bgr.shape[:2])

        def eligible_candidate(index: int, item: dict) -> bool:
            if (
                item["area_px"] < args.min_area
                or item["valid_depth_px"] < args.min_valid_depth_pixels
                or item["xyz_color_camera_m"] is None
            ):
                return False
            if roi is None:
                return True
            x0, y0, x1, y1 = roi
            center_x, center_y = item["center_px"]
            return x0 <= center_x <= x1 and y0 <= center_y <= y1

        eligible = [
            index for index, item in enumerate(statistics)
            if eligible_candidate(index, item)
        ]
        if args.selection == "largest":
            selected = max(eligible, key=lambda index: statistics[index]["area_px"]) if eligible else None
        elif args.selection == "leftmost":
            selected = min(eligible, key=lambda index: statistics[index]["center_px"][0]) if eligible else None
        else:
            selected = max(eligible, key=lambda index: statistics[index]["center_px"][0]) if eligible else None
        overlay_path = output_dir / "overlay.jpg"
        masks_path = output_dir / "masks.npy"
        json_path = output_dir / "localization.json"
        cv2.imwrite(str(overlay_path), draw_crop_box(draw_overlay(color_bgr, masks, statistics, selected), crop_bounds))
        np.save(masks_path, np.asarray(masks, dtype=bool))
        document = {
            "schema_version": 1,
            "source": {
                "capture_dir": str(sample),
                "scene": metadata.get("scene"),
                "camera_source": metadata.get("camera_source"),
                "color_file": metadata["color"]["file"],
                "aligned_depth_file": metadata["depth"]["aligned_to_color_file"],
            },
            "sam3": {
                "checkpoint": str(checkpoint),
                "prompt": args.prompt,
                "threshold": args.threshold,
                "center_crop_fraction": 0.70,
                "center_crop_xyxy": list(crop_bounds),
                "min_area_px": args.min_area,
                "min_valid_depth_pixels": args.min_valid_depth_pixels,
                "mask_count": len(statistics),
                "selected_mask_index": selected,
                "selection": args.selection,
                "roi_xyxy": roi,
            },
            "coordinate_frames": {
                "xyz_color_camera_m": "left hand color optical frame; computed from GDK-aligned depth and color intrinsics",
                "xyz_left_arm_end_link_m": "kLeftHandRGBDToArmLEndLink applied to xyz_color_camera_m",
            },
            "candidates": statistics,
            "selected": None if selected is None else statistics[selected],
            "artifacts": {"overlay": str(overlay_path), "masks": str(masks_path)},
            "safety": "offline file processing only; no GDK control, robot, or gripper API was called",
        }
        json_path.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(output_dir),
            "mask_count": len(statistics),
            "selected_mask_index": selected,
            "selected": None if selected is None else statistics[selected],
        }, ensure_ascii=False, indent=2))
        print("离线 SAM3 定位完成：未调用 GDK 控制、机器人或夹爪 API。", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"左手 D405 SAM3 定位失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
