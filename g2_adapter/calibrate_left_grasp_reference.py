#!/usr/bin/env python3
"""Calibrate the left-grasp visual reference at a manually placed pose.

This tool only opens the two left-hand GDK camera streams and reads GDK TF.
It never creates a Robot object and contains no motion, mode, or gripper API
call.  With ``--apply`` it updates only the visual reference fields in the
JSON configuration after a usable SAM3 target has been observed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

try:
    from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from .localize_left_d405_sam3 import eligible_candidates, parse_roi, stats_for_mask
    from .sam_crop import center_crop_image, expand_crop_masks
except ImportError:
    from camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from localize_left_d405_sam3 import eligible_candidates, parse_roi, stats_for_mask
    from sam_crop import center_crop_image, expand_crop_masks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_SCALE_M = 0.001


def choose_candidate(statistics: list[dict], candidates: list[int], selection: str) -> int | None:
    """Match the grasp-loop target ranking without importing its control code."""
    if not candidates:
        return None
    if selection == "largest":
        return max(candidates, key=lambda index: statistics[index]["area_px"])
    ordered = sorted(candidates, key=lambda index: float(statistics[index]["center_px"][0]))
    if selection == "leftmost":
        return ordered[0]
    if selection == "second-leftmost":
        return ordered[1] if len(ordered) >= 2 else None
    return ordered[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="在操作员手动放好的左手最终夹取点读取 SAM3 参考；不发送机器人或夹爪命令。"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "g2_adapter" / "left_grasp.example.json")
    parser.add_argument("--prompt", default="silver triangular hinge")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=50)
    parser.add_argument("--roi", default="none", metavar="X0,Y0,X1,Y1")
    parser.add_argument(
        "--selection",
        choices=("largest", "leftmost", "second-leftmost", "rightmost"),
        default="second-leftmost",
        help="当前手工对准的铰链排序；默认第二个从左。",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt")
    parser.add_argument("--apply", action="store_true", help="确认后将当前参考写回 --config；省略时只打印建议值。")
    return parser.parse_args()


def atomic_json_write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    if not (
        0.0 <= args.threshold <= 1.0
        and args.min_area >= 1
        and args.min_valid_depth_pixels >= 1
        and args.warmup >= 0
    ):
        print("参数错误：阈值、面积、有效深度像素和 warmup 无效。", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("未检测到 CUDA，不能运行 SAM3 标定。", file=sys.stderr)
        return 2
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if not checkpoint.is_file() or not config_path.is_file():
        print(f"找不到权重或配置：checkpoint={checkpoint}, config={config_path}", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

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
        color = camera.get_latest_image(color_type, 1000.0)
        depth = camera.get_nearest_image(depth_type, int(color.timestamp_ns), 1000.0)
        color_rgb = decode_color_rgb(color, agibot_gdk)
        depth_u16 = decode_depth_u16(depth, agibot_gdk)
        color_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height))
        depth_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height))
        tf = agibot_gdk.TF()
        depth_to_color = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor))
        rgbd_to_end_link = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink))
        aligned_depth = align_depth_to_color(
            depth_u16,
            depth_intrinsic=depth_intrinsic,
            color_intrinsic=color_intrinsic,
            depth_to_color=depth_to_color,
            depth_scale_m=DEPTH_SCALE_M,
        )
        model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=args.threshold)
        sam_input, crop_bounds = center_crop_image(color_rgb)
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            image_state = processor.set_image(Image.fromarray(sam_input))
            result = processor.set_text_prompt(args.prompt, image_state)
        crop_masks = [np.asarray(mask, dtype=bool).squeeze() for mask in result["masks"].detach().cpu().numpy()]
        masks = expand_crop_masks(crop_masks, full_shape=color_rgb.shape[:2], bounds=crop_bounds)
        scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
        statistics = [stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, rgbd_to_end_link) for mask in masks]
        for item, score in zip(statistics, scores):
            item["sam3_confidence"] = float(score)
        roi = parse_roi(args.roi, color_rgb.shape[:2])
        candidates = eligible_candidates(
            statistics,
            min_area=args.min_area,
            min_valid_depth_pixels=args.min_valid_depth_pixels,
            roi=roi,
            require_depth=True,
        )
        selected_index = choose_candidate(statistics, candidates, args.selection)
        if selected_index is None:
            raise RuntimeError("没有符合面积、深度和 ROI 条件的目标；配置没有被修改")
        selected = statistics[selected_index]
        reference = {
            "target_pixel_px": [float(value) for value in selected["center_px"]],
            "pre_descent_depth_m": float(selected["depth_median_m"]),
            "selection": args.selection,
            "selected_mask_index": int(selected_index),
            "selected": selected,
            "mask_count": len(statistics),
            "eligible_candidate_indices": [int(index) for index in candidates],
            "sam_center_crop_xyxy": list(crop_bounds),
            "sam_center_crop_fraction": 0.70,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        if args.apply:
            document = json.loads(config_path.read_text(encoding="utf-8"))
            if document.get("schema_version") != 1:
                raise RuntimeError("配置 schema_version 必须为 1")
            document.setdefault("alignment", {})["target_pixel_px"] = reference["target_pixel_px"]
            document.setdefault("approach", {})["pre_descent_depth_m"] = reference["pre_descent_depth_m"]
            document["manual_final_grasp_reference"] = reference
            atomic_json_write(config_path, document)
            reference["config_updated"] = str(config_path)
        else:
            reference["config_updated"] = None
        print(json.dumps(reference, ensure_ascii=False, indent=2))
        print("只读视觉标定完成：未创建 Robot、未发送运动、模式或夹爪命令。", file=sys.stderr)
        return 0
    except Exception as exc:
        print(f"左手最终夹取参考标定失败：{exc}", file=sys.stderr)
        return 1
    finally:
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


if __name__ == "__main__":
    raise SystemExit(main())
