#!/usr/bin/env python3
"""Guarded one-step GDK end-pose XY mapping calibration for the left D405.

The operator places the left arm at P3 before every run. With
``--allow-motion`` this program moves the same GDK mode-1 end-link controller
used by ``left_grasp_gdk.py`` exactly once: a 1 mm interpolated base_link X
or Y translation at 50 Hz. It never uses the retired custom URDF IK / joint-
servo path. It never commands P2/P3, a gripper, descent, mode change, return
movement, or target reacquisition.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

try:
    from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from .control import CartesianPoseTarget, G2Controller
    from .left_grasp_gdk import choose_candidate
    from .localize_left_d405_sam3 import eligible_candidates, parse_roi, stats_for_mask
    from .sam_crop import center_crop_image, expand_crop_masks
    from .xy_mapping import choose_locked_candidate, single_axis_base_displacement
except ImportError:
    from camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from control import CartesianPoseTarget, G2Controller
    from left_grasp_gdk import choose_candidate
    from localize_left_d405_sam3 import eligible_candidates, parse_roi, stats_for_mask
    from sam_crop import center_crop_image, expand_crop_masks
    from xy_mapping import choose_locked_candidate, single_axis_base_displacement


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_SCALE_M = 0.001
LEFT_END_LINK = "arm_l_end_link"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="左手 D405 的 GDK 末端位姿单次 1 mm XY 标定")
    parser.add_argument("--axis", choices=("x", "y"), required=True, help="要测试的 base_link 水平轴。")
    parser.add_argument("--direction", choices=("positive", "negative"), default="positive", help="该轴的正向或负向。")
    parser.add_argument("--step-m", type=float, default=0.001, help="单次位移；安全上限固定为 0.001 m。")
    parser.add_argument("--duration-s", type=float, default=0.50, help="50 Hz 插补持续时间，0.2..2.0 秒。")
    parser.add_argument("--settle-s", type=float, default=0.8, help="唯一小步结束后、后测量前的等待时间。")
    parser.add_argument("--prompt", default="silver triangular hinge")
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--min-area", type=int, default=1)
    parser.add_argument("--min-valid-depth-pixels", type=int, default=50)
    parser.add_argument("--roi", default="none", metavar="X0,Y0,X1,Y1")
    parser.add_argument("--selection", choices=("largest", "leftmost", "second-leftmost", "rightmost"), default="leftmost")
    parser.add_argument("--stable-frames", type=int, default=3, help="动作前、后各需连续锁定的有效帧数。")
    parser.add_argument("--stability-radius-px", type=float, default=8.0, help="静止时同一目标中心允许的最大偏差。")
    parser.add_argument("--post-lock-radius-px", type=float, default=40.0, help="动作后仅接受初始目标附近的候选，绝不重获取。")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "checkpoints" / "sam3.1_multiplex.pt")
    parser.add_argument("--log", type=Path, default=None, help="默认写入 logs/left_xy_mapping_end_pose_*.jsonl")
    parser.add_argument("--allow-motion", action="store_true", help="唯一运动确认参数；未指定时严格只读。")
    return parser.parse_args()


def record_json(stream, event: dict) -> None:
    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    stream.flush()


def valid_args(args: argparse.Namespace) -> bool:
    return (
        0.0 <= args.threshold <= 1.0
        and args.min_area >= 1
        and args.min_valid_depth_pixels >= 1
        and args.stable_frames >= 2
        and args.stability_radius_px > 0.0
        and args.post_lock_radius_px > 0.0
        and 0.2 <= args.duration_s <= 2.0
        and 0.0 <= args.settle_s <= 5.0
        and args.warmup >= 0
    )


def next_target_measurement(
    *, agibot_gdk, camera, tf, processor, args: argparse.Namespace,
    color_intrinsic: dict | None, depth_intrinsic: dict | None,
    depth_to_color: dict | None, end_link_transform: dict | None,
    locked_center_px: np.ndarray | None, static_reference_px: np.ndarray | None,
) -> tuple[dict | None, dict, dict | None, dict | None, dict | None, dict | None]:
    """Infer one frame and retain the original target only when requested."""
    color_type = agibot_gdk.CameraType.kHandLeftColor
    depth_type = agibot_gdk.CameraType.kHandLeftDepth
    color = camera.get_latest_image(color_type, 1000.0)
    depth = camera.get_nearest_image(depth_type, int(color.timestamp_ns), 1000.0)
    color_rgb = decode_color_rgb(color, agibot_gdk)
    depth_u16 = decode_depth_u16(depth, agibot_gdk)
    if color_intrinsic is None:
        color_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height))
        depth_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height))
        depth_to_color = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor))
        end_link_transform = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink))
    assert color_intrinsic is not None and depth_intrinsic is not None and depth_to_color is not None and end_link_transform is not None
    aligned_depth = align_depth_to_color(depth_u16, depth_intrinsic=depth_intrinsic, color_intrinsic=color_intrinsic, depth_to_color=depth_to_color, depth_scale_m=DEPTH_SCALE_M)
    sam_input, crop_bounds = center_crop_image(color_rgb)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        image_state = processor.set_image(Image.fromarray(sam_input))
        result = processor.set_text_prompt(args.prompt, image_state)
    crop_masks = [np.asarray(mask, dtype=bool).squeeze() for mask in result["masks"].detach().cpu().numpy()]
    masks = expand_crop_masks(crop_masks, full_shape=color_rgb.shape[:2], bounds=crop_bounds)
    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
    statistics = [stats_for_mask(mask, aligned_depth, DEPTH_SCALE_M, color_intrinsic, end_link_transform) for mask in masks]
    for item, score in zip(statistics, scores):
        item["sam3_confidence"] = float(score)
    candidates = eligible_candidates(statistics, min_area=args.min_area, min_valid_depth_pixels=args.min_valid_depth_pixels, roi=parse_roi(args.roi, color_rgb.shape[:2]), require_depth=True)
    if locked_center_px is None:
        selected = choose_candidate(statistics, candidates, args.selection)
        tracking = "initial_selection" if selected is not None else "no_target"
    else:
        selected = choose_locked_candidate(statistics, candidates, reference_center_px=locked_center_px, max_distance_px=args.post_lock_radius_px)
        tracking = "locked" if selected is not None else "lock_lost_no_reacquisition"
    event = {"event": "measurement_frame", "utc": datetime.now(timezone.utc).isoformat(), "tracking": tracking, "mask_count": len(statistics), "selected_mask_index": selected, "selected": None if selected is None else statistics[selected], "sam_center_crop_xyxy": list(crop_bounds), "sam_center_crop_fraction": 0.70}
    if selected is None:
        return None, event, color_intrinsic, depth_intrinsic, depth_to_color, end_link_transform
    item = statistics[selected]
    center = np.asarray(item["center_px"], dtype=np.float64)
    if static_reference_px is not None and float(np.linalg.norm(center - static_reference_px)) > args.stability_radius_px:
        event["tracking"] = "target_not_static"
        event["distance_from_static_reference_px"] = float(np.linalg.norm(center - static_reference_px))
        return None, event, color_intrinsic, depth_intrinsic, depth_to_color, end_link_transform
    return item, event, color_intrinsic, depth_intrinsic, depth_to_color, end_link_transform


def stable_measurement(*, agibot_gdk, camera, tf, processor, args: argparse.Namespace, log, color_intrinsic, depth_intrinsic, depth_to_color, end_link_transform, locked_center_px: np.ndarray | None):
    """Collect a static target observation; failure sends no extra command."""
    samples: list[dict] = []
    initial_center: np.ndarray | None = None
    while len(samples) < args.stable_frames:
        item, event, color_intrinsic, depth_intrinsic, depth_to_color, end_link_transform = next_target_measurement(
            agibot_gdk=agibot_gdk, camera=camera, tf=tf, processor=processor, args=args,
            color_intrinsic=color_intrinsic, depth_intrinsic=depth_intrinsic, depth_to_color=depth_to_color,
            end_link_transform=end_link_transform, locked_center_px=locked_center_px, static_reference_px=initial_center,
        )
        record_json(log, event)
        print(json.dumps(event, ensure_ascii=False))
        if item is None:
            samples.clear()
            initial_center = None
            continue
        if initial_center is None:
            initial_center = np.asarray(item["center_px"], dtype=np.float64)
            if locked_center_px is None:
                locked_center_px = initial_center.copy()
        samples.append(item)
    center = np.median(np.asarray([item["center_px"] for item in samples], dtype=np.float64), axis=0)
    representative = min(samples, key=lambda item: float(np.linalg.norm(np.asarray(item["center_px"], dtype=np.float64) - center)))
    return representative, center, color_intrinsic, depth_intrinsic, depth_to_color, end_link_transform


def send_interpolated_end_pose_step(controller: G2Controller, *, before: CartesianPoseTarget, requested_delta: np.ndarray, duration_s: float) -> int:
    """Issue one smooth GDK mode-1 base-link step at the documented 50 Hz."""
    target = CartesianPoseTarget(tuple(float(value) for value in (np.asarray(before.position_m) + requested_delta)), before.orientation_xyzw)
    steps = max(1, int(round(duration_s * 50.0)))
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        fraction = index / steps
        position = tuple(float(start + fraction * (end - start)) for start, end in zip(before.position_m, target.position_m))
        controller.send_left_end_effector_position_setpoint(CartesianPoseTarget(position, before.orientation_xyzw), life_time_s=0.10)
        deadline += duration_s / steps
        delay = deadline - time.monotonic()
        if delay > 0.0:
            time.sleep(delay)
    return steps


def main() -> int:
    args = parse_args()
    if not valid_args(args):
        print("参数错误：标定参数超出安全范围。", file=sys.stderr)
        return 2
    try:
        requested_base_delta = single_axis_base_displacement(args.axis, args.direction, args.step_m)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    if not torch.cuda.is_available():
        print("未检测到 CUDA，不能运行实时 SAM3。", file=sys.stderr)
        return 2
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        print(f"找不到 SAM3 权重：{checkpoint}", file=sys.stderr)
        return 2
    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    log_path = (args.log or PROJECT_ROOT / "logs" / f"left_xy_mapping_end_pose_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl").expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    initialized = False
    camera = None
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        controller.checked_arm_state()
        camera = agibot_gdk.Camera([agibot_gdk.CameraType.kHandLeftColor, agibot_gdk.CameraType.kHandLeftDepth])
        for _ in range(args.warmup):
            camera.get_latest_image(agibot_gdk.CameraType.kHandLeftColor, 1000.0)
        tf = agibot_gdk.TF()
        model = build_sam3_image_model(checkpoint_path=str(checkpoint), load_from_HF=False)
        processor = Sam3Processor(model, confidence_threshold=args.threshold)
        with log_path.open("a", encoding="utf-8") as log:
            start = {
                "event": "start", "utc": datetime.now(timezone.utc).isoformat(), "mode": "single_motion" if args.allow_motion else "read_only_preview",
                "controller": "gdk_mode1_end_effector_pose_control", "frame": LEFT_END_LINK, "required_start_pose": "left_p3",
                "requested_base_delta_m": [float(value) for value in requested_base_delta],
                "safety": "one interpolated 1 mm GDK end-pose step at most; no gripper, descent, reacquisition, return, or GDK mode change",
            }
            record_json(log, start)
            print(json.dumps({**start, "log": str(log_path)}, ensure_ascii=False))
            before, before_center, color_intrinsic, depth_intrinsic, depth_to_color, end_link_transform = stable_measurement(
                agibot_gdk=agibot_gdk, camera=camera, tf=tf, processor=processor, args=args, log=log,
                color_intrinsic=None, depth_intrinsic=None, depth_to_color=None, end_link_transform=None, locked_center_px=None,
            )
            before_pose = controller.read_end_effector_pose(LEFT_END_LINK)
            preflight = {
                "event": "preflight", "before_center_px": [float(value) for value in before_center],
                "before_end_link_base_link": list(before_pose.position_m), "requested_base_delta_m": [float(value) for value in requested_base_delta],
                "command_controller": "gdk_mode1_end_effector_pose_control",
            }
            if not args.allow_motion:
                preflight["motion_command_sent"] = False
                record_json(log, preflight)
                print(json.dumps(preflight, ensure_ascii=False, indent=2))
                print("只读标定预检完成：未发送运动或夹爪命令。")
                return 0
            status = robot.get_motion_control_status()
            if int(status.control_mode) != 1 or int(status.error_code) != 0:
                raise RuntimeError(f"标定要求健康的 control_mode=1，当前 control_mode={int(status.control_mode)}, error={int(status.error_code)}")
            target_position = np.asarray(before_pose.position_m) + requested_base_delta
            preflight["requested_end_link_target_base_link"] = [float(value) for value in target_position]
            record_json(log, preflight)
            print(json.dumps(preflight, ensure_ascii=False, indent=2))
            commands_sent = send_interpolated_end_pose_step(controller, before=before_pose, requested_delta=requested_base_delta, duration_s=args.duration_s)
            record_json(log, {"event": "single_end_pose_command_sent", "utc": datetime.now(timezone.utc).isoformat(), "commands_sent": commands_sent})
            print("唯一的 GDK 末端位姿 1 mm 命令已发送；脚本不会再发送任何动作命令。")
            time.sleep(args.settle_s)
            after, after_center, _, _, _, _ = stable_measurement(
                agibot_gdk=agibot_gdk, camera=camera, tf=tf, processor=processor, args=args, log=log,
                color_intrinsic=color_intrinsic, depth_intrinsic=depth_intrinsic, depth_to_color=depth_to_color,
                end_link_transform=end_link_transform, locked_center_px=before_center,
            )
            after_pose = controller.read_end_effector_pose(LEFT_END_LINK)
            actual_delta = np.asarray(after_pose.position_m) - np.asarray(before_pose.position_m)
            result = {
                "event": "result", "controller": "gdk_mode1_end_effector_pose_control", "frame": LEFT_END_LINK,
                "requested_base_delta_m": [float(value) for value in requested_base_delta],
                "actual_end_link_base_delta_m": [float(value) for value in actual_delta],
                "before_center_px": [float(value) for value in before_center], "after_center_px": [float(value) for value in after_center],
                "pixel_delta_post_minus_pre_px": [float(value) for value in (after_center - before_center)],
                "before_target": before, "after_target": after, "commands_sent": commands_sent,
                "safety": "completed after exactly one GDK end-pose step; no gripper, descent, reacquisition, return, or GDK mode change",
            }
            record_json(log, result)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except KeyboardInterrupt:
        print("收到 Ctrl-C；不会发送任何额外动作命令。")
        return 0
    except Exception as exc:
        print(f"左手 GDK 末端位姿 XY 标定失败：{exc}", file=sys.stderr)
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    raise SystemExit(main())
