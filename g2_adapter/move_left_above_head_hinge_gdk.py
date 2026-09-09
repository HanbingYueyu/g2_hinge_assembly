#!/usr/bin/env python3
"""Use head RGB-D YOLO to place the left gripper 200 mm above the leftmost hinge."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

try:
    from .bilateral_rgbd_geometry import apply_transform, inverse_transform
    from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from .apriltag_workpiece import quaternion_from_rotation
    from .control import CartesianPoseTarget, G2Controller, quaternion_angle_deg
    from .localize_left_d405_sam3 import depth_points
    from .record_pose import DEFAULT_OUTPUT
    from .yolo_hinge import infer_yolo_masks, load_yolo_segmentation_model
except ImportError:
    from bilateral_rgbd_geometry import apply_transform, inverse_transform
    from camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from apriltag_workpiece import quaternion_from_rotation
    from control import CartesianPoseTarget, G2Controller, quaternion_angle_deg
    from localize_left_d405_sam3 import depth_points
    from record_pose import DEFAULT_OUTPUT
    from yolo_hinge import infer_yolo_masks, load_yolo_segmentation_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_SCALE_M = 0.001
HEAD_LINK = "head_link3"
LEFT_END_LINK = "arm_l_end_link"
TF_FALLBACK_MAX_GAP_MS = 100.0
TF_NEAR_LATEST_MS = 10.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-model", type=Path, default=PROJECT_ROOT / "runs/head_hinge_yolo_seg/head_from_scratch_v1/weights/best.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--sample-timeout-s", type=float, default=15.0)
    parser.add_argument("--max-sample-spread-mm", type=float, default=30.0)
    parser.add_argument("--above-mm", type=float, default=200.0)
    parser.add_argument("--tcp-end-mm", default="0,0,145", help="夹持中心在 arm_l_end_link 下的近似 XYZ，毫米")
    parser.add_argument("--poses", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--grasp-preset", default="left_p2", help="仅兼容显式 --use-joint-preset 调用时使用的关节预设。")
    orientation_group = parser.add_mutually_exclusive_group()
    orientation_group.add_argument(
        "--orientation-only", "--fixed-orientation", dest="orientation_only", action="store_true",
        help="在当前末端位置只旋转到固定前向/俯角姿态，再移动到头部目标（默认）。",
    )
    orientation_group.add_argument(
        "--use-joint-preset", dest="orientation_only", action="store_false",
        help="兼容旧版行为：执行 --grasp-preset 关节预设（会改变位置）。",
    )
    parser.set_defaults(orientation_only=True)
    parser.add_argument(
        "--forward-yaw-deg", type=float, default=0.0,
        help="末端前伸轴相对 base_link +X 的水平偏航角，正值朝 +Y。",
    )
    parser.add_argument(
        "--gripper-pitch-deg", type=float, default=30.0,
        help="末端 +Z 前伸轴向下俯角，正值为向下；默认 30 度。",
    )
    parser.add_argument(
        "--gripper-roll-deg", type=float, default=180.0,
        help="绕末端 +Z 前伸轴的滚转角，默认 180 度。",
    )
    parser.add_argument(
        "--orientation-speed-deg-s", type=float, default=20.0,
        help="仅姿态调整时的末端角速度上限，单位 deg/s。",
    )
    parser.add_argument("--preset-velocity-rad-s", type=float, default=0.08)
    parser.add_argument("--speed-mm-s", type=float, default=10.0)
    parser.add_argument("--max-travel-mm", type=float, default=450.0)
    parser.add_argument("--arrival-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--head-xy-tolerance-mm", type=float, default=50.0, help="夹持中心相对头部目标的最大水平误差")
    parser.add_argument("--save-overlay", type=Path, default=PROJECT_ROOT / "outputs/head_leftmost_hinge_localization.jpg")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--allow-motion", action="store_true")
    return parser.parse_args()


def append_log(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **event}, ensure_ascii=False) + "\n")


def load_left_preset(path: Path, name: str) -> list[float]:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    item = document.get("poses", {}).get(name)
    if not isinstance(item, dict) or item.get("side") != "left":
        raise RuntimeError(f"姿态文件缺少左手预设 {name}")
    values = [float(value) for value in item.get("joint_positions_rad", ())]
    if len(values) != 7 or not np.all(np.isfinite(values)):
        raise RuntimeError(f"{name} 必须包含 7 个有限关节值")
    return values


def fixed_forward_down_orientation(
    *, forward_yaw_deg: float = 0.0,
    gripper_pitch_deg: float = 30.0,
    gripper_roll_deg: float = 180.0,
) -> tuple[float, float, float, float]:
    """Build a base-frame orientation with end +Z forward and pitched down.

    G2's end-link convention uses +Z as the gripper's forward extension axis.
    At zero yaw/pitch, +Z points along ``base_link +X`` and +Y points up.  A
    positive pitch tilts +Z downward while preserving zero roll.  Roll is
    applied around the local +Z axis so callers can tune the tool orientation
    without involving a recorded joint preset.
    """
    values = (forward_yaw_deg, gripper_pitch_deg, gripper_roll_deg)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("末端固定姿态参数必须是有限数值")
    if not -180.0 <= float(forward_yaw_deg) <= 180.0:
        raise ValueError("forward_yaw_deg must be in [-180, 180]")
    if not -89.0 <= float(gripper_pitch_deg) <= 89.0:
        raise ValueError("gripper_pitch_deg must be in [-89, 89]")
    if not -180.0 <= float(gripper_roll_deg) <= 180.0:
        raise ValueError("gripper_roll_deg must be in [-180, 180]")
    yaw = math.radians(float(forward_yaw_deg))
    pitch = math.radians(float(gripper_pitch_deg))
    roll = math.radians(float(gripper_roll_deg))
    sin_pitch, cos_pitch = math.sin(pitch), math.cos(pitch)
    # Columns are end-link +X, +Y, +Z in base_link.  +Z is forward and
    # tilts downward by ``pitch``; +Y is the no-roll up direction.
    base_rotation = np.column_stack((
        np.array([0.0, 1.0, 0.0]),
        np.array([sin_pitch, 0.0, cos_pitch]),
        np.array([cos_pitch, 0.0, -sin_pitch]),
    ))
    cy, sy = math.cos(yaw), math.sin(yaw)
    yaw_rotation = np.array(((cy, -sy, 0.0), (sy, cy, 0.0), (0.0, 0.0, 1.0)))
    cr, sr = math.cos(roll), math.sin(roll)
    local_roll = np.array(((cr, -sr, 0.0), (sr, cr, 0.0), (0.0, 0.0, 1.0)))
    return quaternion_from_rotation(yaw_rotation @ base_rotation @ local_roll)


def gripper_end_target(
    hinge_base_m: np.ndarray,
    orientation_xyzw: tuple[float, float, float, float],
    tcp_end_m: np.ndarray,
    above_m: float,
) -> CartesianPoseTarget:
    """Return end-link pose whose approximate grasp center is above the hinge."""
    x, y, z, w = orientation_xyzw
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    rotation = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ])
    grasp_center_base = np.asarray(hinge_base_m, dtype=float) + np.array([0.0, 0.0, float(above_m)])
    end_position = grasp_center_base - rotation @ np.asarray(tcp_end_m, dtype=float)
    return CartesianPoseTarget(tuple(float(v) for v in end_position), orientation_xyzw)


def select_leftmost_hinge(masks, scores, classes):
    candidates = []
    for index, (mask, score, class_id) in enumerate(zip(masks, scores, classes)):
        if int(class_id) != 0:
            continue
        rows, cols = np.where(mask)
        if rows.size:
            candidates.append((float(cols.mean()), index, float(score)))
    return None if not candidates else min(candidates)[1]


def base_from_head_at(tf, timestamp_ns: int, *, max_fallback_gap_ms: float = TF_FALLBACK_MAX_GAP_MS):
    """Use exact image-time TF, with a tightly bounded latest-TF fallback.

    Camera frames can precede the oldest DDS TF sample by a few milliseconds
    at startup.  GDK rejects that exact query.  Falling back is acceptable
    only when the latest transform is close to the image timestamp.
    """
    latest, latest_timestamp = tf.lookup_transform_latest(
        "base_link", HEAD_LINK, return_timestamp=True
    )
    if latest_timestamp is not None:
        gap_ms = abs(int(latest_timestamp) - int(timestamp_ns)) / 1_000_000.0
        # A camera sample commonly leads the first TF sample by 1--2 ms.  Use
        # the already-read near-synchronous pose directly instead of issuing a
        # query that GDK will reject and print as an error.
        if gap_ms <= min(TF_NEAR_LATEST_MS, float(max_fallback_gap_ms)):
            return inverse_transform(transform_to_dict(latest)), "near_latest", gap_ms
    try:
        raw = transform_to_dict(tf.lookup_transform("base_link", HEAD_LINK, int(timestamp_ns)))
        return inverse_transform(raw), "exact_image_time", 0.0
    except Exception as exact_error:
        if latest_timestamp is None:
            raise RuntimeError(f"精确图像时间 TF 查询失败且最新 TF 无时间戳：{exact_error}") from exact_error
        gap_ms = abs(int(latest_timestamp) - int(timestamp_ns)) / 1_000_000.0
        if gap_ms > float(max_fallback_gap_ms):
            raise RuntimeError(
                f"精确图像时间 TF 查询失败，最近 TF 相差 {gap_ms:.1f} ms，"
                f"超过 {max_fallback_gap_ms:.1f} ms：{exact_error}"
            ) from exact_error
        return inverse_transform(transform_to_dict(latest)), "latest_bounded_fallback", gap_ms


def wait_preset(controller: G2Controller, target: list[float], timeout_s: float = 30.0, tolerance_rad: float = 0.03) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        actual = controller.checked_arm_state()[:7]
        if max(abs(a-b) for a, b in zip(actual, target)) <= tolerance_rad:
            return
        time.sleep(0.1)
    raise RuntimeError("左手抓取姿态预设未在限定时间内到位")


def wait_actual_left_target(controller, stream, target: CartesianPoseTarget, *, timeout_s: float,
                            tolerance_m: float, orientation_tolerance_deg: float | None = None,
                            stable_samples: int = 3) -> CartesianPoseTarget:
    """Keep the 50 Hz stream alive until measured end-link feedback is stable."""
    deadline = time.monotonic() + float(timeout_s)
    consecutive = 0
    actual = controller.read_end_effector_pose(LEFT_END_LINK)
    error_m = float("inf")
    orientation_error_deg = float("inf") if orientation_tolerance_deg is not None else 0.0
    if orientation_tolerance_deg is not None and not 0.0 < orientation_tolerance_deg <= 10.0:
        raise ValueError("orientation_tolerance_deg must be in (0, 10]")
    while time.monotonic() < deadline:
        stream.raise_if_failed()
        actual = controller.read_end_effector_pose(LEFT_END_LINK)
        error_m = float(np.linalg.norm(
            np.asarray(actual.position_m) - np.asarray(target.position_m)
        ))
        if orientation_tolerance_deg is not None:
            orientation_error_deg = quaternion_angle_deg(
                actual.orientation_xyzw, target.orientation_xyzw
            )
        if error_m <= tolerance_m and (
            orientation_tolerance_deg is None or orientation_error_deg <= orientation_tolerance_deg
        ):
            consecutive += 1
            if consecutive >= stable_samples:
                return actual
        else:
            consecutive = 0
        time.sleep(0.10)
    status = controller.motion_status_summary()
    raise RuntimeError(
        f"左手实际末端未稳定到位：last_error_mm={error_m*1000.0:.1f}, "
        f"last_orientation_error_deg={orientation_error_deg:.2f}, "
        f"target={list(target.position_m)}, actual={list(actual.position_m)}, status={status}"
    )


def main() -> int:
    args = parse_args()
    log_path = (args.log or PROJECT_ROOT / "logs" / f"head_hinge_approach_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl").expanduser().resolve()
    try:
        tcp_end_m = np.asarray([float(v) for v in args.tcp_end_mm.split(",")], dtype=float) / 1000.0
        if tcp_end_m.shape != (3,) or not np.all(np.isfinite(tcp_end_m)):
            raise ValueError
    except ValueError:
        print("--tcp-end-mm 必须是三个有限数值，例如 0,0,145", file=sys.stderr)
        return 2
    if not (0 < args.confidence <= 1 and 1 <= args.samples <= 30 and 0 < args.above_mm <= 500
            and 0 < args.speed_mm_s <= 50 and 0 < args.max_travel_mm <= 800
            and 1 <= args.arrival_tolerance_mm <= 30 and 1 <= args.head_xy_tolerance_mm <= 100
            and 1 <= args.orientation_speed_deg_s <= math.degrees(1.0)):
        print("置信度、采样数、高度、速度、最大行程、到位容差或姿态速度参数无效", file=sys.stderr)
        return 2
    try:
        fixed_orientation = fixed_forward_down_orientation(
            forward_yaw_deg=args.forward_yaw_deg,
            gripper_pitch_deg=args.gripper_pitch_deg,
            gripper_roll_deg=args.gripper_roll_deg,
        )
        preset = None if args.orientation_only else load_left_preset(args.poses, args.grasp_preset)
        model = load_yolo_segmentation_model(args.yolo_model, device=args.device)
        import agibot_gdk as gdk
    except Exception as exc:
        print(f"启动失败：{exc}", file=sys.stderr)
        return 2
    initialized = False
    camera = None
    try:
        if gdk.gdk_init() != gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        color_type, depth_type = gdk.CameraType.kHeadColor, gdk.CameraType.kHeadDepth
        camera = gdk.Camera([color_type, depth_type])
        tf = gdk.TF()
        depth_to_color = transform_to_dict(tf.get_tf_from_sensor(gdk.SensorExtrinsicType.kHeadDepthToHeadColor))
        head_from_color = transform_to_dict(tf.get_tf_from_sensor(gdk.SensorExtrinsicType.kHeadRGBDToHeadLink3))
        color_intrinsic = depth_intrinsic = None
        observations = []
        tf_modes = []
        tf_gaps_ms = []
        last_timestamp = None
        deadline = time.monotonic() + args.sample_timeout_s
        last_overlay = None
        while len(observations) < args.samples and time.monotonic() < deadline:
            color = camera.get_latest_image(color_type, 1000.0)
            if int(color.timestamp_ns) == last_timestamp:
                time.sleep(0.01)
                continue
            last_timestamp = int(color.timestamp_ns)
            # Latch the dynamic head pose before YOLO inference.  Inference can
            # take hundreds of milliseconds; asking for "latest" afterwards
            # would compare this image against a much newer robot pose and can
            # incorrectly exceed the bounded fallback window.
            base_from_head, tf_mode, tf_gap_ms = base_from_head_at(tf, last_timestamp)
            depth = camera.get_nearest_image(depth_type, last_timestamp, 1000.0)
            bgr = cv2.cvtColor(decode_color_rgb(color, gdk), cv2.COLOR_RGB2BGR)
            depth_u16 = decode_depth_u16(depth, gdk)
            if color_intrinsic is None:
                color_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height))
                depth_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height))
            aligned = align_depth_to_color(depth_u16, depth_intrinsic=depth_intrinsic, color_intrinsic=color_intrinsic,
                                           depth_to_color=depth_to_color, depth_scale_m=DEPTH_SCALE_M)
            masks, scores, classes = infer_yolo_masks(model, bgr, confidence=args.confidence, imgsz=args.imgsz, device=args.device)
            selected = select_leftmost_hinge(masks, scores, classes)
            if selected is None:
                continue
            inliers, points = depth_points(masks[selected], aligned, DEPTH_SCALE_M, color_intrinsic)
            if len(points) < 20:
                continue
            point_color = np.median(points, axis=0)
            point_base = np.asarray(apply_transform(base_from_head, apply_transform(head_from_color, point_color)), dtype=float)
            observations.append(point_base)
            tf_modes.append(tf_mode)
            tf_gaps_ms.append(tf_gap_ms)
            rows, cols = np.where(masks[selected])
            last_overlay = bgr.copy()
            last_overlay[masks[selected]] = (0.65 * last_overlay[masks[selected]] + 0.35 * np.array([0, 255, 0])).astype(np.uint8)
            cv2.rectangle(last_overlay, (int(cols.min()), int(rows.min())), (int(cols.max()), int(rows.max())), (0, 255, 0), 2)
            cv2.putText(last_overlay, f"leftmost hinge {len(observations)}/{args.samples} depth={point_color[2]:.3f}m",
                        (int(cols.min()), max(22, int(rows.min())-7)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 2)
        if len(observations) < args.samples:
            raise RuntimeError(f"{args.sample_timeout_s:.1f}s 内仅取得 {len(observations)}/{args.samples} 个有效左侧铰链深度样本")
        samples = np.asarray(observations)
        hinge_base = np.median(samples, axis=0)
        spread_mm = float(np.max(np.linalg.norm(samples - hinge_base, axis=1)) * 1000.0)
        if spread_mm > args.max_sample_spread_mm:
            raise RuntimeError(f"铰链定位样本离散 {spread_mm:.1f} mm，超过 {args.max_sample_spread_mm:.1f} mm")
        if last_overlay is not None:
            args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(args.save_overlay), last_overlay)
        result = {"event":"head_leftmost_hinge_localized", "motion_enabled":bool(args.allow_motion),
                  "samples":len(observations), "sample_spread_mm":round(spread_mm,2),
                  "tf_modes":tf_modes, "max_tf_fallback_gap_ms":round(max(tf_gaps_ms),3),
                  "hinge_position_base_link_m":[float(v) for v in hinge_base], "above_mm":args.above_mm,
                  "tcp_end_mm":[float(v*1000) for v in tcp_end_m],
                  "grasp_preset":None if args.orientation_only else args.grasp_preset,
                  "orientation_only":bool(args.orientation_only),
                  "fixed_orientation_xyzw":list(fixed_orientation),
                  "overlay":str(args.save_overlay.resolve())}
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        append_log(log_path, result)
        if not args.allow_motion:
            print("只读定位完成；追加 --allow-motion 才会移动左手。")
            return 0
        robot = gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(gdk, robot, allow_motion=True)
        controller.set_control_mode(1)
        current = controller.read_end_effector_pose(LEFT_END_LINK)
        if args.orientation_only:
            orientation_target = CartesianPoseTarget(
                current.position_m, fixed_orientation,
            )
            orientation_error_deg = quaternion_angle_deg(
                current.orientation_xyzw, fixed_orientation,
            )
            orientation_start = {
                "event": "left_fixed_orientation_start",
                "position_held_m": list(current.position_m),
                "target_orientation_xyzw": list(fixed_orientation),
                "initial_orientation_error_deg": round(orientation_error_deg, 3),
                "orientation_speed_deg_s": args.orientation_speed_deg_s,
                "forward_yaw_deg": args.forward_yaw_deg,
                "gripper_pitch_deg": args.gripper_pitch_deg,
                "gripper_roll_deg": args.gripper_roll_deg,
            }
            print(json.dumps(orientation_start, ensure_ascii=False), flush=True)
            append_log(log_path, orientation_start)
            if orientation_error_deg > 0.05:
                orientation_stream = controller.start_left_end_effector_position_stream(
                    current,
                    rate_hz=50.0,
                    life_time_s=0.10,
                    # The position target is exactly the measured position;
                    # this nonzero limit is required by the stream validator.
                    max_translation_velocity_mps=0.001,
                    allow_orientation_change=True,
                    max_angular_velocity_rad_s=math.radians(args.orientation_speed_deg_s),
                )
                try:
                    orientation_stream.update_target(orientation_target)
                    orientation_timeout_s = max(
                        5.0, orientation_error_deg / args.orientation_speed_deg_s + 3.0,
                    )
                    if not orientation_stream.wait_until_target(orientation_timeout_s):
                        raise RuntimeError("左手末端姿态调整命令未在限定时间内完成")
                    current = wait_actual_left_target(
                        controller,
                        orientation_stream,
                        orientation_target,
                        timeout_s=orientation_timeout_s + 5.0,
                        tolerance_m=0.005,
                        orientation_tolerance_deg=2.0,
                    )
                finally:
                    orientation_stream.stop()
            else:
                print("左手末端已接近记录姿态；保持当前位置，不发送姿态调整目标。", flush=True)
            orientation_done = {
                "event": "left_fixed_orientation_complete",
                "position_held_m": list(current.position_m),
                "actual_orientation_xyzw": list(current.orientation_xyzw),
                "orientation_error_deg": round(
                    quaternion_angle_deg(current.orientation_xyzw, fixed_orientation), 3,
                ),
            }
            print(json.dumps(orientation_done, ensure_ascii=False), flush=True)
            append_log(log_path, orientation_done)
        else:
            # Backward-compatible path for callers that have not opted into
            # the orientation-only coarse approach.
            assert preset is not None
            controller.move_preset(
                "left", preset, velocity_rad_s=args.preset_velocity_rad_s, life_time_s=30.0,
                detail="head YOLO leftmost hinge grasp orientation",
            )
            wait_preset(controller, preset)
            controller.set_control_mode(1)
            current = controller.read_end_effector_pose(LEFT_END_LINK)
        target = gripper_end_target(hinge_base, current.orientation_xyzw, tcp_end_m, args.above_mm / 1000.0)
        travel_mm = float(np.linalg.norm(np.asarray(target.position_m)-np.asarray(current.position_m))*1000.0)
        if travel_mm > args.max_travel_mm:
            raise RuntimeError(f"当前末端到头部目标直线距离 {travel_mm:.1f} mm，超过 --max-travel-mm={args.max_travel_mm:.1f}")
        motion_start = {"event":"left_above_hinge_motion_start", "travel_mm":round(travel_mm,2),
                        "target_end_link_base_m":list(target.position_m), "orientation_xyzw":list(target.orientation_xyzw)}
        print(json.dumps(motion_start, ensure_ascii=False), flush=True)
        append_log(log_path, motion_start)
        stream = controller.start_left_end_effector_position_stream(current, rate_hz=50.0, life_time_s=0.10,
                                                                     max_translation_velocity_mps=args.speed_mm_s/1000.0)
        try:
            stream.update_target(target)
            if not stream.wait_until_target(travel_mm/args.speed_mm_s + 2.0):
                raise RuntimeError("左手位姿流未在限定时间内发送到目标")
            actual = wait_actual_left_target(
                controller, stream, target,
                timeout_s=max(5.0, travel_mm / args.speed_mm_s),
                tolerance_m=args.arrival_tolerance_mm / 1000.0,
            )
            error_mm = float(np.linalg.norm(np.asarray(actual.position_m)-np.asarray(target.position_m))*1000.0)
            actual_center = gripper_end_target(
                np.asarray(actual.position_m), actual.orientation_xyzw, -tcp_end_m, 0.0
            ).position_m
            xy_error_mm = float(np.linalg.norm(np.asarray(actual_center[:2]) - hinge_base[:2]) * 1000.0)
            if xy_error_mm > args.head_xy_tolerance_mm:
                raise RuntimeError(
                    f"夹持中心与头部铰链目标水平误差 {xy_error_mm:.1f} mm，"
                    f"超过 {args.head_xy_tolerance_mm:.1f} mm；不进入左手视觉伺服"
                )
            arrived = {"event":"left_above_hinge_arrived", "position_error_mm":round(error_mm,2),
                       "grasp_center_xy_error_mm":round(xy_error_mm,2),
                       "actual_end_link_base_m":list(actual.position_m)}
            print(json.dumps(arrived, ensure_ascii=False))
            append_log(log_path, arrived)
        finally:
            stream.stop()
        return 0
    except Exception as exc:
        append_log(log_path, {"event":"head_hinge_approach_failed", "error":str(exc)})
        print(f"头部铰链定位/左手预移动失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            try: camera.close_camera()
            except Exception: pass
        if initialized:
            try: gdk.gdk_release()
            except Exception: pass


if __name__ == "__main__":
    raise SystemExit(main())
