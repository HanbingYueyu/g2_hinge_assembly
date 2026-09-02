#!/usr/bin/env python3
"""Post-P4 bilateral XY alignment: right camera observes, left arm moves.

The process assumes the existing left grasp flow has reached P4.  It moves
the right arm once to ``right_post_grasp`` (only with ``--allow-motion``),
then uses the right-hand ``hole`` detector and left-hand ``shaft`` detector to
make bounded, discrete XY updates to the left end-link in GDK mode 1.
``--geometry-mode direct`` uses the official RGB-D extrinsics from both hands
and their dynamic base transforms; legacy ``veto`` keeps pixel-J as the
controller and lets the old right-image RGB-D path deny unsafe corrections.
No Z motion, force action, gripper command, insertion, or automatic regrasp is
implemented here.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    from .bilateral_rgbd_geometry import PairGeometry, apply_transform, estimate_box_depth, estimate_pair_geometry, estimate_point_depth
    from .bilateral_servo import (
        AlignmentPolicy,
        BilateralAlignmentGate,
        BilateralMetricAlignmentGate,
        MetricAlignmentPolicy,
        metric_and_pixel_corrections_agree,
    )
    from .camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from .control import CartesianPoseTarget, G2Controller
    from .hole_target_detector import BoxDetection, infer_detection, load_detection_model, pair_status
    from .record_pose import DEFAULT_OUTPUT
    from .right_hole_target_detect_gdk import RIGHT_ARM_OFFSET, SharedView, handler_class
    from .yolo_hinge import infer_yolo_masks, load_yolo_segmentation_model
except ImportError:
    from bilateral_rgbd_geometry import PairGeometry, apply_transform, estimate_box_depth, estimate_pair_geometry, estimate_point_depth
    from bilateral_servo import (
        AlignmentPolicy,
        BilateralAlignmentGate,
        BilateralMetricAlignmentGate,
        MetricAlignmentPolicy,
        metric_and_pixel_corrections_agree,
    )
    from camera import align_depth_to_color, decode_color_rgb, decode_depth_u16, intrinsic_to_dict, transform_to_dict
    from control import CartesianPoseTarget, G2Controller
    from hole_target_detector import BoxDetection, infer_detection, load_detection_model, pair_status
    from record_pose import DEFAULT_OUTPUT
    from right_hole_target_detect_gdk import RIGHT_ARM_OFFSET, SharedView, handler_class
    from yolo_hinge import infer_yolo_masks, load_yolo_segmentation_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hole_target_det_v2" / "best.pt"
DEFAULT_LEFT_TARGET_MODEL = PROJECT_ROOT / "models" / "left_d405_hinge_shaft_yolo" / "best.pt"
LEFT_END_LINK = "arm_l_end_link"
LEFT_HINGE_CLASS_ID = 0
LEFT_SHAFT_CLASS_ID = 1


def parse_jacobian(value: str | None) -> list[list[float]] | None:
    if value is None or not value.strip():
        return None
    try:
        numbers = [float(item) for item in re.split(r"[,;\s]+", value.strip()) if item]
    except ValueError as exc:
        raise ValueError("--pixel-jacobian 必须是 4 个数字：J00,J01,J10,J11") from exc
    if len(numbers) != 4 or not all(math.isfinite(item) for item in numbers):
        raise ValueError("--pixel-jacobian 必须是 4 个有限数字：J00,J01,J10,J11")
    return [numbers[:2], numbers[2:]]


def load_pose(path: Path, name: str, side: str) -> list[float]:
    document = json.loads(path.expanduser().read_text(encoding="utf-8"))
    item = document.get("poses", {}).get(name)
    if not isinstance(item, dict) or item.get("side") != side:
        raise RuntimeError(f"姿态文件缺少 {side} 侧 {name}")
    values = [float(value) for value in item.get("joint_positions_rad", ())]
    if len(values) != 7 or not all(math.isfinite(value) for value in values):
        raise RuntimeError(f"{name} 必须包含 7 个有限关节值")
    return values


@dataclass(frozen=True)
class LeftTargetObservation:
    """A shaft mask associated with one visible hinge; tip is the mating point."""

    box: BoxDetection
    tip_px: tuple[float, float]
    mask: np.ndarray

    @property
    def class_id(self) -> int:
        return self.box.class_id

    @property
    def class_name(self) -> str:
        return self.box.class_name

    @property
    def confidence(self) -> float:
        return self.box.confidence

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return self.box.xyxy

    @property
    def center_px(self) -> tuple[float, float]:
        return self.tip_px


def infer_left_target_boxes(model, frame_bgr, *, confidence: float, imgsz: int, device: str) -> list[LeftTargetObservation]:
    """Use left hinge/shaft segmentation; expose an associated shaft tip as target."""
    masks, scores, class_ids = infer_yolo_masks(
        model,
        frame_bgr,
        confidence=confidence,
        imgsz=imgsz,
        device=device,
    )
    hinge_centers: list[np.ndarray] = []
    shaft_items: list[tuple[np.ndarray, float]] = []
    for mask, score, class_id in zip(masks, scores, class_ids):
        mask_bool = np.asarray(mask, dtype=bool)
        rows, cols = np.where(mask_bool)
        if rows.size == 0:
            continue
        if int(class_id) == LEFT_HINGE_CLASS_ID:
            hinge_centers.append(np.array((float(cols.mean()), float(rows.mean())), dtype=np.float64))
        elif int(class_id) == LEFT_SHAFT_CLASS_ID:
            shaft_items.append((mask_bool, float(score)))
    if len(hinge_centers) != 1 or not shaft_items:
        return []
    hinge_center = hinge_centers[0]
    ranked: list[tuple[float, np.ndarray, float]] = []
    for mask, score in shaft_items:
        rows, cols = np.where(mask)
        center = np.array((float(cols.mean()), float(rows.mean())), dtype=np.float64)
        ranked.append((float(np.linalg.norm(center - hinge_center)), mask, score))
    ranked.sort(key=lambda item: item[0])
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 20.0:
        return []
    _, mask, score = ranked[0]
    rows, cols = np.where(mask)
    points = np.column_stack((cols.astype(np.float64), rows.astype(np.float64)))
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = vh[0]
    projections = centered @ axis
    endpoints = (points[int(np.argmin(projections))], points[int(np.argmax(projections))])
    tip = max(endpoints, key=lambda point: float(np.linalg.norm(point - hinge_center)))
    x0, x1 = float(cols.min()), float(cols.max())
    y0, y1 = float(rows.min()), float(rows.max())
    return [
        LeftTargetObservation(
            box=BoxDetection(
                class_id=LEFT_SHAFT_CLASS_ID,
                class_name="target",
                confidence=score,
                xyxy=(x0, y0, x1, y1),
            ),
            tip_px=(float(tip[0]), float(tip[1])),
            mask=mask,
        )
    ]


def left_target_from_offset(initial_pose: CartesianPoseTarget, offset_m: Sequence[float]) -> CartesianPoseTarget:
    """Build every left target from one measured origin; keep Z/orientation fixed."""
    values = tuple(float(value) for value in offset_m)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("left XY offset must contain three finite values")
    if abs(values[2]) > 1e-12:
        raise ValueError("bilateral alignment is XY-only; Z offset must be zero")
    return CartesianPoseTarget(
        tuple(float(initial_pose.position_m[index] + values[index]) for index in range(3)),
        initial_pose.orientation_xyzw,
    )


def draw_alignment_overlay(
    frame_bgr,
    detections,
    selected,
    state: str,
    error: tuple[float, float] | None,
    metric_error: tuple[float, float, float] | None = None,
):
    image = frame_bgr.copy()
    colors = {"hole": (0, 255, 0), "target": (0, 128, 255)}
    for item in detections:
        x0, y0, x1, y1 = [int(value) for value in item.xyxy]
        color = colors.get(item.class_name, (255, 0, 0))
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cx, cy = item.center_px
        cv2.circle(image, (int(cx), int(cy)), 4, color, -1)
        cv2.putText(image, f"{item.class_name} {item.confidence:.2f}", (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(image, f"bilateral={state}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    if error is not None:
        cv2.putText(image, f"target-hole=({error[0]:.1f},{error[1]:.1f}) px", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    if metric_error is not None:
        cv2.putText(
            image,
            "metric target-hole=(%.1f,%.1f,%.1f) mm" % tuple(1000.0 * float(value) for value in metric_error),
            (10, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )
    return image


def compact_frame(record: dict) -> dict:
    keys = (
        "event", "frame", "utc", "state", "reason", "motion_enabled",
        "sequence", "detections", "left_detections", "pair_status", "pixel_error_px",
        "target_tip_px_left", "target_source",
        "geometry_mode", "geometry_reason", "geometry_error_m",
        "metric_requested_delta_m", "hole_base_m", "target_base_m",
        "hole_depth_support", "target_depth_support", "right_tf_timestamp_ns", "left_tf_timestamp_ns", "geometry_agrees",
        "requested_delta_m", "accepted_delta_m", "cumulative_offset_m",
        "cumulative_distance_m", "step_index", "p4_verified", "right_fixed",
    )
    return {key: record[key] for key in keys if key in record}


def log_record(stream, record: dict) -> None:
    payload = compact_frame(record) if record.get("event") == "frame" else record
    stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
    stream.flush()


def latest_transform_to_base(tf, source_frame: str) -> tuple[dict, int | None]:
    """Read the dynamic ``T_base_source`` transform with its GDK timestamp."""
    result = tf.lookup_transform_latest("base_link", source_frame, return_timestamp=True)
    if isinstance(result, tuple) and len(result) == 2:
        transform, timestamp = result
    else:  # Defensive compatibility with older bindings.
        transform, timestamp = result, None
    return transform_to_dict(transform), None if timestamp is None else int(timestamp)


def transform_to_base_at(tf, source_frame: str, timestamp_ns: int) -> tuple[dict, int | None]:
    """Prefer the image-time TF; fall back to latest for older GDK bindings."""
    if int(timestamp_ns) > 0 and hasattr(tf, "lookup_transform"):
        try:
            transform = tf.lookup_transform("base_link", source_frame, int(timestamp_ns))
            return transform_to_dict(transform), int(timestamp_ns)
        except Exception:
            pass
    return latest_transform_to_base(tf, source_frame)


def main() -> int:
    parser = argparse.ArgumentParser(description="P4 后右观左动的双臂 XY 定位伺服")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--left-model", type=Path, default=DEFAULT_LEFT_TARGET_MODEL, help="左 D405 hinge/shaft 分割权重；shaft 作为 target")
    parser.add_argument("--poses", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--right-preset", default="right_post_grasp")
    parser.add_argument("--left-p4-preset", default="left_p4")
    parser.add_argument("--p4-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--right-drift-tolerance-rad", type=float, default=0.03)
    parser.add_argument("--gripper-closed-max-open-rad", type=float, default=0.50)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--left-confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--left-imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--pixel-jacobian", default=None, help="右图 target-hole 像素 / 左 base XY 米，J00,J01,J10,J11；未提供时只读")
    parser.add_argument(
        "--geometry-mode",
        choices=("off", "veto", "direct"),
        default="off",
        help="几何路径：off=旧右图 pixel-J，veto=旧右图 RGB-D 否决，direct=左 shaft/右 hole 双 RGB-D 三维 XY",
    )
    parser.add_argument(
        "--approve-rgbd-direct-motion",
        action="store_true",
        help="允许 direct 模式真实左臂 XY 运动；必须与 --allow-motion 同时使用",
    )
    parser.add_argument("--metric-tolerance-mm", type=float, default=2.0)
    parser.add_argument("--metric-pair-jump-mm", type=float, default=8.0)
    parser.add_argument("--metric-max-error-mm", type=float, default=60.0)
    parser.add_argument("--min-depth-pixels", type=int, default=8)
    parser.add_argument("--min-depth-support", type=float, default=0.20)
    parser.add_argument("--max-depth-mad-mm", type=float, default=15.0)
    parser.add_argument("--max-rgbd-skew-ms", type=float, default=100.0)
    parser.add_argument("--pixel-tolerance-px", type=float, default=8.0)
    parser.add_argument("--stable-frames", type=int, default=4)
    parser.add_argument("--aligned-frames", type=int, default=5)
    parser.add_argument("--max-pair-jump-px", type=float, default=25.0)
    parser.add_argument("--max-error-px", type=float, default=260.0)
    parser.add_argument("--max-step-mm", type=float, default=1.5)
    parser.add_argument("--max-total-displacement-mm", type=float, default=10.0)
    parser.add_argument("--max-total-travel-mm", type=float, default=20.0)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--left-speed-mm-s", type=float, default=10.0)
    parser.add_argument("--step-settle-s", type=float, default=0.20)
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示持续运行")
    parser.add_argument("--max-duration-s", type=float, default=60.0, help="单次定位最长运行时间")
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8776, help="0 表示关闭网页")
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--allow-motion", action="store_true", help="允许右臂一次预移动和左臂 XY 伺服")
    parser.add_argument("--skip-right-premove", action="store_true", help="不移动右臂；测试当前观察位")
    args = parser.parse_args()

    try:
        jacobian = parse_jacobian(args.pixel_jacobian)
        policy = AlignmentPolicy(
            pixel_tolerance_px=args.pixel_tolerance_px,
            stable_frames=args.stable_frames,
            aligned_frames=args.aligned_frames,
            max_pair_jump_px=args.max_pair_jump_px,
            max_error_px=args.max_error_px,
            max_step_m=args.max_step_mm / 1000.0,
            max_total_displacement_m=args.max_total_displacement_mm / 1000.0,
            max_total_travel_m=args.max_total_travel_mm / 1000.0,
            max_steps=args.max_steps,
            min_confidence=args.confidence,
        )
        policy.validate()
        if args.allow_motion and args.geometry_mode in {"off", "veto"} and jacobian is None:
            raise ValueError("允许运动时必须提供右相机实测 --pixel-jacobian；当前没有可复用的左 D405 Jacobian")
        if args.geometry_mode == "direct" and args.pixel_jacobian is not None:
            raise ValueError("direct 模式不接受 --pixel-jacobian；避免两套控制律混用")
        if args.geometry_mode == "direct" and args.allow_motion and not args.approve_rgbd_direct_motion:
            raise ValueError("direct 真实运动必须同时提供 --approve-rgbd-direct-motion")
        if args.approve_rgbd_direct_motion and not args.allow_motion:
            raise ValueError("--approve-rgbd-direct-motion 必须与 --allow-motion 同时使用")
        if args.approve_rgbd_direct_motion and args.geometry_mode != "direct":
            raise ValueError("--approve-rgbd-direct-motion 只能用于 --geometry-mode direct")
        if not (
            0.0 <= args.confidence <= 1.0 and 0.0 <= args.left_confidence <= 1.0
            and 0.0 <= args.iou <= 1.0 and args.imgsz > 0 and args.left_imgsz > 0
            and args.max_frames >= 0 and 1.0 <= args.max_duration_s <= 600.0
            and 0.001 <= args.p4_tolerance_rad <= 0.10
            and 0.001 <= args.right_drift_tolerance_rad <= 0.20
            and 0.05 <= args.gripper_closed_max_open_rad <= 0.785
            and 0.1 <= args.left_speed_mm_s <= 50.0 and 0.0 <= args.step_settle_s <= 5.0
            and 0 <= args.web_port <= 65535
            and 0.001 <= args.metric_tolerance_mm <= 100.0
            and args.metric_tolerance_mm < args.metric_pair_jump_mm <= 500.0
            and args.metric_tolerance_mm < args.metric_max_error_mm <= 1000.0
            and 1 <= args.min_depth_pixels <= 100000
            and 0.0 < args.min_depth_support <= 1.0
            and 0.1 <= args.max_depth_mad_mm <= 200.0
            and 1.0 <= args.max_rgbd_skew_ms <= 1000.0
        ):
            raise ValueError("检测、P4、速度、伺服或网页参数无效")
        model_path = args.model.expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"找不到 hole/target 权重：{model_path}")
        left_model_path = args.left_model.expanduser().resolve()
        if args.geometry_mode == "direct" and not left_model_path.is_file():
            raise FileNotFoundError(f"找不到左 hinge/shaft 权重：{left_model_path}")
        left_p4 = load_pose(args.poses, args.left_p4_preset, "left")
        right_target = load_pose(args.poses, args.right_preset, "right")
    except Exception as exc:
        print(f"参数/姿态无效：{exc}", file=sys.stderr)
        return 2

    log_path = (args.log or PROJECT_ROOT / "logs" / f"bilateral_alignment_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.jsonl").expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "mode": "motion" if args.allow_motion else "read_only_preview",
        "stage": "post_p4_bilateral_xy_alignment",
        "camera": "right_hand_color + left_hand_color" if args.geometry_mode == "direct" else "right_hand_color",
        "geometry_mode": args.geometry_mode,
        "moving_arm": "left",
        "model": str(model_path),
        "left_model": str(left_model_path) if args.geometry_mode == "direct" else None,
        "pixel_jacobian": jacobian,
        "right_premove": None if args.skip_right_premove else args.right_preset,
        "p4_preset": args.left_p4_preset,
        "log": str(log_path),
    }, ensure_ascii=False))

    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    initialized = False
    server = None
    shared = None
    stream = None
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        robot = agibot_gdk.Robot()
        time.sleep(1.0)
        controller = G2Controller(agibot_gdk, robot, allow_motion=args.allow_motion)
        current = controller.checked_arm_state()
        p4_error = max(abs(actual - expected) for actual, expected in zip(current[:7], left_p4))
        gripper_position = controller.read_gripper_position("left")
        p4_verified = p4_error <= args.p4_tolerance_rad
        gripper_verified = gripper_position is not None and abs(gripper_position) <= args.gripper_closed_max_open_rad
        if args.allow_motion and (not p4_verified or not gripper_verified):
            raise RuntimeError(
                "拒绝运动：必须先验证左臂处于 P4 且夹爪仍为闭合/持物状态；"
                f"p4_error={p4_error:.4f} rad, gripper={gripper_position} rad"
            )
        right_reference = tuple(current[RIGHT_ARM_OFFSET:RIGHT_ARM_OFFSET + 7])
        if not args.skip_right_premove and args.allow_motion:
            controller.move_preset("right", right_target, velocity_rad_s=0.08, life_time_s=25.0, detail="bilateral right observer premove")
            deadline = time.monotonic() + 25.0
            while time.monotonic() < deadline:
                measured = controller.checked_arm_state()[RIGHT_ARM_OFFSET:RIGHT_ARM_OFFSET + 7]
                if max(abs(actual - expected) for actual, expected in zip(measured, right_target)) <= args.right_drift_tolerance_rad:
                    right_reference = tuple(measured)
                    break
                time.sleep(0.10)
            else:
                raise RuntimeError("右臂观察位预移动超时；未启动左臂伺服")
        elif not args.skip_right_premove:
            print("只读预检：读取 right_post_grasp，但未发送右臂预移动。")

        model = load_detection_model(str(model_path), device=args.device)
        left_model = None
        if args.geometry_mode == "direct":
            left_model = load_yolo_segmentation_model(str(left_model_path), device=args.device)
        color_type = agibot_gdk.CameraType.kHandRightColor
        depth_type = agibot_gdk.CameraType.kHandRightDepth
        left_color_type = agibot_gdk.CameraType.kHandLeftColor
        left_depth_type = agibot_gdk.CameraType.kHandLeftDepth
        need_rgbd = args.geometry_mode != "off"
        need_bilateral = args.geometry_mode == "direct"
        camera_types = [color_type, depth_type, left_color_type, left_depth_type] if need_bilateral else [color_type, depth_type] if need_rgbd else [color_type]
        camera = agibot_gdk.Camera(camera_types)
        tf = agibot_gdk.TF() if need_rgbd else None
        depth_to_color = None
        rgbd_to_right_end = None
        left_depth_to_color = None
        left_rgbd_to_end = None
        geometry_startup_error = None
        if need_rgbd:
            try:
                depth_to_color = transform_to_dict(
                    tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kRightHandDepthToRightHandColor)
                )
                rgbd_to_right_end = transform_to_dict(
                    tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kRightHandRGBDToArmREndLink)
                )
                if need_bilateral:
                    left_depth_to_color = transform_to_dict(
                        tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandDepthToLeftHandColor)
                    )
                    left_rgbd_to_end = transform_to_dict(
                        tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kLeftHandRGBDToArmLEndLink)
                    )
            except Exception as exc:
                geometry_startup_error = f"official_bilateral_rgbd_calibration_unavailable: {exc}"
                depth_to_color = None
                rgbd_to_right_end = None
                left_depth_to_color = None
                left_rgbd_to_end = None
        if args.allow_motion and need_bilateral and (depth_to_color is None or rgbd_to_right_end is None or left_depth_to_color is None or left_rgbd_to_end is None):
            raise RuntimeError(
                "拒绝双相机 RGB-D 运动：GDK 未返回完整左右手官方外参；先用只读 direct 检查运行环境。"
                f" {geometry_startup_error or ''}".strip()
            )
        color_intrinsic = None
        depth_intrinsic = None
        left_color_intrinsic = None
        left_depth_intrinsic = None
        if args.web_port:
            shared = SharedView()
            server = ThreadingHTTPServer((args.web_host, args.web_port), handler_class(shared))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"实时页面：http://{args.web_host}:{args.web_port}")

        pixel_gate = None if jacobian is None else BilateralAlignmentGate(jacobian, policy)
        metric_policy = MetricAlignmentPolicy(
            position_tolerance_m=args.metric_tolerance_mm / 1000.0,
            stable_frames=args.stable_frames,
            aligned_frames=args.aligned_frames,
            max_pair_jump_m=args.metric_pair_jump_mm / 1000.0,
            max_error_m=args.metric_max_error_mm / 1000.0,
            max_step_m=args.max_step_mm / 1000.0,
            max_total_displacement_m=args.max_total_displacement_mm / 1000.0,
            max_total_travel_m=args.max_total_travel_mm / 1000.0,
            max_steps=args.max_steps,
            min_confidence=args.confidence,
        )
        metric_gate = BilateralMetricAlignmentGate(metric_policy) if need_rgbd else None
        initial_pose: CartesianPoseTarget | None = None
        if args.allow_motion:
            if controller.motion_status_summary()["control_mode"] != 1:
                controller.set_control_mode(1)
            initial_pose = controller.read_end_effector_pose(LEFT_END_LINK)
            stream = controller.start_left_end_effector_position_stream(initial_pose, rate_hz=50.0, life_time_s=0.10, max_translation_velocity_mps=args.left_speed_mm_s / 1000.0)
        right_fixed = True
        sequence = 0
        started_at = time.monotonic()
        last_camera_timestamp = 0
        last_left_camera_timestamp = 0
        step_settle_until = 0.0
        aligned = False
        last_state = "acquire"
        fatal_fault = False
        with log_path.open("a", encoding="utf-8") as log:
            log_record(log, {
                "event": "start", "utc": datetime.now(timezone.utc).isoformat(),
                "stage": "post_p4_bilateral_xy_alignment", "model": str(model_path),
                "camera": "right_hand_color + left_hand_color" if need_bilateral else "right_hand_color",
                "motion_enabled": args.allow_motion, "p4_verified": p4_verified,
                "p4_error_rad": p4_error, "left_gripper_position_rad": gripper_position,
                "right_premove": None if args.skip_right_premove else args.right_preset,
                "right_fixed": True, "pixel_jacobian": jacobian,
                "geometry_mode": args.geometry_mode,
                "official_right_calibration": {
                    "depth_to_color": depth_to_color is not None,
                    "rgbd_to_right_end": rgbd_to_right_end is not None,
                },
                "official_left_calibration": {
                    "depth_to_color": left_depth_to_color is not None,
                    "rgbd_to_end": left_rgbd_to_end is not None,
                } if need_bilateral else None,
                "geometry_startup_error": geometry_startup_error,
                "policy": policy.__dict__,
                "metric_policy": metric_policy.__dict__,
            })
            while (args.max_frames == 0 or sequence < args.max_frames) and time.monotonic() - started_at < args.max_duration_s:
                sequence += 1
                frame_record = {
                    "event": "frame", "frame": sequence,
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "motion_enabled": args.allow_motion, "p4_verified": p4_verified,
                    "right_fixed": right_fixed, "sequence": sequence,
                }
                if args.allow_motion:
                    measured_right = controller.checked_arm_state()[RIGHT_ARM_OFFSET:RIGHT_ARM_OFFSET + 7]
                    right_drift = max(abs(actual - expected) for actual, expected in zip(measured_right, right_reference))
                    if right_drift > args.right_drift_tolerance_rad:
                        frame_record.update({"state": "fault_hold", "reason": "right_observer_drift", "right_drift_rad": right_drift})
                        log_record(log, frame_record)
                        print(json.dumps(compact_frame(frame_record), ensure_ascii=False), flush=True)
                        fatal_fault = True
                        break
                try:
                    image = camera.get_latest_image(color_type, 1000.0)
                    camera_timestamp = int(getattr(image, "timestamp_ns", 0))
                    if camera_timestamp > 0 and camera_timestamp <= last_camera_timestamp:
                        frame_record.update({"state": "hold", "reason": "duplicate_or_old_camera_frame"})
                        log_record(log, frame_record)
                        continue
                    if camera_timestamp > 0:
                        last_camera_timestamp = camera_timestamp
                    color_rgb = decode_color_rgb(image, agibot_gdk)
                    frame_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
                except Exception as exc:
                    frame_record.update({"state": "fault_hold", "reason": "color_perception_failure", "error": str(exc)})
                    log_record(log, frame_record)
                    print(json.dumps(compact_frame(frame_record), ensure_ascii=False), flush=True)
                    fatal_fault = True
                    break

                aligned_depth = None
                geometry_reason = geometry_startup_error
                right_end_to_base = None
                tf_timestamp = None
                left_image = None
                left_color_rgb = None
                left_frame_bgr = None
                left_aligned_depth = None
                left_end_to_base = None
                left_tf_timestamp = None
                if need_rgbd:
                    try:
                        if depth_to_color is None or rgbd_to_right_end is None:
                            raise RuntimeError(geometry_startup_error or "official right RGB-D calibration is unavailable")
                        depth_image = camera.get_nearest_image(depth_type, camera_timestamp, 1000.0)
                        depth_timestamp = int(getattr(depth_image, "timestamp_ns", 0))
                        if depth_timestamp <= 0:
                            raise RuntimeError("invalid right depth timestamp")
                        if camera_timestamp > 0 and abs(depth_timestamp - camera_timestamp) > args.max_rgbd_skew_ms * 1_000_000.0:
                            raise RuntimeError(
                                f"right RGB-D timestamp skew exceeds limit: {abs(depth_timestamp - camera_timestamp) / 1_000_000.0:.1f} ms"
                            )
                        depth_u16 = decode_depth_u16(depth_image, agibot_gdk)
                        if color_intrinsic is None:
                            color_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(color_type),
                                width=int(image.width),
                                height=int(image.height),
                            )
                        if depth_intrinsic is None:
                            depth_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(depth_type),
                                width=int(depth_image.width),
                                height=int(depth_image.height),
                            )
                        aligned_depth = align_depth_to_color(
                            depth_u16,
                            depth_intrinsic=depth_intrinsic,
                            color_intrinsic=color_intrinsic,
                            depth_to_color=depth_to_color,
                        )
                        right_end_to_base, tf_timestamp = transform_to_base_at(tf, "arm_r_end_link", camera_timestamp)
                    except Exception as exc:
                        geometry_reason = str(exc)

                if need_bilateral:
                    try:
                        left_image = camera.get_latest_image(left_color_type, 1000.0)
                        left_timestamp = int(getattr(left_image, "timestamp_ns", 0))
                        if left_timestamp <= 0:
                            raise RuntimeError("invalid left color timestamp")
                        if left_timestamp <= last_left_camera_timestamp:
                            raise RuntimeError("duplicate_or_old_left_camera_frame")
                        last_left_camera_timestamp = left_timestamp
                        left_color_rgb = decode_color_rgb(left_image, agibot_gdk)
                        left_frame_bgr = cv2.cvtColor(left_color_rgb, cv2.COLOR_RGB2BGR)
                        left_depth_image = camera.get_nearest_image(left_depth_type, left_timestamp, 1000.0)
                        left_depth_timestamp = int(getattr(left_depth_image, "timestamp_ns", 0))
                        if left_depth_timestamp <= 0:
                            raise RuntimeError("invalid left depth timestamp")
                        if abs(left_depth_timestamp - left_timestamp) > args.max_rgbd_skew_ms * 1_000_000.0:
                            raise RuntimeError(
                                f"left RGB-D timestamp skew exceeds limit: {abs(left_depth_timestamp - left_timestamp) / 1_000_000.0:.1f} ms"
                            )
                        left_depth_u16 = decode_depth_u16(left_depth_image, agibot_gdk)
                        if color_intrinsic is None:
                            color_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(color_type), width=int(image.width), height=int(image.height)
                            )
                        if depth_intrinsic is None:
                            depth_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(depth_type), width=int(depth_image.width), height=int(depth_image.height)
                            )
                        if left_color_intrinsic is None:
                            left_color_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(left_color_type), width=int(left_image.width), height=int(left_image.height)
                            )
                        if left_depth_intrinsic is None:
                            left_depth_intrinsic = intrinsic_to_dict(
                                camera.get_camera_intrinsic(left_depth_type), width=int(left_depth_image.width), height=int(left_depth_image.height)
                            )
                        if left_depth_to_color is None or left_rgbd_to_end is None:
                            raise RuntimeError("official_left_rgbd_calibration_unavailable")
                        left_aligned_depth = align_depth_to_color(
                            left_depth_u16,
                            depth_intrinsic=left_depth_intrinsic,
                            color_intrinsic=left_color_intrinsic,
                            depth_to_color=left_depth_to_color,
                        )
                        left_end_to_base, left_tf_timestamp = transform_to_base_at(tf, LEFT_END_LINK, left_timestamp)
                    except Exception as exc:
                        geometry_reason = f"left_target_perception_unavailable: {exc}"

                try:
                    detections = infer_detection(
                        model,
                        frame_bgr,
                        confidence=args.confidence,
                        iou=args.iou,
                        imgsz=args.imgsz,
                        device=args.device,
                    )
                except Exception as exc:
                    frame_record.update({"state": "fault_hold", "reason": "detector_failure", "error": str(exc)})
                    log_record(log, frame_record)
                    print(json.dumps(compact_frame(frame_record), ensure_ascii=False), flush=True)
                    fatal_fault = True
                    break
                holes = [item for item in detections if item.class_name == "hole"]
                left_targets: list[BoxDetection] = []
                if need_bilateral and left_model is not None and left_color_rgb is not None:
                    try:
                        left_targets = infer_left_target_boxes(
                            left_model,
                            left_frame_bgr,
                            confidence=args.left_confidence,
                            imgsz=args.left_imgsz,
                            device=args.device,
                        )
                    except Exception as exc:
                        geometry_reason = f"left_target_detector_failure: {exc}"
                # Legacy off/veto modes keep their original single-right-camera
                # semantics.  Correct bilateral direct mode uses left shaft as
                # target and deliberately ignores any right-camera target box.
                targets = left_targets if need_bilateral else [item for item in detections if item.class_name == "target"]
                pair_detections = holes + targets
                frame_record["detections"] = [
                    {"camera": "right", "class": item.class_name, "confidence": round(item.confidence, 4), "center_px": [round(value, 1) for value in item.center_px]}
                    for item in holes
                ]
                frame_record["left_detections"] = [
                    {"camera": "left", "class": item.class_name, "confidence": round(item.confidence, 4), "center_px": [round(value, 1) for value in item.center_px]}
                    for item in left_targets
                ] if need_bilateral else []
                if need_bilateral and len(targets) == 1:
                    frame_record["target_tip_px_left"] = [round(float(value), 1) for value in targets[0].tip_px]
                    frame_record["target_source"] = "left_shaft_tip_far_from_unique_hinge"
                frame_record["pair_status"] = pair_status(
                    pair_detections,
                    {"hole": holes[0], "target": targets[0]} if len(holes) == 1 and len(targets) == 1 else {},
                )
                frame_record["geometry_mode"] = args.geometry_mode
                if geometry_reason:
                    frame_record["geometry_reason"] = geometry_reason
                if tf_timestamp is not None:
                    frame_record["right_tf_timestamp_ns"] = tf_timestamp
                if left_tf_timestamp is not None:
                    frame_record["left_tf_timestamp_ns"] = left_tf_timestamp
                if need_rgbd and (len(holes) != 1 or len(targets) != 1) and not geometry_reason:
                    geometry_reason = "require_exactly_one_hole_and_target"
                    frame_record["geometry_reason"] = geometry_reason
                pair_geometry: PairGeometry | None = None
                hole_base_m = None
                target_base_m = None
                hole_depth_support = None
                target_depth_support = None
                if need_rgbd and len(holes) == 1 and len(targets) == 1:
                    try:
                        if aligned_depth is None or color_intrinsic is None or right_end_to_base is None or rgbd_to_right_end is None:
                            raise RuntimeError("right hole RGB-D geometry unavailable")
                        right_hole_depth = estimate_box_depth(
                            aligned_depth,
                            holes[0].xyxy,
                            intrinsic=color_intrinsic,
                            role="hole",
                            min_valid_pixels=args.min_depth_pixels,
                            min_support_fraction=args.min_depth_support,
                            max_mad_m=args.max_depth_mad_mm / 1000.0,
                        )
                        hole_end = apply_transform(rgbd_to_right_end, right_hole_depth.point_color_m)
                        hole_base_m = apply_transform(right_end_to_base, hole_end)
                        hole_depth_support = right_hole_depth.support_fraction
                        if need_bilateral:
                            if left_aligned_depth is None or left_color_intrinsic is None or left_end_to_base is None or left_rgbd_to_end is None:
                                raise RuntimeError("left target RGB-D geometry unavailable")
                            left_target_depth = estimate_point_depth(
                                left_aligned_depth,
                                targets[0].tip_px,
                                intrinsic=left_color_intrinsic,
                                radius_px=4,
                                min_valid_pixels=args.min_depth_pixels,
                                min_support_fraction=args.min_depth_support,
                                max_mad_m=args.max_depth_mad_mm / 1000.0,
                            )
                            target_end = apply_transform(left_rgbd_to_end, left_target_depth.point_color_m)
                            target_base_m = apply_transform(left_end_to_base, target_end)
                            target_depth_support = left_target_depth.support_fraction
                        else:
                            old_pair_geometry = estimate_pair_geometry(
                                aligned_depth,
                                hole_xyxy=holes[0].xyxy,
                                target_xyxy=targets[0].xyxy,
                                color_intrinsic=color_intrinsic,
                                rgbd_to_right_end=rgbd_to_right_end,
                                right_end_to_base=right_end_to_base,
                                min_valid_pixels=args.min_depth_pixels,
                                min_support_fraction=args.min_depth_support,
                                max_mad_m=args.max_depth_mad_mm / 1000.0,
                            )
                            pair_geometry = old_pair_geometry
                            target_base_m = old_pair_geometry.target_base_m
                            target_depth_support = old_pair_geometry.target.support_fraction
                        frame_record.update({
                            "hole_base_m": list(hole_base_m),
                            "target_base_m": list(target_base_m),
                            "hole_depth_support": hole_depth_support,
                            "target_depth_support": target_depth_support,
                        })
                    except Exception as exc:
                        frame_record["geometry_reason"] = str(exc)
                        hole_base_m = None
                        target_base_m = None

                decision = None
                metric_decision = None
                pixel_decision = None
                if stream is not None and time.monotonic() < step_settle_until:
                    frame_record.update({"state": "step_settling", "reason": "waiting_for_left_target_arrival_before_consuming_new_pair"})
                elif args.geometry_mode == "off":
                    if pixel_gate is None:
                        frame_record.update({"state": "calibration_required", "reason": "provide_right_camera_pixel_jacobian_for_motion"})
                    else:
                        can_move = bool(stream is not None and stream.is_target_reached() and not aligned)
                        pixel_decision = pixel_gate.update(
                            sequence=sequence,
                            hole_center_px=None if len(holes) != 1 else holes[0].center_px,
                            target_center_px=None if len(targets) != 1 else targets[0].center_px,
                            hole_confidence=holes[0].confidence if len(holes) == 1 else 0.0,
                            target_confidence=targets[0].confidence if len(targets) == 1 else 0.0,
                            hole_count=len(holes), target_count=len(targets), can_move=can_move,
                        )
                        decision = pixel_decision
                        frame_record.update({
                            "state": decision.state, "reason": decision.reason,
                            "pixel_error_px": decision.pixel_error_px,
                            "requested_delta_m": list(decision.requested_delta_m),
                            "accepted_delta_m": list(decision.accepted_delta_m),
                            "cumulative_offset_m": list(decision.cumulative_offset_m),
                            "cumulative_distance_m": decision.cumulative_distance_m,
                            "step_index": decision.step_index,
                        })
                elif args.geometry_mode == "direct":
                    if metric_gate is None:
                        frame_record.update({"state": "geometry_hold", "reason": "metric_gate_not_initialized"})
                    elif hole_base_m is None or target_base_m is None:
                        metric_gate.reset_observation()
                        frame_record.update({"state": "geometry_hold", "reason": frame_record.get("geometry_reason", "require_one_pair_with_valid_rgbd_depth")})
                    else:
                        can_move = bool(stream is not None and stream.is_target_reached() and not aligned)
                        metric_decision = metric_gate.update(
                            sequence=sequence,
                            hole_position_m=hole_base_m,
                            target_position_m=target_base_m,
                            hole_confidence=holes[0].confidence,
                            target_confidence=targets[0].confidence,
                            can_move=can_move,
                        )
                        decision = metric_decision
                        frame_record.update({
                            "state": metric_decision.state, "reason": metric_decision.reason,
                            "geometry_error_m": metric_decision.error_m,
                            "requested_delta_m": list(metric_decision.requested_delta_m),
                            "accepted_delta_m": list(metric_decision.accepted_delta_m),
                            "cumulative_offset_m": list(metric_decision.cumulative_offset_m),
                            "cumulative_distance_m": metric_decision.cumulative_distance_m,
                            "step_index": metric_decision.step_index,
                            "hole_base_m": list(hole_base_m),
                            "target_base_m": list(target_base_m),
                        })
                else:  # RGB-D veto around the existing pixel-J controller.
                    if pixel_gate is None:
                        frame_record.update({"state": "calibration_required", "reason": "veto_motion_requires_pixel_jacobian"})
                    elif metric_gate is None or pair_geometry is None:
                        pixel_gate.reset_observation()
                        if metric_gate is not None:
                            metric_gate.reset_observation()
                        frame_record.update({"state": "geometry_hold", "reason": frame_record.get("geometry_reason", "require_one_pair_with_valid_rgbd_depth")})
                    else:
                        can_move = bool(stream is not None and stream.is_target_reached() and not aligned)
                        metric_decision = metric_gate.update(
                            sequence=sequence,
                            hole_position_m=pair_geometry.hole_base_m,
                            target_position_m=pair_geometry.target_base_m,
                            hole_confidence=holes[0].confidence,
                            target_confidence=targets[0].confidence,
                            can_move=False,
                        )
                        pixel_decision = pixel_gate.update(
                            sequence=sequence,
                            hole_center_px=holes[0].center_px,
                            target_center_px=targets[0].center_px,
                            hole_confidence=holes[0].confidence,
                            target_confidence=targets[0].confidence,
                            hole_count=1,
                            target_count=1,
                            can_move=can_move,
                        )
                        decision = pixel_decision
                        frame_record.update({
                            "state": pixel_decision.state, "reason": pixel_decision.reason,
                            "pixel_error_px": pixel_decision.pixel_error_px,
                            "metric_error_m": metric_decision.error_m,
                            "metric_requested_delta_m": list(metric_decision.requested_delta_m),
                            "hole_base_m": list(pair_geometry.hole_base_m),
                            "target_base_m": list(pair_geometry.target_base_m),
                        })
                        if pixel_decision.state == "move":
                            agrees = metric_decision.state in {"ready", "waiting_alignment", "aligned"} and metric_and_pixel_corrections_agree(pixel_decision.requested_delta_m, metric_decision.requested_delta_m)
                            frame_record["geometry_agrees"] = agrees
                            if not agrees:
                                pixel_gate.reset_observation()
                                metric_gate.reset_observation()
                                frame_record.update({"state": "geometry_disagreement_hold", "reason": "pixel_and_official_rgbd_corrections_disagree"})
                                decision = None
                        elif pixel_decision.state == "aligned" and metric_decision.state != "aligned":
                            frame_record.update({"state": "geometry_wait", "reason": "pixel_aligned_but_metric_alignment_not_confirmed"})
                            decision = None

                if decision is not None and decision.state == "move" and stream is not None and initial_pose is not None:
                    target_pose = left_target_from_offset(initial_pose, decision.cumulative_offset_m)
                    stream.update_target(target_pose)
                    if args.geometry_mode == "off":
                        pixel_gate.reset_observation()
                    elif args.geometry_mode == "direct":
                        metric_gate.reset_observation()
                    else:
                        pixel_gate.reset_observation()
                        metric_gate.reset_observation()
                    step_settle_until = time.monotonic() + args.step_settle_s
                    frame_record["command_source"] = "left_mode1_end_effector_position_stream"
                elif decision is not None and decision.state == "aligned":
                    aligned = True

                right_overlay = draw_alignment_overlay(
                    frame_bgr,
                    holes if need_bilateral else detections,
                    {},
                    frame_record.get("state", last_state),
                    frame_record.get("pixel_error_px"),
                    frame_record.get("geometry_error_m"),
                )
                if need_bilateral and left_frame_bgr is not None:
                    left_overlay = draw_alignment_overlay(
                        left_frame_bgr,
                        left_targets,
                        {},
                        "left target",
                        None,
                        None,
                    )
                    if left_overlay.shape[0] != right_overlay.shape[0]:
                        scale = right_overlay.shape[0] / float(left_overlay.shape[0])
                        left_overlay = cv2.resize(left_overlay, (int(left_overlay.shape[1] * scale), right_overlay.shape[0]))
                    # Display order follows the operator's view: left hand first,
                    # right hand second.  Detection/TF/control frames are unchanged.
                    overlay = np.hstack((left_overlay, right_overlay))
                else:
                    overlay = right_overlay
                ok, encoded = cv2.imencode(".jpg", overlay, [cv2.IMWRITE_JPEG_QUALITY, 82])
                if shared is not None and ok:
                    shared.update(encoded.tobytes(), {
                        "state": frame_record.get("state", last_state), "sequence": sequence,
                        "camera": "right_hand_color + left_hand_color" if need_bilateral else "right_hand_color",
                        "detections": frame_record.get("detections", []),
                        "left_detections": frame_record.get("left_detections", []),
                        "pair_status": frame_record.get("pair_status"), "pixel_error_px": frame_record.get("pixel_error_px"),
                        "geometry_mode": args.geometry_mode,
                        "geometry_reason": frame_record.get("geometry_reason"),
                        "geometry_error_m": frame_record.get("geometry_error_m"),
                        "hole_base_m": frame_record.get("hole_base_m"), "target_base_m": frame_record.get("target_base_m"),
                        "geometry_agrees": frame_record.get("geometry_agrees"),
                        "accepted_delta_m": frame_record.get("accepted_delta_m"),
                        "cumulative_offset_m": frame_record.get("cumulative_offset_m"),
                        "right_fixed": True, "motion": "left_xy_only" if args.allow_motion else "read_only_preview",
                    })
                log_record(log, frame_record)
                if frame_record.get("state") != last_state:
                    print(json.dumps(compact_frame(frame_record), ensure_ascii=False), flush=True)
                    last_state = frame_record.get("state", last_state)
                if aligned:
                    success_basis = "官方 RGB-D 三维容差" if args.geometry_mode == "direct" else "像素与官方 RGB-D 双重容差" if args.geometry_mode == "veto" else "像素容差"
                    print(f"双臂定位完成：右臂保持观察位，左臂已满足{success_basis}；未下探、未夹爪、未插入。")
                    break
        return 1 if fatal_fault else 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"双臂定位伺服失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if stream is not None:
            stream.stop()
        if server is not None:
            server.shutdown()
            server.server_close()
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
