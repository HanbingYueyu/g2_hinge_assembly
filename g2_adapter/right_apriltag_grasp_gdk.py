#!/usr/bin/env python3
"""G2 right-hand AprilTag localization and a gated grasp sequence.

This replaces the ROS topic graph in the supplied OpenArm package with one
process: GDK right color camera -> AprilTag 41h12 -> PnP -> GDK TF -> G2
end-link targets.  It contains no ROS, MoveIt, Pinocchio, or custom IK.

By default only the calibrated local final move may be executed.  The explicit
``--autonomous-approach`` mode adds a bounded, waypoint-by-waypoint movement
from a coarse QR-visible start to the recorded pregrasp.  Both modes remain
explicitly fail-closed until the right-arm capability probe and tool mappings
are recorded.  When requested, the full motion sequence is
``right_pr1`` planned pre-move -> fresh visual grasp -> +Z lift ->
``right_pr2`` planned transport.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

try:
    from .apriltag_workpiece import (
        AprilTagGeometryError, DEFAULT_TAG_GEOMETRY, GraspTargets, RollingWorkpiecePoseEstimator, TagGeometry, as_transform,
        g2_grasp_targets, pose_distance, quaternion_from_rotation, right_end_from_grasp_tcp_candidate,
        transform_from_dict, inverse_transform_dict, workpiece_from_pregrasp_end_candidate,
        constrained_pregrasp_end_candidate,
    )
    from .camera import decode_color_rgb, intrinsic_to_dict, transform_to_dict
    from .control import CartesianPoseTarget, G2Controller, quaternion_slerp
    from .right_post_grasp_sequence import (
        PRESET_VELOCITY_RAD_S, PR1_POSE_NAME, PR2_POSE_NAME, execute_post_grasp_transport,
        load_pr1_pose, load_pr2_pose, move_right_pr1,
    )
    from .spacemouse_takeover import MotionPauseGate, SpaceMouseTakeoverWorker
    from .spacemouse_teleop import TeleopConfig
except ImportError:
    from apriltag_workpiece import (
        AprilTagGeometryError, DEFAULT_TAG_GEOMETRY, GraspTargets, RollingWorkpiecePoseEstimator, TagGeometry, as_transform,
        g2_grasp_targets, pose_distance, quaternion_from_rotation, right_end_from_grasp_tcp_candidate,
        transform_from_dict, inverse_transform_dict, workpiece_from_pregrasp_end_candidate,
        constrained_pregrasp_end_candidate,
    )
    from camera import decode_color_rgb, intrinsic_to_dict, transform_to_dict
    from control import CartesianPoseTarget, G2Controller, quaternion_slerp
    from right_post_grasp_sequence import (
        PRESET_VELOCITY_RAD_S, PR1_POSE_NAME, PR2_POSE_NAME, execute_post_grasp_transport,
        load_pr1_pose, load_pr2_pose, move_right_pr1,
    )
    from spacemouse_takeover import MotionPauseGate, SpaceMouseTakeoverWorker
    from spacemouse_teleop import TeleopConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "g2_adapter" / "right_apriltag_grasp.example.json"


class VisualLockTimeout(RuntimeError):
    """No fresh, stable 0/1/2 lock arrived before one visual-attempt deadline."""


def utc_stamp() -> str:
    """Collision-resistant UTC timestamp for a log filename."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def timestamped_log_path(path: Path, stamp: str) -> Path:
    """Append a timestamp only to a user-supplied log filename."""
    expanded = path.expanduser()
    suffix = expanded.suffix or ".json"
    return expanded.with_name(f"{expanded.stem}_{stamp}{suffix}")


class SharedView:
    """Thread-safe browser preview state; it never participates in motion."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.overlay_jpeg: bytes | None = None
        self.status: dict = {"state": "starting", "sequence": 0}
        self._sequence = 0

    def update(self, overlay_jpeg: bytes | None, status: dict) -> None:
        with self.lock:
            self._sequence += 1
            self.overlay_jpeg = overlay_jpeg
            next_status = dict(status)
            next_status["sequence"] = self._sequence
            self.status = next_status

    def update_status(self, status: dict) -> None:
        """Publish a status heartbeat while retaining the last camera image."""
        with self.lock:
            self._sequence += 1
            next_status = dict(self.status)
            next_status.update(status)
            next_status["sequence"] = self._sequence
            self.status = next_status

    def snapshot(self) -> tuple[bytes | None, dict]:
        with self.lock:
            return self.overlay_jpeg, dict(self.status)


def handler_class(shared: SharedView):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args) -> None:
            return

        def send_payload(self, payload: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:  # noqa: N802
            image, status = shared.snapshot()
            path = self.path.split("?", 1)[0]
            if path == "/status.json":
                self.send_payload(json.dumps(status, ensure_ascii=False).encode(), "application/json; charset=utf-8")
                return
            if path == "/overlay.jpg":
                if image is None:
                    self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "waiting for camera frame")
                else:
                    self.send_payload(image, "image/jpeg")
                return
            if path not in {"/", "/index.html"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            page = """<!doctype html><meta charset='utf-8'><title>G2 Right AprilTag</title>
<style>body{margin:0;background:#161616;color:#eee;font:15px sans-serif}main{max-width:1280px;margin:auto;padding:14px}img{width:100%;background:#222}pre{white-space:pre-wrap;background:#242424;padding:10px;border-radius:6px}</style>
<main><h2>G2 右手 AprilTag 41h12（只读预览）</h2><img id='overlay' alt='等待相机帧'><pre id='status'>starting</pre></main>
<script>let s=-1;async function tick(){try{let r=await fetch('/status.json',{cache:'no-store'});let j=await r.json();document.querySelector('#status').textContent=JSON.stringify(j,null,2);if(j.sequence!==s){s=j.sequence;document.querySelector('#overlay').src='/overlay.jpg?seq='+s}}catch(e){document.querySelector('#status').textContent=String(e)}}setInterval(tick,250);tick();</script>"""
            self.send_payload(page.encode(), "text/html; charset=utf-8")

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="G2 右手 AprilTag 41h12 工件定位与局部抓取（仅 GDK）")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--face-distance-mm", type=float, default=None, help="仅持续正对伺服，距标定抓取点 10–300 mm，不操作夹爪")
    parser.add_argument(
        "--tag-geometry-profile", type=Path,
        help="覆盖配置中的独立 AprilTag 几何 profile；默认使用电批 Tag 几何",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="0 表示持续观察，正数表示最多帧数")
    parser.add_argument(
        "--terminal-output", choices=("text", "json"), default="text",
        help="终端输出格式；默认单行原地刷新，json 为逐帧 JSON",
    )
    parser.add_argument("--log", type=Path)
    parser.add_argument("--save-target", type=Path, help="把最近一次有效目标写为 JSON；不发送运动")
    parser.add_argument(
        "--calibrate-grasp-tcp", action="store_true",
        help="只读：在人工摆好的最终抓取姿态，计算 right_end_from_grasp_tcp 候选值",
    )
    parser.add_argument(
        "--calibrate-pregrasp", action="store_true",
        help="只读：记录人工确认的安全 pregrasp 为 workpiece_from_pregrasp_end 候选值",
    )
    parser.add_argument("--web-host", default="127.0.0.1")
    parser.add_argument("--web-port", type=int, default=8768, help="实时叠加页面端口；0 表示关闭")
    parser.add_argument("--allow-motion", action="store_true", help="仅与 --execute 同时出现时允许 GDK 动作")
    parser.add_argument("--execute", action="store_true", help="执行已标定的 pregrasp→grasp→闭爪")
    parser.add_argument(
        "--autonomous-approach", action="store_true",
        help="在二维码锁定后，从粗略当前位置分段自主移动到 pregrasp；需同时提供 --execute",
    )
    parser.add_argument(
        "--premove-pr1", action="store_true",
        help="执行抓取前先用 GDK planned joint request 移动到已记录的 right_pr1；需 --execute",
    )
    parser.add_argument(
        "--post-grasp-pr2", action="store_true",
        help="抓取反馈成功后保存当前位姿、沿 base_link +Z 上抬 100 mm，再移动到已记录的 right_pr2；需 --execute",
    )
    parser.add_argument(
        "--poses", type=Path, default=PROJECT_ROOT / "g2_adapter" / "recorded_poses.json",
        help="包含 right_pr1 和 right_pr2 的记录姿态文件",
    )
    parser.add_argument("--spacemouse-device", help="启用 SpaceMouse 临时接管；指定 /dev/hidrawN。")
    parser.add_argument("--spacemouse-speed-mm-s", type=float, default=10.0, help="SpaceMouse 平移速度上限，单位 mm/s。")
    parser.add_argument("--spacemouse-rotation-deg-s", type=float, default=10.0, help="SpaceMouse 旋转速度上限，单位 deg/s。")
    parser.add_argument("--spacemouse-deadzone", type=float, default=0.12, help="SpaceMouse 轴死区。")
    parser.add_argument("--auto-speed-mm-s", type=float, default=None, help="视觉自动末端平移速度，单位 mm/s。")
    parser.add_argument("--auto-rotation-deg-s", type=float, default=None, help="视觉自动末端旋转速度，单位 deg/s。")
    parser.add_argument(
        "--auto-preset-velocity-rad-s",
        type=float,
        default=None,
        help="视觉自动关节预设速度，单位 rad/s。",
    )
    return parser.parse_args()


def require_mapping(document: object, name: str) -> dict:
    if not isinstance(document, dict):
        raise RuntimeError(f"{name} 必须是 JSON 对象")
    return document


def apply_auto_motion_overrides(
    config: dict,
    *,
    speed_mm_s: float | None = None,
    rotation_deg_s: float | None = None,
) -> dict:
    """Return a config copy with optional Cartesian auto-speed overrides.

    Explicit speeds replace configured speeds; timeout budgets are independent.
    """
    updated = deepcopy(config)
    if speed_mm_s is not None:
        speed_mps = float(speed_mm_s) / 1000.0
        autonomous = updated.get("autonomous_approach")
        if isinstance(autonomous, dict) and "speed_mps" in autonomous:
            autonomous["speed_mps"] = speed_mps
        local_motion = updated.get("local_motion")
        if isinstance(local_motion, dict) and "final_speed_mps" in local_motion:
            local_motion["final_speed_mps"] = speed_mps
    if rotation_deg_s is not None:
        autonomous = updated.get("autonomous_approach")
        if isinstance(autonomous, dict) and "angular_speed_deg_s" in autonomous:
            autonomous["angular_speed_deg_s"] = float(rotation_deg_s)
    return updated


def format_terminal_status(
    *, frame: int, visible_tag_ids: Sequence[int], sample_counts: Mapping[object, int],
    pnp_diagnostics: Mapping[str, object], sample_count: int = 20,
) -> str:
    """Format one compact, carriage-return-friendly status line for the terminal."""
    visible = ",".join(str(tag_id) for tag_id in sorted(int(item) for item in visible_tag_ids)) or "-"
    samples = " ".join(
        f"{tag_id}:{int(sample_counts.get(tag_id, sample_counts.get(str(tag_id), 0)))}/{sample_count}"
        for tag_id in (0, 1, 2)
    )
    raw_state = str(pnp_diagnostics.get("state", "unknown"))
    state = {
        "collecting_samples": "collecting",
        "pnp_accepted": "accepted",
        "pnp_reprojection_rejected": "rejected",
        "pnp_ransac_failed": "ransac_failed",
        "stale_tag_samples": "stale",
        "samples_reset": "reset",
    }.get(raw_state, raw_state.replace("\n", " ").replace("\r", " "))
    line = f"帧 {int(frame)} | Tag {visible} | sample {samples} | PnP {state}"
    if {"mean_reprojection_error_px", "max_reprojection_error_px", "inlier_count"} <= pnp_diagnostics.keys():
        line += (
            f" mean={float(pnp_diagnostics['mean_reprojection_error_px']):.2f}px"
            f" max={float(pnp_diagnostics['max_reprojection_error_px']):.2f}px"
            f" inliers={int(pnp_diagnostics['inlier_count'])}"
        )
    return line


def calibration_terminal_message(mode: str, event: str, rejection: str | None) -> str:
    """Return an unambiguous terminal message for a calibration result."""
    if mode == "grasp_tcp_calibration":
        if event == "right_grasp_tcp_calibration_candidate":
            return "\n右手 grasp TCP 标定候选已生成。"
        label = "grasp TCP"
    elif mode == "pregrasp_calibration":
        if event == "right_pregrasp_calibration_candidate":
            return "\n右手 pregrasp 标定候选已生成。"
        label = "pregrasp"
    else:
        return ""
    reason = f"：{rejection}" if rejection else f"（状态：{event}）"
    return f"\n右手 {label} 标定未通过{reason}"


def load_tag_geometry(config: dict, profile_override: Path | str | None = None) -> TagGeometry:
    """Load an optional physical-workpiece profile without changing legacy defaults."""
    inline = config.get("tag_geometry")
    configured = config.get("tag_geometry_profile")
    selected = profile_override if profile_override is not None else configured
    if inline is not None and selected is not None:
        raise RuntimeError("tag_geometry 与 tag_geometry_profile 只能配置一个")
    if inline is not None:
        return TagGeometry.from_mapping(require_mapping(inline, "tag_geometry"))
    if selected is None:
        return DEFAULT_TAG_GEOMETRY
    profile_path = Path(selected).expanduser()
    if not profile_path.is_absolute():
        candidates = (PROJECT_ROOT / profile_path, PROJECT_ROOT / "g2_adapter" / profile_path)
        profile_path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    try:
        document = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"无法读取 tag_geometry_profile: {profile_path}") from exc
    return TagGeometry.from_mapping(require_mapping(document, str(profile_path)), str(profile_path))


def get_number(document: dict, name: str, *, minimum: float = 0.0) -> float:
    value = float(document.get(name, math.nan))
    if not math.isfinite(value) or value < minimum:
        raise RuntimeError(f"配置 {name} 必须是不小于 {minimum} 的有限数")
    return value


def camera_to_base_at(tf, timestamp_ns: int, max_age_ms: float) -> tuple[dict, dict]:
    """Return ``T_base_arm_r_end`` from GDK's reverse-direction TF response.

    GDK 4.1.5 returns ``T_source_target`` from
    ``lookup_transform(target, source)``.  This is the inverse of the usual
    ROS-style spelling.  The Robot motion-status frame pose and the A/B
    fixed-workpiece regression both confirm this convention, so invert here
    once at the GDK boundary rather than leaking the ambiguity into PnP.
    """
    latest_raw, latest_timestamp = tf.lookup_transform_latest("base_link", "arm_r_end_link", return_timestamp=True)
    latest_raw_dict = transform_to_dict(latest_raw)
    latest = inverse_transform_dict(latest_raw_dict)
    latest_timestamp = None if latest_timestamp is None else int(latest_timestamp)
    if latest_timestamp is not None and timestamp_ns <= latest_timestamp:
        try:
            raw = transform_to_dict(tf.lookup_transform("base_link", "arm_r_end_link", timestamp_ns))
            return inverse_transform_dict(raw), {
                "mode": "exact_image_time", "transform_timestamp_ns": timestamp_ns, "age_ms": 0.0,
                "lookup_arguments": ["base_link", "arm_r_end_link"],
                "gdk_returned_transform": "T_arm_r_end_link_base_link",
                "converted_transform": "T_base_link_arm_r_end_link",
                "gdk_raw_transform": raw,
            }
        except Exception as exc:
            reason = str(exc)
    else:
        reason = "image_timestamp_newer_than_latest_tf"
    if latest_timestamp is None:
        raise RuntimeError("GDK has no arm_r_end_link TF timestamp")
    age_ms = (int(timestamp_ns) - latest_timestamp) / 1_000_000.0
    if abs(age_ms) > max_age_ms:
        raise RuntimeError(f"right camera TF fallback age {age_ms:.1f} ms exceeds {max_age_ms:.1f} ms")
    return latest, {
        "mode": "latest_static_fallback", "transform_timestamp_ns": latest_timestamp,
        "age_ms": age_ms, "exact_lookup_error": reason,
        "lookup_arguments": ["base_link", "arm_r_end_link"],
        "gdk_returned_transform": "T_arm_r_end_link_base_link",
        "converted_transform": "T_base_link_arm_r_end_link",
        "gdk_raw_transform": latest_raw_dict,
    }


def detector_for_41h12():
    try:
        from pupil_apriltags import Detector
    except ImportError as exc:
        raise RuntimeError(
            "缺少 pupil_apriltags；tagStandard41h12 不是当前 OpenCV ArUco 字典。"
            "请在 .venv-sam3 安装项目 tags 依赖后重新运行。"
        ) from exc
    return Detector(families="tagStandard41h12", nthreads=2, quad_decimate=1.0, refine_edges=1)


def detect_corners(detector, color_rgb: np.ndarray) -> dict[int, np.ndarray]:
    # Keep detector-native order. The PnP layer evaluates and records each
    # globally consistent cyclic/mirrored corner convention before accepting.
    gray = np.ascontiguousarray(np.dot(color_rgb[..., :3], (0.299, 0.587, 0.114)).astype(np.uint8))
    result: dict[int, np.ndarray] = {}
    for detection in detector.detect(gray, estimate_tag_pose=False):
        tag_id = int(detection.tag_id)
        if tag_id in (0, 1, 2):
            corners = np.asarray(detection.corners, dtype=np.float64)
            if corners.shape == (4, 2) and np.all(np.isfinite(corners)):
                result[tag_id] = corners
    return result


def draw_overlay(color_rgb: np.ndarray, visible: dict[int, np.ndarray], record: dict) -> bytes:
    """Draw detection status in BGR then encode a browser-friendly JPEG."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for the AprilTag preview") from exc
    image = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
    for tag_id, corners in visible.items():
        polygon = np.rint(corners).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [polygon], True, (0, 255, 0), 2, cv2.LINE_AA)
        center = np.mean(corners, axis=0).astype(int)
        cv2.putText(image, f"tag {tag_id}", tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    counts = record.get("sample_counts", {})
    cv2.putText(image, f"tag samples: 0={counts.get(0, 0)} 1={counts.get(1, 0)} 2={counts.get(2, 0)} / 20", (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    locked = str(record.get("event", "")).startswith("stable_apriltag_")
    message = "POSE LOCKED" if locked else "OBSERVING: need all 3 tags"
    color = (0, 220, 0) if locked else (0, 180, 255)
    cv2.putText(image, message, (14, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("failed to encode AprilTag preview")
    return encoded.tobytes()


def publish_shared_view(
    shared: SharedView | None,
    color_rgb: np.ndarray,
    visible: dict[int, np.ndarray],
    record: dict,
    *,
    motion: str,
) -> None:
    """Publish one camera frame and its JSON state to the browser preview.

    The autonomous visual-lock loop runs outside the main observation loop, so
    it must use this same publisher explicitly.  ``SharedView`` owns the
    browser sequence; the camera frame index is only diagnostic and may repeat
    when a nested lock acquisition starts.
    """
    if shared is None:
        return
    pnp = record.get("pnp") or record.get("pnp_diagnostics") or {}
    sample_counts = record.get("sample_counts") or pnp.get("sample_counts", {})
    visible_tag_ids = record.get("visible_tag_ids")
    if visible_tag_ids is None:
        visible_tag_ids = sorted(int(tag_id) for tag_id in visible)
    shared.update(
        draw_overlay(color_rgb, visible, record),
        {
            "state": record.get("event", "frame"),
            "visible_tag_ids": visible_tag_ids,
            "sample_counts": sample_counts,
            "pnp": pnp,
            "arm_r_end_link_pose_base": record.get("arm_r_end_link_pose_base"),
            "right_color_camera_pose_base": record.get("right_color_camera_pose_base"),
            "workpiece_pose_right_color_camera": record.get("workpiece_pose_right_color_camera"),
            "workpiece_pose_base": record.get("workpiece_pose_base"),
            "pregrasp_arm_r_end_link_base": record.get("pregrasp_arm_r_end_link_base"),
            "grasp_arm_r_end_link_base": record.get("grasp_arm_r_end_link_base"),
            "current_to_pregrasp": record.get("current_to_pregrasp"),
            "calibration_required": record.get("calibration_required"),
            "motion": motion,
        },
    )


def target_record(
    targets, workpiece_pose, tf_info: dict, timestamp_ns: int, base_from_right_end: np.ndarray,
) -> dict:
    def serialize(matrix: np.ndarray) -> dict:
        return {
            "translation_m": [round(float(item), 7) for item in matrix[:3, 3]],
            "orientation_xyzw": [round(float(item), 8) for item in quaternion_from_rotation(matrix[:3, :3])],
            "matrix": [[round(float(item), 9) for item in row] for row in matrix],
        }
    pregrasp_distance_m, pregrasp_angle_deg = pose_distance(base_from_right_end, targets.base_from_pregrasp_end)
    return {
        "event": "stable_apriltag_target", "camera_timestamp_ns": timestamp_ns,
        "arm_r_end_link_pose_base": matrix_pose_record(base_from_right_end),
        "workpiece_pose_camera": serialize(workpiece_pose.camera_from_workpiece),
        "workpiece_pose_base": serialize(targets.base_from_workpiece),
        "pregrasp_arm_r_end_link_base": serialize(targets.base_from_pregrasp_end),
        "grasp_arm_r_end_link_base": serialize(targets.base_from_grasp_end),
        "pnp": {"mean_reprojection_error_px": round(workpiece_pose.mean_reprojection_error_px, 4),
                "max_reprojection_error_px": round(workpiece_pose.max_reprojection_error_px, 4),
                "inlier_count": workpiece_pose.inlier_count, "sample_counts": workpiece_pose.sample_counts},
        "current_to_pregrasp": {
            "translation_error_m": round(pregrasp_distance_m, 7),
            "orientation_error_deg": round(pregrasp_angle_deg, 4),
        },
        "gdk_tf": tf_info,
    }


def visual_lock_pose_ready(pose, min_inliers: int = 8) -> bool:
    """Return whether an already PnP-accepted pose is usable for live servo."""
    return pose is not None and int(pose.inlier_count) >= int(min_inliers)


def matrix_pose_record(matrix: np.ndarray) -> dict:
    """Serialize one named frame pose as a base-frame translation/quaternion."""
    return {
        "translation_m": [round(float(item), 7) for item in matrix[:3, 3]],
        "orientation_xyzw": [round(float(item), 8) for item in quaternion_from_rotation(matrix[:3, :3])],
    }


def workpiece_only_record(
    workpiece_pose, base_from_workpiece: np.ndarray, base_from_right_end: np.ndarray,
    right_end_from_camera: np.ndarray, tf_info: dict, timestamp_ns: int,
) -> dict:
    base_from_camera = base_from_right_end @ right_end_from_camera
    return {
        "event": "stable_apriltag_workpiece_pose", "calibration_required": "right_end_from_grasp_tcp",
        "camera_timestamp_ns": timestamp_ns,
        "arm_r_end_link_pose_base": matrix_pose_record(base_from_right_end),
        "right_color_camera_pose_base": matrix_pose_record(base_from_camera),
        "workpiece_pose_right_color_camera": matrix_pose_record(workpiece_pose.camera_from_workpiece),
        "workpiece_pose_base": matrix_pose_record(base_from_workpiece),
        "pnp": {"mean_reprojection_error_px": round(workpiece_pose.mean_reprojection_error_px, 4),
                "max_reprojection_error_px": round(workpiece_pose.max_reprojection_error_px, 4),
                "inlier_count": workpiece_pose.inlier_count, "sample_counts": workpiece_pose.sample_counts},
        "gdk_tf": tf_info,
    }


def acquire_fresh_visual_lock(
    *,
    camera,
    color_type,
    detector,
    estimator,
    intrinsic: np.ndarray,
    distortion,
    tf,
    controller: G2Controller,
    right_end_from_camera: np.ndarray,
    right_end_from_tcp: np.ndarray,
    workpiece_from_pregrasp_end: np.ndarray,
    max_tf_age_ms: float,
    max_end_motion_m: float,
    max_end_rotation_deg: float,
    timeout_s: float,
    last_timestamp_holder: dict,
    require_stationary: bool = True,
    shared: SharedView | None = None,
    min_inlier_count: int = 8,
):
    """Collect a fresh tag lock, optionally requiring a stationary end link."""
    if not 0.2 <= timeout_s <= 10.0:
        raise ValueError("fresh visual lock timeout must be in [0.2, 10] s")
    estimator.reset()
    anchor = controller.read_end_effector_pose("arm_r_end_link")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        image = camera.get_latest_image(color_type, 200.0)
        timestamp_ns = int(image.timestamp_ns)
        if timestamp_ns <= int(last_timestamp_holder.get("timestamp_ns", 0)):
            continue
        last_timestamp_holder["timestamp_ns"] = timestamp_ns
        color = decode_color_rgb(image, controller.gdk)
        current = controller.read_end_effector_pose("arm_r_end_link")
        motion_m = float(np.linalg.norm(np.asarray(current.position_m) - np.asarray(anchor.position_m)))
        motion_deg = quaternion_angle_deg(current.orientation_xyzw, anchor.orientation_xyzw)
        if require_stationary and (motion_m > max_end_motion_m or motion_deg > max_end_rotation_deg):
            estimator.reset()
            anchor = current
            continue
        visible = detect_corners(detector, color)
        timestamp_s = timestamp_ns / 1_000_000_000.0
        estimator.add(visible, timestamp_s=timestamp_s)
        pose = estimator.estimate(intrinsic, distortion, now_s=timestamp_s)
        frame_record = {
            "event": "frame",
            "visible_tag_ids": sorted(visible),
            "sample_counts": estimator.sample_counts,
            "pnp_diagnostics": estimator.last_diagnostics,
        }
        publish_shared_view(
            shared,
            color,
            visible,
            frame_record,
            motion="visual_servo" if not require_stationary else "visual_lock",
        )
        if not visual_lock_pose_ready(pose, min_inlier_count):
            continue
        base_from_end, tf_info = camera_to_base_at(tf, timestamp_ns, max_tf_age_ms)
        if tf_info["mode"] != "exact_image_time":
            estimator.reset()
            continue
        base_from_end_matrix = transform_from_dict(base_from_end)
        base_from_workpiece = base_from_end_matrix @ right_end_from_camera @ pose.camera_from_workpiece
        targets = g2_grasp_targets(
            pose.camera_from_workpiece,
            base_from_end_matrix,
            right_end_from_camera,
            right_end_from_tcp,
            workpiece_from_pregrasp_end=workpiece_from_pregrasp_end,
        )
        targets = horizontal_grasp_targets(targets, right_end_from_tcp)
        publish_shared_view(
            shared,
            color,
            visible,
            target_record(targets, pose, tf_info, timestamp_ns, base_from_end_matrix),
            motion="visual_servo" if not require_stationary else "visual_lock",
        )
        print(json.dumps({
            "event": "fresh_visual_lock",
            "camera_timestamp_ns": timestamp_ns,
            "pnp_mean_reprojection_error_px": round(pose.mean_reprojection_error_px, 4),
            "pnp_inlier_count": pose.inlier_count,
            "workpiece_translation_m": [round(float(value), 6) for value in base_from_workpiece[:3, 3]],
            "pregrasp_translation_m": [round(float(value), 6) for value in targets.base_from_pregrasp_end[:3, 3]],
            "tf_mode": tf_info["mode"],
        }, ensure_ascii=False), flush=True)
        return {
            "targets": targets,
            "base_from_workpiece": base_from_workpiece,
            "pose": pose,
            "tf_info": tf_info,
            "timestamp_ns": timestamp_ns,
        }
    raise VisualLockTimeout("自主视觉伺服在本次等待内未取得新的稳定 0/1/2 标签锁定")


def calibration_candidate_record(
    workpiece_pose, base_from_workpiece: np.ndarray, base_from_right_end: np.ndarray,
    tf_info: dict, timestamp_ns: int,
) -> dict:
    """Record a review-only tool-map candidate; this function never commands GDK."""
    right_end_from_tcp, base_from_pre_tcp, base_from_grasp_tcp = right_end_from_grasp_tcp_candidate(
        base_from_right_end, base_from_workpiece,
    )
    return {
        "event": "right_grasp_tcp_calibration_candidate",
        "camera_timestamp_ns": timestamp_ns,
        "calibration_assumption": (
            "operator manually placed the physical G2 grasp TCP at the source-package final grasp pose; "
            "this result is a candidate requiring review, not an automatic config update"
        ),
        "pnp": {
            "mean_reprojection_error_px": round(workpiece_pose.mean_reprojection_error_px, 4),
            "max_reprojection_error_px": round(workpiece_pose.max_reprojection_error_px, 4),
            "inlier_count": workpiece_pose.inlier_count,
            "sample_counts": workpiece_pose.sample_counts,
        },
        "gdk_tf": tf_info,
        "arm_r_end_link_pose_base": matrix_pose_record(base_from_right_end),
        "grasp_tcp_pose_base": matrix_pose_record(base_from_grasp_tcp),
        "pregrasp_tcp_pose_base": matrix_pose_record(base_from_pre_tcp),
        "right_end_from_grasp_tcp": {
            "matrix": [[round(float(item), 9) for item in row] for row in right_end_from_tcp],
        },
        "config_patch": {
            "right_end_from_grasp_tcp": {
                "matrix": [[round(float(item), 9) for item in row] for row in right_end_from_tcp],
            },
            "calibrated": False,
            "right_mode1_probe_verified": False,
        },
    }


def pregrasp_calibration_candidate_record(
    base_from_workpiece: np.ndarray, base_from_right_end: np.ndarray, grasp_end_target: np.ndarray,
    workpiece_pose, tf_info: dict, timestamp_ns: int,
) -> dict:
    """Record an operator-selected pregrasp relative to the current workpiece."""
    constrained_pregrasp_end, axial_distance_m, lateral_error_m, raw_orientation_error_deg = constrained_pregrasp_end_candidate(
        base_from_right_end, grasp_end_target,
    )
    workpiece_from_pregrasp_end = workpiece_from_pregrasp_end_candidate(
        base_from_workpiece, constrained_pregrasp_end,
    )
    return {
        "event": "right_pregrasp_calibration_candidate",
        "camera_timestamp_ns": timestamp_ns,
        "calibration_assumption": (
            "operator manually placed arm_r_end_link at a collision-free pregrasp with the same final grasp orientation; "
            "this result is a candidate requiring review, not an automatic config update"
        ),
        "pnp": {
            "mean_reprojection_error_px": round(workpiece_pose.mean_reprojection_error_px, 4),
            "max_reprojection_error_px": round(workpiece_pose.max_reprojection_error_px, 4),
            "inlier_count": workpiece_pose.inlier_count,
            "sample_counts": workpiece_pose.sample_counts,
        },
        "gdk_tf": tf_info,
        "raw_pregrasp_arm_r_end_link_pose_base": matrix_pose_record(base_from_right_end),
        "pregrasp_arm_r_end_link_pose_base": matrix_pose_record(constrained_pregrasp_end),
        "grasp_arm_r_end_link_pose_base": matrix_pose_record(grasp_end_target),
        "pregrasp_to_grasp": {
            "axial_distance_m": round(axial_distance_m, 7),
            "raw_lateral_error_m": round(lateral_error_m, 7),
            "raw_orientation_error_deg": round(raw_orientation_error_deg, 4),
            "candidate_orientation_error_deg": 0.0,
        },
        "workpiece_from_pregrasp_end": {
            "matrix": [[round(float(item), 9) for item in row] for row in workpiece_from_pregrasp_end],
        },
        "config_patch": {
            "workpiece_from_pregrasp_end": {
                "matrix": [[round(float(item), 9) for item in row] for row in workpiece_from_pregrasp_end],
            },
            "calibrated": False,
        },
    }


def target_pose(matrix: np.ndarray) -> CartesianPoseTarget:
    return CartesianPoseTarget(tuple(float(item) for item in matrix[:3, 3]), quaternion_from_rotation(matrix[:3, :3]))


def quaternion_angle_deg(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    q0 = np.asarray(first, dtype=np.float64)
    q1 = np.asarray(second, dtype=np.float64)
    q0 /= np.linalg.norm(q0)
    q1 /= np.linalg.norm(q1)
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, abs(float(np.dot(q0, q1)))))))


def orientation_relock_needed(
    locked_orientation: tuple[float, float, float, float],
    candidate_orientation: tuple[float, float, float, float],
    *,
    threshold_deg: float,
) -> bool:
    """Return whether a fresh visual orientation is large enough to relock.

    The visual servo intentionally ignores small orientation changes.  This
    keeps detector noise from continuously changing the commanded wrist pose
    while still allowing a genuinely rotated workpiece to be adopted later.
    """
    if threshold_deg < 0.0:
        raise ValueError("orientation relock threshold must be non-negative")
    return quaternion_angle_deg(locked_orientation, candidate_orientation) > threshold_deg


def orientation_relock_progress(
    candidate_orientation: tuple[float, float, float, float],
    previous_candidate_orientation: tuple[float, float, float, float] | None,
    candidate_count: int,
    *,
    consistency_deg: float,
    required_count: int,
) -> tuple[tuple[float, float, float, float], int, bool]:
    """Accumulate mutually consistent orientation observations before relock."""
    if candidate_count < 0 or consistency_deg < 0.0 or required_count < 1:
        raise ValueError("invalid orientation relock progress limits")
    if (
        previous_candidate_orientation is not None
        and quaternion_angle_deg(previous_candidate_orientation, candidate_orientation) <= consistency_deg
    ):
        next_count = candidate_count + 1
    else:
        next_count = 1
    return candidate_orientation, next_count, next_count >= required_count


def horizontal_grasp_targets(targets, right_end_from_tcp):
    """Keep end +Z horizontal and end +Y up; preserve the calibrated TCP point."""
    grasp = np.asarray(targets.base_from_grasp_end).copy()
    forward = grasp[:3, 2].copy()
    forward[2] = 0.0
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        raise RuntimeError("右手抓取方向接近竖直，无法确定水平朝向；拒绝动作")
    forward /= norm
    up = np.array([0.0, 0.0, 1.0])
    rotation = np.column_stack((np.cross(up, forward), up, forward))
    tcp_offset = np.asarray(right_end_from_tcp)[:3, 3]
    tcp_position = grasp[:3, 3] + grasp[:3, :3] @ tcp_offset
    distance = float(np.linalg.norm(grasp[:3, 3] - targets.base_from_pregrasp_end[:3, 3]))
    grasp[:3, :3] = rotation
    grasp[:3, 3] = tcp_position - rotation @ tcp_offset
    pregrasp = grasp.copy()
    pregrasp[:3, 3] -= forward * distance
    return GraspTargets(targets.base_from_workpiece, pregrasp, grasp)


# STEP red planar faces #201/#2010, midpoint of their eight vertices (mm → m).
RED_GRIP_CENTER_M = np.array([0.0, -0.039665062887915, -0.0413136150943175])


def facing_distance_targets(targets, distance_mm):
    if not math.isfinite(distance_mm) or not 10 <= distance_mm <= 300:
        raise ValueError("正对距离必须在 10–300 mm 之间（末端原点到红色夹取中心）")
    grasp = np.asarray(targets.base_from_grasp_end).copy()
    workpiece = np.asarray(targets.base_from_workpiece)
    grasp[:3, 3] = workpiece[:3, :3] @ RED_GRIP_CENTER_M + workpiece[:3, 3]
    pre = grasp.copy()
    pre[:3, 3] -= grasp[:3, 2] * (distance_mm / 1000.0)
    return GraspTargets(targets.base_from_workpiece, pre, grasp)


def targets_with_locked_orientation(targets, locked_rotation: np.ndarray) -> GraspTargets:
    """Keep live target translations while replacing both end-link rotations.

    AprilTag PnP supplies both translation and orientation.  During tracking,
    translation should follow the current workpiece, but small rotational
    detector changes must not make the wrist hunt.  This helper makes that
    invariant explicit for both pregrasp and grasp targets.
    """
    rotation = np.asarray(locked_rotation, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("locked_orientation must be a finite 3x3 rotation matrix")
    pregrasp = np.asarray(targets.base_from_pregrasp_end, dtype=np.float64).copy()
    grasp = np.asarray(targets.base_from_grasp_end, dtype=np.float64).copy()
    if pregrasp.shape != (4, 4) or grasp.shape != (4, 4):
        raise ValueError("grasp targets must contain 4x4 transforms")
    pregrasp[:3, :3] = rotation
    grasp[:3, :3] = rotation
    return GraspTargets(
        np.asarray(targets.base_from_workpiece, dtype=np.float64).copy()
        if hasattr(targets, "base_from_workpiece")
        else np.eye(4, dtype=np.float64),
        pregrasp,
        grasp,
    )


def move_right_gripper_with_feedback(
    controller: G2Controller,
    target_rad: float,
    *,
    tolerance_rad: float = 0.04,
    timeout_s: float = 6.0,
) -> dict:
    """Drive the right Omnipicker and require its GDK end-state feedback."""
    if not 0.0 < tolerance_rad <= 0.20 or not 0.1 <= timeout_s <= 15.0:
        raise ValueError("right gripper feedback limits are invalid")
    before = controller.read_gripper_position("right")
    attempts = 0
    latest = before
    deadline = time.monotonic() + timeout_s
    while True:
        controller.move_gripper("right", target_rad)
        attempts += 1
        time.sleep(0.10)
        latest = controller.read_gripper_position("right")
        if latest is not None and abs(latest - target_rad) <= tolerance_rad:
            return {
                "target_rad": float(target_rad),
                "before_position_rad": before,
                "actual_position_rad": latest,
                "tolerance_rad": tolerance_rad,
                "attempts": attempts,
            }
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"右夹爪未在 {timeout_s:.1f} s 内到达目标；target={target_rad:.4f} rad, "
                f"actual={latest}；停止后续机械臂动作"
            )


def close_right_gripper(
    controller: G2Controller,
    target_rad: float,
    *,
    settle_s: float = 0.8,
    min_motion_rad: float = 0.02,
) -> dict:
    """Close the right Omnipicker after the arm has actually reached grasp.

    A workpiece can mechanically prevent the jaws from reaching the empty-tool
    zero position, so closing is confirmed by the command plus a short settle
    readback rather than by requiring ``actual == target``.
    """
    if not 0.0 <= settle_s <= 5.0 or not 0.0 <= min_motion_rad <= 0.7853981633974483:
        raise ValueError("right gripper close settle time must be in [0, 5] s and minimum motion in [0, pi/4] rad")
    before = controller.read_gripper_position("right")
    if before is None:
        raise RuntimeError("右夹爪闭合前没有有效位置反馈；停止并判定抓取失败")
    attempts = 0
    latest = before
    deadline = time.monotonic() + settle_s
    while True:
        controller.move_gripper("right", target_rad)
        attempts += 1
        if settle_s:
            time.sleep(min(0.10, max(0.0, deadline - time.monotonic())))
        latest = controller.read_gripper_position("right")
        if not settle_s or time.monotonic() >= deadline:
            break
    if latest is None:
        raise RuntimeError("右夹爪闭合后没有有效位置反馈；停止并判定抓取失败")
    motion_toward_target = None
    motion_toward_target = abs(before - target_rad) - abs(latest - target_rad)
    if abs(before - target_rad) > min_motion_rad and motion_toward_target < min_motion_rad:
        raise RuntimeError(
            "右夹爪闭合命令已重发但反馈未向闭合方向移动；"
            f"before={before:.6f} rad, actual={latest:.6f} rad, "
            f"target={target_rad:.6f} rad；判定抓取失败"
        )
    close_state = "target_reached" if abs(latest - target_rad) <= 0.04 else "blocked_before_target"
    return {
        "target_rad": float(target_rad),
        "before_position_rad": before,
        "actual_position_rad": latest,
        "settle_s": float(settle_s),
        "attempts": attempts,
        "motion_toward_target_rad": motion_toward_target,
        "close_state": close_state,
    }


def close_right_gripper_on_pose_stream(
    controller: G2Controller,
    stream,
    target_rad: float,
    *,
    settle_s: float = 0.8,
    min_motion_rad: float = 0.02,
) -> dict:
    """Close the right Omnipicker while the final pose stream stays active.

    GDK's pose controller accepts the gripper joint target in the same
    ``EndEffectorPose`` request.  Keeping that request at 50 Hz avoids a race
    where a standalone ``right_tool`` command is sent while the last pose
    command is still owned by the pose controller.
    """
    if not 0.0 <= settle_s <= 5.0 or not 0.0 <= min_motion_rad <= 0.7853981633974483:
        raise ValueError("right gripper close settle time must be in [0, 5] s and minimum motion in [0, pi/4] rad")
    before = controller.read_gripper_position("right")
    if before is None:
        raise RuntimeError("右夹爪闭合前没有有效位置反馈；停止并判定抓取失败")
    stream.set_right_gripper_target(target_rad)
    # Allow at least one 50-Hz pose request even when a test/configuration
    # supplies settle_s=0.  The normal execute path uses several seconds.
    deadline = time.monotonic() + max(float(settle_s), 0.12)
    samples = 0
    latest = before
    while True:
        stream.raise_if_failed()
        latest = controller.read_gripper_position("right")
        samples += 1
        if latest is None:
            raise RuntimeError("右夹爪闭合后没有有效位置反馈；停止并判定抓取失败")
        if abs(latest - target_rad) <= 0.04:
            return {
                "target_rad": float(target_rad),
                "before_position_rad": before,
                "actual_position_rad": latest,
                "settle_s": float(settle_s),
                "command_path": "right_end_effector_pose_stream",
                "feedback_samples": samples,
                "motion_toward_target_rad": abs(before - target_rad) - abs(latest - target_rad),
                "close_state": "target_reached",
            }
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            break
        time.sleep(min(0.10, remaining))
    motion_toward_target = abs(before - target_rad) - abs(latest - target_rad)
    if abs(before - target_rad) > min_motion_rad and motion_toward_target < min_motion_rad:
        raise RuntimeError(
            "右夹爪闭合命令已通过右臂位姿流重发但反馈未向闭合方向移动；"
            f"before={before:.6f} rad, actual={latest:.6f} rad, "
            f"target={target_rad:.6f} rad；判定抓取失败"
        )
    return {
        "target_rad": float(target_rad),
        "before_position_rad": before,
        "actual_position_rad": latest,
        "settle_s": float(settle_s),
        "command_path": "right_end_effector_pose_stream",
        "feedback_samples": samples,
        "motion_toward_target_rad": motion_toward_target,
        "close_state": "blocked_before_target",
    }


def bounded_pose_waypoints(
    start: CartesianPoseTarget,
    target: CartesianPoseTarget,
    *,
    max_translation_m: float,
    max_orientation_deg: float,
) -> tuple[CartesianPoseTarget, ...]:
    """Return finite straight-line pose waypoints with bounded increments."""
    if not 0.001 <= max_translation_m <= 0.10:
        raise ValueError("autonomous waypoint translation must be in [0.001, 0.10] m")
    if not 0.1 <= max_orientation_deg <= 30.0:
        raise ValueError("autonomous waypoint orientation must be in [0.1, 30] deg")
    start_position = np.asarray(start.position_m, dtype=np.float64)
    target_position = np.asarray(target.position_m, dtype=np.float64)
    distance_m = float(np.linalg.norm(target_position - start_position))
    angle_deg = quaternion_angle_deg(start.orientation_xyzw, target.orientation_xyzw)
    count = max(1, int(math.ceil(distance_m / max_translation_m)), int(math.ceil(angle_deg / max_orientation_deg)))
    waypoints = []
    for index in range(1, count + 1):
        fraction = index / count
        position = start_position + fraction * (target_position - start_position)
        waypoints.append(CartesianPoseTarget(
            tuple(float(value) for value in position),
            quaternion_slerp(start.orientation_xyzw, target.orientation_xyzw, fraction),
        ))
    return tuple(waypoints)


def execute_autonomous_pregrasp(
    controller: G2Controller,
    current: CartesianPoseTarget,
    pregrasp: CartesianPoseTarget,
    config: dict,
) -> CartesianPoseTarget:
    """Move from a coarse QR-visible start to pregrasp in bounded pose steps."""
    autonomous = require_mapping(config.get("autonomous_approach"), "autonomous_approach")
    if not autonomous.get("enabled"):
        raise RuntimeError("autonomous_approach.enabled=false：未启用二维码自主预接近")
    waypoints = bounded_pose_waypoints(
        current, pregrasp,
        max_translation_m=get_number(autonomous, "max_waypoint_translation_m", minimum=0.001),
        max_orientation_deg=get_number(autonomous, "max_waypoint_orientation_deg", minimum=0.1),
    )
    stream = controller.start_right_end_effector_position_stream(
        current,
        rate_hz=50.0,
        life_time_s=0.10,
        max_translation_velocity_mps=get_number(autonomous, "speed_mps", minimum=0.001),
        allow_orientation_change=True,
        max_angular_velocity_rad_s=math.radians(get_number(autonomous, "angular_speed_deg_s", minimum=0.1)),
    )
    try:
        for index, waypoint in enumerate(waypoints, start=1):
            stream.update_target(waypoint)
            actual = stream.wait_until_actual_target(
                waypoint,
                get_number(autonomous, "waypoint_timeout_s", minimum=0.1),
                position_tolerance_m=get_number(autonomous, "actual_position_tolerance_m", minimum=0.0001),
                orientation_tolerance_deg=get_number(autonomous, "actual_orientation_tolerance_deg", minimum=0.1),
                stable_samples=int(autonomous.get("stable_samples", 3)),
            )
            if actual is None:
                raise RuntimeError(f"自主预接近第 {index}/{len(waypoints)} 段未实际稳定到位；停止且不闭爪")
    finally:
        stream.stop()
    return controller.read_end_effector_pose("arm_r_end_link")


def execute_live_autonomous_pregrasp(
    controller: G2Controller,
    initial_targets,
    acquire_lock,
    config: dict,
    *,
    shared: SharedView | None = None,
    face_distance_mm: float | None = None,
):
    """Receding-horizon visual servo with a bounded hold/reacquire state.

    ``initial_targets`` is retained for the caller's tuple shape, but it is
    never used as a motion reference.  Each fresh lock updates the target
    translation, while the wrist orientation remains at its current visual
    lock.  A persistent orientation change is treated as a candidate while the
    stream holds the current physical pose; only repeated mutually-consistent
    locks can relock the wrist orientation.
    """
    autonomous = require_mapping(config.get("autonomous_approach"), "autonomous_approach")
    if not autonomous.get("enabled"):
        raise RuntimeError("autonomous_approach.enabled=false：未启用二维码自主预接近")
    local_motion = require_mapping(config.get("local_motion"), "local_motion")
    max_local_m = get_number(local_motion, "max_initial_pregrasp_error_m", minimum=0.0)
    max_local_deg = get_number(local_motion, "max_initial_orientation_error_deg", minimum=0.0)
    max_iterations = int(autonomous.get("max_iterations", 80))
    if not 1 <= max_iterations <= 300:
        raise RuntimeError("autonomous_approach.max_iterations 必须在 1–300")
    max_total_duration_s = get_number(autonomous, "max_total_duration_s", minimum=1.0)
    max_target_delta_m = get_number(autonomous, "max_target_delta_m", minimum=0.001)
    max_target_delta_deg = get_number(autonomous, "max_target_delta_deg", minimum=0.1)
    hold_timeout_s = get_number(
        {"visual_lock_hold_timeout_s": autonomous.get("visual_lock_hold_timeout_s", 30.0)},
        "visual_lock_hold_timeout_s", minimum=0.5,
    )
    jump_confirmations = int(autonomous.get("jump_reacquire_confirmations", 3))
    if not 2 <= jump_confirmations <= 10:
        raise RuntimeError("autonomous_approach.jump_reacquire_confirmations 必须在 2–10")
    jump_consistency_m = get_number(
        {"jump_cluster_delta_m": autonomous.get("jump_cluster_delta_m", 0.010)},
        "jump_cluster_delta_m", minimum=0.001,
    )
    jump_consistency_deg = get_number(
        {"jump_cluster_delta_deg": autonomous.get("jump_cluster_delta_deg", 3.0)},
        "jump_cluster_delta_deg", minimum=0.1,
    )
    orientation_relock_threshold_deg = get_number(
        {"orientation_relock_threshold_deg": autonomous.get("orientation_relock_threshold_deg", 2.0)},
        "orientation_relock_threshold_deg", minimum=0.1,
    )
    orientation_relock_confirmations = int(autonomous.get("orientation_relock_confirmations", 3))
    if not 2 <= orientation_relock_confirmations <= 10:
        raise RuntimeError("autonomous_approach.orientation_relock_confirmations 必须在 2–10")
    orientation_relock_consistency_deg = get_number(
        {"orientation_relock_consistency_deg": autonomous.get(
            "orientation_relock_consistency_deg", jump_consistency_deg,
        )},
        "orientation_relock_consistency_deg", minimum=0.1,
    )
    current = controller.read_end_effector_pose("arm_r_end_link")
    deadline = time.monotonic() + max_total_duration_s

    def acquire_lock_while_holding(stream=None):
        """Retry only visual timeouts while preserving the last actual pose."""
        hold_deadline = min(deadline, time.monotonic() + hold_timeout_s)
        attempt = 0
        while True:
            if time.monotonic() >= deadline:
                raise RuntimeError("自主视觉预接近超过总时间预算；停止且不闭爪")
            try:
                if stream is not None:
                    stream.raise_if_failed()
                return acquire_lock(moving=stream is not None)
            except VisualLockTimeout as exc:
                attempt += 1
                if stream is not None:
                    hold_pose = controller.read_end_effector_pose("arm_r_end_link")
                    stream.update_target(hold_pose)
                    stream.raise_if_failed()
                remaining_s = max(0.0, min(deadline, hold_deadline) - time.monotonic())
                print(json.dumps({
                    "event": "visual_lock_waiting",
                    "attempt": attempt,
                    "remaining_hold_s": round(remaining_s, 2),
                    "reason": str(exc),
                    "motion": "hold_current_pose",
                }, ensure_ascii=False), flush=True)
                if shared is not None:
                    shared.update_status({
                        "state": "visual_lock_waiting",
                        "motion": "hold_current_pose",
                        "visual_lock_waiting": {
                            "attempt": attempt,
                            "remaining_hold_s": round(remaining_s, 2),
                            "reason": str(exc),
                        },
                    })
                if time.monotonic() >= hold_deadline:
                    raise RuntimeError(
                        f"二维码在保持 {hold_timeout_s:.1f} s 内未恢复；停止且不闭爪"
                    ) from exc

    # The first lock is fresh too; a stale main-loop target cannot become the
    # start-distance reference. There is no stream yet, so the robot remains
    # at the operator's stationary coarse pose while this lock is acquired.
    first_lock = acquire_lock_while_holding()
    first_targets = first_lock["targets"]
    first_pregrasp = target_pose(first_targets.base_from_pregrasp_end)
    start_distance = float(np.linalg.norm(np.asarray(current.position_m) - np.asarray(first_pregrasp.position_m)))
    start_angle_deg = quaternion_angle_deg(current.orientation_xyzw, first_pregrasp.orientation_xyzw)
    print(json.dumps({
        "event": "autonomous_start_offset_observed",
        "distance_m": round(start_distance, 6),
        "orientation_deg": round(start_angle_deg, 3),
        "motion": "bounded_visual_servo",
        "note": "起点偏差不再作为硬拒绝条件；仍受单步、限速、总时长和最终局部门槛保护",
    }, ensure_ascii=False), flush=True)
    stream = controller.start_right_end_effector_position_stream(
        current,
        rate_hz=50.0,
        life_time_s=0.10,
        max_translation_velocity_mps=get_number(autonomous, "speed_mps", minimum=0.001),
        allow_orientation_change=True,
        max_angular_velocity_rad_s=math.radians(get_number(autonomous, "angular_speed_deg_s", minimum=0.1)),
    )
    previous_lock = None
    locked_rotation = np.asarray(first_targets.base_from_pregrasp_end, dtype=np.float64)[:3, :3].copy()
    orientation_candidate = None
    orientation_candidate_count = 0
    def locked_targets(targets):
        result = targets_with_locked_orientation(targets, locked_rotation)
        if face_distance_mm is not None:
            result = facing_distance_targets(result, face_distance_mm)
        return result

    latest_targets = locked_targets(first_targets)
    pending_lock = first_lock
    jump_candidate = None
    jump_candidate_count = 0
    motion_steps = 0
    last_progress_pose = current
    max_waypoint_translation_m = get_number(
        autonomous, "max_waypoint_translation_m", minimum=0.001,
    )
    max_waypoint_orientation_deg = get_number(
        autonomous, "max_waypoint_orientation_deg", minimum=0.1,
    )

    def prepare_lock(lock):
        """Apply live translation and gate orientation changes before motion."""
        nonlocal locked_rotation, orientation_candidate, orientation_candidate_count
        raw_targets = lock["targets"]
        candidate_orientation = target_pose(raw_targets.base_from_pregrasp_end).orientation_xyzw
        locked_orientation = quaternion_from_rotation(locked_rotation)
        if orientation_relock_needed(
            locked_orientation,
            candidate_orientation,
            threshold_deg=orientation_relock_threshold_deg,
        ):
            previous_orientation_candidate = orientation_candidate
            (
                orientation_candidate,
                orientation_candidate_count,
                ready,
            ) = orientation_relock_progress(
                candidate_orientation,
                orientation_candidate,
                orientation_candidate_count,
                consistency_deg=orientation_relock_consistency_deg,
                required_count=orientation_relock_confirmations,
            )
            hold_pose = controller.read_end_effector_pose("arm_r_end_link")
            stream.update_target(hold_pose)
            stream.raise_if_failed()
            print(json.dumps({
                "event": "visual_orientation_relock_waiting",
                "orientation_delta_deg": round(
                    quaternion_angle_deg(locked_orientation, candidate_orientation), 3,
                ),
                "candidate_consistency_deg": round(
                    quaternion_angle_deg(previous_orientation_candidate, candidate_orientation), 3,
                ) if previous_orientation_candidate is not None else None,
                "candidate_count": orientation_candidate_count,
                "required_count": orientation_relock_confirmations,
                "motion": "hold_current_pose",
            }, ensure_ascii=False), flush=True)
            if shared is not None:
                shared.update_status({
                    "state": "visual_orientation_relock_waiting",
                    "motion": "hold_current_pose",
                    "visual_orientation_relock": {
                        "orientation_delta_deg": round(
                            quaternion_angle_deg(locked_orientation, candidate_orientation), 3,
                        ),
                        "candidate_count": orientation_candidate_count,
                        "required_count": orientation_relock_confirmations,
                    },
                })
            if not ready:
                return None
            locked_rotation = np.asarray(raw_targets.base_from_pregrasp_end, dtype=np.float64)[:3, :3].copy()
            orientation_candidate = None
            orientation_candidate_count = 0
            print(json.dumps({
                "event": "visual_orientation_relocked",
                "orientation_deg": round(
                    quaternion_angle_deg(locked_orientation, candidate_orientation), 3,
                ),
                "motion": "resume_visual_servo",
            }, ensure_ascii=False), flush=True)
        else:
            orientation_candidate = None
            orientation_candidate_count = 0
        return locked_targets(raw_targets)

    try:
        while motion_steps < max_iterations:
            if time.monotonic() >= deadline:
                raise RuntimeError("自主视觉预接近超过总时间预算；停止且不闭爪")
            lock = pending_lock if pending_lock is not None else acquire_lock_while_holding(stream)
            pending_lock = None
            if previous_lock is not None:
                target_delta_m, target_delta_deg = pose_distance(
                    previous_lock["base_from_workpiece"], lock["base_from_workpiece"],
                )
                if target_delta_m > max_target_delta_m or target_delta_deg > max_target_delta_deg:
                    candidate_delta_m = math.inf
                    candidate_delta_deg = math.inf
                    if jump_candidate is not None:
                        candidate_delta_m, candidate_delta_deg = pose_distance(
                            jump_candidate["base_from_workpiece"], lock["base_from_workpiece"],
                        )
                    if (
                        jump_candidate is not None
                        and candidate_delta_m <= jump_consistency_m
                        and candidate_delta_deg <= jump_consistency_deg
                    ):
                        jump_candidate_count += 1
                    else:
                        jump_candidate = lock
                        jump_candidate_count = 1
                    hold_pose = controller.read_end_effector_pose("arm_r_end_link")
                    stream.update_target(hold_pose)
                    print(json.dumps({
                        "event": "visual_target_jump_waiting",
                        "target_delta_m": round(target_delta_m, 6),
                        "target_delta_deg": round(target_delta_deg, 3),
                        "candidate_consistency_m": round(candidate_delta_m, 6) if math.isfinite(candidate_delta_m) else None,
                        "candidate_consistency_deg": round(candidate_delta_deg, 3) if math.isfinite(candidate_delta_deg) else None,
                        "candidate_count": jump_candidate_count,
                        "required_count": jump_confirmations,
                        "motion": "hold_current_pose",
                    }, ensure_ascii=False), flush=True)
                    if jump_candidate_count < jump_confirmations:
                        continue
                    # The candidate survived repeated fresh observations. Use
                    # this latest lock as the new reference and resume bounded
                    # motion from the actual held pose.
                    jump_candidate = None
                    jump_candidate_count = 0
                else:
                    jump_candidate = None
                    jump_candidate_count = 0
            prepared_targets = prepare_lock(lock)
            if prepared_targets is None:
                continue
            previous_lock = lock
            latest_targets = prepared_targets
            pregrasp = target_pose(latest_targets.base_from_pregrasp_end)
            actual = controller.read_end_effector_pose("arm_r_end_link")
            current_distance = float(np.linalg.norm(
                np.asarray(actual.position_m) - np.asarray(pregrasp.position_m),
            ))
            current_angle_deg = quaternion_angle_deg(actual.orientation_xyzw, pregrasp.orientation_xyzw)
            if current_distance <= max_local_m and current_angle_deg <= max_local_deg:
                confirmation = acquire_lock_while_holding(stream)
                confirm_distance, confirm_angle_deg = pose_distance(
                    previous_lock["base_from_workpiece"], confirmation["base_from_workpiece"],
                )
                if confirm_distance > max_target_delta_m or confirm_angle_deg > max_target_delta_deg:
                    jump_candidate = confirmation
                    jump_candidate_count = 1
                    stream.update_target(controller.read_end_effector_pose("arm_r_end_link"))
                    print(json.dumps({
                        "event": "visual_target_jump_waiting",
                        "target_delta_m": round(confirm_distance, 6),
                        "target_delta_deg": round(confirm_angle_deg, 3),
                        "candidate_count": jump_candidate_count,
                        "required_count": jump_confirmations,
                        "motion": "hold_current_pose",
                    }, ensure_ascii=False), flush=True)
                    continue
                prepared_confirmation = prepare_lock(confirmation)
                if prepared_confirmation is None:
                    continue
                latest_targets = prepared_confirmation
                final_pregrasp = target_pose(latest_targets.base_from_pregrasp_end)
                actual = controller.read_end_effector_pose("arm_r_end_link")
                final_distance = float(np.linalg.norm(
                    np.asarray(actual.position_m) - np.asarray(final_pregrasp.position_m),
                ))
                final_angle_deg = quaternion_angle_deg(actual.orientation_xyzw, final_pregrasp.orientation_xyzw)
                if final_distance <= max_local_m and final_angle_deg <= max_local_deg:
                    return actual, latest_targets
                previous_lock = confirmation
                continue
            waypoints = bounded_pose_waypoints(
                actual,
                pregrasp,
                max_translation_m=max_waypoint_translation_m,
                max_orientation_deg=max_waypoint_orientation_deg,
            )
            stream.update_target(waypoints[0])
            stream.raise_if_failed()
            # Keep a short target horizon continuously fed to the 50 Hz GDK
            # stream. Counting physical progress rather than visual updates
            # preserves the bounded-motion budget without stopping at each
            # 5 mm waypoint to reacquire the tags.
            progress_m = float(np.linalg.norm(
                np.asarray(actual.position_m) - np.asarray(last_progress_pose.position_m),
            ))
            progress_deg = quaternion_angle_deg(actual.orientation_xyzw, last_progress_pose.orientation_xyzw)
            if (
                progress_m >= 0.5 * max_waypoint_translation_m
                or progress_deg >= 0.5 * max_waypoint_orientation_deg
            ):
                motion_steps += 1
                last_progress_pose = actual
            time.sleep(0.02)
        raise RuntimeError("自主视觉预接近超过 waypoint 次数预算；停止且不闭爪")
    finally:
        stream.stop()


def execute_local_grasp(
    controller: G2Controller,
    targets,
    config: dict,
    *,
    autonomous_approach: bool = False,
    max_angular_velocity_rad_s: float | None = None,
) -> dict:
    safety = require_mapping(config.get("local_motion"), "local_motion")
    if not config.get("calibrated"):
        raise RuntimeError("config.calibrated=false：禁止动作；先完成 G2 右端链接/抓取 TCP 标定")
    if not config.get("right_mode1_probe_verified"):
        raise RuntimeError("right_mode1_probe_verified=false：未通过受看护的右手 1 mm mode-1 探针，禁止动作")
    current = controller.read_end_effector_pose("arm_r_end_link")
    pregrasp = target_pose(targets.base_from_pregrasp_end)
    grasp = target_pose(targets.base_from_grasp_end)
    current_matrix = np.eye(4, dtype=np.float64)
    # Only pose_distance needs an SE(3); construct it from the known target
    # after separately comparing quaternion angle below.
    current_matrix[:3, 3] = current.position_m
    target_matrix = targets.base_from_pregrasp_end
    start_distance = float(np.linalg.norm(current_matrix[:3, 3] - target_matrix[:3, 3]))
    dot = abs(sum(a * b for a, b in zip(current.orientation_xyzw, pregrasp.orientation_xyzw)))
    start_angle_deg = math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))
    max_start_m = get_number(safety, "max_initial_pregrasp_error_m", minimum=0.0)
    max_start_deg = get_number(safety, "max_initial_orientation_error_deg", minimum=0.0)
    if autonomous_approach:
        current = execute_autonomous_pregrasp(controller, current, pregrasp, config)
        current_position = np.asarray(current.position_m, dtype=np.float64)
        start_distance = float(np.linalg.norm(current_position - target_matrix[:3, 3]))
        start_angle_deg = quaternion_angle_deg(current.orientation_xyzw, pregrasp.orientation_xyzw)
    if not autonomous_approach and (start_distance > max_start_m or start_angle_deg > max_start_deg):
        raise RuntimeError(
            f"当前右端链接不在已标定局部起点：{start_distance * 1000:.1f} mm / {start_angle_deg:.1f} deg；"
            "不得用无规划 mode-1 做大范围靠近"
        )
    if autonomous_approach and (start_distance > max_start_m or start_angle_deg > max_start_deg):
        raise RuntimeError(
            f"自主预接近后仍未进入最终起点范围：{start_distance * 1000:.1f} mm / {start_angle_deg:.1f} deg"
        )
    final_distance, final_angle = pose_distance(targets.base_from_pregrasp_end, targets.base_from_grasp_end)
    if final_angle > 0.05 or final_distance > get_number(safety, "max_final_approach_m", minimum=0.001):
        raise RuntimeError("最终抓取目标不满足已标定的纯平移局部范围")
    gripper = require_mapping(config.get("gripper"), "gripper")
    closed_rad = get_number(gripper, "closed_rad", minimum=-0.7853981633974483)
    close_settle_s = float(gripper.get("close_settle_s", 0.8))
    close_min_motion_rad = float(gripper.get("close_min_motion_rad", 0.02))
    stream_kwargs = {
        "max_translation_velocity_mps": get_number(safety, "final_speed_mps", minimum=0.001),
    }
    if max_angular_velocity_rad_s is not None:
        stream_kwargs["max_angular_velocity_rad_s"] = max_angular_velocity_rad_s
    stream = controller.start_right_end_effector_position_stream(
        CartesianPoseTarget(current.position_m, pregrasp.orientation_xyzw),
        rate_hz=50.0,
        life_time_s=0.10,
        **stream_kwargs,
    )
    try:
        stream.update_target(grasp)
        stable_samples = int(safety.get("final_actual_stable_samples", 10))
        # Speed overrides scale the old timeout, but settling and feedback
        # sampling still need wall-clock time independent of travel speed.
        travel_m = float(np.linalg.norm(np.asarray(grasp.position_m) - current.position_m))
        timeout_s = max(
            get_number(safety, "final_timeout_s", minimum=0.1),
            travel_m / stream_kwargs["max_translation_velocity_mps"] + stable_samples * 0.02 + 1.0,
        )
        actual = stream.wait_until_actual_target(
            grasp, timeout_s,
            position_tolerance_m=get_number(safety, "final_actual_position_tolerance_m", minimum=0.0001),
            orientation_tolerance_deg=get_number(safety, "final_actual_orientation_tolerance_deg", minimum=0.1),
            stable_samples=stable_samples,
        )
        if actual is None:
            feedback = controller.read_end_effector_pose("arm_r_end_link")
            error_mm = float(np.linalg.norm(np.asarray(grasp.position_m) - feedback.position_m)) * 1000
            error_deg = quaternion_angle_deg(grasp.orientation_xyzw, feedback.orientation_xyzw)
            raise RuntimeError(
                "右手最终局部接近未实际稳定到抓取点；停止发送目标且不闭爪："
                f"timeout_s={timeout_s:.3f}, position_error_mm={error_mm:.3f}, "
                f"orientation_error_deg={error_deg:.3f}"
            )
        return close_right_gripper_on_pose_stream(
            controller,
            stream,
            closed_rad,
            settle_s=close_settle_s,
            min_motion_rad=close_min_motion_rad,
        )
    finally:
        stream.stop()


def return_screwdriver(controller, saved_pose, config, emit) -> None:
    """Return via base +Z clearance; release only after both measured arrivals."""
    above = CartesianPoseTarget(
        (saved_pose.position_m[0], saved_pose.position_m[1], saved_pose.position_m[2] + 0.10),
        saved_pose.orientation_xyzw,
    )
    auto = require_mapping(config.get("autonomous_approach"), "autonomous_approach")
    local = require_mapping(config.get("local_motion"), "local_motion")
    speed = get_number(auto, "speed_mps", minimum=0.001)
    angular = get_number(auto, "angular_speed_deg_s", minimum=0.1)
    current = controller.read_end_effector_pose("arm_r_end_link")
    stream = controller.start_right_end_effector_position_stream(
        current, rate_hz=50.0, life_time_s=0.10,
        max_translation_velocity_mps=speed, allow_orientation_change=True,
        max_angular_velocity_rad_s=math.radians(angular),
    )
    try:
        for stage, target in (("above_grasp", above), ("original_grasp", saved_pose)):
            distance = float(np.linalg.norm(np.asarray(target.position_m) - current.position_m))
            duration = max(distance / speed, quaternion_angle_deg(current.orientation_xyzw, target.orientation_xyzw) / angular)
            stream.update_target(target)
            actual = stream.wait_until_actual_target(
                target, duration + 3.0,
                position_tolerance_m=get_number(local, "final_actual_position_tolerance_m", minimum=0.0001),
                orientation_tolerance_deg=get_number(local, "final_actual_orientation_tolerance_deg", minimum=0.1),
                stable_samples=int(local.get("final_actual_stable_samples", 10)),
            )
            if actual is None:
                raise RuntimeError(f"电批放回 {stage} 未实际稳定到位；停止且不松爪")
            current = actual
            emit({"event": "right_return_arrived", "stage": stage, "position_m": list(actual.position_m)})
    finally:
        stream.stop()
    gripper = require_mapping(config.get("gripper"), "gripper")
    opened = move_right_gripper_with_feedback(controller, float(gripper.get("open_rad", -0.785)))
    emit({"event": "right_return_released", **opened})


def wait_for_manual_finish(worker, *, sleep=time.sleep) -> None:
    """Keep takeover alive after completion; only a new long hold finishes."""
    worker.allow_advance_right = True
    print("右手流程已完成；可双击任一按钮接管，右按钮长按 2 秒放回电批后结束程序。", flush=True)
    while True:
        worker.raise_if_failed()
        if worker.advance_requested:
            return
        sleep(0.01)


def main() -> int:
    args = parse_args()
    if args.face_distance_mm is not None:
        if not math.isfinite(args.face_distance_mm) or not 10 <= args.face_distance_mm <= 300 or args.post_grasp_pr2 or (args.execute and not args.autonomous_approach):
            print("正对模式距离需为 10–300 mm，执行需自主接近且不能运输至 PR2", file=sys.stderr)
            return 2
    if (
        args.max_frames < 0 or not 0 <= args.web_port <= 65535
        or (args.execute and not args.allow_motion)
        or (args.execute and args.calibrate_grasp_tcp)
        or (args.execute and args.calibrate_pregrasp)
        or (args.calibrate_grasp_tcp and args.calibrate_pregrasp)
        or (args.autonomous_approach and not args.execute)
        or (args.premove_pr1 and not args.execute)
        or (args.post_grasp_pr2 and not args.execute)
        or (args.post_grasp_pr2 and not args.premove_pr1)
    ):
        print(
            "--max-frames / --web-port 无效；--execute 必须同时提供 --allow-motion，"
            "标定模式不能与动作共用；--autonomous-approach、--premove-pr1 和 "
            "--post-grasp-pr2 必须与 --execute 共用，且 --post-grasp-pr2 必须同时启用 --premove-pr1。",
            file=sys.stderr,
        )
        return 2
    if (
        not 1.0 <= args.spacemouse_speed_mm_s <= 30.0
        or not 1.0 <= args.spacemouse_rotation_deg_s <= 28.6
        or not 0.0 <= args.spacemouse_deadzone <= 0.5
        or (args.auto_speed_mm_s is not None and not 1.0 <= args.auto_speed_mm_s <= 18.0)
        or (args.auto_rotation_deg_s is not None and not 1.0 <= args.auto_rotation_deg_s <= 28.6)
        or (args.auto_preset_velocity_rad_s is not None and not 0.001 <= args.auto_preset_velocity_rad_s <= 0.15)
    ):
        print("参数错误：SpaceMouse 或视觉自动速度参数无效。", file=sys.stderr)
        return 2
    if args.spacemouse_device and not args.execute:
        print("--spacemouse-device 必须与 --execute 共用。", file=sys.stderr)
        return 2
    terminal_status_active = False
    try:
        config = require_mapping(json.loads(args.config.expanduser().read_text(encoding="utf-8")), "config")
        config = apply_auto_motion_overrides(
            config,
            speed_mm_s=args.auto_speed_mm_s,
            rotation_deg_s=args.auto_rotation_deg_s,
        )
        preset_velocity_rad_s = (
            PRESET_VELOCITY_RAD_S
            if args.auto_preset_velocity_rad_s is None
            else args.auto_preset_velocity_rad_s
        )
        pr1_target = load_pr1_pose(args.poses) if args.premove_pr1 else None
        pr2_target = load_pr2_pose(args.poses) if args.post_grasp_pr2 else None
        tag_geometry = load_tag_geometry(config, args.tag_geometry_profile)
        tool_mapping = config.get("right_end_from_grasp_tcp")
        right_end_from_tcp = None if tool_mapping is None else as_transform(tool_mapping, "right_end_from_grasp_tcp")
        pregrasp_mapping = config.get("workpiece_from_pregrasp_end")
        workpiece_from_pregrasp_end = None if pregrasp_mapping is None else as_transform(
            pregrasp_mapping, "workpiece_from_pregrasp_end",
        )
        estimator = RollingWorkpiecePoseEstimator(
            sample_count=int(config.get("tag_sample_count", 20)),
            max_reprojection_error_px=float(config.get("max_reprojection_error_px", 3.0)),
            max_sample_age_s=get_number(require_mapping(config.get("observation"), "observation"), "max_sample_age_s", minimum=0.1),
            geometry=tag_geometry,
        )
        observation = require_mapping(config.get("observation"), "observation")
        max_observation_motion_m = get_number(observation, "max_end_link_motion_m", minimum=0.0)
        max_observation_motion_deg = get_number(observation, "max_end_link_rotation_deg", minimum=0.0)
        max_tf_age_ms = get_number(require_mapping(config.get("tf"), "tf"), "max_age_ms", minimum=0.0)
        calibration_max_reprojection_error_px = get_number(
            config, "calibration_max_reprojection_error_px", minimum=0.0,
        ) if args.calibrate_grasp_tcp or args.calibrate_pregrasp else None
        pregrasp_calibration = require_mapping(config.get("pregrasp_calibration"), "pregrasp_calibration")
        pregrasp_min_distance_m = get_number(pregrasp_calibration, "min_distance_m", minimum=0.001)
        pregrasp_max_distance_m = get_number(pregrasp_calibration, "max_distance_m", minimum=pregrasp_min_distance_m)
        pregrasp_max_lateral_error_m = get_number(
            pregrasp_calibration, "max_lateral_error_m", minimum=0.0,
        )
        pregrasp_max_orientation_error_deg = get_number(
            pregrasp_calibration, "max_orientation_error_deg", minimum=0.1,
        )
        min_visual_lock_inliers = 8
        if args.autonomous_approach:
            autonomous = require_mapping(config.get("autonomous_approach"), "autonomous_approach")
            if not autonomous.get("enabled"):
                raise RuntimeError("autonomous_approach.enabled=false：请先显式启用自主预接近配置")
            get_number(autonomous, "max_waypoint_translation_m", minimum=0.001)
            get_number(autonomous, "max_waypoint_orientation_deg", minimum=0.1)
            get_number(autonomous, "speed_mps", minimum=0.001)
            get_number(autonomous, "angular_speed_deg_s", minimum=0.1)
            get_number(autonomous, "waypoint_timeout_s", minimum=0.1)
            get_number(autonomous, "actual_position_tolerance_m", minimum=0.0001)
            get_number(autonomous, "actual_orientation_tolerance_deg", minimum=0.1)
            if not 1 <= int(autonomous.get("servo_sample_count", 1)) <= 10:
                raise RuntimeError("autonomous_approach.servo_sample_count 必须在 1–10")
            min_visual_lock_inliers = int(autonomous.get("min_visual_lock_inliers", 8))
            if not 6 <= min_visual_lock_inliers <= 12:
                raise RuntimeError("autonomous_approach.min_visual_lock_inliers 必须在 6–12")
            get_number(autonomous, "visual_lock_timeout_s", minimum=0.2)
            get_number(autonomous, "visual_lock_hold_timeout_s", minimum=0.5)
            get_number(autonomous, "max_target_delta_m", minimum=0.001)
            get_number(autonomous, "max_target_delta_deg", minimum=0.1)
            if not 2 <= int(autonomous.get("jump_reacquire_confirmations", 3)) <= 10:
                raise RuntimeError("autonomous_approach.jump_reacquire_confirmations 必须在 2–10")
            get_number(autonomous, "jump_cluster_delta_m", minimum=0.001)
            get_number(autonomous, "jump_cluster_delta_deg", minimum=0.1)
            get_number(
                {"orientation_relock_threshold_deg": autonomous.get("orientation_relock_threshold_deg", 2.0)},
                "orientation_relock_threshold_deg", minimum=0.1,
            )
            if not 2 <= int(autonomous.get("orientation_relock_confirmations", 3)) <= 10:
                raise RuntimeError("autonomous_approach.orientation_relock_confirmations 必须在 2–10")
            get_number(
                {"orientation_relock_consistency_deg": autonomous.get(
                    "orientation_relock_consistency_deg",
                    autonomous.get("jump_cluster_delta_deg", 3.0),
                )},
                "orientation_relock_consistency_deg", minimum=0.1,
            )
            if not 1 <= int(autonomous.get("max_iterations", 80)) <= 300:
                raise RuntimeError("autonomous_approach.max_iterations 必须在 1–300")
            get_number(autonomous, "max_total_duration_s", minimum=1.0)
            if not 1 <= int(autonomous.get("stable_samples", 3)) <= 50:
                raise RuntimeError("autonomous_approach.stable_samples 必须在 1–50")
        if args.execute:
            gripper = require_mapping(config.get("gripper"), "gripper")
            open_rad = get_number(
                {"open_rad": gripper.get("open_rad", -0.785)},
                "open_rad", minimum=-0.7853981633974483,
            )
            closed_rad = get_number(
                {"closed_rad": gripper.get("closed_rad", 0.0)},
                "closed_rad", minimum=-0.7853981633974483,
            )
            if open_rad > 0.0 or closed_rad > 0.0:
                raise RuntimeError("右夹爪 open_rad/closed_rad 必须在 [-pi/4, 0] rad")
            open_tolerance = get_number(
                {"open_tolerance_rad": gripper.get("open_tolerance_rad", 0.05)},
                "open_tolerance_rad", minimum=0.001,
            )
            if open_tolerance > 0.20:
                raise RuntimeError("gripper.open_tolerance_rad 必须不大于 0.20 rad")
            get_number(
                {"open_timeout_s": gripper.get("open_timeout_s", 3.0)},
                "open_timeout_s", minimum=0.1,
            )
            close_settle = get_number(
                {"close_settle_s": gripper.get("close_settle_s", 0.8)},
                "close_settle_s", minimum=0.0,
            )
            if close_settle > 5.0:
                raise RuntimeError("gripper.close_settle_s 必须不大于 5 s")
            close_min_motion = get_number(
                {"close_min_motion_rad": gripper.get("close_min_motion_rad", 0.02)},
                "close_min_motion_rad", minimum=0.0,
            )
            if close_min_motion > 0.7853981633974483:
                raise RuntimeError("gripper.close_min_motion_rad 必须不大于 pi/4 rad")
    except Exception as exc:
        print(f"配置无效：{exc}", file=sys.stderr)
        return 2
    if args.execute and right_end_from_tcp is None:
        print("--execute 需要填写实测 right_end_from_grasp_tcp；当前只允许只读定位。", file=sys.stderr)
        return 2
    if args.execute and workpiece_from_pregrasp_end is None:
        print("--execute 需要填写实测 workpiece_from_pregrasp_end；原 OpenArm pregrasp 偏置不用于 G2 动作。", file=sys.stderr)
        return 2
    if args.calibrate_pregrasp and right_end_from_tcp is None:
        print("--calibrate-pregrasp 需要已有 right_end_from_grasp_tcp，以核验同朝向和最终接近距离。", file=sys.stderr)
        return 2
    run_id = utc_stamp()
    log_path = (
        timestamped_log_path(args.log, run_id)
        if args.log is not None
        else PROJECT_ROOT / "logs" / f"right_apriltag_grasp_{run_id}.jsonl"
    ).expanduser()
    save_target_path = None if args.save_target is None else args.save_target.expanduser()
    try:
        import agibot_gdk
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
    except Exception as exc:
        print(f"GDK 不可用：{exc}", file=sys.stderr)
        return 2
    camera = None
    server = None
    motion_gate: MotionPauseGate | None = None
    takeover_worker: SpaceMouseTakeoverWorker | None = None
    try:
        detector = detector_for_41h12()
        color_type = agibot_gdk.CameraType.kHandRightColor
        camera = agibot_gdk.Camera([color_type])
        tf = agibot_gdk.TF()
        robot = agibot_gdk.Robot()
        if args.spacemouse_device:
            motion_gate = MotionPauseGate()
        controller = G2Controller(
            agibot_gdk,
            robot,
            allow_motion=args.allow_motion,
            motion_gate=motion_gate,
        )
        controller.checked_arm_state()
        if args.spacemouse_device:
            takeover_worker = SpaceMouseTakeoverWorker(
                controller,
                motion_gate,
                device=args.spacemouse_device,
                config=TeleopConfig(
                    side="right",
                    speed_m_s=args.spacemouse_speed_mm_s / 1000.0,
                    angular_speed_rad_s=math.radians(args.spacemouse_rotation_deg_s),
                    deadzone=args.spacemouse_deadzone,
                ),
                allow_advance_right=False,
            )
            takeover_worker.start()
        intrinsic = None
        distortion = ()
        right_end_from_camera = transform_from_dict(transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kRightHandRGBDToArmREndLink)))
        live_estimator = None
        live_timestamp_holder = {"timestamp_ns": 0}
        if args.autonomous_approach:
            autonomous = require_mapping(config.get("autonomous_approach"), "autonomous_approach")
            live_estimator = RollingWorkpiecePoseEstimator(
                sample_count=int(autonomous.get("servo_sample_count", 1)),
                max_reprojection_error_px=float(config.get("max_reprojection_error_px", 3.0)),
                max_sample_age_s=min(
                    float(require_mapping(config.get("observation"), "observation").get("max_sample_age_s", 2.5)),
                    float(autonomous.get("visual_lock_timeout_s", 3.0)),
                ),
                geometry=tag_geometry,
            )
        latest = None
        observation_anchor = None
        shared = SharedView() if args.web_port else None
        if shared is not None:
            server = ThreadingHTTPServer((args.web_host, args.web_port), handler_class(shared))
            threading.Thread(target=server.serve_forever, daemon=True).start()
            print(f"实时只读页面：http://{args.web_host}:{args.web_port}", flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(json.dumps({
                "event": "start",
                "motion": "execute" if args.execute else "read_only",
                "mode": (
                    "grasp_tcp_calibration" if args.calibrate_grasp_tcp
                    else "pregrasp_calibration" if args.calibrate_pregrasp else "observation"
                ),
                "config": str(args.config),
                "tag_geometry_profile": str(
                    args.tag_geometry_profile or config.get("tag_geometry_profile") or "tag_geometry_electric_screwdriver.json"
                ),
                "premove_pr1": bool(args.premove_pr1),
                "post_grasp_pr2": bool(args.post_grasp_pr2),
                "right_pose_file": str(args.poses) if (args.premove_pr1 or args.post_grasp_pr2) else None,
            }, ensure_ascii=False) + "\n")

            if args.premove_pr1:
                if pr1_target is None:
                    raise RuntimeError(f"未加载右手 {PR1_POSE_NAME} 目标；停止启动抓取流程")
                if shared is not None:
                    shared.update_status({
                        "state": "premove_pr1",
                        "motion": "planned_joint_premove",
                        "preset": PR1_POSE_NAME,
                    })
                premove_record = move_right_pr1(
                    controller,
                    pr1_target,
                    velocity_rad_s=preset_velocity_rad_s,
                )
                premove_record = {
                    **premove_record,
                    "event": "right_pr1_premove_arrived",
                    "phase": "before_visual_grasp",
                    "utc": datetime.now(timezone.utc).isoformat(),
                }
                print(json.dumps(premove_record, ensure_ascii=False), flush=True)
                log.write(json.dumps(premove_record, ensure_ascii=False) + "\n")
                log.flush()
                # Do not let frames observed while moving to PR1 satisfy the
                # grasp.  The first lock after arrival must be newly sampled.
                estimator.reset()
                if live_estimator is not None:
                    live_estimator.reset()
                observation_anchor = None
                live_timestamp_holder["timestamp_ns"] = 0
                if shared is not None:
                    shared.update_status({
                        "state": "waiting_fresh_visual_lock",
                        "motion": "hold_current_pose",
                        "preset": PR1_POSE_NAME,
                    })

            def acquire_live_lock(*, moving: bool = False):
                if live_estimator is None or intrinsic is None:
                    raise RuntimeError("自主视觉锁定尚未初始化相机内参")
                lock = acquire_fresh_visual_lock(
                    camera=camera,
                    color_type=color_type,
                    detector=detector,
                    estimator=live_estimator,
                    intrinsic=intrinsic,
                    distortion=distortion,
                    tf=tf,
                    controller=controller,
                    right_end_from_camera=right_end_from_camera,
                    right_end_from_tcp=right_end_from_tcp,
                    workpiece_from_pregrasp_end=workpiece_from_pregrasp_end,
                    max_tf_age_ms=max_tf_age_ms,
                    max_end_motion_m=max_observation_motion_m,
                    max_end_rotation_deg=max_observation_motion_deg,
                    timeout_s=float(require_mapping(config.get("autonomous_approach"), "autonomous_approach").get("visual_lock_timeout_s", 3.0)),
                    last_timestamp_holder=live_timestamp_holder,
                    require_stationary=not moving,
                    shared=shared,
                    min_inlier_count=min_visual_lock_inliers,
                )
                if args.face_distance_mm is not None:
                    lock["targets"] = facing_distance_targets(lock["targets"], args.face_distance_mm)
                return lock

            for index in range(args.max_frames or sys.maxsize):
                if takeover_worker is not None:
                    takeover_worker.raise_if_failed()
                    if motion_gate is not None and motion_gate.is_paused():
                        time.sleep(0.01)
                        continue
                image = camera.get_latest_image(color_type, 1000.0)
                color = decode_color_rgb(image, agibot_gdk)
                if intrinsic is None:
                    meta = intrinsic_to_dict(camera.get_camera_intrinsic(color_type), width=int(image.width), height=int(image.height))
                    intrinsic = np.asarray(((meta["fx"], 0.0, meta["ppx"]), (0.0, meta["fy"], meta["ppy"]), (0.0, 0.0, 1.0)), dtype=np.float64)
                    distortion = meta["distortion"]
                current_end_raw, _ = tf.lookup_transform_latest("base_link", "arm_r_end_link", return_timestamp=True)
                current_end_matrix = transform_from_dict(inverse_transform_dict(transform_to_dict(current_end_raw)))
                if observation_anchor is None:
                    observation_anchor = current_end_matrix
                else:
                    shift_m, shift_deg = pose_distance(observation_anchor, current_end_matrix)
                    if shift_m > max_observation_motion_m or shift_deg > max_observation_motion_deg:
                        estimator.reset()
                        observation_anchor = current_end_matrix
                visible = detect_corners(detector, color)
                timestamp_s = float(image.timestamp_ns) / 1_000_000_000.0
                estimator.add(visible, timestamp_s=timestamp_s)
                pose = estimator.estimate(intrinsic, distortion, now_s=timestamp_s)
                record = {
                    "event": "frame", "frame": index + 1, "visible_tag_ids": sorted(visible),
                    "sample_counts": estimator.sample_counts, "pnp_diagnostics": estimator.last_diagnostics,
                }
                if pose is not None:
                    base_from_end, tf_info = camera_to_base_at(tf, int(image.timestamp_ns), max_tf_age_ms)
                    base_from_end_matrix = transform_from_dict(base_from_end)
                    base_from_workpiece = base_from_end_matrix @ right_end_from_camera @ pose.camera_from_workpiece
                    if args.calibrate_grasp_tcp:
                        if tf_info["mode"] != "exact_image_time":
                            raise RuntimeError("标定要求 GDK 精确图像时间 TF；latest fallback 不能用于工具映射")
                        assert calibration_max_reprojection_error_px is not None
                        if pose.mean_reprojection_error_px > calibration_max_reprojection_error_px:
                            raise RuntimeError(
                                f"标定拒绝：PnP 平均重投影误差 {pose.mean_reprojection_error_px:.3f} px "
                                f"超过 calibration_max_reprojection_error_px={calibration_max_reprojection_error_px:.3f} px；"
                                "请调整标签视角并保持机械臂静止后重试"
                            )
                        record.update(calibration_candidate_record(
                            pose, base_from_workpiece, base_from_end_matrix, tf_info, int(image.timestamp_ns),
                        ))
                    elif args.calibrate_pregrasp:
                        rejection_reason = None
                        if tf_info["mode"] != "exact_image_time":
                            rejection_reason = "pregrasp 标定要求 GDK 精确图像时间 TF；latest fallback 不能用于工具映射"
                        elif pose.mean_reprojection_error_px > calibration_max_reprojection_error_px:
                            rejection_reason = (
                                f"PnP 平均重投影误差 {pose.mean_reprojection_error_px:.3f} px 超过 "
                                f"calibration_max_reprojection_error_px={calibration_max_reprojection_error_px:.3f} px"
                            )
                        else:
                            assert right_end_from_tcp is not None
                            grasp_target = g2_grasp_targets(
                                pose.camera_from_workpiece, base_from_end_matrix, right_end_from_camera, right_end_from_tcp,
                            ).base_from_grasp_end
                            _candidate, axial_distance_m, lateral_error_m, orientation_error_deg = constrained_pregrasp_end_candidate(
                                base_from_end_matrix, grasp_target,
                            )
                            if not pregrasp_min_distance_m <= axial_distance_m <= pregrasp_max_distance_m:
                                rejection_reason = (
                                    f"沿接近轴距 grasp {axial_distance_m * 1000:.1f} mm；要求 "
                                    f"{pregrasp_min_distance_m * 1000:.1f}–{pregrasp_max_distance_m * 1000:.1f} mm"
                                )
                            elif lateral_error_m > pregrasp_max_lateral_error_m:
                                rejection_reason = (
                                    f"横向偏差 {lateral_error_m * 1000:.1f} mm；要求不超过 "
                                    f"{pregrasp_max_lateral_error_m * 1000:.1f} mm"
                                )
                            elif orientation_error_deg > pregrasp_max_orientation_error_deg:
                                rejection_reason = (
                                    f"当前末端与 grasp 朝向差 {orientation_error_deg:.2f}°；要求不超过 "
                                    f"{pregrasp_max_orientation_error_deg:.2f}°"
                                )
                        if rejection_reason is not None:
                            record.update({
                                "event": "pregrasp_calibration_rejected",
                                "calibration_required": "workpiece_from_pregrasp_end",
                                "calibration_rejection": rejection_reason,
                            })
                        else:
                            record.update(pregrasp_calibration_candidate_record(
                                base_from_workpiece, base_from_end_matrix, grasp_target,
                                pose, tf_info, int(image.timestamp_ns),
                            ))
                    elif right_end_from_tcp is None:
                        record.update(workpiece_only_record(
                            pose, base_from_workpiece, base_from_end_matrix,
                            right_end_from_camera, tf_info, int(image.timestamp_ns),
                        ))
                    else:
                        targets = g2_grasp_targets(
                            pose.camera_from_workpiece, base_from_end_matrix, right_end_from_camera, right_end_from_tcp,
                            workpiece_from_pregrasp_end=workpiece_from_pregrasp_end,
                        )
                        targets = horizontal_grasp_targets(targets, right_end_from_tcp)
                        if args.face_distance_mm is not None:
                            targets = facing_distance_targets(targets, args.face_distance_mm)
                        if workpiece_from_pregrasp_end is None:
                            record.update(workpiece_only_record(
                                pose, base_from_workpiece, base_from_end_matrix,
                                right_end_from_camera, tf_info, int(image.timestamp_ns),
                            ))
                            record["calibration_required"] = "workpiece_from_pregrasp_end"
                            record["grasp_arm_r_end_link_base"] = matrix_pose_record(targets.base_from_grasp_end)
                        else:
                            latest = (targets, pose, tf_info, int(image.timestamp_ns), base_from_end_matrix)
                            record.update(target_record(*latest))
                if args.terminal_output == "text":
                    print(
                        format_terminal_status(
                            frame=index + 1,
                            visible_tag_ids=record["visible_tag_ids"],
                            sample_counts=record["sample_counts"],
                            pnp_diagnostics=record["pnp_diagnostics"],
                            sample_count=estimator.sample_count,
                        ),
                        end="\r",
                        flush=True,
                    )
                    terminal_status_active = True
                else:
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                publish_shared_view(
                    shared,
                    color,
                    visible,
                    record,
                    motion="none" if not args.execute else "pending_execute",
                )
                if args.calibrate_grasp_tcp and pose is not None:
                    if save_target_path is not None:
                        save_target_path.parent.mkdir(parents=True, exist_ok=True)
                        save_target_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    if args.terminal_output == "text":
                        print(
                            calibration_terminal_message(
                                "grasp_tcp_calibration", record["event"], record.get("calibration_rejection"),
                            ),
                            flush=True,
                        )
                    return 0
                if args.calibrate_pregrasp and pose is not None:
                    if save_target_path is not None:
                        save_target_path.parent.mkdir(parents=True, exist_ok=True)
                        save_target_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    if args.terminal_output == "text":
                        print(
                            calibration_terminal_message(
                                "pregrasp_calibration", record["event"], record.get("calibration_rejection"),
                            ),
                            flush=True,
                        )
                    return 0 if record["event"] == "right_pregrasp_calibration_candidate" else 2
                if latest is not None and pose is not None:
                    if save_target_path is not None and pose is not None:
                        save_target_path.parent.mkdir(parents=True, exist_ok=True)
                        save_target_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                    if args.execute:
                        if tf_info["mode"] != "exact_image_time":
                            raise RuntimeError("执行抓取要求 GDK 精确图像时间 TF；latest fallback 仅可用于只读诊断")
                        before_control_mode = controller.motion_status_summary()
                        if before_control_mode["mode"] not in {0, 1} or before_control_mode["error_code"] != 0:
                            raise RuntimeError(
                                f"执行前机器人处于规划态或有错误，拒绝切换控制模式：{before_control_mode}"
                            )
                        controller.set_control_mode(1)
                        after_control_mode = controller.motion_status_summary()
                        if (
                            after_control_mode["mode"] not in {0, 1}
                            or after_control_mode["control_mode"] != 1
                            or after_control_mode["error_code"] != 0
                        ):
                            raise RuntimeError(
                                f"未进入健康 motion mode=0/1、control_mode=1，拒绝动作：{after_control_mode}"
                            )
                        control_mode_record = {
                            "event": "right_control_mode_ready",
                            "requested_control_mode": 1,
                            "before": before_control_mode,
                            "after": after_control_mode,
                            "motion": "mode_switch_only",
                        }
                        print(json.dumps(control_mode_record, ensure_ascii=False), flush=True)
                        log.write(json.dumps(control_mode_record, ensure_ascii=False) + "\n")
                        log.flush()
                        gripper = require_mapping(config.get("gripper"), "gripper")
                        if args.face_distance_mm is not None:
                            while True:
                                execute_live_autonomous_pregrasp(
                                    controller, latest[0], acquire_live_lock, config, shared=shared,
                                    face_distance_mm=args.face_distance_mm,
                                )
                                print(json.dumps({"event": "right_facing_distance_reached", "distance_mm": args.face_distance_mm}), flush=True)
                                time.sleep(0.10)
                        open_record = move_right_gripper_with_feedback(
                            controller,
                            get_number(
                                {"open_rad": gripper.get("open_rad", -0.785)},
                                "open_rad", minimum=-0.7853981633974483,
                            ),
                            tolerance_rad=get_number(
                                {"open_tolerance_rad": gripper.get("open_tolerance_rad", 0.05)},
                                "open_tolerance_rad", minimum=0.001,
                            ),
                            timeout_s=get_number(
                                {"open_timeout_s": gripper.get("open_timeout_s", 3.0)},
                                "open_timeout_s", minimum=0.1,
                            ),
                        )
                        open_record = {"event": "right_gripper_open_confirmed", **open_record}
                        print(json.dumps(open_record, ensure_ascii=False), flush=True)
                        log.write(json.dumps(open_record, ensure_ascii=False) + "\n")
                        log.flush()
                        execute_targets = latest[0]
                        if args.autonomous_approach:
                            _actual, execute_targets = execute_live_autonomous_pregrasp(
                                controller, execute_targets, acquire_live_lock, config, shared=shared,
                            )
                        close_record = execute_local_grasp(
                            controller,
                            execute_targets,
                            config,
                            autonomous_approach=args.autonomous_approach,
                            max_angular_velocity_rad_s=(
                                None
                                if args.auto_rotation_deg_s is None
                                else math.radians(args.auto_rotation_deg_s)
                            ),
                        )
                        close_record = {"event": "right_gripper_closed", **close_record}
                        saved_grasp_pose = controller.read_end_effector_pose("arm_r_end_link")
                        def emit_return_event(event):
                            event = {"utc": datetime.now(timezone.utc).isoformat(), **event}
                            print(json.dumps(event, ensure_ascii=False), flush=True)
                            log.write(json.dumps(event, ensure_ascii=False) + "\n")
                            log.flush()
                        emit_return_event({
                            "event": "right_grasp_return_pose_saved",
                            "frame": "base_link", "tip": "arm_r_end_link",
                            "position_m": list(saved_grasp_pose.position_m),
                            "orientation_xyzw": list(saved_grasp_pose.orientation_xyzw),
                        })
                        print(json.dumps(close_record, ensure_ascii=False), flush=True)
                        log.write(json.dumps(close_record, ensure_ascii=False) + "\n")
                        log.flush()
                        if args.post_grasp_pr2:
                            if pr2_target is None:
                                raise RuntimeError(f"未加载右手 {PR2_POSE_NAME} 目标；停止前往 PR2")
                            post_grasp_events = execute_post_grasp_transport(
                                controller,
                                pr2_target,
                                lift_speed_mps=(
                                    None
                                    if args.auto_speed_mm_s is None
                                    else args.auto_speed_mm_s / 1000.0
                                ),
                                preset_velocity_rad_s=preset_velocity_rad_s,
                            )
                            for event in post_grasp_events:
                                event["utc"] = datetime.now(timezone.utc).isoformat()
                                print(json.dumps(event, ensure_ascii=False), flush=True)
                                log.write(json.dumps(event, ensure_ascii=False) + "\n")
                                log.flush()
                        print(
                            (
                                ("PR1 预移动 + 二维码自主预接近 + 局部抓取 + 上抬 100 mm + PR2 已完成"
                                 if args.autonomous_approach
                                 else "PR1 预移动 + 局部抓取 + 上抬 100 mm + PR2 已完成")
                                if args.post_grasp_pr2
                                else (("PR1 预移动 + 二维码自主预接近 + 局部抓取序列已完成"
                                       if args.autonomous_approach
                                       else "PR1 预移动 + 局部抓取序列已完成")
                                      if args.premove_pr1
                                      else ("二维码自主预接近 + 局部抓取序列已完成"
                                            if args.autonomous_approach else "局部抓取序列已完成"))
                            )
                            + "；请在现场核验夹爪状态。"
                        )
                        if takeover_worker is not None:
                            wait_for_manual_finish(takeover_worker)
                            takeover_worker.stop_for_final_motion()
                            controller.motion_gate = None
                            return_screwdriver(controller, saved_grasp_pose, config, emit_return_event)
                        return 0
                    if args.max_frames == 0:
                        continue
                elif pose is not None:
                    if save_target_path is not None:
                        save_target_path.parent.mkdir(parents=True, exist_ok=True)
                        save_target_path.write_text(
                            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                        )
                if args.max_frames == 0:
                    continue
        return 0 if latest is not None or right_end_from_tcp is None else 1
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"右手 AprilTag 流程失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if takeover_worker is not None:
            takeover_worker.stop()
        if terminal_status_active:
            print(flush=True)
        if server is not None:
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


if __name__ == "__main__":
    raise SystemExit(main())
