#!/usr/bin/env python3
import argparse
import contextlib
import io
import json
import time
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from extract_sam31_3d_refs import infer_image, mask_stats, select_masks
from sam3.model_builder import build_sam3_predictor


WINDOW_NAME = "SAM3.1 RealSense Hinge 3D"


def configure_auto_exposure(device, exposure_limit: int) -> None:
    sensor = device.first_depth_sensor()
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 1)
    if (
        sensor.supports(rs.option.auto_exposure_limit_toggle)
        and sensor.supports(rs.option.auto_exposure_limit)
    ):
        sensor.set_option(rs.option.auto_exposure_limit_toggle, 1)
        limits = sensor.get_option_range(rs.option.auto_exposure_limit)
        value = min(max(float(exposure_limit), limits.min), limits.max)
        sensor.set_option(rs.option.auto_exposure_limit, value)


def find_device(context, requested_serial: str):
    devices = list(context.query_devices())
    if requested_serial and requested_serial.lower() != "auto":
        for device in devices:
            serial = device.get_info(rs.camera_info.serial_number)
            if serial == requested_serial:
                return device
        available = ", ".join(
            device.get_info(rs.camera_info.serial_number) for device in devices
        ) or "none"
        raise RuntimeError(
            f"D435 serial {requested_serial} not found; available serials: {available}"
        )

    for device in devices:
        name = device.get_info(rs.camera_info.name)
        if "D435" in name:
            return device
    raise RuntimeError("No RealSense D435 device found")


def put_text(img, text, xy, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_detections(bgr: np.ndarray, detections: list[dict], status: dict) -> np.ndarray:
    vis = bgr.copy()
    colors = {
        "silver_metal_hinge": (80, 255, 80),
        "slotted_hinge_shaft": (0, 165, 255),
    }
    for det in detections:
        color = colors.get(det["label"], (255, 255, 255))
        x0, y0, x1, y1 = det["bbox_xyxy"]
        u, v = det["center_px"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color, 2)
        cv2.circle(vis, (int(u), int(v)), 5, color, -1)
        x, y, z = det["center_xyz_m"]
        label = det["label"].replace("_", " ")
        text = f"{label} #{det['rank']} z={z:.3f} xyz=({x:.3f},{y:.3f},{z:.3f})"
        put_text(vis, text, (x0, max(18, y0 - 6)), color, 0.43)
        # This is the principal image direction of the *selected* mask.  In
        # slotted_hinge_shaft mode the mask is the lower slotted cylinder, not
        # the complete hinge plate.
        angle_deg = det.get("angle_deg")
        if angle_deg is not None:
            angle_rad = float(np.deg2rad(angle_deg))
            half_length = 0.6 * float(max(x1 - x0, y1 - y0))
            dx = int(round(np.cos(angle_rad) * half_length))
            dy = int(round(np.sin(angle_rad) * half_length))
            center = (int(round(u)), int(round(v)))
            cv2.line(
                vis,
                (center[0] - dx, center[1] - dy),
                (center[0] + dx, center[1] + dy),
                (0, 165, 255),
                3,
                cv2.LINE_AA,
            )
            put_text(
                vis,
                f"{'shaft' if det['label'] == 'slotted_hinge_shaft' else 'hinge'} axis={float(angle_deg):+.1f} deg",
                (x0, min(vis.shape[0] - 12, y1 + 18)),
                (0, 165, 255),
                0.43,
            )

    age = time.monotonic() - status.get("last_time", time.monotonic())
    detection_name = status.get("detection_name", "hinge")
    header = f"{detection_name}={status.get('hinge_count', 0)} infer={status.get('latency_ms', 0):.0f}ms age={age:.1f}s"
    if status.get("busy"):
        header += " RUNNING"
    if status.get("error"):
        header = "ERROR: " + str(status["error"])[:120]
    put_text(vis, header, (10, 24), (255, 255, 255), 0.58)
    put_text(vis, "q/esc quit, s save latest", (10, vis.shape[0] - 14), (255, 255, 255), 0.52)
    return vis


def select_target_masks(label: str, stats: list[dict], args) -> list[int]:
    """Apply target-specific selection without changing the normal hinge flow."""
    if label != "slotted_hinge_shaft":
        return select_masks(label, stats, args)

    valid = [
        index
        for index, stat in enumerate(stats)
        if stat.get("area_px", 0) >= args.shaft_min_area
        and stat.get("center_xyz_m") is not None
    ]
    if not valid:
        return []

    # A part-level SAM prompt can occasionally return a small fragmented mask
    # as well as the actual slotted shaft.  The shaft is chosen by its usable
    # area, not by the whole-hinge selection rule used in the normal mode.
    return [max(valid, key=lambda index: stats[index]["area_px"])]


def detections_for_label(
    predictor,
    bgr: np.ndarray,
    depth_u16: np.ndarray,
    label: str,
    prompt: str,
    threshold: float,
    depth_scale: float,
    color_intr: dict,
    args,
    include_masks: bool = False,
) -> tuple[list[dict], int]:
    masks = infer_image(predictor, bgr, prompt, threshold)
    if masks is None:
        masks = np.zeros((0, bgr.shape[0], bgr.shape[1]), dtype=bool)
    stats = [mask_stats(m, bgr, depth_u16, depth_scale, color_intr) for m in masks]
    selected = select_target_masks(label, stats, args)
    detections = []
    for rank, mask_idx in enumerate(selected):
        s = stats[mask_idx]
        axis_measurement = estimate_axis_3d(
            masks[mask_idx], depth_u16, depth_scale, color_intr
        )
        center_xyz = s["center_xyz_m"]
        if label == "slotted_hinge_shaft" and axis_measurement is not None:
            # The fitted line centroid is the appropriate 3-D shaft point;
            # the raw mask centroid is retained separately for diagnostics.
            center_xyz = axis_measurement["point_camera_m"]
        detection = {
            "label": label,
            "prompt": prompt,
            "mask_index": int(mask_idx),
            "rank": int(rank),
            "area_px": int(s["area_px"]),
            "bbox_xyxy": s["bbox_xyxy"],
            "center_px": [float(s["center_px"][0]), float(s["center_px"][1])],
            "center_xyz_m": [float(v) for v in center_xyz],
            "mask_centroid_xyz_m": [float(v) for v in s["center_xyz_m"]],
            "depth_median_m": None if s["depth_median_m"] is None else float(s["depth_median_m"]),
            "mean_gray": float(s["mean_gray"]),
            "angle_deg": s["angle_deg"],
            "axis_3d_camera_unit": None if axis_measurement is None else axis_measurement["direction_camera_unit"],
            "axis_point_camera_m": None if axis_measurement is None else axis_measurement["point_camera_m"],
            "axis_linearity": None if axis_measurement is None else axis_measurement["linearity"],
            "feature_type": "slotted_cylindrical_shaft" if label == "slotted_hinge_shaft" else "whole_hinge",
        }
        # Optional, in-memory transport for a consumer that needs the exact
        # selected mask.  Default callers retain the existing JSON schema.
        if include_masks:
            detection["_selected_mask"] = masks[mask_idx].astype(bool, copy=False)
        detections.append(detection)
    return detections, len(stats)


def estimate_axis_3d(
    mask: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    color_intr: dict,
) -> dict | None:
    """Robust 3-D line fit of the selected target mask.

    In the ordinary hinge mode this remains only a diagnostic PCA direction.
    In slotted_hinge_shaft mode the input mask is specifically the lower
    notched cylinder, so the fitted line is the measured mechanical axis.
    """
    rows, cols = np.where(mask)
    if len(cols) < 30:
        return None
    depths_m = depth_u16[rows, cols].astype(np.float64) * float(depth_scale)
    valid = np.isfinite(depths_m) & (depths_m > 0.02) & (depths_m < 3.0)
    if int(valid.sum()) < 30:
        return None
    rows = rows[valid].astype(np.float64)
    cols = cols[valid].astype(np.float64)
    depths_m = depths_m[valid]
    low_depth, high_depth = np.percentile(depths_m, [2.0, 98.0])
    # Drop edge/background depths while retaining the full shaft length.
    inlier = (depths_m >= low_depth) & (depths_m <= high_depth)
    if int(inlier.sum()) < 30:
        return None
    rows, cols, depths_m = rows[inlier], cols[inlier], depths_m[inlier]
    points = np.column_stack(
        (
            (cols - float(color_intr["ppx"])) / float(color_intr["fx"]) * depths_m,
            (rows - float(color_intr["ppy"])) / float(color_intr["fy"]) * depths_m,
            depths_m,
        )
    )
    if len(points) > 8000:
        sample = np.linspace(0, len(points) - 1, 8000, dtype=np.int64)
        points = points[sample]

    # Two passes of line-distance rejection prevent a few plate/background
    # pixels at the mask boundary from steering the shaft axis.
    for _ in range(2):
        center = np.mean(points, axis=0)
        centered = points - center
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        values, vectors = np.linalg.eigh(covariance)
        direction = vectors[:, int(np.argmax(values))]
        distances = np.linalg.norm(np.cross(centered, direction), axis=1)
        median_distance = float(np.median(distances))
        limit = max(0.003, 2.5 * median_distance)
        keep = distances <= limit
        if int(keep.sum()) < 30 or int(keep.sum()) == len(points):
            break
        points = points[keep]

    center = np.mean(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-9:
        return None
    direction = direction / norm
    # A line has no intrinsic sign. Pick one only for stable JSON/display.
    dominant = int(np.argmax(np.abs(direction)))
    if direction[dominant] < 0.0:
        direction = -direction
    sorted_values = np.sort(values)
    linearity = float(sorted_values[-1] / max(sorted_values[-2], 1e-12))
    return {
        "direction_camera_unit": [float(value) for value in direction],
        "point_camera_m": [float(value) for value in center],
        "linearity": linearity,
    }


def shaft_search_bounds(image_shape: tuple[int, int], args) -> tuple[int, int, int, int]:
    """Convert the normalized right-D405 shaft search window to pixels."""
    height, width = image_shape[:2]
    x0_ratio, y0_ratio, x1_ratio, y1_ratio = args.shaft_search_roi
    x0 = max(0, min(width - 1, int(round(x0_ratio * width))))
    y0 = max(0, min(height - 1, int(round(y0_ratio * height))))
    x1 = max(x0 + 1, min(width, int(round(x1_ratio * width))))
    y1 = max(y0 + 1, min(height, int(round(y1_ratio * height))))
    return x0, y0, x1, y1


def draw_shaft_search_roi(image: np.ndarray, args) -> None:
    x0, y0, x1, y1 = shaft_search_bounds(image.shape, args)
    cv2.rectangle(image, (x0, y0), (x1, y1), (255, 0, 255), 1, cv2.LINE_AA)
    put_text(image, "shaft search", (x0, max(18, y0 - 5)), (255, 0, 255), 0.42)


def detect_slotted_hinge_shaft(
    bgr: np.ndarray,
    depth_u16: np.ndarray,
    depth_scale: float,
    color_intr: dict,
    args,
) -> tuple[list[dict], int]:
    """Detect the thin notched shaft directly from its local image geometry.

    This deliberately does not segment, localize, or use the large hinge plate.
    The right D405 view is fixed for this operation: the notched shaft appears
    as a narrow, near-vertical metallic feature in the configured search ROI.
    We locate its strongest long vertical edge, gather only nearby bright shaft
    pixels, then fit their depth points to produce the mechanical axis.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    x0, y0, x1, y1 = shaft_search_bounds(gray.shape, args)
    roi = gray[y0:y1, x0:x1]
    blurred = cv2.GaussianBlur(roi, (3, 3), 0)
    edges = cv2.Canny(blurred, args.shaft_canny_low, args.shaft_canny_high)
    lines = cv2.HoughLinesP(
        edges,
        rho=1.0,
        theta=np.pi / 180.0,
        threshold=args.shaft_hough_threshold,
        minLineLength=args.shaft_min_line_px,
        maxLineGap=args.shaft_max_line_gap_px,
    )

    candidates = []
    min_y = y0 + args.shaft_min_lower_fraction * (y1 - y0)
    if lines is not None:
        for line in lines[:, 0]:
            local_xa, local_ya, local_xb, local_yb = (int(value) for value in line)
            xa, ya = local_xa + x0, local_ya + y0
            xb, yb = local_xb + x0, local_yb + y0
            dx, dy = xb - xa, yb - ya
            length = float(np.hypot(dx, dy))
            if length < args.shaft_min_line_px or abs(dy) < 1:
                continue
            tilt_from_vertical = float(np.degrees(np.arctan2(abs(dx), abs(dy))))
            if tilt_from_vertical > args.shaft_max_tilt_deg:
                continue
            mid_y = 0.5 * (ya + yb)
            if mid_y < min_y:
                continue

            line_pixels = np.zeros_like(gray, dtype=np.uint8)
            cv2.line(line_pixels, (xa, ya), (xb, yb), 255, 3, cv2.LINE_AA)
            brightness = float(np.percentile(gray[line_pixels > 0], 75))
            if brightness < args.shaft_min_line_brightness:
                continue
            lower_score = (mid_y - min_y) / max(float(y1 - min_y), 1.0)
            score = length * (0.5 + brightness / 255.0) * (1.0 + 0.35 * lower_score)
            candidates.append((score, xa, ya, xb, yb, length, brightness))

    if not candidates:
        return [], 0

    _, xa, ya, xb, yb, line_length, line_brightness = max(candidates, key=lambda item: item[0])
    pad = args.shaft_mask_half_width_px
    support = np.zeros_like(gray, dtype=np.uint8)
    cv2.line(support, (xa, ya), (xb, yb), 255, 2 * pad + 1, cv2.LINE_AA)
    # Do not let the mask grow into the plate above or unrelated objects below.
    ymin, ymax = max(y0, min(ya, yb) - 3), min(y1 - 1, max(ya, yb) + 3)
    support[:ymin, :] = 0
    support[ymax + 1 :, :] = 0
    shaft_mask = (support > 0) & (gray >= args.shaft_pixel_brightness)
    if int(shaft_mask.sum()) < args.shaft_min_area:
        # The reflective shaft often has a dark edge.  The narrow support is
        # still a safer target than broadening to the hinge plate.
        shaft_mask = support > 0

    stat = mask_stats(shaft_mask, bgr, depth_u16, depth_scale, color_intr)
    if stat.get("center_xyz_m") is None or stat.get("area_px", 0) < args.shaft_min_area:
        return [], len(candidates)
    measurement = estimate_axis_3d(shaft_mask, depth_u16, depth_scale, color_intr)
    if measurement is None:
        return [], len(candidates)

    center_px = stat["center_px"]
    direction = measurement["direction_camera_unit"]
    angle_deg = float(np.degrees(np.arctan2(yb - ya, xb - xa)))
    detection = {
        "label": "slotted_hinge_shaft",
        "prompt": "direct geometric slotted shaft detector",
        "mask_index": 0,
        "rank": 0,
        "area_px": int(stat["area_px"]),
        "bbox_xyxy": stat["bbox_xyxy"],
        "center_px": [float(center_px[0]), float(center_px[1])],
        "center_xyz_m": measurement["point_camera_m"],
        "mask_centroid_xyz_m": [float(value) for value in stat["center_xyz_m"]],
        "depth_median_m": None if stat["depth_median_m"] is None else float(stat["depth_median_m"]),
        "mean_gray": float(stat["mean_gray"]),
        "angle_deg": angle_deg,
        "axis_3d_camera_unit": direction,
        "axis_point_camera_m": measurement["point_camera_m"],
        "axis_linearity": measurement["linearity"],
        "feature_type": "slotted_cylindrical_shaft",
        "detector": "direct_vertical_metal_geometry",
        "line_length_px": line_length,
        "line_brightness_p75": line_brightness,
        "search_roi_xyxy": [x0, y0, x1, y1],
        "candidate_count": len(candidates),
    }
    return [detection], len(candidates)


def inference_worker(shared, lock, event, stop, predictor, color_intr, depth_scale, args):
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    while not stop.is_set():
        if not event.wait(0.1):
            continue
        event.clear()
        with lock:
            color = shared.get("pending_color")
            depth = shared.get("pending_depth")
            shared["pending_color"] = None
            shared["pending_depth"] = None
        if color is None or depth is None:
            with lock:
                shared["busy"] = False
            continue

        start = time.monotonic()
        try:
            if args.target_mode == "slotted_hinge_shaft":
                target, raw_count = detect_slotted_hinge_shaft(
                    color, depth, depth_scale, color_intr, args
                )
                target_label = "slotted_hinge_shaft"
                target_prompt = "direct geometric slotted shaft detector"
            else:
                target_label = "silver_metal_hinge"
                target_prompt = args.hinge_prompt
                target, raw_count = detections_for_label(
                    predictor,
                    color,
                    depth,
                    target_label,
                    target_prompt,
                    args.threshold,
                    depth_scale,
                    color_intr,
                    args,
                )
            detections = target
            overlay = draw_detections(
                color,
                detections,
                {
                    "hinge_count": len(target),
                    "detection_name": "slotted shaft" if args.target_mode == "slotted_hinge_shaft" else "hinge",
                    "latency_ms": (time.monotonic() - start) * 1000,
                    "last_time": time.monotonic(),
                },
            )
            if args.target_mode == "slotted_hinge_shaft":
                draw_shaft_search_roi(overlay, args)
            payload = {
                "timestamp_unix": time.time(),
                "camera_name": args.camera_name,
                "camera_serial": args.camera_serial,
                "source_frame": args.source_frame,
                "target_mode": args.target_mode,
                "hinge_prompt": args.hinge_prompt,
                "shaft_prompt": args.shaft_prompt if args.target_mode == "slotted_hinge_shaft" else None,
                "threshold": args.threshold,
                "raw_mask_counts": {
                    target_label: raw_count,
                },
                "detections": detections,
            }
            (out_dir / args.json_name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            cv2.imwrite(str(out_dir / "live_3d_latest_raw.jpg"), color)
            cv2.imwrite(str(out_dir / "live_3d_latest_overlay.jpg"), overlay)
            with lock:
                shared["overlay"] = overlay
                shared["detections"] = detections
                shared["hinge_count"] = len(target)
                shared["detection_name"] = "slotted shaft" if args.target_mode == "slotted_hinge_shaft" else "hinge"
                shared["latency_ms"] = (time.monotonic() - start) * 1000
                shared["last_time"] = time.monotonic()
                shared["error"] = None
                shared["completed"] = shared.get("completed", 0) + 1
        except Exception as exc:
            with lock:
                shared["error"] = repr(exc)
        finally:
            with lock:
                shared["busy"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sam3.1_multiplex.pt")
    parser.add_argument(
        "--target-mode",
        choices=["hinge", "slotted_hinge_shaft"],
        default="hinge",
        help="hinge: normal SAM whole-hinge detection; slotted_hinge_shaft: direct geometric detection of the notched shaft only.",
    )
    parser.add_argument("--hinge-prompt", default="silver metal hinge")
    parser.add_argument(
        "--shaft-prompt",
        default="slotted cylindrical hinge shaft with a notch",
        help="Reserved prompt description for logs in --target-mode slotted_hinge_shaft.",
    )
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--infer-interval", type=float, default=0.75)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out-dir", default="outputs/live_3d")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--max-inferences", type=int, default=0)
    parser.add_argument("--hinge-min-area", type=int, default=1500)
    parser.add_argument("--shaft-min-area", type=int, default=80)
    parser.add_argument(
        "--shaft-search-roi",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        default=(0.35, 0.08, 0.63, 0.40),
        help="Normalized right-D405 ROI for the slotted shaft search.",
    )
    parser.add_argument("--shaft-canny-low", type=int, default=50)
    parser.add_argument("--shaft-canny-high", type=int, default=130)
    parser.add_argument("--shaft-hough-threshold", type=int, default=9)
    parser.add_argument("--shaft-min-line-px", type=int, default=18)
    parser.add_argument("--shaft-max-line-gap-px", type=int, default=4)
    parser.add_argument("--shaft-max-tilt-deg", type=float, default=22.0)
    parser.add_argument("--shaft-min-lower-fraction", type=float, default=0.28)
    parser.add_argument("--shaft-min-line-brightness", type=float, default=85.0)
    parser.add_argument("--shaft-pixel-brightness", type=int, default=65)
    parser.add_argument("--shaft-mask-half-width-px", type=int, default=12)
    parser.add_argument("--serial", default="346522070661")
    parser.add_argument("--auto-exposure-limit", type=int, default=4000)
    parser.add_argument("--camera-name", default="Chest D435")
    parser.add_argument("--window-name", default=WINDOW_NAME)
    parser.add_argument("--json-name", default="live_3d_latest.json")
    parser.add_argument("--source-frame", default="")
    args = parser.parse_args()

    if not (0.0 <= args.shaft_search_roi[0] < args.shaft_search_roi[2] <= 1.0):
        parser.error("--shaft-search-roi requires 0 <= X0 < X1 <= 1")
    if not (0.0 <= args.shaft_search_roi[1] < args.shaft_search_roi[3] <= 1.0):
        parser.error("--shaft-search-roi requires 0 <= Y0 < Y1 <= 1")

    predictor = None
    if args.target_mode == "hinge":
        checkpoint = Path(args.checkpoint).expanduser().resolve()
        print(f"loading SAM3.1 checkpoint={checkpoint}")
        muted = io.StringIO()
        with contextlib.redirect_stdout(muted):
            predictor = build_sam3_predictor(
                version="sam3.1",
                checkpoint_path=str(checkpoint),
                use_fa3=False,
                use_rope_real=False,
                async_loading_frames=False,
                default_output_prob_thresh=args.threshold,
            )
        print("model loaded")
    else:
        print("slotted-shaft mode: direct geometry detector; SAM model is not used")

    context = rs.context()
    device = find_device(context, args.serial)
    actual_serial = device.get_info(rs.camera_info.serial_number)
    args.camera_serial = actual_serial
    configure_auto_exposure(device, args.auto_exposure_limit)

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(actual_serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    try:
        configure_auto_exposure(profile.get_device(), args.auto_exposure_limit)
    except RuntimeError as exc:
        print(f"warning: cannot set AE on {actual_serial}: {exc}")
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    color_intr_dict = {
        "fx": color_intr.fx,
        "fy": color_intr.fy,
        "ppx": color_intr.ppx,
        "ppy": color_intr.ppy,
    }

    lock = Lock()
    event = Event()
    stop = Event()
    shared = {
        "pending_color": None,
        "pending_depth": None,
        "overlay": None,
        "busy": False,
        "last_time": time.monotonic(),
        "latency_ms": 0.0,
        "hinge_count": 0,
        "detection_name": "slotted shaft" if args.target_mode == "slotted_hinge_shaft" else "hinge",
        "completed": 0,
        "error": None,
    }
    worker = Thread(
        target=inference_worker,
        args=(shared, lock, event, stop, predictor, color_intr_dict, depth_scale, args),
        daemon=True,
    )
    worker.start()
    print(
        f"opened {args.camera_name}: S/N {actual_serial} "
        f"{args.width}x{args.height}@{args.fps}, "
        f"AE limit={args.auto_exposure_limit}",
        flush=True,
    )
    next_infer = 0.0

    try:
        for _ in range(args.warmup):
            pipeline.wait_for_frames()
        if not args.no_window:
            cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

        while not stop.is_set():
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()

            now = time.monotonic()
            with lock:
                busy = shared["busy"]
                completed = shared["completed"]
            if now >= next_infer and not busy:
                with lock:
                    shared["pending_color"] = color.copy()
                    shared["pending_depth"] = depth.copy()
                    shared["busy"] = True
                event.set()
                next_infer = now + max(0.05, args.infer_interval)

            with lock:
                status = dict(shared)
                overlay = shared["overlay"].copy() if shared["overlay"] is not None else color.copy()

            if not args.no_window:
                if shared.get("overlay") is None:
                    overlay = draw_detections(color, [], status)
                cv2.imshow(args.window_name, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    out_dir = Path(args.out_dir).expanduser().resolve()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(out_dir / f"saved_3d_overlay_{stamp}.jpg"), overlay)

            if args.max_inferences > 0 and completed >= args.max_inferences:
                break
    finally:
        stop.set()
        event.set()
        worker.join(timeout=5.0)
        pipeline.stop()
        if not args.no_window:
            cv2.destroyAllWindows()
        if predictor is not None:
            del predictor
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
