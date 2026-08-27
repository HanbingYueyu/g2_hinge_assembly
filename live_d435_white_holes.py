#!/usr/bin/env python3
"""Locate bright circular hole rims from the chest D435 RGB-D stream.

The output position is expressed in ``chest_camera_optical_frame``: +X right,
+Y down and +Z forward.  Depth is the median on a ring around the detected
white rim, which is more reliable than sampling the dark/empty hole centre.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs


WINDOW_NAME = "D435 White Hole 3D"


def configure_auto_exposure(device, exposure_limit: int) -> None:
    sensor = device.first_depth_sensor()
    if sensor.supports(rs.option.enable_auto_exposure):
        sensor.set_option(rs.option.enable_auto_exposure, 1)
    if (sensor.supports(rs.option.auto_exposure_limit_toggle)
            and sensor.supports(rs.option.auto_exposure_limit)):
        sensor.set_option(rs.option.auto_exposure_limit_toggle, 1)
        limits = sensor.get_option_range(rs.option.auto_exposure_limit)
        sensor.set_option(
            rs.option.auto_exposure_limit,
            min(max(float(exposure_limit), limits.min), limits.max),
        )


def find_d435(context: rs.context, requested_serial: str):
    devices = list(context.query_devices())
    for device in devices:
        name = device.get_info(rs.camera_info.name)
        serial = device.get_info(rs.camera_info.serial_number)
        if requested_serial and requested_serial.lower() != "auto":
            if serial == requested_serial:
                if "D435" not in name:
                    raise RuntimeError(f"serial {serial} is {name}, not D435")
                return device
        elif "D435" in name:
            return device
    available = ", ".join(
        f"{device.get_info(rs.camera_info.name)}({device.get_info(rs.camera_info.serial_number)})"
        for device in devices
    ) or "none"
    raise RuntimeError(f"Chest D435 {requested_serial} not found; available: {available}")


def ring_depth_xyz(
    depth_u16: np.ndarray,
    center: tuple[float, float],
    radius: float,
    intrinsics: dict,
    depth_scale: float,
) -> tuple[list[float] | None, float | None, int]:
    """Use valid depth on the white rim, avoiding an empty dark hole centre."""
    height, width = depth_u16.shape[:2]
    u, v = center
    r_outer = max(radius * 1.15, 3.0)
    r_inner = max(radius * 0.45, 1.0)
    x0 = max(0, int(np.floor(u - r_outer)))
    x1 = min(width, int(np.ceil(u + r_outer)) + 1)
    y0 = max(0, int(np.floor(v - r_outer)))
    y1 = min(height, int(np.ceil(v + r_outer)) + 1)
    if x1 <= x0 or y1 <= y0:
        return None, None, 0
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance2 = (xx - u) ** 2 + (yy - v) ** 2
    ring = (distance2 >= r_inner ** 2) & (distance2 <= r_outer ** 2)
    values = depth_u16[y0:y1, x0:x1][ring]
    values = values[values > 0]
    if values.size < 12:
        return None, None, int(values.size)
    z = float(np.median(values)) * depth_scale
    x = (u - intrinsics["ppx"]) * z / intrinsics["fx"]
    y = (v - intrinsics["ppy"]) * z / intrinsics["fy"]
    return [float(x), float(y), float(z)], z, int(values.size)


def circle_appearance(
    gray: np.ndarray, saturation: np.ndarray, center: tuple[float, float], radius: float,
    min_gray: int, max_saturation: int,
) -> tuple[float, float, float]:
    """Return dark-centre mean, rim mean and bright-rim fraction."""
    height, width = gray.shape[:2]
    u, v = center
    outer = max(radius * 1.15, 3.0)
    x0 = max(0, int(np.floor(u - outer)))
    x1 = min(width, int(np.ceil(u + outer)) + 1)
    y0 = max(0, int(np.floor(v - outer)))
    y1 = min(height, int(np.ceil(v + outer)) + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    distance2 = (xx - u) ** 2 + (yy - v) ** 2
    inner = distance2 <= (max(radius * 0.42, 1.0) ** 2)
    rim = ((distance2 >= (max(radius * 0.55, 1.0) ** 2))
           & (distance2 <= outer ** 2))
    inner_values = gray[y0:y1, x0:x1][inner]
    rim_values = gray[y0:y1, x0:x1][rim]
    rim_saturation = saturation[y0:y1, x0:x1][rim]
    if inner_values.size == 0 or rim_values.size == 0:
        return 255.0, 0.0, 0.0
    bright_rim = (rim_values >= min_gray) & (rim_saturation <= max_saturation)
    return (
        float(np.mean(inner_values)),
        float(np.mean(rim_values)),
        float(np.mean(bright_rim)),
    )


def detect_white_holes(
    bgr: np.ndarray,
    depth_u16: np.ndarray | None,
    intrinsics: dict | None,
    depth_scale: float | None,
    args: argparse.Namespace,
) -> tuple[list[dict], np.ndarray]:
    """Find bright, low-saturation, approximately circular rim contours."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([0, 0, args.min_gray], dtype=np.uint8),
        np.array([180, args.max_saturation, 255], dtype=np.uint8),
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # A white rim has a dark centre.  Hough circles find that ring even when
    # the white pixels are not a single connected contour; the appearance
    # checks reject ordinary bright table/background patches.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        cv2.GaussianBlur(gray, (7, 7), 1.5),
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=args.min_circle_distance,
        param1=args.hough_edge_threshold,
        param2=args.hough_accumulator_threshold,
        minRadius=int(args.min_radius),
        maxRadius=int(args.max_radius),
    )

    detections = []
    saturation = hsv[:, :, 1]
    for u, v, radius in ([] if circles is None else circles[0]):
        # The target fixture is in the lower work area.  Excluding the upper
        # part of the chest view removes circular robot parts behind the work
        # surface (the false detection seen at y≈45 px).
        if v < args.min_center_y_fraction * bgr.shape[0]:
            continue
        area = float(np.pi * radius * radius)
        if area < args.min_area or area > args.max_area:
            continue
        inner_gray, rim_gray, bright_rim_fraction = circle_appearance(
            gray, saturation, (float(u), float(v)), float(radius),
            args.min_gray, args.max_saturation,
        )
        rim_contrast = rim_gray - inner_gray
        if (inner_gray > args.max_inner_gray
                or rim_contrast < args.min_rim_contrast
                or bright_rim_fraction < args.min_rim_white_fraction):
            continue
        x0 = max(0, int(round(u - radius)))
        y0 = max(0, int(round(v - radius)))
        x1 = min(bgr.shape[1], int(round(u + radius)))
        y1 = min(bgr.shape[0], int(round(v + radius)))

        xyz = depth_median = None
        valid_depth = 0
        if depth_u16 is not None and intrinsics is not None and depth_scale is not None:
            xyz, depth_median, valid_depth = ring_depth_xyz(
                depth_u16, (u, v), radius, intrinsics, depth_scale
            )
        if depth_u16 is not None and xyz is None:
            continue
        detections.append({
            "label": "white_hole",
            "rank": 0,
            "center_px": [float(u), float(v)],
            "center_xyz_m": xyz,
            "depth_median_m": depth_median,
            "valid_depth_px": valid_depth,
            "radius_px": float(radius),
            "area_px": area,
            "inner_gray": inner_gray,
            "rim_gray": rim_gray,
            "rim_contrast": rim_contrast,
            "bright_rim_fraction": bright_rim_fraction,
            "confidence": rim_contrast + 100.0 * bright_rim_fraction,
            "bbox_xyxy": [x0, y0, x1, y1],
        })

    detections.sort(key=lambda item: item["confidence"], reverse=True)
    for rank, detection in enumerate(detections):
        detection["rank"] = rank
    return detections, mask


def draw_overlay(bgr: np.ndarray, detections: list[dict], mask: np.ndarray, status: str) -> np.ndarray:
    view = bgr.copy()
    for detection in detections:
        u, v = detection["center_px"]
        radius = int(round(detection["radius_px"]))
        cv2.circle(view, (int(round(u)), int(round(v))), radius, (0, 255, 255), 2)
        cv2.drawMarker(
            view, (int(round(u)), int(round(v))), (0, 0, 255),
            cv2.MARKER_CROSS, 12, 2,
        )
        xyz = detection["center_xyz_m"]
        text = f"#{detection['rank']} px=({u:.0f},{v:.0f})"
        if xyz is not None:
            text += f" xyz=({xyz[0]:+.3f},{xyz[1]:+.3f},{xyz[2]:.3f})m"
        x0, y0, _, _ = detection["bbox_xyxy"]
        cv2.putText(view, text, (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(view, text, (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view, f"white holes={len(detections)} {status}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, f"white holes={len(detections)} {status}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(view, "q/esc quit, s save", (10, view.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(view, "q/esc quit, s save", (10, view.shape[0] - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    return view


def save_result(out_dir: Path, payload: dict, raw: np.ndarray, overlay: np.ndarray) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "chest_white_holes_latest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)
    cv2.imwrite(str(out_dir / "chest_white_holes_latest_raw.jpg"), raw)
    cv2.imwrite(str(out_dir / "chest_white_holes_latest_overlay.jpg"), overlay)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chest D435 white circular-hole 3D locator")
    parser.add_argument("--serial", default="346522070661")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--auto-exposure-limit", type=int, default=4000)
    parser.add_argument("--min-gray", type=int, default=150)
    parser.add_argument("--max-saturation", type=int, default=105)
    parser.add_argument("--min-area", type=float, default=80.0)
    parser.add_argument("--max-area", type=float, default=8000.0)
    # The validated chest-D435 target is a roughly 20 px radius ring at the
    # normal working distance.  Starting at 10 px rejects tiny edge noise.
    parser.add_argument("--min-radius", type=float, default=10.0)
    parser.add_argument("--max-radius", type=float, default=45.0)
    parser.add_argument("--min-center-y-fraction", type=float, default=0.40)
    parser.add_argument("--min-circle-distance", type=float, default=35.0)
    parser.add_argument("--hough-edge-threshold", type=float, default=80.0)
    parser.add_argument("--hough-accumulator-threshold", type=float, default=24.0)
    parser.add_argument("--max-inner-gray", type=float, default=110.0)
    parser.add_argument("--min-rim-contrast", type=float, default=25.0)
    parser.add_argument("--min-rim-white-fraction", type=float, default=0.08)
    parser.add_argument("--out-dir", default="outputs/white_holes")
    parser.add_argument("--no-window", action="store_true")
    # Offline mode is used for repeatable image checks; live operation ignores it.
    parser.add_argument("--image", default=None)
    parser.add_argument("--depth", default=None)
    parser.add_argument("--fx", type=float, default=615.0)
    parser.add_argument("--fy", type=float, default=615.0)
    parser.add_argument("--ppx", type=float, default=320.0)
    parser.add_argument("--ppy", type=float, default=240.0)
    parser.add_argument("--depth-scale", type=float, default=0.001)
    return parser.parse_args()


def run_offline(args: argparse.Namespace) -> int:
    image = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"cannot read --image {args.image}")
    depth = None
    if args.depth:
        depth = cv2.imread(args.depth, cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f"cannot read --depth {args.depth}")
    intrinsics = {"fx": args.fx, "fy": args.fy, "ppx": args.ppx, "ppy": args.ppy}
    detections, mask = detect_white_holes(image, depth, intrinsics, args.depth_scale, args)
    print(json.dumps(detections, indent=2), flush=True)
    overlay = draw_overlay(image, detections, mask, "offline")
    save_result(Path(args.out_dir).expanduser(), {
        "mode": "offline", "source_frame": "chest_camera_optical_frame",
        "detections": detections,
    }, image, overlay)
    return 0


def run_live(args: argparse.Namespace) -> int:
    context = rs.context()
    device = find_d435(context, args.serial)
    serial = device.get_info(rs.camera_info.serial_number)
    configure_auto_exposure(device, args.auto_exposure_limit)
    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    profile = pipeline.start(config)
    configure_auto_exposure(profile.get_device(), args.auto_exposure_limit)
    depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
    stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr = stream.get_intrinsics()
    intrinsics = {
        "fx": float(color_intr.fx), "fy": float(color_intr.fy),
        "ppx": float(color_intr.ppx), "ppy": float(color_intr.ppy),
        "width": int(color_intr.width), "height": int(color_intr.height),
    }
    align = rs.align(rs.stream.color)
    out_dir = Path(args.out_dir).expanduser().resolve()
    print(f"opened chest D435 S/N {serial}: {args.width}x{args.height}@{args.fps}, "
          f"AE limit={args.auto_exposure_limit}", flush=True)
    if not args.no_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    try:
        for _ in range(30):
            pipeline.wait_for_frames()
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color = np.asanyarray(color_frame.get_data()).copy()
            depth = np.asanyarray(depth_frame.get_data()).copy()
            detections, mask = detect_white_holes(color, depth, intrinsics, depth_scale, args)
            overlay = draw_overlay(color, detections, mask, f"D435 {serial}")
            payload = {
                "timestamp_unix": time.time(),
                "camera_name": "Chest D435",
                "camera_serial": serial,
                "source_frame": "chest_camera_optical_frame",
                "color_intrinsics": intrinsics,
                "depth_scale_m_per_unit": depth_scale,
                "parameters": {
                    "min_gray": args.min_gray,
                    "max_saturation": args.max_saturation,
                    "min_area": args.min_area,
                    "max_area": args.max_area,
                    "min_center_y_fraction": args.min_center_y_fraction,
                    "max_inner_gray": args.max_inner_gray,
                    "min_rim_contrast": args.min_rim_contrast,
                    "min_rim_white_fraction": args.min_rim_white_fraction,
                },
                "detections": detections,
            }
            save_result(out_dir, payload, color, overlay)
            if not args.no_window:
                cv2.imshow(WINDOW_NAME, overlay)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(str(out_dir / f"white_holes_{stamp}.jpg"), overlay)
    finally:
        pipeline.stop()
        if not args.no_window:
            cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.image:
        return run_offline(args)
    return run_live(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
