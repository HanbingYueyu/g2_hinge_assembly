#!/usr/bin/env python3
"""Run D405 hinge tracking plus chest-D435 white-hole localization.

Both cameras share one predictor so opening the second preview does not load a
second copy of the checkpoint into GPU memory.  Each camera gets its own
window and JSON file; the ROS bridge consumes both files independently.  The
chest stream does not run SAM hinge inference by default, but it continuously
locates white circular holes in the validated lower work area.
"""

import argparse
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

from live_d435_white_holes import detect_white_holes, draw_overlay as draw_white_hole_overlay, save_result as save_white_hole_result

from extract_sam31_3d_refs import infer_image, mask_stats, select_masks
from live_sam31_3d_realsense import draw_detections, detections_for_label, put_text
from sam3.model_builder import build_sam3_predictor


CAMERA_SPECS = (
    {
        "key": "chest",
        "display_name": "Chest D435",
        "window_name": "SAM3.1 Chest D435 Hinge 3D",
        "serial_param": "chest_serial",
        "source_frame": "chest_camera_optical_frame",
        "json_name": "live_3d_latest.json",
        "raw_name": "live_3d_latest_raw.jpg",
        "overlay_name": "live_3d_latest_overlay.jpg",
    },
    {
        "key": "left_d405",
        "display_name": "Left D405",
        "window_name": "SAM3.1 Left D405 Hinge 3D",
        "serial_param": "left_serial",
        "source_frame": "left_d405_camera_optical_frame",
        "json_name": "live_3d_d405.json",
        "raw_name": "live_3d_d405_raw.jpg",
        "overlay_name": "live_3d_d405_overlay.jpg",
        # Data-only sidecars from the exact RGB-D frame used by the existing
        # left-D405 SAM hinge result.  They never command either arm.
        "depth_name": "live_3d_d405_depth.png",
        "mask_name": "live_3d_d405_hinge_mask.png",
    },
)


def configure_auto_exposure(device, exposure_limit: int) -> None:
    """Match the known-good camera viewer's D400 auto-exposure settings."""
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


def device_info(device) -> tuple[str, str]:
    return (
        device.get_info(rs.camera_info.name),
        device.get_info(rs.camera_info.serial_number),
    )


def find_device(context, requested_serial: str, model_hint: str):
    devices = list(context.query_devices())
    by_serial = {device_info(device)[1]: device for device in devices}
    if requested_serial and requested_serial.lower() != "auto":
        device = by_serial.get(requested_serial)
        if device is None:
            available = ", ".join(
                f"{name}({serial})" for name, serial in map(device_info, devices)
            ) or "none"
            raise RuntimeError(
                f"{model_hint} serial {requested_serial} not found; available: {available}"
            )
        actual_name, _ = device_info(device)
        if model_hint.lower() not in actual_name.lower():
            raise RuntimeError(
                f"serial {requested_serial} is reported as {actual_name}, "
                f"not the requested {model_hint}"
            )
        return device

    matches = [
        device
        for device in devices
        if model_hint.lower() in device_info(device)[0].lower()
    ]
    if not matches:
        raise RuntimeError(f"No RealSense device matching {model_hint} found")
    return matches[0]


def start_camera(context, spec, serial, args):
    device = find_device(context, serial, spec["display_name"].split()[-1])
    name, actual_serial = device_info(device)
    configure_auto_exposure(device, args.auto_exposure_limit)

    pipeline = rs.pipeline(context)
    config = rs.config()
    config.enable_device(actual_serial)
    config.enable_stream(
        rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps
    )
    config.enable_stream(
        rs.stream.depth, args.width, args.height, rs.format.z16, args.fps
    )
    profile = pipeline.start(config)
    # Apply again to the active sensor: this is needed by some D405 firmware
    # versions because the pre-start setting is cached by librealsense.
    try:
        configure_auto_exposure(
            profile.get_device(), args.auto_exposure_limit
        )
    except RuntimeError as exc:
        print(f"warning: cannot set AE on {actual_serial}: {exc}", file=sys.stderr)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = float(depth_sensor.get_depth_scale())
    color_intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    color_intr_dict = {
        "fx": color_intr.fx,
        "fy": color_intr.fy,
        "ppx": color_intr.ppx,
        "ppy": color_intr.ppy,
        "width": color_intr.width,
        "height": color_intr.height,
    }
    return {
        **spec,
        "serial": actual_serial,
        "model": name,
        "pipeline": pipeline,
        "align": rs.align(rs.stream.color),
        "depth_scale": depth_scale,
        "color_intr": color_intr_dict,
        "latest_color": None,
        "latest_depth": None,
        "overlay": None,
        "detections": [],
        "busy": False,
        "next_infer": 0.0,
        "last_time": time.monotonic(),
        "latency_ms": 0.0,
        "hinge_count": 0,
        "completed": 0,
        "error": None,
        "actual_exposure": -1,
        # Chest D435 remains owned by this process.  White-hole localization
        # reuses the aligned RGB-D pair, so it never opens a second pipeline
        # for the same physical camera.
        "next_white_hole": 0.0,
    }


def annotate_camera_overlay(image: np.ndarray, state: dict, status: dict) -> np.ndarray:
    overlay = draw_detections(image, status.get("detections", []), status)
    put_text(
        overlay,
        f"{state['display_name']}  S/N {state['serial']}  "
        f"AE=ON limit={status['ae_limit']} actual={status.get('actual_exposure', -1)}",
        (10, 48),
        (0, 220, 255),
        0.52,
    )
    if not status.get("detection_enabled", True):
        put_text(
            overlay,
            "SAM hinge detection: OFF (camera preview only)",
            (10, 74),
            (0, 180, 255),
            0.55,
        )
    return overlay


def write_image_atomic(path: Path, image: np.ndarray) -> None:
    """Prevent readers from observing a half-written RGB-D sidecar."""
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if not cv2.imwrite(str(temporary), image):
        raise RuntimeError(f"cannot write image {temporary}")
    temporary.replace(path)


def inference_worker(states, shared, lock, event, stop, predictor, args):
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    while not stop.is_set():
        if not event.wait(0.1):
            continue
        event.clear()
        with lock:
            task = shared.get("pending_task")
            shared["pending_task"] = None
        if task is None:
            continue

        key, color, depth, capture_timestamp_unix = task
        state = states[key]
        with lock:
            shared["worker_active"] = True
        start = time.monotonic()
        try:
            hinge, hinge_raw = detections_for_label(
                predictor,
                color,
                depth,
                "silver_metal_hinge",
                args.hinge_prompt,
                args.threshold,
                state["depth_scale"],
                state["color_intr"],
                args,
                include_masks=(key == "left_d405"),
            )
            # The selected mask is only used to emit a registration sidecar;
            # remove it before the established JSON/RViz paths see the data.
            selected_mask = None
            if key == "left_d405" and hinge:
                selected_mask = hinge[0].pop("_selected_mask", None)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            status = {
                "hinge_count": len(hinge),
                "latency_ms": elapsed_ms,
                "last_time": time.monotonic(),
                "detections": hinge,
                "ae_limit": args.auto_exposure_limit,
                "actual_exposure": state["actual_exposure"],
            }
            overlay = annotate_camera_overlay(color, state, status)
            payload = {
                "timestamp_unix": time.time(),
                # Wall time when the RGB-D pair was copied, before SAM
                # inference.  The visual-servo node uses this to query the
                # corresponding historical camera TF instead of combining an
                # old image with the arm's current pose.
                "capture_timestamp_unix": capture_timestamp_unix,
                "camera_name": state["display_name"],
                "camera_model": state["model"],
                "camera_serial": state["serial"],
                "source_frame": state["source_frame"],
                # The image-target servo needs the actual stream intrinsics.
                # Its target is deliberately below the optical centre, so
                # center_xyz_m alone is not a valid image-target error.
                "color_intrinsics": state["color_intr"],
                # Registration sidecars retain raw z16 depth.  Export the
                # physical scale with the intrinsics so a consumer never
                # guesses millimetres versus D405 0.1-mm depth units.
                "depth_scale_m_per_unit": state["depth_scale"],
                "hinge_prompt": args.hinge_prompt,
                "threshold": args.threshold,
                "auto_exposure_limit": args.auto_exposure_limit,
                "raw_mask_counts": {"silver_metal_hinge": hinge_raw},
                "detections": hinge,
            }
            if key == "left_d405":
                payload["registration_sidecars"] = {
                    "color_bgr": state["raw_name"],
                    "depth_u16": state["depth_name"],
                    "hinge_mask": state["mask_name"],
                }
            json_path = out_dir / state["json_name"]
            temp_json_path = json_path.with_suffix(json_path.suffix + ".tmp")
            temp_json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            # The bridge polls these files. Replace atomically so it never
            # observes a half-written JSON document during inference.
            temp_json_path.replace(json_path)
            write_image_atomic(out_dir / state["raw_name"], color)
            write_image_atomic(out_dir / state["overlay_name"], overlay)
            if key == "left_d405":
                write_image_atomic(out_dir / state["depth_name"], depth)
                mask_image = np.zeros(depth.shape, dtype=np.uint8)
                if selected_mask is not None:
                    mask_image[selected_mask] = 255
                write_image_atomic(out_dir / state["mask_name"], mask_image)
            with lock:
                state["overlay"] = overlay
                state["detections"] = hinge
                state["hinge_count"] = len(hinge)
                state["latency_ms"] = elapsed_ms
                state["last_time"] = time.monotonic()
                state["error"] = None
                state["completed"] += 1
                shared["completed_total"] += 1
        except Exception as exc:
            with lock:
                state["error"] = repr(exc)
        finally:
            with lock:
                state["busy"] = False
                shared["worker_active"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual RealSense SAM3.1 hinge detector")
    parser.add_argument("--checkpoint", default="checkpoints/sam3.1_multiplex.pt")
    parser.add_argument("--hinge-prompt", default="silver metal hinge")
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--infer-interval", type=float, default=0.75)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--out-dir", default="outputs/live_3d")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--max-inferences", type=int, default=0)
    parser.add_argument("--hinge-min-area", type=int, default=1500)
    parser.add_argument("--auto-exposure-limit", type=int, default=4000)
    # Parameters below are the validated D435 white-hole work-area filter.
    # In particular, objects above 40% of the image height are excluded;
    # this rejects the circular robot parts above the tabletop.
    parser.add_argument("--white-hole-interval", type=float, default=0.20)
    parser.add_argument("--white-hole-min-gray", type=int, default=150)
    parser.add_argument("--white-hole-max-saturation", type=int, default=105)
    parser.add_argument("--white-hole-min-area", type=float, default=80.0)
    parser.add_argument("--white-hole-max-area", type=float, default=8000.0)
    parser.add_argument("--white-hole-min-radius", type=float, default=10.0)
    parser.add_argument("--white-hole-max-radius", type=float, default=45.0)
    parser.add_argument("--white-hole-min-center-y-fraction", type=float, default=0.40)
    parser.add_argument("--white-hole-min-circle-distance", type=float, default=35.0)
    parser.add_argument("--white-hole-hough-edge-threshold", type=float, default=80.0)
    parser.add_argument("--white-hole-hough-accumulator-threshold", type=float, default=24.0)
    parser.add_argument("--white-hole-max-inner-gray", type=float, default=110.0)
    parser.add_argument("--white-hole-min-rim-contrast", type=float, default=25.0)
    parser.add_argument("--white-hole-min-rim-white-fraction", type=float, default=0.08)
    parser.add_argument("--chest-serial", default="346522070661")
    parser.add_argument("--left-serial", default="260322275186")
    parser.add_argument(
        "--disable-chest-detection",
        action="store_true",
        help="Open and preview the chest D435 but do not schedule SAM inference for it.",
    )
    parser.add_argument(
        "--disable-white-hole-detection",
        action="store_true",
        help=(
            "Keep the chest D435 stream open but disable white-hole localization, "
            "result output, and the D435 White Hole 3D preview."
        ),
    )
    return parser.parse_args()


def white_hole_args(args: argparse.Namespace) -> argparse.Namespace:
    """Adapt the dual-stream CLI options to the standalone white-hole API."""
    return argparse.Namespace(
        min_gray=args.white_hole_min_gray,
        max_saturation=args.white_hole_max_saturation,
        min_area=args.white_hole_min_area,
        max_area=args.white_hole_max_area,
        min_radius=args.white_hole_min_radius,
        max_radius=args.white_hole_max_radius,
        min_center_y_fraction=args.white_hole_min_center_y_fraction,
        min_circle_distance=args.white_hole_min_circle_distance,
        hough_edge_threshold=args.white_hole_hough_edge_threshold,
        hough_accumulator_threshold=args.white_hole_hough_accumulator_threshold,
        max_inner_gray=args.white_hole_max_inner_gray,
        min_rim_contrast=args.white_hole_min_rim_contrast,
        min_rim_white_fraction=args.white_hole_min_rim_white_fraction,
    )


def main() -> int:
    args = parse_args()
    white_args = white_hole_args(args)
    white_hole_out_dir = Path("outputs/white_holes").expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    print(f"loading SAM3.1 checkpoint={checkpoint}", flush=True)
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
    print("model loaded; opening D435 and D405", flush=True)

    context = rs.context()
    states = {}
    for spec in CAMERA_SPECS:
        serial = getattr(args, spec["serial_param"])
        try:
            state = start_camera(context, spec, serial, args)
            state["detection_enabled"] = not (
                spec["key"] == "chest" and args.disable_chest_detection
            )
            states[state["key"]] = state
            detection_mode = "SAM=OFF preview-only" if not state["detection_enabled"] else "SAM=ON"
            print(
                f"opened {state['display_name']}: {state['model']} S/N {state['serial']} "
                f"{args.width}x{args.height}@{args.fps}, AE limit={args.auto_exposure_limit}, "
                f"{detection_mode}",
                flush=True,
            )
        except Exception:
            for opened in states.values():
                opened["pipeline"].stop()
            raise

    lock = Lock()
    event = Event()
    stop = Event()
    shared = {"pending_task": None, "worker_active": False, "completed_total": 0}
    worker = Thread(
        target=inference_worker,
        args=(states, shared, lock, event, stop, predictor, args),
        daemon=True,
    )
    worker.start()

    try:
        for state in states.values():
            for _ in range(args.warmup):
                state["pipeline"].wait_for_frames()
        if not args.no_window:
            for state in states.values():
                cv2.namedWindow(state["window_name"], cv2.WINDOW_NORMAL)
                cv2.resizeWindow(state["window_name"], args.width, args.height)
            if "chest" in states and not args.disable_white_hole_detection:
                cv2.namedWindow("D435 White Hole 3D", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("D435 White Hole 3D", args.width, args.height)

        while not stop.is_set():
            for state in states.values():
                try:
                    frames = state["align"].process(
                        state["pipeline"].wait_for_frames(1000)
                    )
                    color_frame = frames.get_color_frame()
                    depth_frame = frames.get_depth_frame()
                    if not color_frame or not depth_frame:
                        continue
                    color = np.asanyarray(color_frame.get_data()).copy()
                    depth = np.asanyarray(depth_frame.get_data()).copy()
                    capture_timestamp_unix = time.time()
                    try:
                        actual_exposure = color_frame.get_frame_metadata(
                            rs.frame_metadata_value.actual_exposure
                        )
                    except RuntimeError:
                        actual_exposure = -1
                    with lock:
                        state["latest_color"] = color
                        state["latest_depth"] = depth
                        state["actual_exposure"] = actual_exposure
                        worker_busy = (
                            shared["pending_task"] is not None
                            or shared["worker_active"]
                        )
                        now = time.monotonic()
                        white_hole_due = (
                            not args.disable_white_hole_detection
                            and state["key"] == "chest"
                            and now >= state["next_white_hole"]
                        )
                        if white_hole_due:
                            state["next_white_hole"] = now + max(
                                0.05, args.white_hole_interval
                            )
                        if (
                            state["detection_enabled"]
                            and now >= state["next_infer"]
                            and not state["busy"]
                            and not worker_busy
                        ):
                            state["busy"] = True
                            state["next_infer"] = now + max(0.05, args.infer_interval)
                            shared["pending_task"] = (
                                state["key"], color, depth, capture_timestamp_unix
                            )
                            event.set()
                        status = {
                            "hinge_count": state["hinge_count"],
                            "latency_ms": state["latency_ms"],
                            "last_time": state["last_time"],
                            "busy": state["busy"],
                            "error": state["error"],
                            "detections": list(state["detections"]),
                            "ae_limit": args.auto_exposure_limit,
                            "actual_exposure": state["actual_exposure"],
                            "detection_enabled": state["detection_enabled"],
                        }
                        overlay = (
                            state["overlay"].copy()
                            if state["overlay"] is not None
                            else annotate_camera_overlay(color, state, status)
                        )
                    if white_hole_due:
                        try:
                            white_detections, white_mask = detect_white_holes(
                                color,
                                depth,
                                state["color_intr"],
                                state["depth_scale"],
                                white_args,
                            )
                            white_overlay = draw_white_hole_overlay(
                                color,
                                white_detections,
                                white_mask,
                                f"D435 {state['serial']} work-area y>={args.white_hole_min_center_y_fraction:.2f}",
                            )
                            save_white_hole_result(
                                white_hole_out_dir,
                                {
                                    "timestamp_unix": time.time(),
                                    "capture_timestamp_unix": capture_timestamp_unix,
                                    "camera_name": "Chest D435",
                                    "camera_serial": state["serial"],
                                    "source_frame": state["source_frame"],
                                    "color_intrinsics": state["color_intr"],
                                    "parameters": vars(white_args),
                                    "detections": white_detections,
                                },
                                color,
                                white_overlay,
                            )
                            if not args.no_window:
                                cv2.imshow("D435 White Hole 3D", white_overlay)
                        except Exception as exc:
                            # White-hole output must never interrupt D405 SAM
                            # tracking or the existing grasp flow.
                            print(f"white-hole D435 processing failed: {exc}", file=sys.stderr)
                    if not args.no_window:
                        cv2.imshow(state["window_name"], overlay)
                except RuntimeError as exc:
                    with lock:
                        state["error"] = repr(exc)
                    print(f"read {state['serial']} failed: {exc}", file=sys.stderr)

            if not args.no_window:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    out_dir = Path(args.out_dir).expanduser().resolve()
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime("%Y%m%d_%H%M%S")
                    with lock:
                        for state in states.values():
                            if state["overlay"] is not None:
                                cv2.imwrite(
                                    str(out_dir / f"saved_{state['key']}_{stamp}.jpg"),
                                    state["overlay"],
                                )

            with lock:
                completed = shared["completed_total"]
            if args.max_inferences > 0 and completed >= args.max_inferences:
                break
    finally:
        stop.set()
        event.set()
        worker.join(timeout=5.0)
        for state in states.values():
            try:
                state["pipeline"].stop()
            except RuntimeError:
                pass
        if not args.no_window:
            cv2.destroyAllWindows()
        del predictor
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
