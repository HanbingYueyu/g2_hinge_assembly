#!/usr/bin/env python3
"""Simultaneous right-D405 white-hole and slotted-shaft 3-D measurement.

Both targets are segmented by their reviewed-mask models on the same RGB-D
frame.  Each target's mask plus aligned depth yields one 3-D centre.  The ROS
image overlay projects a direct line between those two centres.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs

from live_right_d405_shaft_seg_inference import ShaftSegmenter, add_3d, configure_auto_exposure, find_device, select_component


WINDOW_NAME = "Right D405 White Hole + Slotted Shaft 3D"


class ShaftLocalTracker:
    """Lock one shaft instance, then associate masks only locally.

    The segmentation network is still used to produce a mask.  The important
    difference is identity handling: after a short stable acquisition period,
    candidates elsewhere in the image are deliberately ignored.  A one-frame
    false positive on a cable or on the opposite arm can therefore never
    replace a previously locked shaft.  If the local association is genuinely
    lost, global acquisition is restarted and must pass the same multi-frame
    stability test before anything is published again.
    """

    def __init__(
        self,
        acquire_stable_frames: int,
        acquire_jitter_px: float,
        search_radius_px: float,
        association_max_depth_delta_m: float,
        filter_alpha: float,
    ) -> None:
        self.acquire_stable_frames = max(1, int(acquire_stable_frames))
        self.acquire_jitter_px = max(1.0, float(acquire_jitter_px))
        self.search_radius_px = max(10.0, float(search_radius_px))
        self.association_max_depth_delta_m = max(
            0.001, float(association_max_depth_delta_m)
        )
        self.filter_alpha = min(1.0, max(0.05, float(filter_alpha)))
        self.state = "ACQUIRING"
        self.acquire_count = 0
        self.acquire_center: np.ndarray | None = None
        self.acquire_depth_m: float | None = None
        self.locked_center: np.ndarray | None = None
        self.locked_depth_m: float | None = None

    def reset(self) -> None:
        self.state = "ACQUIRING"
        self.acquire_count = 0
        self.acquire_center = None
        self.acquire_depth_m = None
        self.locked_center = None
        self.locked_depth_m = None

    def restart_global_acquisition(self, reason: str) -> tuple[None, str]:
        """Drop a stale local lock without permanently disabling detection.

        The next full-frame model inference is treated exactly like initial
        acquisition: it must remain spatially/depth stable for several frames
        before the shaft is allowed back into the 3-D result.  This preserves
        the anti-false-positive property while allowing a valid shaft to be
        found again after an occlusion or a larger P4 motion.
        """
        self.reset()
        return None, (
            f"shaft tracking lost ({reason}); restarting global acquisition "
            f"({self.acquire_stable_frames} stable frames required)"
        )

    def status(self) -> dict:
        return {
            "state": self.state,
            "acquire_count": self.acquire_count,
            "acquire_stable_frames_required": self.acquire_stable_frames,
            "search_radius_px": self.search_radius_px,
            "locked_center_px": (
                None
                if self.locked_center is None
                else [float(self.locked_center[0]), float(self.locked_center[1])]
            ),
            "locked_depth_m": self.locked_depth_m,
        }

    @staticmethod
    def _center(detection: dict) -> np.ndarray:
        return np.asarray(detection["center_px"], dtype=float)

    def acquire(
        self, detection: dict | None, intrinsics: dict
    ) -> tuple[dict | None, str]:
        """Accept a global candidate only until one identity is locked."""
        if detection is None:
            self.acquire_count = 0
            self.acquire_center = None
            self.acquire_depth_m = None
            return None, "waiting for initial shaft candidate"
        center = self._center(detection)
        depth = float(detection["depth_median_m"])
        if (
            self.acquire_center is None
            or self.acquire_depth_m is None
            or float(np.linalg.norm(center - self.acquire_center))
            > self.acquire_jitter_px
            or abs(depth - self.acquire_depth_m)
            > self.association_max_depth_delta_m
        ):
            self.acquire_count = 1
            self.acquire_center = center
            self.acquire_depth_m = depth
        else:
            self.acquire_count += 1
            alpha = self.filter_alpha
            self.acquire_center = alpha * center + (1.0 - alpha) * self.acquire_center
            self.acquire_depth_m = alpha * depth + (1.0 - alpha) * self.acquire_depth_m
        if self.acquire_count < self.acquire_stable_frames:
            return None, (
                "shaft acquiring "
                f"{self.acquire_count}/{self.acquire_stable_frames}"
            )
        self.state = "LOCKED"
        self.locked_center = self.acquire_center.copy()
        self.locked_depth_m = float(self.acquire_depth_m)
        return self._filtered_detection(detection, intrinsics), "shaft locked; local tracking only"

    def associate(
        self, detection: dict | None, intrinsics: dict
    ) -> tuple[dict | None, str]:
        """Update only the locked instance, then safely reacquire if lost."""
        if self.state != "LOCKED" or self.locked_center is None or self.locked_depth_m is None:
            return None, "shaft tracker is not locked"
        if detection is None:
            return self.restart_global_acquisition("no local candidate")
        center = self._center(detection)
        depth = float(detection["depth_median_m"])
        displacement = float(np.linalg.norm(center - self.locked_center))
        depth_delta = abs(depth - self.locked_depth_m)
        if displacement > self.search_radius_px or depth_delta > self.association_max_depth_delta_m:
            return self.restart_global_acquisition(
                "local association mismatch "
                f"pixel={displacement:.1f}px depth={depth_delta * 1000.0:.1f}mm"
            )
        return self._filtered_detection(detection, intrinsics), "shaft tracked locally"

    def _filtered_detection(self, detection: dict, intrinsics: dict) -> dict:
        """Return the current mask with a low-pass filtered 2-D/3-D centre."""
        center = self._center(detection)
        depth = float(detection["depth_median_m"])
        assert self.locked_center is not None
        assert self.locked_depth_m is not None
        alpha = self.filter_alpha
        self.locked_center = alpha * center + (1.0 - alpha) * self.locked_center
        self.locked_depth_m = alpha * depth + (1.0 - alpha) * self.locked_depth_m
        result = dict(detection)
        result["center_px"] = [float(self.locked_center[0]), float(self.locked_center[1])]
        result["center_xyz_m"] = [
            float((self.locked_center[0] - intrinsics["ppx"]) / intrinsics["fx"] * self.locked_depth_m),
            float((self.locked_center[1] - intrinsics["ppy"]) / intrinsics["fy"] * self.locked_depth_m),
            float(self.locked_depth_m),
        ]
        result["depth_median_m"] = float(self.locked_depth_m)
        result["feature_type"] = "tracked_local_trained_segmentation"
        return result


def restrict_probability_to_local_search(
    probability: np.ndarray, center_px: np.ndarray, radius_px: float
) -> np.ndarray:
    """Mask all segmentation probability outside the locked local window."""
    rows, cols = np.ogrid[: probability.shape[0], : probability.shape[1]]
    radius_sq = float(radius_px) ** 2
    inside = (cols - float(center_px[0])) ** 2 + (rows - float(center_px[1])) ** 2 <= radius_sq
    return np.where(inside, probability, 0.0)


def overlay_mask(view: np.ndarray, mask: np.ndarray | None, color: tuple[int, int, int]) -> np.ndarray:
    if mask is None:
        return view
    layer = np.zeros_like(view)
    layer[mask] = color
    return cv2.addWeighted(view, 1.0, layer, 0.36, 0.0)


def draw_detection(view: np.ndarray, detection: dict | None, color: tuple[int, int, int], title: str) -> None:
    if detection is None:
        return
    x0, y0, x1, y1 = detection["bbox_xyxy"]
    cv2.rectangle(view, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
    u, v = detection["center_px"]
    cv2.circle(view, (round(u), round(v)), 4, color, -1)
    cv2.putText(view, f"{title} p={detection['mean_probability']:.2f} z={detection['depth_median_m']:.3f}m", (x0, max(18, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_world_overlay(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("ready") is not True or time.time() - float(payload.get("timestamp_unix", 0.0)) > 1.0:
        return None
    return payload


def draw_world_overlay(view: np.ndarray, overlay_payload: dict | None) -> None:
    if overlay_payload is None:
        return
    overlay = overlay_payload.get("overlay", {})
    origin = overlay.get("origin_px")
    if not isinstance(origin, list) or len(origin) != 2:
        return
    origin_xy = tuple(round(value) for value in origin)
    colors = {"x": (0, 0, 255), "y": (0, 255, 0), "z": (255, 0, 0)}  # BGR
    for name, endpoint in overlay.get("world_axes_end_px", {}).items():
        if name not in colors or not isinstance(endpoint, list) or len(endpoint) != 2:
            continue
        endpoint_xy = tuple(round(value) for value in endpoint)
        cv2.arrowedLine(view, origin_xy, endpoint_xy, colors[name], 2, cv2.LINE_AA, tipLength=0.12)
        cv2.putText(view, name.upper(), endpoint_xy, cv2.FONT_HERSHEY_SIMPLEX, 0.50, colors[name], 2, cv2.LINE_AA)
    shaft_center = overlay.get("shaft_center_px")
    if isinstance(shaft_center, list) and len(shaft_center) == 2:
        shaft_center_xy = tuple(round(value) for value in shaft_center)
        cv2.line(view, origin_xy, shaft_center_xy, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(view, shaft_center_xy, 5, (0, 0, 255), -1)
    cv2.circle(view, origin_xy, 5, (0, 255, 255), -1)
    distance = float(overlay_payload.get("shaft_center_to_hole_distance_m", 0.0)) * 1000.0
    delta = overlay_payload.get("shaft_center_minus_hole_world_xyz_m")
    if isinstance(delta, list) and len(delta) == 3:
        delta_text = f"dXYZ=({delta[0] * 1000.0:+.1f}, {delta[1] * 1000.0:+.1f}, {delta[2] * 1000.0:+.1f}) mm"
    else:
        delta_text = "dXYZ=waiting for world TF"
    cv2.putText(view, f"world origin: white hole   center distance={distance:.1f} mm", (10, view.shape[0] - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(view, delta_text, (10, view.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 165, 255), 2, cv2.LINE_AA)


def start_stream_with_retries(context: rs.context, args: argparse.Namespace):
    """Start the right D405 and require actual RGB-D frames before success.

    A D405 can accept ``pipeline.start`` while its USB stream is still coming
    up.  In that case the old code exited after its first 5 s frame timeout,
    leaving the automatic fourth-preset flow with no camera window.
    """
    attempts = max(1, int(args.stream_start_attempts))
    retry_delay = max(0.1, float(args.stream_retry_delay))
    reset_after = max(0, int(args.stream_reset_after))
    last_error: RuntimeError | None = None
    for attempt in range(1, attempts + 1):
        pipeline = None
        started = False
        try:
            pipeline = rs.pipeline(context)
            config = rs.config()
            config.enable_device(args.serial)
            config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
            config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
            profile = pipeline.start(config)
            started = True
            configure_auto_exposure(profile.get_device(), args.auto_exposure_limit)
            depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
            intrinsics_raw = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            intrinsics = {
                "fx": intrinsics_raw.fx,
                "fy": intrinsics_raw.fy,
                "ppx": intrinsics_raw.ppx,
                "ppy": intrinsics_raw.ppy,
            }
            for _ in range(max(1, args.warmup)):
                pipeline.wait_for_frames(1000)
            print(
                f"right D405 stream ready on attempt {attempt}/{attempts}; "
                f"serial={args.serial}",
                flush=True,
            )
            return pipeline, rs.align(rs.stream.color), depth_scale, intrinsics
        except RuntimeError as exc:
            last_error = exc
            print(
                f"right D405 stream attempt {attempt}/{attempts} failed: {exc}",
                flush=True,
            )
            if started and pipeline is not None:
                try:
                    pipeline.stop()
                except RuntimeError:
                    pass
            if reset_after and attempt == reset_after:
                try:
                    print(
                        f"right D405 has no frames; resetting only serial {args.serial} "
                        "before the remaining retries",
                        flush=True,
                    )
                    find_device(context, args.serial).hardware_reset()
                    # A hardware reset removes and re-enumerates this one USB
                    # device.  Recreate the context after it comes back.
                    time.sleep(max(3.0, retry_delay))
                    context = rs.context()
                except RuntimeError as reset_exc:
                    print(f"right D405 hardware reset failed: {reset_exc}", flush=True)
            if attempt < attempts:
                time.sleep(retry_delay)
    raise RuntimeError(
        f"right D405 {args.serial} produced no RGB-D frames after {attempts} attempts"
    ) from last_error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", default="260322275349")
    parser.add_argument("--source-frame", default="right_d405_camera_optical_frame")
    parser.add_argument("--hole-checkpoint", default="runs/right_d405_new_target_seg_torch/positive_only_v5/best.pt")
    parser.add_argument("--shaft-checkpoint", default="runs/right_d405_shaft_seg_torch/mixed_auto_p4_v7/best.pt")
    parser.add_argument("--out-dir", default="outputs/right_d405_hole_shaft_measurement")
    parser.add_argument("--world-overlay-json", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--stream-start-attempts", type=int, default=8)
    parser.add_argument("--stream-retry-delay", type=float, default=1.0)
    parser.add_argument(
        "--stream-reset-after",
        type=int,
        default=3,
        help="After this many failed stream starts, reset only the right D405 once (0 disables).",
    )
    parser.add_argument("--auto-exposure-limit", type=int, default=4000)
    parser.add_argument("--inference-hz", type=float, default=6.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hole-threshold", type=float, default=0.50)
    parser.add_argument("--hole-min-area-px", type=int, default=400)
    parser.add_argument("--hole-max-area-px", type=int, default=3500)
    parser.add_argument("--shaft-threshold", type=float, default=0.50)
    parser.add_argument("--shaft-min-area-px", type=int, default=500)
    parser.add_argument("--shaft-max-area-px", type=int, default=5500)
    parser.add_argument("--shaft-min-mean-probability", type=float, default=0.70)
    parser.add_argument("--shaft-max-aspect-ratio", type=float, default=2.05)
    parser.add_argument("--shaft-min-short-side-px", type=int, default=18)
    parser.add_argument("--shaft-min-fill-ratio", type=float, default=0.45)
    parser.add_argument(
        "--shaft-acquire-stable-frames",
        type=int,
        default=4,
        help="Global shaft candidate must be stable for this many inferences before locking.",
    )
    parser.add_argument(
        "--shaft-acquire-jitter-px",
        type=float,
        default=22.0,
        help="Maximum 2-D centre change while acquiring the initial shaft lock.",
    )
    parser.add_argument(
        "--shaft-track-search-radius-px",
        type=float,
        default=100.0,
        help="After lock, only this-radius local mask search may update the shaft.",
    )
    parser.add_argument(
        "--shaft-track-max-depth-delta-m",
        type=float,
        default=0.040,
        help="Maximum local association depth change before declaring the shaft lost.",
    )
    parser.add_argument(
        "--shaft-track-filter-alpha",
        type=float,
        default=0.35,
        help="EMA coefficient for a locked shaft centre/depth; 1.0 disables smoothing.",
    )
    parser.add_argument(
        "--disable-shaft-validation",
        action="store_true",
        help=(
            "Show the raw shaft-model component without the additional "
            "confidence, shape, and hole-distance false-positive gates."
        ),
    )
    parser.add_argument(
        "--shaft-max-hole-distance-m",
        type=float,
        default=0.160,
        help="Reject a shaft candidate implausibly far from the hole at P4.",
    )
    args = parser.parse_args()

    hole_checkpoint, shaft_checkpoint = Path(args.hole_checkpoint).expanduser().resolve(), Path(args.shaft_checkpoint).expanduser().resolve()
    for name, checkpoint in (("white-hole", hole_checkpoint), ("shaft", shaft_checkpoint)):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {checkpoint}")
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "right_d405_hole_shaft_measurement_latest.json"
    overlay_path = Path(args.world_overlay_json).expanduser().resolve() if args.world_overlay_json else out_dir / "right_d405_hole_shaft_overlay_latest.json"
    hole_segmenter = ShaftSegmenter(hole_checkpoint, args.device)
    shaft_segmenter = ShaftSegmenter(shaft_checkpoint, args.device)
    hole_gate = argparse.Namespace(threshold=args.hole_threshold, min_area_px=args.hole_min_area_px, max_area_px=args.hole_max_area_px)
    shaft_gate = argparse.Namespace(
        threshold=args.shaft_threshold,
        min_area_px=args.shaft_min_area_px,
        max_area_px=args.shaft_max_area_px,
        min_mean_probability=(0.0 if args.disable_shaft_validation else args.shaft_min_mean_probability),
        max_aspect_ratio=(float("inf") if args.disable_shaft_validation else args.shaft_max_aspect_ratio),
        min_short_side_px=(0 if args.disable_shaft_validation else args.shaft_min_short_side_px),
        min_fill_ratio=(0.0 if args.disable_shaft_validation else args.shaft_min_fill_ratio),
    )
    validation_mode = "OFF (raw model component)" if args.disable_shaft_validation else "ON"
    shaft_tracker = ShaftLocalTracker(
        args.shaft_acquire_stable_frames,
        args.shaft_acquire_jitter_px,
        args.shaft_track_search_radius_px,
        args.shaft_track_max_depth_delta_m,
        args.shaft_track_filter_alpha,
    )
    print(
        f"loaded white-hole model={hole_checkpoint}; shaft model={shaft_checkpoint}; "
        f"device={hole_segmenter.device}; shaft validation={validation_mode}; "
        f"shaft tracker=acquire {shaft_tracker.acquire_stable_frames} then local-only "
        "with safe global re-acquisition after loss",
        flush=True,
    )

    context = rs.context()
    device = find_device(context, args.serial)
    configure_auto_exposure(device, args.auto_exposure_limit)
    pipeline, align, depth_scale, intrinsics = start_stream_with_retries(context, args)
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        hole_component = shaft_component = None
        hole_detection = shaft_detection = None
        inference_ms, next_inference, inference_timestamp_unix = 0.0, 0.0, 0.0
        shaft_rejection_reason = ""
        shaft_tracking_status = "waiting for initial shaft candidate"
        while True:
            frames = align.process(pipeline.wait_for_frames())
            color_frame, depth_frame = frames.get_color_frame(), frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue
            color, depth = np.asanyarray(color_frame.get_data()).copy(), np.asanyarray(depth_frame.get_data()).copy()
            now = time.monotonic()
            if now >= next_inference:
                start = time.monotonic()
                hole_component = select_component(hole_segmenter.probabilities(color), hole_gate)
                shaft_probability = shaft_segmenter.probabilities(color)
                if shaft_tracker.state == "LOCKED" and shaft_tracker.locked_center is not None:
                    shaft_probability = restrict_probability_to_local_search(
                        shaft_probability,
                        shaft_tracker.locked_center,
                        shaft_tracker.search_radius_px,
                    )
                shaft_component = select_component(shaft_probability, shaft_gate)
                hole_detection = None if hole_component is None else add_3d(hole_component, depth, depth_scale, intrinsics, "white_hole")
                raw_shaft_detection = None if shaft_component is None else add_3d(shaft_component, depth, depth_scale, intrinsics, "slotted_hinge_shaft")
                shaft_rejection_reason = ""
                if hole_detection is not None and raw_shaft_detection is not None:
                    hole_xyz = np.asarray(hole_detection["center_xyz_m"], dtype=float)
                    shaft_xyz = np.asarray(raw_shaft_detection["center_xyz_m"], dtype=float)
                    pair_distance = float(np.linalg.norm(shaft_xyz - hole_xyz))
                    if (
                        not args.disable_shaft_validation
                        and pair_distance > args.shaft_max_hole_distance_m
                    ):
                        shaft_component = None
                        raw_shaft_detection = None
                        shaft_rejection_reason = (
                            "shaft rejected: hole distance="
                            f"{pair_distance * 1000.0:.1f}mm > "
                            f"{args.shaft_max_hole_distance_m * 1000.0:.1f}mm"
                        )
                        print(shaft_rejection_reason, flush=True)
                if shaft_tracker.state == "ACQUIRING":
                    shaft_detection, shaft_tracking_status = shaft_tracker.acquire(
                        raw_shaft_detection, intrinsics
                    )
                    # Do not show or publish an unconfirmed candidate as the
                    # grasped shaft.  The mask remains visible only once the
                    # identity has completed the stable lock.
                    if shaft_detection is None:
                        shaft_component = None
                elif shaft_tracker.state == "LOCKED":
                    shaft_detection, shaft_tracking_status = shaft_tracker.associate(
                        raw_shaft_detection, intrinsics
                    )
                    if shaft_detection is None:
                        shaft_component = None
                else:
                    # This branch is only retained for compatibility with an
                    # externally restored legacy state.  Never leave a live
                    # P4 measurement permanently LOST: restart the guarded
                    # multi-frame global acquisition instead.
                    shaft_tracker.reset()
                    shaft_detection, shaft_tracking_status = shaft_tracker.acquire(
                        raw_shaft_detection, intrinsics
                    )
                    if shaft_detection is None:
                        shaft_component = None
                inference_ms = (time.monotonic() - start) * 1000.0
                inference_timestamp_unix = time.time()
                next_inference = now + 1.0 / max(args.inference_hz, 0.1)
            view = overlay_mask(color.copy(), shaft_component["mask"] if shaft_component is not None else None, (0, 0, 255))
            view = overlay_mask(view, hole_component["mask"] if hole_component is not None else None, (0, 255, 255))
            draw_detection(view, shaft_detection, (0, 0, 255), "shaft")
            draw_detection(view, hole_detection, (0, 255, 255), "white hole")
            draw_world_overlay(view, load_world_overlay(overlay_path))
            status = "READY" if hole_detection is not None and shaft_detection is not None else "WAITING FOR WHITE HOLE + SLOTTED SHAFT"
            lines = [
                "Right D405: white hole + slotted shaft 3D measurement",
                f"{status}; inference={inference_ms:.0f}ms; q/esc quit, s save debug",
                "yellow=white-hole mask; red=shaft mask; camera view connects their 3-D centres",
                shaft_tracking_status,
            ]
            if shaft_rejection_reason:
                lines.append(shaft_rejection_reason)
            for row, text in enumerate(lines):
                y = 25 + row * 22
                cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW_NAME, view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("s"):
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(str(out_dir / f"hole_shaft_{stamp}.jpg"), view)
                cv2.imwrite(str(out_dir / f"hole_shaft_depth_{stamp}.png"), depth)
                print(f"saved debug frame {stamp}", flush=True)
            payload = {
                "timestamp_unix": time.time(),
                "inference_timestamp_unix": inference_timestamp_unix,
                "camera_serial": args.serial,
                "source_frame": args.source_frame,
                "intrinsics": {**intrinsics, "width": args.width, "height": args.height},
                "hole_detection": hole_detection,
                "shaft_detection": shaft_detection,
                "shaft_rejection_reason": shaft_rejection_reason,
                "shaft_tracking": shaft_tracker.status(),
                "shaft_tracking_status": shaft_tracking_status,
                "ready_for_world_measurement": (
                    hole_detection is not None
                    and shaft_detection is not None
                    and shaft_tracker.state == "LOCKED"
                ),
                "positive_only_warning": True,
            }
            write_json_atomic(json_path, payload)
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
