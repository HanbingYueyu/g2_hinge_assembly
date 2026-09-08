#!/usr/bin/env python3
"""Preview YOLO + STEP hole-pattern PnP using only the left colour camera."""
import argparse
import json
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

try:
    from .left_step_rgb import load_profile, detect_hole_centres, match_holes, OcclusionTrackerPool
    from .yolo_hinge import load_yolo_segmentation_model, infer_yolo_masks
except ImportError:
    from left_step_rgb import load_profile, detect_hole_centres, match_holes, OcclusionTrackerPool
    from yolo_hinge import load_yolo_segmentation_model, infer_yolo_masks


def analyse(bgr, model, profile, intrinsic, device, trackers=None):
    masks, scores, classes = infer_yolo_masks(model, bgr, confidence=.25, imgsz=640, device=device)
    source = bgr.copy()
    eligible = [i for i, cls in enumerate(classes) if int(cls) == 0]
    holes_by_index = {i: detect_hole_centres(source, masks[i]) for i in eligible}
    tracked = {}
    if trackers is not None:
        centres = [np.argwhere(masks[i])[:, ::-1].mean(axis=0) for i in eligible]
        tracked = dict(zip(eligible, trackers.update(profile, [holes_by_index[i] for i in eligible], centres, intrinsic)))
    records = []
    for index, (mask, score, class_id) in enumerate(zip(masks, scores, classes)):
        if int(class_id) != 0:
            continue
        holes = holes_by_index[index]
        result = dict(tracked[index]) if index in tracked else match_holes(profile, holes, intrinsic)
        result["yolo_confidence"] = float(score)
        records.append(result)
        for point in holes:
            cv2.circle(bgr, tuple(np.round(point).astype(int)), 4, (0, 255, 255), 1)
        if result["accepted"]:
            for index, point in zip(result["matched_model_indices"], result["matched_pixels"]):
                cv2.putText(bgr, str(index), tuple(np.round(point).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0), 1)
            point = tuple(np.round(result["grasp_pixel_px"]).astype(int))
            cv2.drawMarker(bgr, point, (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
    return {"state": "rgb_step_preview", "motion_enabled": False, "depth_used": False,
            "geometry_verified": bool(profile.get("verified")), "detections": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--save-reference", type=Path, help="Capture 20 stationary RGB frames after manually aligning the open gripper; no motion")
    parser.add_argument("--yolo-model", default="models/left_d405_hinge_shaft_yolo/best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image", type=Path, help="Offline RGB image instead of GDK")
    parser.add_argument("--intrinsics", type=Path, help="Required for offline images; fx,fy,ppx,ppy,width,height,distortion")
    parser.add_argument("--output", type=Path, default=Path("outputs/left_step/rgb_preview.jpg"))
    parser.add_argument("--web-host", default="0.0.0.0")
    parser.add_argument("--web-port", type=int, default=8777)
    args = parser.parse_args()
    profile = load_profile(args.profile)
    if args.image and not args.intrinsics:
        parser.error("--image requires --intrinsics")
    if args.image and args.save_reference:
        parser.error("reference capture requires live camera")
    model = load_yolo_segmentation_model(args.yolo_model, device=args.device)
    if args.image:
        bgr = cv2.imread(str(args.image))
        if bgr is None:
            parser.error("cannot read image")
        intrinsic = json.loads(args.intrinsics.read_text())
        if (intrinsic["width"], intrinsic["height"]) != (bgr.shape[1], bgr.shape[0]):
            parser.error("image size differs from intrinsics")
        record = analyse(bgr, model, profile, intrinsic, args.device)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), bgr):
            raise RuntimeError("cannot write preview")
        print(json.dumps(record, ensure_ascii=False))
        return
    import agibot_gdk as gdk
    try:
        from .camera import decode_color_rgb, intrinsic_to_dict
        from .live_left_d405_sam3_web import SharedView, handler_class
    except ImportError:
        from camera import decode_color_rgb, intrinsic_to_dict
        from live_left_d405_sam3_web import SharedView, handler_class
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK initialization failed")
    camera = server = None
    thread = None
    try:
        camera_type = gdk.CameraType.kHandLeftColor
        camera = gdk.Camera([camera_type])
        shared = SharedView()
        server = ThreadingHTTPServer((args.web_host, args.web_port), handler_class(shared))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"RGB STEP preview: http://127.0.0.1:{args.web_port}", flush=True)
        previous_timestamp = None
        samples = []
        trackers = OcclusionTrackerPool()
        while True:
            frame = camera.get_latest_image(camera_type, 1000.0)
            if frame.timestamp_ns == previous_timestamp:
                time.sleep(.01)
                continue
            previous_timestamp = frame.timestamp_ns
            intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(camera_type), width=int(frame.width), height=int(frame.height))
            bgr = cv2.cvtColor(decode_color_rgb(frame, gdk), cv2.COLOR_RGB2BGR)
            record = analyse(bgr, model, profile, intrinsic, args.device, trackers=None if args.save_reference else trackers)
            if args.save_reference:
                detections = record["detections"]
                # Never silently select another hinge when the intended one is missing.
                if len(detections) == 1 and detections[0].get("accepted"):
                    point = np.asarray(detections[0]["grasp_pixel_px"])
                    if samples and np.linalg.norm(point - np.mean(samples, axis=0)) > 2:
                        samples = []
                    samples.append(point)
                else:
                    samples = []
                record["reference_samples"] = len(samples)
                if len(samples) >= 20:
                    reference = {"target_pixel_px": np.median(samples, axis=0).tolist(),
                                 "sample_count": len(samples), "intrinsic": intrinsic,
                                 "source_sha256": profile["source_sha256"],
                                 "grasp_center_m": profile["grasp_center_m"]}
                    args.save_reference.parent.mkdir(parents=True, exist_ok=True)
                    args.save_reference.write_text(json.dumps(reference, indent=2) + "\n")
                    print(f"Saved STEP alignment reference: {args.save_reference}", flush=True)
                    break
            record["camera_timestamp_ns"] = int(frame.timestamp_ns)
            ok, jpeg = cv2.imencode(".jpg", bgr)
            if ok:
                shared.update(jpeg.tobytes(), record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if thread:
            thread.join(timeout=2)
        if camera:
            camera.close_camera()
        gdk.gdk_release()


if __name__ == "__main__":
    main()
