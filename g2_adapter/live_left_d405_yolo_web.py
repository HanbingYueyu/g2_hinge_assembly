#!/usr/bin/env python3
"""Read-only left D405 RGB YOLO preview; no robot, gripper, depth, or SAM."""

from __future__ import annotations

import argparse
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

from camera import decode_color_rgb
from live_left_d405_sam3_web import SharedView, handler_class
from sam_crop import center_crop_bounds
from yolo_hinge import infer_yolo_masks, load_yolo_segmentation_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-model",
        default="models/left_d405_hinge_yolo/best.pt",
    )
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=0.70,
        help="YOLO 全图推理时的中央有效目标区域宽高比例，默认 0.70。",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8775)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or args.imgsz <= 0 or not 0.0 < args.center_fraction <= 1.0:
        parser.error("invalid confidence or image size")

    model = load_yolo_segmentation_model(args.yolo_model, device=args.device)
    import agibot_gdk as gdk

    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK initialization failed")

    camera = server = thread = None
    try:
        stream = gdk.CameraType.kHandLeftColor
        camera = gdk.Camera([stream])
        shared = SharedView()
        server = ThreadingHTTPServer(
            (args.host, args.port),
            handler_class(shared, title="G2 左手 D405 / YOLO（只读）"),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"Left D405 YOLO: http://127.0.0.1:{args.port}", flush=True)
        print("只读模式：不加载 SAM、不访问机器人和夹爪、不发送运动命令。", flush=True)

        last_timestamp = None
        sequence = 0
        while True:
            frame = camera.get_latest_image(stream, 1000.0)
            if frame.timestamp_ns == last_timestamp:
                time.sleep(0.01)
                continue
            last_timestamp = frame.timestamp_ns
            bgr = cv2.cvtColor(decode_color_rgb(frame, gdk), cv2.COLOR_RGB2BGR)
            started = time.monotonic()
            valid_roi = center_crop_bounds(bgr.shape, fraction=args.center_fraction)
            masks, scores, classes = infer_yolo_masks(
                model,
                bgr,
                confidence=args.confidence,
                imgsz=args.imgsz,
                device=args.device,
            )
            detections = []
            x0, y0, x1, y1 = valid_roi
            for mask, score, cls in zip(masks, scores, classes):
                rows, cols = np.nonzero(mask)
                if not len(rows):
                    continue
                class_id = int(cls)
                label = model.names[class_id]
                center_x, center_y = float(cols.mean()), float(rows.mean())
                valid_target = x0 <= center_x <= x1 and y0 <= center_y <= y1
                colour = (0, 220, 0) if class_id == 0 and valid_target else (0, 180, 255)
                if not valid_target:
                    colour = (120, 120, 120)
                bgr[mask] = (
                    0.65 * bgr[mask] + 0.35 * np.asarray(colour)
                ).astype(np.uint8)
                box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
                cv2.rectangle(bgr, tuple(box[:2]), tuple(box[2:]), colour, 2)
                cv2.putText(
                    bgr,
                    f"{label} {float(score):.2f}"
                    + ("" if valid_target else " outside valid ROI"),
                    (box[0], max(15, box[1] - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    colour,
                    2,
                )
                detections.append(
                    {
                        "class_id": class_id,
                        "label": label,
                        "confidence": float(score),
                        "bbox_xyxy": box,
                        "valid_target": bool(valid_target),
                    }
                )

            cv2.rectangle(bgr, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2)
            cv2.putText(
                bgr,
                f"YOLO valid target region {args.center_fraction:.0%}",
                (x0 + 8, max(y0 + 24, 24)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            ok, jpeg = cv2.imencode(".jpg", bgr)
            if ok:
                sequence += 1
                shared.update(
                    jpeg.tobytes(),
                    {
                        "state": "left_d405_yolo",
                        "sequence": sequence,
                        "motion_enabled": False,
                        "camera_timestamp_ns": int(frame.timestamp_ns),
                        "detections": detections,
                        "yolo_valid_roi_xyxy": list(valid_roi),
                        "yolo_valid_region_fraction": args.center_fraction,
                        "inference_ms": round((time.monotonic() - started) * 1000, 1),
                    },
                )
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
