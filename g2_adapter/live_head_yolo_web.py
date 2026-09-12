"""Read-only head RGB YOLO segmentation preview; no robot-control API."""
import argparse
import threading
import time
from http.server import ThreadingHTTPServer

import cv2
import numpy as np

from camera import decode_color_rgb
from yolo_hinge import load_yolo_segmentation_model, infer_yolo_masks
from live_left_d405_sam3_web import SharedView, handler_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo-model", default="models/head_hinge_yolo/best.pt")
    parser.add_argument("--confidence", type=float, default=.50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8778)
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1 or args.imgsz <= 0:
        parser.error("invalid confidence or image size")
    model = load_yolo_segmentation_model(args.yolo_model, device=args.device)
    import agibot_gdk as gdk
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK initialization failed")
    camera = server = thread = None
    try:
        stream = gdk.CameraType.kHeadColor
        camera = gdk.Camera([stream])
        shared = SharedView()
        server = ThreadingHTTPServer((args.host, args.port), handler_class(shared, title="G2 头部相机 / YOLO（只读）"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"Head YOLO: http://127.0.0.1:{args.port}", flush=True)
        last_timestamp = None
        sequence = 0
        while True:
            frame = camera.get_latest_image(stream, 1000.0)
            if frame.timestamp_ns == last_timestamp:
                time.sleep(.01)
                continue
            last_timestamp = frame.timestamp_ns
            bgr = cv2.cvtColor(decode_color_rgb(frame, gdk), cv2.COLOR_RGB2BGR)
            start = time.monotonic()
            masks, scores, classes = infer_yolo_masks(model, bgr, confidence=args.confidence, imgsz=args.imgsz, device=args.device)
            detections = []
            for mask, score, cls in zip(masks, scores, classes):
                rows, cols = np.nonzero(mask)
                if not len(rows):
                    continue
                label = model.names[int(cls)]
                colour = (0, 220, 0) if int(cls) == 0 else (0, 180, 255)
                bgr[mask] = (.65 * bgr[mask] + .35 * np.array(colour)).astype(np.uint8)
                box = [int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max())]
                cv2.rectangle(bgr, tuple(box[:2]), tuple(box[2:]), colour, 2)
                cv2.putText(bgr, f"{label} {score:.2f}", (box[0], max(15, box[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .6, colour, 2)
                detections.append(dict(class_id=int(cls), label=label, confidence=float(score), bbox_xyxy=box))
            ok, jpeg = cv2.imencode(".jpg", bgr)
            if ok:
                sequence += 1
                shared.update(jpeg.tobytes(), dict(state="head_rgb_yolo", sequence=sequence, motion_enabled=False,
                    camera_timestamp_ns=int(frame.timestamp_ns), detections=detections,
                    inference_ms=round((time.monotonic()-start)*1000, 1)))
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
