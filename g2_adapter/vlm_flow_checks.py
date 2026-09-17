"""Non-blocking semantic checks over the bilateral Web JPEG cache."""

from __future__ import annotations

import time
import warnings
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


RESULT_EVENTS = {"gripper": "vlm_gripper_result", "pin": "vlm_pin_result", "c-hole": "vlm_c_hole_result"}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "vlm"


def _save_inputs(stage: str, images: dict[str, bytes], statuses: dict) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths = []
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        for camera, data in images.items():
            name, _, variant = camera.partition("_")
            suffix = f"_{variant}" if variant else ""
            path = OUTPUT_DIR / f"{timestamp}_{name}_{stage}{suffix}.jpg"
            with path.open("xb") as output:
                output.write(data)
            paths.append(str(path))
    except OSError as exc:
        statuses["image_save_error"] = str(exc)
        warnings.warn(f"[VLM 警告] 图片保存失败：{exc}", RuntimeWarning, stacklevel=2)
    statuses["saved_images"] = paths


def parse_crop_xyxy(value: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(part.strip()) for part in str(value).split(","))
    except ValueError as exc:
        raise ValueError("VLM crop must be x0,y0,x1,y1") from exc
    if len(result) != 4 or result[0] < 0 or result[1] < 0 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("VLM crop must be x0,y0,x1,y1 with positive area")
    return result


def crop_jpeg(jpeg: bytes, crop_xyxy: tuple[int, int, int, int]) -> bytes:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("VLM input is not a decodable JPEG")
    height, width = image.shape[:2]
    x0, y0, x1, y1 = crop_xyxy
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("VLM crop is outside the cached frame")
    ok, encoded = cv2.imencode(".jpg", image[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("failed to encode VLM gripper crop")
    return encoded.tobytes()


def _warning(task: str, error: Exception | str, statuses: dict) -> dict:
    return {"event": "vlm_check_warning", "task": task, "ok": False, "error": str(error), "input_status": statuses}


def _safe_check(client, task: str, images: list[bytes], statuses: dict, clock=time.monotonic) -> dict:
    started = clock()
    try:
        result = client.run_bytes(task, images)
        if task == "gripper" and result.get("state") not in {"attached", "not_attached", "dropped", "unknown"}:
            raise ValueError("invalid gripper VLM state")
        if task == "pin" and result.get("state") not in {"inserted", "not_inserted", "unknown"}:
            raise ValueError("invalid pin VLM state")
        if task == "c-hole" and result.get("obvious_c_leak") not in {True, False, None}:
            raise ValueError("invalid c-hole VLM result")
        return {"event": RESULT_EVENTS[task], "task": task, "ok": True, "elapsed_s": round(clock() - started, 3), "input_status": statuses, "result": result}
    except Exception as exc:
        record = _warning(task, exc, statuses)
        record["elapsed_s"] = round(clock() - started, 3)
        return record


def _snapshots(web, names: tuple[str, ...]):
    if web is None:
        raise RuntimeError("Web JPEG cache is disabled")
    deadline = time.monotonic() + 2.0
    while True:
        frames, statuses = {}, {}
        for name in names:
            frame, status = web.snapshot_raw_pane(name)
            statuses[name] = status
            if frame and status.get('raw_age_s', 0.0) <= 1.0:
                frames[name] = bytes(frame)
        if len(frames) == len(names):
            return frames, statuses
        if time.monotonic() >= deadline:
            missing = ', '.join(name for name in names if name not in frames)
            raise RuntimeError(f"Camera {missing} has no fresh original JPEG")
        time.sleep(.05)


def run_gripper_check(web, client, crop_xyxy=None, clock=time.monotonic) -> dict:
    try:
        frames, statuses = _snapshots(web, ("left",))
        if client is None:
            raise RuntimeError("VLM client is unavailable")
        images = [frames["left"]]
    except Exception as exc:
        return _warning("gripper", exc, locals().get("statuses", {}))
    _save_inputs("vlm_gripper_check", {"left": images[0]}, statuses)
    return _safe_check(client, "gripper", images, statuses, clock)


def run_assembly_checks(web, client, clock=time.monotonic) -> list[dict]:
    try:
        frames, statuses = _snapshots(web, ("head", "left", "right"))
        if client is None:
            raise RuntimeError("VLM client is unavailable")
    except Exception as exc:
        return [_warning("pin", exc, locals().get("statuses", {})), _warning("c-hole", exc, locals().get("statuses", {}))]
    _save_inputs("vlm_assembly_check", frames, statuses)
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="vlm-assembly") as pool:
        pin = pool.submit(_safe_check, client, "pin", [frames["right"], frames["left"], frames["head"]], statuses, clock)
        c_hole = pool.submit(_safe_check, client, "c-hole", [frames["left"], frames["right"]], statuses, clock)
        return [pin.result(), c_hole.result()]
