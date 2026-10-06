"""Non-blocking semantic checks over the bilateral Web JPEG cache."""

from __future__ import annotations

import time
import warnings
import threading
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np


RESULT_EVENTS = {
    "gripper": "vlm_gripper_result",
    "pin": "vlm_pin_result",
    "c-hole": "vlm_c_hole_result",
    "convex-inserted": "vlm_convex_inserted_result",
    "four-screws": "vlm_four_screws_result",
    "right-fixture-grasp": "vlm_right_fixture_grasp_result",
    "hole-alignment": "vlm_hole_alignment_result",
}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "vlm"
_RUN_NAME = None
_RUN_LOCK = threading.RLock()


def start_vlm_run() -> Path:
    """Select one timestamp directory before any producers start saving."""
    global _RUN_NAME
    with _RUN_LOCK:
        now = datetime.now()
        name = now.strftime('%Y%m%d_%H%M%S_%f')
        while name == _RUN_NAME or (OUTPUT_DIR / name).exists():
            now += timedelta(microseconds=1)
            name = now.strftime('%Y%m%d_%H%M%S_%f')
        _RUN_NAME = name
        return OUTPUT_DIR / name


def _save_inputs(stage: str, images: dict[str, bytes], statuses: dict) -> None:
    with _RUN_LOCK:
        if _RUN_NAME is None:
            start_vlm_run()
        directory = OUTPUT_DIR / _RUN_NAME
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    paths = []
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for camera, data in images.items():
            name, _, variant = camera.partition("_")
            suffix = f"_{variant}" if variant else ""
            path = directory / f"{timestamp}_{name}_{stage}{suffix}.jpg"
            with path.open("xb") as output:
                output.write(data)
            paths.append(str(path))
    except OSError as exc:
        statuses["image_save_error"] = str(exc)
        warnings.warn(f"[VLM 警告] 图片保存失败：{exc}", RuntimeWarning, stacklevel=2)
    statuses["saved_images"] = paths
    statuses["image_directory"] = str(directory)


def parse_crop_xyxy(value: str) -> tuple[int, int, int, int]:
    try:
        result = tuple(int(part.strip()) for part in str(value).split(","))
    except ValueError as exc:
        raise ValueError("VLM crop must be x0,y0,x1,y1") from exc
    if len(result) != 4 or result[0] < 0 or result[1] < 0 or result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("VLM crop must be x0,y0,x1,y1 with positive area")
    return result


def central_gripper_crop(jpeg: bytes) -> bytes:
    return crop_jpeg(jpeg, None)


def crop_jpeg(jpeg: bytes, crop_xyxy: tuple[int, int, int, int] | None = None) -> bytes:
    image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("VLM input is not a decodable JPEG")
    height, width = image.shape[:2]
    if crop_xyxy is None:
        crop_width, crop_height = max(1, round(width * .4)), max(1, round(height * .6))
        x0, y0 = (width - crop_width) // 2, (height - crop_height) // 2
        x1, y1 = x0 + crop_width, y0 + crop_height
    else:
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
        if task == "convex-inserted" and result.get("decision") not in {"已插入", "未插入", "不确定", None}:
            raise ValueError("invalid convex-inserted VLM decision")
        if task == "four-screws" and result.get("latest_decision") not in {"yes", "no", "unknown", None}:
            raise ValueError("invalid four-screws VLM decision")
        if task == "right-fixture-grasp" and result.get("decision", result.get("latest_decision")) not in {"抓住", "未抓住", "不确定", "yes", "no", "unknown", None}:
            raise ValueError("invalid right-fixture-grasp VLM decision")
        return {"event": RESULT_EVENTS[task], "task": task, "ok": True, "elapsed_s": round(clock() - started, 3), "input_status": statuses, "result": result}
    except Exception as exc:
        record = _warning(task, exc, statuses)
        record["elapsed_s"] = round(clock() - started, 3)
        return record


def _snapshots(web, names: tuple[str, ...], *, not_before: float | None = None):
    if web is None:
        raise RuntimeError("Web JPEG cache is disabled")
    deadline = time.monotonic() + 2.0
    while True:
        frames, statuses = {}, {}
        for name in names:
            sampled_at = time.monotonic()
            frame, status = web.snapshot_raw_pane(name)
            statuses[name] = status
            age = status.get('raw_age_s', 0.0)
            if frame and age <= 1.0 and (not_before is None or sampled_at - age >= not_before):
                frames[name] = bytes(frame)
        if len(frames) == len(names):
            return frames, statuses
        if time.monotonic() >= deadline:
            missing = ', '.join(name for name in names if name not in frames)
            raise RuntimeError(f"Camera {missing} has no fresh original JPEG")
        time.sleep(.05)


def _snapshot_sequence(web, name: str, *, count: int = 4, interval_s: float = .25,
                       not_before: float | None = None):
    """Collect a short original-JPEG sequence without touching processed panes."""
    if count < 1:
        raise ValueError("snapshot sequence count must be positive")
    frames: list[bytes] = []
    statuses: dict = {}
    for index in range(count):
        current, current_status = _snapshots(web, (name,), not_before=not_before)
        frames.append(bytes(current[name]))
        statuses = current_status
        if index + 1 < count and interval_s > 0:
            time.sleep(interval_s)
    return frames, statuses


def _fixture_sequence_board_jpeg(frames: list[bytes]) -> bytes:
    """Build the V32 four-panel sequence while retaining raw frames on disk."""
    if len(frames) != 4:
        raise ValueError("fixture sequence board requires four frames")
    tiles = []
    for frame in frames:
        image = cv2.imdecode(np.frombuffer(frame, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("fixture sequence contains an undecodable JPEG")
        tiles.append(cv2.resize(image, (510, 428), interpolation=cv2.INTER_AREA))
    board = np.zeros((856, 1020, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        x = (index % 2) * 510
        y = (index // 2) * 428
        board[y:y + 428, x:x + 510] = tile
        cv2.putText(
            board,
            f"frame {index + 1}",
            (x + 12, y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            .7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".jpg", board, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("failed to encode right-fixture sequence board")
    return encoded.tobytes()


def run_gripper_check(web, client, crop_xyxy=None, clock=time.monotonic, *, reference=None, frames_override=None) -> dict:
    try:
        frames, statuses = frames_override if frames_override is not None else _snapshots(web, ("left",))
        if client is None:
            raise RuntimeError("VLM client is unavailable")
        from vlm_checks.src.migrated_checks import gripper_roi
        crop = gripper_roi(frames['left'])
        images = [frames["left"]]
        age = statuses.get('left', {}).get('raw_age_s', 0.)
        statuses['image_time'] = datetime.fromtimestamp(time.time() - age).astimezone().isoformat(timespec='milliseconds')
    except Exception as exc:
        return _warning("gripper", exc, locals().get("statuses", {}))
    saved = {'left': images[0], 'left_crop': crop}
    _save_inputs("vlm_gripper_check", saved, statuses)
    return _safe_check(client, "gripper", images, statuses, clock)


def run_assembly_checks(web, client, clock=time.monotonic, *, include_c_hole: bool = True) -> list[dict]:
    try:
        frames, statuses = _snapshots(web, ("head", "left", "right"))
        if client is None:
            raise RuntimeError("VLM client is unavailable")
    except Exception as exc:
        tasks = ("pin", "c-hole") if include_c_hole else ("pin",)
        return [_warning(task, exc, locals().get("statuses", {})) for task in tasks]
    _save_inputs("vlm_assembly_check", frames, statuses)
    with ThreadPoolExecutor(max_workers=2 if include_c_hole else 1, thread_name_prefix="vlm-assembly") as pool:
        pin = pool.submit(_safe_check, client, "pin", [frames["right"], frames["left"], frames["head"]], statuses, clock)
        if include_c_hole:
            c_hole = pool.submit(_safe_check, client, "c-hole", [frames["left"], frames["right"]], statuses, clock)
            return [pin.result(), c_hole.result()]
        return [pin.result()]


def run_c_hole_check(web, client, clock=time.monotonic) -> dict:
    """Check the visible C-hole area using fresh, unprocessed left/right frames."""
    try:
        frames, statuses = _snapshots(web, ("left", "right"))
        if client is None:
            raise RuntimeError("VLM client is unavailable")
    except Exception as exc:
        return _warning("c-hole", exc, locals().get("statuses", {}))
    _save_inputs("vlm_c_hole_check", frames, statuses)
    return _safe_check(client, "c-hole", [frames["left"], frames["right"]], statuses, clock)


def run_convex_insert_check(web, client, clock=time.monotonic, *, frames_override=None) -> dict:
    """Read-only strict convex-shaft insertion check using the original right view."""
    try:
        if frames_override is None:
            frames, statuses = _snapshots(web, ("right",))
            images = [frames["right"]]
            saved = {"right": images[0]}
        else:
            images = [bytes(frames_override[-1])]
            statuses = {"right": {"raw_age_s": 0.0}}
            saved = {"right": images[0]}
        if client is None:
            raise RuntimeError("VLM client is unavailable")
    except Exception as exc:
        return _warning("convex-inserted", exc, locals().get("statuses", {}))
    _save_inputs("vlm_convex_insert_check", saved, statuses)
    return _safe_check(client, "convex-inserted", images, statuses, clock)


def run_four_screws_check(web, client, clock=time.monotonic, *, frames_override=None) -> dict:
    """Read-only four-screw check over four fresh original right-camera frames."""
    try:
        if frames_override is None:
            images, statuses = _snapshot_sequence(web, "right", not_before=time.monotonic())
        else:
            images = [bytes(frame) for frame in frames_override]
            statuses = {"right": {"raw_age_s": 0.0}}
        if len(images) != 4:
            raise ValueError("four-screws VLM check requires four frames")
        if client is None:
            raise RuntimeError("VLM client is unavailable")
    except Exception as exc:
        return _warning("four-screws", exc, locals().get("statuses", {}))
    _save_inputs(
        "vlm_four_screws_check",
        {f"right_frame{index}": frame for index, frame in enumerate(images, 1)},
        statuses,
    )
    return _safe_check(client, "four-screws", images, statuses, clock)


def run_right_fixture_grasp_check(web, client, clock=time.monotonic, *, frames_override=None) -> dict:
    """V47: four original frames spaced 0.25 seconds, contact ROI and V32/V40."""
    try:
        if frames_override is None:
            images, statuses = _snapshot_sequence(web, "right", interval_s=0.25)
        else:
            images = [bytes(frame) for frame in frames_override]
            statuses = {"right": {"raw_age_s": 0.0}}
        if len(images) != 4:
            raise ValueError("right-fixture-grasp VLM check requires four frames")
        if client is None:
            raise RuntimeError("VLM client is unavailable")
        from vlm_checks.src.migrated_checks import fixture_board
        sequence_board = fixture_board(images)
    except Exception as exc:
        return _warning("right-fixture-grasp", exc, locals().get("statuses", {}))
    _save_inputs(
        "vlm_right_fixture_grasp_check",
        {
            **{f"right_frame{index}": frame for index, frame in enumerate(images, 1)},
            "right_sequence_board": sequence_board,
        },
        statuses,
    )
    return _safe_check(client, "right-fixture-grasp", images, statuses, clock)


def run_hole_alignment_review(web, client, emit, *, sleep=time.sleep):
    """Three fresh right-camera ROI judgments followed by a conservative majority."""
    from vlm_checks.src.migrated_checks import alignment_roi, alignment_vote
    emit({"event": "vlm_check_start", "task": "hole-alignment"})
    records, results = [], []
    for index in range(1, 4):
        if index > 1:
            sleep(1.)
        statuses = {}
        try:
            frames, statuses = _snapshots(web, ("right",))
            crop = alignment_roi(frames["right"])
            _save_inputs("vlm_hole_alignment_check", {"right": frames["right"], "right_crop": crop}, statuses)
            record = _safe_check(client, "hole-alignment", [frames["right"]], statuses)
        except Exception as exc:
            record = _warning("hole-alignment", exc, statuses)
        record.update(review_round=index, review_total=3)
        records.append(record)
        results.append(record.get('result', {'decision': '不确定'}))
        emit(record)
    summary = {"event": RESULT_EVENTS['hole-alignment'], "task": 'hole-alignment',
               "ok": any(record.get('ok') for record in records),
               "aggregate": True, "result": alignment_vote(results)}
    emit(summary)
    return [*records, summary]


def run_assembly_review(
    web, client, emit, *, sleep=time.sleep, include_convex: bool = False, include_c_hole: bool = True,
    include_pin: bool = True,
):
    if include_convex:
        emit({"event": "vlm_check_start", "task": "convex-inserted"})
    records = []
    convex_votes = []
    for round_index in range(1,4):
        if round_index > 1:
            sleep(1.)
        legacy = (run_assembly_checks(web, client, include_c_hole=include_c_hole)
                  if include_pin else ([run_c_hole_check(web, client)] if include_c_hole else []))
        for record in legacy:
            record = dict(record)
            record['review_round'] = round_index
            records.append(record)
            emit(record)
        if include_convex:
            record = dict(run_convex_insert_check(web, client))
            record['review_round'] = round_index
            records.append(record)
            emit(record)
            convex_votes.append(record.get('result', {}).get('decision') if record.get('ok') else None)
    if include_convex:
        decision = ('已插入' if convex_votes.count('已插入') >= 2 else
                    '未插入' if convex_votes.count('未插入') >= 2 else '不确定')
        summary = {'event': RESULT_EVENTS['convex-inserted'], 'task': 'convex-inserted',
                   'ok': any(vote is not None for vote in convex_votes),
                   'aggregate': True, 'result': {'decision': decision}}
        records.append(summary)
        emit(summary)
    return records


def run_c_hole_review(web, client, emit, *, sleep=time.sleep):
    """Repeat the read-only C-hole check three times after the operator stage."""
    records = []
    for round_index in range(1, 4):
        if round_index > 1:
            sleep(1.)
        record = dict(run_c_hole_check(web, client))
        record['review_round'] = round_index
        records.append(record)
        emit(record)
    return records
