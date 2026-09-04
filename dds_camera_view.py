#!/usr/bin/env python3
"""
直接订阅 cosine camera topic，并通过 Flask 提供多相机 Web 预览。

这个脚本不导入 agibot_gdk。它使用机器上已有的
libcosine_bus_ffi.so 订阅 CosineDataPacket：

默认按 camera_device_map.json 的 current_product 订阅该机型全部相机。
可用 --product 切换 G2 / G2 Air / G2 Max。网页端默认勾选全部相机，
也可手动取消；/get_all_images 只返回选中的图。

支持网页一键录制：通过 DDS `/dlb_msgs/record_video_request`
交给 camera_dlb 落盘（H265/ZST，非 OpenCV 软编，不卡顿）。

彩色相机发布的 JPEG 不做解码/重编码，直接转为 base64。Z16 深度图
使用 OpenCV 转成伪彩色 JPEG。Web 接口和 test/camera_view.py 保持兼容：

    GET /
    GET /get_available_cameras
    GET /get_all_images
    POST /record/start
    POST /record/stop
    GET /record/status
"""

from __future__ import annotations

import argparse
import atexit
import base64
import ctypes
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple


APP_DIR = os.environ.get("COSINE_APP_DIR", "/home/agi/app")
SYSTEM_DIST_PACKAGES = "/usr/lib/python3/dist-packages"
DDS_PUB = os.environ.get("DDS_PUB", os.path.join(APP_DIR, "bin", "dds_pub"))
DDS_ECHO = os.environ.get("DDS_ECHO", os.path.join(APP_DIR, "bin", "dds_echo"))
DDS_RECORD_TOPIC = "/dlb_msgs/record_video_request"
DDS_RECORD_TYPE = "dlb_msgs.RecordRequest"
DDS_RECORD_RESPONSE_TOPIC = "/dlb_msgs/record_video_response"
DDS_RECORD_RESPONSE_TYPE = "dlb_msgs.RecordResponse"
# camera_dlb 单次 START 的最长录制窗口（毫秒）；点停止会提前 STOP
DEFAULT_DDS_POST_WIN_MS = 1_800_000
DATA_RECORD_ROOT = os.environ.get("DDS_RECORD_ROOT", "/data/record")

# camera_dlb/VPU 对 G300 腕部 848x530 的 H265 花屏；这些 topic 改从 DDS JPEG 旁路录 MP4
HAND_JPEG_FALLBACK_TOPICS = {
    "/camera/hand_left_color",
    "/camera/hand_right_color",
}

from camera_map import product_ids, viewer_cameras  # noqa: E402

DEFAULT_PRODUCT = os.environ.get("CAMERA_PRODUCT", "g2_thor")
DEFAULT_CAMERAS: Sequence[Tuple[str, str]] = viewer_cameras(DEFAULT_PRODUCT)


def bootstrap_runtime() -> None:
    """补齐 cosine 动态库路径和 Aorta discovery 环境，然后重新执行自身。"""
    lib_root = os.path.join(APP_DIR, "lib")
    if not os.path.isdir(lib_root):
        raise RuntimeError(f"cosine lib directory does not exist: {lib_root}")

    lib_dirs = [lib_root]
    for name in sorted(os.listdir(lib_root)):
        path = os.path.join(lib_root, name)
        if os.path.isdir(path):
            lib_dirs.append(path)

    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_dirs = [p for p in current.split(":") if p]
    missing = [p for p in lib_dirs if p not in current_dirs]

    env_defaults = {
        "LOCATOR_IP": "10.42.1.101",
        "AORTA_DISCOVERY_URI": "http://10.42.1.101:2379",
        "AORTA_URI": "http://10.42.1.101",
        "COSINE_BUS_DEFAULT_MIDDLEWARE": "aorta",
    }
    env_needs_update = any(not os.environ.get(key) for key in env_defaults)

    if (missing or env_needs_update) and os.environ.get("_COSINE_FFI_REEXEC") != "1":
        env = os.environ.copy()
        env["_COSINE_FFI_REEXEC"] = "1"
        env["LD_LIBRARY_PATH"] = ":".join(lib_dirs + current_dirs)
        for key, value in env_defaults.items():
            env.setdefault(key, value)
        os.execve(sys.executable, [sys.executable, *sys.argv], env)

    # Ubuntu 的 cv2 通常来自 dist-packages。将系统包放在前面，避免
    # /usr/local 下 NumPy 2.x 与系统 OpenCV 的 NumPy 1.x ABI 冲突。
    if SYSTEM_DIST_PACKAGES in sys.path:
        sys.path.remove(SYSTEM_DIST_PACKAGES)
    sys.path.insert(0, SYSTEM_DIST_PACKAGES)


bootstrap_runtime()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from flask import Flask, jsonify, render_template_string, request  # noqa: E402


class CosineQos(ctypes.Structure):
    _fields_ = [
        ("reliability", ctypes.c_uint32),
        ("history", ctypes.c_uint32),
        ("depth", ctypes.c_int32),
        ("durability", ctypes.c_uint32),
        ("frame_skip_rate", ctypes.c_uint32),
    ]


class CosineDataPacket(ctypes.Structure):
    _fields_ = [
        ("send_timestamp", ctypes.c_uint64),
        ("recv_timestamp", ctypes.c_uint64),
        ("source_timestamp", ctypes.c_uint64),
        ("sequence_number", ctypes.c_uint64),
        ("type", ctypes.c_uint32),
        ("width", ctypes.c_int32),
        ("height", ctypes.c_int32),
        ("bit_depth", ctypes.c_int32),
        ("encoding", ctypes.c_uint32),
        ("color_format", ctypes.c_uint32),
        ("tensor_dtype", ctypes.c_uint32),
        ("tensor_ndim", ctypes.c_uint32),
        ("tensor_shape", ctypes.POINTER(ctypes.c_uint64)),
        ("proto_type_name", ctypes.c_char_p),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
        ("data_size", ctypes.c_uint32),
    ]


PACKET_CALLBACK = ctypes.CFUNCTYPE(
    None, ctypes.POINTER(CosineDataPacket), ctypes.c_void_p
)


@dataclass(frozen=True)
class Frame:
    payload: bytes
    width: int
    height: int
    bit_depth: int
    encoding: int
    color_format: int
    source_timestamp: int
    sequence_number: int
    received_monotonic: float
    received_wall_time: float


class CameraState:
    def __init__(self, index: int, topic: str, name: str) -> None:
        self.index = index
        self.topic = topic
        self.name = name
        self.lock = threading.Lock()
        self.encode_lock = threading.Lock()
        self.frame: Optional[Frame] = None
        self.receive_count = 0
        self.arrival_times: deque[float] = deque()
        self.cached_key: Optional[Tuple[int, int, int]] = None
        self.cached_data_url: Optional[str] = None
        self.cached_format = ""
        self.last_error = ""
        self.last_error_log = 0.0

    def update(self, frame: Frame) -> None:
        now = frame.received_monotonic
        with self.lock:
            self.frame = frame
            self.receive_count += 1
            self.arrival_times.append(now)
            cutoff = now - 2.0
            while self.arrival_times and self.arrival_times[0] < cutoff:
                self.arrival_times.popleft()
            self.cached_key = None
            self.cached_data_url = None
            self.cached_format = ""

    def set_error(self, text: str) -> None:
        with self.lock:
            self.last_error = text

    def snapshot(self) -> Tuple[Optional[Frame], int, float, str]:
        with self.lock:
            fps = 0.0
            if len(self.arrival_times) >= 2:
                duration = self.arrival_times[-1] - self.arrival_times[0]
                if duration > 0:
                    fps = (len(self.arrival_times) - 1) / duration
            return self.frame, self.receive_count, fps, self.last_error

    @staticmethod
    def _frame_key(frame: Frame) -> Tuple[int, int, int]:
        # 某些 publisher 不填写 sequence_number，timestamp + 数据长度作为补充。
        return frame.sequence_number, frame.source_timestamp, len(frame.payload)

    def data_url(self) -> Tuple[str, str]:
        with self.encode_lock:
            frame, _, _, _ = self.snapshot()
            if frame is None:
                raise RuntimeError("尚未收到图像")

            key = self._frame_key(frame)
            with self.lock:
                if self.cached_key == key and self.cached_data_url:
                    return self.cached_data_url, self.cached_format

            data_url, image_format = encode_frame(frame)

            # 编码期间可能又收到新帧，只缓存与当前帧仍一致的结果。
            with self.lock:
                if self.frame is not None and self._frame_key(self.frame) == key:
                    self.cached_key = key
                    self.cached_data_url = data_url
                    self.cached_format = image_format
            return data_url, image_format


def bytes_to_data_url(payload: bytes, mime: str) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def z16_plane_from_frame(frame: Frame) -> np.ndarray:
    """Unpack Z16 depth, honoring V4L row padding (bytesperline > width*2)."""
    height, width = frame.height, frame.width
    nbytes = len(frame.payload)
    if height <= 0 or nbytes < height * 2:
        raise ValueError(
            f"无效的 Z16 图像: {width}x{height}, bytes={nbytes}"
        )

    row_bytes = nbytes // height
    if row_bytes % 2 != 0:
        raise ValueError(
            f"无效的 Z16 stride: row_bytes={row_bytes} for {width}x{height}"
        )

    stride_w = row_bytes // 2
    plane = np.frombuffer(
        frame.payload[: row_bytes * height], dtype=np.uint16
    ).reshape(height, stride_w)
    # Padded rows: stride_w >= declared width. Tight rows: stride_w < width.
    if stride_w >= width:
        return plane[:, :width]
    return plane


def depth_to_bgr(frame: Frame) -> np.ndarray:
    """Z16 → pseudo-color; map by this frame's valid min/max only (no fixed range)."""
    depth = z16_plane_from_frame(frame)
    # 0 / 65535 are typical invalid/saturated markers on these streams.
    valid = (depth > 0) & (depth < np.iinfo(np.uint16).max)
    colored = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)

    if np.any(valid):
        # Light denoise for preview only; fill holes so 0s don't bleed in.
        filled = depth.copy()
        fill_v = int(np.median(depth[valid]))
        filled[~valid] = fill_v
        depth_f = cv2.medianBlur(filled, 3).astype(np.float32)

        values = depth_f[valid]
        lo = float(values.min())
        hi = float(values.max())
        if hi <= lo:
            hi = lo + 1.0
        normalized = (np.clip(depth_f, lo, hi) - lo) * (255.0 / (hi - lo))
        normalized = np.clip(normalized, 0, 255).astype(np.uint8)
        normalized[~valid] = 0
        cmap = getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(normalized, cmap)
        colored[~valid] = 0
        text = (
            f"Z16 {int(depth[valid].min())}-{int(depth[valid].max())} mm "
            f"fill {100.0 * np.mean(valid):.1f}%"
        )
    else:
        text = "Z16 no valid depth"

    cv2.putText(
        colored,
        text,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return colored


def depth_to_jpeg(frame: Frame) -> bytes:
    colored = depth_to_bgr(frame)
    ok, buffer = cv2.imencode(
        ".jpg",
        colored,
        [cv2.IMWRITE_JPEG_QUALITY, 85, cv2.IMWRITE_JPEG_OPTIMIZE, 1],
    )
    if not ok:
        raise RuntimeError("深度图 JPEG 编码失败")
    return buffer.tobytes()


def _payload_looks_like_z16(frame: Frame) -> bool:
    """Heuristic: Z16 is uint16 rows; payload length is a multiple of height."""
    height, width = frame.height, frame.width
    if height <= 0 or width <= 0:
        return False
    nbytes = len(frame.payload)
    if nbytes < height * width:
        return False
    row_bytes = nbytes // height
    return row_bytes >= width and row_bytes % 2 == 0


def raw_image_to_jpeg(frame: Frame) -> bytes:
    pixels = frame.width * frame.height
    if frame.width <= 0 or frame.height <= 0:
        raise ValueError(f"无效图像尺寸: {frame.width}x{frame.height}")

    if len(frame.payload) >= pixels * 3:
        image = np.frombuffer(frame.payload[: pixels * 3], dtype=np.uint8).reshape(
            frame.height, frame.width, 3
        )
        # CosineDataPacket 的彩色原始格式通常标记为 RGB。
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif len(frame.payload) >= pixels:
        image = np.frombuffer(frame.payload[:pixels], dtype=np.uint8).reshape(
            frame.height, frame.width
        )
    else:
        raise ValueError(
            f"无法识别的未压缩图像: {frame.width}x{frame.height}, "
            f"bytes={len(frame.payload)}"
        )

    ok, buffer = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80]
    )
    if not ok:
        raise RuntimeError("原始图像 JPEG 编码失败")
    return buffer.tobytes()


def encode_frame(frame: Frame) -> Tuple[str, str]:
    payload = frame.payload
    if payload.startswith(b"\xff\xd8"):
        return bytes_to_data_url(payload, "image/jpeg"), "JPEG"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        return bytes_to_data_url(payload, "image/png"), "PNG"

    if frame.bit_depth == 16 or _payload_looks_like_z16(frame):
        jpeg = depth_to_jpeg(frame)
        return bytes_to_data_url(jpeg, "image/jpeg"), "Z16→JPEG"

    jpeg = raw_image_to_jpeg(frame)
    return bytes_to_data_url(jpeg, "image/jpeg"), "RAW→JPEG"


def frame_to_bgr(frame: Frame) -> Tuple[np.ndarray, str]:
    """Decode a Cosine frame to BGR uint8 for VideoWriter."""
    payload = frame.payload
    if payload.startswith(b"\xff\xd8"):
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("JPEG 解码失败")
        return image, "JPEG"
    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("PNG 解码失败")
        return image, "PNG"
    if frame.bit_depth == 16 or _payload_looks_like_z16(frame):
        return depth_to_bgr(frame), "Z16→BGR"
    jpeg = raw_image_to_jpeg(frame)
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("RAW→JPEG 解码失败")
    return image, "RAW→BGR"


def safe_record_stem(topic: str, name: str) -> str:
    base = topic.strip("/").replace("/", "_") or name
    base = re.sub(r"[^A-Za-z0-9_.\-]+", "_", base).strip("._")
    return base or "camera"


def topic_camera_key(topic: str) -> str:
    """`/camera/hand_left_color` → `hand_left_color`."""
    parts = [p for p in topic.strip("/").split("/") if p]
    return parts[-1] if parts else safe_record_stem(topic, "camera")


@dataclass
class DdsCamRecord:
    topic: str
    name: str
    key: str
    storage_folder: str
    frames: int = 0
    files: List[str] = field(default_factory=list)
    last_error: str = ""
    jpeg_mp4: str = ""
    jpeg_frames: int = 0
    source: str = "dds_h265"  # dds_h265 | jpeg_fallback


@dataclass
class JpegFallbackRecorder:
    topic: str
    name: str
    key: str
    path: str
    writer: Optional[Any] = None
    width: int = 0
    height: int = 0
    frames: int = 0
    last_key: Optional[Tuple[int, int, int]] = None
    last_bgr: Optional[Any] = None
    last_error: str = ""


class CosineCameraViewer:
    MAX_PACKET_BYTES = 64 * 1024 * 1024

    def __init__(
        self,
        cameras: Sequence[Tuple[str, str]],
        host: str,
        port: int,
        record_dir: Optional[str] = None,
    ) -> None:
        self.host = host
        self.port = port
        self.record_root = os.path.abspath(
            record_dir
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "recordings")
        )
        self.states = [
            CameraState(index, topic, name)
            for index, (topic, name) in enumerate(cameras)
        ]
        self.state_by_topic: Dict[str, CameraState] = {
            state.topic: state for state in self.states
        }
        self.lib: Optional[ctypes.CDLL] = None
        self.subscribers: List[ctypes.c_void_p] = []
        self.callbacks: List[PACKET_CALLBACK] = []
        self.callback_holders: List[dict] = []
        self.initialized = False
        self.cleanup_lock = threading.Lock()
        self.cleaned = False
        self._record_lock = threading.RLock()
        self._record_active = False
        self._record_dir = ""
        self._record_meta_path = ""
        self._record_started_at = ""
        self._record_uuid = ""
        self._record_post_win_ms = DEFAULT_DDS_POST_WIN_MS
        self._record_mono_start = 0.0
        self._record_props: List[Dict[str, Any]] = []
        self._recorders: Dict[str, DdsCamRecord] = {}
        self._jpeg_recorders: Dict[str, JpegFallbackRecorder] = {}
        self._jpeg_stop = threading.Event()
        self._jpeg_thread: Optional[threading.Thread] = None
        self._jpeg_fps = 30.0
        self.app = Flask(__name__)
        self.app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
        self.app.json.ensure_ascii = False
        self._setup_routes()

    def _load_library(self) -> None:
        path = os.path.join(APP_DIR, "lib", "libcosine_bus_ffi.so")
        self.lib = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        self.lib.cosine_bus_initialize.argtypes = []
        self.lib.cosine_bus_initialize.restype = ctypes.c_bool
        self.lib.cosine_bus_release.argtypes = []
        self.lib.cosine_bus_release.restype = None
        self.lib.cosine_create_subscriber.argtypes = [
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.POINTER(CosineQos),
            PACKET_CALLBACK,
            ctypes.c_void_p,
        ]
        self.lib.cosine_create_subscriber.restype = ctypes.c_void_p
        self.lib.cosine_destroy_subscriber.argtypes = [ctypes.c_void_p]
        self.lib.cosine_destroy_subscriber.restype = None
        self.lib.cosine_free_data_packet_from_subscriber.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(CosineDataPacket),
        ]
        self.lib.cosine_free_data_packet_from_subscriber.restype = ctypes.c_bool

    def _make_callback(self, state: CameraState, holder: dict) -> PACKET_CALLBACK:
        @PACKET_CALLBACK
        def on_packet(
            packet_ptr: ctypes.POINTER(CosineDataPacket), _user_data: ctypes.c_void_p
        ) -> None:
            try:
                if not packet_ptr:
                    return
                packet = packet_ptr.contents
                size = int(packet.data_size)
                if size <= 0 or size > self.MAX_PACKET_BYTES or not packet.data:
                    raise ValueError(f"非法 packet data_size={size}")
                if packet.width <= 0 or packet.height <= 0:
                    raise ValueError(
                        f"非法图像尺寸 {packet.width}x{packet.height}"
                    )

                # FFI packet 的数据只在回调期间有效，必须先复制再 free。
                payload = ctypes.string_at(packet.data, size)
                now_monotonic = time.monotonic()
                frame = Frame(
                    payload=payload,
                    width=int(packet.width),
                    height=int(packet.height),
                    bit_depth=int(packet.bit_depth),
                    encoding=int(packet.encoding),
                    color_format=int(packet.color_format),
                    source_timestamp=int(packet.source_timestamp),
                    sequence_number=int(packet.sequence_number),
                    received_monotonic=now_monotonic,
                    received_wall_time=time.time(),
                )
                state.update(frame)
            except Exception as exc:
                state.set_error(str(exc))
                now = time.monotonic()
                if now - state.last_error_log >= 2.0:
                    print(f"⚠️ {state.topic} 回调错误: {exc}", flush=True)
                    state.last_error_log = now
            finally:
                subscriber = holder.get("subscriber")
                if subscriber and packet_ptr and self.lib is not None:
                    try:
                        self.lib.cosine_free_data_packet_from_subscriber(
                            subscriber, packet_ptr
                        )
                    except Exception as exc:
                        print(
                            f"⚠️ {state.topic} 释放 packet 失败: {exc}",
                            flush=True,
                        )

        return on_packet

    def start_subscribers(self) -> None:
        self._load_library()
        assert self.lib is not None

        print(
            f"FFI struct size: packet={ctypes.sizeof(CosineDataPacket)}, "
            f"qos={ctypes.sizeof(CosineQos)}",
            flush=True,
        )
        if ctypes.sizeof(CosineDataPacket) != 96:
            raise RuntimeError(
                f"CosineDataPacket ABI 不匹配: {ctypes.sizeof(CosineDataPacket)} != 96"
            )
        if not self.lib.cosine_bus_initialize():
            raise RuntimeError("cosine_bus_initialize() 失败")
        self.initialized = True

        # reliability=2(best effort), history=1, depth=4, durability=2。
        qos = CosineQos(2, 1, 4, 2, 1)
        for state in self.states:
            holder: dict = {"subscriber": None}
            callback = self._make_callback(state, holder)
            subscriber = self.lib.cosine_create_subscriber(
                state.topic.encode("utf-8"),
                b"CosineDataPacket",
                ctypes.byref(qos),
                callback,
                None,
            )
            if not subscriber:
                print(f"❌ 订阅创建失败: {state.topic}", flush=True)
                continue

            holder["subscriber"] = subscriber
            self.subscribers.append(subscriber)
            self.callbacks.append(callback)
            self.callback_holders.append(holder)
            print(f"✅ 已订阅 {state.topic} ({state.name})", flush=True)

        if not self.subscribers:
            raise RuntimeError("所有相机 topic 均订阅失败")

    def wait_for_frames(self, timeout: float) -> None:
        deadline = time.monotonic() + max(0.0, timeout)
        last_count = -1
        while time.monotonic() < deadline:
            ready = sum(1 for state in self.states if state.snapshot()[0] is not None)
            if ready != last_count:
                print(
                    f"⏳ 等待首帧: {ready}/{len(self.states)} 个相机已就绪",
                    flush=True,
                )
                last_count = ready
            if ready == len(self.states):
                break
            time.sleep(0.1)

        for state in self.states:
            frame, count, fps, error = state.snapshot()
            if frame is None:
                print(
                    f"⚠️ 未收到 {state.topic} 首帧"
                    + (f": {error}" if error else ""),
                    flush=True,
                )
            else:
                magic = frame.payload[:4].hex()
                print(
                    f"📷 {state.topic}: {frame.width}x{frame.height}, "
                    f"bytes={len(frame.payload)}, prefix={magic}, "
                    f"frames={count}, fps={fps:.1f}",
                    flush=True,
                )

    def cleanup(self) -> None:
        try:
            self.stop_recording()
        except Exception as exc:
            print(f"⚠️ stop_recording 失败: {exc}", flush=True)
        with self.cleanup_lock:
            if self.cleaned:
                return
            self.cleaned = True
            print("🧹 正在停止 cosine subscribers...", flush=True)

            if self.lib is not None:
                for subscriber in reversed(self.subscribers):
                    try:
                        self.lib.cosine_destroy_subscriber(subscriber)
                    except Exception as exc:
                        print(f"⚠️ destroy subscriber 失败: {exc}", flush=True)
                for holder in self.callback_holders:
                    holder["subscriber"] = None
                self.subscribers.clear()

                if self.initialized:
                    try:
                        self.lib.cosine_bus_release()
                    except Exception as exc:
                        print(f"⚠️ cosine_bus_release 失败: {exc}", flush=True)
                    self.initialized = False

            print("✅ cosine 资源已释放", flush=True)

    def get_recording_status(self) -> Dict[str, Any]:
        with self._record_lock:
            self._refresh_record_file_stats()
            elapsed = (
                max(0.0, time.monotonic() - self._record_mono_start)
                if self._record_active and self._record_mono_start
                else 0.0
            )
            cams = []
            for rec in self._recorders.values():
                entry = {
                    "topic": rec.topic,
                    "name": rec.name,
                    "key": rec.key,
                    "path": rec.storage_folder,
                    "frames": rec.frames,
                    "files": list(rec.files),
                    "error": rec.last_error,
                    "source": rec.source,
                }
                if rec.jpeg_mp4:
                    entry["jpeg_mp4"] = rec.jpeg_mp4
                    entry["jpeg_frames"] = rec.jpeg_frames
                cams.append(entry)
            return {
                "active": self._record_active,
                "mode": "dds_camera_dlb+jpeg_fallback",
                "dir": self._record_dir,
                "uuid": self._record_uuid,
                "started_at": self._record_started_at,
                "elapsed_sec": round(elapsed, 1),
                "post_win_ms": self._record_post_win_ms,
                "cameras": cams,
            }

    def _refresh_record_file_stats(self) -> None:
        for rec in self._recorders.values():
            folder = rec.storage_folder
            if not folder or not os.path.isdir(folder):
                continue
            files = sorted(
                f
                for f in os.listdir(folder)
                if not f.startswith(".") and os.path.isfile(os.path.join(folder, f))
            )
            rec.files = [os.path.join(folder, f) for f in files]
            frames = 0
            for f in files:
                if not f.endswith(".txt"):
                    continue
                txt_path = os.path.join(folder, f)
                try:
                    # 大文件用 wc，避免整文件读入拖慢 /record/status
                    out = subprocess.check_output(
                        ["wc", "-l", txt_path], text=True, timeout=2
                    )
                    frames = int(out.strip().split()[0])
                except Exception:
                    try:
                        with open(txt_path, "rb") as fh:
                            frames = sum(1 for _ in fh)
                    except OSError:
                        pass
            rec.frames = frames

    @staticmethod
    def _build_record_message(command: str, req_uuid: str, props: Sequence[Dict[str, Any]]) -> str:
        parts = [f'uuid: "{req_uuid}"', f"command: {command}"]
        for prop in props:
            post_win = int(prop.get("post_win", DEFAULT_DDS_POST_WIN_MS))
            key = str(prop["key"])
            folder = str(prop["storage_folder"])
            parts.append(
                f'record_props {{ post_win: {post_win} key: "{key}" '
                f'storage_folder: "{folder}" }}'
            )
        return " ".join(parts)

    def _run_dds_pub(self, message: str, timeout_sec: float = 20.0) -> Tuple[bool, str]:
        if not os.path.isfile(DDS_PUB):
            return False, f"找不到 dds_pub: {DDS_PUB}"
        cmd = [DDS_PUB, "--once", DDS_RECORD_TOPIC, DDS_RECORD_TYPE, message]
        env = os.environ.copy()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return False, "dds_pub 超时"
        except OSError as exc:
            return False, f"dds_pub 启动失败: {exc}"
        out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
        if proc.returncode != 0:
            return False, out or f"dds_pub exit={proc.returncode}"
        # 文本里通常有 Published once / Parse text format to message success
        if "Parse text format to message success" not in out and "Published once" not in out:
            # 有些版本只打 glog，只要 returncode==0 也视为成功
            pass
        return True, out

    def _ensure_jpeg_writer(self, rec: JpegFallbackRecorder, width: int, height: int) -> bool:
        if rec.writer is not None:
            return True
        if width <= 0 or height <= 0:
            return False
        os.makedirs(os.path.dirname(rec.path), exist_ok=True)
        for fourcc_name in ("mp4v", "avc1", "XVID"):
            fourcc = cv2.VideoWriter_fourcc(*fourcc_name)
            writer = cv2.VideoWriter(rec.path, fourcc, self._jpeg_fps, (width, height))
            if writer.isOpened():
                rec.writer = writer
                rec.width = width
                rec.height = height
                print(
                    f"💾 JPEG旁路录像 {rec.topic} {width}x{height} → {rec.path}",
                    flush=True,
                )
                return True
            writer.release()
        rec.last_error = f"无法创建 JPEG VideoWriter: {rec.path}"
        print(f"⚠️ {rec.last_error}", flush=True)
        return False

    def _jpeg_write_catchup(self, rec: JpegFallbackRecorder, now_mono: float) -> None:
        if rec.writer is None or rec.last_bgr is None:
            return
        expected = int(max(0.0, now_mono - self._record_mono_start) * self._jpeg_fps)
        pad = int(self._jpeg_fps * 2)
        while rec.frames < expected and pad > 0:
            rec.writer.write(rec.last_bgr)
            rec.frames += 1
            pad -= 1

    def _jpeg_fallback_loop(self) -> None:
        interval = min(1.0 / max(1.0, self._jpeg_fps), 0.02)
        while not self._jpeg_stop.is_set():
            loop_start = time.monotonic()
            now_mono = loop_start
            with self._record_lock:
                items = list(self._jpeg_recorders.items())
                active = self._record_active
            if not active:
                break
            for topic, rec in items:
                state = self.state_by_topic.get(topic)
                if state is None:
                    continue
                frame, _, _, _ = state.snapshot()
                if frame is not None:
                    key = CameraState._frame_key(frame)
                    if key != rec.last_key:
                        try:
                            bgr, _kind = frame_to_bgr(frame)
                            h, w = bgr.shape[:2]
                            if not self._ensure_jpeg_writer(rec, w, h):
                                continue
                            if (w, h) != (rec.width, rec.height):
                                bgr = cv2.resize(bgr, (rec.width, rec.height))
                            rec.last_bgr = bgr
                            rec.last_key = key
                            rec.last_error = ""
                        except Exception as exc:
                            rec.last_error = str(exc)
                self._jpeg_write_catchup(rec, now_mono)
            elapsed = time.monotonic() - loop_start
            self._jpeg_stop.wait(max(0.0, interval - elapsed))

        now_mono = time.monotonic()
        with self._record_lock:
            for rec in self._jpeg_recorders.values():
                self._jpeg_write_catchup(rec, now_mono)
                if rec.writer is not None:
                    try:
                        rec.writer.release()
                    except Exception:
                        pass
                    rec.writer = None

    def _start_jpeg_fallback(self, selected: Sequence[CameraState], meta_dir: str) -> None:
        jpeg_recs: Dict[str, JpegFallbackRecorder] = {}
        for state in selected:
            if state.topic not in HAND_JPEG_FALLBACK_TOPICS:
                continue
            key = topic_camera_key(state.topic)
            path = os.path.join(meta_dir, "jpeg_fallback", f"{key}.mp4")
            jpeg_recs[state.topic] = JpegFallbackRecorder(
                topic=state.topic, name=state.name, key=key, path=path
            )
            if state.topic in self._recorders:
                self._recorders[state.topic].source = "jpeg_fallback"
                self._recorders[state.topic].jpeg_mp4 = path
        if not jpeg_recs:
            return
        self._jpeg_recorders = jpeg_recs
        self._jpeg_stop.clear()
        self._jpeg_thread = threading.Thread(
            target=self._jpeg_fallback_loop,
            name="hand-jpeg-fallback-recorder",
            daemon=True,
        )
        self._jpeg_thread.start()
        print(
            f"🩹 腕部 JPEG 旁路录制已启动: {', '.join(jpeg_recs.keys())}",
            flush=True,
        )

    def _stop_jpeg_fallback(self) -> None:
        self._jpeg_stop.set()
        thread = self._jpeg_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=8.0)
        with self._record_lock:
            for rec in self._jpeg_recorders.values():
                if rec.writer is not None:
                    try:
                        rec.writer.release()
                    except Exception:
                        pass
                    rec.writer = None
                if rec.topic in self._recorders:
                    self._recorders[rec.topic].jpeg_frames = rec.frames
                    self._recorders[rec.topic].jpeg_mp4 = rec.path
            self._jpeg_thread = None

    def _sudo_install_jpeg_into_data_dir(self, out_dir: str) -> None:
        """把旁路 MP4 拷进 /data/record/<uuid>/（目录属 root，需 sudo）。"""
        password = os.environ.get("ROBOT_SUDO_PASSWORD", "1")
        for rec in list(self._jpeg_recorders.values()):
            if not rec.path or not os.path.isfile(rec.path):
                continue
            dest_dir = os.path.join(out_dir, "camera", rec.key)
            dest = os.path.join(dest_dir, f"{rec.key}_jpeg.mp4")
            note = os.path.join(dest_dir, "README_jpeg_fallback.txt")
            script = (
                f"mkdir -p {dest_dir!r} && "
                f"cp -f {rec.path!r} {dest!r} && "
                f"printf '%s\\n' "
                f"'hand_*_color 的 *.h265 在 848x530 下 VPU 编码花屏；请用 *_jpeg.mp4' "
                f"> {note!r} && chmod a+r {dest!r} {note!r}"
            )
            try:
                proc = subprocess.run(
                    ["sudo", "-S", "bash", "-lc", script],
                    input=password + "\n",
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                if proc.returncode == 0:
                    print(f"✅ 已安装腕部可用录像 → {dest}", flush=True)
                    if rec.topic in self._recorders:
                        self._recorders[rec.topic].jpeg_mp4 = dest
                else:
                    print(
                        f"⚠️ sudo 拷贝 JPEG 旁路失败: {proc.stderr or proc.stdout}",
                        flush=True,
                    )
            except Exception as exc:
                print(f"⚠️ sudo 拷贝 JPEG 旁路异常: {exc}", flush=True)

    def start_recording(
        self,
        indices: Optional[Sequence[int]] = None,
        fps: float = 30.0,
        post_win_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        del fps  # DDS/camera_dlb 按源流帧率写盘，不使用软编 fps
        with self._record_lock:
            if self._record_active:
                return {
                    "success": False,
                    "error": "已在录制中",
                    "recording": self.get_recording_status(),
                }
            if indices is None:
                selected = list(self.states)
            else:
                index_set = set(int(i) for i in indices)
                selected = [s for s in self.states if s.index in index_set]
            if not selected:
                return {"success": False, "error": "没有可录制的相机"}

            req_uuid = str(uuid.uuid4())
            out_dir = os.path.join(DATA_RECORD_ROOT, req_uuid)
            post_win = int(post_win_ms or DEFAULT_DDS_POST_WIN_MS)
            post_win = max(1000, min(post_win, 3_600_000))

            recorders: Dict[str, DdsCamRecord] = {}
            props: List[Dict[str, Any]] = []
            for state in selected:
                key = topic_camera_key(state.topic)
                folder = os.path.join(out_dir, "camera", key)
                recorders[state.topic] = DdsCamRecord(
                    topic=state.topic,
                    name=state.name,
                    key=key,
                    storage_folder=folder,
                )
                props.append(
                    {
                        "key": state.topic,
                        "storage_folder": folder,
                        "post_win": post_win,
                    }
                )
            message = self._build_record_message("START", req_uuid, props)

        ok, detail = self._run_dds_pub(message)
        if not ok:
            return {
                "success": False,
                "error": f"DDS START 失败: {detail}",
                "recording": self.get_recording_status(),
            }

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        meta_dir = os.path.join(self.record_root, f"session_{stamp}_{req_uuid[:8]}")
        os.makedirs(meta_dir, exist_ok=True)
        meta = {
            "mode": "dds_camera_dlb",
            "uuid": req_uuid,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "post_win_ms": post_win,
            "data_dir": out_dir,
            "command_topic": DDS_RECORD_TOPIC,
            "hand_jpeg_fallback": sorted(HAND_JPEG_FALLBACK_TOPICS),
            "hand_jpeg_note": (
                "848x530 腕部 color 的 camera_dlb H265 花屏(VPU对齐)；"
                "同目录 *_jpeg.mp4 为 DDS JPEG 旁路，请用它回放"
            ),
            "cameras": [
                {
                    "topic": s.topic,
                    "name": s.name,
                    "index": s.index,
                    "key": topic_camera_key(s.topic),
                    "storage_folder": os.path.join(
                        out_dir, "camera", topic_camera_key(s.topic)
                    ),
                }
                for s in selected
            ],
            "start_pub": detail[-500:],
        }
        meta_path = os.path.join(meta_dir, "session.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)

        with self._record_lock:
            if self._record_active:
                self._run_dds_pub(self._build_record_message("STOP", req_uuid, props))
                return {
                    "success": False,
                    "error": "已在录制中",
                    "recording": self.get_recording_status(),
                }
            self._recorders = recorders
            self._record_props = props
            self._record_dir = out_dir
            self._record_meta_path = meta_path
            self._record_uuid = req_uuid
            self._record_started_at = meta["started_at"]
            self._record_post_win_ms = post_win
            self._record_mono_start = time.monotonic()
            self._record_active = True
            self._start_jpeg_fallback(selected, meta_dir)

        print(f"🔴 DDS 录制 START uuid={req_uuid} → {out_dir}", flush=True)
        print(f"   dds_pub: {message[:180]}...", flush=True)
        return {
            "success": True,
            "message": f"DDS 开始录制 {len(selected)} 路相机",
            "recording": self.get_recording_status(),
        }

    def stop_recording(self) -> Dict[str, Any]:
        with self._record_lock:
            if not self._record_active:
                return {
                    "success": True,
                    "message": "未在录制",
                    "recording": self.get_recording_status(),
                }
            # 先占住停止权，避免与下一次 START / 重复 STOP 竞态
            self._record_active = False
            req_uuid = self._record_uuid
            props = list(self._record_props)
            out_dir = self._record_dir
            meta_path = self._record_meta_path
            mono_start = self._record_mono_start
            jpeg_recs = dict(self._jpeg_recorders)
            jpeg_thread = self._jpeg_thread
            self._jpeg_recorders = {}
            self._jpeg_thread = None

        # 只停本次会话卸下的 JPEG 线程，避免误杀下一场录制
        self._jpeg_stop.set()
        if jpeg_thread is not None and jpeg_thread.is_alive():
            jpeg_thread.join(timeout=8.0)
        for rec in jpeg_recs.values():
            if rec.writer is not None:
                try:
                    rec.writer.release()
                except Exception:
                    pass
                rec.writer = None
        with self._record_lock:
            self._jpeg_recorders = jpeg_recs
            for rec in jpeg_recs.values():
                if rec.topic in self._recorders:
                    self._recorders[rec.topic].jpeg_frames = rec.frames
                    self._recorders[rec.topic].jpeg_mp4 = rec.path

        message = self._build_record_message("STOP", req_uuid, props)
        ok, detail = self._run_dds_pub(message)
        if not ok:
            print(f"⚠️ DDS STOP 发布失败: {detail}", flush=True)
        else:
            # camera_dlb 收尾写文件
            time.sleep(1.2)

        self._sudo_install_jpeg_into_data_dir(out_dir)

        duration_sec = max(0.0, time.monotonic() - mono_start) if mono_start else 0.0
        with self._record_lock:
            self._refresh_record_file_stats()
            status = self.get_recording_status()

        if meta_path and os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    meta = json.load(fh)
                meta["ended_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                meta["duration_sec"] = round(duration_sec, 3)
                meta["stop_pub"] = (detail or "")[-500:]
                meta["cameras_written"] = status.get("cameras", [])
                with open(meta_path, "w", encoding="utf-8") as fh:
                    json.dump(meta, fh, ensure_ascii=False, indent=2)
            except Exception as exc:
                print(f"⚠️ 更新 session.json 失败: {exc}", flush=True)

        print(f"⏹ DDS 录制 STOP uuid={req_uuid} → {out_dir}", flush=True)
        return {
            "success": True,
            "message": "DDS 录制已停止" if ok else f"已请求停止，但 dds_pub 异常: {detail}",
            "recording": status,
        }

    def camera_json(self, state: CameraState) -> Optional[dict]:
        frame, count, fps, error = state.snapshot()
        if frame is None:
            return None
        age_ms = max(0.0, (time.monotonic() - frame.received_monotonic) * 1000.0)
        return {
            "index": state.index,
            "type": state.topic,
            "topic": state.topic,
            "name": state.name,
            "shape": [frame.width, frame.height],
            "fps": round(fps, 1),
            "available": age_ms < 3000.0,
            "age_ms": round(age_ms, 1),
            "frames": count,
            "error": error,
        }

    def image_json(self, state: CameraState) -> dict:
        frame, count, fps, error = state.snapshot()
        if frame is None:
            return {
                "index": state.index,
                "success": False,
                "available": False,
                "error": error or "尚未收到图像",
                "topic": state.topic,
            }

        try:
            data_url, image_format = state.data_url()
            age_ms = max(
                0.0, (time.monotonic() - frame.received_monotonic) * 1000.0
            )
            stale = age_ms >= 3000.0
            info = (
                f"尺寸: {frame.width}x{frame.height} | "
                f"接收: {fps:.1f} FPS | {image_format} | "
                f"延迟: {age_ms:.0f}ms"
            )
            if stale:
                info += " | ⚠️ 数据已停止更新"
            return {
                "index": state.index,
                "success": True,
                "available": not stale,
                "image": data_url,
                "info": info,
                "topic": state.topic,
                "width": frame.width,
                "height": frame.height,
                "fps": round(fps, 1),
                "age_ms": round(age_ms, 1),
                "frames": count,
                "source_timestamp": frame.source_timestamp,
                "sequence_number": frame.sequence_number,
                "encoding": frame.encoding,
                "color_format": frame.color_format,
                "format": image_format,
                "cached": True,
            }
        except Exception as exc:
            state.set_error(str(exc))
            return {
                "index": state.index,
                "success": False,
                "available": False,
                "error": f"图像处理失败: {exc}",
                "topic": state.topic,
            }

    def _setup_routes(self) -> None:
        @self.app.after_request
        def disable_cache(response):
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            return response

        @self.app.route("/")
        def index():
            return render_template_string(HTML_TEMPLATE)

        @self.app.route("/get_available_cameras")
        def get_available_cameras():
            cameras = []
            waiting = []
            all_cameras = []
            for state in self.states:
                data = self.camera_json(state)
                if data is None:
                    item = {
                        "index": state.index,
                        "topic": state.topic,
                        "type": state.topic,
                        "name": state.name,
                        "shape": [0, 0],
                        "fps": 0.0,
                        "available": False,
                        "ready": False,
                    }
                    waiting.append(item)
                else:
                    data["ready"] = True
                    item = data
                    cameras.append(data)
                all_cameras.append(item)
            return jsonify(
                {
                    "success": True,
                    "cameras": cameras,
                    "waiting": waiting,
                    "all_cameras": all_cameras,
                    "subscriber_count": len(self.subscribers),
                }
            )

        @self.app.route("/get_all_images")
        def get_all_images():
            start = time.monotonic()
            raw_indices = request.args.get("indices", "").strip()
            selected_states = self.states
            if raw_indices:
                try:
                    selected_indices = {
                        int(value)
                        for value in raw_indices.split(",")
                        if value.strip()
                    }
                except ValueError:
                    return jsonify(
                        {
                            "success": False,
                            "error": "indices 必须是逗号分隔的整数",
                        }
                    ), 400
                selected_states = [
                    state
                    for state in self.states
                    if state.index in selected_indices
                ]

            images = [self.image_json(state) for state in selected_states]
            elapsed_ms = (time.monotonic() - start) * 1000.0
            return jsonify(
                {
                    "success": True,
                    "images": images,
                    "processing_time_ms": round(elapsed_ms, 1),
                    "selected_count": len(selected_states),
                    "camera_count": sum(
                        1 for image in images if image.get("success")
                    ),
                }
            )

        @self.app.route("/healthz")
        def healthz():
            ready = sum(
                1 for state in self.states if state.snapshot()[0] is not None
            )
            return jsonify(
                {
                    "success": True,
                    "ready_cameras": ready,
                    "configured_cameras": len(self.states),
                    "subscribers": len(self.subscribers),
                    "middleware": os.environ.get(
                        "COSINE_BUS_DEFAULT_MIDDLEWARE", ""
                    ),
                    "recording": self.get_recording_status(),
                }
            )

        @self.app.route("/record/status")
        def record_status():
            return jsonify({"success": True, "recording": self.get_recording_status()})

        @self.app.route("/record/start", methods=["POST"])
        def record_start():
            payload = request.get_json(silent=True) or {}
            indices = payload.get("indices")
            fps = float(payload.get("fps") or 30)
            post_win_ms = payload.get("post_win_ms")
            if post_win_ms is not None:
                try:
                    post_win_ms = int(post_win_ms)
                except (TypeError, ValueError):
                    return jsonify(
                        {"success": False, "error": "post_win_ms 非法"}
                    ), 400
            if indices is None and payload.get("selected_only"):
                raw = payload.get("selected_indices") or []
                try:
                    indices = [int(v) for v in raw]
                except (TypeError, ValueError):
                    return jsonify(
                        {"success": False, "error": "selected_indices 非法"}
                    ), 400
            result = self.start_recording(
                indices=indices, fps=fps, post_win_ms=post_win_ms
            )
            code = 200 if result.get("success") else 400
            return jsonify(result), code

        @self.app.route("/record/stop", methods=["POST"])
        def record_stop():
            return jsonify(self.stop_recording())

    def run(self, first_frame_timeout: float) -> None:
        self.start_subscribers()
        self.wait_for_frames(first_frame_timeout)
        print("=" * 72, flush=True)
        print("🌐 DDS/Aorta 相机 Web 查看器已启动（未使用 agibot_gdk）", flush=True)
        print(f"🌐 监听地址: http://{self.host}:{self.port}", flush=True)
        print(f"🌐 局域网地址: http://172.22.34.161:{self.port}", flush=True)
        print("=" * 72, flush=True)
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>G2 / G2S 相机实时画面</title>
    <style>
        :root {
            color-scheme: dark;
            --bg: #0c111b;
            --panel: #151d2b;
            --panel2: #1b2637;
            --line: #2d3d55;
            --text: #edf4ff;
            --muted: #91a4bf;
            --ok: #40d58b;
            --warn: #ffbf5b;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: radial-gradient(circle at top, #18243a, var(--bg) 44%);
            color: var(--text);
            font-family: Inter, "PingFang SC", "Microsoft YaHei", sans-serif;
        }
        .container { max-width: 1500px; margin: 0 auto; padding: 22px; }
        .header {
            display: flex; align-items: end; justify-content: space-between;
            gap: 16px; margin-bottom: 18px;
        }
        h1 { margin: 0; font-size: 28px; }
        .subtitle { color: var(--muted); margin-top: 6px; font-size: 14px; }
        .status {
            padding: 9px 13px; border: 1px solid var(--line);
            background: rgba(21, 29, 43, .85); border-radius: 999px;
            color: var(--muted); white-space: nowrap;
        }
        .dot {
            display: inline-block; width: 9px; height: 9px; border-radius: 50%;
            margin-right: 8px; background: var(--warn);
        }
        .status.online .dot { background: var(--ok); box-shadow: 0 0 10px var(--ok); }
        .selector-panel {
            margin-bottom: 18px; padding: 15px; border: 1px solid var(--line);
            border-radius: 13px; background: rgba(21, 29, 43, .88);
        }
        .selector-head {
            display: flex; align-items: center; justify-content: space-between;
            gap: 12px; margin-bottom: 12px;
        }
        .selector-title { font-size: 16px; font-weight: 700; }
        .selector-note { color: var(--muted); font-size: 12px; margin-top: 3px; }
        .selector-actions { display: flex; flex-wrap: wrap; gap: 7px; }
        button {
            padding: 7px 11px; border: 1px solid var(--line); border-radius: 8px;
            background: #202d41; color: var(--text); cursor: pointer;
        }
        button:hover { border-color: #5b789e; background: #293a53; }
        .camera-selector {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr));
            gap: 8px;
        }
        .camera-option {
            display: flex; align-items: center; gap: 9px; min-height: 42px;
            padding: 8px 10px; border: 1px solid #273750; border-radius: 9px;
            background: #111927; cursor: pointer;
        }
        .camera-option:hover { border-color: #4d688c; }
        .camera-option input { width: 17px; height: 17px; accent-color: #50a8ff; }
        .option-text { min-width: 0; flex: 1; }
        .option-name { display: block; font-size: 13px; }
        .option-topic {
            display: block;
            overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
            color: var(--muted); font-size: 10px;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .online-badge {
            flex: none; padding: 2px 6px; border-radius: 999px; font-size: 10px;
            color: var(--muted); background: #283343;
        }
        .online-badge.ready { color: #071a10; background: var(--ok); }
        .cameras-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(390px, 1fr));
            gap: 16px;
        }
        .camera-card {
            overflow: hidden; border: 1px solid var(--line); border-radius: 13px;
            background: linear-gradient(155deg, var(--panel2), var(--panel));
            box-shadow: 0 12px 28px rgba(0, 0, 0, .22);
        }
        .camera-head { padding: 13px 15px 11px; }
        .camera-title { font-size: 17px; font-weight: 700; }
        .camera-topic {
            margin-top: 4px; color: #75bfff; font-size: 12px;
            font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
        }
        .camera-info { margin-top: 7px; color: var(--muted); font-size: 13px; }
        .image-wrap {
            position: relative; min-height: 240px; display: flex;
            align-items: center; justify-content: center; background: #03060a;
        }
        .camera-image {
            display: block; width: 100%; height: 330px; object-fit: contain;
            background: #03060a;
        }
        .waiting { color: var(--muted); padding: 48px 20px; text-align: center; }
        .footer { color: var(--muted); text-align: center; margin: 18px 0 4px; font-size: 13px; }
        @media (max-width: 600px) {
            .container { padding: 12px; }
            .header { align-items: flex-start; flex-direction: column; }
            .selector-head { align-items: flex-start; flex-direction: column; }
            .cameras-grid { grid-template-columns: 1fr; }
            .camera-selector { grid-template-columns: 1fr; }
            .camera-image { height: 270px; }
        }
    </style>
</head>
<body>
    <main class="container">
        <div class="header">
            <div>
                <h1>G2 / G2S 相机实时画面</h1>
                <div class="subtitle">CosineDataPacket 直订阅 · 绕过 agibot_gdk · 按选择传图</div>
            </div>
            <div class="status" id="status"><span class="dot"></span><span id="status-text">连接中</span></div>
        </div>
        <section class="selector-panel">
            <div class="selector-head">
                <div>
                    <div class="selector-title">选择显示相机</div>
                    <div class="selector-note">支持完整 G2S camera map；选择会保存在当前浏览器</div>
                </div>
                <div class="selector-actions">
                    <button type="button" id="select-online">选择在线</button>
                    <button type="button" id="select-all">全选</button>
                    <button type="button" id="select-none">清空</button>
                    <button type="button" id="record-toggle">DDS开始录制</button>
                </div>
            </div>
            <div class="selector-note" id="record-status">录制未开始（DDS H265；腕部 color 额外 JPEG 旁路 → *_jpeg.mp4）</div>
            <div class="camera-selector" id="camera-selector">
                <div class="waiting">正在获取相机列表...</div>
            </div>
        </section>
        <section class="cameras-grid" id="cameras-grid">
            <div class="waiting">请选择需要显示的相机</div>
        </section>
        <div class="footer" id="footer">准备获取图像</div>
    </main>

    <script>
        const selectionStorageKey = "g2s-camera-view-selected-topics-v2-all";
        let allCameras = [];
        let selectedTopics = new Set();
        let selectionInitialized = false;
        let listSignature = "";
        let stopped = false;
        let recordingActive = false;

        try {
            const saved = localStorage.getItem(selectionStorageKey);
            if (saved !== null) {
                const topics = JSON.parse(saved);
                if (Array.isArray(topics)) {
                    selectedTopics = new Set(topics);
                    selectionInitialized = true;
                }
            }
        } catch (error) {
            console.warn("读取相机选择失败:", error);
        }

        function escapeHtml(text) {
            const div = document.createElement("div");
            div.textContent = text;
            return div.innerHTML;
        }

        function saveSelection() {
            localStorage.setItem(
                selectionStorageKey,
                JSON.stringify(Array.from(selectedTopics))
            );
        }

        function selectedCameras() {
            return allCameras.filter(camera => selectedTopics.has(camera.topic));
        }

        function renderCameraSelector() {
            const selector = document.getElementById("camera-selector");
            if (!allCameras.length) {
                selector.innerHTML = '<div class="waiting">没有相机配置</div>';
                return;
            }
            selector.innerHTML = allCameras.map(camera => `
                <label class="camera-option">
                    <input
                        type="checkbox"
                        class="camera-toggle"
                        data-index="${camera.index}"
                        ${selectedTopics.has(camera.topic) ? "checked" : ""}
                    >
                    <span class="option-text">
                        <span class="option-name">${escapeHtml(camera.name)}</span>
                        <span class="option-topic">${escapeHtml(camera.topic)}</span>
                    </span>
                    <span class="online-badge ${camera.ready ? "ready" : ""}">
                        ${camera.ready ? `${camera.fps.toFixed(1)} FPS` : "等待数据"}
                    </span>
                </label>
            `).join("");

            document.querySelectorAll(".camera-toggle").forEach(input => {
                input.addEventListener("change", event => {
                    const index = Number(event.currentTarget.dataset.index);
                    const camera = allCameras.find(item => item.index === index);
                    if (!camera) return;
                    if (event.currentTarget.checked) {
                        selectedTopics.add(camera.topic);
                    } else {
                        selectedTopics.delete(camera.topic);
                    }
                    saveSelection();
                    renderCameraCards(selectedCameras());
                });
            });
        }

        function renderCameraCards(cameras) {
            const signature = cameras.map(c => `${c.index}:${c.topic}`).join("|");
            if (signature === listSignature) return;
            listSignature = signature;
            const grid = document.getElementById("cameras-grid");
            if (!cameras.length) {
                grid.innerHTML = '<div class="waiting">请选择至少一路相机</div>';
                return;
            }
            grid.innerHTML = cameras.map(camera => `
                <article class="camera-card">
                    <div class="camera-head">
                        <div class="camera-title">${escapeHtml(camera.name)}</div>
                        <div class="camera-topic">${escapeHtml(camera.topic)}</div>
                        <div class="camera-info" id="info-${camera.index}">
                            ${camera.ready ? "等待图像..." : "已订阅，尚未收到数据"}
                        </div>
                    </div>
                    <div class="image-wrap">
                        <img class="camera-image" id="image-${camera.index}" alt="${escapeHtml(camera.name)}">
                    </div>
                </article>
            `).join("");
        }

        async function refreshCameraList() {
            try {
                const response = await fetch("/get_available_cameras", {cache: "no-store"});
                const data = await response.json();
                if (!data.success) throw new Error(data.error || "相机列表接口失败");
                allCameras = data.all_cameras || [
                    ...(data.cameras || []),
                    ...(data.waiting || [])
                ].sort((a, b) => a.index - b.index);

                if (!selectionInitialized) {
                    // 默认勾选机型配置中的全部相机（含尚未出帧的）
                    selectedTopics = new Set(allCameras.map(camera => camera.topic));
                    selectionInitialized = true;
                    saveSelection();
                } else {
                    const knownTopics = new Set(allCameras.map(camera => camera.topic));
                    selectedTopics = new Set(
                        Array.from(selectedTopics).filter(topic => knownTopics.has(topic))
                    );
                    // 若本地存的是空选择，回退为全选
                    if (!selectedTopics.size && allCameras.length) {
                        selectedTopics = new Set(allCameras.map(camera => camera.topic));
                        saveSelection();
                    }
                }

                renderCameraSelector();
                renderCameraCards(selectedCameras());
            } catch (error) {
                console.error("camera list:", error);
            }
        }

        async function pollImages() {
            if (stopped) return;
            const started = performance.now();
            const cameras = selectedCameras();
            if (!cameras.length) {
                document.getElementById("status").classList.add("online");
                document.getElementById("status-text").textContent = "未选择相机";
                document.getElementById("footer").textContent = "请选择需要显示的相机";
                setTimeout(pollImages, 250);
                return;
            }

            try {
                const indices = cameras.map(camera => camera.index).join(",");
                const response = await fetch(
                    `/get_all_images?indices=${encodeURIComponent(indices)}`,
                    {cache: "no-store"}
                );
                const data = await response.json();
                if (!data.success) throw new Error(data.error || "接口返回失败");
                let good = 0;
                for (const imageData of data.images) {
                    const image = document.getElementById(`image-${imageData.index}`);
                    const info = document.getElementById(`info-${imageData.index}`);
                    if (!image || !info) continue;
                    if (imageData.success && imageData.image) {
                        image.src = imageData.image;
                        info.textContent = imageData.info;
                        good += 1;
                    } else {
                        info.textContent = imageData.error || "无法获取图像";
                    }
                }
                const status = document.getElementById("status");
                status.classList.add("online");
                document.getElementById("status-text").textContent =
                    `${good}/${cameras.length} 路已显示`;
                document.getElementById("footer").textContent =
                    `服务端处理 ${data.processing_time_ms}ms · 浏览器往返 ${(performance.now() - started).toFixed(1)}ms`;
            } catch (error) {
                document.getElementById("status").classList.remove("online");
                document.getElementById("status-text").textContent = `连接异常: ${error}`;
            } finally {
                setTimeout(pollImages, 100);
            }
        }

        function applySelection(cameras) {
            selectedTopics = new Set(cameras.map(camera => camera.topic));
            selectionInitialized = true;
            saveSelection();
            renderCameraSelector();
            renderCameraCards(selectedCameras());
        }

        function renderRecordStatus(rec) {
            const el = document.getElementById("record-status");
            const btn = document.getElementById("record-toggle");
            if (!el || !btn) return;
            recordingActive = !!(rec && rec.active);
            if (!recordingActive) {
                btn.textContent = "DDS开始录制";
                el.textContent = rec && rec.dir
                    ? `上次录制: ${rec.dir}`
                    : "录制未开始（DDS H265；腕部 color 用 *_jpeg.mp4）";
                return;
            }
            const totalFrames = (rec.cameras || []).reduce(
                (sum, cam) => sum + (cam.frames || 0), 0
            );
            const elapsed = rec.elapsed_sec != null ? `${rec.elapsed_sec}s` : "";
            btn.textContent = `停止录制 ${elapsed}`;
            el.textContent =
                `DDS录制中 uuid=${rec.uuid || ""} → ${rec.dir || ""} | ` +
                `${(rec.cameras || []).length} 路 | 已写 ${totalFrames} 帧`;
        }

        async function refreshRecordStatus() {
            try {
                const response = await fetch("/record/status", {cache: "no-store"});
                const data = await response.json();
                if (data.success) renderRecordStatus(data.recording || {});
            } catch (error) {
                console.warn("record status:", error);
            }
        }

        async function toggleRecording() {
            const starting = !recordingActive;
            const url = starting ? "/record/start" : "/record/stop";
            try {
                const response = await fetch(url, {
                    method: "POST",
                    headers: starting ? {"Content-Type": "application/json"} : undefined,
                    body: starting ? JSON.stringify({fps: 30}) : undefined,
                });
                const data = await response.json();
                if (!data.success) {
                    alert(data.error || "录制操作失败");
                }
                renderRecordStatus(data.recording || {});
            } catch (error) {
                alert(`录制请求失败: ${error}`);
            }
        }

        document.getElementById("select-online").addEventListener(
            "click",
            () => applySelection(allCameras.filter(camera => camera.ready))
        );
        document.getElementById("select-all").addEventListener(
            "click",
            () => applySelection(allCameras)
        );
        document.getElementById("select-none").addEventListener(
            "click",
            () => applySelection([])
        );
        document.getElementById("record-toggle").addEventListener(
            "click",
            () => toggleRecording()
        );
        window.addEventListener("beforeunload", () => { stopped = true; });
        refreshCameraList();
        refreshRecordStatus();
        setInterval(refreshCameraList, 2000);
        setInterval(refreshRecordStatus, 1000);
        pollImages();
    </script>
</body>
</html>
"""


def parse_camera(value: str) -> Tuple[str, str]:
    """解析 --camera '/topic=显示名'。显示名可省略。"""
    topic, separator, name = value.partition("=")
    topic = topic.strip()
    if not topic.startswith("/"):
        raise argparse.ArgumentTypeError("camera topic 必须以 / 开头")
    if not separator:
        name = topic.rsplit("/", 1)[-1]
    name = name.strip() or topic
    return topic, name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="直接订阅 CosineDataPacket 并提供相机 Web 预览"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Flask 监听地址")
    parser.add_argument("--port", type=int, default=5001, help="Flask 监听端口")
    parser.add_argument(
        "--product",
        default=DEFAULT_PRODUCT,
        choices=product_ids(),
        help="机型相机表（见 camera_device_map.json），默认 g2_thor",
    )
    parser.add_argument(
        "--camera",
        action="append",
        type=parse_camera,
        help="订阅相机，格式 /camera/topic=显示名；可重复指定",
    )
    parser.add_argument(
        "--first-frame-timeout",
        type=float,
        default=5.0,
        help="启动 Web 服务前等待相机首帧的秒数",
    )
    parser.add_argument(
        "--record-dir",
        default="",
        help="录像输出根目录（默认脚本旁 recordings/）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cameras = args.camera or list(viewer_cameras(args.product))
    viewer = CosineCameraViewer(
        cameras=cameras,
        host=args.host,
        port=args.port,
        record_dir=args.record_dir or None,
    )

    def signal_handler(signum, _frame) -> None:
        print(f"\n收到信号 {signum}，准备退出...", flush=True)
        viewer.cleanup()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(viewer.cleanup)

    try:
        viewer.run(first_frame_timeout=args.first_frame_timeout)
        return 0
    finally:
        viewer.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())

