#!/usr/bin/env python3
"""Record any subset of the six G2 RGB-D streams in one GDK process."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

try:
    from .camera import decode_color_rgb, decode_depth_u16
except ImportError:
    from camera import decode_color_rgb, decode_depth_u16


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STREAM_TYPES = {
    "head_color": "kHeadColor",
    "head_depth": "kHeadDepth",
    "left_color": "kHandLeftColor",
    "left_depth": "kHandLeftDepth",
    "right_color": "kHandRightColor",
    "right_depth": "kHandRightDepth",
}
DEPTH_STREAMS = {name for name in STREAM_TYPES if name.endswith("_depth")}


def parse_streams(value: str) -> tuple[str, ...]:
    """Parse ``all`` or a comma-separated subset in canonical stream order."""
    text = str(value).strip().lower()
    if text == "all":
        return tuple(STREAM_TYPES)
    requested = [item.strip() for item in text.split(",") if item.strip()]
    if not requested:
        raise ValueError("--streams 不能为空")
    unknown = [name for name in requested if name not in STREAM_TYPES]
    if unknown:
        raise ValueError(f"未知相机流：{', '.join(unknown)}")
    if len(set(requested)) != len(requested):
        raise ValueError("--streams 不能包含重复相机流")
    selected = set(requested)
    return tuple(name for name in STREAM_TYPES if name in selected)


def measured_fps(timestamps_ns) -> float:
    """Estimate native FPS from the median positive GDK timestamp delta."""
    values = [int(value) for value in timestamps_ns]
    deltas = [later - earlier for earlier, later in zip(values, values[1:])]
    if any(delta < 0 for delta in deltas):
        raise ValueError("GDK 时间戳发生倒退，无法计算实测帧率")
    deltas = [delta for delta in deltas if delta > 0]
    if not deltas:
        raise ValueError("实测帧率至少两个不同的 GDK 时间戳才能计算")
    fps = 1_000_000_000.0 / statistics.median(deltas)
    if not math.isfinite(fps) or fps <= 0.0:
        raise ValueError("无法从 GDK 时间戳计算有效帧率")
    if fps > 120.0:
        raise ValueError(f"GDK 实测帧率 {fps:.3f} 超过 120 FPS，时间戳可能异常")
    return fps


def depth_preview_bgr(depth_u16: np.ndarray, *, near_mm: int, far_mm: int) -> np.ndarray:
    """Convert Z16 depth to a viewable color image without changing raw data."""
    depth = np.asarray(depth_u16)
    if depth.ndim != 2 or depth.dtype != np.uint16:
        raise ValueError("深度帧必须是二维 uint16 Z16 图像")
    if not 0 <= int(near_mm) < int(far_mm) <= 65535:
        raise ValueError("深度伪彩范围必须满足 0 <= near < far <= 65535")
    invalid = (depth == 0) | (depth == np.iinfo(np.uint16).max)
    clipped = np.clip(depth, int(near_mm), int(far_mm)).astype(np.float32)
    normalized = ((clipped - int(near_mm)) * (255.0 / (int(far_mm) - int(near_mm)))).astype(np.uint8)
    preview = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    preview[invalid] = 0
    return preview


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="只读录制 G2 任意一路或多路 RGB-D 相机")
    parser.add_argument(
        "--streams", default="all",
        help="逗号分隔：head_color,head_depth,left_color,left_depth,right_color,right_depth；默认 all",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "datasets" / "camera_videos",
        help="输出根目录；程序会在其中创建带时间戳的录制目录",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="视频/采样帧率；0 表示按各路 GDK 实测帧率保存")
    parser.add_argument("--seconds", type=float, default=0.0, help="录制秒数；0 表示按 Ctrl-C 停止")
    parser.add_argument("--warmup", type=int, default=10, help="开始录制前每路读取的不同时间戳帧数")
    parser.add_argument("--warmup-timeout", type=float, default=15.0, help="等待各路预热帧的最长秒数，默认 15")
    parser.add_argument("--max-gap-seconds", type=float, default=2.0, help="实测帧率模式允许补帧的最大时间戳间隔，默认 2 秒")
    parser.add_argument("--depth-near-mm", type=int, default=200, help="深度伪彩近端，默认 200 mm")
    parser.add_argument("--depth-far-mm", type=int, default=2000, help="深度伪彩远端，默认 2000 mm")
    return parser.parse_args(argv)


@dataclass
class StreamOutput:
    name: str
    camera_type: object
    fps: float
    session_dir: Path
    writer: object | None = None
    timestamp_stream: object | None = None
    preserve_timeline: bool = False
    max_gap_seconds: float = 2.0
    frames: int = 0
    captured_frames: int = 0
    last_timestamp_ns: int | None = None
    first_timestamp_ns: int | None = None
    last_frame_bgr: np.ndarray | None = None

    @property
    def raw_dir(self) -> Path:
        return self.session_dir / f"{self.name}_raw_z16"

    def open_writer(self, frame_bgr: np.ndarray) -> None:
        height, width = frame_bgr.shape[:2]
        path = self.session_dir / f"{self.name}.mp4"
        self.writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建视频文件：{path}")
        self.timestamp_stream = (self.session_dir / f"{self.name}_timestamps.jsonl").open("w", encoding="utf-8")
        if self.name in DEPTH_STREAMS:
            self.raw_dir.mkdir(parents=True, exist_ok=True)

    def write(self, frame_bgr: np.ndarray, timestamp_ns: int, raw_depth: np.ndarray | None = None) -> bool:
        stamp = int(timestamp_ns)
        if self.last_timestamp_ns == stamp:
            return False
        if self.last_timestamp_ns is not None and stamp < self.last_timestamp_ns:
            raise RuntimeError(f"{self.name} GDK 时间戳倒退：{stamp} < {self.last_timestamp_ns}")
        if (
            self.preserve_timeline and self.last_timestamp_ns is not None
            and stamp - self.last_timestamp_ns > self.max_gap_seconds * 1_000_000_000
        ):
            raise RuntimeError(
                f"{self.name} GDK 时间戳向前跳变超过 {self.max_gap_seconds:g} 秒，停止以避免大量补帧"
            )
        if self.writer is None:
            self.open_writer(frame_bgr)
        assert self.timestamp_stream is not None
        if self.first_timestamp_ns is None:
            self.first_timestamp_ns = stamp
        self.captured_frames += 1
        raw_file = self._save_raw_depth(raw_depth, stamp)
        if self.preserve_timeline and self.last_frame_bgr is not None:
            position = round((stamp - self.first_timestamp_ns) * self.fps / 1_000_000_000.0)
            if position < self.frames:
                self.timestamp_stream.write(json.dumps({
                    "frame": None,
                    "source_timestamp_ns": stamp,
                    "duplicated": False,
                    "dropped": True,
                    "raw_z16_file": raw_file,
                }, ensure_ascii=False) + "\n")
                self.timestamp_stream.flush()
                self.last_frame_bgr = frame_bgr.copy()
                self.last_timestamp_ns = stamp
                return True
            while self.frames < position:
                self._write_video_frame(self.last_frame_bgr, self.last_timestamp_ns, duplicated=True)
        self._write_video_frame(frame_bgr, stamp, duplicated=False, raw_file=raw_file)
        self.last_frame_bgr = frame_bgr.copy()
        self.last_timestamp_ns = stamp
        return True

    def _save_raw_depth(self, raw_depth: np.ndarray | None, stamp: int) -> str | None:
        if raw_depth is None:
            return None
        raw_file = f"{self.name}_raw_z16/{self.captured_frames:08d}_{stamp}.png"
        if not cv2.imwrite(str(self.session_dir / raw_file), raw_depth):
            raise RuntimeError(f"无法保存原始深度帧：{raw_file}")
        return raw_file

    def _write_video_frame(
        self, frame_bgr: np.ndarray, source_timestamp_ns: int, *,
        duplicated: bool, raw_file: str | None = None,
    ) -> None:
        assert self.writer is not None and self.timestamp_stream is not None
        self.writer.write(frame_bgr)
        self.frames += 1
        self.timestamp_stream.write(json.dumps({
            "frame": self.frames,
            "source_timestamp_ns": int(source_timestamp_ns),
            "duplicated": duplicated,
            "raw_z16_file": raw_file,
        }, ensure_ascii=False) + "\n")
        self.timestamp_stream.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        if self.timestamp_stream is not None:
            self.timestamp_stream.close()


def _collect_timestamps(
    camera, outputs: dict[str, StreamOutput], count: int, *, timeout_s: float,
    clock=time.monotonic, sleep=time.sleep,
) -> dict[str, list[int]]:
    samples = {name: [] for name in outputs}
    if count == 0:
        return samples
    deadline = clock() + timeout_s
    while any(len(values) < count for values in samples.values()):
        if clock() >= deadline:
            stalled = ", ".join(
                f"{name}={len(values)}/{count}" for name, values in samples.items() if len(values) < count
            )
            raise TimeoutError(f"相机预热超时，未取得足够的新帧：{stalled}")
        for name, output in outputs.items():
            if len(samples[name]) >= count:
                continue
            remaining_ms = max(1.0, min(1000.0, (deadline - clock()) * 1000.0))
            if remaining_ms <= 1.0 and clock() >= deadline:
                break
            image = camera.get_latest_image(output.camera_type, remaining_ms)
            stamp = int(image.timestamp_ns)
            if not samples[name] or samples[name][-1] != stamp:
                samples[name].append(stamp)
        sleep(0.001)
    return samples


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        stream_names = parse_streams(args.streams)
        if not math.isfinite(args.fps) or args.fps < 0.0 or args.fps > 120.0:
            raise ValueError("--fps 必须在 [0, 120] 范围内")
        if not math.isfinite(args.seconds) or args.seconds < 0.0 or args.warmup < 0:
            raise ValueError("--seconds 必须为非负有限数，--warmup 不能为负")
        if not math.isfinite(args.warmup_timeout) or args.warmup_timeout <= 0.0:
            raise ValueError("--warmup-timeout 必须为正有限数")
        if not math.isfinite(args.max_gap_seconds) or args.max_gap_seconds <= 0.0:
            raise ValueError("--max-gap-seconds 必须为正有限数")
        if not 0 <= args.depth_near_mm < args.depth_far_mm <= 65535:
            raise ValueError("深度伪彩范围必须满足 0 <= near < far <= 65535")
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    try:
        import agibot_gdk
    except ImportError:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_sam3_python.sh 运行。", file=sys.stderr)
        return 2

    initialized = False
    camera = None
    outputs: dict[str, StreamOutput] = {}
    result_label = "录制完成"
    session_dir = args.output_dir.expanduser().resolve() / datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
            raise RuntimeError("GDK 初始化失败")
        initialized = True
        types = {name: getattr(agibot_gdk.CameraType, STREAM_TYPES[name]) for name in stream_names}
        camera = agibot_gdk.Camera(list(types.values()))
        provisional = {
            name: StreamOutput(name, camera_type, args.fps, session_dir)
            for name, camera_type in types.items()
        }
        sample_count = max(2, args.warmup) if args.fps == 0.0 else args.warmup
        samples = _collect_timestamps(
            camera, provisional, sample_count, timeout_s=args.warmup_timeout,
        )
        session_dir.mkdir(parents=True, exist_ok=False)
        for name, output in provisional.items():
            output.fps = measured_fps(samples[name]) if args.fps == 0.0 else float(args.fps)
            output.preserve_timeline = args.fps == 0.0
            output.max_gap_seconds = args.max_gap_seconds
            outputs[name] = output
        metadata = {
            "streams": list(stream_names),
            "requested_fps": float(args.fps),
            "video_fps": {name: round(output.fps, 6) for name, output in outputs.items()},
            "actual_rate_timeline_preserved": args.fps == 0.0,
            "max_gap_seconds": args.max_gap_seconds,
            "depth_raw_format": "lossless uint16 PNG; values preserve GDK Z16",
            "depth_preview_range_mm": [args.depth_near_mm, args.depth_far_mm],
        }
        (session_dir / "recording.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
        )
        print(f"开始录制：{session_dir}")
        for name, output in outputs.items():
            suffix = "MP4 + 原始 Z16 PNG" if name in DEPTH_STREAMS else "MP4"
            print(f"  {name}: {output.fps:.3f} FPS，{suffix}")
        print("按 Ctrl-C 停止；录制过程不发送任何机器人或夹爪命令。")
        started = time.monotonic()
        interval = 0.0 if args.fps == 0.0 else 1.0 / args.fps
        while args.seconds == 0.0 or time.monotonic() - started < args.seconds:
            loop_started = time.monotonic()
            for name, output in outputs.items():
                image = camera.get_latest_image(output.camera_type, 1000.0)
                stamp = int(image.timestamp_ns)
                if output.last_timestamp_ns == stamp:
                    continue
                if name in DEPTH_STREAMS:
                    raw = decode_depth_u16(image, agibot_gdk)
                    preview = depth_preview_bgr(raw, near_mm=args.depth_near_mm, far_mm=args.depth_far_mm)
                    output.write(preview, stamp, raw)
                else:
                    rgb = decode_color_rgb(image, agibot_gdk)
                    output.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), stamp)
            delay = max(0.0, interval - (time.monotonic() - loop_started)) if interval else 0.001
            time.sleep(delay)
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        result_label = "录制失败，已保存已有数据"
        print(f"相机视频录制失败：{exc}", file=sys.stderr)
        return 1
    finally:
        for output in outputs.values():
            output.close()
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        if initialized:
            try:
                agibot_gdk.gdk_release()
            except Exception:
                pass
        if outputs:
            counts = {name: output.frames for name, output in outputs.items()}
            print(f"{result_label}：{session_dir}，frames={counts}")


if __name__ == "__main__":
    raise SystemExit(main())
