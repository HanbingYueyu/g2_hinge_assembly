"""Fresh, fixed-identity temporal aggregation for bilateral RGB-D depth."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

try:
    from .bilateral_rgbd_geometry import hole_box_depth_support, hole_ring_depth_support
except ImportError:
    from bilateral_rgbd_geometry import hole_box_depth_support, hole_ring_depth_support


@dataclass(frozen=True)
class DepthWindowResult:
    depth_u16: np.ndarray | None
    box_xyxy: tuple[float, float, float, float] | None
    diagnostics: dict


class FreshHoleDepthAccumulator:
    """Aggregate only fresh frames belonging to one locked hole box."""

    def __init__(
        self,
        *,
        window_frames: int = 5,
        min_temporal_hits: int = 3,
        max_window_ms: float = 500.0,
        max_center_jitter_px: float = 8.0,
        max_size_change_fraction: float = 0.25,
        max_temporal_mad_mm: float = 15.0,
        admission_min_valid_pixels: int = 1,
        admission_min_support_fraction: float = 0.0,
        admission_min_depth_m: float = 0.03,
        admission_max_depth_m: float = 2.0,
        admission_depth_scale_m: float = 0.001,
        sampling_region: str = "ring",
    ) -> None:
        if window_frames < 1 or min_temporal_hits < 1 or min_temporal_hits > window_frames:
            raise ValueError("invalid temporal depth window size")
        if not math.isfinite(max_window_ms) or max_window_ms <= 0.0:
            raise ValueError("max_window_ms must be positive and finite")
        if not math.isfinite(max_center_jitter_px) or max_center_jitter_px <= 0.0:
            raise ValueError("max_center_jitter_px must be positive and finite")
        if not math.isfinite(max_size_change_fraction) or not 0.0 <= max_size_change_fraction < 1.0:
            raise ValueError("max_size_change_fraction must be in [0, 1)")
        if not math.isfinite(max_temporal_mad_mm) or max_temporal_mad_mm <= 0.0:
            raise ValueError("max_temporal_mad_mm must be positive and finite")
        if admission_min_valid_pixels < 1:
            raise ValueError("admission_min_valid_pixels must be positive")
        if not math.isfinite(admission_min_support_fraction) or not 0.0 <= admission_min_support_fraction <= 1.0:
            raise ValueError("admission_min_support_fraction must be in [0, 1]")
        if not math.isfinite(admission_min_depth_m) or not math.isfinite(admission_max_depth_m) or admission_min_depth_m <= 0.0 or admission_max_depth_m < admission_min_depth_m:
            raise ValueError("admission depth range is invalid")
        if not math.isfinite(admission_depth_scale_m) or admission_depth_scale_m <= 0.0:
            raise ValueError("admission_depth_scale_m must be positive and finite")
        if sampling_region not in {"ring", "box"}:
            raise ValueError("sampling_region must be 'ring' or 'box'")
        self.window_frames = int(window_frames)
        self.min_temporal_hits = int(min_temporal_hits)
        self.max_window_ms = float(max_window_ms)
        self.max_center_jitter_px = float(max_center_jitter_px)
        self.max_size_change_fraction = float(max_size_change_fraction)
        self.max_temporal_mad_mm = float(max_temporal_mad_mm)
        self.admission_min_valid_pixels = int(admission_min_valid_pixels)
        self.admission_min_support_fraction = float(admission_min_support_fraction)
        self.admission_min_depth_m = float(admission_min_depth_m)
        self.admission_max_depth_m = float(admission_max_depth_m)
        self.admission_depth_scale_m = float(admission_depth_scale_m)
        self.sampling_region = str(sampling_region)
        self._samples: list[tuple[int, tuple[float, float, float, float], np.ndarray]] = []
        self._shape: tuple[int, int] | None = None
        self._last_reset_reason: str | None = None
        self._last_seen_timestamp_ns: int | None = None

    @property
    def box_xyxy(self) -> tuple[float, float, float, float] | None:
        if not self._samples:
            return None
        return tuple(
            float(value)
            for value in np.median(np.asarray([sample[1] for sample in self._samples], dtype=np.float64), axis=0)
        )

    def reset(self, reason: str = "manual_reset") -> None:
        self._samples.clear()
        self._shape = None
        self._last_reset_reason = str(reason)

    def _start_sample(self, timestamp_ns: int, box: tuple[float, float, float, float], depth: np.ndarray) -> None:
        self._samples[:] = [(timestamp_ns, box, depth.copy())]
        self._shape = (int(depth.shape[0]), int(depth.shape[1]))

    def _box_stable(self, box: tuple[float, float, float, float]) -> bool:
        anchor = self.box_xyxy
        if anchor is None:
            return True
        center = ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)
        anchor_center = ((anchor[0] + anchor[2]) * 0.5, (anchor[1] + anchor[3]) * 0.5)
        if math.hypot(center[0] - anchor_center[0], center[1] - anchor_center[1]) > self.max_center_jitter_px:
            return False
        width = max(1.0, box[2] - box[0])
        height = max(1.0, box[3] - box[1])
        anchor_width = max(1.0, anchor[2] - anchor[0])
        anchor_height = max(1.0, anchor[3] - anchor[1])
        return (
            abs(width - anchor_width) / anchor_width <= self.max_size_change_fraction
            and abs(height - anchor_height) / anchor_height <= self.max_size_change_fraction
        )

    def _diagnostics(
        self,
        *,
        state: str,
        reason: str | None = None,
        support: dict[str, float | int] | None = None,
    ) -> dict:
        if self._samples:
            window_span_ms = (self._samples[-1][0] - self._samples[0][0]) / 1_000_000.0
        else:
            window_span_ms = 0.0
        diagnostics = {
            "state": state,
            "window_frames": len(self._samples),
            "window_span_ms": round(window_span_ms, 1),
            "temporal_min_hits": self.min_temporal_hits,
            "max_temporal_mad_mm": self.max_temporal_mad_mm,
            "reset_reason": self._last_reset_reason,
        }
        if reason is not None:
            diagnostics["reason"] = reason
        if support is not None:
            diagnostics.update({
                "frame_depth_region": self.sampling_region,
                "frame_raw_valid_pixels": support["raw_valid_pixels"],
                "frame_valid_pixels": support["valid_pixels"],
                "frame_support_fraction": round(float(support["support_fraction"]), 4),
            })
            if self.sampling_region == "box":
                diagnostics["frame_box_pixels"] = support["box_pixels"]
            else:
                diagnostics["frame_ring_pixels"] = support["ring_pixels"]
        return diagnostics

    def add(
        self,
        depth_u16: np.ndarray,
        timestamp_ns: int,
        xyxy: tuple[float, float, float, float],
    ) -> DepthWindowResult:
        depth = np.asarray(depth_u16)
        timestamp = int(timestamp_ns)
        box = tuple(float(value) for value in xyxy)
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError("temporal depth must be a uint16 image")
        if timestamp <= 0 or len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("temporal depth sample metadata is invalid")
        if self._last_seen_timestamp_ns is not None and timestamp <= self._last_seen_timestamp_ns:
            self.reset("duplicate_or_old_depth_frame")
            diagnostics = self._diagnostics(state="skipped", reason="duplicate_or_old_depth_frame")
            self._last_reset_reason = None
            return DepthWindowResult(None, self.box_xyxy, diagnostics)
        self._last_seen_timestamp_ns = timestamp
        if self._shape is not None and self._shape != (int(depth.shape[0]), int(depth.shape[1])):
            self.reset("depth_shape_changed")
        if self._samples and not self._box_stable(box):
            self.reset("hole_box_unstable_reset")
        if self._samples and timestamp - self._samples[0][0] > int(self.max_window_ms * 1_000_000.0):
            self.reset("hole_depth_window_stale")
        if self.sampling_region == "box":
            support = hole_box_depth_support(
                depth,
                box,
                depth_scale_m=self.admission_depth_scale_m,
                min_depth_m=self.admission_min_depth_m,
                max_depth_m=self.admission_max_depth_m,
            )
        else:
            support = hole_ring_depth_support(
                depth,
                box,
                depth_scale_m=self.admission_depth_scale_m,
                min_depth_m=self.admission_min_depth_m,
                max_depth_m=self.admission_max_depth_m,
            )
        if (
            int(support["valid_pixels"]) < self.admission_min_valid_pixels
            or float(support["support_fraction"]) < self.admission_min_support_fraction
        ):
            diagnostics = self._diagnostics(
                state="skipped",
                reason="skipped_no_hole_depth",
                support=support,
            )
            self._last_reset_reason = None
            return DepthWindowResult(None, self.box_xyxy, diagnostics)
        self._samples.append((timestamp, box, depth.copy()))
        self._shape = (int(depth.shape[0]), int(depth.shape[1]))
        diagnostics = self._diagnostics(state="collecting", support=support)
        self._last_reset_reason = None
        if len(self._samples) < self.window_frames:
            return DepthWindowResult(None, self.box_xyxy, diagnostics)

        stack = np.stack([sample[2] for sample in self._samples], axis=0)
        valid = stack > 0
        hit_count = valid.sum(axis=0)
        stable = hit_count >= self.min_temporal_hits
        work = stack.astype(np.float32, copy=False)
        work[~valid] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(all="ignore"):
                median = np.nanmedian(work, axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            temporal_mad_u16 = np.nanmedian(np.abs(work - median[None, ...]), axis=0)
        stable &= np.isfinite(temporal_mad_u16) & (temporal_mad_u16 <= self.max_temporal_mad_mm)
        result = np.where(stable, median, 0.0)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0).astype(np.uint16)
        stable_values = result[stable].astype(np.float64) * 0.001
        temporal_mad_m = None
        if stable_values.size:
            temporal_mad_m = float(np.median(temporal_mad_u16[stable])) * 0.001
        diagnostics.update({
            "state": "ready",
            "temporal_stable_pixels": int(np.count_nonzero(stable)),
            "temporal_mad_mm": None if temporal_mad_m is None else round(1000.0 * temporal_mad_m, 2),
        })
        return DepthWindowResult(result, self.box_xyxy, diagnostics)
