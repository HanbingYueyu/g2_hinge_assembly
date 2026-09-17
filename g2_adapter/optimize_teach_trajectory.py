"""Offline joint-path simplification; playback provides quintic interpolation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

try:
    from .teach_trajectory import SafetyError, build_trajectory, load_trajectory, trajectory_document
    from .teach_trajectory_gdk import _atomic_write_json
except ImportError:
    from teach_trajectory import SafetyError, build_trajectory, load_trajectory, trajectory_document
    from teach_trajectory_gdk import _atomic_write_json


def optimize(trajectory, *, tolerance_deg=1.0, jitter_deg=.2, median_window=5,
             max_velocity_rad_s=.1, keep_indices=()):
    """Return sparse retimed keyframes compatible with the existing replay."""
    for value in (tolerance_deg, max_velocity_rad_s):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("简化容差和关节速度必须为正有限值")
    if not math.isfinite(jitter_deg) or jitter_deg < 0:
        raise ValueError("抖动阈值必须为非负有限值")
    if not isinstance(median_window, int) or median_window < 1 or median_window % 2 == 0:
        raise ValueError("滤波窗口必须为正奇数")
    size = len(trajectory.samples)
    anchors = {0, size-1}
    for index in keep_indices:
        if not isinstance(index, int) or not 0 <= index < size:
            raise ValueError("保留点索引超出轨迹样本范围")
        anchors.add(index)
    original = [s.joint_positions_rad for s in trajectory.samples]
    points = list(original)
    radius = median_window // 2
    jitter = math.radians(jitter_deg)
    # Only accept small median corrections; never flatten a large narrow turn.
    for index in range(1, size-1):
        if index in anchors:
            continue
        neighbours = original[max(0, index-radius):min(size, index+radius+1)]
        median = tuple(statistics.median(q[j] for q in neighbours) for j in range(7))
        if max(abs(a-b) for a,b in zip(median, original[index])) <= jitter:
            points[index] = median
    tolerance = math.radians(tolerance_deg)
    selected = set(anchors)
    ordered_anchors = sorted(anchors)
    stack = list(zip(ordered_anchors, ordered_anchors[1:]))
    while stack:
        first, last = stack.pop()
        chord = tuple(b-a for a,b in zip(points[first], points[last]))
        norm_sq = sum(q*q for q in chord)
        chord_max = max(abs(q) for q in chord)
        maximum_error, split = tolerance, None
        furthest, furthest_index = 0.0, first
        for index in range(first+1, last):
            projection = sum((q-a)*d for q,a,d in zip(points[index], points[first], chord)) / norm_sq if norm_sq else 0.0
            fraction = min(1., max(0., projection))
            error = max(abs(q-(a+fraction*d)) for q,a,d in zip(points[index], points[first], chord))
            if error > maximum_error:
                maximum_error, split = error, index
            # Preserve substantial ordered backtracking, even along a straight chord.
            reverse_error = (furthest-fraction)*chord_max
            if reverse_error > maximum_error and first < furthest_index < last:
                maximum_error, split = reverse_error, furthest_index
            if fraction > furthest:
                furthest, furthest_index = fraction, index
        if split is not None:
            selected.add(split)
            stack.extend(((first, split), (split, last)))
    indices = sorted(selected)
    samples = [(0, points[indices[0]])]
    elapsed_ns = 0
    for previous, index in zip(indices, indices[1:]):
        distance = max(abs(a-b) for a,b in zip(points[previous], points[index]))
        # max(smoothstep5 derivative)=1.875. Round UP to a 10ms control tick.
        ticks = max(1, math.ceil(1.875*distance/max_velocity_rad_s/.01))
        elapsed_ns += ticks*10_000_000
        samples.append((elapsed_ns, points[index]))
    result = build_trajectory(trajectory.side, samples)
    report = {"original_samples": size, "keyframes": len(indices), "key_indices": indices,
              "original_duration_s": trajectory.duration_ns/1e9, "optimized_duration_s": result.duration_ns/1e9,
              "tolerance_deg": tolerance_deg, "jitter_deg": jitter_deg, "median_window": median_window,
              "max_velocity_rad_s": max_velocity_rad_s, "interpolation": "quintic_zero_velocity_acceleration_at_keyframes"}
    return result, report


def main(argv=None):
    parser = argparse.ArgumentParser(description="离线优化示教轨迹，不连接机器人，覆盖优化结果但保留原始输入")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="默认输入文件名加 _optimized")
    parser.add_argument("--tolerance-deg", type=float, default=1.)
    parser.add_argument("--jitter-deg", type=float, default=.2)
    parser.add_argument("--median-window", type=int, default=5)
    parser.add_argument("--max-velocity-rad-s", type=float, default=.1, help="1倍回放时每关节峰值速度，默认0.1rad/s")
    parser.add_argument("--keep-index", action="append", type=int, default=[], help="必保留原始样本索引，从0开始，可重复指定")
    args = parser.parse_args(argv)
    try:
        source = args.input.expanduser().resolve()
        target = args.output.expanduser().resolve() if args.output else source.with_name(source.stem+"_optimized.json")
        if target == source:
            raise ValueError(f"优化输出不能覆盖原始输入文件：{target}")
        original = json.loads(source.read_text(encoding="utf-8"))
        result, report = optimize(load_trajectory(original), tolerance_deg=args.tolerance_deg,
                                  jitter_deg=args.jitter_deg, median_window=args.median_window,
                                  max_velocity_rad_s=args.max_velocity_rad_s, keep_indices=args.keep_index)
        document = trajectory_document(result)
        document["optimization"] = {"source": str(source), **report}
        _atomic_write_json(target, document)
        print(f"优化完成：{report['original_samples']}个样本 → {report['keyframes']}个关键点；"
              f"{report['original_duration_s']:.2f}s → {report['optimized_duration_s']:.2f}s（1倍回放）。")
        print(f"已保存：{target}\n起终点不变；中间路径及停顿已改变，请先慢放验证。")
        return 0
    except (OSError, ValueError, TypeError, SafetyError) as exc:
        print(f"轨迹优化失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
