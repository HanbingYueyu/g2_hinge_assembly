"""Translate a taught right-arm Cartesian path using two head RGB-D target points.

The target-pair yaw is used only to locate the new endpoint. Every Cartesian
sample receives the same XY translation; its Z and orientation are unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from .cartesian_teach import load_cartesian_document, load_joint_document
from .cartesian_teach_trajectory_gdk import DEFAULT_URDF
from .optimize_cartesian_teach_trajectory import main as optimize_main
from .two_hole_rgbd import project_color_pixel_to_base_z


TABLE = "irregular_table_hole"
TIP = "round_hinge_tip_hole"


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dataset_dir(session: Path) -> Path:
    if session.parent.name != "two_hole_right_teach" or session.parent.parent.name != "trajectories":
        raise ValueError(f"Unexpected teaching session path: {session}")
    directory = session.parent.parent.parent / "datasets" / "two_hole_right_teach" / session.name
    if not directory.is_dir():
        raise ValueError(f"Dataset directory does not exist: {directory}")
    return directory


def _reference_dir(root: Path, *, latest: bool) -> Path:
    if (root / "sample_0000.json").is_file():
        return root
    candidates = sorted({p.parent for p in root.rglob("sample_0011.json")}, key=lambda p: str(p))
    if not candidates:
        raise ValueError(f"No complete 12-sample reference in {root}")
    if not latest and len(candidates) != 1:
        raise ValueError(f"Expected one teaching reference in {root}; found {len(candidates)}")
    return candidates[-1]


def read_target_pair(reference: Path) -> dict:
    files = sorted(reference.glob("sample_*.json"))
    if len(files) != 12:
        raise ValueError(f"Expected exactly 12 localized samples in {reference}; found {len(files)}")
    calibration = json.loads((reference / "calibration.json").read_text(encoding="utf-8"))
    if calibration.get("camera") != "GDK kHeadColor + kHeadDepth":
        raise ValueError(f"Not a head RGB-D calibration: {reference}")
    coordinates: dict[str, list[tuple[float, float, float]]] = {TABLE: [], TIP: []}
    measured_tip_z_disagreement = []
    for path in files:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("camera") != "GDK kHeadColor + kHeadDepth RGB-D":
            raise ValueError(f"Not a head RGB-D localization sample: {path}")
        if record.get("tf", {}).get("mode") != "exact_image_time":
            raise ValueError(f"Reference lacks image-time base_link TF: {path}")
        for key, class_id in ((TABLE, 0), (TIP, 1)):
            target = record["targets"][key]
            if target.get("class_id") != class_id:
                raise ValueError(f"Target class mismatch in {path}")
            values = tuple(float(x) for x in target["point_base_link_m"])
            if len(values) != 3 or not all(math.isfinite(x) for x in values):
                raise ValueError(f"Invalid base_link point in {path}")
            if key == TABLE:
                coordinates[key].append(values)
            else:
                table_z = coordinates[TABLE][-1][2]
                projected = project_color_pixel_to_base_z(
                    target["center_head_color_px"], table_z,
                    calibration["color_intrinsic"], calibration["head_from_color"],
                    record["base_from_head_link3"],
                )
                coordinates[key].append(projected)
                measured_tip_z_disagreement.append(abs(values[2] - table_z))
    result = {key: tuple(statistics.median(p[i] for p in points) for i in range(3))
              for key, points in coordinates.items()}
    max_xy_deviation = max(
        math.hypot(p[0] - result[key][0], p[1] - result[key][1])
        for key, points in coordinates.items() for p in points
    )
    if max_xy_deviation > 0.012:
        raise ValueError(f"Target localization XY spread {max_xy_deviation*1000:.1f} mm exceeds 12 mm")
    table, tip = result[TABLE], result[TIP]
    vector = (tip[0] - table[0], tip[1] - table[1])
    spacing = math.hypot(*vector)
    if not 0.05 <= spacing <= 0.20:
        raise ValueError(f"Unexpected target spacing: {spacing*1000:.1f} mm")
    return {
        "reference_dir": str(reference.resolve()),
        "table_base_link_m": list(table),
        "tip_base_link_m": list(tip),
        "midpoint_xy_m": [(table[0] + tip[0])/2, (table[1] + tip[1])/2],
        "yaw_rad": math.atan2(vector[1], vector[0]),
        "spacing_m": spacing,
        "max_xy_deviation_m": max_xy_deviation,
        "tip_raw_depth_z_disagreement_m": statistics.median(measured_tip_z_disagreement),
        "tip_position_method": "head_color_ray_intersect_table_base_link_z",
    }


def endpoint_translation(
    old_pair: dict, new_pair: dict, taught_endpoint_m: tuple[float, float, float],
    *, max_pair_translation_m: float = 0.12,
    max_yaw_change_deg: float = 90.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float], dict]:
    old_mid, new_mid = old_pair["midpoint_xy_m"], new_pair["midpoint_xy_m"]
    midpoint_translation = math.dist(old_mid, new_mid)
    if midpoint_translation > max_pair_translation_m:
        raise ValueError(f"Target midpoint moved {midpoint_translation*1000:.1f} mm; limit is {max_pair_translation_m*1000:.1f} mm")
    spacing_change = abs(new_pair["spacing_m"] - old_pair["spacing_m"])
    if spacing_change > 0.015:
        raise ValueError(f"Target spacing changed {spacing_change*1000:.1f} mm; expected a rigid pair")
    for key in ("table_base_link_m", "tip_base_link_m"):
        z_change = abs(new_pair[key][2] - old_pair[key][2])
        if z_change > 0.02:
            raise ValueError(f"{key} Z changed {z_change*1000:.1f} mm; expected fixed height")
    yaw = math.atan2(
        math.sin(new_pair["yaw_rad"] - old_pair["yaw_rad"]),
        math.cos(new_pair["yaw_rad"] - old_pair["yaw_rad"]),
    )
    if abs(math.degrees(yaw)) > max_yaw_change_deg:
        raise ValueError(f"Target yaw changed {math.degrees(yaw):.1f} deg; limit is {max_yaw_change_deg:.1f} deg")
    local_x, local_y = taught_endpoint_m[0] - old_mid[0], taught_endpoint_m[1] - old_mid[1]
    cosine, sine = math.cos(yaw), math.sin(yaw)
    new_endpoint = (
        new_mid[0] + cosine*local_x - sine*local_y,
        new_mid[1] + sine*local_x + cosine*local_y,
        taught_endpoint_m[2],
    )
    offset = (new_endpoint[0] - taught_endpoint_m[0],
              new_endpoint[1] - taught_endpoint_m[1], 0.0)
    return new_endpoint, offset, {
        "pair_midpoint_translation_m": midpoint_translation,
        "pair_spacing_change_m": spacing_change,
        "pair_yaw_change_deg": math.degrees(yaw),
    }


def translate_document(document: dict, offset: tuple[float, float, float]) -> dict:
    trajectory = load_cartesian_document(document)
    if trajectory.side != "right" or trajectory.base_frame != "base_link":
        raise ValueError("Expected a right-arm base_link Cartesian trajectory")
    shifted = copy.deepcopy(document)
    for sample in shifted["samples"]:
        x, y, z = sample["position_m"]
        sample["position_m"] = [x + offset[0], y + offset[1], z]
    load_cartesian_document(shifted)
    return shifted


def plan(session: Path, *, new_reference: Path | None = None) -> Path:
    session = session.expanduser().resolve()
    (session / "current_retargeted_plan.txt").unlink(missing_ok=True)
    dataset_dir = _dataset_dir(session)
    old_reference = _reference_dir(dataset_dir / "reference", latest=False)
    new_reference = new_reference or _reference_dir(dataset_dir / "relocalization", latest=True)
    source = session / "right_cartesian_raw.json"
    original_solved = session / "right_cartesian_solved.json"
    source_document = json.loads(source.read_text(encoding="utf-8"))
    original_joint_document = json.loads(original_solved.read_text(encoding="utf-8"))
    original_joint, report = load_joint_document(original_joint_document)
    if not report.replay_ready or original_joint.source_sha256 != _hash_file(source):
        raise ValueError("Teaching trajectory is not a verified solution of this raw path")
    urdf_hash = _hash_file(DEFAULT_URDF)
    if source_document.get("urdf_sha256") != urdf_hash or original_joint.urdf_sha256 != urdf_hash:
        raise ValueError("Current URDF differs from the teaching trajectory")
    trajectory = load_cartesian_document(source_document)
    old_pair = read_target_pair(old_reference)
    new_pair = read_target_pair(new_reference)
    old_endpoint = trajectory.samples[-1].position_m
    new_endpoint, offset, diagnostics = endpoint_translation(old_pair, new_pair, old_endpoint)
    shifted = translate_document(source_document, offset)
    shifted["retargeting"] = {
        "method": "two_target_yaw_locates_endpoint_then_constant_xy_translation",
        "original_raw_sha256": _hash_file(source),
        "original_reference": old_pair["reference_dir"],
        "new_reference": new_pair["reference_dir"],
        "translation_base_link_m": list(offset),
    }
    output = session / "retargeted" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    raw_output = output / "right_cartesian_raw.json"
    raw_output.write_text(json.dumps(shifted, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    solved_output = output / "right_cartesian_solved.json"
    result = optimize_main([
        "--input", str(raw_output), "--output", str(solved_output),
        "--rate-hz", "100", "--position-tolerance-mm", "1.0",
        "--orientation-tolerance-deg", "1.0", "--jitter-position-mm", "2.0",
        "--jitter-orientation-deg", "1.0", "--outlier-position-mm", "8",
        "--outlier-orientation-deg", "5", "--pause-position-mm", "0.01",
        "--pause-orientation-deg", "0.01", "--median-window", "7",
        "--smoothing-window-ms", "1000", "--max-velocity-rad-s", "1.0",
        "--max-acceleration-rad-s2", "20.0", "--ik-position-tolerance-mm", "0.1",
        "--ik-reference-weight", "1e-5",
    ])
    summary = {
        "old_pair": old_pair,
        "new_pair": new_pair,
        "old_start_base_link_m": list(trajectory.samples[0].position_m),
        "old_end_base_link_m": list(old_endpoint),
        "new_start_base_link_m": shifted["samples"][0]["position_m"],
        "new_end_base_link_m": list(new_endpoint),
        "constant_translation_base_link_m": list(offset),
        "trajectory_orientation_changed": False,
        "trajectory_z_changed": False,
        "ik_position_tolerance_mm": 0.1,
        "ik_reference_weight": 1e-5,
        "diagnostics": diagnostics,
        "replay_ready": result == 0,
    }
    (output / "plan_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if result != 0:
        raise RuntimeError(f"Shifted trajectory failed IK/validation; see {output}")
    (session / "current_retargeted_plan.txt").write_text(str(output) + "\n", encoding="utf-8")
    print(f"Retargeted plan saved to {output}", flush=True)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--new-reference", type=Path,
                        help="A 12-sample directory; defaults to the latest session/relocalization subdir")
    args = parser.parse_args()
    try:
        plan(args.session, new_reference=args.new_reference)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Two-hole retargeting failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
