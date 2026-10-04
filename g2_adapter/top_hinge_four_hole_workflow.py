#!/usr/bin/env python3
"""Independent two-hole retarget/replay followed by three CAD-spaced plate holes.

The original two-hole commands and modules are unchanged. No robot motion is
sent unless --allow-motion is explicit. The appended motion uses GDK's
right-end-effector pose stream: +20 mm Z, XY at fixed Z, -20 mm Z, repeated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import secrets
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from .cartesian_teach import load_cartesian_document, load_joint_document, quaternion_to_matrix
from .cartesian_teach_trajectory_gdk import (
    DEFAULT_URDF, _joint_map, replay_trajectory, verify_live_model,
)
from .control import CartesianPoseTarget, G2Controller, quaternion_angle_deg
from .g2_kinematics import UrdfKinematics
from .optimize_cartesian_teach_trajectory import main as optimize_main
from .retarget_two_hole_cartesian import (
    _dataset_dir, _reference_dir, endpoint_translation, read_target_pair,
    translate_document,
)
from .safety import ARM_POSITION_LIMITS
from .top_hinge_step_geometry import DEFAULT_STEP, HingeGeometry, measure_top_hinge

CURRENT_SESSION = Path("trajectories/two_hole_right_teach/current_session.txt")
LIFT_M = 0.020
SEGMENT_IK_SPACING_M = 0.005
FIRST_HOLE_MAX_INITIAL_ERROR_M = 0.003
FIRST_HOLE_ARRIVAL_TOLERANCE_M = 0.001
FIRST_HOLE_JOINT_TOLERANCE_RAD = 0.003


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _session_path(value: Path | None) -> Path:
    path = value or Path(CURRENT_SESSION.read_text(encoding="utf-8").strip())
    path = path.expanduser().resolve()
    if not (path / "right_cartesian_raw.json").is_file() or not (path / "right_cartesian_solved.json").is_file():
        raise ValueError(f"Session has no complete taught right-arm trajectory: {path}")
    _dataset_dir(path)
    return path


def _new_run_dirs(session: Path) -> tuple[Path, Path]:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    plan_dir = session / "four_hole_runs" / run_id
    capture_root = _dataset_dir(session) / "four_hole_relocalization" / run_id
    plan_dir.mkdir(parents=True, exist_ok=False)
    return plan_dir, capture_root


def _capture_reference(capture_root: Path, *, count: int, interval_s: float) -> Path:
    capture_root.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [sys.executable, "-m", "g2_adapter.localize_two_holes_head_rgbd_gdk",
         "--count", str(count), "--interval", str(interval_s),
         "--out-dir", str(capture_root)],
        check=True,
    )
    candidates = [path for path in capture_root.iterdir() if path.is_dir()]
    if len(candidates) != 1 or not (candidates[0] / f"sample_{count-1:04d}.json").is_file():
        raise ValueError(f"Expected one complete {count}-sample head RGB-D reference in {capture_root}")
    return candidates[0]


def _retarget_independent(
    session: Path, new_reference: Path, output: Path,
    geometry: HingeGeometry, cad_handedness: str,
) -> dict:
    old_reference = _reference_dir(_dataset_dir(session) / "reference", latest=False)
    source = session / "right_cartesian_raw.json"
    original_solved = session / "right_cartesian_solved.json"
    original, report = load_joint_document(_json(original_solved))
    original_raw = _json(source)
    if original.source_sha256 != _sha256(source):
        raise ValueError("Original taught solution does not match its raw path")
    acceleration_only = bool(report.errors) and all(
        error.startswith("joint acceleration ") for error in report.errors
    )
    if not report.replay_ready and not acceleration_only:
        raise ValueError("Original taught trajectory has validation errors: " + "; ".join(report.errors))
    urdf_sha = _sha256(DEFAULT_URDF)
    if original_raw.get("urdf_sha256") != urdf_sha or original.urdf_sha256 != urdf_sha:
        raise ValueError("Current URDF differs from the teaching trajectory")
    trajectory = load_cartesian_document(original_raw)
    old_pair = read_target_pair(old_reference)
    new_pair = read_target_pair(new_reference)
    taught_end = trajectory.samples[-1].position_m
    new_end, offset, diagnostics = _first_hole_translation(
        geometry, old_pair, new_pair, taught_end, cad_handedness,
    )
    shifted = translate_document(original_raw, offset)
    if not np.allclose(shifted["samples"][-1]["position_m"], new_end, rtol=0, atol=1e-9):
        raise ValueError("First-hole endpoint differs from the taught endpoint plus the two-target offset")
    if shifted["samples"][-1]["orientation_xyzw"] != original_raw["samples"][-1]["orientation_xyzw"]:
        raise ValueError("First-hole end-link orientation changed during translation")
    shifted["retargeting"] = {
        "method": "cad_first_hole_displacement_then_constant_xy_translation",
        "original_raw_sha256": _sha256(source),
        "original_reference": old_pair["reference_dir"],
        "new_reference": new_pair["reference_dir"],
        "translation_base_link_m": list(offset),
    }
    raw_output = output / "right_cartesian_raw.json"
    solved_output = output / "right_cartesian_solved.json"
    _save_json(raw_output, shifted)
    result = optimize_main([
        "--input", str(raw_output), "--output", str(solved_output),
        "--rate-hz", "100", "--position-tolerance-mm", "1.0",
        "--orientation-tolerance-deg", "1.0", "--jitter-position-mm", "2.0",
        "--jitter-orientation-deg", "1.0", "--outlier-position-mm", "8",
        "--outlier-orientation-deg", "5", "--pause-position-mm", "0.01",
        "--pause-orientation-deg", "0.01", "--median-window", "7",
        "--smoothing-window-ms", "1000", "--max-velocity-rad-s", "1.0",
        "--max-acceleration-rad-s2", "50.0", "--ik-position-tolerance-mm", "0.1",
        "--ik-reference-weight", "1e-5",
    ])
    if result != 0:
        raise ValueError("Retargeted taught trajectory failed offline IK validation")
    solved, solved_report = load_joint_document(_json(solved_output))
    if not solved_report.replay_ready or solved.source_sha256 != _sha256(raw_output):
        raise ValueError("New solved trajectory does not match the new raw path")
    summary = {
        "old_pair": old_pair, "new_pair": new_pair,
        "old_start_base_link_m": list(trajectory.samples[0].position_m),
        "old_end_base_link_m": list(taught_end),
        "new_start_base_link_m": shifted["samples"][0]["position_m"],
        "new_end_base_link_m": list(new_end),
        "constant_translation_base_link_m": list(offset),
        "trajectory_orientation_changed": False,
        "trajectory_z_changed": False,
        "diagnostics": diagnostics,
        "original_validation_errors": list(report.errors),
        "original_acceleration_exceeded_teach_limit": acceleration_only,
        "replay_ready": True,
    }
    _save_json(output / "plan_summary.json", summary)
    return summary


def _current_cad_rotation(
    geometry: HingeGeometry, old_pair: dict, new_pair: dict,
) -> tuple[np.ndarray, dict]:
    """Align the two recognized STEP anchors with the fresh RGB-D pair."""
    cad_vector = np.asarray(geometry.red_xy_mm) - np.asarray(geometry.assembly_table_xy_mm)
    observed = np.asarray(new_pair["tip_base_link_m"][:2]) - np.asarray(new_pair["table_base_link_m"][:2])
    cad_spacing = float(np.linalg.norm(cad_vector))
    observed_spacing = float(np.linalg.norm(observed) * 1000)
    if abs(observed_spacing - cad_spacing) > 15:
        raise ValueError(
            f"Recognized target spacing {observed_spacing:.1f} mm differs from assembly STEP "
            f"{cad_spacing:.1f} mm by more than 15 mm"
        )
    cad_yaw = math.atan2(cad_vector[1], cad_vector[0])
    observed_yaw = math.atan2(observed[1], observed[0])
    current_yaw = math.atan2(math.sin(observed_yaw - cad_yaw), math.cos(observed_yaw - cad_yaw))
    old_pair_yaw = float(old_pair["yaw_rad"])
    new_pair_yaw = float(new_pair["yaw_rad"])
    delta = math.atan2(math.sin(new_pair_yaw - old_pair_yaw), math.cos(new_pair_yaw - old_pair_yaw))
    cosine, sine = math.cos(current_yaw), math.sin(current_yaw)
    return np.array(((cosine, -sine), (sine, cosine)), dtype=float), {
        "cad_recognized_pair_spacing_mm": cad_spacing,
        "camera_recognized_pair_spacing_mm": observed_spacing,
        "measured_pair_yaw_change_deg": math.degrees(delta),
        "current_cad_to_base_yaw_deg": math.degrees(current_yaw),
        "orientation_source": "assembly_step_red_table_pair_to_fresh_head_rgbd_red_table_pair",
    }


def _apply_handedness(rotation: np.ndarray, geometry: HingeGeometry,
                      cad_handedness: str) -> np.ndarray:
    if cad_handedness not in {"proper", "mirrored"}:
        raise ValueError("CAD handedness must be proper or mirrored")
    if cad_handedness == "proper":
        return rotation
    axis = np.asarray(geometry.red_xy_mm) - np.asarray(geometry.assembly_table_xy_mm)
    axis /= np.linalg.norm(axis)
    perpendicular = np.array((-axis[1], axis[0]))
    reflection = np.outer(axis, axis) - np.outer(perpendicular, perpendicular)
    return rotation @ reflection


def _first_hole_translation(
    geometry: HingeGeometry, old_pair: dict, new_pair: dict,
    taught_endpoint_m: tuple[float, float, float], cad_handedness: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float], dict]:
    # Preserve the original pair-motion limits, but never rotate the taught
    # end-link position: its fixed offset from the physical hole is unknown.
    _, _, diagnostics = endpoint_translation(old_pair, new_pair, taught_endpoint_m)
    first_from_table_mm = (
        np.asarray(geometry.green_xy_mm) - np.asarray(geometry.assembly_table_xy_mm)
    )
    first_holes = []
    for pair in (old_pair, new_pair):
        rotation, _ = _current_cad_rotation(geometry, pair, pair)
        rotation = _apply_handedness(rotation, geometry, cad_handedness)
        first_holes.append(
            np.asarray(pair["table_base_link_m"][:2])
            + rotation @ first_from_table_mm / 1000.0
        )
    delta_xy = first_holes[1] - first_holes[0]
    new_endpoint = (
        float(taught_endpoint_m[0] + delta_xy[0]),
        float(taught_endpoint_m[1] + delta_xy[1]),
        float(taught_endpoint_m[2]),
    )
    offset = (float(delta_xy[0]), float(delta_xy[1]), 0.0)
    diagnostics = {
        **diagnostics,
        "first_hole_position_method": "assembly_two_targets_plus_cad_first_hole_xy",
        "cad_handedness": cad_handedness,
        "old_first_hole_base_link_xy_m": first_holes[0].tolist(),
        "new_first_hole_base_link_xy_m": first_holes[1].tolist(),
    }
    return new_endpoint, offset, diagnostics


def _append_stages(
    geometry: HingeGeometry, endpoint_xyz_m: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float],
    rotation: np.ndarray, mapping: dict, cad_handedness: str = "proper",
) -> tuple[list[dict], dict]:
    rotation = _apply_handedness(rotation, geometry, cad_handedness)
    mapping = {**mapping, "cad_handedness": cad_handedness,
               "cad_to_base_xy_determinant": float(np.linalg.det(rotation))}
    green_xy = np.asarray(endpoint_xyz_m[:2], dtype=float)
    base_z = float(endpoint_xyz_m[2])
    current_xy = green_xy
    stages = []
    for name, cad_xy in geometry.next_holes_xy_mm:
        cad_offset = np.asarray(cad_xy) - np.asarray(geometry.green_xy_mm)
        target_xy = green_xy + rotation @ cad_offset / 1000.0
        for stage, point in (
            ("lift_20mm", (current_xy[0], current_xy[1], base_z + LIFT_M)),
            ("xy_at_lifted_z", (target_xy[0], target_xy[1], base_z + LIFT_M)),
            ("descend_20mm", (target_xy[0], target_xy[1], base_z)),
        ):
            stages.append({
                "hole": name, "stage": stage,
                "position_base_link_m": [float(value) for value in point],
                "orientation_xyzw": list(orientation_xyzw),
            })
        current_xy = target_xy
    return stages, mapping


def _preflight_ik(
    stages: list[dict], start_xyz_m: tuple[float, float, float],
    orientation_xyzw: tuple[float, float, float, float], solved_path: Path,
) -> dict:
    solved, report = load_joint_document(_json(solved_path))
    if not report.replay_ready:
        raise ValueError("Retargeted trajectory is not replay ready")
    model = UrdfKinematics(DEFAULT_URDF)
    pose = np.eye(4)
    pose[:3, :3] = quaternion_to_matrix(orientation_xyzw)
    q = np.asarray(solved.points[-1].joint_positions_rad, dtype=float)
    previous = np.asarray(start_xyz_m, dtype=float)
    samples = 0
    max_error = 0.0
    for stage in stages:
        target = np.asarray(stage["position_base_link_m"], dtype=float)
        distance = float(np.linalg.norm(target - previous))
        for step in range(1, max(1, math.ceil(distance / SEGMENT_IK_SPACING_M)) + 1):
            fraction = step / max(1, math.ceil(distance / SEGMENT_IK_SPACING_M))
            pose[:3, 3] = previous + fraction * (target - previous)
            q, diagnostics = model.solve_pose(
                "base_link", "arm_r_end_link", solved.fixed_joint_positions,
                solved.joint_names, pose, q, joint_limits=ARM_POSITION_LIMITS[7:],
                reference_weight=1e-5, position_tolerance_m=0.0001,
                orientation_tolerance_rad=0.001,
            )
            samples += 1
            max_error = max(max_error, diagnostics.position_error_m)
        previous = target
    return {
        "sample_spacing_m": SEGMENT_IK_SPACING_M,
        "samples_checked": samples,
        "max_position_error_m": max_error,
        "all_stages_reachable": True,
    }


def build_append_plan(
    plan_dir: Path, *, geometry: HingeGeometry | None = None,
    cad_handedness: str = "proper",
) -> dict:
    geometry = geometry or measure_top_hinge(DEFAULT_STEP)
    raw = load_cartesian_document(_json(plan_dir / "right_cartesian_raw.json"))
    summary = _json(plan_dir / "plan_summary.json")
    if raw.side != "right" or raw.base_frame != "base_link":
        raise ValueError("Expected base_link right end-link taught path")
    endpoint = raw.samples[-1]
    if np.linalg.norm(np.asarray(endpoint.position_m) - np.asarray(summary["new_end_base_link_m"])) > 0.0002:
        raise ValueError("Retargeted endpoint and summary disagree")
    rotation, mapping = _current_cad_rotation(
        geometry, summary["old_pair"], summary["new_pair"],
    )
    stages, mapping = _append_stages(geometry, endpoint.position_m,
                                     endpoint.orientation_xyzw, rotation, mapping,
                                     cad_handedness)
    if len(stages) != 9 or any(
        quaternion_angle_deg(stage["orientation_xyzw"], endpoint.orientation_xyzw) > 1e-6
        for stage in stages
    ):
        raise ValueError("Expected nine unchanged-orientation lift/XY/descent stages")
    preflight = _preflight_ik(
        stages, endpoint.position_m, endpoint.orientation_xyzw,
        plan_dir / "right_cartesian_solved.json",
    )
    plan = {
        "schema_version": 1,
        "kind": "g2_right_top_hinge_three_followup_holes",
        "cad": geometry.as_dict(),
        "new_reference": summary["new_pair"]["reference_dir"],
        "retargeted_taught_endpoint_base_link_m": list(endpoint.position_m),
        "retargeted_taught_orientation_xyzw": list(endpoint.orientation_xyzw),
        "mapping": mapping,
        "move_order": [name for name, _ in geometry.next_holes_xy_mm],
        "lift_m": LIFT_M,
        "stages": stages,
        "offline_ik": preflight,
    }
    _save_json(plan_dir / "three_hole_append_plan.json", plan)
    return plan


def _settle_replay_final_joints(robot, solved, *, timeout_s: float = 5.0) -> float:
    """Hold the replay's last right-arm joint target until feedback converges."""
    names = solved.joint_names
    target = solved.points[-1].joint_positions_rad
    deadline = time.monotonic() + timeout_s
    stable = 0
    while time.monotonic() < deadline:
        status = robot.get_motion_control_status()
        if int(status.control_mode) != 1 or int(status.error_code) != 0:
            raise RuntimeError("Position control became unhealthy while settling the first hole")
        robot.move_arm_joint_servo(list(target), 0.01, 1, False)
        current = _joint_map(robot)
        error = max(abs(current[name] - value) for name, value in zip(names, target))
        stable = stable + 1 if error <= FIRST_HOLE_JOINT_TOLERANCE_RAD else 0
        if stable >= 3:
            return error
        time.sleep(0.01)
    raise RuntimeError("Right arm did not settle at the taught final joint target")


def _followup_target(nominal_xyz: np.ndarray, anchor: np.ndarray,
                     original: np.ndarray,
                     stream_initial_pose: CartesianPoseTarget) -> CartesianPoseTarget:
    xyz = anchor + (nominal_xyz - original)
    return CartesianPoseTarget(
        tuple(float(value) for value in xyz),
        stream_initial_pose.orientation_xyzw,
    )


def _replay_then_followups(
    plan_dir: Path, append_plan: dict, *,
    speed_scale: float, start_velocity_rad_s: float,
    cartesian_speed_mps: float,
    max_velocity_rad_s: float, max_acceleration_rad_s2: float,
) -> None:
    import agibot_gdk as gdk

    raw_path = plan_dir / "right_cartesian_raw.json"
    solved_path = plan_dir / "right_cartesian_solved.json"
    solved, report = load_joint_document(_json(solved_path))
    if not report.replay_ready or solved.source_sha256 != _sha256(raw_path):
        raise ValueError("Shifted trajectory was not validated against its raw path")
    if solved.urdf_sha256 != _sha256(DEFAULT_URDF):
        raise ValueError("URDF changed since shifted trajectory optimization")
    if gdk.gdk_init() != gdk.GDKRes.kSuccess:
        raise RuntimeError("GDK initialization failed")
    stream = None
    try:
        robot = gdk.Robot()
        model = UrdfKinematics(DEFAULT_URDF)
        verify_live_model(model, robot, "right")
        replay_trajectory(
            gdk, robot, solved, report, speed_scale=speed_scale,
            start_velocity_rad_s=start_velocity_rad_s,
            max_velocity_rad_s=max_velocity_rad_s,
            max_acceleration_rad_s2=max_acceleration_rad_s2,
        )
        final_joint_error = _settle_replay_final_joints(robot, solved)
        print(f"首孔末端关节收敛误差={final_joint_error:.5f} rad", flush=True)
        controller = G2Controller(gdk, robot, allow_motion=True)
        controller.set_position_control_phase(True)
        measured = controller.read_end_effector_pose("arm_r_end_link")
        planned_xyz = np.asarray(append_plan["retargeted_taught_endpoint_base_link_m"], dtype=float)
        position_gap = float(np.linalg.norm(np.asarray(measured.position_m) - planned_xyz))
        angle_gap = quaternion_angle_deg(
            measured.orientation_xyzw, append_plan["retargeted_taught_orientation_xyzw"]
        )
        print(
            f"首孔回放终点：计划={planned_xyz.tolist()}，实测={list(measured.position_m)}，"
            f"位置误差={position_gap*1000:.2f} mm，姿态误差={angle_gap:.3f}°",
            flush=True,
        )
        if position_gap > FIRST_HOLE_MAX_INITIAL_ERROR_M or angle_gap > 0.2:
            raise RuntimeError(
                f"First replay endpoint not reached precisely: {position_gap*1000:.1f} mm, {angle_gap:.2f} deg"
            )
        gripper_names, gripper_positions = controller.read_gripper_state("right")
        if len(gripper_names) != 1 or len(gripper_positions) != 1:
            raise RuntimeError("Cannot read right gripper state before continued hole motion")
        # The stream locks this exact quaternion at startup. Feedback after
        # settling may differ slightly; never substitute it into later targets.
        held_orientation = measured.orientation_xyzw
        stream = controller.start_right_end_effector_position_stream(
            measured, rate_hz=50.0, life_time_s=0.10,
            max_translation_velocity_mps=cartesian_speed_mps,
            allow_orientation_change=False, control_mode=1,
        )
        stream.set_right_gripper_target(gripper_positions[0])
        # Joint replay can finish inside its 0.03 rad joint tolerance while
        # the tool is still millimetres from the taught Cartesian endpoint.
        # Settle at the exact retargeted endpoint before the CAD follow-ups.
        first_hole_target = CartesianPoseTarget(
            tuple(float(value) for value in planned_xyz), held_orientation,
        )
        stream.update_target(first_hole_target)
        settled = stream.wait_until_actual_target(
            first_hole_target,
            position_gap / cartesian_speed_mps + 8.0,
            position_tolerance_m=FIRST_HOLE_ARRIVAL_TOLERANCE_M,
            orientation_tolerance_deg=0.5, stable_samples=3,
        )
        if settled is None:
            raise RuntimeError("First taught hole did not settle within 1 mm; follow-up holes were not started")
        settled_gap = float(np.linalg.norm(np.asarray(settled.position_m) - planned_xyz))
        print(f"首孔精确到位：剩余位置误差={settled_gap*1000:.2f} mm", flush=True)
        # Rebase follow-up displacements on the actual reached first hole.
        anchor = np.asarray(settled.position_m, dtype=float)
        original = planned_xyz
        previous = anchor
        for item in append_plan["stages"]:
            nominal = np.asarray(item["position_base_link_m"], dtype=float)
            target = _followup_target(nominal, anchor, original, measured)
            target_xyz = np.asarray(target.position_m, dtype=float)
            # The GDK stream sends a Cartesian pose at 50 Hz and limits each
            # translational step. Orientation is held byte-for-byte constant.
            distance = float(np.linalg.norm(target_xyz - previous))
            stream.update_target(target)
            actual = stream.wait_until_actual_target(
                target, distance / cartesian_speed_mps + 8.0,
                position_tolerance_m=0.0015,
                orientation_tolerance_deg=0.5, stable_samples=3,
            )
            if actual is None:
                raise RuntimeError(f"Right arm did not reach {item['hole']} / {item['stage']}; stopped")
            print(
                f"已到位：{item['hole']} {item['stage']} "
                f"base_link={[round(value, 5) for value in actual.position_m]}",
                flush=True,
            )
            previous = target_xyz
    finally:
        if stream is not None:
            stream.stop()
        gdk.gdk_release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="Defaults to current_session.txt")
    parser.add_argument("--new-reference", type=Path,
                        help="Use an existing 12-sample head RGB-D reference; otherwise capture now")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--start-velocity-rad-s", type=float, default=0.15)
    parser.add_argument("--cartesian-speed-mm-s", type=float, default=10.0)
    parser.add_argument("--cad-handedness", choices=("proper", "mirrored"),
                        help="Verified against the physical first hole; required for motion")
    parser.add_argument("--max-velocity-rad-s", type=float, default=0.5)
    parser.add_argument("--max-acceleration-rad-s2", type=float, default=30.0)
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args()
    if args.count != 12 or args.interval < 0:
        parser.error("The existing retargeter requires exactly 12 localized samples")
    if not all(math.isfinite(value) and value > 0 for value in (
        args.speed_scale, args.start_velocity_rad_s, args.cartesian_speed_mm_s,
        args.max_velocity_rad_s, args.max_acceleration_rad_s2,
    )):
        parser.error("All velocities and speed scales must be finite and positive")
    if args.cartesian_speed_mm_s > 30:
        parser.error("Local Cartesian speed must not exceed 30 mm/s")
    if args.allow_motion and args.new_reference is not None:
        parser.error("Motion requires a fresh live head RGB-D capture; omit --new-reference")
    if args.allow_motion and args.cad_handedness is None:
        parser.error("Motion requires --cad-handedness proper or mirrored after checking the physical first hole")
    session = _session_path(args.session)
    plan_dir, capture_root = _new_run_dirs(session)
    try:
        reference = (args.new_reference.expanduser().resolve() if args.new_reference
                     else _capture_reference(capture_root, count=args.count, interval_s=args.interval))
        geometry = measure_top_hinge(DEFAULT_STEP)
        summary = _retarget_independent(
            session, reference, plan_dir, geometry, args.cad_handedness or "proper",
        )
        print(
            "首孔目标：示教末端=" + str(summary["old_end_base_link_m"])
            + "，本次平移=" + str(summary["constant_translation_base_link_m"])
            + "，新末端=" + str(summary["new_end_base_link_m"]),
            flush=True,
        )
        append = build_append_plan(
            plan_dir, geometry=geometry, cad_handedness=args.cad_handedness or "proper",
        )
        _save_json(plan_dir / "workflow_info.json", {
            "session": str(session), "new_reference": str(reference),
            "retargeted_plan": str(plan_dir),
            "motion_requested": args.allow_motion,
            "speed_scale": args.speed_scale,
            "start_velocity_rad_s": args.start_velocity_rad_s,
            "cartesian_speed_mm_s": args.cartesian_speed_mm_s,
            "cad_handedness": args.cad_handedness,
            "initial_retarget_translation_base_link_m": summary["constant_translation_base_link_m"],
        })
        print(f"四孔独立计划：{plan_dir}", flush=True)
        print(json.dumps({
            "mapping": append["mapping"],
            "next_holes": append["move_order"],
            "stages": len(append["stages"]),
            "offline_ik": append["offline_ik"],
        }, ensure_ascii=False, indent=2), flush=True)
        if args.allow_motion:
            _replay_then_followups(
                plan_dir, append, speed_scale=args.speed_scale,
                start_velocity_rad_s=args.start_velocity_rad_s,
                cartesian_speed_mps=args.cartesian_speed_mm_s / 1000,
                max_velocity_rad_s=args.max_velocity_rad_s,
                max_acceleration_rad_s2=args.max_acceleration_rad_s2,
            )
        else:
            print("仅完成相机定位与离线计划；未发送机器人运动命令。", flush=True)
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Four-hole workflow failed; run kept at {plan_dir}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
