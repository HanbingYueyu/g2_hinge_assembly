"""RGB-only CAD landmark pose estimation. No depth or robot-control imports.

Profiles contain measured STEP hole centres in metres, in a common model frame.
The matcher requires six of seven holes: ambiguous patterns fail closed.
"""
from __future__ import annotations

import itertools
import time
import json
from pathlib import Path

import cv2
import numpy as np


def load_profile(path):
    profile = json.loads(Path(path).read_text())
    points = np.asarray(profile["hole_centers_m"], dtype=np.float64)
    target = np.asarray(profile["grasp_center_m"], dtype=np.float64)
    if points.shape != (7, 3) or target.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(target).all():
        raise ValueError("STEP profile requires seven finite 3D hole centres and a grasp centre")
    if np.linalg.matrix_rank(points - points.mean(axis=0), tol=1e-6) < 2:
        raise ValueError("STEP hole pattern is degenerate")
    return profile


def camera_matrix(intrinsic):
    k = np.array([[intrinsic["fx"], 0, intrinsic["ppx"]],
                  [0, intrinsic["fy"], intrinsic["ppy"]], [0, 0, 1]], dtype=float)
    distortion = np.asarray(intrinsic.get("distortion", []), dtype=float)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or not np.isfinite(distortion).all():
        raise ValueError("invalid camera calibration")
    if distortion.size not in (0, 4, 5, 8, 12, 14):
        raise ValueError("unsupported OpenCV distortion coefficients")
    return k, distortion


def load_alignment_reference(path, profile):
    reference = json.loads(Path(path).read_text())
    pixel = np.asarray(reference["target_pixel_px"], dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all():
        raise ValueError("invalid STEP alignment pixel")
    if reference.get("source_sha256") != profile["source_sha256"] or reference.get("grasp_center_m") != profile["grasp_center_m"]:
        raise ValueError("STEP model/target changed: recapture alignment reference")
    if reference.get("sample_count", 0) < 20:
        raise ValueError("STEP alignment requires 20 stationary frames")
    return reference


def servo_pixel(item):
    if "step_pose" not in item:
        return item["center_px"]
    pose = item["step_pose"]
    if not pose.get("accepted"):
        raise ValueError("STEP pose unavailable")
    return pose["grasp_pixel_px"]


def detect_hole_centres(bgr, mask):
    """Ellipse candidates from RGB edges inside the filled segmentation contour.

    Filling the silhouette deliberately retains holes absent from a YOLO mask.
    Duplicate inner/outer edges are merged; elongated reflections are rejected.
    """
    silhouette = np.zeros(mask.shape, np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.empty((0, 2))
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(silhouette, [largest], -1, 255, -1)
    area = cv2.contourArea(largest)
    if area < 100:
        return np.empty((0, 2))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 130)
    edges[silhouette == 0] = 0
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        if len(contour) < 12:
            continue
        (x, y), (a, b), angle = cv2.fitEllipse(contour)
        if min(a, b) < 5 or min(a, b) / max(a, b) < .55 or not .0008 * area < a * b < .09 * area:
            continue
        theta = np.deg2rad(angle)
        delta = contour[:, 0, :].astype(float) - [x, y]
        local = delta @ np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        radii = np.sqrt((local[:, 0] / (a / 2)) ** 2 + (local[:, 1] / (b / 2)) ** 2)
        if np.mean(np.abs(radii - 1)) > .12 or cv2.contourArea(contour) < .5 * np.pi * a * b / 4:
            continue
        if not (0 <= round(x) < mask.shape[1] and 0 <= round(y) < mask.shape[0]) or not silhouette[round(y), round(x)]:
            continue
        candidates.append((float(np.mean(np.abs(radii - 1))), np.array([x, y]), min(a, b)))
    merged = []
    for _, point, diameter in sorted(candidates, key=lambda value: value[0]):
        if all(np.linalg.norm(point - other) > max(3, .25 * diameter) for other in merged):
            merged.append(point)
    # Too many candidates is an ambiguous observation, not permission to guess.
    return np.asarray(merged, dtype=float).reshape(-1, 2)


def solve_correspondences(object_points, image_points, intrinsic, target):
    """Solve and validate a *known* correspondence, preserving distortion."""
    points = np.asarray(object_points, dtype=np.float64)
    pixels = np.asarray(image_points, dtype=np.float64)
    k, dist = camera_matrix(intrinsic)
    ok, rvec, tvec = cv2.solvePnP(points, pixels, k, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(points, pixels, k, dist, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]
    camera_points = points @ rotation.T + tvec.reshape(3)
    camera_target = rotation @ np.asarray(target) + tvec.reshape(3)
    if np.min(camera_points[:, 2]) <= 0 or not .05 < camera_target[2] < 2:
        return None
    projected = cv2.projectPoints(points, rvec, tvec, k, dist)[0].reshape(-1, 2)
    errors = np.linalg.norm(projected - pixels, axis=1)
    if errors.mean() > 2 or errors.max() > 4:
        return None
    target_px = cv2.projectPoints(np.asarray(target, dtype=float).reshape(1, 3), rvec, tvec, k, dist)[0].reshape(2)
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, tvec.reshape(3)
    return {"camera_from_model": transform.tolist(), "grasp_xyz_camera_m": camera_target.tolist(),
            "grasp_pixel_px": target_px.tolist(), "mean_error_px": float(errors.mean()),
            "max_error_px": float(errors.max()), "matched_pixels": pixels.tolist()}


def _match_subset(profile, pixels, intrinsic, model_indices):
    """Similarity seeds followed by metric PnP; limited to near-frontal views."""
    points = np.asarray(profile["hole_centers_m"], dtype=float)[list(model_indices)]
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    _, _, axes = np.linalg.svd(points - points.mean(axis=0))
    plane = (points - points.mean(axis=0)) @ axes[:2].T
    distances = np.linalg.norm(plane[:, None] - plane[None, :], axis=2)
    i, j = np.unravel_index(distances.argmax(), distances.shape)
    solutions, seen = [], set()
    # Both plane handednesses are considered. Distinct good poses are rejected.
    for sign in (1, -1):
        model = plane * [1, sign]
        source = complex(*(model[j] - model[i]))
        for a, b in itertools.permutations(range(len(pixels)), 2):
            scale = complex(*(pixels[b] - pixels[a])) / source
            delta = model - model[i]
            z = (delta[:, 0] + 1j * delta[:, 1]) * scale
            predicted = np.column_stack((z.real, z.imag)) + pixels[a]
            residual = np.linalg.norm(predicted[:, None] - pixels[None, :], axis=2)
            indices = residual.argmin(axis=1)
            if len(set(indices)) != len(points) or residual[np.arange(len(points)), indices].max() > .10 * np.linalg.norm(pixels[b] - pixels[a]):
                continue
            key = tuple(indices)
            if key in seen:
                continue
            seen.add(key)
            result = solve_correspondences(points, pixels[indices], intrinsic, profile["grasp_center_m"])
            if result is not None:
                transform = np.asarray(result["camera_from_model"])
                centre_camera = transform[:3, :3] @ points.mean(axis=0) + transform[:3, 3]
                # The selected top rims are seen from STEP +Z. Reject the
                # backside solution of the nearly symmetric hole pattern.
                if np.dot(transform[:3, 2], centre_camera) < 0:
                    solutions.append(result)
    if not solutions:
        return {"accepted": False, "reason": "pnp_rejected", "candidate_count": len(pixels)}
    solutions.sort(key=lambda result: result["mean_error_px"])
    best = solutions[0]
    for other in solutions[1:]:
        if other["mean_error_px"] <= best["mean_error_px"] + .5:
            if np.linalg.norm(np.array(other["grasp_xyz_camera_m"]) - best["grasp_xyz_camera_m"]) > .002:
                return {"accepted": False, "reason": "ambiguous_pose"}
    return {**best, "accepted": True, "reason": "matched", "candidate_count": len(pixels),
            "matched_model_indices": list(model_indices), "matched_count": len(points),
            "geometry_verified": bool(profile.get("verified", False)), "motion_enabled": False}


def match_holes(profile, pixels, intrinsic):
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if not 6 <= len(pixels) <= 20:
        return {"accepted": False, "reason": "need_6_to_20_hole_candidates", "candidate_count": len(pixels)}
    if len(pixels) >= 7:
        full = _match_subset(profile, pixels, intrinsic, tuple(range(7)))
        if full["accepted"] or full["reason"] == "ambiguous_pose":
            return full
    solutions = []
    for subset in itertools.combinations(range(7), 6):
        result = _match_subset(profile, pixels, intrinsic, subset)
        if result["reason"] == "ambiguous_pose":
            return {"accepted": False, "reason": "ambiguous_pose", "candidate_count": len(pixels)}
        if result["accepted"]:
            solutions.append(result)
    if not solutions:
        return {"accepted": False, "reason": "pnp_rejected", "candidate_count": len(pixels)}
    solutions.sort(key=lambda result: result["mean_error_px"])
    best = solutions[0]
    for other in solutions[1:]:
        if other["mean_error_px"] <= best["mean_error_px"] + .5:
            if np.linalg.norm(np.asarray(other["grasp_xyz_camera_m"]) - best["grasp_xyz_camera_m"]) > .002:
                return {"accepted": False, "reason": "ambiguous_pose", "candidate_count": len(pixels)}
    return best


class OcclusionTracker:
    """One physical hinge. Never seed a new identity from a four-hole fit."""
    def __init__(self):
        self.pose = None

    def update(self, profile, pixels, intrinsic, now):
        full = match_holes(profile, pixels, intrinsic)
        if full["accepted"]:
            self.pose = full
            return full
        if self.pose is None:
            return full
        pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
        if not 4 <= len(pixels) <= 20 or full["reason"] == "ambiguous_pose":
            return full
        points = np.asarray(profile["hole_centers_m"])
        transform = np.asarray(self.pose["camera_from_model"])
        k, dist = camera_matrix(intrinsic)
        predicted = cv2.projectPoints(points, cv2.Rodrigues(transform[:3, :3])[0], transform[:3, 3], k, dist)[0].reshape(-1, 2)
        distances = np.linalg.norm(predicted[:, None] - pixels[None, :], axis=2)
        separation = np.linalg.norm(predicted[:, None] - predicted[None, :], axis=2)
        np.fill_diagonal(separation, np.inf)
        gate = min(10., .3 * separation.min())
        pairs = []
        for model_index, row in enumerate(distances):
            candidate = int(row.argmin())
            # Mutual nearest neighbour, unique within a tight projected gate.
            if row[candidate] <= gate and distances[:, candidate].argmin() == model_index and np.count_nonzero(row <= gate) == 1 and np.count_nonzero(distances[:, candidate] <= gate) == 1:
                pairs.append((model_index, candidate))
        failed = {"accepted": False, "reason": "occluded_correspondence_rejected", "candidate_count": len(pixels)}
        if len(pairs) < 4:
            return failed
        ids, observations = zip(*pairs)
        subset = points[list(ids)]
        singular = np.linalg.svd(subset - subset.mean(axis=0), compute_uv=False)
        if singular[1] < .006 or singular[1] / singular[0] < .2:
            return failed
        result = solve_correspondences(subset, pixels[list(observations)], intrinsic, profile["grasp_center_m"])
        if result is None:
            return failed
        new = np.asarray(result["camera_from_model"])
        angle = np.arccos(np.clip((np.trace(new[:3, :3] @ transform[:3, :3].T) - 1) / 2, -1, 1))
        if angle > np.deg2rad(5) or np.linalg.norm(np.asarray(result["grasp_xyz_camera_m"]) - self.pose["grasp_xyz_camera_m"]) > .005:
            return failed
        result.update(accepted=True, reason="occlusion_tracked", matched_model_indices=list(ids),
                      matched_count=len(ids), candidate_count=len(pixels), motion_enabled=False,
                      geometry_verified=bool(profile.get("verified")))
        self.pose = result
        return result


class OcclusionTrackerPool:
    """Associate hinges by segmentation centre, never by changing YOLO index."""
    def __init__(self):
        self.entries = []

    def update(self, profile, holes, centres, intrinsic, now=None):
        now = time.monotonic() if now is None else now
        old = list(self.entries)
        new, results, used = [], [], set()
        associated = set()
        for hole_set, centre in zip(holes, centres):
            centre = np.asarray(centre)
            possible = [i for i, (c, _, _) in enumerate(old) if np.linalg.norm(c - centre) < 30]
            associated.update(possible)
            tracker = OcclusionTracker()
            if len(possible) == 1 and possible[0] not in used:
                i = possible[0]
                # If two current hinges could claim one prior identity, reset both.
                if sum(np.linalg.norm(np.asarray(c) - old[i][0]) < 30 for c in centres) == 1:
                    tracker = old[i][1]
                    used.add(i)
            results.append(tracker.update(profile, hole_set, intrinsic, now))
            new.append((centre, tracker, now))
        # Missing detections retain the prior pose, but never emit a result
        # or command. Reappearance still needs spatial and geometric checks.
        self.entries = new + [entry for i, entry in enumerate(old) if i not in associated]
        return results
