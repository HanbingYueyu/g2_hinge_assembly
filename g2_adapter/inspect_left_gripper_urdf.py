"""Read-only left gripper FK and closed-loop consistency audit."""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def vector(text, default="0 0 0"):
    result = np.fromstring(text or default, sep=" ")
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("invalid URDF vector")
    return result


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def origin(element):
    result = np.eye(4)
    if element is None:
        return result
    r, p, y = vector(element.get("rpy"))
    result[:3, :3] = rotation([0, 0, 1], y) @ rotation([0, 1, 0], p) @ rotation([1, 0, 0], r)
    result[:3, 3] = vector(element.get("xyz"))
    return result


def audit(path, opening_rad):
    if not np.isfinite(opening_rad) or not -np.pi / 4 <= opening_rad <= 0:
        raise ValueError("opening must be in [-pi/4, 0] rad")
    root = ET.parse(path).getroot()
    joints = {joint.get("name"): joint for joint in root.findall("joint")}
    values = {"idx31_gripper_l_inner_joint1": opening_rad}

    def angle(name):
        if name in values:
            return values[name]
        joint = joints[name]
        mimic = joint.find("mimic")
        if mimic is not None:
            values[name] = angle(mimic.get("joint")) * float(mimic.get("multiplier", "1")) + float(mimic.get("offset", "0"))
            return values[name]
        if joint.get("type") != "fixed":
            raise ValueError(f"missing joint value: {name}")
        return 0.0

    frames = {"arm_l_end_link": np.eye(4)}
    for _ in range(len(joints)):
        changed = False
        for name, joint in joints.items():
            parent, child = joint.find("parent").get("link"), joint.find("child").get("link")
            if parent not in frames or child in frames:
                continue
            transform = origin(joint.find("origin"))
            if joint.get("type") == "revolute":
                transform[:3, :3] = transform[:3, :3] @ rotation(vector(joint.find("axis").get("xyz")), angle(name))
            elif joint.get("type") != "fixed":
                raise ValueError("unsupported joint type")
            frames[child] = frames[parent] @ transform
            changed = True
        if not changed:
            break
    loops = []
    for loop in root.findall("loop_joint"):
        a, b = loop.find("link1"), loop.find("link2")
        if a.get("link") not in frames or b.get("link") not in frames:
            continue
        pa = (frames[a.get("link")] @ origin(a))[:3, 3]
        pb = (frames[b.get("link")] @ origin(b))[:3, 3]
        loops.append({"joint": loop.get("name"), "closure_error_mm": float(np.linalg.norm(pa - pb) * 1000),
                      "anchor1_end_m": pa.tolist(), "anchor2_end_m": pb.tolist()})
    return {"opening_rad": opening_rad, "motion_enabled": False,
            "end_from_links": {name: matrix.tolist() for name, matrix in frames.items()},
            "closed_loop_checks": loops,
            "note": "FK uses URDF mimic equations. Link origins are not fingertip contact points."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", type=Path, default=Path("robot_description/G2_t2v2_crsB_omnipicker.urdf"))
    parser.add_argument("--opening-rad", type=float, default=-.51025)
    args = parser.parse_args()
    print(json.dumps(audit(args.urdf, args.opening_rad), indent=2))


if __name__ == "__main__":
    main()
