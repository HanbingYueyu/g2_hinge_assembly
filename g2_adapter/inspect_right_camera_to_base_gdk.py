#!/usr/bin/env python3
"""Read-only verification of the official right-camera -> base transform chain.

This probe creates only GDK ``Camera`` and ``TF`` objects.  It never creates a
``Robot`` and never sends an arm or gripper command.  The point, when supplied,
is interpreted in the right color/RGB-D optical frame in metres.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

try:
    from .apriltag_workpiece import inverse_transform_dict
    from .bilateral_rgbd_geometry import right_color_point_to_base
    from .camera import intrinsic_to_dict, transform_to_dict
except ImportError:
    from apriltag_workpiece import inverse_transform_dict
    from bilateral_rgbd_geometry import right_color_point_to_base
    from camera import intrinsic_to_dict, transform_to_dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = PROJECT_ROOT / "robot_description" / "G2_t2v2_crsB_omnipicker.urdf"


def parse_point(value: str | None) -> list[float] | None:
    if value is None:
        return None
    try:
        point = [float(item) for item in value.replace(";", ",").split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--point 需要 x,y,z 三个数字（单位米）") from exc
    if len(point) != 3 or not all(math.isfinite(item) for item in point):
        raise argparse.ArgumentTypeError("--point 需要 x,y,z 三个有限数字（单位米）")
    return point


def urdf_chain(path: Path, leaf: str) -> list[str]:
    root = ET.parse(path).getroot()
    parent_by_child = {}
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is not None and child is not None:
            parent_by_child[child.attrib["link"]] = parent.attrib["link"]
    chain = [leaf]
    seen = {leaf}
    while chain[-1] in parent_by_child:
        parent = parent_by_child[chain[-1]]
        if parent in seen:
            raise RuntimeError("URDF parent loop detected")
        chain.append(parent)
        seen.add(parent)
    return list(reversed(chain))


def main() -> int:
    parser = argparse.ArgumentParser(description="只读核验右手 RGB-D 孔坐标到 base_link 的 GDK 链路")
    parser.add_argument("--point", type=parse_point, help="右彩色/RGBD 相机坐标点 x,y,z（米），可选")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    parser.add_argument("--tf-static-fallback-max-gap-ms", type=float, default=25.0)
    args = parser.parse_args()

    try:
        import agibot_gdk
    except ImportError as exc:
        print(f"agibot_gdk 不可用：{exc}")
        return 2
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败")
        return 2
    camera = None
    try:
        color_type = agibot_gdk.CameraType.kHandRightColor
        depth_type = agibot_gdk.CameraType.kHandRightDepth
        camera = agibot_gdk.Camera([color_type, depth_type])
        color = camera.get_latest_image(color_type, 1000.0)
        depth = camera.get_nearest_image(depth_type, int(color.timestamp_ns), 1000.0)
        color_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(color_type), width=int(color.width), height=int(color.height))
        depth_intrinsic = intrinsic_to_dict(camera.get_camera_intrinsic(depth_type), width=int(depth.width), height=int(depth.height))
        tf = agibot_gdk.TF()
        depth_to_color = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kRightHandDepthToRightHandColor))
        right_end_from_color = transform_to_dict(tf.get_tf_from_sensor(agibot_gdk.SensorExtrinsicType.kRightHandRGBDToArmREndLink))
        color_timestamp = int(color.timestamp_ns)
        latest_transform, latest_timestamp = tf.lookup_transform_latest("base_link", "arm_r_end_link", return_timestamp=True)
        latest_raw = transform_to_dict(latest_transform)
        latest_timestamp = int(latest_timestamp) if latest_timestamp is not None else None
        tf_mode = "latest_static_fallback"
        tf_error = None
        latest_gap_ms = None if latest_timestamp is None else (latest_timestamp - color_timestamp) / 1_000_000.0
        if latest_timestamp is not None and color_timestamp <= latest_timestamp and latest_gap_ms <= args.tf_static_fallback_max_gap_ms:
            try:
                raw_dynamic = transform_to_dict(tf.lookup_transform("base_link", "arm_r_end_link", color_timestamp))
                base_from_right_end = inverse_transform_dict(raw_dynamic)
                tf_mode = "exact_image_time"
                tf_timestamp = color_timestamp
            except Exception as exc:
                tf_error = str(exc)
                raw_dynamic = latest_raw
                base_from_right_end = inverse_transform_dict(raw_dynamic)
                tf_timestamp = latest_timestamp
        else:
            raw_dynamic = latest_raw
            base_from_right_end = inverse_transform_dict(raw_dynamic)
            tf_timestamp = latest_timestamp
            if latest_timestamp is not None:
                tf_error = (
                    "image_timestamp_newer_than_latest_tf"
                    if color_timestamp > latest_timestamp
                    else f"image_timestamp_older_than_latest_tf_by_{latest_gap_ms:.1f}ms"
                )
        result = {
            "event": "right_camera_to_base_probe",
            "motion": "read_only",
            "point_frame_assumption": "right_hand_color_optical",
            "chain": "P_base=T_base_arm_r_end(T_arm_r_end_right_rgbd P_right_color)",
            "color": {
                "shape": [int(color.height), int(color.width)],
                "timestamp_ns": color_timestamp,
                "intrinsic": color_intrinsic,
            },
            "depth": {
                "shape": [int(depth.height), int(depth.width)],
                "timestamp_ns": int(depth.timestamp_ns),
                "timestamp_skew_ms": abs(int(depth.timestamp_ns) - color_timestamp) / 1_000_000.0,
                "intrinsic": depth_intrinsic,
            },
            "gdk_extrinsics": {
                "right_depth_to_color": depth_to_color,
                "right_rgbd_to_arm_r_end": right_end_from_color,
            },
            "dynamic_tf": {
                "target_frame": "base_link",
                "source_frame": "arm_r_end_link",
                "query_timestamp_ns": color_timestamp,
                "transform_timestamp_ns": tf_timestamp,
                "age_ms": None if tf_timestamp is None else (color_timestamp - tf_timestamp) / 1_000_000.0,
                "mode": tf_mode,
                "lookup_error": tf_error,
                "lookup_arguments": ["base_link", "arm_r_end_link"],
                "gdk_returned_transform": "T_arm_r_end_link_base_link",
                "gdk_raw_transform": raw_dynamic,
                "converted_transform": "T_base_link_arm_r_end_link",
                "base_from_arm_r_end": base_from_right_end,
            },
            "urdf_chain_to_arm_r_end": urdf_chain(args.urdf.expanduser().resolve(), "arm_r_end_link"),
            "urdf_gripper_camera_chain_not_used": urdf_chain(args.urdf.expanduser().resolve(), "gripper_r_camera_link"),
        }
        if args.point is not None:
            result["point_right_color_m"] = args.point
            result["point_base_m"] = list(right_color_point_to_base(args.point, right_end_from_color, base_from_right_end))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("只读完成：未保存图像，未创建 Robot，未发送运动或夹爪命令。")
        return 0
    except Exception as exc:
        print(f"右相机到 base_link 核验失败：{exc}")
        return 1
    finally:
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        try:
            agibot_gdk.gdk_release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
