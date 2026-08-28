"""GDK RGB-D acquisition helpers without any ROS or RealSense dependency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class GdkRgbdFrame:
    color_rgb: np.ndarray
    depth_u16: np.ndarray
    color_timestamp_ns: int
    depth_timestamp_ns: int


def intrinsic_to_dict(intrinsic, *, width: int, height: int) -> dict[str, float | int | list[float]]:
    """Convert GDK's ordered ``fx, fy, cx, cy`` intrinsic object to JSON data."""
    values = [float(value) for value in intrinsic.intrinsic]
    if len(values) != 4:
        raise CameraError(f"expected 4 camera intrinsic values, got {len(values)}")
    return {
        "fx": values[0],
        "fy": values[1],
        "ppx": values[2],
        "ppy": values[3],
        "width": int(width),
        "height": int(height),
        "distortion": [float(value) for value in intrinsic.distortion],
    }


def transform_to_dict(transform) -> dict[str, list[float]]:
    """Return a GDK Transform as JSON-friendly translation and quaternion data."""
    return {
        "translation_m": [
            float(transform.translation.x),
            float(transform.translation.y),
            float(transform.translation.z),
        ],
        "rotation_xyzw": [
            float(transform.rotation.x),
            float(transform.rotation.y),
            float(transform.rotation.z),
            float(transform.rotation.w),
        ],
    }


def _rotation_matrix_from_quaternion_xyzw(quaternion: list[float] | np.ndarray) -> np.ndarray:
    x, y, z, w = (float(value) for value in quaternion)
    norm = x * x + y * y + z * z + w * w
    if norm <= np.finfo(np.float64).eps:
        raise CameraError("camera extrinsic quaternion has zero length")
    x, y, z, w = x / norm**0.5, y / norm**0.5, z / norm**0.5, w / norm**0.5
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def align_depth_to_color(
    depth_u16: np.ndarray,
    *,
    depth_intrinsic: dict,
    color_intrinsic: dict,
    depth_to_color: dict,
    depth_scale_m: float = 0.001,
) -> np.ndarray:
    """Project raw depth-camera Z16 pixels into the color-camera image plane.

    GDK returns the left hand color and depth streams as independent optical
    sensors.  Their reported shapes may coincide, but their intrinsics do not;
    a same-pixel lookup would therefore be uncalibrated.  ``depth_to_color``
    is the GDK transform returned for
    ``kLeftHandDepthToLeftHandColor`` (or its right-hand equivalent).
    """
    if depth_u16.dtype != np.uint16 or depth_u16.ndim != 2:
        raise CameraError("depth_u16 must be a two-dimensional uint16 image")
    if not np.isfinite(depth_scale_m) or depth_scale_m <= 0:
        raise CameraError("depth_scale_m must be positive and finite")
    output_height = int(color_intrinsic["height"])
    output_width = int(color_intrinsic["width"])
    if output_height <= 0 or output_width <= 0:
        raise CameraError("color intrinsic image shape is invalid")
    for name, intrinsic in (("depth", depth_intrinsic), ("color", color_intrinsic)):
        if float(intrinsic["fx"]) <= 0 or float(intrinsic["fy"]) <= 0:
            raise CameraError(f"{name} focal length must be positive")

    rows, cols = np.nonzero(depth_u16)
    aligned = np.zeros((output_height, output_width), dtype=np.uint16)
    if rows.size == 0:
        return aligned
    z_depth_m = depth_u16[rows, cols].astype(np.float64) * float(depth_scale_m)
    points_depth = np.stack((
        (cols - float(depth_intrinsic["ppx"])) * z_depth_m / float(depth_intrinsic["fx"]),
        (rows - float(depth_intrinsic["ppy"])) * z_depth_m / float(depth_intrinsic["fy"]),
        z_depth_m,
    ), axis=1)
    rotation = _rotation_matrix_from_quaternion_xyzw(depth_to_color["rotation_xyzw"])
    translation = np.asarray(depth_to_color["translation_m"], dtype=np.float64)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise CameraError("camera extrinsic translation is invalid")
    points_color = points_depth @ rotation.T + translation
    positive = points_color[:, 2] > np.finfo(np.float64).eps
    points_color = points_color[positive]
    if points_color.size == 0:
        return aligned
    projected_cols = np.rint(
        points_color[:, 0] * float(color_intrinsic["fx"]) / points_color[:, 2]
        + float(color_intrinsic["ppx"])
    ).astype(np.int64)
    projected_rows = np.rint(
        points_color[:, 1] * float(color_intrinsic["fy"]) / points_color[:, 2]
        + float(color_intrinsic["ppy"])
    ).astype(np.int64)
    in_bounds = (
        (projected_cols >= 0) & (projected_cols < output_width)
        & (projected_rows >= 0) & (projected_rows < output_height)
    )
    if not np.any(in_bounds):
        return aligned
    projected_depth_units = np.rint(
        points_color[in_bounds, 2] / float(depth_scale_m)
    ).clip(1, np.iinfo(np.uint16).max).astype(np.uint16)
    flat_indices = (
        projected_rows[in_bounds] * output_width + projected_cols[in_bounds]
    )
    z_buffer = np.full(output_height * output_width, np.iinfo(np.uint16).max, dtype=np.uint16)
    np.minimum.at(z_buffer, flat_indices, projected_depth_units)
    z_buffer[z_buffer == np.iinfo(np.uint16).max] = 0
    return z_buffer.reshape(output_height, output_width)


def decode_color_rgb(image, gdk) -> np.ndarray:
    import cv2

    data = np.asarray(image.data, dtype=np.uint8)
    if data.size == 0:
        raise CameraError("GDK returned an empty color image")
    if image.encoding in (gdk.Encoding.JPEG, gdk.Encoding.PNG):
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            raise CameraError("failed to decode compressed GDK color image")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    elif image.encoding == gdk.Encoding.UNCOMPRESSED:
        shape = (int(image.height), int(image.width), 3)
        if image.color_format == gdk.ColorFormat.RGB:
            rgb = data.reshape(shape)
        elif image.color_format == gdk.ColorFormat.BGR:
            rgb = cv2.cvtColor(data.reshape(shape), cv2.COLOR_BGR2RGB)
        elif image.color_format == gdk.ColorFormat.GRAY8:
            rgb = cv2.cvtColor(
                data.reshape(int(image.height), int(image.width)),
                cv2.COLOR_GRAY2RGB,
            )
        else:
            raise CameraError(f"unsupported GDK color format: {image.color_format}")
    else:
        raise CameraError(f"unsupported GDK color encoding: {image.encoding}")
    return np.ascontiguousarray(rgb, dtype=np.uint8)


def decode_depth_u16(image, gdk) -> np.ndarray:
    raw = np.asarray(image.data, dtype=np.uint8)
    if raw.size == 0:
        raise CameraError("GDK returned an empty depth image")
    if image.encoding == gdk.Encoding.PNG:
        import cv2

        depth = cv2.imdecode(raw, cv2.IMREAD_UNCHANGED)
    elif image.encoding == gdk.Encoding.UNCOMPRESSED:
        expected = int(image.height) * int(image.width) * 2
        if raw.nbytes != expected:
            raise CameraError(
                f"raw depth has {raw.nbytes} bytes, expected {expected}"
            )
        depth = raw.view("<u2").reshape(int(image.height), int(image.width))
    else:
        raise CameraError(f"unsupported GDK depth encoding: {image.encoding}")
    if depth is None or depth.dtype != np.uint16:
        raise CameraError("GDK depth is not uint16")
    return np.ascontiguousarray(depth)


class GdkCameraSource:
    def __init__(self, gdk, camera, *, timeout_ms: float = 1000.0) -> None:
        self.gdk = gdk
        self.camera = camera
        self.timeout_ms = float(timeout_ms)

    def stream_types(self, position: str):
        mapping = {
            "head": (
                self.gdk.CameraType.kHeadColor,
                self.gdk.CameraType.kHeadDepth,
            ),
            "left": (
                self.gdk.CameraType.kHandLeftColor,
                self.gdk.CameraType.kHandLeftDepth,
            ),
            "right": (
                self.gdk.CameraType.kHandRightColor,
                self.gdk.CameraType.kHandRightDepth,
            ),
        }
        try:
            return mapping[position]
        except KeyError as exc:
            raise ValueError("position must be head, left, or right") from exc

    def capture_rgbd(self, position: str) -> GdkRgbdFrame:
        color_type, depth_type = self.stream_types(position)
        color = self.camera.get_latest_image(color_type, self.timeout_ms)
        color_timestamp = int(color.timestamp_ns)
        if color_timestamp <= 0:
            raise CameraError("invalid GDK color timestamp")
        depth = self.camera.get_nearest_image(
            depth_type, color_timestamp, self.timeout_ms
        )
        depth_timestamp = int(depth.timestamp_ns)
        if depth_timestamp <= 0:
            raise CameraError("invalid GDK depth timestamp")
        return GdkRgbdFrame(
            color_rgb=decode_color_rgb(color, self.gdk),
            depth_u16=decode_depth_u16(depth, self.gdk),
            color_timestamp_ns=color_timestamp,
            depth_timestamp_ns=depth_timestamp,
        )
