"""GDK RGB-D acquisition helpers without any ROS or RealSense dependency."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


class CameraError(RuntimeError):
    pass


@dataclass(frozen=True)
class GdkRgbdFrame:
    color_rgb: np.ndarray
    depth_u16: np.ndarray
    color_timestamp_ns: int
    depth_timestamp_ns: int


def decode_color_rgb(image, gdk) -> np.ndarray:
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

