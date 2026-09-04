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


def rescale_intrinsic_to_image(
    intrinsic: dict,
    *,
    calibration_width: int,
    calibration_height: int,
    image_width: int,
    image_height: int,
) -> dict[str, float | int | list[float]]:
    """Express a calibration matrix in the pixel grid of an image stream.

    This is only valid when the caller has independently verified that the
    reported calibration belongs to a different native resolution.  It is
    deliberately opt-in: equal color/depth array shapes do not prove that a
    scale is required.  Scaling the focal lengths and principal point is the
    only operation; Brown-Conrady distortion coefficients are dimensionless
    and are kept unchanged.  The returned mapping is a copy and never mutates
    the GDK metadata held by the caller.
    """
    try:
        calibration_width = int(calibration_width)
        calibration_height = int(calibration_height)
        image_width = int(image_width)
        image_height = int(image_height)
    except (TypeError, ValueError) as exc:
        raise CameraError("intrinsic calibration/image sizes must be integers") from exc
    if min(calibration_width, calibration_height, image_width, image_height) <= 0:
        raise CameraError("intrinsic calibration/image sizes must be positive")
    scale_x = float(image_width) / float(calibration_width)
    scale_y = float(image_height) / float(calibration_height)
    result = dict(intrinsic)
    for key in ("fx", "fy", "ppx", "ppy"):
        try:
            value = float(intrinsic[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise CameraError(f"intrinsic is missing finite {key}") from exc
        if not np.isfinite(value):
            raise CameraError(f"intrinsic {key} must be finite")
    result["fx"] = float(intrinsic["fx"]) * scale_x
    result["fy"] = float(intrinsic["fy"]) * scale_y
    result["ppx"] = float(intrinsic["ppx"]) * scale_x
    result["ppy"] = float(intrinsic["ppy"]) * scale_y
    result["width"] = image_width
    result["height"] = image_height
    return result


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


def _distortion_coefficients(intrinsic: dict) -> tuple[float, ...]:
    values = intrinsic.get("distortion", ())
    try:
        coefficients = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise CameraError("camera distortion must contain finite values") from exc
    if coefficients and not np.all(np.isfinite(coefficients)):
        raise CameraError("camera distortion must contain finite values")
    return coefficients


def _undistort_normalized_pixels(
    distorted_x: np.ndarray, distorted_y: np.ndarray, coefficients: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Invert OpenCV/RealSense Brown-Conrady distortion on an image grid."""
    if len(coefficients) < 4 or not any(abs(value) > 1e-15 for value in coefficients):
        return distorted_x, distorted_y
    k1, k2, p1, p2 = coefficients[:4]
    k3 = coefficients[4] if len(coefficients) >= 5 else 0.0
    k4 = coefficients[5] if len(coefficients) >= 6 else 0.0
    k5 = coefficients[6] if len(coefficients) >= 7 else 0.0
    k6 = coefficients[7] if len(coefficients) >= 8 else 0.0
    x = np.asarray(distorted_x, dtype=np.float64).copy()
    y = np.asarray(distorted_y, dtype=np.float64).copy()
    for _ in range(8):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        denominator = np.where(np.abs(denominator) <= 1e-12, 1.0, denominator)
        radial = numerator / denominator
        radial = np.where(np.abs(radial) <= 1e-12, 1.0, radial)
        tangential_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        tangential_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (distorted_x - tangential_x) / radial
        y = (distorted_y - tangential_y) / radial
    return x, y


def _distort_normalized_pixels(
    x: np.ndarray, y: np.ndarray, coefficients: tuple[float, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply OpenCV/RealSense Brown-Conrady distortion to normalized pixels."""
    if len(coefficients) < 4 or not any(abs(value) > 1e-15 for value in coefficients):
        return x, y
    k1, k2, p1, p2 = coefficients[:4]
    k3 = coefficients[4] if len(coefficients) >= 5 else 0.0
    k4 = coefficients[5] if len(coefficients) >= 6 else 0.0
    k5 = coefficients[6] if len(coefficients) >= 7 else 0.0
    k6 = coefficients[7] if len(coefficients) >= 8 else 0.0
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    numerator = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    denominator = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
    denominator = np.where(np.abs(denominator) <= 1e-12, 1.0, denominator)
    radial = numerator / denominator
    return (
        x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
        y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y,
    )


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
    depth_distortion = _distortion_coefficients(depth_intrinsic)
    depth_x_distorted = (cols - float(depth_intrinsic["ppx"])) / float(depth_intrinsic["fx"])
    depth_y_distorted = (rows - float(depth_intrinsic["ppy"])) / float(depth_intrinsic["fy"])
    depth_x, depth_y = _undistort_normalized_pixels(
        depth_x_distorted, depth_y_distorted, depth_distortion
    )
    points_depth = np.stack((
        depth_x * z_depth_m,
        depth_y * z_depth_m,
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
    color_distortion = _distortion_coefficients(color_intrinsic)
    color_x = points_color[:, 0] / points_color[:, 2]
    color_y = points_color[:, 1] / points_color[:, 2]
    color_x_distorted, color_y_distorted = _distort_normalized_pixels(
        color_x, color_y, color_distortion
    )
    finite_projection = np.isfinite(color_x_distorted) & np.isfinite(color_y_distorted)
    if not np.any(finite_projection):
        return aligned
    color_x_distorted = color_x_distorted[finite_projection]
    color_y_distorted = color_y_distorted[finite_projection]
    points_color = points_color[finite_projection]
    projected_cols_float = np.rint(
        color_x_distorted * float(color_intrinsic["fx"]) + float(color_intrinsic["ppx"])
    )
    projected_rows_float = np.rint(
        color_y_distorted * float(color_intrinsic["fy"]) + float(color_intrinsic["ppy"])
    )
    safe_integer = (
        np.isfinite(projected_cols_float)
        & np.isfinite(projected_rows_float)
        & (np.abs(projected_cols_float) < np.iinfo(np.int64).max / 2.0)
        & (np.abs(projected_rows_float) < np.iinfo(np.int64).max / 2.0)
    )
    if not np.any(safe_integer):
        return aligned
    projected_cols = projected_cols_float[safe_integer].astype(np.int64)
    projected_rows = projected_rows_float[safe_integer].astype(np.int64)
    points_color = points_color[safe_integer]
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


def depth_on_color_grid(
    depth_u16: np.ndarray,
    *,
    registration_mode: str,
    depth_intrinsic: dict,
    color_intrinsic: dict,
    depth_to_color: dict | None,
    depth_scale_m: float = 0.001,
    depth_calibration_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return a depth image whose pixels can be indexed by color detections.

    ``registered_same_pixel`` is intentionally explicit and is only for a
    stream that has been independently verified to be registered.  Equal array
    shapes are checked, but are not treated as proof of registration.  When
    ``depth_calibration_size`` is supplied for ``reproject``, the raw depth
    calibration is first expressed in the raw depth array's pixel domain.
    """
    if registration_mode == "registered_same_pixel":
        if depth_u16.dtype != np.uint16 or depth_u16.ndim != 2:
            raise CameraError("registered depth must be a two-dimensional uint16 image")
        expected_shape = (
            int(color_intrinsic["height"]),
            int(color_intrinsic["width"]),
        )
        if depth_u16.shape != expected_shape:
            raise CameraError(
                "registered depth shape must match the color grid: "
                f"depth={depth_u16.shape}, color={expected_shape}"
            )
        # Keep the ownership boundary explicit: downstream temporal windows may
        # retain this array after the camera buffer is reused.
        return np.ascontiguousarray(depth_u16.copy())
    if registration_mode == "reproject":
        if depth_to_color is None:
            raise CameraError("reproject mode requires a depth_to_color transform")
        if depth_calibration_size is not None:
            try:
                calibration_width, calibration_height = depth_calibration_size
            except (TypeError, ValueError) as exc:
                raise CameraError(
                    "depth_calibration_size must be (width, height)"
                ) from exc
            depth_intrinsic = rescale_intrinsic_to_image(
                depth_intrinsic,
                calibration_width=calibration_width,
                calibration_height=calibration_height,
                image_width=int(depth_u16.shape[1]),
                image_height=int(depth_u16.shape[0]),
            )
        return align_depth_to_color(
            depth_u16,
            depth_intrinsic=depth_intrinsic,
            color_intrinsic=color_intrinsic,
            depth_to_color=depth_to_color,
            depth_scale_m=depth_scale_m,
        )
    raise CameraError(
        "registration_mode must be 'registered_same_pixel' or 'reproject'"
    )


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
