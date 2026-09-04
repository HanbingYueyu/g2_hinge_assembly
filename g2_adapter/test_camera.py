from __future__ import annotations

import unittest

import numpy as np

from camera import CameraError, align_depth_to_color, decode_depth_u16, depth_on_color_grid, rescale_intrinsic_to_image


INTRINSIC = {"fx": 1.0, "fy": 1.0, "ppx": 0.0, "ppy": 0.0, "width": 3, "height": 2}
IDENTITY = {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}


class CameraAlignmentTests(unittest.TestCase):
    def test_rescale_intrinsic_to_image_scales_both_axes(self) -> None:
        intrinsic = {
            "fx": 395.0,
            "fy": 396.0,
            "ppx": 317.0,
            "ppy": 243.0,
            "width": 640,
            "height": 480,
            "distortion": [0.1, -0.2, 0.0, 0.0],
        }
        actual = rescale_intrinsic_to_image(
            intrinsic,
            calibration_width=640,
            calibration_height=480,
            image_width=1280,
            image_height=1056,
        )
        self.assertEqual(actual["width"], 1280)
        self.assertEqual(actual["height"], 1056)
        self.assertEqual(actual["fx"], 790.0)
        self.assertEqual(actual["fy"], 871.2)
        self.assertEqual(actual["ppx"], 634.0)
        self.assertEqual(actual["ppy"], 534.6)
        self.assertEqual(actual["distortion"], intrinsic["distortion"])
        self.assertEqual(intrinsic["fx"], 395.0)

    def test_reproject_can_use_depth_calibration_size(self) -> None:
        depth = np.array([[1000]], dtype=np.uint16)
        depth_intrinsic = {
            "fx": 1.0,
            "fy": 1.0,
            "ppx": 0.0,
            "ppy": 0.0,
            "width": 1,
            "height": 1,
        }
        color_intrinsic = {
            "fx": 1.0,
            "fy": 1.0,
            "ppx": 0.0,
            "ppy": 0.0,
            "width": 2,
            "height": 2,
        }
        actual = depth_on_color_grid(
            depth,
            registration_mode="reproject",
            depth_intrinsic=depth_intrinsic,
            color_intrinsic=color_intrinsic,
            depth_to_color=IDENTITY,
            depth_calibration_size=(1, 1),
        )
        self.assertEqual(int(actual[0, 0]), 1000)
        self.assertEqual(int(actual[0, 1]), 0)
        self.assertEqual(int(actual[1, 0]), 0)

    def test_raw_z16_depth_does_not_need_opencv(self) -> None:
        class Encoding:
            PNG = "png"
            UNCOMPRESSED = "raw"

        class Image:
            encoding = Encoding.UNCOMPRESSED
            width = 2
            height = 1
            data = np.array([0xE8, 0x03, 0xD0, 0x07], dtype=np.uint8)

        gdk = type("Gdk", (), {"Encoding": Encoding})
        actual = decode_depth_u16(Image(), gdk)
        np.testing.assert_array_equal(actual, np.array([[1000, 2000]], dtype=np.uint16))

    def test_identity_alignment_keeps_depth_pixels(self) -> None:
        depth = np.array([[0, 1000, 0], [2000, 0, 3000]], dtype=np.uint16)
        aligned = align_depth_to_color(
            depth,
            depth_intrinsic=INTRINSIC,
            color_intrinsic=INTRINSIC,
            depth_to_color=IDENTITY,
        )
        np.testing.assert_array_equal(aligned, depth)

    def test_z_buffer_keeps_nearest_projected_depth(self) -> None:
        depth = np.array([[1000, 2000]], dtype=np.uint16)
        depth_intrinsic = {"fx": 2.0, "fy": 1.0, "ppx": 0.0, "ppy": 0.0, "width": 2, "height": 1}
        color_intrinsic = {"fx": 1.0, "fy": 1.0, "ppx": 0.0, "ppy": 0.0, "width": 2, "height": 1}
        aligned = align_depth_to_color(
            depth,
            depth_intrinsic=depth_intrinsic,
            color_intrinsic=color_intrinsic,
            depth_to_color=IDENTITY,
        )
        self.assertEqual(int(aligned[0, 0]), 1000)

    def test_registered_same_pixel_depth_keeps_color_grid(self) -> None:
        depth = np.array([[0, 1000, 0], [2000, 0, 3000]], dtype=np.uint16)
        actual = depth_on_color_grid(
            depth,
            registration_mode="registered_same_pixel",
            depth_intrinsic=INTRINSIC,
            color_intrinsic=INTRINSIC,
            depth_to_color=None,
        )
        np.testing.assert_array_equal(actual, depth)
        self.assertIsNot(actual, depth)

    def test_registered_same_pixel_rejects_shape_mismatch(self) -> None:
        depth = np.zeros((1, 3), dtype=np.uint16)
        with self.assertRaises(CameraError):
            depth_on_color_grid(
                depth,
                registration_mode="registered_same_pixel",
                depth_intrinsic=INTRINSIC,
                color_intrinsic=INTRINSIC,
                depth_to_color=None,
            )

    def test_reproject_requires_transform(self) -> None:
        depth = np.zeros((2, 3), dtype=np.uint16)
        with self.assertRaises(CameraError):
            depth_on_color_grid(
                depth,
                registration_mode="reproject",
                depth_intrinsic=INTRINSIC,
                color_intrinsic=INTRINSIC,
                depth_to_color=None,
            )
