from __future__ import annotations

import unittest

import numpy as np

from camera import align_depth_to_color, decode_depth_u16


INTRINSIC = {"fx": 1.0, "fy": 1.0, "ppx": 0.0, "ppy": 0.0, "width": 3, "height": 2}
IDENTITY = {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]}


class CameraAlignmentTests(unittest.TestCase):
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
