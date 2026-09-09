import unittest
import numpy as np

from move_left_above_head_hinge_gdk import (
    base_from_head_at,
    fixed_forward_down_orientation,
    gripper_end_target,
    select_leftmost_hinge,
)
from control import quaternion_angle_deg


class HeadHingeMoveTests(unittest.TestCase):
    def test_selects_leftmost_hinge_and_ignores_other_class(self):
        masks = []
        for x in (8, 2, 0):
            mask = np.zeros((10, 12), dtype=bool)
            mask[3:6, x:x+2] = True
            masks.append(mask)
        self.assertEqual(select_leftmost_hinge(masks, [0.8]*3, [0, 0, 1]), 1)

    def test_target_accounts_for_tcp_and_above_height(self):
        target = gripper_end_target(np.array([0.5, 0.2, 0.7]), (0, 0, 0, 1), np.array([0, 0, 0.145]), 0.2)
        self.assertTrue(np.allclose(target.position_m, [0.5, 0.2, 0.755]))

    def test_fixed_orientation_points_forward_and_down_thirty_degrees(self):
        # Keep the axis-direction assertion independent of the default tool
        # roll; the default left-hand orientation intentionally adds a 90°
        # clockwise roll around the forward axis.
        orientation = fixed_forward_down_orientation(gripper_roll_deg=0.0)
        x, y, z, w = orientation
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        self.assertTrue(np.allclose(rotation[:, 2], [np.sqrt(3) / 2, 0.0, -0.5], atol=1e-6))
        self.assertTrue(np.allclose(rotation[:, 1], [0.5, 0.0, np.sqrt(3) / 2], atol=1e-6))

    def test_default_orientation_adds_one_hundred_eighty_degree_roll(self):
        zero_roll = fixed_forward_down_orientation(gripper_roll_deg=0.0)
        default = fixed_forward_down_orientation()
        self.assertAlmostEqual(quaternion_angle_deg(zero_roll, default), 180.0, places=5)

    def test_fixed_orientation_supports_yaw_and_pitch_adjustment(self):
        orientation = fixed_forward_down_orientation(forward_yaw_deg=90.0, gripper_pitch_deg=0.0)
        x, y, z, w = orientation
        rotation = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        self.assertTrue(np.allclose(rotation[:, 2], [0.0, 1.0, 0.0], atol=1e-6))

    def test_tf_uses_bounded_latest_fallback(self):
        class Vector:
            x = y = z = 0.0
        class Rotation:
            x = y = z = 0.0
            w = 1.0
        class Transform:
            translation = Vector()
            rotation = Rotation()
        class TF:
            def lookup_transform(self, *_args):
                raise RuntimeError("before oldest transform")
            def lookup_transform_latest(self, *_args, **_kwargs):
                return Transform(), 1_001_000_000
        transform, mode, gap_ms = base_from_head_at(TF(), 1_000_000_000)
        self.assertEqual(mode, "near_latest")
        self.assertAlmostEqual(gap_ms, 1.0)
        self.assertTrue(np.allclose(transform["translation_m"], [0, 0, 0]))


if __name__ == "__main__":
    unittest.main()
