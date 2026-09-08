import unittest
from pathlib import Path

import numpy as np

from inspect_left_gripper_urdf import audit


class GripperUrdfTests(unittest.TestCase):
    path = Path(__file__).resolve().parents[1] / "robot_description/G2_t2v2_crsB_omnipicker.urdf"

    def test_fixed_base_transform(self):
        result = audit(self.path, 0)
        matrix = np.array(result["end_from_links"]["gripper_l_base_link"])
        np.testing.assert_allclose(matrix[:3, 3], [0, 0, .0051])
        np.testing.assert_allclose(matrix[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)

    def test_open_loop_mismatch_is_reported_not_hidden(self):
        closed = audit(self.path, 0)["closed_loop_checks"]
        opened = audit(self.path, -.51025)["closed_loop_checks"]
        self.assertEqual(len(opened), 2)
        self.assertTrue(all(item["closure_error_mm"] < 1 for item in closed))
        self.assertTrue(all(item["closure_error_mm"] > 20 for item in opened))
        for item in opened:
            distance = np.linalg.norm(np.array(item["anchor1_end_m"]) - item["anchor2_end_m"]) * 1000
            self.assertAlmostEqual(item["closure_error_mm"], distance)

    def test_invalid_openings_rejected(self):
        for value in [float("nan"), .1, -1]:
            with self.assertRaises(ValueError):
                audit(self.path, value)


if __name__ == "__main__":
    unittest.main()
