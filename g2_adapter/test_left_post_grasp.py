import json
import tempfile
import unittest
from pathlib import Path

from run_left_post_grasp_sequence import PRE_P4_LIFT_M, load_presets


class LeftPostGraspSequenceTests(unittest.TestCase):
    def test_pre_p4_lift_is_100_mm(self):
        self.assertAlmostEqual(PRE_P4_LIFT_M, 0.10)

    def test_loads_p3_then_p4(self):
        document = {
            "schema_version": 1,
            "poses": {
                "left_p3": {"side": "left", "joint_positions_rad": [3.0] * 7},
                "left_p4": {"side": "left", "joint_positions_rad": [4.0] * 7},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                load_presets(path),
                [("left_p3", [3.0] * 7), ("left_p4", [4.0] * 7)],
            )


if __name__ == "__main__":
    unittest.main()
