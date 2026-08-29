import json
import tempfile
import unittest
from pathlib import Path

from run_left_premove_sequence import load_presets


class LeftPremoveSequenceTests(unittest.TestCase):
    def test_loads_only_p2_then_p3_even_when_legacy_p1_is_present(self):
        document = {
            "schema_version": 1,
            "poses": {
                "left_p1": {"side": "left", "joint_positions_rad": [1.0] * 7},
                "left_p2": {"side": "left", "joint_positions_rad": [2.0] * 7},
                "left_p3": {"side": "left", "joint_positions_rad": [3.0] * 7},
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "poses.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assertEqual(
                load_presets(path),
                [("left_p2", [2.0] * 7), ("left_p3", [3.0] * 7)],
            )


if __name__ == "__main__":
    unittest.main()
