from __future__ import annotations

import unittest

from hole_target_detector import BoxDetection, best_per_class, pair_status


class HoleTargetDetectorTests(unittest.TestCase):
    def test_best_per_class_and_pair(self) -> None:
        detections = [
            BoxDetection(0, "hole", 0.60, (10, 20, 30, 40)),
            BoxDetection(0, "hole", 0.85, (11, 21, 31, 41)),
            BoxDetection(1, "target", 0.70, (50, 60, 70, 80)),
        ]
        selected = best_per_class(detections)
        self.assertEqual(selected["hole"].confidence, 0.85)
        self.assertEqual(pair_status(detections, selected), "multiple_candidates_preview_only")

    def test_missing_target_is_not_a_pair(self) -> None:
        detections = [BoxDetection(0, "hole", 0.9, (10, 20, 30, 40))]
        selected = best_per_class(detections)
        self.assertEqual(pair_status(detections, selected), "missing_target")

    def test_single_pair(self) -> None:
        detections = [
            BoxDetection(0, "hole", 0.9, (10, 20, 30, 40)),
            BoxDetection(1, "target", 0.8, (50, 60, 70, 80)),
        ]
        selected = best_per_class(detections)
        self.assertEqual(pair_status(detections, selected), "paired")
        self.assertEqual(selected["target"].center_px, (60.0, 70.0))


if __name__ == "__main__":
    unittest.main()

