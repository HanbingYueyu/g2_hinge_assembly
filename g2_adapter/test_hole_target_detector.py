from __future__ import annotations

import unittest

import numpy as np

from hole_target_detector import (
    BoxDetection,
    best_per_class,
    center_crop,
    pair_status,
    select_locked_hole,
    translate_detection,
)


class HoleTargetDetectorTests(unittest.TestCase):
    def test_center_crop_returns_center_and_offset(self) -> None:
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        cropped, offset = center_crop(frame, 60.0)
        self.assertEqual(cropped.shape[:2], (60, 120))
        self.assertEqual(offset, (40, 20))

    def test_translate_detection_maps_crop_pixels_to_source(self) -> None:
        item = BoxDetection(0, "hole", 0.8, (1.0, 2.0, 11.0, 12.0))
        mapped = translate_detection(item, (40, 20))
        self.assertEqual(mapped.xyxy, (41.0, 22.0, 51.0, 32.0))

    def test_locked_hole_does_not_switch_to_far_candidate(self) -> None:
        near = BoxDetection(0, "hole", 0.4, (95.0, 95.0, 105.0, 105.0))
        far = BoxDetection(0, "hole", 0.99, (295.0, 295.0, 305.0, 305.0))
        selected, reason = select_locked_hole([near, far], (100.0, 100.0), max_jump_px=40.0)
        self.assertIs(selected, near)
        self.assertEqual(reason, "temporal_lock")
        selected, reason = select_locked_hole([far], (100.0, 100.0), max_jump_px=40.0)
        self.assertIsNone(selected)
        self.assertEqual(reason, "locked_hole_jump_exceeds_limit")

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
