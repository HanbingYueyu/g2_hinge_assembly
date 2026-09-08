import unittest

import numpy as np

try:
    from .yolo_hinge import associate_shaft_to_hinge, infer_yolo_masks, resize_masks_to_full_image, review_matches, reviewed_target_matches_selected, split_yolo_candidates
except ImportError:
    from yolo_hinge import associate_shaft_to_hinge, infer_yolo_masks, resize_masks_to_full_image, review_matches, reviewed_target_matches_selected, split_yolo_candidates


class _FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def detach(self):
        return self

    def float(self):
        return self

    def int(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class _FakeMasks:
    def __init__(self, value):
        self.data = _FakeTensor(value)


class _FakeBoxes:
    def __init__(self, confidence, classes):
        self.conf = _FakeTensor(confidence)
        self.cls = _FakeTensor(classes)


class _FakeResult:
    def __init__(self):
        self.masks = _FakeMasks([[[1, 0], [1, 1]], [[0, 1], [0, 1]]])
        self.boxes = _FakeBoxes([0.9, 0.8], [0, 1])


class _FakeModel:
    def predict(self, **_kwargs):
        return [_FakeResult()]


class _EmptyModel:
    def predict(self, **_kwargs):
        return []


class YoloReviewTests(unittest.TestCase):
    def test_infer_returns_matching_class_ids(self):
        masks, scores, class_ids = infer_yolo_masks(
            _FakeModel(), np.zeros((2, 2, 3), dtype=np.uint8), confidence=0.1, imgsz=640, device="cpu"
        )
        self.assertEqual(len(masks), 2)
        np.testing.assert_allclose(scores, [0.9, 0.8])
        np.testing.assert_array_equal(class_ids, [0, 1])

    def test_infer_empty_frame_returns_empty_class_ids(self):
        masks, scores, class_ids = infer_yolo_masks(
            _EmptyModel(), np.zeros((2, 2, 3), dtype=np.uint8), confidence=0.1, imgsz=640, device="cpu"
        )
        self.assertEqual(masks, [])
        self.assertEqual(scores.size, 0)
        self.assertEqual(class_ids.size, 0)

    def test_associate_only_nearby_shaft_with_selected_hinge(self):
        statistics = [
            {"center_px": [100.0, 100.0], "yolo_class_id": 0},
            {"center_px": [130.0, 110.0], "yolo_class_id": 1},
            {"center_px": [500.0, 500.0], "yolo_class_id": 1},
        ]
        result = associate_shaft_to_hinge(
            statistics, 0, max_center_distance_px=100.0, max_horizontal_distance_px=100.0
        )
        self.assertEqual(result["shaft_index"], 1)
        self.assertEqual(result["reason"], "matched_nearest_shaft")

    def test_association_rejects_neighbouring_shaft_with_smaller_euclidean_distance(self):
        statistics = [
            {"center_px": [620.0, 434.0], "yolo_class_id": 0},
            {"center_px": [630.0, 695.0], "yolo_class_id": 1},
            {"center_px": [394.0, 509.0], "yolo_class_id": 1},
        ]
        result = associate_shaft_to_hinge(
            statistics, 0, shaft_indices=[1, 2], max_center_distance_px=320.0
        )
        self.assertEqual(result["shaft_index"], 1)

    def test_split_keeps_hinge_for_grasp_and_shaft_as_auxiliary(self):
        statistics = [
            {"yolo_class_id": 0, "yolo_confidence": 0.91},
            {"yolo_class_id": 1, "yolo_confidence": 0.88},
            {"yolo_class_id": 2, "yolo_confidence": 0.99},
            {"yolo_class_id": 1, "yolo_confidence": 0.10},
        ]
        hinges, shafts = split_yolo_candidates(
            statistics, [0, 1, 2, 3], confidence_threshold=0.25
        )
        self.assertEqual(hinges, [0])
        self.assertEqual(shafts, [1])

    def test_split_handles_hinge_only_and_shaft_only_frames(self):
        hinges, shafts = split_yolo_candidates(
            [{"yolo_class_id": 0, "yolo_confidence": 0.8}], [0], confidence_threshold=0.25
        )
        self.assertEqual((hinges, shafts), ([0], []))
        hinges, shafts = split_yolo_candidates(
            [{"yolo_class_id": 1, "yolo_confidence": 0.8}], [0], confidence_threshold=0.25
        )
        self.assertEqual((hinges, shafts), ([], [0]))

    def test_association_rejects_ambiguous_shafts(self):
        statistics = [
            {"center_px": [100.0, 100.0], "yolo_class_id": 0},
            {"center_px": [130.0, 100.0], "yolo_class_id": 1},
            {"center_px": [131.0, 100.0], "yolo_class_id": 1},
        ]
        result = associate_shaft_to_hinge(statistics, 0, shaft_indices=[1, 2], max_center_distance_px=100.0)
        self.assertIsNone(result["shaft_index"])
        self.assertEqual(result["reason"], "ambiguous_nearby_shafts")

    def test_matching_target_is_accepted(self):
        accepted, detail = review_matches(
            {"center_px": [500.0, 400.0], "depth_median_m": 0.200},
            {"center_px": [510.0, 405.0], "depth_median_m": 0.215},
            max_pixel_error=20.0,
            max_depth_error_m=0.02,
        )
        self.assertTrue(accepted)
        self.assertEqual(detail["reason"], "matched")

    def test_disagreement_is_rejected(self):
        accepted, detail = review_matches(
            {"center_px": [500.0, 400.0], "depth_median_m": 0.200},
            {"center_px": [650.0, 400.0], "depth_median_m": 0.260},
            max_pixel_error=20.0,
            max_depth_error_m=0.02,
        )
        self.assertFalse(accepted)
        self.assertEqual(detail["reason"], "selected_target_disagreement")

    def test_review_association_reports_distance_without_accepting_other_hinge(self):
        same, distance = reviewed_target_matches_selected(
            {"center_px": [650.0, 480.0]},
            {"yolo_center_px": [500.0, 400.0], "accepted": True, "fresh": True},
            max_distance_px=120.0,
        )
        self.assertFalse(same)
        self.assertAlmostEqual(distance, np.hypot(150.0, 80.0))

    def test_review_association_accepts_current_candidate(self):
        same, distance = reviewed_target_matches_selected(
            {"center_px": [505.0, 404.0]},
            {"yolo_center_px": [500.0, 400.0]},
            max_distance_px=10.0,
        )
        self.assertTrue(same)
        self.assertAlmostEqual(distance, np.hypot(5.0, 4.0))

    def test_resize_mask_to_full_image_uses_nearest_neighbor(self):
        mask = np.array([[True, False], [False, True]])
        expanded = resize_masks_to_full_image([mask], full_shape=(4, 6))
        self.assertEqual(expanded[0].shape, (4, 6))
        self.assertTrue(expanded[0][0, 0])
        self.assertFalse(expanded[0][0, 5])
        self.assertTrue(expanded[0][3, 5])


if __name__ == "__main__":
    unittest.main()
