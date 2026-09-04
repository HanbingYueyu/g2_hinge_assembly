import unittest

import numpy as np

from bilateral_depth_accumulator import FreshHoleDepthAccumulator


class FreshHoleDepthAccumulatorTests(unittest.TestCase):
    def test_empty_hole_ring_is_skipped_without_advancing_window(self):
        accumulator = FreshHoleDepthAccumulator(window_frames=3, min_temporal_hits=2)
        box = (40.0, 40.0, 60.0, 60.0)
        empty = np.zeros((80, 80), dtype=np.uint16)
        empty[0:5, 0:5] = 500
        result = accumulator.add(empty, 1_000_000_000, box)
        self.assertIsNone(result.depth_u16)
        self.assertEqual(result.diagnostics["state"], "skipped")
        self.assertEqual(result.diagnostics["reason"], "skipped_no_hole_depth")
        self.assertEqual(result.diagnostics["window_frames"], 0)
        self.assertEqual(result.diagnostics["frame_valid_pixels"], 0)

    def test_insufficient_hole_ring_is_skipped(self):
        accumulator = FreshHoleDepthAccumulator(
            window_frames=3,
            admission_min_valid_pixels=2,
            admission_min_support_fraction=0.10,
        )
        depth = np.zeros((80, 80), dtype=np.uint16)
        depth[40, 50] = 500
        result = accumulator.add(depth, 1_000_000_000, (40.0, 40.0, 60.0, 60.0))
        self.assertEqual(result.diagnostics["reason"], "skipped_no_hole_depth")
        self.assertEqual(result.diagnostics["frame_raw_valid_pixels"], 1)
        self.assertEqual(result.diagnostics["window_frames"], 0)

    def test_box_admission_uses_box_pixels_not_ring(self):
        accumulator = FreshHoleDepthAccumulator(
            window_frames=2,
            min_temporal_hits=1,
            admission_min_valid_pixels=1,
            admission_min_support_fraction=0.0,
            sampling_region="box",
        )
        depth = np.zeros((80, 80), dtype=np.uint16)
        # The valid sample is in the box center, which is deliberately
        # excluded by the hole ring mask.
        depth[50, 50] = 500
        box = (40.0, 40.0, 60.0, 60.0)
        first = accumulator.add(depth, 1_000_000_000, box)
        self.assertIsNone(first.depth_u16)
        self.assertEqual(first.diagnostics["frame_depth_region"], "box")
        self.assertEqual(first.diagnostics["frame_box_pixels"], 441)
        self.assertEqual(first.diagnostics["frame_valid_pixels"], 1)
        self.assertEqual(first.diagnostics["window_frames"], 1)
        second = accumulator.add(depth, 1_100_000_000, box)
        self.assertIsNotNone(second.depth_u16)

    def test_stale_window_resets_even_when_current_frame_is_skipped(self):
        accumulator = FreshHoleDepthAccumulator(window_frames=3, max_window_ms=500.0)
        box = (40.0, 40.0, 60.0, 60.0)
        valid = np.zeros((80, 80), dtype=np.uint16)
        valid[40:45, 40:45] = 500
        empty = np.zeros((80, 80), dtype=np.uint16)
        self.assertEqual(accumulator.add(valid, 1_000_000_000, box).diagnostics["window_frames"], 1)
        result = accumulator.add(empty, 1_600_000_000, box)
        self.assertEqual(result.diagnostics["reason"], "skipped_no_hole_depth")
        self.assertEqual(result.diagnostics["reset_reason"], "hole_depth_window_stale")
        self.assertEqual(result.diagnostics["window_frames"], 0)

    def test_returns_temporal_median_after_minimum_window(self):
        accumulator = FreshHoleDepthAccumulator(window_frames=3, min_temporal_hits=2, max_window_ms=500.0)
        box = (40.0, 40.0, 60.0, 60.0)
        depth_a = np.zeros((80, 80), dtype=np.uint16)
        depth_a[40:45, 40:45] = 500
        depth_b = depth_a.copy()
        depth_b[40:45, 40:45] = 520
        depth_c = depth_a.copy()
        depth_c[40:45, 40:45] = 510
        self.assertIsNone(accumulator.add(depth_a, 1_000_000_000, box).depth_u16)
        self.assertIsNone(accumulator.add(depth_b, 1_100_000_000, box).depth_u16)
        result = accumulator.add(depth_c, 1_200_000_000, box)
        self.assertIsNotNone(result.depth_u16)
        self.assertEqual(int(result.depth_u16[42, 42]), 510)
        self.assertEqual(result.diagnostics["state"], "ready")

    def test_unstable_box_starts_a_new_window(self):
        accumulator = FreshHoleDepthAccumulator(window_frames=3, min_temporal_hits=2, max_center_jitter_px=2.0)
        depth = np.full((20, 20), 500, dtype=np.uint16)
        self.assertIsNone(accumulator.add(depth, 1_000_000_000, (5.0, 5.0, 10.0, 10.0)).depth_u16)
        result = accumulator.add(depth, 1_100_000_000, (10.0, 10.0, 15.0, 15.0))
        self.assertIsNone(result.depth_u16)
        self.assertEqual(result.diagnostics["window_frames"], 1)
        self.assertEqual(result.diagnostics["reset_reason"], "hole_box_unstable_reset")


if __name__ == "__main__":
    unittest.main()
