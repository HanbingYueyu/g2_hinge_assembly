import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import record_camera_video as recorder


class CameraSelectionTests(unittest.TestCase):
    def test_accepts_one_or_several_of_the_six_streams(self):
        self.assertEqual(recorder.parse_streams("right_color"), ("right_color",))
        self.assertEqual(
            recorder.parse_streams("head_color,left_depth,right_depth"),
            ("head_color", "left_depth", "right_depth"),
        )

    def test_all_selects_every_stream_once(self):
        self.assertEqual(
            recorder.parse_streams("all"),
            (
                "head_color", "head_depth", "left_color",
                "left_depth", "right_color", "right_depth",
            ),
        )

    def test_rejects_unknown_and_duplicate_streams(self):
        with self.assertRaisesRegex(ValueError, "未知相机流"):
            recorder.parse_streams("head_color,other")
        with self.assertRaisesRegex(ValueError, "重复"):
            recorder.parse_streams("left_color,left_color")


class FrameRateTests(unittest.TestCase):
    def test_measures_actual_fps_from_positive_gdk_timestamp_deltas(self):
        timestamps = [1_000_000_000, 1_033_000_000, 1_066_000_000, 1_099_000_000]
        self.assertAlmostEqual(recorder.measured_fps(timestamps), 30.303, places=3)

    def test_requires_two_distinct_timestamps(self):
        with self.assertRaisesRegex(ValueError, "至少两个"):
            recorder.measured_fps([1_000_000_000, 1_000_000_000])

    def test_rejects_regressing_timestamps(self):
        with self.assertRaisesRegex(ValueError, "倒退"):
            recorder.measured_fps([1_000_000_000, 1_033_000_000, 1_020_000_000])

    def test_rejects_an_implausibly_high_measured_rate(self):
        with self.assertRaisesRegex(ValueError, "120"):
            recorder.measured_fps([1_000_000_000, 1_001_000_000])


class WarmupTests(unittest.TestCase):
    def test_times_out_and_names_a_stalled_stream(self):
        class Image:
            timestamp_ns = 123

        class Camera:
            def get_latest_image(self, camera_type, timeout_ms):
                return Image()

        clock_values = iter([0.0, 0.1, 0.2, 1.1])
        output = recorder.StreamOutput("right_color", object(), 30.0, Path("unused"))
        with self.assertRaisesRegex(TimeoutError, "right_color.*1/2"):
            recorder._collect_timestamps(
                Camera(), {"right_color": output}, 2, timeout_s=1.0,
                clock=lambda: next(clock_values), sleep=lambda _: None,
            )


class StreamOutputTests(unittest.TestCase):
    def test_actual_rate_mode_fills_timestamp_gaps_to_preserve_duration(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = recorder.StreamOutput(
                "right_color", object(), 10.0, Path(temporary), preserve_timeline=True,
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            self.assertTrue(output.write(frame, 1_000_000_000))
            self.assertTrue(output.write(frame, 1_300_000_000))
            output.close()

            self.assertEqual(output.captured_frames, 2)
            self.assertEqual(output.frames, 4)
            records = [json.loads(line) for line in (Path(temporary) / "right_color_timestamps.jsonl").read_text().splitlines()]
            self.assertEqual([item["duplicated"] for item in records], [False, True, True, False])

    def test_actual_rate_mode_does_not_write_twice_into_one_time_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = recorder.StreamOutput(
                "right_color", object(), 10.0, Path(temporary), preserve_timeline=True,
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            output.write(frame, 1_000_000_000)
            output.write(frame, 1_040_000_000)
            output.write(frame, 1_100_000_000)
            output.close()

            self.assertEqual(output.captured_frames, 3)
            self.assertEqual(output.frames, 2)
            records = [json.loads(line) for line in (Path(temporary) / "right_color_timestamps.jsonl").read_text().splitlines()]
            self.assertEqual(sum(item.get("dropped", False) for item in records), 1)

    def test_actual_rate_mode_rejects_an_excessive_forward_jump(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = recorder.StreamOutput(
                "right_color", object(), 10.0, Path(temporary), preserve_timeline=True,
                max_gap_seconds=1.0,
            )
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            output.write(frame, 1_000_000_000)
            with self.assertRaisesRegex(RuntimeError, "跳变"):
                output.write(frame, 3_000_000_000)
            output.close()

    def test_rejects_a_regressing_timestamp(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = recorder.StreamOutput("right_color", object(), 30.0, Path(temporary))
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            output.write(frame, 2)
            with self.assertRaisesRegex(RuntimeError, "倒退"):
                output.write(frame, 1)
            output.close()


class DepthEncodingTests(unittest.TestCase):
    def test_depth_preview_is_three_channel_and_invalid_pixels_are_black(self):
        depth = np.asarray([[0, 200], [1100, 65535]], dtype=np.uint16)
        preview = recorder.depth_preview_bgr(depth, near_mm=200, far_mm=2000)
        self.assertEqual(preview.shape, (2, 2, 3))
        self.assertEqual(preview.dtype, np.uint8)
        np.testing.assert_array_equal(preview[0, 0], [0, 0, 0])
        np.testing.assert_array_equal(preview[1, 1], [0, 0, 0])


if __name__ == "__main__":
    unittest.main()
