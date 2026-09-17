import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from optimize_teach_trajectory import optimize, main
from teach_trajectory import build_trajectory, trajectory_document, iter_smooth_targets, load_trajectory


def path(points):
    return build_trajectory("right", [(i * 100_000_000, q + [0.]*(7-len(q))) for i, q in enumerate(points)])


class OptimizeTests(unittest.TestCase):
    def test_idle_and_small_backtracking_removed_with_exact_endpoints(self):
        source = path([[0], [0], [.10], [.105], [.10], [.2], [.2]])
        result, report = optimize(source, median_window=1)
        self.assertEqual(len(result.samples), 2)
        self.assertEqual(result.samples[0].joint_positions_rad, source.samples[0].joint_positions_rad)
        self.assertEqual(result.samples[-1].joint_positions_rad, source.samples[-1].joint_positions_rad)
        self.assertLess(result.duration_ns, source.duration_ns * 10)

    def test_large_turn_and_large_collinear_reversal_preserved(self):
        for points in ([[0, 0], [.2, 0], [.2, .2]], [[0], [.6], [.2], [1.]]):
            with self.subTest(points=points):
                result, _ = optimize(path(points), median_window=1)
                self.assertEqual([s.joint_positions_rad for s in result.samples], [s.joint_positions_rad for s in path(points).samples])

    def test_pinned_original_sample_preserved_exactly(self):
        source = path([[0], [.001], [.002], [.003], [.004]])
        result, report = optimize(source, keep_indices=[2])
        self.assertEqual(report["key_indices"], [0, 2, 4])
        self.assertEqual(result.samples[1].joint_positions_rad, source.samples[2].joint_positions_rad)

    def test_quintic_replay_speed_bound_for_all_seven_joints(self):
        source = path([[0]*7, [.2, -.3, .4, -.1, .05, -.15, .25]])
        result, _ = optimize(source, max_velocity_rad_s=.1)
        targets = list(iter_smooth_targets(result, rate_hz=100, speed_scale=1))
        for before, after in zip(targets, targets[1:]):
            seconds = (after.t_ns-before.t_ns)/1e9
            self.assertLessEqual(max(abs(a-b)/seconds for a,b in zip(after.joint_positions_rad, before.joint_positions_rad)), .1000001)

    def test_stationary_and_single_sample_keep_endpoints_without_long_idle(self):
        for points in ([[.1]], [[.1], [.1], [.1]]):
            result, _ = optimize(path(points))
            self.assertEqual(result.samples[-1].joint_positions_rad, (.1, 0., 0., 0., 0., 0., 0.))
            self.assertEqual(len(result.samples), 1 if len(points)==1 else 2)

    def test_invalid_options_and_pin_rejected(self):
        for kwargs in ({"tolerance_deg": 0}, {"max_velocity_rad_s": float("nan")}, {"median_window": 4}, {"keep_indices": [99]}):
            with self.assertRaises(ValueError):
                optimize(path([[0], [.1]]), **kwargs)

    def test_cli_overwrites_optimized_output_but_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "right.json"
            original = json.dumps(trajectory_document(path([[0], [0], [.1]])))
            source.write_text(original)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--input", str(source)]), 0)
            output = source.with_name("right_optimized.json")
            load_trajectory(json.loads(output.read_text()))
            output.write_text("old optimized result")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--input", str(source)]), 0)
            load_trajectory(json.loads(output.read_text()))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--input", str(source), "--output", str(source)]), 1)
            self.assertEqual(source.read_text(), original)

    def test_cli_invalid_joint_data_reports_error_without_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "bad.json"
            document = trajectory_document(path([[0], [.1]]))
            document["samples"][1]["joint_positions_rad"][0] = 100
            source.write_text(json.dumps(document))
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--input", str(source)]), 1)
            self.assertFalse(source.with_name("bad_optimized.json").exists())


if __name__ == "__main__":
    unittest.main()
