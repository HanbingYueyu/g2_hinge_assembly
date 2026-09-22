import json
import math
import tempfile
import unittest
from pathlib import Path

try:
    from .cartesian_teach import (
        BODY_JOINT_NAMES,
        CartesianSample,
        CartesianTrajectory,
        cartesian_document,
        load_joint_document,
    )
    from .cartesian_trajectory_optimizer import optimize_cartesian_trajectory
    from .optimize_cartesian_teach_trajectory import main
except ImportError:
    from cartesian_teach import (
        BODY_JOINT_NAMES,
        CartesianSample,
        CartesianTrajectory,
        cartesian_document,
        load_joint_document,
    )
    from cartesian_trajectory_optimizer import optimize_cartesian_trajectory
    from optimize_cartesian_teach_trajectory import main


class CartesianOptimizationTests(unittest.TestCase):
    def _trajectory(self, samples):
        return CartesianTrajectory(
            "left", "base_link", "arm_l_end_link",
            tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8)),
            {name: 0.0 for name in BODY_JOINT_NAMES},
            tuple(samples),
        )

    def test_removes_stationary_samples_and_preserves_absolute_endpoints(self):
        samples = tuple(
            CartesianSample(
                index * 100_000_000,
                (0.0 if index < 3 else (index - 2) * 0.01, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
                (0.01 * index,) * 7,
            )
            for index in range(6)
        )
        result, report = optimize_cartesian_trajectory(
            self._trajectory(samples),
            position_tolerance_m=0.0005,
            orientation_tolerance_rad=math.radians(0.5),
            jitter_position_m=0.0002,
            jitter_orientation_rad=math.radians(0.2),
            median_window=3,
        )

        self.assertLess(len(result.samples), len(samples))
        self.assertEqual(result.samples[0].position_m, samples[0].position_m)
        self.assertEqual(result.samples[-1].position_m, samples[-1].position_m)
        self.assertEqual(result.samples[0].t_ns, 0)
        self.assertTrue(all(a.t_ns < b.t_ns for a, b in zip(result.samples, result.samples[1:])))
        self.assertEqual(report["original_samples"], len(samples))
        self.assertEqual(report["keyframes"], len(result.samples))

    def test_preserves_a_cartesian_bend_above_tolerance(self):
        samples = (
            CartesianSample(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0,) * 7),
            CartesianSample(100_000_000, (0.05, 0.05, 0.0), (0.0, 0.0, 0.0, 1.0), (0.1,) * 7),
            CartesianSample(200_000_000, (0.10, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.2,) * 7),
        )
        result, _ = optimize_cartesian_trajectory(
            self._trajectory(samples),
            position_tolerance_m=0.001,
            orientation_tolerance_rad=math.radians(0.5),
            jitter_position_m=0.0002,
            jitter_orientation_rad=math.radians(0.2),
            median_window=1,
        )

        self.assertEqual(len(result.samples), 3)
        self.assertEqual(result.samples[1].position_m, samples[1].position_m)

    def test_repairs_an_isolated_large_position_spike_before_simplifying(self):
        samples = tuple(
            CartesianSample(
                index * 100_000_000,
                position,
                (0.0, 0.0, 0.0, 1.0),
                (0.01 * index,) * 7,
            )
            for index, position in enumerate(((0.00, 0.0, 0.0), (0.01, 0.0, 0.0),
                                               (0.20, 0.0, 0.0), (0.03, 0.0, 0.0),
                                               (0.04, 0.0, 0.0)))
        )
        result, report = optimize_cartesian_trajectory(
            self._trajectory(samples),
            position_tolerance_m=0.001,
            orientation_tolerance_rad=math.radians(0.5),
            jitter_position_m=0.001,
            jitter_orientation_rad=math.radians(0.5),
            median_window=3,
            outlier_position_m=0.02,
            outlier_orientation_rad=math.radians(3.0),
        )

        self.assertLess(max(sample.position_m[0] for sample in result.samples), 0.1)
        self.assertEqual(report["isolated_outliers_repaired"], 1)


class CombinedOptimizationSolveCliTests(unittest.TestCase):
    def test_cli_optimizes_raw_cartesian_and_writes_solved_document(self):
        root = Path(__file__).resolve().parents[1]
        urdf = root / "robot_description/G2_t2v2_crsB_omnipicker.urdf"
        names = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
        fixed = {name: 0.0 for name in BODY_JOINT_NAMES}
        try:
            from .g2_kinematics import UrdfKinematics
            from .cartesian_teach import matrix_to_quaternion
        except ImportError:
            from g2_kinematics import UrdfKinematics
            from cartesian_teach import matrix_to_quaternion

        model = UrdfKinematics(urdf)
        joints = (0.1, -0.4, 0.2, -0.8, 0.1, 0.3, -0.2)
        pose = model.forward("base_link", "arm_l_end_link", {**fixed, **dict(zip(names, joints))})
        raw = CartesianTrajectory(
            "left", "base_link", "arm_l_end_link", names, fixed,
            (CartesianSample(0, tuple(pose[:3, 3]), matrix_to_quaternion(pose[:3, :3]), joints),),
        )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "raw.json"
            output = Path(directory) / "solved.json"
            source.write_text(json.dumps(cartesian_document(raw, urdf_sha256="ignored")), encoding="utf-8")

            code = main([
                "--input", str(source), "--output", str(output), "--urdf", str(urdf),
                "--rate-hz", "100", "--max-velocity-rad-s", "1.0",
                "--max-acceleration-rad-s2", "10.0",
            ])

            self.assertEqual(code, 0)
            document = json.loads(output.read_text(encoding="utf-8"))
            trajectory, report = load_joint_document(document)
            self.assertEqual(document["kind"], "g2_solved_absolute_joint_trajectory")
            self.assertTrue(report.replay_ready, report.errors)
            self.assertEqual(len(trajectory.points), 1)
            self.assertIn("optimization", document)


if __name__ == "__main__":
    unittest.main()
