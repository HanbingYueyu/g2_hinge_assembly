import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cartesian_teach import (
    CartesianSample,
    CartesianTrajectory,
    JointTrajectory,
    JointTrajectoryPoint,
    matrix_to_quaternion,
    resample_cartesian_trajectory,
    solve_cartesian_trajectory,
    trim_leading_stationary,
    validate_joint_trajectory,
)
from g2_kinematics import UrdfKinematics


class CartesianTrajectoryTests(unittest.TestCase):
    def test_resample_interpolates_position_and_quaternion_without_moving_endpoints(self):
        trajectory = CartesianTrajectory(
            side="left",
            base_frame="base_link",
            tip_frame="arm_l_end_link",
            joint_names=tuple(f"j{i}" for i in range(7)),
            fixed_joint_positions={},
            samples=(
                CartesianSample(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0,) * 7),
                CartesianSample(1_000_000_000, (1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (1.0,) * 7),
            ),
        )

        result = resample_cartesian_trajectory(trajectory, rate_hz=2.0)

        self.assertEqual([sample.t_ns for sample in result.samples], [0, 500_000_000, 1_000_000_000])
        self.assertEqual(result.samples[0], trajectory.samples[0])
        self.assertEqual(result.samples[-1], trajectory.samples[-1])
        self.assertAlmostEqual(result.samples[1].position_m[0], 0.5)
        np.testing.assert_allclose(
            result.samples[1].orientation_xyzw,
            (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
            atol=1e-8,
        )

    def test_validation_rejects_joint_velocity_above_limit(self):
        trajectory = JointTrajectory(
            side="left",
            joint_names=tuple(f"j{i}" for i in range(7)),
            fixed_joint_positions={},
            source_sha256="0" * 64,
            urdf_sha256="1" * 64,
            points=(
                JointTrajectoryPoint(0, (0.0,) * 7, 0.0, 0.0),
                JointTrajectoryPoint(100_000_000, (0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 0.0, 0.0),
            ),
        )

        report = validate_joint_trajectory(
            trajectory,
            joint_limits=((-1.0, 1.0),) * 7,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=10.0,
            max_position_error_m=0.001,
            max_orientation_error_rad=0.01,
        )

        self.assertFalse(report.replay_ready)
        self.assertIn("joint velocity", " ".join(report.errors))

    def test_leading_idle_is_trimmed_but_frame_before_motion_is_kept(self):
        samples = (
            CartesianSample(0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0,) * 7),
            CartesianSample(10, (0.0001, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0,) * 7),
            CartesianSample(20, (0.001, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0,) * 7),
        )
        trajectory = CartesianTrajectory("left", "base_link", "tip", tuple(f"j{i}" for i in range(7)), {}, samples)

        trimmed, removed = trim_leading_stationary(trajectory, translation_threshold_m=0.0005, rotation_threshold_rad=0.01)

        self.assertEqual(removed, 1)
        self.assertEqual([sample.t_ns for sample in trimmed.samples], [0, 10])
        self.assertEqual(trimmed.samples[0].position_m, samples[1].position_m)


class UrdfKinematicsTests(unittest.TestCase):
    def setUp(self):
        self.urdf = Path(__file__).resolve().parents[1] / "robot_description/G2_t2v2_crsB_omnipicker.urdf"
        self.model = UrdfKinematics(self.urdf)
        self.names = tuple(f"idx2{i}_arm_l_joint{i}" for i in range(1, 8))
        self.fixed = {
            "idx01_body_joint1": -0.2,
            "idx02_body_joint2": 0.4,
            "idx03_body_joint3": -0.2,
            "idx04_body_joint4": 0.0,
            "idx05_body_joint5": 0.0,
        }

    def test_pose_ik_recovers_a_nearby_reachable_g2_pose(self):
        initial = np.array((0.1, -0.4, 0.2, -0.8, 0.1, 0.3, -0.2))
        desired = np.array((0.12, -0.42, 0.18, -0.78, 0.11, 0.29, -0.19))
        target = self.model.forward(
            "base_link",
            "arm_l_end_link",
            {**self.fixed, **dict(zip(self.names, desired))},
        )

        solution, diagnostics = self.model.solve_pose(
            "base_link",
            "arm_l_end_link",
            self.fixed,
            self.names,
            target,
            initial,
            reference=desired,
            joint_limits=(
                (-3.0718, 3.0718), (-2.0595, 2.0595), (-3.0718, 3.0718),
                (-2.4959, 1.0123), (-3.0718, 3.0718), (-1.0123, 1.0123),
                (-1.5359, 1.5359),
            ),
        )

        self.assertLess(diagnostics.position_error_m, 1e-4)
        self.assertLess(diagnostics.orientation_error_rad, 1e-3)
        self.assertLess(float(np.max(np.abs(solution - initial))), 0.2)

    def test_entire_cartesian_path_is_solved_before_it_is_marked_replay_ready(self):
        configurations = (
            (0.10, -0.40, 0.20, -0.80, 0.10, 0.30, -0.20),
            (0.11, -0.41, 0.19, -0.79, 0.105, 0.295, -0.195),
            (0.12, -0.42, 0.18, -0.78, 0.11, 0.29, -0.19),
        )
        samples = []
        for index, joints in enumerate(configurations):
            pose = self.model.forward(
                "base_link", "arm_l_end_link",
                {**self.fixed, **dict(zip(self.names, joints))},
            )
            samples.append(CartesianSample(
                index * 500_000_000,
                tuple(pose[:3, 3]),
                matrix_to_quaternion(pose[:3, :3]),
                joints,
            ))
        trajectory = CartesianTrajectory(
            "left", "base_link", "arm_l_end_link", self.names, self.fixed, tuple(samples),
        )

        solved, report = solve_cartesian_trajectory(
            trajectory,
            self.model,
            joint_limits=(
                (-3.0718, 3.0718), (-2.0595, 2.0595), (-3.0718, 3.0718),
                (-2.4959, 1.0123), (-3.0718, 3.0718), (-1.0123, 1.0123),
                (-1.5359, 1.5359),
            ),
            source_sha256="0" * 64,
            urdf_sha256="1" * 64,
            max_velocity_rad_s=1.0,
            max_acceleration_rad_s2=10.0,
        )

        self.assertTrue(report.replay_ready, report.errors)
        self.assertEqual(len(solved.points), 3)
        self.assertLess(report.max_position_error_m, 1e-4)
        self.assertLess(report.max_orientation_error_rad, 1e-3)


if __name__ == "__main__":
    unittest.main()
