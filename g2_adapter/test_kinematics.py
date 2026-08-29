import tempfile
import unittest
from pathlib import Path

import numpy as np

from kinematics import UrdfKinematics


URDF = """<robot name="test">
<link name="base"/><link name="one"/><link name="tip"/>
<joint name="j1" type="revolute"><parent link="base"/><child link="one"/>
  <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
<joint name="fixed" type="fixed"><parent link="one"/><child link="tip"/>
  <origin xyz="1 0 0" rpy="0 0 0"/></joint>
</robot>"""


class KinematicsTests(unittest.TestCase):
    def setUp(self):
        handle = tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False)
        handle.write(URDF)
        handle.close()
        self.path = Path(handle.name)
        self.model = UrdfKinematics(self.path)

    def tearDown(self):
        self.path.unlink()

    def test_forward(self):
        pose = self.model.forward("base", "tip", {"j1": np.pi / 2})
        np.testing.assert_allclose(pose[:3, 3], [0, 1, 0], atol=1e-8)

    def test_jacobian(self):
        jacobian = self.model.jacobian("base", "tip", {"j1": 0}, ["j1"])
        np.testing.assert_allclose(jacobian[:, 0], [0, 1, 0, 0, 0, 1])

    def test_damped_delta_is_bounded(self):
        delta = self.model.damped_joint_delta(
            "base", "tip", {"j1": 0}, ["j1"], [0, 0.1, 0], max_joint_delta=0.01
        )
        self.assertLessEqual(abs(delta[0]), 0.01)

    def test_translation_waypoints_reach_requested_translation_with_bounded_increments(self):
        waypoints = self.model.translation_waypoints(
            "base",
            "tip",
            {"j1": 0.0},
            ["j1"],
            [np.sqrt(1.0 - 0.01**2) - 1.0, 0.01, 0.0],
            max_cartesian_step_m=0.002,
            max_joint_delta=0.005,
            tolerance_m=3e-5,
        )

        self.assertGreaterEqual(len(waypoints), 5)
        self.assertTrue(all(abs(next_value - previous) <= 0.005 + 1e-12 for previous, next_value in zip((0.0, *[item[0] for item in waypoints[:-1]]), [item[0] for item in waypoints])))
        pose = self.model.forward("base", "tip", {"j1": waypoints[-1][0]})
        np.testing.assert_allclose(pose[:3, 3], [np.sqrt(1.0 - 0.01**2), 0.01, 0.0], atol=1e-5)


if __name__ == "__main__":
    unittest.main()
