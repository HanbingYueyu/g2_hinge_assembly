import unittest
import numpy as np
from left_step_xy import xy_error, bounded_xy


class MetricXYTests(unittest.TestCase):
    def test_rotation_camera_offset_and_compensation(self):
        base = np.eye(4)
        base[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        base[:3, 3] = [1, 2, 3]
        camera = np.eye(4)
        camera[:3, 3] = [.01, 0, 0]
        np.testing.assert_allclose(xy_error(base, camera, [.02, .03, .2], [0, 0, .145], [.001, -.002]), [-.029, .028])

    def test_no_vertical_motion_and_bounded_step(self):
        step = bounded_xy([.03, .04], .002)
        np.testing.assert_allclose(step, [.0012, .0016, 0])
        np.testing.assert_allclose(bounded_xy([0, 0], .002), [0, 0, 0])

    def test_tcp_height_matters_when_wrist_is_tilted(self):
        base = np.eye(4)
        base[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
        np.testing.assert_allclose(xy_error(base, np.eye(4), [0, 0, .15], [0, 0, .145], [0, 0]), [.005, 0])
