import unittest
import numpy as np
from control import CartesianTranslationSmoother


class SmoothingTests(unittest.TestCase):
    def test_start_and_brake(self):
        smoother = CartesianTranslationSmoother()
        position = np.zeros(3)
        velocities = []
        for _ in range(250):
            new = smoother.step(position, np.array([.1, 0., 0.]), .03, .02, 0., .005)
            velocities.append((new - position) / .02)
            position = new
        self.assertLess(velocities[0][0], .03)
        self.assertAlmostEqual(position[0], .1, places=5)
        self.assertLessEqual(max(np.linalg.norm(v) for v in velocities), .030001)
        self.assertLessEqual(max(np.linalg.norm(b-a) for a,b in zip(velocities, velocities[1:])), .002001)

    def test_reverse_preserves_velocity_continuity(self):
        smoother = CartesianTranslationSmoother()
        position = np.zeros(3)
        for _ in range(20):
            position = smoother.step(position, np.array([1., 0., 0.]), .03, .02, 0., .005)
        before = smoother.velocity.copy()
        smoother.step(position, np.array([-1., 0., 0.]), .03, .02, 0., .005)
        self.assertLessEqual(np.linalg.norm(smoother.velocity-before), .002001)

    def test_error_slowdown_and_hard_stop(self):
        for error in (0., .0045):
            smoother = CartesianTranslationSmoother()
            for _ in range(30):
                smoother.step(np.zeros(3), np.ones(3), .03, .02, error, .005)
            if error == 0.:
                full = np.linalg.norm(smoother.velocity)
            else:
                self.assertLess(np.linalg.norm(smoother.velocity), full)
        result = smoother.step(np.zeros(3), np.ones(3), .03, .02, .005, .005)
        np.testing.assert_array_equal(result, np.zeros(3))
        np.testing.assert_array_equal(smoother.velocity, np.zeros(3))

class VisualVelocityTests(unittest.TestCase):
    def test_vision_lookahead_allows_cruise_speed(self):
        from control import VisualVelocityFilter
        filt = VisualVelocityFilter()
        smoother = CartesianTranslationSmoother()
        position = np.zeros(3)
        desired = position.copy()
        for tick in range(50):
            if tick % 2 == 0:
                desired = position + filt.update(np.array([.02, 0., 0.]), .03, tick*.02+1.)
            position = smoother.step(position, desired, .03, .02, 0., .005)
        self.assertGreater(position[0], .023)


    def test_filter_bounds_horizon_and_resets_after_gap(self):
        from control import VisualVelocityFilter
        filt = VisualVelocityFilter()
        first = filt.update(np.array([.02, 0., 0.]), .015, 1.)
        self.assertLessEqual(np.linalg.norm(first), .015001)
        reversed_step = filt.update(np.array([-.02, 0., 0.]), .015, 1.033)
        self.assertGreater(reversed_step[0], 0.)
        fresh = filt.update(np.array([-.02, 0., 0.]), .015, 2.)
        self.assertLess(fresh[0], 0.)

class FeedbackHeadroomTests(unittest.TestCase):
    def test_cached_feedback_does_not_overshoot_hold_boundary(self):
        sm = CartesianTranslationSmoother()
        sm.velocity[:] = [.03, 0., 0.]
        position = np.array([.0048, 0., 0.])
        for _ in range(5):
            position = sm.step(position, np.ones(3), .03, .02,
                               np.linalg.norm(position), .005, feedback_horizon_s=.05)
            self.assertLess(np.linalg.norm(position), .005)
        self.assertGreater(np.linalg.norm(sm.velocity), 0.)
