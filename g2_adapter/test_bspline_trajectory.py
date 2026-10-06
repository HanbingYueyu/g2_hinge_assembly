import unittest
import numpy as np
from cartesian_teach import CartesianSample, CartesianTrajectory
from bspline_trajectory import spline_basis, smooth_bspline


class BsplineTests(unittest.TestCase):
    def trajectory(self):
        t=np.linspace(0.,1.,101)
        samples=tuple(CartesianSample(int(u*1e9),(float(u+.15*np.sin(12*u)),float(.05*np.sin(8*u)),0.),(0.,0.,0.,1.),(0.,)*7) for u in t)
        return CartesianTrajectory('left','base_link','arm_l_end_link',tuple('abcdefg'),{},samples)

    def test_basis_is_convex_and_clamped(self):
        b=spline_basis(np.linspace(0.,1.,1001),6)
        self.assertTrue((b>=0).all())
        np.testing.assert_allclose(b.sum(axis=1),1.,atol=1e-12)
        np.testing.assert_array_equal(b[0],[1.,0.,0.,0.,0.,0.])
        np.testing.assert_array_equal(b[-1],[0.,0.,0.,0.,0.,1.])

    def test_endpoints_are_exact_and_no_endpoint_overshoot(self):
        raw=self.trajectory()
        result,report=smooth_bspline(raw)
        self.assertEqual(result.samples[0].position_m,raw.samples[0].position_m)
        self.assertEqual(result.samples[-1].position_m,raw.samples[-1].position_m)
        self.assertEqual(result.samples[0].orientation_xyzw,raw.samples[0].orientation_xyzw)
        p=np.array([s.position_m for s in result.samples]);a=np.array(raw.samples[0].position_m);d=np.array(raw.samples[-1].position_m)-a
        progress=(p-a)@d/np.dot(d,d)
        self.assertGreaterEqual(progress.min(),-1e-8)
        self.assertLessEqual(progress.max(),1.+1e-8)
        controls=np.array(report['control_positions_m'])
        original=np.array([s.position_m for s in raw.samples])
        self.assertTrue((controls>=original.min(axis=0)-1e-8).all())
        self.assertTrue((controls<=original.max(axis=0)+1e-8).all())

    def test_very_short_and_closed_paths_are_rejected(self):
        raw=self.trajectory()
        raw=CartesianTrajectory(raw.side,raw.base_frame,raw.tip_frame,raw.joint_names,{},(raw.samples[0],raw.samples[0]))
        with self.assertRaises(ValueError):smooth_bspline(raw)
