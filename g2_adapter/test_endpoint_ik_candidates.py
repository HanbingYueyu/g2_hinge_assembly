import unittest
from unittest.mock import Mock, patch
import numpy as np
from g2_kinematics import KinematicsError
from endpoint_ik_candidates import solve_candidates, submit_endpoint

class CandidateTests(unittest.TestCase):
    def test_selects_nearest_unique_solution(self):
        model=Mock()
        model.solve_pose.side_effect=[(np.ones(7)*.4,None),(np.ones(7)*.1,None)]+[(np.ones(7)*.4,None)]*8
        result=solve_candidates(model,Mock(),{},np.eye(4),np.zeros(7),'left')
        np.testing.assert_allclose(result,.1)
        self.assertGreater(model.solve_pose.call_count,1)
    def test_retries_convergence_failure(self):
        model=Mock()
        model.solve_pose.side_effect=[KinematicsError('local minimum'),(np.zeros(7),None)]
        np.testing.assert_allclose(solve_candidates(model,Mock(),{},np.eye(4),np.zeros(7),'right'),0)
    def test_unexpected_failure_propagates(self):
        model=Mock()
        model.solve_pose.side_effect=RuntimeError('cancelled')
        with self.assertRaisesRegex(RuntimeError,'cancelled'):
            solve_candidates(model,Mock(),{},np.eye(4),np.zeros(7),'left')

    def test_ambiguous_gdk_failure_never_retries(self):
        controller=Mock()
        controller.move_preset.side_effect=RuntimeError('JointControl failed')
        with patch('endpoint_ik_candidates.log_ik') as log:
            with self.assertRaisesRegex(RuntimeError,'JointControl failed'):
                submit_endpoint(controller,'right',np.zeros(7),velocity_rad_s=.1,control_mode=1)
        controller.move_preset.assert_called_once()
        self.assertFalse(log.call_args.args[0]['automatic_retry'])

    def test_all_nonconvergent_seeds_stop(self):
        model=Mock()
        model.solve_pose.side_effect=KinematicsError('unreachable')
        with patch('endpoint_ik_candidates.log_ik'):
            with self.assertRaises(KinematicsError):
                solve_candidates(model,Mock(),{},np.eye(4),np.zeros(7),'left')
        self.assertEqual(model.solve_pose.call_count,8)

class SubmissionTests(unittest.TestCase):
    def controller(self):
        c=Mock()
        c.checked_arm_state.return_value=(0.,)*14
        c.motion_status_summary.return_value={'error_code':0,'control_mode':3}
        return c
    def test_coupled_rejection_tries_next_candidate(self):
        from endpoint_ik_candidates import submit_candidates
        c=self.controller()
        c.move_preset.side_effect=[RuntimeError('JointControl failed'),None]
        reason='JointControlRequest failed\nCoupledBoundConstraint[1]\nPath generator failed at segment 0'
        with patch('endpoint_ik_candidates.capture_native_stderr') as capture, patch('endpoint_ik_candidates.log_ik'):
            capture.return_value.__enter__.return_value.read.return_value=reason
            chosen=submit_candidates(c,'left',[np.zeros(7),np.ones(7)*.1],velocity_rad_s=.2,control_mode=3)
        np.testing.assert_allclose(chosen,.1)
        self.assertEqual(c.move_preset.call_count,2)
    def test_ambiguous_failure_stops(self):
        from endpoint_ik_candidates import submit_candidates
        c=self.controller();c.move_preset.side_effect=RuntimeError('JointControl failed')
        with patch('endpoint_ik_candidates.capture_native_stderr') as capture, patch('endpoint_ik_candidates.log_ik'):
            capture.return_value.__enter__.return_value.read.return_value='timeout'
            with self.assertRaises(RuntimeError):
                submit_candidates(c,'left',[np.zeros(7),np.ones(7)],velocity_rad_s=.2,control_mode=3)
        c.move_preset.assert_called_once()
    def test_movement_blocks_retry_even_with_planning_message(self):
        from endpoint_ik_candidates import submit_candidates
        c=self.controller();c.checked_arm_state.side_effect=[(0.,)*14,(.01,)*14]
        c.move_preset.side_effect=RuntimeError('JointControl failed')
        with patch('endpoint_ik_candidates.capture_native_stderr') as capture, patch('endpoint_ik_candidates.log_ik'):
            capture.return_value.__enter__.return_value.read.return_value='JointControlRequest failed\nCoupledBoundConstraint[1]\nPath generator failed at segment 0'
            with self.assertRaises(RuntimeError):
                submit_candidates(c,'left',[np.zeros(7),np.ones(7)],velocity_rad_s=.2,control_mode=3)
        c.move_preset.assert_called_once()

class NativeCaptureTests(unittest.TestCase):
    def test_captures_native_fd_writes_and_restores_stderr(self):
        import os
        from endpoint_ik_candidates import capture_native_stderr
        before=os.fstat(2)
        with capture_native_stderr() as capture:
            os.write(2,b'Path generator failed at segment 0\n')
            self.assertIn('segment 0',capture.read())
        after=os.fstat(2)
        self.assertEqual((before.st_dev,before.st_ino),(after.st_dev,after.st_ino))
    def test_exception_also_restores_stderr(self):
        import os
        from endpoint_ik_candidates import capture_native_stderr
        before=os.fstat(2)
        with self.assertRaises(RuntimeError):
            with capture_native_stderr():
                raise RuntimeError('abort')
        after=os.fstat(2)
        self.assertEqual((before.st_dev,before.st_ino),(after.st_dev,after.st_ino))
