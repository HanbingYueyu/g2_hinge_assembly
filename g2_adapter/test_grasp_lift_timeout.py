import unittest
from unittest.mock import Mock
from control import CartesianPoseTarget
from grasp_lift_diagnostics import stop_timed_out_lift
from safety import SafetyError

class LiftTimeoutTests(unittest.TestCase):
    def test_timeout_preserves_jaw_and_logs_pose_errors(self):
        controller=Mock();controller.read_gripper_position.return_value=-.02
        controller.checked_arm_state.return_value=(0.,)*14
        stream=Mock();stream.pose_snapshot.return_value={'force_hold':True}
        origin=CartesianPoseTarget((.4,.2,.5),(0.,0.,0.,1.))
        target=CartesianPoseTarget((.4,.2,.52),origin.orientation_xyzw)
        actual=CartesianPoseTarget((.4,.2,.512),origin.orientation_xyzw)
        records=[]
        with self.assertRaisesRegex(SafetyError,'运动未到位'):
            stop_timed_out_lift(controller,stream,{'origin':origin,'target':target,'started_at':10.,'timeout_s':2.,'deadline':12.},actual,records.append,now=12.1)
        self.assertAlmostEqual(records[0]['position_error_mm'],8.)
        self.assertAlmostEqual(records[0]['actual_lift_mm'],12.)
        self.assertEqual(records[0]['orientation_error_deg'],0.)
        controller.start_left_gripper_position_stream.assert_not_called()
        controller.move_gripper.assert_not_called()
        stream.stop.assert_called_once()
        controller.hold_joint_positions.assert_called_once()
