import unittest
from unittest.mock import Mock, patch
from bilateral_runtime import BilateralRuntime, OPEN_GRIPPER_RAD, GRIPPER_OPEN_TOLERANCE_RAD
from safety import SafetyError


class ReturnGripperTests(unittest.TestCase):
    def test_return_uses_existing_stream_feedback_before_emitting_success(self):
        runtime = BilateralRuntime.__new__(BilateralRuntime)
        runtime.controller = Mock()
        runtime.emit = Mock(return_value={'event': 'left_gripper_opened'})
        with patch('left_grasp_gdk.open_left_gripper_with_feedback', return_value={
            'target_rad': OPEN_GRIPPER_RAD, 'actual_position_rad': -.76}) as open_jaw:
            runtime.open_left_gripper(hold_pose_stream=True)
        open_jaw.assert_called_once_with(runtime.controller, target_rad=OPEN_GRIPPER_RAD,
                                        tolerance_rad=GRIPPER_OPEN_TOLERANCE_RAD, timeout_s=6.)
        runtime.controller.move_gripper.assert_not_called()
        runtime.emit.assert_called_once()

    def test_feedback_failure_does_not_report_success(self):
        runtime = BilateralRuntime.__new__(BilateralRuntime)
        runtime.controller = Mock()
        runtime.emit = Mock()
        with patch('left_grasp_gdk.open_left_gripper_with_feedback', side_effect=SafetyError('timeout')):
            with self.assertRaises(SafetyError):
                runtime.open_left_gripper(hold_pose_stream=True)
        runtime.emit.assert_not_called()
