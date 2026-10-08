import unittest
from control import G2Controller, MotionDisabledError
from test_control import FakeGdk, FakeRobot

class LeftGraspModeTests(unittest.TestCase):
    def test_new_left_grasp_clears_previous_tool_phase(self):
        robot=FakeRobot()
        controller=G2Controller(FakeGdk,robot,allow_motion=True)
        controller.set_position_control_phase(True)
        controller.begin_left_grasp_control()
        self.assertFalse(controller.position_control_phase)
        self.assertEqual(controller.motion_status_summary()['control_mode'],3)
    def test_readonly_does_not_clear_phase_or_send_commands(self):
        controller=G2Controller(FakeGdk,FakeRobot(),allow_motion=False)
        controller.set_position_control_phase(True)
        with self.assertRaises(MotionDisabledError):
            controller.begin_left_grasp_control()
        self.assertTrue(controller.position_control_phase)
