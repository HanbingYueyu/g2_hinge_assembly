import math
import unittest

from demo_joint_impedance_move import parse_args, validate_args, move_joint_impedance
from control import G2Controller
from test_control import FakeGdk, FakeRobot


class FakeController:
    def __init__(self, follows=True):
        self.robot = self
        self.state = [0.] * 14
        self.events = []
        self.follows = follows

    def checked_arm_state(self):
        return tuple(self.state)

    def set_control_mode(self, mode):
        self.events.append(('mode', mode))

    def move_preset(self, side, target, **kw):
        self.events.append(('move', side, list(target), kw))
        if self.follows:
            offset = 0 if side == 'left' else 7
            self.state[offset:offset+7] = target

    def prepare_joint_impedance_control(self, sides):
        self.events.append(('standard', tuple(sides)))

    def move_arm_joint_servo(self, target, timestep, group, low_latency):
        assert timestep == .01 and low_latency is False
        self.events.append(('hold', group, list(target)))

    def hold_joint_positions(self, side, positions):
        group = 0 if side == 'left' else 1
        self.events.append(('hold', group, list(positions)))


class JointImpedanceDemoTests(unittest.TestCase):
    def test_right_arm_uses_seven_joint_target_and_rebases_before_standard_impedance(self):
        args = parse_args(['--side', 'right'])
        validate_args(args)
        controller = FakeController()
        result = move_joint_impedance(controller, args)
        self.assertEqual(controller.events[0], ('mode', 3))
        self.assertEqual(controller.events[1][1], 'right')
        self.assertEqual(len(controller.events[1][2]), 7)
        self.assertAlmostEqual(controller.events[1][2][-1], math.radians(5))
        self.assertEqual(tuple(controller.events[1][3]['joint_impedance_kp']),
                         (400, 400, 400, 500, 300, 200, 200))
        self.assertEqual(controller.state[:7], [0.] * 7)
        self.assertEqual(controller.events[-2], ('hold', 1, list(result)))
        self.assertEqual(controller.events[-1], ('standard', ('right',)))

    def test_timeout_holds_measured_not_unreached_target(self):
        controller = FakeController(follows=False)
        now = [0.]
        def sleep(dt):
            now[0] += dt
        with self.assertRaisesRegex(RuntimeError, '未到达'):
            move_joint_impedance(controller, parse_args([]), clock=lambda: now[0], sleep=sleep)
        self.assertEqual(controller.events[-2], ('hold', 0, [0.] * 7))
        self.assertEqual(controller.events[-1], ('standard', ('left',)))

    def test_interrupt_rebases_before_restoring_standard_impedance(self):
        controller = FakeController(follows=False)
        def interrupt(dt):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            move_joint_impedance(controller, parse_args([]), sleep=interrupt)
        self.assertEqual(controller.events[-2], ('hold', 0, [0.] * 7))
        self.assertEqual(controller.events[-1], ('standard', ('left',)))

    def test_custom_kp_reaches_planner_then_cleanup_restores_standard_mode_3(self):
        robot = FakeRobot()
        observed = []
        def submit(request):
            observed.append((robot.control_mode, tuple(
                item.kp for item in robot.joint_impedance_calls[-1][0]
            )))
            robot.requests.append(request)
        robot.joint_control_request = submit
        controller = G2Controller(FakeGdk, robot, allow_motion=True)
        args = parse_args(['--side', 'right', '--delta-deg', *(['0'] * 7),
                           '--kp', '50', '50', '50', '60', '30', '30', '30'])
        move_joint_impedance(controller, args, sleep=lambda _: None)
        self.assertEqual(observed, [(3, (50, 50, 50, 60, 30, 30, 30))])
        self.assertEqual(robot.control_mode, 3)
        self.assertEqual(
            [(group, tuple(item.kp for item in values)) for values, group in robot.joint_impedance_calls][-1],
            (1, (400, 400, 400, 500, 300, 200, 200)),
        )

    def test_invalid_kp_and_velocity_are_rejected(self):
        for tokens in (['--kp', '5000', '50', '50', '60', '30', '30', '30'],
                       ['--velocity-rad-s', '0'], ['--delta-deg', 'nan', '0','0','0','0','0','0']):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                validate_args(parse_args(tokens))
