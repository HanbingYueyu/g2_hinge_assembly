import math
import unittest
from types import SimpleNamespace

from demo_joint_impedance_move import parse_args, validate_args, move_joint_impedance


class FakeController:
    def __init__(self, follows=True):
        self.gdk = SimpleNamespace(Impedance=lambda **kw: SimpleNamespace(**kw))
        self.robot = self
        self.state = [0.] * 14
        self.events = []
        self.follows = follows

    def checked_arm_state(self):
        return tuple(self.state)

    def set_control_mode(self, mode):
        self.events.append(('mode', mode))

    def set_joint_impedance(self, values, group):
        self.events.append(('kp', group, [v.kp for v in values]))

    def move_preset(self, side, target, **kw):
        self.events.append(('move', side, list(target), kw))
        if self.follows:
            offset = 0 if side == 'left' else 7
            self.state[offset:offset+7] = target

    def move_arm_joint_servo(self, target, timestep, group, low_latency):
        assert timestep == .01 and low_latency is False
        self.events.append(('hold', group, list(target)))


class JointImpedanceDemoTests(unittest.TestCase):
    def test_right_arm_uses_seven_joint_target_and_rebases_before_position_mode(self):
        args = parse_args(['--side', 'right'])
        validate_args(args)
        controller = FakeController()
        result = move_joint_impedance(controller, args)
        self.assertEqual(controller.events[0], ('mode', 3))
        self.assertEqual(controller.events[1], ('kp', 1, [400,400,400,500,300,200,200]))
        self.assertEqual(controller.events[2][1], 'right')
        self.assertEqual(len(controller.events[2][2]), 7)
        self.assertAlmostEqual(controller.events[2][2][-1], math.radians(5))
        self.assertEqual(controller.state[:7], [0.] * 7)
        self.assertEqual(controller.events[-2], ('hold', 1, list(result)))
        self.assertEqual(controller.events[-1], ('mode', 1))

    def test_timeout_holds_measured_not_unreached_target(self):
        controller = FakeController(follows=False)
        now = [0.]
        def sleep(dt):
            now[0] += dt
        with self.assertRaisesRegex(RuntimeError, '未到达'):
            move_joint_impedance(controller, parse_args([]), clock=lambda: now[0], sleep=sleep)
        self.assertEqual(controller.events[-2], ('hold', 0, [0.] * 7))
        self.assertEqual(controller.events[-1], ('mode', 1))

    def test_interrupt_rebases_before_leaving_impedance(self):
        controller = FakeController(follows=False)
        def interrupt(dt):
            raise KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            move_joint_impedance(controller, parse_args([]), sleep=interrupt)
        self.assertEqual(controller.events[-2], ('hold', 0, [0.] * 7))
        self.assertEqual(controller.events[-1], ('mode', 1))

    def test_invalid_kp_and_velocity_are_rejected(self):
        for tokens in (['--kp', '5000', '50', '50', '60', '30', '30', '30'],
                       ['--velocity-rad-s', '0'], ['--delta-deg', 'nan', '0','0','0','0','0','0']):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                validate_args(parse_args(tokens))
