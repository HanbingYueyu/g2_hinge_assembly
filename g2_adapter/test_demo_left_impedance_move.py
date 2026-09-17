import unittest
from types import SimpleNamespace

from control import CartesianPoseTarget
from demo_left_impedance_move import move_left_impedance, validate_args, parse_args


class ImpedanceDemoTests(unittest.TestCase):
    def test_default_stiffness_matches_measured_right_arm(self):
        args = parse_args([])
        validate_args(args)
        self.assertEqual(tuple(args.kp), (2000, 2000, 2000, 200, 200, 200))

    def test_rejected_mode_entry_does_not_send_cleanup_motion(self):
        writes = []
        class Controller:
            def read_end_effector_pose(self, frame):
                return CartesianPoseTarget((0, 0, .5), (0, 0, 0, 1))
            def enable_left_cartesian_impedance(self, kp):
                raise RuntimeError('mode entry rejected')
            def motion_status_summary(self):
                return {'control_mode': 1}
            def _send_left_cartesian_pose(self, *args):
                writes.append('motion')
            def set_control_mode(self, mode):
                writes.append('mode')
        with self.assertRaisesRegex(RuntimeError, 'rejected'):
            move_left_impedance(Controller(), delta_mm=(0,0,20), speed_mm_s=5, kp=(600,)*3+(30,)*3)
        self.assertEqual(writes, [])

    def run_demo(self, interrupt=False):
        events = []
        origin = CartesianPoseTarget((.3, -.2, .5), (0., 0., 1., 0.))
        now = [0.0]
        class Stream:
            def update_target(self, target):
                events.append(('target', target))
            def raise_if_failed(self):
                pass
            def stop(self):
                events.append(('stop',))
        class Controller:
            def read_end_effector_pose(self, frame):
                self.frame = frame
                return origin
            def enable_left_cartesian_impedance(self, kp):
                events.append(('impedance', kp))
            def start_left_cartesian_impedance_stream(self, initial, **kwargs):
                events.append(('start', initial))
                return Stream()
            def _send_left_cartesian_pose(self, pose, lifetime):
                events.append(('hold', pose))
            def set_control_mode(self, mode):
                events.append(('mode', mode))
        def sleep(seconds):
            if interrupt:
                raise KeyboardInterrupt()
            now[0] += seconds
        return events, lambda: move_left_impedance(Controller(), delta_mm=(0, 0, 20), speed_mm_s=5, kp=(600,)*3+(30,)*3, clock=lambda: now[0], sleep=sleep), origin

    def test_ramps_target_with_fixed_orientation_then_holds_measured_pose(self):
        events, run, origin = self.run_demo()
        result = run()
        targets = [event[1] for event in events if event[0] == 'target']
        self.assertEqual(targets[-1].position_m, (.3, -.2, .52))
        self.assertTrue(all(p.orientation_xyzw == origin.orientation_xyzw for p in targets))
        self.assertLessEqual(max(b.position_m[2]-a.position_m[2] for a,b in zip(targets,targets[1:])), .000101)
        self.assertEqual(events[-3:], [('stop',), ('hold', origin), ('mode', 1)])
        self.assertEqual(result['duration_s'], 4)

    def test_interrupt_stops_stream_before_restoring_mode(self):
        events, run, origin = self.run_demo(interrupt=True)
        with self.assertRaises(KeyboardInterrupt):
            run()
        self.assertEqual(events[-3:], [('stop',), ('hold', origin), ('mode', 1)])

    def test_rejects_invalid_speed_before_hardware_is_opened(self):
        for speed in (0, -1, float('nan')):
            with self.subTest(speed=speed), self.assertRaises(ValueError):
                validate_args(SimpleNamespace(x_mm=0, y_mm=0, z_mm=20, speed_mm_s=speed, kp=(600,)*3+(30,)*3))
