import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import left_grasp_gdk as flow


class LeftRgbPerceptionTests(unittest.TestCase):
    def test_full_flow_perception_reads_only_color_and_skips_duplicate_frames(self):
        from run_bilateral_flow_gdk import StageSettings, parse_args
        args = flow.parse_args(StageSettings(parse_args([])).left_argv())
        args.min_area = 1
        frames = iter([1, 1, 2])

        class Camera:
            def get_latest_image(self, *_):
                return SimpleNamespace(timestamp_ns=next(frames), width=20, height=20)

            def get_nearest_image(self, *_):
                raise AssertionError('Left RGB flow requested depth')

            def get_camera_intrinsic(self, kind):
                if kind != 'color':
                    raise AssertionError('Left RGB flow requested depth calibration')
                return None

        class Tf:
            def get_tf_from_sensor(self, kind):
                if kind != 'color_to_end':
                    raise AssertionError('Left RGB flow requested depth extrinsics')
                return None

        gdk = SimpleNamespace(SensorExtrinsicType=SimpleNamespace(
            kLeftHandRGBDToArmLEndLink='color_to_end'))
        worker = flow.PersistentPerception(camera=Camera(), color_type='color',
            depth_type='depth', gdk=gdk, tf=Tf(), processor=None,
            yolo_model=SimpleNamespace(names={0: 'hinge'}), args=args,
            shared=None, yolo_model_path=args.yolo_model)
        mask = np.zeros((20, 20), dtype=bool)
        mask[6:14, 6:14] = True
        with patch.object(flow, 'decode_color_rgb', return_value=np.zeros((20, 20, 3), dtype=np.uint8)), \
             patch.object(flow, 'intrinsic_to_dict', return_value={}), \
             patch.object(flow, 'transform_to_dict', return_value={}), \
             patch.object(flow, 'align_depth_to_color', side_effect=AssertionError('Unexpected alignment')), \
             patch.object(flow, 'infer_yolo_masks', return_value=([mask], [.9], [0])) as infer:
            worker._run()
        self.assertIsInstance(worker._failure, StopIteration)
        self.assertEqual(worker._sequence, 2)
        self.assertEqual(infer.call_count, 2)
        self.assertEqual(worker._latest['statistics'][0]['area_px'], 64)
        self.assertEqual(worker._latest['statistics'][0]['center_px'], [9.5, 9.5])

    def test_rgb_verification_rejects_unstable_area_even_with_stable_depth(self):
        args = flow.parse_args([])
        samples = [dict(center_delta_px=0., area_fraction_change=.5, depth_change_m=0.)] * 3
        report = flow.rgb_grasp_verification(samples, args)
        self.assertFalse(report['held_with_gripper'])

    def test_rgb_verification_accepts_stable_held_mask_without_depth(self):
        args = flow.parse_args([])
        samples = [dict(center_delta_px=1., area_fraction_change=.01)] * 3
        report = flow.rgb_grasp_verification(samples, args)
        self.assertTrue(report['held_with_gripper'])


if __name__ == '__main__':
    unittest.main()

class VisualReacquisitionTests(unittest.TestCase):
    def test_large_image_error_has_velocity_lookahead(self):
        from control import VisualVelocityFilter
        delta = flow.continuous_visual_xy_delta(
            np.array([100., 0.]), np.eye(2)*1000.,
            VisualVelocityFilter(), speed_mps=.03, now=1.)
        self.assertGreater(np.linalg.norm(delta), .002)
        self.assertLessEqual(np.linalg.norm(delta), .015001)

    def test_reacquisition_hold_discards_old_velocity_and_target(self):
        from unittest.mock import Mock
        from control import CartesianPoseTarget, VisualVelocityFilter
        controller = Mock()
        measured = CartesianPoseTarget((.4,.2,.9), (0,0,0,1))
        controller.read_end_effector_pose.return_value = measured
        stream = Mock()
        filt = VisualVelocityFilter()
        filt.update(np.ones(3), .03, 1.)
        with patch.object(flow, 'measured_left_end_link_pose', return_value=measured):
            held = flow.hold_visual_motion_for_reacquisition(controller, stream, measured, filt)
        stream.hold_at_pose.assert_called_once_with(held)
        np.testing.assert_array_equal(filt.velocity, np.zeros(3))
        self.assertIsNone(filt.updated_at)

class VisualAlignmentSettlingTests(unittest.TestCase):
    def test_center_confirmation_cancels_lookahead_until_new_correction(self):
        from unittest.mock import Mock
        from control import CartesianPoseTarget, VisualVelocityFilter
        pose = CartesianPoseTarget((.4,.2,.9), (0,0,0,1))
        stream = Mock()
        filt = VisualVelocityFilter()
        center = SimpleNamespace(reason='xy_center_confirmation', camera_delta_m=None)
        with patch.object(flow, 'measured_left_end_link_pose', return_value=pose):
            held, target = flow.settle_visual_alignment(Mock(), stream, pose, filt, center, False)
            self.assertTrue(held)
            self.assertEqual(target, pose)
            flow.settle_visual_alignment(Mock(), stream, pose, filt, center, True)
        stream.hold_at_pose.assert_called_once()
        correction = SimpleNamespace(reason='xy_correction', camera_delta_m=(.01,0,0))
        held, _ = flow.settle_visual_alignment(Mock(), stream, pose, filt, correction, True)
        self.assertFalse(held)
        stream.set_force_hold.assert_called_once_with(False)
