import unittest
from pathlib import Path

import numpy as np

from left_grasp_gdk import (
    prepare_yolo_input,
    project_end_point_to_color_pixel,
    screen_gripper_target_pixel,
    servo_pixel,
    urdf_gripper_center_end_m,
    yolo_box_fraction_grasp_pixel,
)


class YoloBoxGraspPixelTests(unittest.TestCase):
    def test_yolo_inference_uses_full_frame_and_returns_central_valid_roi(self):
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        bgr, valid_roi = prepare_yolo_input(rgb)
        self.assertEqual(bgr.shape, rgb.shape)
        self.assertEqual(valid_roi, (30, 15, 170, 85))

    def test_uses_box_55_percent_up_from_bottom(self):
        item = {"bbox_xyxy": [3.0, 2.0, 18.0, 11.0]}
        result = yolo_box_fraction_grasp_pixel(item)
        self.assertAlmostEqual(result[0], 10.5)
        self.assertAlmostEqual(result[1], 6.05)

    def test_gripper_target_is_screen_center_and_55_percent_up_from_bottom(self):
        np.testing.assert_allclose(screen_gripper_target_pixel((1056, 1280, 3)), [640.0, 475.2])

    def test_supports_an_explicit_fraction(self):
        item = {"bbox_xyxy": [2.0, 1.0, 8.0, 9.0]}
        self.assertEqual(yolo_box_fraction_grasp_pixel(item, 0.5), [5.0, 5.0])

    def test_servo_prefers_yolo_box_landmark(self):
        item = {"center_px": [2.0, 3.0], "yolo_box_grasp_pixel_px": [5.0, 7.0]}
        self.assertEqual(servo_pixel(item), [5.0, 7.0])

    def test_servo_uses_yolo_landmark_even_if_legacy_step_data_is_present(self):
        item = {
            "center_px": [2.0, 3.0],
            "yolo_box_grasp_pixel_px": [5.0, 7.0],
            "step_pose": {"accepted": True, "grasp_pixel_px": [11.0, 13.0]},
        }
        self.assertEqual(servo_pixel(item), [5.0, 7.0])

    def test_urdf_center_is_midpoint_of_fixed_symmetric_anchors(self):
        root = Path(__file__).resolve().parents[1]
        center = urdf_gripper_center_end_m(
            root / "robot_description/G2_t2v2_crsB_omnipicker.urdf", -0.51025
        )
        np.testing.assert_allclose(center, [0.0, 0.0, 0.1101], atol=1e-9)

    def test_projects_end_point_through_inverse_camera_extrinsic(self):
        pixel = project_end_point_to_color_pixel(
            np.asarray([0.0, 0.0, 1.0]),
            {"translation_m": [0.0, 0.0, 0.0], "rotation_xyzw": [0.0, 0.0, 0.0, 1.0]},
            {"fx": 500.0, "fy": 500.0, "ppx": 320.0, "ppy": 240.0},
        )
        self.assertEqual(pixel, [320.0, 240.0])


if __name__ == "__main__":
    unittest.main()
