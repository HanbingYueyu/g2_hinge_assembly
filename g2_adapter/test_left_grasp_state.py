import unittest
from dataclasses import replace

from left_grasp_state import GraspParameters, GraspPhase, LeftGraspState, Observation, requires_calibrated_config


def observation(*, center=(320.0, 240.0), depth=0.20):
    return Observation(center, depth, 400.0, 400.0, 320.0, 240.0)


class LeftGraspStateTests(unittest.TestCase):
    def setUp(self):
        self.parameters = GraspParameters(
            pixel_tolerance=12.0,
            center_confirmations=3,
            max_xy_step_m=0.002,
            target_u_px=320.0,
            target_v_px=240.0,
            pre_descent_depth_m=0.145,
            pre_descent_confirmations=3,
            pre_descent_step_m=0.001,
            pre_descent_duration_s=0.1,
            final_descent_speed_mps=0.03,
            final_descent_duration_s=0.1,
        )

    def test_coarse_completion_never_requests_descent(self):
        fine = replace(self.parameters, target_u_px=380)
        machine = LeftGraspState(fine, self.parameters)
        for frame in range(3):
            step = machine.update(observation(), now=frame)
        self.assertEqual(step.reason, "coarse_xy_complete_wait_step")
        self.assertEqual(step.phase, GraspPhase.ALIGN_XY)
        self.assertIsNone(step.base_delta_m)
        self.assertFalse(machine.coarse_active)
        self.assertEqual(machine.parameters.target_u_px, 380)
        self.assertEqual(machine.update(observation(), now=4).reason, "xy_correction")
        machine.reset_for_regrasp()
        self.assertTrue(machine.coarse_active)
        self.assertEqual(machine.parameters.target_u_px, 320)

    def test_xy_then_force_guarded_descent_then_grasp(self):
        machine = LeftGraspState(self.parameters)
        correction = machine.update(observation(center=(420.0, 240.0)), now=0.0)
        self.assertEqual(correction.phase, GraspPhase.ALIGN_XY)
        self.assertEqual(correction.reason, "xy_correction")
        self.assertLessEqual(correction.camera_delta_m[0], 0.002)
        for now in (0.1, 0.2, 0.3):
            step = machine.update(observation(), now=now)
        self.assertEqual(step.phase, GraspPhase.FINAL_DESCENT)
        step = machine.update(observation(depth=0.12), now=0.4)
        self.assertEqual(step.base_delta_m, (0.0, 0.0, -0.003))
        self.assertEqual(step.motion_duration_s, 0.1)
        step = machine.confirm_final_descent()
        self.assertEqual(step.phase, GraspPhase.GRASP)
        self.assertTrue(step.should_close_gripper)
        self.assertEqual(machine.confirm_grasp().phase, GraspPhase.GRASPED)

    def test_uses_calibrated_grasp_pixel_not_camera_principal_point(self):
        parameters = GraspParameters(
            pixel_tolerance=12.0,
            center_confirmations=1,
            max_xy_step_m=0.002,
            target_u_px=264.5,
            target_v_px=554.4,
            pre_descent_depth_m=0.145,
            pre_descent_confirmations=3,
            pre_descent_step_m=0.001,
            pre_descent_duration_s=0.1,
            final_descent_speed_mps=0.01,
            final_descent_duration_s=0.5,
        )
        machine = LeftGraspState(parameters)
        step = machine.update(observation(center=(264.5, 554.4), depth=0.134), now=0.0)
        self.assertEqual(step.phase, GraspPhase.FINAL_DESCENT)
        self.assertIsNone(step.camera_delta_m)

    def test_force_guarded_descent_ignores_depth_as_a_motion_trigger(self):
        machine = LeftGraspState(self.parameters)
        for now in (0.1, 0.2, 0.3):
            machine.update(observation(), now=now)
        step = machine.update(observation(depth=0.05), now=0.4)
        self.assertEqual(step.phase, GraspPhase.FINAL_DESCENT)
        self.assertEqual(step.reason, "force_guarded_final_descent")
        self.assertFalse(step.should_close_gripper)

    def test_reacquired_target_must_align_again(self):
        machine = LeftGraspState(self.parameters)
        for now in (0.1, 0.2, 0.3):
            machine.update(observation(), now=now)
        self.assertEqual(machine.phase, GraspPhase.FINAL_DESCENT)
        machine.reset_alignment()
        self.assertEqual(machine.phase, GraspPhase.ALIGN_XY)
        self.assertEqual(machine.centered_frames, 0)

    def test_centered_target_plans_fixed_timed_descent_then_allows_close(self):
        """A depth fluctuation must not replace the configured 5 mm / 0.5 s descent."""
        parameters = GraspParameters(
            pixel_tolerance=12.0,
            center_confirmations=1,
            max_xy_step_m=0.002,
            target_u_px=320.0,
            target_v_px=240.0,
            pre_descent_depth_m=0.145,
            pre_descent_confirmations=3,
            pre_descent_step_m=0.001,
            pre_descent_duration_s=0.1,
            final_descent_speed_mps=0.01,
            final_descent_duration_s=0.5,
        )
        machine = LeftGraspState(parameters)

        aligned = machine.update(observation(depth=0.145), now=0.0)
        self.assertEqual(aligned.phase, GraspPhase.FINAL_DESCENT)
        self.assertEqual(aligned.reason, "xy_centered_force_primary_descent")

        descent = machine.update(observation(depth=0.001), now=0.1)
        self.assertEqual(descent.base_delta_m, (0.0, 0.0, -0.005))
        self.assertEqual(descent.motion_duration_s, 0.5)
        self.assertFalse(descent.should_close_gripper)

        close = machine.confirm_final_descent()
        self.assertEqual(close.phase, GraspPhase.GRASP)
        self.assertTrue(close.should_close_gripper)

    def test_descent_only_motion_test_is_permitted_while_full_grasp_stays_disarmed(self):
        self.assertFalse(requires_calibrated_config(allow_motion=True, stop_after="descend"))
        self.assertFalse(requires_calibrated_config(allow_motion=True, stop_after="xy"))
        self.assertTrue(requires_calibrated_config(allow_motion=True, stop_after="none"))
        self.assertFalse(requires_calibrated_config(allow_motion=False, stop_after="descend"))

    def test_depth_cannot_request_an_unbounded_pre_descent(self):
        parameters = GraspParameters(
            pixel_tolerance=12.0,
            center_confirmations=1,
            max_xy_step_m=0.002,
            target_u_px=320.0,
            target_v_px=240.0,
            pre_descent_depth_m=0.145,
            pre_descent_confirmations=3,
            pre_descent_step_m=0.001,
            pre_descent_duration_s=0.1,
            final_descent_speed_mps=0.01,
            final_descent_duration_s=0.5,
        )
        machine = LeftGraspState(parameters)

        reached = machine.update(observation(depth=0.216), now=0.0)
        self.assertEqual(reached.phase.value, "final_descent")
        final = machine.update(observation(depth=0.216), now=0.1)
        self.assertEqual(final.base_delta_m, (0.0, 0.0, -0.005))
        self.assertEqual(final.motion_duration_s, 0.5)


if __name__ == "__main__":
    unittest.main()
