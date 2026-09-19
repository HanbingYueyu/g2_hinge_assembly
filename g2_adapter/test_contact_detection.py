import unittest

from contact_detection import ForceContactDetector, ForceContactParameters


class ContactDetectionTests(unittest.TestCase):
    def setUp(self):
        self.parameters = ForceContactParameters(
            frame_name="arm_l_end_link",
            force_delta_threshold_n=3.0,
            consecutive_samples=3,
            baseline_samples=20,
            sample_period_s=0.02,
            require_for_grasp=True,
            relief_max_lift_m=0.005,
            relief_force_delta_threshold_n=1.0,
            relief_consecutive_samples=3,
        )

    def test_requires_three_consecutive_force_samples(self):
        detector = ForceContactDetector(self.parameters, (0.0, 0.0, 0.0))
        for sample in ((3.1, 0.0, 0.0), (3.2, 0.0, 0.0)):
            detected, delta, hits = detector.update(sample)
            self.assertFalse(detected)
            self.assertGreater(delta, 3.0)
            self.assertIn(hits, (1, 2))
        detected, delta, hits = detector.update((3.3, 0.0, 0.0))
        self.assertTrue(detected)
        self.assertGreater(delta, 3.0)
        self.assertEqual(hits, 3)

    def test_subthreshold_sample_resets_contact_debounce(self):
        detector = ForceContactDetector(self.parameters, (0.0, 0.0, 0.0))
        detector.update((3.1, 0.0, 0.0))
        detector.update((3.1, 0.0, 0.0))
        detected, _delta, hits = detector.update((2.9, 0.0, 0.0))
        self.assertFalse(detected)
        self.assertEqual(hits, 0)

    def test_allows_a_higher_close_force_limit_than_contact_trigger(self):
        parameters = ForceContactParameters(
            frame_name="arm_l_end_link",
            force_delta_threshold_n=2.0,
            consecutive_samples=1,
            baseline_samples=20,
            sample_period_s=0.02,
            require_for_grasp=True,
            relief_max_lift_m=0.015,
            relief_force_delta_threshold_n=3.0,
            relief_consecutive_samples=3,
        )
        parameters.validate()

    def test_relief_extra_lift_defaults_to_zero_and_is_bounded(self):
        self.assertEqual(self.parameters.relief_extra_lift_m, 0.0)
        base = dict(
            frame_name="arm_l_end_link",
            force_delta_threshold_n=2.0,
            consecutive_samples=1,
            baseline_samples=20,
            sample_period_s=0.02,
            require_for_grasp=True,
            relief_max_lift_m=0.015,
            relief_force_delta_threshold_n=1.0,
            relief_consecutive_samples=3,
        )
        for value in (0.0, 0.01, 0.02):
            ForceContactParameters(**base, relief_extra_lift_m=value).validate()
        for value in (-0.001, 0.021, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ForceContactParameters(**base, relief_extra_lift_m=value).validate()


if __name__ == "__main__":
    unittest.main()
