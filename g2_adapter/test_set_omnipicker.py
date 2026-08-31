import unittest

from set_omnipicker import CLOSE_RAD, OPEN_RAD, target_position_rad


class OmnipickerTargetTests(unittest.TestCase):
    def test_open_percentage_maps_between_measured_endpoints(self):
        self.assertEqual(target_position_rad(target=None, open_percent=0.0)[0], CLOSE_RAD)
        self.assertEqual(target_position_rad(target=None, open_percent=100.0)[0], OPEN_RAD)
        self.assertAlmostEqual(target_position_rad(target=None, open_percent=50.0)[0], OPEN_RAD / 2.0)

    def test_named_target_stays_backward_compatible(self):
        self.assertEqual(target_position_rad(target="open", open_percent=None)[0], OPEN_RAD)
        self.assertEqual(target_position_rad(target="close", open_percent=None)[0], CLOSE_RAD)

    def test_rejects_percentage_outside_measured_range(self):
        with self.assertRaises(ValueError):
            target_position_rad(target=None, open_percent=100.1)


if __name__ == "__main__":
    unittest.main()
