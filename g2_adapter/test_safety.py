import math
import unittest

from safety import (
    ARM_JOINT_NAMES,
    OMNIPICKER_MIN,
    SafetyError,
    clamp_omnipicker,
    reorder_arm_positions,
    validate_arm_target,
    validate_segment,
)


class SafetyTests(unittest.TestCase):
    def test_reorders_by_name(self):
        states = [
            {"name": name, "motor_position": index / 100.0}
            for index, name in reversed(list(enumerate(ARM_JOINT_NAMES)))
        ]
        self.assertEqual(reorder_arm_positions({"states": states})[0], 0.0)
        self.assertEqual(reorder_arm_positions({"states": states})[-1], 0.13)

    def test_missing_state_rejected(self):
        with self.assertRaises(SafetyError):
            reorder_arm_positions({"states": []})

    def test_nan_and_joint_limit_rejected(self):
        with self.assertRaises(SafetyError):
            validate_arm_target([math.nan] * 14)
        target = [0.0] * 14
        target[3] = 1.5
        with self.assertRaises(SafetyError):
            validate_arm_target(target)

    def test_velocity_limit_rejected(self):
        start = [0.0] * 14
        end = [0.0] * 14
        end[0] = 0.2
        with self.assertRaises(SafetyError):
            validate_segment(start, end, 1.0, 0.15)
        validate_segment(start, end, 2.0, 0.15)

    def test_gripper_clamp(self):
        self.assertEqual(clamp_omnipicker(1.0), 0.0)
        self.assertEqual(clamp_omnipicker(-2.0), OMNIPICKER_MIN)


if __name__ == "__main__":
    unittest.main()
