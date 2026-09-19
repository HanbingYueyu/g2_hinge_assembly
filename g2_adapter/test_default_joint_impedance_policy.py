"""Guard the formal motion entry points against position-mode fallbacks."""

import ast
from pathlib import Path
import unittest


# Explicit position recovery and the Cartesian mode-2 demo are intentionally
# absent. Add new formal motion entry points here when introducing them.
FORMAL_MOTION_FILES = (
    "control.py",
    "teach_trajectory_gdk.py",
    "left_teach_placement.py",
    "right_teach_placement.py",
    "run_spacemouse_gdk.py",
    "spacemouse_teleop.py",
    "spacemouse_takeover.py",
    "run_bilateral_flow_gdk.py",
    "bilateral_runtime.py",
    "bilateral_stages.py",
    "move_left_above_head_hinge_gdk.py",
    "right_apriltag_grasp_gdk.py",
    "right_hole_target_detect_gdk.py",
    "set_arm_orientation_gdk.py",
    "set_omnipicker.py",
    "left_seat_motion.py",
    "run_preset.py",
    "run_left_premove_sequence.py",
    "run_left_post_grasp_sequence.py",
    "left_grasp_gdk.py",
    "demo_joint_impedance_move.py",
)


def _is_control_mode(node) -> bool:
    """Return True when ``node`` reads a ``control_mode`` status field."""
    if isinstance(node, ast.Attribute) and node.attr == "control_mode":
        return True
    if isinstance(node, ast.Subscript):
        index = node.slice
        if isinstance(index, ast.Constant) and index.value == "control_mode":
            return True
    if (
        isinstance(node, ast.Call)
        and len(node.args) == 1
        and isinstance(node.func, ast.Name)
        and node.func.id in {"int", "float"}
    ):
        return _is_control_mode(node.args[0])
    return False


def _constant_int(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _mentions_position_mode_comparison(node: ast.Compare) -> bool:
    operands = [node.left, *node.comparators]
    for index, operator in enumerate(node.ops):
        left, right = operands[index], operands[index + 1]
        if isinstance(operator, (ast.Eq, ast.NotEq)):
            if _is_control_mode(left) and _constant_int(right) == 1:
                return True
            if _is_control_mode(right) and _constant_int(left) == 1:
                return True
        if isinstance(operator, (ast.In, ast.NotIn)) and _is_control_mode(left):
            if not isinstance(right, (ast.Set, ast.Tuple, ast.List)):
                continue
            values = [_constant_int(item) for item in right.elts]
            if 1 in values and 3 not in values:
                return True
    return False


def violations_in_source(source: str, filename: str = "<source>") -> list[str]:
    tree = ast.parse(source, filename=filename)
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "set_control_mode":
                arguments = [*node.args, *(item.value for item in node.keywords)]
                if any(_constant_int(value) == 1 for value in arguments):
                    violations.append(f"{filename}:{node.lineno}")
        elif isinstance(node, ast.Compare) and _mentions_position_mode_comparison(node):
            violations.append(f"{filename}:{node.lineno}")
    return violations


class DefaultJointImpedancePolicyTests(unittest.TestCase):
    def test_formal_motion_has_no_direct_position_mode_fallback(self):
        violations = []
        for filename in FORMAL_MOTION_FILES:
            path = Path(__file__).resolve().parent / filename
            violations.extend(
                violations_in_source(path.read_text(encoding="utf-8"), filename)
            )
        self.assertEqual(violations, [], "Formal motion must prepare joint impedance, not mode 1")

    def test_detector_flags_the_fallback_patterns(self):
        flagged = violations_in_source(
            "robot.set_control_mode(1)\n"
            "robot.set_control_mode(control_mode=1)\n"
            "if status.control_mode == 1:\n    pass\n"
            "if status['control_mode'] != 1:\n    pass\n"
            "if int(status.control_mode) in (1,):\n    pass\n"
        )
        self.assertEqual(len(flagged), 5)

    def test_detector_allows_mode_one_three_diagnostic(self):
        allowed = violations_in_source(
            "if int(status.control_mode) not in {1, 3}:\n    pass\n"
            "if status['control_mode'] != 3:\n    pass\n"
        )
        self.assertEqual(allowed, [])


if __name__ == "__main__":
    unittest.main()
