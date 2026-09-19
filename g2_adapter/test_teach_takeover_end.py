import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from left_teach_placement import replay_left_placement
from right_teach_placement import replay_right_placement
from spacemouse_takeover import MotionPauseGate, TeachReplayTakenOver
from teach_trajectory import build_trajectory, trajectory_document
from test_teach_trajectory_gdk import FakeGdk, FakeRobot


class TeachTakeoverTests(unittest.TestCase):
    def test_takeover_during_last_feedback_read_is_not_reported_as_complete(self):
        gate = MotionPauseGate()
        with self.assertRaises(TeachReplayTakenOver):
            with gate.automatic_teach_motion(gate.pause_generation):
                gate.pause()
                gate.resume()

    def test_takeover_ends_replay_without_resume_or_endpoint_wait_for_both_arms(self):
        for side, replay in (("left", replay_left_placement), ("right", replay_right_placement)):
            with self.subTest(side=side), tempfile.TemporaryDirectory() as directory:
                gate = MotionPauseGate()
                class Robot(FakeRobot):
                    def move_arm_joint_servo(self, command, *args):
                        self.servo_commands.append(command)
                        offset = 0 if side == "left" else 7
                        self.positions[offset:offset+7] = command
                        if len(self.servo_commands) > 1:
                            return
                        # Pause from a separate HID thread after the first target.
                        thread = threading.Thread(target=gate.pause)
                        thread.start()
                        self.pause_thread = thread
                robot = Robot()
                path = Path(directory) / "demo.json"
                path.write_text(json.dumps(trajectory_document(build_trajectory(side, [(0, [0]*7), (100_000_000, [.1]*7)]))))
                args = SimpleNamespace(**{f"{side}_teach_trajectory": path, f"{side}_teach_speed_scale": .5, f"{side}_teach_start_velocity_rad_s": .05})
                outcome = []
                def run():
                    try:
                        outcome.append(replay(SimpleNamespace(robot=robot, gdk=FakeGdk(), gate=gate), args))
                    except Exception as exc:
                        outcome.append(exc)
                with patch("sys.stdout"):
                    worker = threading.Thread(target=run, daemon=True)
                    worker.start()
                    worker.join(1)
                    completed_without_resume = not worker.is_alive()
                    gate.resume()  # Always clean up a failing implementation.
                    worker.join(2)
                self.assertTrue(completed_without_resume, "示教仍在等待接管退出")
                self.assertEqual(len(robot.servo_commands), 1)
                self.assertTrue(outcome[0]["interrupted_by_takeover"])
                self.assertEqual(robot.control_mode, 3)
                self.assertEqual(
                    tuple(item.kp for item in robot.impedance_calls[-1][0]),
                    (400, 400, 400, 500, 300, 200, 200),
                )


if __name__ == "__main__":
    unittest.main()
