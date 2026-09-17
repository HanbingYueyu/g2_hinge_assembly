import unittest

from bilateral_stages import DEFAULT_PLAN, Stage, StageSequencer, stage_names


class BilateralStagePlanTests(unittest.TestCase):
    def test_plan_covers_the_nine_operator_steps_in_order(self):
        steps = [spec.user_step for spec in DEFAULT_PLAN]
        self.assertEqual(steps, ["1", "2", "3", "3", "4", "5", "5", "5", "5", "6-7", "8", "8", "9"])
        self.assertEqual(stage_names()[0], "home")
        self.assertEqual(stage_names()[-1], "finish")

    def test_only_the_two_gates_wait_for_a_spacemouse_advance(self):
        waiting = [spec.stage for spec in DEFAULT_PLAN if spec.waits_for_advance]
        self.assertEqual(waiting, [Stage.GATE_LEFT, Stage.GATE_RIGHT])

    def test_ll1_and_force_descent_precede_gate_then_release_lift_and_ll2(self):
        order = stage_names()
        self.assertLess(order.index("right_to_lr1"), order.index("gate_left"))
        self.assertLess(order.index("right_to_lr1"), order.index("left_to_ll1_and_seat"))
        self.assertLess(order.index("left_to_ll1_and_seat"), order.index("gate_left"))
        self.assertLess(order.index("gate_left"), order.index("left_release_lift_to_ll2"))
        self.assertLess(order.index("right_grasp"), order.index("gate_right"))
        self.assertLess(order.index("gate_right"), order.index("right_return"))

    def test_vision_stages_name_their_camera_pane_and_gates_do_not(self):
        expected = {
            Stage.HEAD_LOCALIZE: "head",
            Stage.LEFT_GRASP: "left",
            Stage.LEFT_TO_LL1_AND_SEAT: None,
            Stage.LEFT_RELEASE_LIFT_TO_LL2: None,
            Stage.RIGHT_TO_LR1: "right",
            Stage.RIGHT_GRASP: "right",
            Stage.RIGHT_RETURN: "right",
        }
        for spec in DEFAULT_PLAN:
            if spec.waits_for_advance:
                self.assertIsNone(spec.camera_pane, spec.stage)
        for spec in DEFAULT_PLAN:
            if spec.stage in expected:
                self.assertEqual(spec.camera_pane, expected[spec.stage], spec.stage)


class BilateralSequencerTests(unittest.TestCase):
    def test_sequence_walks_every_stage_once(self):
        sequencer = StageSequencer()
        seen = []
        while not sequencer.is_done:
            seen.append(sequencer.current.stage)
            sequencer.complete_current()
        self.assertEqual(seen, [spec.stage for spec in DEFAULT_PLAN])
        self.assertTrue(sequencer.summary()["finished"])
        self.assertEqual(len(sequencer.summary()["completed"]), len(DEFAULT_PLAN))

    def test_complete_current_returns_the_next_stage_and_none_at_the_end(self):
        sequencer = StageSequencer()
        self.assertEqual(sequencer.complete_current(), DEFAULT_PLAN[1])
        self.assertIsNone(sequencer.current if sequencer.is_done else None)

    def test_skip_to_starts_at_the_requested_stage(self):
        sequencer = StageSequencer(start_at="gate_right")
        self.assertEqual(sequencer.current.stage, Stage.GATE_RIGHT)
        self.assertEqual(len(sequencer.remaining), 3)
        with self.assertRaises(ValueError):
            StageSequencer(start_at="not_a_stage")

    def test_failure_records_the_stage_and_stops_the_summary(self):
        sequencer = StageSequencer()
        sequencer.complete_current()
        sequencer.fail_current("头部没有检测到 hole")
        summary = sequencer.summary()
        self.assertFalse(summary["finished"])
        self.assertEqual(summary["failure"]["stage"], "head_localize")
        self.assertIn("hole", summary["failure"]["error"])

    def test_payload_is_merged_into_the_completed_stage_record(self):
        sequencer = StageSequencer()
        sequencer.complete_current({"frames": 15, "targets": 2})
        self.assertEqual(sequencer.results[0]["frames"], 15)
        self.assertEqual(sequencer.results[0]["user_step"], "1")

    def test_empty_plan_is_rejected(self):
        with self.assertRaises(ValueError):
            StageSequencer(plan=())


if __name__ == "__main__":
    unittest.main()
