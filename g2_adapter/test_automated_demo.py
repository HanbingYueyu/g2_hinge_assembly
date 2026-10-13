import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import repeated_grasp_demo as demo
from right_teach_placement import run_right_teach_sequence
from dianpi_runner import DianpiCycleRunner, DianpiTriggerResult


class AutomatedDemoTests(unittest.TestCase):
    def test_single_modes_return_to_idle(self):
        for mode in ('5','6'):
            selection=demo.DemoModeSelection()
            selection.choose(mode)
            self.assertEqual(selection.wait_first(), mode)
            self.assertIsNone(selection.next_round())
            self.assertIsNone(selection.active)

    def test_right_four_points_are_automatic_and_each_has_callback(self):
        args=SimpleNamespace(right_teach_trajectories=[Path(str(i)) for i in range(4)],right_teach_trajectory=None)
        runtime=Mock()
        runtime.automatic_demo_mode='5'
        with patch('right_teach_placement.replay_right_placement',return_value={'sent':3}):
            run_right_teach_sequence(object(),args,runtime)
        runtime.wait_for_advance.assert_not_called()
        self.assertEqual(runtime.on_right_teach_point.call_count,4)

    def test_mode5_waits_three_seconds_without_trigger(self):
        now=[0.]
        runtime=Mock()
        runtime.clock=lambda:now[0]
        runtime.sleep=lambda dt:now.__setitem__(0,now[0]+dt)
        runner=Mock()
        proxy=demo.AutomaticDemoRuntime(runtime, '5', runner)
        proxy.on_right_teach_point(1, {})
        runner.trigger.assert_not_called()
        runtime.lift_after_screw.assert_not_called()
        self.assertGreaterEqual(now[0],3.)

    def test_mode6_triggers_once_then_checks_completion(self):
        now=[0.]
        runtime=Mock()
        runtime.clock=lambda:now[0]
        runtime.sleep=lambda dt:now.__setitem__(0,now[0]+dt)
        runner=Mock(running=False,last_error=None)
        started=[]
        def trigger(**kwargs):
            started.append(now[0])
            return DianpiTriggerResult.STARTED
        runner.trigger.side_effect=trigger
        proxy=demo.AutomaticDemoRuntime(runtime,'6',runner)
        proxy.on_right_teach_point(1,{})
        runner.trigger.assert_called_once_with(automatic=True)
        runtime.lift_after_screw.assert_called_once_with(distance_mm=10.0)
        self.assertGreaterEqual(started[0],1.)
        self.assertLess(now[0],1.05)

    def test_trigger_policy_blocks_manual_and_p10(self):
        runner=DianpiCycleRunner(event_sink=lambda _:None)
        runner.set_trigger_policy(enabled=False,automatic_only=True)
        self.assertEqual(runner.trigger(),DianpiTriggerResult.DISABLED)
        self.assertEqual(runner.trigger(automatic=True),DianpiTriggerResult.DISABLED)
        runner.close()

class FullFlowScopeTests(unittest.TestCase):
    def test_full_flow_enables_teach_and_restores_context(self):
        from run_bilateral_flow_gdk import FlowContext, StageSettings, teach_enabled
        original=SimpleNamespace(teach_enable=True,repeat_grasp_demo=True,right_teach_trajectories=[1,2,3,4])
        runner=Mock()
        runner.set_trigger_policy.return_value=(True,False)
        services=SimpleNamespace(dianpi_runner=runner)
        runtime=Mock()
        ctx=FlowContext(gdk=None,services=services,runtime=runtime,settings=StageSettings(original))
        def drive(context,args):
            self.assertTrue(teach_enabled(args))
            self.assertEqual(args.from_stage,'home')
            self.assertEqual(args.through_stage,'finish')
            self.assertIsInstance(context.runtime,demo.AutomaticDemoRuntime)
            return 0
        with patch('run_bilateral_flow_gdk._drive',side_effect=drive), patch('left_teach_placement.validate_teach_args'), patch('right_teach_placement.validate_right_teach_args'):
            demo._run_full_flow_demo(ctx,'5')
        self.assertIs(ctx.runtime,runtime)
        self.assertIs(ctx.args,original)
        self.assertEqual(runner.set_trigger_policy.call_count,2)

class ScrewCompletionTests(unittest.TestCase):
    def runtime(self):
        now=[0.]
        runtime=Mock()
        runtime.clock=lambda:now[0]
        runtime.sleep=lambda dt:now.__setitem__(0,now[0]+dt)
        return runtime,now

    def test_completion_error_cannot_be_mistaken_for_success(self):
        from safety import SafetyError
        runtime,now=self.runtime()
        class Runner:
            last_error=None
            def trigger(self, **kwargs): return DianpiTriggerResult.STARTED
            @property
            def running(self):
                self.last_error=RuntimeError('cycle failed')
                return False
        with self.assertRaises(SafetyError):
            demo.AutomaticDemoRuntime(runtime,'6',Runner()).on_right_teach_point(1,{})

    def test_mode6_waits_until_cycle_finishes(self):
        runtime,now=self.runtime()
        class Runner:
            last_error=None
            def trigger(self,**kwargs):
                self.started=now[0]
                return DianpiTriggerResult.STARTED
            @property
            def running(self): return now[0] < self.started+5.
        demo.AutomaticDemoRuntime(runtime,'6',Runner()).on_right_teach_point(1,{})
        self.assertGreaterEqual(now[0],6.)
        self.assertLess(now[0],6.05)

    def test_interrupted_point_never_triggers_screwdriver(self):
        runtime=Mock(automatic_demo_mode='6')
        args=SimpleNamespace(right_teach_trajectories=[Path(str(i)) for i in range(4)],right_teach_trajectory=None)
        with patch('right_teach_placement.replay_right_placement',return_value={'interrupted_by_takeover':True}):
            with self.assertRaises(RuntimeError): run_right_teach_sequence(object(),args,runtime)
        runtime.on_right_teach_point.assert_not_called()

class PointTakeoverTests(unittest.TestCase):
    def test_takeover_during_delay_blocks_screwdriver_start(self):
        from safety import SafetyError
        from spacemouse_takeover import MotionPauseGate
        runtime=Mock()
        now=[0.]
        runtime.clock=lambda:now[0]
        runtime.gate=MotionPauseGate()
        def sleep(dt):
            now[0]+=dt
            runtime.gate.pause()
        runtime.sleep=sleep
        runner=Mock()
        with self.assertRaises(SafetyError):
            demo.AutomaticDemoRuntime(runtime,'6',runner).on_right_teach_point(1,{})
        runner.trigger.assert_not_called()
