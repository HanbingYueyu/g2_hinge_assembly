import unittest
from dianpi_runner import DianpiCycleRunner,DianpiTriggerResult
from test_dianpi_runner import FakeController

class GpioPolicyTests(unittest.TestCase):
    def test_gpio_enabled_without_enabling_other_sources(self):
        controller=FakeController(block=True)
        runner=DianpiCycleRunner(controller_factory=lambda:controller,event_sink=lambda e:None)
        try:
            runner.set_trigger_policy(enabled=False,automatic_only=True)
            runner.set_gpio_trigger_override(True)
            self.assertEqual(runner.trigger(),DianpiTriggerResult.DISABLED)
            self.assertEqual(runner.trigger(automatic=True),DianpiTriggerResult.DISABLED)
            self.assertEqual(runner.trigger(source='gpio'),DianpiTriggerResult.STARTED)
            self.assertEqual(runner.trigger(source='gpio'),DianpiTriggerResult.BUSY)
        finally:
            controller.release.set();runner.close()
    def test_override_can_be_restored(self):
        runner=DianpiCycleRunner(event_sink=lambda e:None)
        runner.set_trigger_policy(enabled=False,automatic_only=True)
        previous=runner.set_gpio_trigger_override(True)
        self.assertIsNone(previous)
        runner.set_gpio_trigger_override(previous)
        self.assertEqual(runner.trigger(source='gpio'),DianpiTriggerResult.DISABLED)
        runner.close()
