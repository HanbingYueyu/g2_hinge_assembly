import unittest
from test_vlm_tts import FakeInteraction,load_module

class ForcedSpeechTests(unittest.TestCase):
    def test_closed_jaw_announces_success_without_vlm(self):
        interaction=FakeInteraction();a=load_module().VlmTtsAnnouncer(interaction)
        try:
            self.assertTrue(a.handle_event({'event':'right_gripper_closed'}))
            self.assertFalse(a.handle_event({'event':'vlm_right_fixture_grasp_result','ok':True,'result':{'decision':'未抓住'}}))
        finally:a.close()
        self.assertEqual(interaction.messages,['电批抓取成功'])
