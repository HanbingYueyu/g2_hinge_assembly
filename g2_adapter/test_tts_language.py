import unittest
from unittest.mock import Mock,patch
from recorded_tts import LanguageSpeechPlayer

class LanguageTests(unittest.TestCase):
    def test_g2_english_and_switch_back(self):
        interaction=Mock();p=LanguageSpeechPlayer('g2',interaction,language='en')
        p.play_tts('开始打螺丝')
        interaction.play_tts.assert_called_with('Now fastening the screws.')
        self.assertEqual(p.toggle_language(),'zh')
        p.play_tts('开始打螺丝')
        interaction.play_tts.assert_called_with('开始打螺丝')
    def test_local_english_uses_matching_file(self):
        p=LanguageSpeechPlayer('local',language='en')
        with patch('recorded_tts.subprocess.run') as run:
            p.play_tts('开始打螺丝')
        self.assertIn('g2_denoised_flow_21_en_20261013/13_drive_screws.wav',run.call_args.args[0][-1])
    def test_all_21_entries_match(self):
        p=LanguageSpeechPlayer('g2',Mock())
        self.assertEqual(len(p.translations),21)
    def test_demo_key_nine_switches_without_changing_mode(self):
        import os,threading
        from repeated_grasp_demo import DemoKeyboard,DemoModeSelection
        read_fd,write_fd=os.pipe()
        selection=DemoModeSelection();selection.active='6'
        switched=threading.Event()
        def switch():
            switched.set()
            return 'en'
        with os.fdopen(read_fd,'r') as stream:
            keyboard=DemoKeyboard(selection,stream,language_switch=switch)
            keyboard.start()
            try:
                os.write(write_fd,b'9\n')
                self.assertTrue(switched.wait(1))
                self.assertEqual(selection.active,'6')
                self.assertIsNone(selection.pending)
            finally:
                keyboard.stop();os.close(write_fd)
