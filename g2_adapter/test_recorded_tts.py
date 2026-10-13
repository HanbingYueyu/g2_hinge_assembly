import unittest
from unittest.mock import Mock,patch
from recorded_tts import RecordedAudioPlayer

class AudioTests(unittest.TestCase):
    def test_all_21_phrases_have_existing_files(self):
        p=RecordedAudioPlayer('local')
        self.assertEqual(len(p.files),21)
        self.assertTrue(all(f.is_file() for f in p.files.values()))
    def test_local_uses_audio_file_without_gdk(self):
        p=RecordedAudioPlayer('local')
        with patch('recorded_tts.subprocess.run') as run:
            p.play_tts('开始打螺丝')
        self.assertEqual(run.call_args.args[0][0:2],['aplay','-q'])
        self.assertTrue(run.call_args.args[0][-1].endswith('13_drive_screws.wav'))
    def test_unknown_phrase_does_not_invent_audio(self):
        p=RecordedAudioPlayer('local')
        with self.assertRaisesRegex(ValueError,'台词'):
            p.play_tts('不存在的台词')
    def test_local_player_does_not_inherit_gdk_libraries(self):
        import os
        p=RecordedAudioPlayer('local')
        with patch.dict(os.environ,{'LD_LIBRARY_PATH':'/gdk/lib','LD_PRELOAD':'gdk.so'}), patch('recorded_tts.subprocess.run') as run:
            p.play_tts('开始打螺丝')
        self.assertNotIn('LD_LIBRARY_PATH',run.call_args.kwargs['env'])
        self.assertNotIn('LD_PRELOAD',run.call_args.kwargs['env'])
