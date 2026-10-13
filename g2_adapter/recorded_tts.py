"""Play recorded workflow phrases through the development machine speakers."""
from pathlib import Path
import os
import shutil
import subprocess
import wave

AUDIO_DIR = Path(__file__).resolve().parents[1] / 'models/tts/g2_denoised_flow_21_20261006'

class RecordedAudioPlayer:
    """Expose play_tts for the existing asynchronous announcement queue."""
    def __init__(self, mode='local', *, audio_dir=AUDIO_DIR):
        if mode != 'local':
            raise ValueError('RecordedAudioPlayer only supports local audio')
        root=Path(audio_dir).resolve()
        self.files={}
        for line in (root/'台词清单.txt').read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            text,filename=line.split(' — ',1)
            phrase=text.split('. ',1)[1]
            path=(root/filename.strip()).resolve()
            if path.parent!=root or not path.is_file():
                raise ValueError(f'台词音频文件不存在或路径无效：{filename}')
            self.files[phrase]=path
        if shutil.which('aplay') is None:
            raise RuntimeError('本机未安装 aplay，请安装 alsa-utils')

    def play_tts(self, message):
        if message not in self.files:
            raise ValueError(f'台词清单中没有对应音频：{message}')
        path=self.files[message]
        with wave.open(str(path),'rb') as wav:
            duration=wav.getnframes()/wav.getframerate()
        # Blocking here serializes audio in the existing background speech worker.
        environment = os.environ.copy()
        # GDK ships an ALSA build configured for the robot filesystem. System
        # aplay must use host libraries; preserve desktop audio/session variables.
        environment.pop('LD_LIBRARY_PATH', None)
        environment.pop('LD_PRELOAD', None)
        try:
            subprocess.run(['aplay','-q',str(path)],check=True,env=environment,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,
                           timeout=duration+5.)
        except subprocess.CalledProcessError as exc:
            reason = (exc.stderr or b'').decode('utf-8', errors='replace').strip()
            raise RuntimeError(f'本机音频播放失败：{reason or str(exc)}') from exc
