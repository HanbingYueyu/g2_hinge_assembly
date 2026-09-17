"""Read terminal Enter without writing to the robot from this thread."""

import os
import select
import sys
import threading


class KeyboardStageAdvance:
    def __init__(self, worker, stream=None):
        self.worker = worker
        self.stream = sys.stdin if stream is None else stream
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        try:
            self._fd = self.stream.fileno()
        except (AttributeError, OSError, ValueError):
            return
        self._thread = threading.Thread(target=self._run, name='g2-stage-enter', daemon=True)
        self._thread.start()

    def _run(self):
        pending = b''
        while not self._stop.is_set():
            try:
                readable, _, _ = select.select([self._fd], [], [], .1)
                if not readable:
                    continue
                data = os.read(self._fd, 1024)
                if not data:
                    return
                pending += data
                while b'\n' in pending:
                    line, pending = pending.split(b'\n', 1)
                    if not line.strip():
                        self.worker.request_keyboard_advance()
            except (OSError, ValueError):
                return

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
