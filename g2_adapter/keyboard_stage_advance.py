"""Read terminal Enter/Space without writing to the robot from this thread."""

import os
import select
import sys
import termios
import threading
import tty


class KeyboardStageAdvance:
    def __init__(self, worker, stream=None):
        self.worker = worker
        self.stream = sys.stdin if stream is None else stream
        self._stop = threading.Event()
        self._thread = None
        self._fd = None
        self._saved_terminal = None
        self._cbreak = False

    def start(self):
        try:
            self._fd = self.stream.fileno()
        except (AttributeError, OSError, ValueError):
            return
        # A real terminal is normally in canonical mode, which would delay a
        # lone space until Enter.  cbreak keeps Ctrl-C working while making
        # Enter is an immediate stage confirmation and Space is an immediate
        # electric-screwdriver trigger.
        try:
            self._saved_terminal = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._cbreak = True
        except (termios.error, OSError, ValueError):
            self._saved_terminal = None
            self._cbreak = False
        self._thread = threading.Thread(target=self._run, name='g2-stage-enter', daemon=True)
        self._thread.start()

    def _run(self):
        pending = b''
        try:
            while not self._stop.is_set():
                try:
                    readable, _, _ = select.select([self._fd], [], [], .1)
                    if not readable:
                        continue
                    data = os.read(self._fd, 1024)
                    if not data:
                        return
                    if self._cbreak:
                        for byte in data:
                            if byte == 0x20:
                                self.worker.request_keyboard_dianpi()
                            elif byte in (0x0A, 0x0D):
                                self.worker.request_keyboard_advance()
                        continue
                    pending += data
                    while b'\n' in pending:
                        line, pending = pending.split(b'\n', 1)
                        if line.strip() == b'' and b' ' in line:
                            self.worker.request_keyboard_dianpi()
                        elif not line.strip():
                            self.worker.request_keyboard_advance()
                except (OSError, ValueError):
                    return
        finally:
            self._restore_terminal()

    def _restore_terminal(self):
        if self._saved_terminal is not None and self._fd is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_terminal)
            except (termios.error, OSError, TypeError, ValueError):
                pass
        self._saved_terminal = None
        self._cbreak = False

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
        self._restore_terminal()
