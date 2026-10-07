"""File-only timing diagnostics for hinge return, without changing motion."""
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import threading
import time

LOG_PATH = Path(__file__).resolve().parents[1] / 'logs' / 'hinge_return_timing.jsonl'
_LOCK = threading.Lock()


@contextmanager
def timed_return_step(label, *, enabled=True):
    started = time.monotonic()
    try:
        yield
    finally:
        elapsed = time.monotonic() - started
        if enabled and elapsed > .5:
            record = {'time': datetime.now().astimezone().isoformat(timespec='milliseconds'),
                      'step': label, 'elapsed_s': round(elapsed, 3)}
            try:
                with _LOCK:
                    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                    with LOG_PATH.open('a', encoding='utf-8') as output:
                        output.write(json.dumps(record, ensure_ascii=False) + '\n')
            except OSError:
                # Optional diagnostics must not interrupt motion or print.
                pass
