"""Portable gripper evidence; failure requests are handled by the motion owner."""
import threading
import time
from collections import deque

try:
    from .vlm_flow_checks import _snapshots, crop_jpeg, run_gripper_check
    from .safety import SafetyError
except ImportError:
    from vlm_flow_checks import _snapshots, crop_jpeg, run_gripper_check
    from safety import SafetyError


class VlmRegraspRequested(SafetyError):
    """Unwind current automatic motion before restarting left pickup."""


class GripperMonitor:
    def __init__(self, web, client, emit, crop=None):
        self.web, self.client, self.emit, self.crop = web, client, emit, crop
        self.reference = None
        self.votes = deque(maxlen=5)
        self.phase = 'idle'
        self.phase_started = time.monotonic()
        self.epoch = 0
        self.latest = None
        self.regrasp_reason = None
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread = None

    def set_phase(self, phase):
        with self.lock:
            if self.phase == phase:
                return
            self.phase = phase
            self.phase_started = time.monotonic()
            self.epoch += 1
            self.reference = None
            self.votes.clear()
            self.latest = None
            self.regrasp_reason = None

    def raise_if_regrasp_requested(self):
        with self.lock:
            reason = self.regrasp_reason
        if reason is not None:
            raise VlmRegraspRequested(reason)

    def finish_before_release(self):
        with self.lock:
            self.raise_if_regrasp_requested()
            self.set_phase('idle')

    def sample(self):
        with self.lock:
            phase, epoch, reference = self.phase, self.epoch, self.reference
            phase_started = self.phase_started
        if phase == 'idle':
            return None
        frames_override = None
        if phase == 'post_grasp':
            try:
                frames_override = _snapshots(self.web, ('left',))
                age = frames_override[1].get('left', {}).get('raw_age_s')
                read_started = frames_override[1].get('left', {}).get('frame_read_started_s')
                if read_started is not None and read_started < phase_started:
                    return None
                if age is not None and time.monotonic() - age < phase_started:
                    # Never use the still-fresh PRE-grasp cached frame as the
                    # first POST-grasp reference; wait for a new publication.
                    return None
            except Exception as exc:
                record = {'event': 'vlm_check_warning', 'task': 'gripper', 'ok': False, 'error': str(exc)}
                with self.lock:
                    if self.epoch != epoch or self.stop_event.is_set():
                        return None
                    self.votes.append('unknown')
                self.emit(record)
                return record
            with self.lock:
                if self.epoch != epoch:
                    return None
                self.reference = None
        record = run_gripper_check(self.web, self.client, self.crop,
                                   frames_override=frames_override)
        with self.lock:
            if epoch != self.epoch or self.stop_event.is_set():
                return None
            state = record.get('result', {}).get('state', 'unknown') if record['ok'] else 'unknown'
            self.votes.append(state)
            tail = list(self.votes)[-3:]
            stable = tail[-1] if len(tail) == 3 and len(set(tail)) == 1 else 'unknown'
            record.update(workflow_phase=phase, stable_state=stable, votes=list(self.votes))
            self.latest = record
            failure_tail = list(self.votes)[-2:]
            if phase == 'post_grasp' and len(failure_tail) == 2 and all(v in {'not_attached', 'dropped'} for v in failure_tail):
                record['stable_state'] = 'failed'
                self.regrasp_reason = 'VLM 连续两次确认夹持失败'
        self.emit(record)
        return record

    def start(self):
        def run():
            while not self.stop_event.is_set():
                record = self.sample()
                # No extra delay after a real HTTP request. Back off only when
                # idle/no new evidence or unavailable, avoiding a busy loop.
                if record is None or not record.get('ok'):
                    self.stop_event.wait(.1)
        self.thread = threading.Thread(target=run, name='vlm-gripper', daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.set_phase('idle')
        if self.thread is not None:
            self.thread.join(timeout=.2)
        # In-flight HTTP may finish later; it owns no camera/GDK objects and
        # epoch/stop checks suppress publishing its obsolete response.
