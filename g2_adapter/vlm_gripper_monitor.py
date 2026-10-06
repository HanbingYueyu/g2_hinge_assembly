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
        self.confirmed = threading.Event()
        self.stop_event = threading.Event()
        self.thread = None

    def set_phase(self, phase):
        with self.lock:
            if self.phase == phase:
                return
            self.phase = phase
            self.phase_started = time.monotonic()
            self.epoch += 1
            self.confirmed.clear()
            self.reference = None
            self.votes.clear()
            self.latest = None
            self.regrasp_reason = None

    def raise_if_regrasp_requested(self):
        with self.lock:
            reason = self.regrasp_reason
        if reason is not None:
            raise VlmRegraspRequested(reason)

    def wait_for_attached(self, *, timeout_s, guard=None):
        deadline = time.monotonic() + timeout_s
        while True:
            if guard is not None:
                guard()
            with self.lock:
                self.raise_if_regrasp_requested()
                if self.phase != 'post_grasp' or self.stop_event.is_set():
                    raise SafetyError('铰链抓取监测未处于本次抓取阶段；拒绝插入')
                if self.confirmed.is_set() and self.latest and self.latest.get('stable_state') == 'attached':
                    return self.latest
            if time.monotonic() >= deadline:
                raise SafetyError('VLM 铰链抓取确认超时或不确定；保持夹持，拒绝插入')
            time.sleep(.02)

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
            stable = 'unknown'
            if len(self.votes) == 5:
                for candidate in ('attached', 'not_attached'):
                    if self.votes.count(candidate) >= 3:
                        stable = candidate
            record.update(workflow_phase=phase, stable_state=stable, votes=list(self.votes))
            self.latest = record
            if phase == 'post_grasp' and stable == 'not_attached':
                self.regrasp_reason = 'VLM 最近五次多数票确认未抓住（至少三票）'
        self.emit(record)
        with self.lock:
            if self.epoch == epoch and phase == 'post_grasp':
                if stable == 'attached':
                    self.confirmed.set()
                else:
                    self.confirmed.clear()
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
