"""Shared bounded multi-start endpoint IK; does not send motion commands."""
import json
import time
from datetime import datetime
from pathlib import Path
import numpy as np
try:
    from .g2_kinematics import KinematicsError
    from .safety import ARM_POSITION_LIMITS, LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES
except ImportError:
    from g2_kinematics import KinematicsError
    from safety import ARM_POSITION_LIMITS, LEFT_ARM_JOINT_NAMES, RIGHT_ARM_JOINT_NAMES


def log_ik(record):
    try:
        path=Path(__file__).resolve().parents[1]/'logs'/'endpoint_ik_candidates.jsonl'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps({'time':datetime.now().astimezone().isoformat(),**record},ensure_ascii=False)+'\n')
    except OSError:
        pass


def solve_candidates(model,controller,state,transform,initial,side,*,return_all=False):
    if side not in ('left','right'):
        raise ValueError('invalid arm side')
    names=LEFT_ARM_JOINT_NAMES if side=='left' else RIGHT_ARM_JOINT_NAMES
    limits=np.asarray(ARM_POSITION_LIMITS[:7] if side=='left' else ARM_POSITION_LIMITS[7:])
    initial=np.asarray(initial,dtype=float)
    seeds=[initial.copy()]
    for joint in (1,3,5):
        for sign in (-1,1):
            seed=initial.copy()
            seed[joint]+=sign*.35
            seeds.append(np.clip(seed,limits[:,0],limits[:,1]))
    seeds.append(.7*initial+.3*limits.mean(axis=1))
    candidates=[]
    errors=[]
    started=time.monotonic()
    for seed in seeds:
        controller.check_automatic_guard()
        if time.monotonic()-started>12:
            break
        try:
            # Solve the exact pose first; rank feasible solutions by current-joint distance.
            # A pose-penalty reference term can otherwise prevent tight convergence.
            solution,_=model.solve_pose('base_link',f'arm_{"l" if side=="left" else "r"}_end_link',
                state,names,transform,seed,reference=initial,joint_limits=limits,
                reference_weight=0.,adaptive_steps=True,max_iterations=350,
                position_tolerance_m=.001,orientation_tolerance_rad=np.radians(1.))
            solution=np.asarray(solution,dtype=float)
            if solution.shape!=(7,) or not np.all(np.isfinite(solution)) or np.any(solution<limits[:,0]) or np.any(solution>limits[:,1]):
                raise KinematicsError('IK returned invalid joint target')
            if not any(np.max(np.abs(solution-q))<.001 for q in candidates):
                candidates.append(solution.copy())
            # Already at the target: avoid searching remote redundant configurations.
            if np.max(np.abs(solution-initial))<.01:
                break
        except KinematicsError as exc:
            errors.append(str(exc))
    candidates.sort(key=lambda q:(float(np.linalg.norm(q-initial)),float(np.max(np.abs(q-initial)))))
    log_ik({'event':'solve','side':side,'target_transform':transform.tolist(),
            'initial':initial.tolist(),'joint_state':state,'candidates':[q.tolist() for q in candidates],
            'errors':errors,'elapsed_s':time.monotonic()-started})
    if not candidates:
        raise KinematicsError(f'{side} 终点 IK 多初值未收敛；保持不动。最后结果：{errors[-1] if errors else "求解预算耗尽"}')
    return candidates if return_all else candidates[0]


def submit_endpoint(controller,side,solution,*,velocity_rad_s,control_mode):
    try:
        controller.check_automatic_guard()
        controller.move_preset(side,solution,velocity_rad_s=velocity_rad_s,
            life_time_s=30.,detail=f'{side} endpoint IK',control_mode=control_mode)
    except Exception as exc:
        # The installed binding loses the GDK failure classification: never retry blindly.
        log_ik({'event':'joint_request_failed','side':side,'solution':list(map(float,solution)),
                'velocity_rad_s':velocity_rad_s,'error':str(exc),'automatic_retry':False})
        raise


# Native GDK writes the detailed rejection to fd 2, not to the Python exception.
# Serialize captures: fd redirection is process-wide. Persist all captured text.
import os
import tempfile
import threading
from contextlib import contextmanager
_CAPTURE_LOCK = threading.Lock()

@contextmanager
def capture_native_stderr():
    with _CAPTURE_LOCK, tempfile.TemporaryFile(mode='w+b') as captured:
        saved = os.dup(2)
        try:
            os.dup2(captured.fileno(), 2)
            class Reader:
                def read(self):
                    captured.flush()
                    return os.pread(captured.fileno(), os.fstat(captured.fileno()).st_size, 0).decode('utf-8', errors='replace')
            yield Reader()
        finally:
            os.dup2(saved, 2)
            os.close(saved)


def submit_candidates(controller,side,candidates,*,velocity_rad_s,control_mode):
    """Try another IK branch only after an explicit pre-execution rejection."""
    for index,solution in enumerate(candidates):
        controller.check_automatic_guard()
        before=np.asarray(controller.checked_arm_state(),dtype=float)
        started=time.monotonic()
        failure=None
        with capture_native_stderr() as captured:
            try:
                controller.move_preset(side,solution,velocity_rad_s=velocity_rad_s,
                    life_time_s=30.,detail=f'{side} endpoint IK candidate {index+1}',control_mode=control_mode)
            except Exception as exc:
                failure=exc
            native_log=captured.read()
        if failure is None:
            log_ik({'event':'joint_request_accepted','side':side,'candidate_index':index+1,
                    'solution':list(map(float,solution)),'native_log':native_log})
            return solution
        retry=False
        # Exact exception + explicit segment-zero coupled rejection. Neither timeout
        # nor a generic planner message permits retry. Unchanged measured arms and
        # healthy control mode provide additional conservative checks.
        explicit=(type(failure) is RuntimeError and str(failure)=='JointControl failed'
                  and native_log.count('JointControlRequest failed')==1
                  and 'CoupledBoundConstraint[' in native_log
                  and 'Path generator failed at segment 0' in native_log
                  and not any(word in native_log.lower() for word in ('timeout','timed out','communication')))
        if explicit and time.monotonic()-started<2.:
            after=np.asarray(controller.checked_arm_state(),dtype=float)
            status=controller.motion_status_summary()
            retry=(before.shape==(14,) and after.shape==(14,)
                   and np.all(np.isfinite(before)) and np.all(np.isfinite(after))
                   and np.max(np.abs(after-before))<.001
                   and status['error_code']==0 and status['control_mode']==control_mode)
        log_ik({'event':'joint_request_failed','side':side,'candidate_index':index+1,
                'solution':list(map(float,solution)),'velocity_rad_s':velocity_rad_s,
                'error':str(failure),'native_log':native_log,
                'automatic_retry':bool(retry and index+1<len(candidates))})
        if not retry or index+1==len(candidates):
            raise failure
    raise KinematicsError('没有可提交的 IK 候选解')
