"""Distinguish a verification-lift motion timeout from loss of the grasp."""
import time
from datetime import datetime,timezone
import numpy as np
try:
    from .control import quaternion_angle_deg
    from .safety import SafetyError
except ImportError:
    from control import quaternion_angle_deg
    from safety import SafetyError


def lift_record(controller,stream,verify,actual,event,*,now=None):
    now=time.monotonic() if now is None else now
    target=verify['target'];origin=verify['origin']
    delta=np.asarray(actual.position_m)-np.asarray(target.position_m)
    return {'event':event,'utc':datetime.now(timezone.utc).isoformat(),
        'origin_position_m':list(origin.position_m),'target_position_m':list(target.position_m),
        'actual_position_m':list(actual.position_m),
        'target_orientation_xyzw':list(target.orientation_xyzw),'actual_orientation_xyzw':list(actual.orientation_xyzw),
        'position_error_mm':float(np.linalg.norm(delta)*1000.),'xyz_error_mm':(delta*1000.).tolist(),
        'orientation_error_deg':quaternion_angle_deg(actual.orientation_xyzw,target.orientation_xyzw),
        'requested_lift_mm':(target.position_m[2]-origin.position_m[2])*1000.,
        'actual_lift_mm':(actual.position_m[2]-origin.position_m[2])*1000.,
        'elapsed_s':now-verify['started_at'],'timeout_s':verify['timeout_s'],
        'gripper_position_rad':controller.read_gripper_position('left'),
        'motion_status':controller.motion_status_summary(),'stream':stream.pose_snapshot()}


def stop_timed_out_lift(controller,stream,verify,actual,emit,*,now=None):
    record=lift_record(controller,stream,verify,actual,'post_grasp_verify_lift_motion_timeout',now=now)
    record.update({'grasp_failed':False,'action':'stop_motion_preserve_gripper','automatic_regrasp':False})
    emit(record)
    # Replace the movement destination, then stop its sole writer before joint hold.
    # Do not send any new gripper command or open it on a motion timeout.
    try:
        stream.hold_at_pose(actual)
    finally:
        stream.stop()
    controller.hold_joint_positions('left',controller.checked_arm_state()[:7])
    raise SafetyError('抓取后验证上抬运动未到位；保持夹爪闭合，停止后续移动；详情见日志')
