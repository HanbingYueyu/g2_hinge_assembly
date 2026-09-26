"""Endpoint IK followed by the same planned joint request as a left preset."""
import math
import time
import numpy as np

try:
    from .cartesian_teach_trajectory_gdk import DEFAULT_URDF, _joint_map, verify_live_model
    from .cartesian_teach import quaternion_to_matrix
    from .g2_kinematics import UrdfKinematics, rotation_error
    from .safety import LEFT_ARM_JOINT_NAMES, ARM_POSITION_LIMITS, SafetyError
except ImportError:
    from cartesian_teach_trajectory_gdk import DEFAULT_URDF, _joint_map, verify_live_model
    from cartesian_teach import quaternion_to_matrix
    from g2_kinematics import UrdfKinematics, rotation_error
    from safety import LEFT_ARM_JOINT_NAMES, ARM_POSITION_LIMITS, SafetyError


def move_left_hole_joint_target(controller, target, *, velocity_rad_s,
                                timeout_s=30., sleep_fn=time.sleep, clock=time.monotonic):
    if not math.isfinite(velocity_rad_s) or velocity_rad_s <= 0:
        raise ValueError('hole joint velocity must be positive and finite')
    model = UrdfKinematics(DEFAULT_URDF.resolve())
    verify_live_model(model, controller.robot, 'left')
    state = _joint_map(controller.robot)
    initial = [state[name] for name in LEFT_ARM_JOINT_NAMES]
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(target.orientation_xyzw)
    transform[:3, 3] = target.position_m
    solution, _ = model.solve_pose('base_link', 'arm_l_end_link', state,
        LEFT_ARM_JOINT_NAMES, transform, initial,
        joint_limits=ARM_POSITION_LIMITS[:7], reference_weight=0.)
    controller.check_automatic_guard()
    controller.move_preset('left', solution, velocity_rad_s=velocity_rad_s,
        life_time_s=30., detail='left endpoint IK', control_mode=3)
    deadline = clock() + timeout_s
    stable = 0
    while True:
        controller.check_automatic_guard()
        controller._require_joint_impedance_mode('left endpoint arrival')
        actual = controller.read_end_effector_pose('arm_l_end_link')
        position_error = np.linalg.norm(np.asarray(actual.position_m) - transform[:3, 3])
        angle_error = np.linalg.norm(rotation_error(transform[:3, :3],
            quaternion_to_matrix(actual.orientation_xyzw)))
        stable = stable + 1 if position_error <= .005 and angle_error <= math.radians(2.) else 0
        if stable >= 3:
            return actual
        if clock() >= deadline:
            raise SafetyError(f'左手终点 IK 关节规划未到位：位置误差 {position_error*1000:.1f} mm，姿态误差 {math.degrees(angle_error):.2f}°')
        sleep_fn(.05)
