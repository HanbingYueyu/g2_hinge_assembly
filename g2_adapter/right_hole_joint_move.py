"""Move a grasped screwdriver to an endpoint target with planned right joints."""
import math
import time

import numpy as np

try:
    from .cartesian_teach_trajectory_gdk import DEFAULT_URDF, _joint_map, verify_live_model
    from .cartesian_teach import quaternion_to_matrix
    from .g2_kinematics import UrdfKinematics, rotation_error
    from .safety import RIGHT_ARM_JOINT_NAMES, ARM_POSITION_LIMITS, SafetyError
except ImportError:
    from cartesian_teach_trajectory_gdk import DEFAULT_URDF, _joint_map, verify_live_model
    from cartesian_teach import quaternion_to_matrix
    from g2_kinematics import UrdfKinematics, rotation_error
    from safety import RIGHT_ARM_JOINT_NAMES, ARM_POSITION_LIMITS, SafetyError


def move_right_hole_joint_target(controller, target, *, velocity_rad_s,
                                 timeout_s=30., sleep_fn=time.sleep, clock=time.monotonic, control_mode=1):
    """Plan the endpoint pose as right joints and verify the live endpoint arrived."""
    if not math.isfinite(velocity_rad_s) or velocity_rad_s <= 0:
        raise ValueError('right endpoint joint velocity must be positive and finite')
    if control_mode not in (1, 3):
        raise ValueError('right endpoint joint control_mode must be 1 or 3')
    model = UrdfKinematics(DEFAULT_URDF.resolve())
    verify_live_model(model, controller.robot, 'right')
    state = _joint_map(controller.robot)
    initial = [state[name] for name in RIGHT_ARM_JOINT_NAMES]
    transform = np.eye(4)
    transform[:3, :3] = quaternion_to_matrix(target.orientation_xyzw)
    transform[:3, 3] = target.position_m
    solution, _ = model.solve_pose('base_link', 'arm_r_end_link', state,
        RIGHT_ARM_JOINT_NAMES, transform, initial,
        joint_limits=ARM_POSITION_LIMITS[7:], reference_weight=0.)
    controller.check_automatic_guard()
    # Carrying the tool defaults to position mode; pre-grasp clearance can use mode 3.
    controller.move_preset('right', solution, velocity_rad_s=velocity_rad_s,
        life_time_s=30., detail='right endpoint IK', control_mode=control_mode)
    # A large joint move at half the preset speed can legitimately exceed 30 s.
    expected_travel_s = max(abs(float(goal) - start) for goal, start in zip(solution, initial)) / velocity_rad_s
    deadline = clock() + max(timeout_s, expected_travel_s + 10.)
    stable = 0
    while True:
        controller.check_automatic_guard()
        controller._require_joint_impedance_mode('right endpoint arrival')
        actual = controller.read_end_effector_pose('arm_r_end_link')
        position_error = np.linalg.norm(np.asarray(actual.position_m) - transform[:3, 3])
        angle_error = np.linalg.norm(rotation_error(transform[:3, :3],
            quaternion_to_matrix(actual.orientation_xyzw)))
        stable = stable + 1 if position_error <= .005 and angle_error <= math.radians(2.) else 0
        if stable >= 3:
            return actual
        if clock() >= deadline:
            raise SafetyError(f'右手终点 IK 关节规划未到位：位置误差 {position_error*1000:.1f} mm，姿态误差 {math.degrees(angle_error):.2f}°')
        sleep_fn(.05)
