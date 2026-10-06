"""Offline constrained cubic B-spline fitting; no robot dependencies."""
import numpy as np
try:
    from .cartesian_teach import CartesianSample, CartesianTrajectory
except ImportError:
    from cartesian_teach import CartesianSample, CartesianTrajectory


def spline_basis(u, count):
    if count < 4:
        raise ValueError('三次 B 样条至少需要 4 个控制点')
    u=np.asarray(u,dtype=float)
    knots=np.r_[np.zeros(4),np.linspace(0.,1.,count-2)[1:-1],np.ones(4)]
    b=((u[:,None]>=knots[:-1]) & (u[:,None]<knots[1:])).astype(float)
    for degree in range(1,4):
        next_b=np.zeros((len(u),len(knots)-degree-1))
        for i in range(next_b.shape[1]):
            left=knots[i+degree]-knots[i]
            right=knots[i+degree+1]-knots[i+1]
            if left:next_b[:,i]+=(u-knots[i])/left*b[:,i]
            if right:next_b[:,i]+=(knots[i+degree+1]-u)/right*b[:,i+1]
        b=next_b
    b[u==1.]=0.
    b[u==1.,-1]=1.
    return b


def _fit(b, values, strength):
    count=b.shape[1]
    difference=np.diff(np.eye(count),n=2,axis=0)
    matrix=b.T@b/len(b)+strength*(difference.T@difference)
    rhs=b.T@values/len(b)
    controls=np.empty((count,values.shape[1]))
    controls[0],controls[-1]=values[0],values[-1]
    controls[1:-1]=np.linalg.solve(matrix[1:-1,1:-1],rhs[1:-1]-matrix[1:-1,0,None]*controls[0]-matrix[1:-1,-1,None]*controls[-1])
    return controls


def _bounded_controls(controls, positions):
    lo,hi=positions.min(axis=0),positions.max(axis=0)
    start,end=positions[0],positions[-1]
    direction=end-start
    norm=float(direction@direction)
    if norm < 1e-12:
        raise ValueError('起终点过近，无法定义端点防过冲平面')
    # Alternating projection onto two convex sets: the recording's XYZ box
    # and the slab between planes perpendicular to the endpoint direction.
    for _ in range(200):
        controls[1:-1]=np.clip(controls[1:-1],lo,hi)
        progress=(controls[1:-1]-start)@direction/norm
        controls[1:-1]+=(np.clip(progress,0.,1.)-progress)[:,None]*direction
    if np.any(controls<lo-1e-8) or np.any(controls>hi+1e-8):
        raise ValueError('控制点约束未收敛，拒绝输出曲线')
    return controls


def smooth_bspline(trajectory, *, control_points=6, strength=.1, rate_hz=100., duration_s=None):
    samples=trajectory.samples
    if len(samples)<4 or not isinstance(control_points,int) or not 4<=control_points<=32:
        raise ValueError('至少需要 4 个样本；控制点数须为 4–32')
    if not np.isfinite(strength) or strength<=0 or not np.isfinite(rate_hz) or rate_hz<=0:
        raise ValueError('平滑强度和采样率必须为正有限数')
    times=np.array([s.t_ns for s in samples],dtype=float)/1e9
    if np.any(np.diff(times)<=0):raise ValueError('采样时间必须严格递增')
    u=(times-times[0])/(times[-1]-times[0])
    positions=np.array([s.position_m for s in samples])
    orientations=np.array([s.orientation_xyzw for s in samples])
    orientations[orientations@orientations[0]<0]*=-1
    references=np.array([s.reference_joint_positions_rad for s in samples])
    b=spline_basis(u,control_points)
    controls=_bounded_controls(_fit(b,positions,strength),positions)
    rotations=_fit(b,orientations,strength)
    joints=_fit(b,references,strength)
    duration=times[-1]-times[0] if duration_s is None else duration_s
    if not np.isfinite(duration) or duration<=0:raise ValueError('轨迹时长必须为正有限数')
    ticks=max(2,int(np.ceil(duration*rate_hz)))
    t=np.linspace(0.,duration,ticks+1)
    fraction=t/duration
    # Quintic time law: analytic start/end velocity and acceleration are zero.
    progress=10*fraction**3-15*fraction**4+6*fraction**5
    progress=np.clip(progress,0.,1.)
    basis=spline_basis(progress,control_points)
    path=basis@controls
    quaternion=basis@rotations
    norms=np.linalg.norm(quaternion,axis=1)
    if norms.min()<.2:raise ValueError('姿态拟合接近奇异，拒绝输出曲线')
    quaternion/=norms[:,None]
    reference=basis@joints
    result=[CartesianSample(int(round(ti*1e9)),tuple(p),tuple(q),tuple(j)) for ti,p,q,j in zip(t,path,quaternion,reference)]
    # Preserve original endpoint tuples exactly, including quaternion sign.
    for i,original in ((0,samples[0]),(-1,samples[-1])):
        result[i]=CartesianSample(result[i].t_ns,original.position_m,original.orientation_xyzw,original.reference_joint_positions_rad)
    report={'method':'constrained_cubic_bspline','control_points':control_points,'strength':strength,
            'duration_s':duration,'rate_hz':rate_hz,'control_positions_m':controls.tolist(),
            'endpoint_positions_preserved':True,'endpoint_orientations_preserved':True,
            'overshoot_constraints':'XYZ recording box and endpoint progress slab',
            'time_law':'quintic_zero_endpoint_velocity_and_acceleration'}
    return CartesianTrajectory(trajectory.side,trajectory.base_frame,trajectory.tip_frame,trajectory.joint_names,trajectory.fixed_joint_positions,tuple(result)),report
