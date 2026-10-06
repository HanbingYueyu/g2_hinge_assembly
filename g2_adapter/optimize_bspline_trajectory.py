#!/usr/bin/env python3
"""Create an independent B-spline preview and validated solved trajectory."""
import argparse
import hashlib
import json
import math
from pathlib import Path
try:
    from .bspline_trajectory import smooth_bspline
    from .cartesian_teach import load_cartesian_document,cartesian_document,joint_document,solve_cartesian_trajectory
    from .cartesian_teach_trajectory_gdk import DEFAULT_URDF
    from .g2_kinematics import UrdfKinematics
    from .safety import ARM_POSITION_LIMITS
    from .optimize_cartesian_teach_trajectory import _atomic_write_json
except ImportError:
    from bspline_trajectory import smooth_bspline
    from cartesian_teach import load_cartesian_document,cartesian_document,joint_document,solve_cartesian_trajectory
    from cartesian_teach_trajectory_gdk import DEFAULT_URDF
    from g2_kinematics import UrdfKinematics
    from safety import ARM_POSITION_LIMITS
    from optimize_cartesian_teach_trajectory import _atomic_write_json


def main(argv=None):
    p=argparse.ArgumentParser(description='独立 B 样条优化和整轨 IK，不连接机器人')
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--urdf',type=Path,default=DEFAULT_URDF)
    p.add_argument('--control-points',type=int,default=6)
    p.add_argument('--strength',type=float,default=.1)
    p.add_argument('--rate-hz',type=float,default=100.)
    p.add_argument('--duration-s',type=float)
    p.add_argument('--ik-position-tolerance-mm',type=float,default=.0001)
    p.add_argument('--max-velocity-rad-s',type=float,default=1.)
    p.add_argument('--max-acceleration-rad-s2',type=float,default=20.)
    args=p.parse_args(argv)
    if args.input.resolve()==args.output.resolve():p.error('输出不能覆盖原始文件')
    raw=load_cartesian_document(json.loads(args.input.read_text()))
    raw_doc=json.loads(args.input.read_text())
    digest=hashlib.sha256(args.urdf.read_bytes()).hexdigest()
    if raw_doc.get('urdf_sha256')!=digest:p.error('URDF 与原始轨迹模型不一致')
    fitted,optimization=smooth_bspline(raw,control_points=args.control_points,strength=args.strength,
                                      rate_hz=args.rate_hz,duration_s=args.duration_s)
    preview=args.output.with_name(args.output.stem+'_cartesian.json')
    if preview.resolve()==args.input.resolve():p.error('预览文件不能覆盖原始文件')
    _atomic_write_json(preview,cartesian_document(fitted,urdf_sha256=digest,optimization=optimization))
    print(f'样条目标已生成：{preview}；{len(fitted.samples)} 点，开始 IK',flush=True)
    solved,report=solve_cartesian_trajectory(fitted,UrdfKinematics(args.urdf),
        joint_limits=ARM_POSITION_LIMITS[:7] if raw.side=='left' else ARM_POSITION_LIMITS[7:],
        source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),urdf_sha256=digest,
        max_velocity_rad_s=args.max_velocity_rad_s,max_acceleration_rad_s2=args.max_acceleration_rad_s2,
        ik_position_tolerance_m=args.ik_position_tolerance_mm/1000.,
        max_orientation_error_rad=1e-5,ik_reference_weight=1e-8)
    _atomic_write_json(args.output,joint_document(solved,report,optimization=optimization))
    print(json.dumps({'output':str(args.output),'replay_ready':report.replay_ready,
        'max_fk_error_mm':report.max_position_error_m*1000,
        'max_fk_error_deg':math.degrees(report.max_orientation_error_rad),
        'max_velocity_rad_s':report.max_velocity_rad_s,
        'max_acceleration_rad_s2':report.max_acceleration_rad_s2,'errors':report.errors},ensure_ascii=False),flush=True)
    return 0 if report.replay_ready else 1

if __name__=='__main__':raise SystemExit(main())
