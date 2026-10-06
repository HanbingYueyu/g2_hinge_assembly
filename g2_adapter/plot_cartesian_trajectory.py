#!/usr/bin/env python3
"""Offline 3D comparison of recorded Cartesian poses and solved joint paths."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .cartesian_teach import load_cartesian_document, load_joint_document
    from .g2_kinematics import UrdfKinematics
except ImportError:
    from cartesian_teach import load_cartesian_document, load_joint_document
    from g2_kinematics import UrdfKinematics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF = PROJECT_ROOT / 'robot_description/G2_t2v2_crsB_omnipicker.urdf'


def load_curves(raw_path, optimized_path, urdf_path):
    raw = load_cartesian_document(json.loads(raw_path.read_text()))
    solved, _report = load_joint_document(json.loads(optimized_path.read_text()))
    if raw.side != solved.side:
        raise ValueError('原始和优化轨迹必须属于同一机械臂')
    if not raw.samples or not solved.points:
        raise ValueError('轨迹不能为空')
    digest = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
    if digest != solved.urdf_sha256:
        raise ValueError('URDF 与优化轨迹记录的模型不一致，无法可靠计算末端曲线')
    expected_tip = 'arm_l_end_link' if raw.side == 'left' else 'arm_r_end_link'
    if raw.base_frame != 'base_link' or raw.tip_frame != expected_tip:
        raise ValueError('原始轨迹坐标系与优化轨迹的末端坐标系不一致')
    model = UrdfKinematics(urdf_path)
    positions = []
    for point in solved.points:
        joints = dict(solved.fixed_joint_positions)
        joints.update(zip(solved.joint_names, point.joint_positions_rad))
        positions.append(model.forward(raw.base_frame, raw.tip_frame, joints)[:3, 3])
    return raw, np.asarray([s.position_m for s in raw.samples])*1000., np.asarray(positions)*1000.


def main(argv=None):
    parser = argparse.ArgumentParser(description='离线绘制原始和优化后末端三维曲线，不连接机器人。')
    parser.add_argument('--raw', type=Path, required=True, help='原始 *_raw.json')
    parser.add_argument('--optimized', type=Path, required=True, help='优化并求解后的 *_solved.json')
    parser.add_argument('--urdf', type=Path, default=DEFAULT_URDF)
    parser.add_argument('--output', type=Path, help='输出 PNG/SVG/PDF，默认保存到原始文件同目录')
    parser.add_argument('--spline', type=Path, help='额外叠加样条优化的 solved 文件')
    parser.add_argument('--show', action='store_true', help='打开可用鼠标旋转、缩放的三维窗口，需要桌面显示环境')
    args = parser.parse_args(argv)
    try:
        trajectory, original, optimized = load_curves(args.raw, args.optimized, args.urdf)
    except (ValueError, OSError, KeyError) as exc:
        parser.error(str(exc))
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 8), layout='constrained')
    ax = fig.add_subplot(111, projection='3d')
    curves = [(original, 'Original', '#7b8794', '--'),
              (optimized, 'Current optimized (FK)', '#0072b2', '-')]
    if args.spline:
        _, _, spline = load_curves(args.raw, args.spline, args.urdf)
        curves.append((spline, 'B-spline optimized (FK)', '#d55e00', '-'))
    for points, label, color, style in curves:
        ax.plot(*points.T, label=label, color=color, linestyle=style, linewidth=1.5, alpha=.85)
        ax.scatter(*points[0], color=color, marker='o', s=45)
        ax.scatter(*points[-1], color=color, marker='X', s=65)
    ax.set(xlabel='base_link X (mm)', ylabel='base_link Y (mm)', zlabel='base_link Z (mm)',
           title=f'{trajectory.side} end-link: original vs optimized\nCircle = start; X = end')
    combined = np.vstack([c[0] for c in curves])
    center = (combined.max(axis=0)+combined.min(axis=0))/2
    half = max(float(np.ptp(combined, axis=0).max())*.55, 1.)
    for setter, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(mid-half, mid+half)
    ax.set_box_aspect((1, 1, 1))
    ax.legend()
    output = args.output or args.raw.with_name(args.raw.stem+'_comparison.png')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    print(f'已保存：{output.resolve()}')
    for points, name, _, _ in curves:
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        print(f'{name}：{len(points)} 点，路径长度 {length:.2f} mm')
    for points, name, _, _ in curves[1:]:
        print(f'{name}：起点差 {np.linalg.norm(original[0]-points[0]):.6f} mm；终点差 {np.linalg.norm(original[-1]-points[-1]):.6f} mm')
    if args.show:
        plt.show()
    else:
        plt.close(fig)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
