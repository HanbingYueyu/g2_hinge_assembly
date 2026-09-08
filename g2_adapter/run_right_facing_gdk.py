#!/usr/bin/env python3
"""Right workpiece facing servo; distance is measured back from calibrated grasp."""
import argparse
from pathlib import Path
import subprocess
import sys


def build_command(args):
    command = [sys.executable, str(Path(__file__).with_name("right_apriltag_grasp_gdk.py")),
               "--config", str(args.config), "--face-distance-mm", str(args.distance_mm),
               "--auto-speed-mm-s", str(args.speed_mm_s),
               "--auto-rotation-deg-s", str(args.rotation_deg_s),
               "--web-port", str(args.web_port)]
    if args.allow_motion:
        command += ["--execute", "--autonomous-approach", "--allow-motion"]
    return command


def main():
    parser = argparse.ArgumentParser(description="右手保持水平正对电批并持续视觉伺服；不操作夹爪。Ctrl+C 停止。")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("right_apriltag_grasp.example.json"))
    parser.add_argument("--distance-mm", type=float, default=100, help="末端原点沿水平前伸轴到 STEP 红色夹取中心的距离，10–300 mm")
    parser.add_argument("--speed-mm-s", type=float, default=10)
    parser.add_argument("--rotation-deg-s", type=float, default=10)
    parser.add_argument("--web-port", type=int, default=8767)
    parser.add_argument("--allow-motion", action="store_true")
    args = parser.parse_args()
    try:
        return subprocess.call(build_command(args), cwd=Path(__file__).resolve().parents[1])
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
