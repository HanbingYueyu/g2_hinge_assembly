#!/usr/bin/env python3
"""Preview or explicitly apply a validated project-owned Thor camera config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "thor_camera_conf_hand_depth.json"
HEAD_DEPTH_DIAGNOSTIC_CONFIG = (
    PROJECT_ROOT / "config" / "thor_camera_conf_head_depth_diagnostic.json"
)
PROFILE_REQUIREMENTS = {
    DEFAULT_CONFIG.resolve(): {
        "enabled": {"head_color", "head_depth", "hand_left_depth", "hand_right_depth"},
        "disabled": {"head_stereo_left", "head_stereo_right", "head_back_fisheye",
                     "head_left_fisheye", "head_right_fisheye"},
    },
    HEAD_DEPTH_DIAGNOSTIC_CONFIG.resolve(): {
        "enabled": {"head_color", "head_depth", "hand_left_color", "hand_right_color"},
        "disabled": {"hand_left_depth", "hand_right_depth", "head_stereo_left",
                     "head_stereo_right", "head_back_fisheye", "head_left_fisheye",
                     "head_right_fisheye"},
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="预览或显式应用受校验的 Thor 相机发布配置。"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="唯一会调用 GDK set_dev_camera_config() 的确认参数。",
    )
    return parser.parse_args()


def load_and_validate(path: Path) -> tuple[Path, dict]:
    candidate = path.expanduser()
    # Commands are intended to work from any current directory.  Keep relative
    # profile paths rooted at this project rather than at the caller's shell.
    resolved = (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()
    document = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("相机配置根节点必须是对象")
    if resolved not in PROFILE_REQUIREMENTS:
        allowed = ", ".join(str(item) for item in PROFILE_REQUIREMENTS)
        raise ValueError(f"只允许应用项目内已验证的配置：{allowed}")
    enabled = {
        str(value.get("name"))
        for value in document.values()
        if isinstance(value, dict) and value.get("publish") is True
    }
    requirements = PROFILE_REQUIREMENTS[resolved]
    missing = requirements["enabled"] - enabled
    unexpected = requirements["disabled"] & enabled
    if missing or unexpected:
        raise ValueError(
            f"配置不符合安全预设：缺少启用={sorted(missing)}，"
            f"错误启用={sorted(unexpected)}"
        )
    return resolved, document


def main() -> int:
    args = parse_args()
    try:
        config_path, document = load_and_validate(args.config)
    except Exception as exc:
        print(f"配置检查失败：{exc}", file=sys.stderr)
        return 2
    published = {
        value["name"]: value["fps"]
        for value in document.values()
        if isinstance(value, dict) and value.get("publish") is True
    }
    print(json.dumps({
        "config": str(config_path),
        "apply": args.apply,
        "published_streams_fps": published,
    }, indent=2, ensure_ascii=False))
    if not args.apply:
        print("只读预览；未初始化 GDK，未修改相机配置。")
        print("现场确认后，使用同一命令加 --apply。")
        return 0

    try:
        import agibot_gdk
    except ImportError as exc:
        print("agibot_gdk 不可用；请通过 scripts/run_gdk_python.sh 运行。", file=sys.stderr)
        return 2
    if agibot_gdk.gdk_init() != agibot_gdk.GDKRes.kSuccess:
        print("GDK 初始化失败。", file=sys.stderr)
        return 2
    camera = None
    try:
        camera = agibot_gdk.Camera()
        camera.set_dev_camera_config(str(config_path))
        print("GDK 已接受相机配置；尚未执行模式切换。")
        print("请由操作员执行官方 mode_switch --mode develop，并重启机器人/GDK 服务使模式与相机配置生效后，再运行 inspect_cameras.py。")
        return 0
    except Exception as exc:
        print(f"应用相机配置失败：{exc}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            try:
                camera.close_camera()
            except Exception:
                pass
        try:
            agibot_gdk.gdk_release()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
