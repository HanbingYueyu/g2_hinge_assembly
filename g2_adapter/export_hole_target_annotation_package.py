#!/usr/bin/env python3
"""Export a portable, non-GDK hole/target annotation package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "hole_target_det_v2" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出包含视频、YOLO 预标工具和模型的最小标注包")
    parser.add_argument("--video", type=Path, action="append", default=[], help="要打包的视频，可重复指定")
    parser.add_argument("--video-dir", type=Path, default=None, help="打包目录下全部 mp4/avi/mov")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "exports" / "hole_target_annotation_package.zip")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    videos = [path.expanduser().resolve() for path in args.video]
    if args.video_dir is not None:
        root = args.video_dir.expanduser().resolve()
        videos.extend(sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}))
    unique_videos = []
    seen = set()
    for path in videos:
        if path in seen:
            continue
        seen.add(path)
        if not path.is_file():
            raise SystemExit(f"视频不存在：{path}")
        unique_videos.append(path)
    if not unique_videos:
        raise SystemExit("请至少指定一个 --video 或 --video-dir")
    model = args.model.expanduser().resolve()
    if not model.is_file():
        raise SystemExit(f"模型不存在：{model}")
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"输出已存在：{output}；如需覆盖请加 --overwrite")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="hole_target_package_") as temporary:
        root = Path(temporary) / "hole_target_annotation_package"
        (root / "tools").mkdir(parents=True)
        (root / "models" / "hole_target_det_v2").mkdir(parents=True)
        (root / "videos").mkdir(parents=True)
        shutil.copy2(PROJECT_ROOT / "g2_adapter" / "annotate_hole_target_video.py", root / "tools" / "annotate_hole_target_video.py")
        shutil.copy2(PROJECT_ROOT / "g2_adapter" / "hole_target_detector.py", root / "tools" / "hole_target_detector.py")
        shutil.copy2(PROJECT_ROOT / "g2_adapter" / "merge_hole_target_annotations.py", root / "tools" / "merge_hole_target_annotations.py")
        shutil.copy2(model, root / "models" / "hole_target_det_v2" / "best.pt")
        used_names = set()
        manifest_files = []
        for source in unique_videos:
            name = source.name
            if name in used_names:
                raise SystemExit(f"视频文件名重复，请先重命名：{name}")
            used_names.add(name)
            destination = root / "videos" / name
            shutil.copy2(source, destination)
            manifest_files.append({"path": f"videos/{name}", "size_bytes": destination.stat().st_size, "sha256": sha256(destination)})
        requirements = "opencv-python\nultralytics>=8.0\n"
        (root / "requirements.txt").write_text(requirements, encoding="utf-8")
        readme = """# hole/target 标注包

解压并进入目录：

```bash
mkdir -p hole_target_annotation_package
unzip hole_target_annotation_package.zip -d hole_target_annotation_package
cd hole_target_annotation_package
```

安装依赖：

```bash
python3 -m pip install -r requirements.txt
```

开始标注（CPU 可运行，GPU 更快）：

```bash
python3 tools/annotate_hole_target_video.py \\
  --source videos/视频文件.mp4 \\
  --out-dir annotations/视频名称 \\
  --model models/hole_target_det_v2/best.pt \\
  --stride 6 \\
  --confidence 0.10 \\
  --host 0.0.0.0 --port 8771
```

浏览器打开 `http://127.0.0.1:8771`。标注结果写入 `annotations/`，类别为 `0=hole`、`1=target`。
网页顶部可输入标注帧号并跳转；帧号按抽帧后的序号计算（从 1 开始）。
初始默认标注 `hole`，类别选择会保持；按 `T` 切换 `target`；右键画布可复制上一帧位置；按 Enter 保存并进入下一帧。
每台电脑处理不同视频时使用不同端口或顺序运行，最后合并各自的 `annotations/*/images` 和 `labels`。

合并多台电脑的标注：

```bash
python3 tools/merge_hole_target_annotations.py \\
  --input-dir annotations \\
  --out-dir datasets/hole_target_yolo_v1 \\
  --require-one-each
```
只保留恰好一个 hole 和一个 target 的样本，空标签及其他数量的标签不会写入输出。
"""
        (root / "README.md").write_text(readme, encoding="utf-8")
        manifest = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "classes": ["hole", "target"], "files": manifest_files + [{"path": "models/hole_target_det_v2/best.pt", "size_bytes": (root / "models/hole_target_det_v2/best.pt").stat().st_size, "sha256": sha256(root / "models/hole_target_det_v2/best.pt")}], "tool": "tools/annotate_hole_target_video.py"}
        (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        compression = zipfile.ZIP_STORED
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix(), compress_type=zipfile.ZIP_STORED if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".pt"} else compression)
    print(json.dumps({"output": str(output), "videos": [path.name for path in unique_videos], "size_bytes": output.stat().st_size, "contents": ["videos/", "tools/", "models/hole_target_det_v2/best.pt", "requirements.txt", "README.md", "manifest.json"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
