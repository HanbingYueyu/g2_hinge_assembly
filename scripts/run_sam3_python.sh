#!/usr/bin/env bash
# Run offline SAM3 tools in their isolated Python 3.12 environment.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="$project_root/.venv-sam3/bin/python"

if [[ ! -x "$python_bin" ]]; then
  echo "未找到 .venv-sam3。请先按 MIGRATION.md 安装 SAM3 环境。" >&2
  exit 2
fi

cd "$project_root"
exec "$python_bin" "$@"
