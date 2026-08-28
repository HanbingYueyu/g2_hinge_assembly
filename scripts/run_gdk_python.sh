#!/usr/bin/env bash
# Run a Python 3.10 program in this project with the official GDK runtime.
G2_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${G2_PROJECT_ROOT}/scripts/gdk_env.sh"
set -euo pipefail
export UV_CACHE_DIR="${UV_CACHE_DIR:-${G2_PROJECT_ROOT}/.uv-cache}"

if ! command -v uv >/dev/null 2>&1; then
    echo "未找到 uv；请安装 uv 后重试。" >&2
    exit 2
fi

cd "${G2_PROJECT_ROOT}"
exec uv run --no-project --python 3.10 python "$@"
