#!/usr/bin/env bash
# Build the GDK CPython 3.12 binding into .venv-sam3 without touching GDK 3.10.
# The official env.sh appends to LD_LIBRARY_PATH/PYTHONPATH even when they are
# initially unset, so load it before enabling nounset.
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_python="${project_root}/.venv-sam3/bin/python"

if [[ ! -x "${venv_python}" ]]; then
    echo "未找到 ${venv_python}；请先创建并安装 .venv-sam3。" >&2
    exit 2
fi
if ! command -v uv >/dev/null 2>&1; then
    echo "未找到 uv。" >&2
    exit 2
fi

source "${project_root}/scripts/gdk_env.sh"
set -u
binding_source="${G2_GDK_RUNTIME}/gdk/build_dep/python/pybind"
if [[ ! -f "${binding_source}/cpp/bind_gdk.cpp" ]]; then
    echo "未找到可编译的 GDK pybind 源码：${binding_source}" >&2
    exit 2
fi

# Do not invoke the upstream setup.py. Its custom install step deletes the
# official GDK 3.10 agibot_gdk directory. This recipe only installs to .venv-sam3.
uv pip install --python "${venv_python}" \
    "pybind11>=2.11" "protobuf==5.28.3" "setuptools>=45" wheel

G2_GDK_RUNTIME="${G2_GDK_RUNTIME}" \
    uv pip install --python "${venv_python}" --no-build-isolation \
    "${project_root}/gdk_py312_binding"

"${project_root}/scripts/run_gdk_sam3_python.sh" -c '
import agibot_gdk
print("GDK 3.12 binding:", agibot_gdk.__file__)
print("Camera available:", hasattr(agibot_gdk, "Camera"))
'
