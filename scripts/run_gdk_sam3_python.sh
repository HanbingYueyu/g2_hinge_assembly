#!/usr/bin/env bash
# Run GDK under the project-local CPython 3.12 binding. The historical
# filename is kept so existing commands remain compatible; it does not load
# SAM3 by itself.
# The official env.sh appends to variables that can be initially unset.
set -eo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_python="${project_root}/.venv-sam3/bin/python"
if [[ ! -x "${venv_python}" ]]; then
    echo "未找到 .venv-sam3。" >&2
    exit 2
fi

source "${project_root}/scripts/gdk_env.sh"
set -u

# env.sh adds the official CPython 3.10 module directory to PYTHONPATH. Keep
# its dynamic-library environment, but never expose that incompatible .so to
# CPython 3.12; the venv holds the freshly compiled extension instead.
# Project-root packages such as dianpi/ are resolved by the tools themselves
# (appended to sys.path), so the repo's sam3/ checkout cannot shadow the
# installed sam3 package.
unset PYTHONPATH

cd "${project_root}"
exec "${venv_python}" "$@"
