#!/usr/bin/env bash
set -euo pipefail
VLM_PACKAGE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Prefer the project environment; VLM calls do not initialize GDK.
VLM_PYTHON="${VLM_PYTHON:-${VLM_PACKAGE_ROOT}/../.venv-sam3/bin/python}"
if [[ ! -x "$VLM_PYTHON" ]]; then
    VLM_PYTHON=python3
fi
exec "$VLM_PYTHON" "$VLM_PACKAGE_ROOT/src/vlm_three_checks.py" "$@"
