#!/usr/bin/env bash
# Source this file to load the official GDK runtime for this project.
# G2_GDK_RUNTIME can override the default official installation directory.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "请使用: source scripts/gdk_env.sh" >&2
    exit 2
fi

if [[ "${G2_GDK_ENV_LOADED:-}" == "1" ]]; then
    return 0
fi

G2_PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
G2_PROJECT_LOG_DIR="${G2_PROJECT_ROOT}/logs"
mkdir -p "${G2_PROJECT_LOG_DIR}"
# GDK uses glog; set this before Python imports agibot_gdk.
export GLOG_log_dir="${G2_PROJECT_LOG_DIR}"

G2_GDK_RUNTIME="${G2_GDK_RUNTIME:-${HOME}/.cache/agibot/app}"
G2_GDK_ENV_FILE="${G2_GDK_RUNTIME}/env.sh"

if [[ ! -f "${G2_GDK_ENV_FILE}" ]]; then
    echo "未找到 GDK 运行时: ${G2_GDK_ENV_FILE}" >&2
    echo "请安装官方 GDK，或设置 G2_GDK_RUNTIME 指向其 app 目录。" >&2
    return 2
fi

source "${G2_GDK_ENV_FILE}" "${G2_GDK_RUNTIME}"
export G2_GDK_ENV_LOADED=1
