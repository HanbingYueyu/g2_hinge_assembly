#!/usr/bin/env bash
set -euo pipefail

G2_HOST="${G2_HOST:-10.42.1.101}"
G2_SSH_TARGET="${G2_SSH_TARGET:-agi@${G2_HOST}}"
G2_PTP_SCRIPT="${G2_PTP_SCRIPT:-/home/flyfuture/.cache/agibot/app/gdk/scripts/ptp_hard.sh}"

usage() {
    cat <<'USAGE'
Usage: ./scripts/start_voice_services.sh

在开发机执行：检查 G2 网络与 genie_app.service，通过 SSH 启动/复用 G2
语音服务和远程 GDK 桥接，将 G2 音量设置为 100，随后在本机前台启动 ptp_hard.sh。

可选环境变量：
  SSHPASS          SSH 密码；未设置时会安全地交互输入
  G2_HOST          G2 地址（默认 10.42.1.101）
  G2_SSH_TARGET    SSH 用户和地址（默认 agi@$G2_HOST）
  G2_PTP_SCRIPT    本机 PTP 脚本路径

Ctrl+C 停止本机 PTP；不会停止 G2 上的两个语音服务。
USAGE
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ "$#" -ne 0 ]]; then
    usage >&2
    exit 2
fi

for tool in ping timeout ssh sshpass base64 tr sudo; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
        echo "缺少命令：${tool}" >&2
        exit 2
    fi
done

if [[ ! -f "${G2_PTP_SCRIPT}" ]]; then
    echo "找不到本机 PTP 脚本：${G2_PTP_SCRIPT}" >&2
    exit 2
fi

if [[ -z "${SSHPASS:-}" ]]; then
    if [[ ! -r /dev/tty ]]; then
        echo "请设置 SSHPASS，或在交互式终端运行以输入 G2 SSH 密码。" >&2
        exit 2
    fi
    read -r -s -p "G2 SSH 密码：" SSHPASS </dev/tty
    printf '\n' >/dev/tty
    if [[ -z "${SSHPASS}" ]]; then
        echo "SSH 密码不能为空。" >&2
        exit 2
    fi
    export SSHPASS
fi

echo "[网络检查] 检查 G2 ${G2_HOST}..."
if ! ping -c 1 -W 2 "${G2_HOST}" >/dev/null 2>&1; then
    echo "无法 ping 通 G2：${G2_HOST}" >&2
    exit 1
fi
if ! timeout 3 bash -c 'exec 3<>/dev/tcp/$1/2379' _ "${G2_HOST}"; then
    echo "无法连接 G2 DDS 端口 ${G2_HOST}:2379。" >&2
    exit 1
fi

remote_script="$(cat <<'REMOTE'
#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/home/agi/app
if ! systemctl is-active --quiet genie_app.service; then
    echo "G2 主程序 genie_app.service 未处于 active 状态。"
    systemctl status genie_app.service --no-pager -l || true
    exit 1
fi

start_service() {
    local unit="$1"
    shift
    if systemctl is-active --quiet "${unit}"; then
        echo "[已运行] ${unit}"
        return 0
    fi

    systemctl reset-failed "${unit}" >/dev/null 2>&1 || true
    echo "[启动] ${unit}"
    systemd-run --unit="${unit}" \
        --working-directory="${APP_DIR}" \
        --property=Restart=no \
        "$@"

    for _ in $(seq 1 15); do
        if systemctl is-active --quiet "${unit}"; then
            echo "[运行中] ${unit}"
            return 0
        fi
        if systemctl is-failed --quiet "${unit}"; then
            break
        fi
        sleep 1
    done

    echo "服务启动失败：${unit}"
    journalctl -u "${unit}" -n 40 --no-pager || true
    return 1
}

start_service agibot-voice-env-manual.service \
    /bin/bash -lc '
        set -a
        source /home/agi/app/conf/sys/run.conf
        set +a
        export LD_LIBRARY_PATH="$(find /home/agi/app/lib -type f -name "*.so" -printf "%h\n" | sort -u | paste -sd:):${LD_LIBRARY_PATH:-}"
        exec /home/agi/app/bin/agibot_voice
    '

start_service gdk-lumina-remote-env-manual.service \
    /bin/bash -lc '
        set -a
        source /home/agi/app/conf/sys/run.conf
        set +a
        export AORTA_DISPATCHER_THREAD_NUM=5
        export COSINE_BUS_ENABLE_TOPIC_CONFIG_MERGE=1
        export COSINE_IDL_CONVERTER_LIBS=/home/agi/app/lib/liblumina_idl_proto_bridge.so
        export COSINE_BUS_CONFIG_PATH=/home/agi/app/conf/dds/base_remote_fastdds.prototxt
        export LD_LIBRARY_PATH="$(find /home/agi/app/lib -type f -name "*.so" -printf "%h\n" | sort -u | paste -sd:):${LD_LIBRARY_PATH:-}"
        exec /home/agi/app/bin/gdk_lumina_adapter /home/agi/app/gdk/config/base_remote_bridge_config.json
    '

echo "G2 语音服务和远程 GDK 桥接均已运行。"
REMOTE
)"

remote_payload="$(printf '%s' "${remote_script}" | base64 | tr -d '\n')"
remote_command="printf '%s' '${remote_payload}' | base64 -d | sudo bash -s"

echo "[G2 服务] 通过 SSH 启动语音服务和远程 GDK 桥接..."
sshpass -e ssh -tt -o ConnectTimeout=8 "${G2_SSH_TARGET}" "${remote_command}"

echo "[音量] 设置 G2 音量为 100..."
G2_VOICE_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${G2_VOICE_SCRIPT_DIR}/run_gdk_python.sh" - <<'PYTHON'
import time
import agibot_gdk as gdk

if gdk.gdk_init() != gdk.GDKRes.kSuccess:
    raise RuntimeError("GDK 初始化失败，无法设置音量")
try:
    interaction = gdk.Interaction()
    for attempt in range(3):
        try:
            interaction.set_volume(100)
            print("G2 音量已设置为 100")
            break
        except RuntimeError:
            if attempt == 2:
                raise
            time.sleep(1.0)
finally:
    gdk.gdk_release()
PYTHON

echo "[PTP] 启动开发机时间同步；保持此终端运行，按 Ctrl+C 停止 PTP。"
sudo bash "${G2_PTP_SCRIPT}"
