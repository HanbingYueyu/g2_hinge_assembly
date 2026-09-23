#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="${repo_root}/scripts/start_voice_services.sh"

if [[ ! -x "${launcher}" ]]; then
    echo "FAIL: executable launcher not found: ${launcher}" >&2
    exit 1
fi

help_output="$("${launcher}" --help)"
grep -Fq 'SSHPASS' <<<"${help_output}"
grep -Fq 'ptp_hard.sh' <<<"${help_output}"

temp_dir="$(mktemp -d)"
trap 'rm -rf "${temp_dir}"' EXIT
mock_bin="${temp_dir}/bin"
mkdir -p "${mock_bin}"
call_log="${temp_dir}/calls.log"

cat >"${mock_bin}/ping" <<'MOCK'
#!/usr/bin/env bash
exit 0
MOCK

cat >"${mock_bin}/timeout" <<'MOCK'
#!/usr/bin/env bash
exit 0
MOCK

cat >"${mock_bin}/sshpass" <<'MOCK'
#!/usr/bin/env bash
printf 'sshpass' >>"${CALL_LOG}"
printf ' <%s>' "$@" >>"${CALL_LOG}"
printf '\n' >>"${CALL_LOG}"
exit 0
MOCK

cat >"${mock_bin}/sudo" <<'MOCK'
#!/usr/bin/env bash
printf 'sudo' >>"${CALL_LOG}"
printf ' <%s>' "$@" >>"${CALL_LOG}"
printf '\n' >>"${CALL_LOG}"
exit 0
MOCK

chmod +x "${mock_bin}"/*
CALL_LOG="${call_log}" \
PATH="${mock_bin}:${PATH}" \
SSHPASS=test-password \
G2_PTP_SCRIPT=/bin/true \
G2_HOST=192.0.2.10 \
G2_SSH_TARGET=agi@192.0.2.10 \
"${launcher}" >"${temp_dir}/launcher.out"

grep -q '^sshpass ' "${call_log}"
grep -q '^sudo ' "${call_log}"
[[ "$(wc -l <"${call_log}")" -eq 2 ]]
! grep -Fq '<1>' "${call_log}"

echo "PASS: launcher help and SSH-then-PTP orchestration"
