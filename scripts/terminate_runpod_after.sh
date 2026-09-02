#!/usr/bin/env bash

# Run this guard outside the Pod so container failures cannot disable the hard limit.
set -u
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNPODCTL_WRAPPER="${SCRIPT_DIR}/runpodctl_project.sh"
RUNPOD_S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
RUNPOD_READINESS_HELPER="${SCRIPT_DIR}/runpod_readiness.py"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: terminate_runpod_after.sh POD_ID DELAY_SECONDS [LOG_FILE]" >&2
    exit 2
fi

POD_ID="$1"
DELAY_SECONDS="$2"
LOG_FILE="${3:-${HOME:-/tmp}/.local/state/runpod-guards/${POD_ID}.log}"
MAX_ATTEMPTS="${RUNPOD_GUARD_MAX_ATTEMPTS:-3}"
RETRY_SECONDS="${RUNPOD_GUARD_RETRY_SECONDS:-30}"
LIFECYCLE_KEY="${RUNPOD_GUARD_LIFECYCLE_KEY:-}"
POLL_SECONDS="${RUNPOD_GUARD_POLL_SECONDS:-30}"
READY_FILE="${RUNPOD_GUARD_READY_FILE:-}"
REQUIRE_LIFECYCLE="${RUNPOD_GUARD_REQUIRE_LIFECYCLE:-0}"
RUNPOD_GUARD_VOLUME_ROOT="${RUNPOD_GUARD_VOLUME_ROOT:-/runpod-volume}"
RUNPOD_GUARD_RUN_ID="${RUNPOD_GUARD_RUN_ID:-}"
GUARD_S3_CONNECT_TIMEOUT="${RUNPOD_GUARD_S3_CONNECT_TIMEOUT:-5}"
GUARD_S3_READ_TIMEOUT="${RUNPOD_GUARD_S3_READ_TIMEOUT:-15}"
GUARD_S3_MAX_ATTEMPTS="${RUNPOD_GUARD_S3_MAX_ATTEMPTS:-1}"

if [[ -n "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "The hard-limit guard must run outside the RunPod Pod" >&2
    exit 2
fi
if [[ ! "${POD_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid Pod ID" >&2
    exit 2
fi
if [[ ! "${DELAY_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "DELAY_SECONDS must be a non-negative integer" >&2
    exit 2
fi
if [[ "${DELAY_SECONDS}" -eq 0 && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "A zero-second hard limit is allowed only in test mode" >&2
    exit 2
fi
if [[ ! "${MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ || ! "${RETRY_SECONDS}" =~ ^[0-9]+$ ]]; then
    echo "Invalid guard retry settings" >&2
    exit 2
fi
if [[ ! "${POLL_SECONDS}" =~ ^[0-9]+$ \
    || ( "${POLL_SECONDS}" == "0" && "${RUNPOD_TEST_MODE:-0}" != "1" ) ]]; then
    echo "RUNPOD_GUARD_POLL_SECONDS must be a positive integer outside test mode" >&2
    exit 2
fi
if [[ ! "${GUARD_S3_CONNECT_TIMEOUT}" =~ ^[1-9][0-9]*$ \
    || ! "${GUARD_S3_READ_TIMEOUT}" =~ ^[1-9][0-9]*$ \
    || ! "${GUARD_S3_MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Guard S3 timeout and retry settings must be positive integers" >&2
    exit 2
fi
if [[ "${REQUIRE_LIFECYCLE}" != "0" && "${REQUIRE_LIFECYCLE}" != "1" ]]; then
    echo "RUNPOD_GUARD_REQUIRE_LIFECYCLE must be 0 or 1" >&2
    exit 2
fi
if [[ ! "${RUNPOD_GUARD_VOLUME_ROOT}" =~ ^/[A-Za-z0-9._/-]+$ \
    || ( "${RUNPOD_GUARD_VOLUME_ROOT}" != "/" \
        && "${RUNPOD_GUARD_VOLUME_ROOT}" == */ ) ]]; then
    echo "RUNPOD_GUARD_VOLUME_ROOT must be a canonical absolute path" >&2
    exit 2
fi
if [[ -n "${RUNPOD_GUARD_RUN_ID}" \
    && ( ! "${RUNPOD_GUARD_RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$ \
        || "${RUNPOD_GUARD_RUN_ID}" == *--* ) ]]; then
    echo "RUNPOD_GUARD_RUN_ID must be a safe 1-120 character run ID" >&2
    exit 2
fi
if [[ -n "${READY_FILE}" ]]; then
    if [[ "${READY_FILE}" != /* || "$(dirname "${READY_FILE}")" != "$(dirname "${LOG_FILE}")" ]]; then
        echo "Guard readiness file must be an absolute sibling of the guard log" >&2
        exit 2
    fi
fi
case "${LIFECYCLE_KEY}" in
    ""|lifecycle/stage1/cpu-preparation.json|\
        lifecycle/stage1/mixed-finalization.json|\
        lifecycle/stage1/training.json|lifecycle/stage1/validation.json) ;;
    *)
        echo "Unsupported lifecycle marker key" >&2
        exit 2
        ;;
esac
if [[ ( "${LIFECYCLE_KEY}" == "lifecycle/stage1/training.json" \
        || "${LIFECYCLE_KEY}" == "lifecycle/stage1/validation.json" ) \
    && -z "${RUNPOD_GUARD_RUN_ID}" ]]; then
    echo "GPU lifecycle guards require RUNPOD_GUARD_RUN_ID" >&2
    exit 2
fi
if ! command -v runpodctl >/dev/null 2>&1; then
    echo "runpodctl is required on the external guard host" >&2
    exit 127
fi
if [[ ! -f "${RUNPODCTL_WRAPPER}" || ! -r "${RUNPODCTL_WRAPPER}" \
    || ! -f "${RUNPOD_READINESS_HELPER}" || ! -r "${RUNPOD_READINESS_HELPER}" ]]; then
    echo "Project guard dependencies are unavailable" >&2
    exit 127
fi
runpod_assert_project_env_file "${LOCAL_PROJECT_ROOT}"

mkdir -p "$(dirname "${LOG_FILE}")"

runpod_guard_s3() {
    RUNPOD_S3_CONNECT_TIMEOUT_OVERRIDE="${GUARD_S3_CONNECT_TIMEOUT}" \
    RUNPOD_S3_READ_TIMEOUT_OVERRIDE="${GUARD_S3_READ_TIMEOUT}" \
    RUNPOD_S3_MAX_ATTEMPTS_OVERRIDE="${GUARD_S3_MAX_ATTEMPTS}" \
        bash "${RUNPOD_S3_WRAPPER}" "$@"
}

publish_guard_ready() {
    if [[ -z "${READY_FILE}" ]]; then
        return 0
    fi
    local ready_tmp="${READY_FILE}.tmp.$$"
    printf '{"state":"armed","pid":%d,"pod_id":"%s","run_id":"%s","lifecycle_key":"%s","armed_at":"%s"}\n' \
        "$$" "${POD_ID}" "${RUNPOD_GUARD_RUN_ID}" "${LIFECYCLE_KEY}" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        > "${ready_tmp}"
    mv "${ready_tmp}" "${READY_FILE}"
}

printf '[%s] guard armed for Pod %s; terminate after %s seconds\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${POD_ID}" "${DELAY_SECONDS}" >> "${LOG_FILE}"

termination_reason="hard-limit"
last_marker_json='{}'
matching_run_lifecycle_observed=0
pod_terminal_key=""
case "${LIFECYCLE_KEY}" in
    lifecycle/stage1/training.json|lifecycle/stage1/validation.json)
        if [[ -n "${RUNPOD_GUARD_RUN_ID}" ]]; then
            pod_terminal_key="lifecycle/runs/${RUNPOD_GUARD_RUN_ID}/pods/${POD_ID}/terminal.json"
        fi
        ;;
esac
rejected_lifecycle_keys=""
if [[ -n "${LIFECYCLE_KEY}" ]]; then
    IFS=',' read -r -a LIFECYCLE_KEYS <<< "${LIFECYCLE_KEY}"
    env_file="$(runpod_project_env_file "${LOCAL_PROJECT_ROOT}")"
    volume_id=""
    if [[ -r "${RUNPOD_S3_WRAPPER}" ]] \
        && command -v aws >/dev/null 2>&1 \
        && command -v python3 >/dev/null 2>&1 \
        && volume_id="$(runpod_read_project_env_value \
            "${env_file}" RUNPOD_NETWORK_VOLUME_ID 2>/dev/null)" \
        && [[ "${volume_id}" =~ ^[A-Za-z0-9_-]+$ ]]; then
        printf '[%s] lifecycle monitor armed for %s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${LIFECYCLE_KEY}" >> "${LOG_FILE}"
        publish_guard_ready
        guard_started=${SECONDS}
        while [[ $((SECONDS - guard_started)) -lt ${DELAY_SECONDS} ]]; do
            terminal_state=""
            terminal_key=""
            if [[ -n "${pod_terminal_key}" ]]; then
                pod_terminal_json=""
                expected_terminal_lifecycle_kind=stage1-training
                if [[ "${LIFECYCLE_KEY}" == "lifecycle/stage1/validation.json" ]]; then
                    expected_terminal_lifecycle_kind=stage1-validation
                fi
                if pod_terminal_json="$(runpod_guard_s3 s3 cp \
                    "s3://${volume_id}/${pod_terminal_key}" - \
                    --only-show-errors 2>/dev/null)" \
                    && printf '%s' "${pod_terminal_json}" | python3 -c \
                        'import json, re, sys
try:
    payload = json.load(sys.stdin)
except (TypeError, ValueError):
    raise SystemExit(2)
expected_kind, expected_pod_id, expected_run_id = sys.argv[1:4]
run_id = payload.get("wandb_run_id", "") if isinstance(payload, dict) else ""
valid = (
    isinstance(payload, dict)
    and type(payload.get("schema_version")) is int
    and payload.get("schema_version") == 1
    and payload.get("kind") == "gpu-workflow-terminal"
    and payload.get("lifecycle_kind") == expected_kind
    and payload.get("state") == "failed"
    and payload.get("reason") == "gpu_workflow_lease_contended"
    and payload.get("exit_code") == 75
    and payload.get("pod_id") == expected_pod_id
    and run_id == expected_run_id
    and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", run_id) is not None
    and "--" not in run_id
    and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,159}", str(payload.get("launch_id", ""))
    ) is not None
)
raise SystemExit(0 if valid else 2)' \
                        "${expected_terminal_lifecycle_kind}" "${POD_ID}" \
                        "${RUNPOD_GUARD_RUN_ID}" 2>/dev/null; then
                    terminal_state="lease-contended"
                    terminal_key="${pod_terminal_key}"
                fi
            fi
            if [[ -n "${pod_terminal_key}" ]]; then
                matching_run_lifecycle_observed=0
            fi
            for lifecycle_candidate in "${LIFECYCLE_KEYS[@]}"; do
                if [[ -n "${terminal_state}" ]]; then
                    break
                fi
                marker_json=""
                if ! marker_json="$(runpod_guard_s3 s3 cp \
                    "s3://${volume_id}/${lifecycle_candidate}" - \
                    --only-show-errors 2>/dev/null)"; then
                    continue
                fi
                case "${lifecycle_candidate}" in
                    lifecycle/stage1/cpu-preparation.json)
                        expected_lifecycle_kind="stage1-cpu-preparation"
                        ;;
                    lifecycle/stage1/mixed-finalization.json)
                        expected_lifecycle_kind="stage1-mixed-finalization"
                        ;;
                    lifecycle/stage1/training.json)
                        expected_lifecycle_kind="stage1-training"
                        ;;
                    lifecycle/stage1/validation.json)
                        expected_lifecycle_kind="stage1-validation"
                        ;;
                    *)
                        continue
                        ;;
                esac
                if ! marker_state="$(printf '%s' "${marker_json}" | \
                    python3 "${RUNPOD_READINESS_HELPER}" guard-lifecycle-state \
                        --expected-kind "${expected_lifecycle_kind}" \
                        --expected-pod-id "${POD_ID}" \
                        --active-run-id "${RUNPOD_GUARD_RUN_ID}" \
                        2>/dev/null)"; then
                    case ",${rejected_lifecycle_keys}," in
                        *",${lifecycle_candidate},"*) ;;
                        *)
                            printf '[%s] rejected non-canonical lifecycle marker: %s\n' \
                                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
                                "${lifecycle_candidate}" >> "${LOG_FILE}"
                            rejected_lifecycle_keys="${rejected_lifecycle_keys:+${rejected_lifecycle_keys},}${lifecycle_candidate}"
                            ;;
                    esac
                    continue
                fi
                if [[ "${lifecycle_candidate}" == "lifecycle/stage1/training.json" \
                    || "${lifecycle_candidate}" == "lifecycle/stage1/validation.json" ]]; then
                    last_marker_json="${marker_json}"
                    matching_run_lifecycle_observed=1
                fi
                if [[ "${marker_state}" == "ready" \
                    || "${marker_state}" == "failed" \
                    || "${marker_state}" == "timed_out" \
                    || "${marker_state}" == "waiting_for_provider" \
                    || "${marker_state}" == "waiting_for_budget" \
                    || "${marker_state}" == "waiting_for_resume" \
                    || "${marker_state}" == "waiting_for_preparation" \
                    || "${marker_state}" == "downloaded" ]]; then
                    terminal_state="${marker_state}"
                    terminal_key="${lifecycle_candidate}"
                    break
                fi
            done
            if [[ -n "${terminal_state}" ]]; then
                if [[ "${terminal_state}" == "lease-contended" ]]; then
                    termination_reason="workflow-lease-contended"
                    printf '[%s] matching Pod reported GPU workflow lease contention (%s)\n' \
                        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${terminal_key}" \
                        >> "${LOG_FILE}"
                else
                    termination_reason="lifecycle-${terminal_state}"
                    printf '[%s] matching Pod lifecycle reached terminal state: %s (%s)\n' \
                        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${terminal_state}" \
                        "${terminal_key}" >> "${LOG_FILE}"
                fi
                break
            fi
            elapsed=$((SECONDS - guard_started))
            remaining=$((DELAY_SECONDS - elapsed))
            if [[ ${remaining} -le 0 ]]; then
                break
            fi
            sleep_seconds=${POLL_SECONDS}
            if [[ ${sleep_seconds} -gt ${remaining} ]]; then
                sleep_seconds=${remaining}
            fi
            sleep "${sleep_seconds}"
        done
    else
        printf '[%s] lifecycle monitor unavailable; hard-limit protection remains active\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
        if [[ "${REQUIRE_LIFECYCLE}" == "1" ]]; then
            printf '[%s] required lifecycle monitor could not be armed\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
            exit 4
        fi
        publish_guard_ready
        sleep "${DELAY_SECONDS}"
    fi
else
    publish_guard_ready
    sleep "${DELAY_SECONDS}"
fi

if [[ "${termination_reason}" == "hard-limit" \
    && ( "${LIFECYCLE_KEY}" == "lifecycle/stage1/training.json" \
        || "${LIFECYCLE_KEY}" == "lifecycle/stage1/validation.json" ) \
    && "${volume_id:-}" =~ ^[A-Za-z0-9_-]+$ \
    && -r "${RUNPOD_S3_WRAPPER}" ]] \
    && command -v aws >/dev/null 2>&1 \
    && command -v python3 >/dev/null 2>&1; then
    timeout_json="$(printf '%s' "${last_marker_json}" | python3 -c \
        'import datetime, json, pathlib, re, sys
try:
    previous = json.load(sys.stdin)
except (TypeError, ValueError):
    previous = {}
if not isinstance(previous, dict):
    previous = {}
expected_kind = "stage1-validation" if sys.argv[3].endswith("validation.json") else "stage1-training"
active_run_id = sys.argv[5]
if previous:
    previous_schema = previous.get("schema_version")
    previous_run_id = previous.get("wandb_run_id", "")
    previous_is_valid = (
        isinstance(previous, dict)
        and type(previous_schema) is int
        and previous_schema == 1
        and previous.get("kind") == expected_kind
        and previous.get("pod_id") == sys.argv[1]
        and (not active_run_id or previous_run_id == active_run_id)
    )
    if not previous_is_valid:
        previous = {}
payload = {
    "schema_version": 1,
    "kind": expected_kind,
    "state": "timed_out",
    "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "pod_id": sys.argv[1],
    "launch_id": "external-guard-hard-limit",
    "exit_code": 124,
    "timed_out": True,
    "hard_limit_seconds": int(sys.argv[2]),
    "reason": "external_guard_hard_limit",
}
if payload["kind"] == "stage1-training":
    payload["training_completed"] = bool(previous.get("training_completed", False))
    payload["resume_discovery_required"] = True
else:
    payload["validation_completed"] = False
    payload["resume_validation_required"] = True
for key in (
    "max_runtime_seconds",
    "log_path",
    "recovery_path",
    "checkpoint",
    "result_path",
):
    if previous.get(key) not in (None, ""):
        payload[key] = previous[key]
previous_run_id = previous.get("wandb_run_id", "")
if active_run_id:
    payload["wandb_run_id"] = active_run_id
elif previous_run_id:
    payload["wandb_run_id"] = previous_run_id
run_id = payload.get("wandb_run_id")
run_scoped_keys = (
    "checkpoint",
    "result_path",
    "log_path",
    "recovery_path",
)
if not run_id and any(payload.get(key) not in (None, "") for key in run_scoped_keys):
    raise ValueError("Lifecycle run-scoped paths require wandb_run_id")
if run_id:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,119}", str(run_id)) is None or "--" in str(run_id):
        raise ValueError("Lifecycle contains an invalid wandb_run_id")
    volume_root = pathlib.PurePosixPath(sys.argv[4])
    checkpoint = payload.get("checkpoint")
    if checkpoint:
        checkpoint_path = pathlib.PurePosixPath(str(checkpoint))
        expected_parent = volume_root / "savedModel" / str(run_id)
        if checkpoint_path.parent != expected_parent or re.fullmatch(
            r"checkpoint-[0-9]{6,}", checkpoint_path.name
        ) is None:
            raise ValueError("Lifecycle contains a non-canonical checkpoint path")
    expected_evaluation_root = volume_root / "evaluations" / str(run_id)
    expected_evaluation_paths = {
        "result_path": expected_evaluation_root / "validation-benchmark.json",
    }
    for key, expected in expected_evaluation_paths.items():
        if payload.get(key) and pathlib.PurePosixPath(str(payload[key])) != expected:
            raise ValueError("Lifecycle contains a non-canonical {}".format(key))
    expected_log_root = volume_root / "logs" / str(run_id)
    for key in ("log_path", "recovery_path"):
        if not payload.get(key):
            continue
        candidate = pathlib.PurePosixPath(str(payload[key]))
        if ".." in candidate.parts or "." in candidate.parts:
            raise ValueError("Lifecycle contains a non-canonical {}".format(key))
        try:
            candidate.relative_to(expected_log_root)
        except ValueError:
            raise ValueError("Lifecycle contains a non-canonical {}".format(key))
        if key == "recovery_path" and candidate.name != "recovery.json":
            raise ValueError("Lifecycle recovery_path must identify recovery.json")
    if payload["kind"] == "stage1-training":
        payload["checkpoint_search_pattern"] = str(volume_root / "savedModel" / str(run_id) / "checkpoint-*")
    else:
        payload["result_search_pattern"] = str(expected_evaluation_root / "validation-benchmark.json")
json.dump(payload, sys.stdout, sort_keys=True)' \
        "${POD_ID}" "${DELAY_SECONDS}" "${LIFECYCLE_KEY}" \
        "${RUNPOD_GUARD_VOLUME_ROOT}" "${RUNPOD_GUARD_RUN_ID}" \
        2>/dev/null || true)"
    if [[ -n "${timeout_json}" ]] \
        && printf '%s' "${timeout_json}" | runpod_guard_s3 s3 cp - \
            "s3://${volume_id}/lifecycle/runs/${RUNPOD_GUARD_RUN_ID}/timeouts/${POD_ID}.json" \
            --only-show-errors >/dev/null 2>&1; then
        printf '[%s] published per-Pod timed_out audit before hard-limit termination\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
        if [[ "${matching_run_lifecycle_observed}" == "1" ]]; then
            if printf '%s' "${timeout_json}" | runpod_guard_s3 s3 cp - \
                "s3://${volume_id}/${LIFECYCLE_KEY}" \
                --only-show-errors >/dev/null 2>&1; then
                printf '[%s] published owned timed_out singleton before hard-limit termination\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
            else
                printf '[%s] unable to publish owned timed_out singleton; termination still required\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
            fi
        else
            printf '[%s] singleton timed_out update skipped: no current same-Pod same-run lifecycle ownership\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
        fi
    else
        printf '[%s] unable to publish per-Pod timed_out audit; termination still required\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
    fi
fi

printf '[%s] termination triggered by %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${termination_reason}" >> "${LOG_FILE}"

attempt=1
while [[ ${attempt} -le ${MAX_ATTEMPTS} ]]; do
    printf '[%s] terminate attempt %d/%d for Pod %s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${attempt}" "${MAX_ATTEMPTS}" "${POD_ID}" \
        >> "${LOG_FILE}"
    if bash "${RUNPODCTL_WRAPPER}" pod delete "${POD_ID}" >> "${LOG_FILE}" 2>&1; then
        printf '[%s] terminate request succeeded\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
            >> "${LOG_FILE}"
        exit 0
    fi
    attempt=$((attempt + 1))
    if [[ ${attempt} -le ${MAX_ATTEMPTS} ]]; then
        sleep "${RETRY_SECONDS}"
    fi
done

printf '[%s] all terminate attempts failed; manual intervention is required\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
exit 1
