#!/usr/bin/env bash

# Upload an explicit Stage 1 source manifest to the RunPod network volume.
set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
S3_WRAPPER="${SCRIPT_DIR}/runpod_s3_project.sh"
READINESS_HELPER="${SCRIPT_DIR}/runpod_readiness.py"
CODE_MARKER_KEY="lifecycle/stage1/code.json"

# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"
runpod_load_s3_env "${LOCAL_PROJECT_ROOT}"

MODE=dry-run
if [[ $# -gt 1 ]]; then
    echo "Usage: sync_project_to_runpod_volume.sh [--dry-run|--apply]" >&2
    exit 2
fi
if [[ $# -eq 1 ]]; then
    case "$1" in
        --dry-run) MODE=dry-run ;;
        --apply) MODE=apply ;;
        *)
            echo "Usage: sync_project_to_runpod_volume.sh [--dry-run|--apply]" >&2
            exit 2
            ;;
    esac
fi

if [[ ! "${RUNPOD_NETWORK_VOLUME_ID}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "RUNPOD_NETWORK_VOLUME_ID contains invalid characters" >&2
    exit 2
fi
if [[ ! -f "${S3_WRAPPER}" || ! -r "${S3_WRAPPER}" ]]; then
    echo "RunPod S3 wrapper is unavailable: ${S3_WRAPPER}" >&2
    exit 127
fi
if [[ ! -f "${READINESS_HELPER}" ]]; then
    echo "RunPod readiness helper is missing: ${READINESS_HELPER}" >&2
    exit 127
fi
if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required to create the source readiness manifest" >&2
    exit 127
fi

REMOTE_PROJECT_DIR="${RUNPOD_REMOTE_PROJECT_DIR:-ts_multimodal_LLM}"
if [[ ! "${REMOTE_PROJECT_DIR}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "RUNPOD_REMOTE_PROJECT_DIR contains invalid characters" >&2
    exit 2
fi

MANIFEST_RELATIVE=()
MANIFEST_ABSOLUTE=()
MANIFEST_SIZE=()
TOTAL_BYTES=0
MAX_FILE_BYTES=$((10 * 1024 * 1024))

append_manifest_file() {
    local relative_path="$1"
    local absolute_path="${LOCAL_PROJECT_ROOT}/${relative_path}"
    local file_size=""

    if [[ ! "${relative_path}" =~ ^[A-Za-z0-9._/-]+$ ]]; then
        echo "Manifest path contains unsupported characters: ${relative_path}" >&2
        exit 2
    fi
    if [[ ! -f "${absolute_path}" || -L "${absolute_path}" ]]; then
        echo "Manifest file is missing or is a symlink: ${relative_path}" >&2
        exit 2
    fi

    file_size="$(wc -c < "${absolute_path}" | tr -d '[:space:]')"
    if [[ ! "${file_size}" =~ ^[0-9]+$ || "${file_size}" -gt "${MAX_FILE_BYTES}" ]]; then
        echo "Manifest file exceeds the 10 MB safety limit: ${relative_path}" >&2
        exit 2
    fi

    MANIFEST_RELATIVE+=("${relative_path}")
    MANIFEST_ABSOLUTE+=("${absolute_path}")
    MANIFEST_SIZE+=("${file_size}")
    TOTAL_BYTES=$((TOTAL_BYTES + file_size))
}

# Poetry lock resolution is intentionally remote-only.  Do not upload or
# fingerprint poetry.lock because the remote Python/platform determines it.
for root_file in README.md pyproject.toml; do
    append_manifest_file "${root_file}"
done

while IFS= read -r absolute_path; do
    append_manifest_file "${absolute_path#${LOCAL_PROJECT_ROOT}/}"
done < <(
    {
        find "${LOCAL_PROJECT_ROOT}/src" -type f -name '*.py' -print
        find "${LOCAL_PROJECT_ROOT}/configs" -type f \( -name '*.yaml' -o -name '*.yml' \) -print
        find "${LOCAL_PROJECT_ROOT}/scripts" -type f \( -name '*.sh' -o -name '*.py' \) -print
        find "${LOCAL_PROJECT_ROOT}/tests" -type f -name '*.py' -print
    } | LC_ALL=C sort
)

DATA_PIPELINE_PATHS=(
    src/fin_ts_multimodal/cli/download_market_data.py
    src/fin_ts_multimodal/cli/prepare_data.py
    src/fin_ts_multimodal/cli/prefetch_models.py
    src/fin_ts_multimodal/cli/verify_stage1_data.py
    scripts/lib/runpod_paths.sh
    scripts/prefetch_hf_models.sh
    scripts/runpod_cpu_finalize.sh
    scripts/runpod_cpu_prepare.sh
)
while IFS= read -r absolute_path; do
    DATA_PIPELINE_PATHS+=("${absolute_path#${LOCAL_PROJECT_ROOT}/}")
done < <(find "${LOCAL_PROJECT_ROOT}/src/fin_ts_multimodal/data" -type f -name '*.py' -print | LC_ALL=C sort)

render_code_manifest() {
    local marker_state="$1"
    local manifest_command=(
        python3 "${READINESS_HELPER}" code-manifest
        --project-root "${LOCAL_PROJECT_ROOT}"
        --remote-project-dir "${REMOTE_PROJECT_DIR}"
        --state "${marker_state}"
    )
    local pipeline_path

    for pipeline_path in "${DATA_PIPELINE_PATHS[@]}"; do
        manifest_command+=(--pipeline-path "${pipeline_path}")
    done
    manifest_command+=("${MANIFEST_RELATIVE[@]}")
    "${manifest_command[@]}"
}

if [[ ${#MANIFEST_RELATIVE[@]} -eq 0 ]]; then
    echo "Upload manifest is empty" >&2
    exit 2
fi

# Scan every selected file for actual project secrets and high-confidence token formats.
ENV_FILE="$(runpod_project_env_file "${LOCAL_PROJECT_ROOT}")"
SENSITIVE_VALUES=()
while IFS= read -r env_line || [[ -n "${env_line}" ]]; do
    env_line="${env_line%$'\r'}"
    case "${env_line}" in
        export\ [A-Za-z_][A-Za-z0-9_]*=*) env_assignment="${env_line#export }" ;;
        [A-Za-z_][A-Za-z0-9_]*=*) env_assignment="${env_line}" ;;
        *) continue ;;
    esac
    env_key="${env_assignment%%=*}"
    case "${env_key}" in
        RUNPOD_EODHD_SECRET_NAME|RUNPOD_HF_SECRET_NAME|RUNPOD_WANDB_SECRET_NAME)
            # Secret identifiers are safe metadata; their values are never credentials.
            continue
            ;;
        *API_KEY|*TOKEN|*SECRET*|*PASSWORD*|RUNPOD_S3_ACCESS_KEY_ID)
            env_value="$(runpod_read_project_env_value "${ENV_FILE}" "${env_key}")"
            if [[ ${#env_value} -ge 8 ]]; then
                SENSITIVE_VALUES+=("${env_value}")
            fi
            ;;
    esac
done < "${ENV_FILE}"

for manifest_index in "${!MANIFEST_ABSOLUTE[@]}"; do
    manifest_path="${MANIFEST_ABSOLUTE[${manifest_index}]}"
    manifest_relative="${MANIFEST_RELATIVE[${manifest_index}]}"
    for secret_value in "${SENSITIVE_VALUES[@]}"; do
        if grep -Fq -- "${secret_value}" "${manifest_path}"; then
            echo "Security scan found a project secret in: ${manifest_relative}" >&2
            exit 3
        fi
    done
    if grep -Eq -- \
        '(-----BEGIN ([A-Z]+ )?PRIVATE KEY-----|rpa_[A-Za-z0-9_-]{16,}|rps_[A-Za-z0-9_-]{16,}|hf_[A-Za-z0-9]{20,}|sk-proj-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,})' \
        "${manifest_path}"; then
        echo "Security scan found a credential pattern in: ${manifest_relative}" >&2
        exit 3
    fi
done

printf 'Upload manifest: %d files, %d bytes\n' "${#MANIFEST_RELATIVE[@]}" "${TOTAL_BYTES}"
if [[ "${MODE}" == "dry-run" ]]; then
    printf 'Dry-run only; use --apply to upload.\n'
    render_code_manifest ready | python3 "${READINESS_HELPER}" check-code --marker - >/dev/null
    printf 'The code readiness manifest can be generated successfully.\n'
    for relative_path in "${MANIFEST_RELATIVE[@]}"; do
        printf '  %s\n' "${relative_path}"
    done
    exit 0
fi

bash "${S3_WRAPPER}" s3api head-bucket \
    --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
    >/dev/null

# Invalidate any older ready marker before changing remote source files.
render_code_manifest syncing | bash "${S3_WRAPPER}" s3 cp \
    - "s3://${RUNPOD_NETWORK_VOLUME_ID}/${CODE_MARKER_KEY}" \
    --only-show-errors

for manifest_index in "${!MANIFEST_ABSOLUTE[@]}"; do
    manifest_path="${MANIFEST_ABSOLUTE[${manifest_index}]}"
    manifest_relative="${MANIFEST_RELATIVE[${manifest_index}]}"
    printf '[%d/%d] Uploading %s\n' \
        "$((manifest_index + 1))" "${#MANIFEST_RELATIVE[@]}" "${manifest_relative}"
    bash "${S3_WRAPPER}" s3 cp \
        "${manifest_path}" \
        "s3://${RUNPOD_NETWORK_VOLUME_ID}/${REMOTE_PROJECT_DIR}/${manifest_relative}" \
        --only-show-errors
done

for manifest_index in "${!MANIFEST_ABSOLUTE[@]}"; do
    manifest_relative="${MANIFEST_RELATIVE[${manifest_index}]}"
    remote_size="$(bash "${S3_WRAPPER}" s3api head-object \
        --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
        --key "${REMOTE_PROJECT_DIR}/${manifest_relative}" \
        --query ContentLength \
        --output text)"
    if [[ "${remote_size}" != "${MANIFEST_SIZE[${manifest_index}]}" ]]; then
        echo "Remote size verification failed for: ${manifest_relative}" >&2
        exit 4
    fi
done

REMOTE_VENV_PREFIX="${REMOTE_PROJECT_DIR}/.venv/"

check_remote_key_safety() {
    local remote_key="$1"

    case "${remote_key}" in
        */.env|*/.env.*|*/artifacts/*|*/.ssh/*|*/id_ed25519*|\
        */id_rsa*|*.pem|*.key|*/credentials.json|*/secrets.json)
            echo "Remote safety check found a forbidden object: ${remote_key}" >&2
            exit 4
            ;;
    esac
}

scan_remote_prefix() {
    local prefix="$1"
    local listing=""
    local entries=""
    local entry_type=""
    local entry_value=""

    listing="$(bash "${S3_WRAPPER}" s3api list-objects-v2 \
        --bucket "${RUNPOD_NETWORK_VOLUME_ID}" \
        --prefix "${prefix}" \
        --delimiter / \
        --output json)"
    entries="$(printf '%s\n' "${listing}" | python3 -c '
import json
import sys

payload = json.load(sys.stdin)
if not isinstance(payload, dict):
    raise ValueError("ListObjectsV2 response must be an object")

for item in payload.get("Contents", []):
    key = item.get("Key")
    if not isinstance(key, str) or any(character in key for character in "\r\n\t"):
        raise ValueError("ListObjectsV2 returned an unsupported object key")
    print("key\t" + key)

for item in payload.get("CommonPrefixes", []):
    child_prefix = item.get("Prefix")
    if not isinstance(child_prefix, str) or any(
        character in child_prefix for character in "\r\n\t"
    ):
        raise ValueError("ListObjectsV2 returned an unsupported common prefix")
    print("prefix\t" + child_prefix)
')"

    while IFS=$'\t' read -r entry_type entry_value; do
        [[ -z "${entry_type}" ]] && continue
        case "${entry_type}" in
            key)
                check_remote_key_safety "${entry_value}"
                ;;
            prefix)
                if [[ "${entry_value}" == "${REMOTE_VENV_PREFIX}" ]]; then
                    # The RunPod setup creates this persistent environment from the approved image.
                    continue
                fi
                if [[ "${entry_value}" != "${prefix}"* \
                    || "${entry_value}" == "${prefix}" ]]; then
                    echo "Remote safety check returned an invalid child prefix: ${entry_value}" >&2
                    exit 4
                fi
                scan_remote_prefix "${entry_value}"
                ;;
            *)
                echo "Remote safety check returned an invalid entry type: ${entry_type}" >&2
                exit 4
                ;;
        esac
    done <<< "${entries}"
}

scan_remote_prefix "${REMOTE_PROJECT_DIR}/"

# Publish ready only after every source object and safety check has passed.
render_code_manifest ready | bash "${S3_WRAPPER}" s3 cp \
    - "s3://${RUNPOD_NETWORK_VOLUME_ID}/${CODE_MARKER_KEY}" \
    --only-show-errors

bash "${S3_WRAPPER}" s3 cp \
    "s3://${RUNPOD_NETWORK_VOLUME_ID}/${CODE_MARKER_KEY}" - \
    --only-show-errors \
    | python3 "${READINESS_HELPER}" check-code \
        --marker - \
        --project-root "${LOCAL_PROJECT_ROOT}" \
        >/dev/null

printf 'Upload, remote verification, and code readiness publication completed successfully.\n'
