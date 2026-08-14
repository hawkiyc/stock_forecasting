#!/usr/bin/env bash

# Populate the persistent Hugging Face cache before starting expensive training.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-/runpod-volume}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/ts_multimodal_LLM}"
POETRY_VERSION="${POETRY_VERSION:-2.4.0}"
RUNPOD_CONFIG="${RUNPOD_CONFIG:-configs/stage1_kronos_base_lora.yaml}"
POETRY_BIN="${RUNPOD_POETRY_BIN:-${NETWORK_VOLUME_ROOT}/tools/poetry/${POETRY_VERSION}/bin/poetry}"
PROJECT_VENV="${PROJECT_ROOT}/.venv"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
export RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404}"
RUNPOD_EXPECTED_TORCH_VERSION="${RUNPOD_EXPECTED_TORCH_VERSION:-2.9.1+cu128}"
RUNPOD_EXPECTED_CUDA_PREFIX="${RUNPOD_EXPECTED_CUDA_PREFIX:-12.8}"
RUNPOD_EXPECTED_UBUNTU_VERSION="${RUNPOD_EXPECTED_UBUNTU_VERSION:-24.04}"
RUNPOD_ROLE="${RUNPOD_ROLE:-gpu-train}"
HF_PREFETCH_MAX_WORKERS="${HF_PREFETCH_MAX_WORKERS:-4}"
HF_PREFETCH_MAX_ATTEMPTS="${HF_PREFETCH_MAX_ATTEMPTS:-3}"
HF_PREFETCH_RETRY_BACKOFF_SECONDS="${HF_PREFETCH_RETRY_BACKOFF_SECONDS:-15}"
export HF_PREFETCH_MAX_ATTEMPTS HF_PREFETCH_RETRY_BACKOFF_SECONDS
RUNTIME_VERIFIER="${PROJECT_ROOT}/scripts/verify_runpod_runtime.py"

if [[ -z "${RUNPOD_POD_ID:-}" && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "Model prefetch must run inside a RunPod Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "cpu-prep" && "${RUNPOD_ROLE}" != "gpu-train" ]]; then
    echo "RUNPOD_ROLE must be cpu-prep or gpu-train" >&2
    exit 2
fi
if [[ ! "${HF_PREFETCH_MAX_WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "HF_PREFETCH_MAX_WORKERS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${HF_PREFETCH_MAX_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "HF_PREFETCH_MAX_ATTEMPTS must be a positive integer" >&2
    exit 2
fi
if [[ ! "${HF_PREFETCH_RETRY_BACKOFF_SECONDS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "HF_PREFETCH_RETRY_BACKOFF_SECONDS must be non-negative" >&2
    exit 2
fi
CUDA_DEVICE_ARGUMENTS=()
if [[ "${RUNPOD_ROLE}" == "gpu-train" ]]; then
    CUDA_DEVICE_ARGUMENTS+=(--require-cuda-device)
fi

runpod_validate_absolute_path "${NETWORK_VOLUME_ROOT}" NETWORK_VOLUME_ROOT
case "${NETWORK_VOLUME_ROOT}" in
    /workspace|/workspace/*)
        echo "NETWORK_VOLUME_ROOT must never use /workspace" >&2
        exit 2
        ;;
esac
runpod_validate_path_in_root \
    "${PROJECT_ROOT}" "${NETWORK_VOLUME_ROOT}" PROJECT_ROOT NETWORK_VOLUME_ROOT
runpod_validate_path_in_root \
    "${POETRY_BIN}" "${NETWORK_VOLUME_ROOT}" POETRY_BIN NETWORK_VOLUME_ROOT
if [[ ! -x "${POETRY_BIN}" ]]; then
    echo "Persistent Poetry not found; run scripts/setup_runpod_environment.sh first" >&2
    exit 127
fi
if [[ ! -x "${PROJECT_VENV}/bin/python" ]]; then
    echo "Persistent project .venv not found; run scripts/setup_runpod_environment.sh first" >&2
    exit 127
fi
if ! grep -Eq '^include-system-site-packages = true$' "${PROJECT_VENV}/pyvenv.cfg"; then
    echo "Persistent project .venv does not inherit RunPod image packages" >&2
    exit 3
fi
if [[ "${RUNPOD_CONFIG}" == /* ]]; then
    CONFIG_PATH="${RUNPOD_CONFIG}"
else
    if [[ ! "${RUNPOD_CONFIG}" =~ ^[A-Za-z0-9._/-]+$ \
        || "/${RUNPOD_CONFIG}/" == *"/../"* \
        || "/${RUNPOD_CONFIG}/" == *"/./"* \
        || "${RUNPOD_CONFIG}" == *"//"* ]]; then
        echo "RUNPOD_CONFIG must be a safe relative path" >&2
        exit 2
    fi
    CONFIG_PATH="${PROJECT_ROOT}/${RUNPOD_CONFIG}"
fi
runpod_validate_path_in_root \
    "${CONFIG_PATH}" "${NETWORK_VOLUME_ROOT}" CONFIG_PATH NETWORK_VOLUME_ROOT
if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Config not found: ${CONFIG_PATH}" >&2
    exit 2
fi

# The image may export cache paths under ephemeral /workspace; never inherit them.
export HF_HOME="${NETWORK_VOLUME_ROOT}/cache/huggingface"
export MODEL_CACHE_MANIFEST="${NETWORK_VOLUME_ROOT}/cache/hf-models.json"
export RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
# Keep Xet range requests serialized while allowing repository file prefetch concurrency.
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-0}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-1}"
export HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY="${HF_XET_RECONSTRUCT_WRITE_SEQUENTIALLY:-1}"
for path_name in HF_HOME MODEL_CACHE_MANIFEST TMPDIR; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
mkdir -p "${HF_HOME}" "$(dirname "${MODEL_CACHE_MANIFEST}")" "${TMPDIR}"
if [[ "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    if [[ "${RUNPOD_PYTHON_BIN}" != /* || ! -x "${RUNPOD_PYTHON_BIN}" ]]; then
        echo "RunPod image Python not found: ${RUNPOD_PYTHON_BIN}" >&2
        exit 127
    fi
    case "${RUNPOD_PYTHON_BIN}" in
        "${NETWORK_VOLUME_ROOT}"/*|/workspace/*)
            echo "RUNPOD_PYTHON_BIN must come from the RunPod image" >&2
            exit 2
            ;;
    esac
    if [[ ! -f "${RUNTIME_VERIFIER}" ]]; then
        echo "RunPod runtime verifier not found: ${RUNTIME_VERIFIER}" >&2
        exit 2
    fi
    IMAGE_RUNTIME_METADATA="${TMPDIR}/prefetch-image-runtime.json"
    "${RUNPOD_PYTHON_BIN}" "${RUNTIME_VERIFIER}" \
        --role image \
        --expected-torch "${RUNPOD_EXPECTED_TORCH_VERSION}" \
        --expected-cuda-prefix "${RUNPOD_EXPECTED_CUDA_PREFIX}" \
        --expected-ubuntu "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
        --network-volume-root "${NETWORK_VOLUME_ROOT}" \
        "${CUDA_DEVICE_ARGUMENTS[@]}" \
        --output "${IMAGE_RUNTIME_METADATA}"
    "${PROJECT_VENV}/bin/python" "${RUNTIME_VERIFIER}" \
        --role venv \
        --expected-torch "${RUNPOD_EXPECTED_TORCH_VERSION}" \
        --expected-cuda-prefix "${RUNPOD_EXPECTED_CUDA_PREFIX}" \
        --expected-ubuntu "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
        --network-volume-root "${NETWORK_VOLUME_ROOT}" \
        "${CUDA_DEVICE_ARGUMENTS[@]}" \
        --reference "${IMAGE_RUNTIME_METADATA}"
fi
cd "${PROJECT_ROOT}"
"${POETRY_BIN}" env use "${PROJECT_VENV}/bin/python"
printf 'Hugging Face prefetch policy: max_workers=%s, max_attempts=%s, retry_backoff=%ss, etag_timeout=%ss, download_timeout=%ss, disable_xet=%s, xet_range_gets=%s\n' \
    "${HF_PREFETCH_MAX_WORKERS}" "${HF_PREFETCH_MAX_ATTEMPTS}" \
    "${HF_PREFETCH_RETRY_BACKOFF_SECONDS}" \
    "${HF_HUB_ETAG_TIMEOUT}" "${HF_HUB_DOWNLOAD_TIMEOUT}" \
    "${HF_HUB_DISABLE_XET}" "${HF_XET_NUM_CONCURRENT_RANGE_GETS}"
set +e
"${POETRY_BIN}" run python -m fin_ts_multimodal.cli.prefetch_models \
    --config "${RUNPOD_CONFIG}" \
    --max-workers "${HF_PREFETCH_MAX_WORKERS}" "$@"
PREFETCH_EXIT_CODE=$?
set -e
if [[ ${PREFETCH_EXIT_CODE} -eq 137 ]]; then
    echo "Model prefetch was killed by the Pod memory limit; use a CPU Pod with more RAM" >&2
fi
exit "${PREFETCH_EXIT_CODE}"
