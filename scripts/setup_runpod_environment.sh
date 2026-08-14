#!/usr/bin/env bash

# Install reproducible tooling and source dependencies onto the network volume.
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/runpod_paths.sh
source "${SCRIPT_DIR}/lib/runpod_paths.sh"

NETWORK_VOLUME_ROOT="${NETWORK_VOLUME_ROOT:-/runpod-volume}"
PROJECT_ROOT="${PROJECT_ROOT:-${NETWORK_VOLUME_ROOT}/stock_forecasting}"
POETRY_VERSION="${POETRY_VERSION:-2.4.0}"
POETRY_ROOT="${NETWORK_VOLUME_ROOT}/tools/poetry/${POETRY_VERSION}"
POETRY_BIN="${POETRY_ROOT}/bin/poetry"
RUNPOD_PYTHON_BIN="${RUNPOD_PYTHON_BIN:-/usr/local/bin/python}"
export RUNPOD_IMAGE="${RUNPOD_IMAGE:-runpod/pytorch:1.0.7-cu1281-torch291-ubuntu2404}"
RUNPOD_EXPECTED_TORCH_VERSION="${RUNPOD_EXPECTED_TORCH_VERSION:-2.9.1+cu128}"
RUNPOD_EXPECTED_CUDA_PREFIX="${RUNPOD_EXPECTED_CUDA_PREFIX:-12.8}"
RUNPOD_EXPECTED_UBUNTU_VERSION="${RUNPOD_EXPECTED_UBUNTU_VERSION:-24.04}"
RUNPOD_ROLE="${RUNPOD_ROLE:-gpu-train}"
KRONOS_REPOSITORY="https://github.com/shiyu-coder/Kronos.git"
KRONOS_COMMIT="67b630e67f6a18c9e9be918d9b4337c960db1e9a"
KRONOS_ROOT="${KRONOS_ROOT:-${NETWORK_VOLUME_ROOT}/third_party/Kronos}"
PROJECT_VENV="${PROJECT_ROOT}/.venv"
RUNTIME_VERIFIER="${PROJECT_ROOT}/scripts/verify_runpod_runtime.py"
POETRY_OWNERSHIP_VERIFIER="${PROJECT_ROOT}/scripts/verify_poetry_runtime_ownership.py"
POETRY_LOCK_FILTER="${PROJECT_ROOT}/scripts/filter_runpod_poetry_lock.py"
IMAGE_RUNTIME_METADATA="${NETWORK_VOLUME_ROOT}/tmp/runpod-image-runtime.json"
VENV_RUNTIME_METADATA="${PROJECT_VENV}/runpod-runtime.json"
# The image and SSH session may provide ephemeral cache overrides; never inherit them.
export PIP_CACHE_DIR="${NETWORK_VOLUME_ROOT}/cache/pip"
export TMPDIR="${NETWORK_VOLUME_ROOT}/tmp"
POETRY_LOCK_WORK_ROOT="${TMPDIR}/poetry-lock-install-$$"
POETRY_LOCK_INSTALL="${POETRY_LOCK_WORK_ROOT}/poetry.lock.install"
POETRY_LOCK_BACKUP="${POETRY_LOCK_WORK_ROOT}/poetry.lock.full"

if [[ -z "${RUNPOD_POD_ID:-}" \
    && "${RUNPOD_SETUP_DRY_RUN:-0}" != "1" \
    && "${RUNPOD_TEST_MODE:-0}" != "1" ]]; then
    echo "RunPod environment setup must run inside a RunPod Pod" >&2
    exit 2
fi
if [[ "${RUNPOD_ROLE}" != "cpu-prep" && "${RUNPOD_ROLE}" != "gpu-train" ]]; then
    echo "RUNPOD_ROLE must be cpu-prep or gpu-train" >&2
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
for path_name in \
    PROJECT_ROOT POETRY_ROOT POETRY_BIN KRONOS_ROOT PROJECT_VENV RUNTIME_VERIFIER \
    POETRY_OWNERSHIP_VERIFIER POETRY_LOCK_FILTER IMAGE_RUNTIME_METADATA \
    VENV_RUNTIME_METADATA POETRY_LOCK_WORK_ROOT POETRY_LOCK_INSTALL POETRY_LOCK_BACKUP \
    PIP_CACHE_DIR TMPDIR; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    echo "pyproject.toml not found at ${PROJECT_ROOT}" >&2
    exit 2
fi
runpod_validate_absolute_path "${RUNPOD_PYTHON_BIN}" RUNPOD_PYTHON_BIN
if [[ ! -x "${RUNPOD_PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${RUNPOD_PYTHON_BIN}" >&2
    exit 127
fi
case "${RUNPOD_PYTHON_BIN}" in
    "${NETWORK_VOLUME_ROOT}"/*|/workspace/*)
        echo "RUNPOD_PYTHON_BIN must come from the RunPod image" >&2
        exit 2
        ;;
esac
if ! "${RUNPOD_PYTHON_BIN}" -c \
    'import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)'; then
    echo "The approved RunPod image environment requires Python 3.12" >&2
    exit 2
fi
if ! command -v git >/dev/null 2>&1; then
    echo "git is required to provision Kronos" >&2
    exit 127
fi
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

if [[ "${RUNPOD_SETUP_DRY_RUN:-0}" == "1" ]]; then
    printf 'Would install Poetry %s at %s\n' "${POETRY_VERSION}" "${POETRY_ROOT}"
    printf 'Would verify image Python %s with PyTorch %s and CUDA %s\n' \
        "${RUNPOD_PYTHON_BIN}" "${RUNPOD_EXPECTED_TORCH_VERSION}" \
        "${RUNPOD_EXPECTED_CUDA_PREFIX}"
    printf 'RunPod role: %s; physical CUDA device required: %s\n' \
        "${RUNPOD_ROLE}" "$([[ "${RUNPOD_ROLE}" == "gpu-train" ]] && printf yes || printf no)"
    printf 'Would create a Python 3.12 system-site-packages venv at %s\n' "${PROJECT_VENV}"
    printf 'Would provision Kronos %s at commit %s\n' "${KRONOS_REPOSITORY}" "${KRONOS_COMMIT}"
    printf 'Would install locked project and dev dependencies for %s without replacing image PyTorch\n' \
        "${PROJECT_ROOT}"
    printf 'Would reject any PyTorch or CUDA package managed by Poetry\n'
    exit 0
fi

if [[ ! -f "${RUNTIME_VERIFIER}" ]]; then
    echo "RunPod runtime verifier not found: ${RUNTIME_VERIFIER}" >&2
    exit 2
fi
if [[ ! -f "${POETRY_OWNERSHIP_VERIFIER}" ]]; then
    echo "Poetry ownership verifier not found: ${POETRY_OWNERSHIP_VERIFIER}" >&2
    exit 2
fi
if [[ ! -f "${POETRY_LOCK_FILTER}" ]]; then
    echo "Poetry lock filter not found: ${POETRY_LOCK_FILTER}" >&2
    exit 2
fi
"${RUNPOD_PYTHON_BIN}" "${RUNTIME_VERIFIER}" \
    --role image \
    --expected-torch "${RUNPOD_EXPECTED_TORCH_VERSION}" \
    --expected-cuda-prefix "${RUNPOD_EXPECTED_CUDA_PREFIX}" \
    --expected-ubuntu "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}" \
    "${CUDA_DEVICE_ARGUMENTS[@]}" \
    --output "${IMAGE_RUNTIME_METADATA}"

if [[ -d "${KRONOS_ROOT}/.git" ]]; then
    CURRENT_COMMIT="$(git -C "${KRONOS_ROOT}" rev-parse HEAD)"
    CURRENT_ORIGIN="$(git -C "${KRONOS_ROOT}" remote get-url origin)"
    if [[ "${CURRENT_COMMIT}" != "${KRONOS_COMMIT}" ]]; then
        echo "Existing Kronos checkout is at ${CURRENT_COMMIT}; refusing to overwrite it" >&2
        exit 3
    fi
    if [[ "${CURRENT_ORIGIN}" != "${KRONOS_REPOSITORY}" ]]; then
        echo "Existing Kronos checkout has an unexpected origin; refusing to use it" >&2
        exit 3
    fi
elif [[ -e "${KRONOS_ROOT}" ]]; then
    echo "KRONOS_ROOT exists but is not the pinned Git checkout; refusing to overwrite it" >&2
    exit 3
else
    mkdir -p "$(dirname "${KRONOS_ROOT}")"
    git clone --no-checkout "${KRONOS_REPOSITORY}" "${KRONOS_ROOT}"
    git -C "${KRONOS_ROOT}" checkout --detach "${KRONOS_COMMIT}"
fi

if [[ ! -x "${POETRY_BIN}" ]]; then
    if [[ -x "${POETRY_ROOT}/bin/python" ]]; then
        printf 'Resuming fixed Poetry installation at %s\n' "${POETRY_ROOT}"
    elif [[ -e "${POETRY_ROOT}" ]]; then
        echo "Poetry destination exists but is incomplete; refusing to overwrite it" >&2
        exit 3
    else
        mkdir -p "$(dirname "${POETRY_ROOT}")"
        "${RUNPOD_PYTHON_BIN}" -m venv "${POETRY_ROOT}"
    fi
    "${POETRY_ROOT}/bin/python" -m pip install \
        --disable-pip-version-check "poetry==${POETRY_VERSION}"
fi

export POETRY_CACHE_DIR="${NETWORK_VOLUME_ROOT}/cache/pypoetry"
# Install from the lock view without resolving omitted image packages back in.
export POETRY_INSTALLER_RE_RESOLVE=false
export HF_HOME="${NETWORK_VOLUME_ROOT}/cache/huggingface"
export KRONOS_ROOT
export RUNPOD_VOLUME_ROOT="${NETWORK_VOLUME_ROOT}"
for path_name in POETRY_CACHE_DIR HF_HOME; do
    runpod_validate_path_in_root \
        "${!path_name}" "${NETWORK_VOLUME_ROOT}" "${path_name}" NETWORK_VOLUME_ROOT
done
mkdir -p "${POETRY_CACHE_DIR}" "${HF_HOME}"
if [[ ! -x "${PROJECT_VENV}/bin/python" ]]; then
    if [[ -e "${PROJECT_VENV}" ]]; then
        echo "Project .venv exists but is incomplete; refusing to overwrite it" >&2
        exit 3
    fi
    # Reuse the pinned RunPod image's CUDA-enabled PyTorch while Poetry manages the environment.
    "${RUNPOD_PYTHON_BIN}" -m venv --system-site-packages "${PROJECT_VENV}"
fi
if ! grep -Eq '^include-system-site-packages = true$' "${PROJECT_VENV}/pyvenv.cfg"; then
    echo "Project .venv must enable system-site-packages; refusing to use it" >&2
    exit 3
fi
"${PROJECT_VENV}/bin/python" "${RUNTIME_VERIFIER}" \
    --role venv \
    --expected-torch "${RUNPOD_EXPECTED_TORCH_VERSION}" \
    --expected-cuda-prefix "${RUNPOD_EXPECTED_CUDA_PREFIX}" \
    --expected-ubuntu "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}" \
    "${CUDA_DEVICE_ARGUMENTS[@]}" \
    --reference "${IMAGE_RUNTIME_METADATA}"
cd "${PROJECT_ROOT}"
"${POETRY_BIN}" env use "${PROJECT_VENV}/bin/python"
"${POETRY_BIN}" lock --no-interaction
"${RUNPOD_PYTHON_BIN}" "${POETRY_OWNERSHIP_VERIFIER}" \
    --project-root "${PROJECT_ROOT}" --allow-transitive-image-packages

# Keep the generated lock authoritative and use a temporary install view that
# leaves CUDA and PyTorch packages supplied by the RunPod image untouched.
mkdir -p "${POETRY_LOCK_WORK_ROOT}"
cp -p "${PROJECT_ROOT}/poetry.lock" "${POETRY_LOCK_BACKUP}"
restore_poetry_lock() {
    if [[ -f "${POETRY_LOCK_BACKUP}" ]]; then
        cp -p "${POETRY_LOCK_BACKUP}" "${PROJECT_ROOT}/poetry.lock"
    fi
}
trap restore_poetry_lock EXIT
"${RUNPOD_PYTHON_BIN}" "${POETRY_LOCK_FILTER}" \
    --input "${POETRY_LOCK_BACKUP}" --output "${POETRY_LOCK_INSTALL}"
cp -p "${POETRY_LOCK_INSTALL}" "${PROJECT_ROOT}/poetry.lock"
"${POETRY_BIN}" install --no-interaction
restore_poetry_lock
trap - EXIT
"${PROJECT_VENV}/bin/python" -m pip check
if ! "${PROJECT_VENV}/bin/python" -c \
    'import numpy as np; print(f"Verified project NumPy: {np.__version__}")'; then
    echo "Persistent Poetry environment is missing NumPy; dependency installation is incomplete" >&2
    exit 3
fi
"${PROJECT_VENV}/bin/python" "${RUNTIME_VERIFIER}" \
    --role venv \
    --expected-torch "${RUNPOD_EXPECTED_TORCH_VERSION}" \
    --expected-cuda-prefix "${RUNPOD_EXPECTED_CUDA_PREFIX}" \
    --expected-ubuntu "${RUNPOD_EXPECTED_UBUNTU_VERSION}" \
    --network-volume-root "${NETWORK_VOLUME_ROOT}" \
    "${CUDA_DEVICE_ARGUMENTS[@]}" \
    --reference "${IMAGE_RUNTIME_METADATA}" \
    --output "${VENV_RUNTIME_METADATA}"
printf 'RunPod environment is ready. Poetry: %s\n' "${POETRY_BIN}"
printf 'Verified runtime metadata: %s\n' "${VENV_RUNTIME_METADATA}"
