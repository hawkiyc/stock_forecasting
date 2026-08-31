#!/usr/bin/env bash

# Deploy the restricted Cloud Run relay and activate it only after live verification.
set -Eeuo pipefail
set +x
umask 077

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
ENV_TEMPLATE="${PROJECT_ROOT}/.env.example"
ENV_HELPER="${SCRIPT_DIR}/update_runpod_env.py"
RELAY_SOURCE="${PROJECT_ROOT}/cloudrun/tpex-relay"
SERVICE_ACCOUNT_NAME="stock-forecasting-tpex-relay"
# shellcheck source=lib/runpod_project_env.sh
source "${SCRIPT_DIR}/lib/runpod_project_env.sh"

if [[ $# -ne 0 ]]; then
    echo "Usage: bash scripts/deploy_tpex_cloud_run_relay.sh" >&2
    exit 2
fi
for command_name in curl gcloud python3; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "${command_name} is required on the local control machine" >&2
        exit 127
    fi
done
if [[ ! -f "${RELAY_SOURCE}/package.json" || ! -f "${RELAY_SOURCE}/src/server.mjs" ]]; then
    echo "Cloud Run TPEx relay source is incomplete" >&2
    exit 2
fi

runpod_load_tpex_relay_deploy_env "${PROJECT_ROOT}"
export -n RUNPOD_API_KEY TPEX_PROXY_SHARED_SECRET
if [[ "${GCP_CLOUD_RUN_REGION}" != "asia-east1" ]]; then
    echo "GCP_CLOUD_RUN_REGION must be asia-east1 for the TPEx relay" >&2
    exit 2
fi

active_account="$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null)"
if [[ -z "${active_account}" || "${active_account}" == *$'\n'* ]]; then
    echo "Exactly one active gcloud account is required; run gcloud auth login" >&2
    exit 2
fi
verified_project="$(gcloud projects describe "${GCP_PROJECT_ID}" \
    --format='value(projectId)' 2>/dev/null || true)"
if [[ "${verified_project}" != "${GCP_PROJECT_ID}" ]]; then
    echo "The active gcloud account cannot access GCP_PROJECT_ID" >&2
    exit 2
fi
project_number="$(gcloud projects describe "${GCP_PROJECT_ID}" \
    --format='value(projectNumber)')"
if [[ ! "${project_number}" =~ ^[1-9][0-9]*$ ]]; then
    echo "GCP did not return a valid project number" >&2
    exit 2
fi

RUNPOD_API_KEY="${RUNPOD_API_KEY}" \
    python3 "${SCRIPT_DIR}/create_runpod_tpex_proxy_secret.py" --check-access

printf 'Enabling the APIs required for source deployment and Secret Manager.\n'
gcloud services enable \
    run.googleapis.com \
    cloudbuild.googleapis.com \
    artifactregistry.googleapis.com \
    secretmanager.googleapis.com \
    iam.googleapis.com \
    --project="${GCP_PROJECT_ID}" \
    --quiet

build_service_account="${project_number}-compute@developer.gserviceaccount.com"
if ! gcloud iam service-accounts describe "${build_service_account}" \
    --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
    echo "The default source-build service account is unavailable after API enablement" >&2
    exit 2
fi
gcloud projects add-iam-policy-binding "${GCP_PROJECT_ID}" \
    --member="serviceAccount:${build_service_account}" \
    --role="roles/run.builder" \
    --condition=None \
    --quiet >/dev/null

service_account_email="${SERVICE_ACCOUNT_NAME}@${GCP_PROJECT_ID}.iam.gserviceaccount.com"
existing_service_account="$(gcloud iam service-accounts list \
    --project="${GCP_PROJECT_ID}" \
    --filter="email=${service_account_email}" \
    --format='value(email)')"
if [[ -z "${existing_service_account}" ]]; then
    gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
        --project="${GCP_PROJECT_ID}" \
        --display-name="Stock forecasting TPEx relay" \
        --description="Dedicated runtime identity for the restricted TPEx Cloud Run relay"
elif [[ "${existing_service_account}" != "${service_account_email}" ]]; then
    echo "Unable to resolve the dedicated TPEx relay service account safely" >&2
    exit 2
fi

if ! gcloud secrets describe "${GCP_TPEX_RELAY_SECRET}" \
    --project="${GCP_PROJECT_ID}" >/dev/null 2>&1; then
    gcloud secrets create "${GCP_TPEX_RELAY_SECRET}" \
        --project="${GCP_PROJECT_ID}" \
        --replication-policy=automatic \
        --labels=application=stock-forecasting,component=tpex-relay
fi
secret_version_resource="$({
    printf '%s' "${TPEX_PROXY_SHARED_SECRET}"
} | gcloud secrets versions add "${GCP_TPEX_RELAY_SECRET}" \
    --project="${GCP_PROJECT_ID}" \
    --data-file=- \
    --format='value(name)')"
secret_version="${secret_version_resource##*/}"
if [[ ! "${secret_version}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Secret Manager did not return a numeric secret version" >&2
    exit 2
fi
gcloud secrets add-iam-policy-binding "${GCP_TPEX_RELAY_SECRET}" \
    --project="${GCP_PROJECT_ID}" \
    --member="serviceAccount:${service_account_email}" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet >/dev/null

printf 'Deploying the restricted TPEx relay to Cloud Run asia-east1.\n'
gcloud run deploy "${GCP_TPEX_RELAY_SERVICE}" \
    --project="${GCP_PROJECT_ID}" \
    --region="${GCP_CLOUD_RUN_REGION}" \
    --source="${RELAY_SOURCE}" \
    --service-account="${service_account_email}" \
    --set-secrets="TPEX_PROXY_SHARED_SECRET=${GCP_TPEX_RELAY_SECRET}:${secret_version}" \
    --set-env-vars="TPEX_RELAY_REGION=${GCP_CLOUD_RUN_REGION}" \
    --execution-environment=gen2 \
    --ingress=all \
    --allow-unauthenticated \
    --cpu=1 \
    --memory=512Mi \
    --concurrency=1 \
    --min=0 \
    --max=1 \
    --timeout=60s \
    --cpu-throttling \
    --cpu-boost \
    --no-session-affinity \
    --deploy-health-check \
    --quiet

TPEX_PROXY_URL="$(gcloud run services describe "${GCP_TPEX_RELAY_SERVICE}" \
    --project="${GCP_PROJECT_ID}" \
    --region="${GCP_CLOUD_RUN_REGION}" \
    --format='value(status.url)')"
export TPEX_PROXY_URL
if [[ ! "${TPEX_PROXY_URL}" =~ ^https://[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?(\.[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?)*\.run\.app/?$ ]]; then
    echo "Cloud Run did not publish an approved run.app service URL" >&2
    exit 2
fi

if TPEX_PROXY_SHARED_SECRET="${TPEX_PROXY_SHARED_SECRET}" \
    bash "${SCRIPT_DIR}/verify_tpex_cloud_run_relay.sh" --environment; then
    :
else
    verification_status=$?
    printf '%s\n' \
        'Cloud Run service deployment completed, but live TPEx verification failed.' \
        'This run did not create/update the RunPod TPEx Secret or local TPEX_PROXY_URL.' \
        >&2
    exit "${verification_status}"
fi

if RUNPOD_TPEX_PROXY_SECRET_NAME="$(
    RUNPOD_API_KEY="${RUNPOD_API_KEY}" \
    TPEX_PROXY_SHARED_SECRET="${TPEX_PROXY_SHARED_SECRET}" \
        python3 "${SCRIPT_DIR}/create_runpod_tpex_proxy_secret.py"
)"; then
    :
else
    secret_status=$?
    printf '%s\n' \
        'Cloud Run relay verification passed, but RunPod Secret creation failed.' \
        'This run did not activate the new relay URL or Secret reference in local .env.' \
        >&2
    exit "${secret_status}"
fi

if {
    printf 'TPEX_PROXY_URL\0%s\0' "${TPEX_PROXY_URL}"
    printf 'RUNPOD_TPEX_PROXY_SECRET_NAME\0%s\0' \
        "${RUNPOD_TPEX_PROXY_SECRET_NAME}"
} | python3 "${ENV_HELPER}" apply-null \
    --env-file "${ENV_FILE}" \
    --template "${ENV_TEMPLATE}" \
    --require-key TPEX_PROXY_URL \
    --require-key RUNPOD_TPEX_PROXY_SECRET_NAME; then
    :
else
    activation_status=$?
    printf '%s\n' \
        'Cloud Run and RunPod Secret setup passed, but local relay activation failed.' \
        'The previous local TPEX_PROXY_URL and Secret reference remain authoritative.' \
        >&2
    exit "${activation_status}"
fi

unset RUNPOD_API_KEY TPEX_PROXY_SHARED_SECRET
printf 'Deployed and verified restricted Cloud Run TPEx relay: %s\n' "${TPEX_PROXY_URL}"
printf 'Created RunPod Secret reference: %s\n' "${RUNPOD_TPEX_PROXY_SECRET_NAME}"
