#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${HOME}/Lianjunyang/SiganusMorph"
ENV_FILE="${PROJECT_ROOT}/deployment/remote_worker/worker_runtime.env"
PYTHON="${PROJECT_ROOT}/.venv-worker/bin/python"

cd "${PROJECT_ROOT}"
set -a
source "${ENV_FILE}"
set +a

exec nice -n 10 "${PYTHON}" scripts/remote_measurement_worker.py "$@"
