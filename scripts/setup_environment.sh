#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/environment.yml"
ENV_NAME="${PINN_ENV_NAME:-pinn-framework-env}"
CONDA_BIN="${CONDA_BIN:-conda}"

if ! command -v "${CONDA_BIN}" >/dev/null 2>&1; then
  echo "conda was not found. Install Miniconda/Mambaforge first, then rerun this script." >&2
  exit 1
fi

if "${CONDA_BIN}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Updating existing conda environment: ${ENV_NAME}"
  "${CONDA_BIN}" env update -n "${ENV_NAME}" -f "${ENV_FILE}" --prune
else
  echo "Creating conda environment from ${ENV_FILE}"
  "${CONDA_BIN}" env create -f "${ENV_FILE}"
fi

echo "Checking environment imports..."
"${CONDA_BIN}" run -n "${ENV_NAME}" python "${PROJECT_ROOT}/scripts/check_environment.py"

if [[ "${PINN_SKIP_SMOKE:-0}" != "1" ]]; then
  echo "Running operator smoke training..."
  "${CONDA_BIN}" run -n "${ENV_NAME}" python "${PROJECT_ROOT}/scripts/check_environment.py" --smoke
fi

cat <<EOF

Environment is ready.

Activate it with:
  conda activate ${ENV_NAME}

Train the default operator with:
  python scripts/train_operator.py --config configs/train_operator_config.yaml
EOF
