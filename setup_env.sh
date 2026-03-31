#!/usr/bin/env bash
# One-time / repeatable local setup: venv, dependencies, gcloud auth.
# Usage: bash setup_env.sh
# On Windows: Git Bash or WSL.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ -z "${PYTHON:-}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
  else
    PYTHON="python"
  fi
fi

echo "Using interpreter: $PYTHON"
"$PYTHON" -m venv .venv

# shellcheck source=/dev/null
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo ">>> Application Default Credentials (used by google-cloud-* SDKs)"
gcloud auth application-default login

echo ""
echo ">>> Active gcloud account and project (Vertex uses project from config or .env)"
gcloud auth list --filter=status:ACTIVE --format="value(account)"
if [[ -n "${GCP_PROJECT_ID:-}" ]]; then
  gcloud config set project "$GCP_PROJECT_ID"
else
  echo "Tip: export GCP_PROJECT_ID=your-project-id then re-run to set the default project."
fi

echo ""
echo "Done. Activate the venv with: source .venv/bin/activate (Linux/macOS/Git Bash)"
echo "Or on Windows CMD: .venv\\Scripts\\activate.bat"
