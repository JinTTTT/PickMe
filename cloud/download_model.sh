#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
exec "$GR00T_REPO/.venv/bin/hf" download nvidia/GR00T-N1.7-3B --local-dir "$BASE_MODEL"
