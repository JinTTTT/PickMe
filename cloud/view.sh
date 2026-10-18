#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
export DISPLAY=:99
cd "$PROJECT_DIR"
exec "$ISAACLAB_PY" -u scripts/view_env.py --viz kit --run_policy 600 --keep_open "$@"
