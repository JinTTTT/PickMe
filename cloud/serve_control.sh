#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$PROJECT_DIR"
if [ -z "$CHECKPOINT" ]; then
    [ -f logs/control-checkpoint.txt ] || { echo 'No trained checkpoint selected yet. Finish training first.' >&2; exit 1; }
    export CHECKPOINT=$(cat logs/control-checkpoint.txt)
fi
exec bash start_server.sh
