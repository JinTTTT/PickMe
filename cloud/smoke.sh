#!/usr/bin/env bash
set -euo pipefail
export DATASET_NAME="${SMOKE_DATASET_NAME:-cloud_smoke_$(date -u +%Y%m%d_%H%M%S)}"
export NUM_EPISODES=3
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$PROJECT_DIR"
printf '%s\n' "$DATASET_ROOT" > logs/smoke-dataset.txt
bash collect.sh
jq -e '.total_episodes == 3 and .total_frames > 0 and .fps == 30' "$DATASET_ROOT/meta/info.json"
bash prepare_dataset.sh
printf 'SMOKE_PASSED dataset=%s\n' "$DATASET_ROOT"
