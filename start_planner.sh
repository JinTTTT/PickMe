#!/bin/bash
# Terminal: Cosmos-Reason2 high-level planner for eval_in_sim.py --planner_host.
# Runs in the GR00T uv venv (has transformers + access to the gated Cosmos-Reason2-2B weights).
#   ./start_planner.sh                      # port 5556, nvidia/Cosmos-Reason2-2B
#   COSMOS_MODEL=nvidia/Cosmos-Reason2-8B ./start_planner.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
gr00t_activate
echo "Starting Cosmos planner: model=$COSMOS_MODEL port=$PLANNER_PORT"
exec uv run python "$PROJECT_DIR/scripts/cosmos_planner_server.py" \
    --model "$COSMOS_MODEL" --host 0.0.0.0 --port "$PLANNER_PORT" "$@"
