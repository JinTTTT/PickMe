#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$PROJECT_DIR"
mkdir -p logs
exec 9>logs/learned-control.lock
flock -w 15 9 || { echo 'A learned controller is already running; press Ctrl-C in its terminal before sending another command.' >&2; exit 1; }
export DISPLAY=:99
xdpyinfo -display "$DISPLAY" >/dev/null
scene="${CONTROL_SCENE:-cube}"
task_args=()
[ "$#" -eq 0 ] || task_args=(--task_text "$*")
echo "Scene: $scene | GR00T/Cosmos instruction: ${*:-<scene default>}"
exec "$ISAACLAB_PY" -u scripts/eval_in_sim.py --viz kit --scene "$scene" --num_episodes "${CONTROL_EPISODES:-1}" --seed "${CONTROL_SEED:-0}" --keep_open \
    "${task_args[@]}" --policy_host "$POLICY_HOST" --policy_port "$POLICY_PORT" \
    --results_file logs/live-control.json --record_dir logs/live-control
