#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$PROJECT_DIR"
export CHECKPOINT=${1:?Pass the checkpoint directory}
[ -f "$CHECKPOINT/trainer_state.json" ]
step=${CHECKPOINT##*checkpoint-}
case "$step" in ''|*[!0-9]*) echo 'Expected checkpoint-N directory'; exit 1 ;; esac
exec 9>logs/checkpoint-evaluation.lock
flock -n 9 || { echo 'Another checkpoint evaluation is active'; exit 1; }
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 18000 ]; do
    echo 'Waiting for enough GPU memory to evaluate the checkpoint...'
    sleep 15
done
[ "$(ss -lnt 'sport = :5557' | wc -l)" -eq 1 ] || { echo 'Port 5557 is already in use'; exit 1; }
session="control-eval-server-$step"
cleanup() { tmux kill-session -t "=$session" 2>/dev/null || true; }
trap cleanup EXIT
printf -v launch '%q ' env CHECKPOINT="$CHECKPOINT" POLICY_PORT=5557 bash "$PROJECT_DIR/start_server.sh"
printf -v server_log '%q' "$PROJECT_DIR/logs/eval-server-$step.log"
tmux new-session -d -s "$session" "$launch > $server_log 2>&1"
ready=0
for attempt in {1..120}; do
    tmux has-session -t "=$session" 2>/dev/null || { tail -n 30 "logs/eval-server-$step.log"; exit 1; }
    if [ "$(ss -lnt 'sport = :5557' | wc -l)" -gt 1 ]; then ready=1; break; fi
    sleep 1
done
[ "$ready" = 1 ] || { echo 'Policy server did not become ready'; exit 1; }
POLICY_PORT=5557 bash eval.sh --num_episodes 3 --seed 1001 \
    --results_file "logs/eval-$step.json" --record_dir "logs/eval-$step"
jq --arg checkpoint "$CHECKPOINT" '. + {checkpoint: $checkpoint}' "logs/eval-$step.json" > "logs/eval-$step.tmp"
mv "logs/eval-$step.tmp" "logs/eval-$step.json"
score=$(jq -er '.successes' "logs/eval-$step.json")
best=-1
[ ! -f logs/control-best.json ] || best=$(jq -er '.successes' logs/control-best.json)
if [ "$score" -ge "$best" ]; then
    cp "logs/eval-$step.json" logs/control-best.json
    if [ "$score" -gt 0 ]; then printf '%s\n' "$CHECKPOINT" > logs/control-checkpoint.txt; fi
fi
printf 'CHECKPOINT_EVALUATED step=%s successes=%s/3\n' "$step" "$score"
