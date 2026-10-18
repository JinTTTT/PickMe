#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$PROJECT_DIR"
while tmux has-session -t '=control-train' 2>/dev/null || tmux has-session -t '=control-validate' 2>/dev/null; do
    sleep 15
done
[ "$(cat logs/control-train.exit)" = 0 ]
[ "$(cat logs/control-validate.exit)" = 0 ]
jq -e '.successes > 0' logs/control-best.json
jq -r '.checkpoint' logs/control-best.json > logs/control-checkpoint.txt
[ "$(ss -lnt 'sport = :5555' | wc -l)" -eq 1 ] || { echo 'Port 5555 is already in use'; exit 1; }
tmux new-session -d -s control-policy "cd ~/work/PickMe && bash cloud/serve_control.sh > logs/control-policy.log 2>&1; echo \$? > logs/control-policy.exit"
ready=0
for attempt in {1..120}; do
    tmux has-session -t '=control-policy' 2>/dev/null || { tail -n 30 logs/control-policy.log; exit 1; }
    if [ "$(ss -lnt 'sport = :5555' | wc -l)" -gt 1 ]; then ready=1; break; fi
    sleep 1
done
[ "$ready" = 1 ]
tmux new-session -d -s control-viewer "cd ~/work/PickMe && CONTROL_EPISODES=3 CONTROL_SEED=2001 bash cloud/command_robot.sh > logs/live-control.log 2>&1; echo \$? > logs/live-control.exit"
echo "LEARNED_CONTROLLER_LAUNCHED checkpoint=$(cat logs/control-checkpoint.txt)"
