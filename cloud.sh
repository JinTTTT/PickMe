#!/usr/bin/env bash
set -euo pipefail
TARGET=${SSH_TARGET:-ubuntu@81.85.1.206}
IDENTITY=${SSH_IDENTITY:-$HOME/.ssh/id_ed25519}
POLICY_PORT=${POLICY_PORT:-5556}
CONTROL_SCENE=${CONTROL_SCENE:-cube}
ssh_args=(-i "$IDENTITY" -o BatchMode=yes -o ConnectTimeout=15)
quote() {
    local value="$1"
    printf "'"
    while [[ "$value" == *"'"* ]]; do
        printf '%s' "${value%%\'*}" "'\\''"
        value="${value#*\'}"
    done
    printf "%s'" "$value"
}
case "${1:-}" in
  ui)
    ssh "${ssh_args[@]}" "$TARGET" 'tmux has-session -t desktop 2>/dev/null || tmux new-session -d -s desktop "cd ~/work/PickMe && bash cloud/desktop.sh > logs/desktop.log 2>&1"'
    echo 'Open http://localhost:6080/vnc.html?autoconnect=true&resize=scale (keep this terminal open)'
    exec ssh "${ssh_args[@]}" -N -o ExitOnForwardFailure=yes -L 127.0.0.1:6080:127.0.0.1:6080 "$TARGET" ;;
  view)
    ssh "${ssh_args[@]}" "$TARGET" 'tmux has-session -t viewer 2>/dev/null || tmux new-session -d -s viewer "cd ~/work/PickMe && bash cloud/view.sh > logs/viewer.log 2>&1"' ;;
  logs) exec ssh "${ssh_args[@]}" "$TARGET" 'tail -n 50 ~/work/PickMe/logs/setup.log ~/work/PickMe/logs/viewer.log' ;;
  login) exec ssh -t "${ssh_args[@]}" "$TARGET" "~/work/Isaac-GR00T/.venv/bin/hf auth login" ;;
  check) exec ssh "${ssh_args[@]}" "$TARGET" "cd ~/work/PickMe && bash cloud/doctor.sh" ;;
  train)
    exec ssh "${ssh_args[@]}" "$TARGET" 'tmux new-session -d -s control-train "cd ~/work/PickMe && bash cloud/train_control.sh > logs/control-train.log 2>&1; echo \$? > logs/control-train.exit"' ;;
  serve)
    exec ssh "${ssh_args[@]}" "$TARGET" "tmux new-session -d -s control-policy \"cd ~/work/PickMe && POLICY_PORT=$(quote "$POLICY_PORT") bash cloud/serve_control.sh > logs/control-policy.log 2>&1; echo \\\$? > logs/control-policy.exit\"" ;;
  serve-base)
    exec ssh "${ssh_args[@]}" "$TARGET" "tmux new-session -d -s control-policy \"cd ~/work/PickMe && source env.sh && CHECKPOINT=\\\"$BASE_MODEL\\\" POLICY_PORT=$(quote \"$POLICY_PORT\") bash start_server.sh > logs/control-policy.log 2>&1; echo \\\$? > logs/control-policy.exit\"" ;;
  control)
    shift
    ssh "${ssh_args[@]}" "$TARGET" 'tmux kill-session -t =control-viewer 2>/dev/null || true'
    command="cd ~/work/PickMe && POLICY_PORT=$(quote "$POLICY_PORT") CONTROL_SCENE=$(quote "$CONTROL_SCENE") bash cloud/command_robot.sh"
    for argument in "$@"; do command="$command $(quote "$argument")"; done
    exec ssh -t "${ssh_args[@]}" "$TARGET" "$command" ;;
  hanoi)
    shift
    ssh "${ssh_args[@]}" "$TARGET" 'tmux has-session -t desktop 2>/dev/null || tmux new-session -d -s desktop "cd ~/work/PickMe && bash cloud/desktop.sh > logs/desktop.log 2>&1"'
    command="cd ~/work/PickMe && bash cloud/solve_hanoi.sh"
    for argument in "$@"; do command="$command $(quote "$argument")"; done
    exec ssh -t "${ssh_args[@]}" "$TARGET" "$command" ;;
  progress)
    exec ssh "${ssh_args[@]}" "$TARGET" 'tail -n 5 ~/work/PickMe/logs/control-data.log; test ! -f ~/work/PickMe/logs/control-train.log || tail -n 8 ~/work/PickMe/logs/control-train.log' ;;
  ssh) exec ssh -t "${ssh_args[@]}" "$TARGET" ;;
  *) echo 'Usage: bash cloud.sh ui|view|hanoi [solver options]|logs|check|login|train|serve|serve-base|control "instruction"|progress|ssh  (hanoi = scripted solver; CONTROL_SCENE=hanoi control = learned policy)'; exit 1 ;;
esac
