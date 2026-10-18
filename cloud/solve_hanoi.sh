#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
cd "$PROJECT_DIR"
mkdir -p logs
# Share the lock with command_robot.sh so two controllers cannot cover the desktop.
exec 9>logs/learned-control.lock
flock -w 15 9 || { echo 'Another controller is running; press Ctrl-C in its terminal first.' >&2; exit 1; }
export DISPLAY=:99
for attempt in {1..50}; do
    xdpyinfo -display "$DISPLAY" >/dev/null 2>&1 && break
    sleep 0.2
done
xdpyinfo -display "$DISPLAY" >/dev/null
echo 'Starting the scripted Tower of Hanoi solver (no model server needed).'
exec "$ISAACLAB_PY" -u scripts/solve_hanoi.py --viz kit --keep_open "$@"