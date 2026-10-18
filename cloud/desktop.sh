#!/usr/bin/env bash
set -euo pipefail
export DISPLAY=:99
children=()
cleanup() { trap - EXIT INT TERM; kill "${children[@]}" 2>/dev/null || true; }
trap cleanup EXIT
trap 'exit 0' INT TERM
Xvfb :99 -screen 0 1600x900x24 -nolisten tcp &
children+=("$!")
for attempt in {1..50}; do
    xdpyinfo -display :99 >/dev/null 2>&1 && break
    sleep 0.2
done
xdpyinfo -display :99 >/dev/null
fluxbox &
children+=("$!")
x11vnc -display :99 -localhost -rfbport 5901 -nopw -forever -shared -noxdamage &
children+=("$!")
websockify --web=/usr/share/novnc 127.0.0.1:6080 127.0.0.1:5901 &
children+=("$!")
wait -n "${children[@]}"
