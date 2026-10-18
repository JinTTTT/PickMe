#!/usr/bin/env bash
# Cloud install for JinTTTT/PickMe; Ubuntu 24.04, Python 3.12, RTX GPU.
set -euo pipefail
PROJECT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK=${WORK:-$HOME/work}
export PATH="$HOME/.local/bin:$PATH"
export OMNI_KIT_ACCEPT_EULA=YES
export DEBIAN_FRONTEND=noninteractive
mkdir -p "$WORK" "$PROJECT_DIR/logs"
sudo apt-get update
sudo -E apt-get install -y git git-lfs curl ffmpeg tmux rsync python3.12-venv python3.12-dev build-essential cmake jq libglu1-mesa libxrandr2 libxinerama1 libxcursor1 libxi6 libxkbcommon0 libvulkan1 vulkan-tools libxt6 libsm6 xvfb x11-utils x11vnc fluxbox novnc websockify
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh -o "$WORK/install-uv.sh"
    sh "$WORK/install-uv.sh"
fi
/usr/bin/python3.12 -m venv "$WORK/env_isaaclab"
source "$WORK/env_isaaclab/bin/activate"
uv pip install --python "$VIRTUAL_ENV/bin/python" --upgrade pip
uv pip install --python "$VIRTUAL_ENV/bin/python" 'isaacsim[all,extscache]==6.1.0' --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match
if [ ! -d "$WORK/IsaacLab" ]; then
    git clone --no-checkout https://github.com/isaac-sim/IsaacLab.git "$WORK/IsaacLab"
    git -C "$WORK/IsaacLab" checkout ae37b028ea415c91ea2bc32609efcd759ed2b974
fi
[ "$(git -C "$WORK/IsaacLab" rev-parse HEAD)" = ae37b028ea415c91ea2bc32609efcd759ed2b974 ]
(cd "$WORK/IsaacLab" && ./isaaclab.sh --install core)
uv pip install --python "$VIRTUAL_ENV/bin/python" --no-deps -e "$WORK/IsaacLab/source/isaaclab_teleop"
uv pip install --python "$VIRTUAL_ENV/bin/python" pyzmq msgpack msgpack-numpy h5py imageio pyarrow
[ -f "$WORK/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/contrib/stack/config/so101/pose_ik_action.py" ]
deactivate
if [ ! -d "$WORK/Isaac-GR00T" ]; then
    git clone --no-checkout https://github.com/NVIDIA/Isaac-GR00T.git "$WORK/Isaac-GR00T"
    git -C "$WORK/Isaac-GR00T" checkout dac54b313160c201c450f97df0fd74ebd379c892
fi
[ "$(git -C "$WORK/Isaac-GR00T" rev-parse HEAD)" = dac54b313160c201c450f97df0fd74ebd379c892 ]
(cd "$WORK/Isaac-GR00T" && git lfs install --local && git lfs pull && git submodule update --init --recursive && uv sync --frozen --python /usr/bin/python3.12)
printf 'IsaacSim=6.1.0\nIsaacLab=ae37b028ea415c91ea2bc32609efcd759ed2b974\nGR00T=dac54b313160c201c450f97df0fd74ebd379c892\n' > "$PROJECT_DIR/logs/stack-versions.txt"
echo 'STACK_INSTALL_COMPLETE'
