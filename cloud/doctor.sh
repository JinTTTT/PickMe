#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
"$ISAACLAB_PY" - <<'PY'
import importlib.metadata as m
import torch
from isaaclab.app.sim_launcher import launch_simulation
assert torch.cuda.is_available(), 'Isaac Lab CUDA unavailable'
x = torch.randn(128, 128, device='cuda')
assert torch.isfinite(x @ x).all().item()
print('SIM_ENV_OK', 'isaacsim', m.version('isaacsim'), 'torch', torch.__version__, torch.cuda.get_device_name(0))
PY
"$GR00T_REPO/.venv/bin/python" - <<'PY'
import torch
import torchcodec
import gr00t
assert torch.cuda.is_available(), 'GR00T CUDA unavailable'
x = torch.randn(128, 128, device='cuda')
assert torch.isfinite(x @ x).all().item()
print('GR00T_ENV_OK', 'torch', torch.__version__, torch.cuda.get_device_name(0))
PY
