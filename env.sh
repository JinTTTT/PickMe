#!/usr/bin/env bash
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK="${WORK:-$HOME/work}"
export GR00T_REPO="$WORK/Isaac-GR00T"
export ISAACLAB_DIR="$WORK/IsaacLab"
export ISAACLAB_PY="$WORK/env_isaaclab/bin/python"
export OMNI_KIT_ACCEPT_EULA=YES
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"
export TASK="${TASK:-pick up the red block and place it on the green pad}"
export DATASETS_DIR="${DATASETS_DIR:-$WORK/datasets}"
export DATASET_NAME="${DATASET_NAME:-so101_sim_pickplace}"
export DATASET_ROOT="${DATASET_ROOT:-$DATASETS_DIR/local/$DATASET_NAME}"
export NUM_EPISODES="${NUM_EPISODES:-50}"
export BASE_MODEL="${BASE_MODEL:-$GR00T_REPO/checkpoints/GR00T-N1.7-3B}"
export MODALITY_CONFIG="$PROJECT_DIR/gr00t/so101_sim_config.py"
export EMBODIMENT_TAG=NEW_EMBODIMENT
export FINETUNE_OUTPUT="${FINETUNE_OUTPUT:-$WORK/finetune/$DATASET_NAME}"
export CHECKPOINT="${CHECKPOINT:-}"
export POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
export POLICY_PORT="${POLICY_PORT:-5555}"
unset HF_HUB_OFFLINE
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
gr00t_activate() {
    cd "$GR00T_REPO"
    source .venv/bin/activate
}
