#!/usr/bin/env bash
# Run on the VM. Stage and train the learned SO-101 controller in persistent tmux.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
[ ! -f "$PROJECT_DIR/cloud/control.env" ] || source "$PROJECT_DIR/cloud/control.env"
export DATASET_NAME="${CONTROL_DATASET_NAME:-so101_cosmos_pickplace_20261018}"
export DATASET_ROOT="$DATASETS_DIR/local/$DATASET_NAME"
export TRAIN_DATASET="$DATASET_ROOT"
export FINETUNE_OUTPUT="$WORK/finetune/$DATASET_NAME"
export MAX_STEPS="${MAX_STEPS:-4000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
cd "$PROJECT_DIR"
# Authentication must be valid before any model initialization/training.
"$GR00T_REPO/.venv/bin/hf" download nvidia/Cosmos-Reason2-2B config.json
while tmux has-session -t '=control-preflight' 2>/dev/null; do
    echo 'Waiting for the model training preflight...'
    sleep 10
done
[ "$(cat logs/control-preflight.exit)" = 0 ] || { echo 'Training preflight failed; inspect logs/control-preflight.log'; exit 1; }
if [ "${CONTROL_WAIT_FOR_COLLECTION:-1}" = 1 ]; then
    while tmux has-session -t '=control-data' 2>/dev/null; do
        echo 'Waiting for training demonstration collection and dataset preparation...'
        sleep 30
    done
    [ "$(cat logs/control-data.exit)" = 0 ] || { echo 'Dataset preparation failed; inspect logs/control-data.log'; exit 1; }
fi
minimum=${CONTROL_MIN_EPISODES:-100}
case "$minimum" in ''|*[!0-9]*) echo 'Invalid CONTROL_MIN_EPISODES'; exit 1 ;; esac
jq -e ".total_episodes >= $minimum" "$DATASET_ROOT/meta/info.json"
[ -f "$DATASET_ROOT/meta/stats.json" ]
if [ -d "$FINETUNE_OUTPUT" ]; then
    echo "Output already exists: $FINETUNE_OUTPUT. Review it before resuming or starting another run." >&2
    exit 1
fi
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1)" -lt 65000 ]; do
    echo 'Waiting for 65 GB of free GPU memory before starting training...'
    sleep 30
done
bash finetune.sh
checkpoint="$FINETUNE_OUTPUT/checkpoint-$MAX_STEPS"
[ -f "$checkpoint/config.json" ]
printf '%s\n' "$checkpoint" > logs/control-final-checkpoint.txt
echo "TRAINING_COMPLETE checkpoint=$checkpoint"
