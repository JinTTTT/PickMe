#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
source "$PROJECT_DIR/cloud/control.env"
cd "$PROJECT_DIR"
output="$WORK/finetune/$CONTROL_DATASET_NAME"
for step in $(seq "$SAVE_STEPS" "$SAVE_STEPS" "$MAX_STEPS"); do
    checkpoint="$output/checkpoint-$step"
    while [ ! -f "$checkpoint/trainer_state.json" ] || [ ! -f "$checkpoint/processor_config.json" ]; do
        if [ -f logs/control-train.exit ] && [ "$(cat logs/control-train.exit)" != 0 ]; then
            echo 'Training failed; inspect logs/control-train.log'; exit 1
        fi
        sleep 10
    done
    bash cloud/evaluate_checkpoint.sh "$checkpoint" > "logs/eval-$step.log" 2>&1
    tail -n 4 "logs/eval-$step.log"
done
[ -f logs/control-checkpoint.txt ] || { echo 'No checkpoint completed a successful held-out episode.'; exit 1; }
echo "VALIDATED_CONTROLLER=$(cat logs/control-checkpoint.txt)"
