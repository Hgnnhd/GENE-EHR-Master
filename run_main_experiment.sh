#!/usr/bin/env bash
# Main experiment, step by step. Run from anywhere:
#
#   bash run_main_experiment.sh <step> [<step> ...]
#   bash run_main_experiment.sh all                  # data -> pretrain -> main -> ablations -> evaluate
#
# Settings (environment variables, defaults in brackets):
#   GPUS        GPU ids for parallel jobs                   [0,1,2,3,4,5]
#   LABEL_MODE  verified | provisional_observed (debug only) [verified]
#   PY          python interpreter                          [python]
#   RESUME=1    continue interrupted pretraining from last.pt
#
# Steps:
#   data          1. build cohorts, outcomes, model inputs, pretraining corpus; validate
#   pretrain      2. masked-code pretraining of ehr_transformer, behrt, medbert (variant A, one GPU each)
#   pretrain_b       same for sensitivity B (pretraining codes before 2011 only)
#   main          3. main experiment: ehr_transformer + every baseline, joint 10-cancer competing-risk
#                    training; pretrained models LOAD their step-2 weights and fine-tune
#   finetune         one pretrained model on one GPU (loads weights), e.g. MODEL=behrt GPU=0
#   scratch          one pretrained model trained from scratch, no weights loaded, e.g. MODEL=ehr_transformer GPU=0
#   ablation_pretrain   ehr_transformer / behrt / medbert from scratch (pretraining ablation)
#   ablation_single     ehr_transformer, one model per cancer site (joint-training ablation)
#   sensitivity_b       pretrained models fine-tuned from the variant-B weights
#   evaluate      4. evaluate everything trained so far (IPCW AUROC / Brier / calibration at 1, 3, 5 years)
set -euo pipefail
cd "$(dirname "$0")"

GPUS=${GPUS:-0,1,2,3,4,5}
LABEL_MODE=${LABEL_MODE:-verified}
PY=${PY:-python}
MODEL=${MODEL:-ehr_transformer}
GPU=${GPU:-0}
PRETRAINED="ehr_transformer behrt medbert"
mkdir -p logs

gpu_at() { cut -d, -f"$(( $1 % $(tr ',' '\n' <<<"$GPUS" | wc -l) + 1 ))" <<<"$GPUS"; }

data() {
  $PY source_code/01_build_data.py
}

pretrain_variant() {  # $1 = main | strict_2011; the three models in parallel, one GPU each
  local i=0 pids=() m
  for m in $PRETRAINED; do
    echo "pretraining $m (variant $1) on GPU $(gpu_at $i) -> logs/pretrain_${m}_$1.log"
    $PY source_code/02_pretrain.py --model "$m" --pretrain-variant "$1" --device "cuda:$(gpu_at $i)" \
      ${RESUME:+--resume} > "logs/pretrain_${m}_$1.log" 2>&1 &
    pids+=($!); i=$((i + 1))
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  echo "curves: data/processed/models/pretrained/$1/<model>/training_curves.png"
}
pretrain()   { pretrain_variant main; }
pretrain_b() { pretrain_variant strict_2011; }

main() {  # all models; missing pretrained weights are pretrained first
  $PY source_code/03_train_models.py --gpus "$GPUS" --label-mode "$LABEL_MODE" --no-evaluate
}

finetune() {  # a single pretrained model: loads models/pretrained/main/$MODEL/encoder.pt
  $PY source_code/03_train_models.py --model "$MODEL" --device "cuda:$GPU" --label-mode "$LABEL_MODE" --no-evaluate
}

scratch() {  # the same model without pretraining (random initialisation)
  $PY source_code/03_train_models.py --model "$MODEL" --device "cuda:$GPU" --label-mode "$LABEL_MODE" --no-pretrain --no-evaluate
}

ablation_pretrain() {
  $PY source_code/03_train_models.py --gpus "$GPUS" --models "${PRETRAINED// /,}" --no-pretrain \
    --label-mode "$LABEL_MODE" --no-evaluate
}

ablation_single() {
  $PY source_code/03_train_models.py --gpus "$GPUS" --models ehr_transformer --single-site all \
    --label-mode "$LABEL_MODE" --no-evaluate
}

sensitivity_b() {
  $PY source_code/03_train_models.py --gpus "$GPUS" --models "${PRETRAINED// /,}" --pretrain-variant strict_2011 \
    --label-mode "$LABEL_MODE" --no-evaluate
}

evaluate() {
  $PY source_code/04_evaluate.py --label-mode "$LABEL_MODE"
}

all() {
  data; pretrain; pretrain_b; main; ablation_pretrain; ablation_single; sensitivity_b; evaluate
}

[ $# -gt 0 ] || { sed -n '2,32p' "$0"; exit 1; }
for step in "$@"; do
  case "$step" in
    data|pretrain|pretrain_b|main|finetune|scratch|ablation_pretrain|ablation_single|sensitivity_b|evaluate|all)
      echo "=== $step ($(date '+%F %T')) ==="; "$step" ;;
    *) echo "unknown step: $step"; sed -n '2,32p' "$0"; exit 1 ;;
  esac
done
