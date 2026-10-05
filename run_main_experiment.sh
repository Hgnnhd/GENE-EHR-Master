#!/usr/bin/env bash
# 主实验脚本：按顺序执行，每一步一个命令（在任意目录运行均可）。
#
#   bash run_main_experiment.sh data                      # 1. 数据构建与校验
#   bash run_main_experiment.sh pretrain                  # 2. 三个模型预训练（方案 A），各占一张卡同时跑
#   bash run_main_experiment.sh main                      # 3. 主实验：主模型 + 全部基线，预训练模型载入权重后微调
#   bash run_main_experiment.sh ablation_pretrain         #    消融①：三个预训练模型改为从零训练
#   bash run_main_experiment.sh ablation_single           #    消融②：主模型逐个癌种单独训练
#   bash run_main_experiment.sh pretrain_b sensitivity_b  #    敏感性分析 B：只用 2011 年前数据预训练，再微调
#   bash run_main_experiment.sh evaluate                  # 4. 评估全部模型
#   bash run_main_experiment.sh all                       #    以上全部，按顺序执行
#
#   # 单个模型
#   MODEL=behrt GPU=2 bash run_main_experiment.sh finetune   # 载入预训练权重
#   MODEL=behrt GPU=2 bash run_main_experiment.sh scratch    # 从零训练
#
# 可一次写多个步骤，按顺序执行。环境变量（方括号内为默认值）：
#   GPUS        并行任务使用的 GPU 编号                     [0,1,2,3,4,5]
#   LABEL_MODE  verified | provisional_observed（仅调试）   [verified]
#   PY          Python 解释器                               [python]
#   MODEL, GPU  finetune / scratch 使用的模型和 GPU         [ehr_transformer, 0]
#   RESUME=1    预训练从 last.pt 接着训练
#
# 日志写在 logs/；说明文档见 docs/experiment_commands.md。
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

usage() { sed -n '2,/^[^#]/p' "$0" | grep '^#' | sed 's/^# \{0,1\}//'; }
[ $# -gt 0 ] || { usage; exit 1; }
for step in "$@"; do
  case "$step" in
    data|pretrain|pretrain_b|main|finetune|scratch|ablation_pretrain|ablation_single|sensitivity_b|evaluate|all)
      echo "=== $step ($(date '+%F %T')) ==="; "$step" ;;
    *) echo "unknown step: $step"; usage; exit 1 ;;
  esac
done
