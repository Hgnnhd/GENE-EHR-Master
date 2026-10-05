# 主实验运行命令

全部命令写在项目根目录的 [`run_main_experiment.sh`](../run_main_experiment.sh)，每一步一个命令，可在任意目录运行。研究定义见 [人群与数据协议](cohort_protocol.md)。

## 命令清单

```bash
bash run_main_experiment.sh data                      # 1. 数据构建与校验
bash run_main_experiment.sh pretrain                  # 2. 三个模型预训练（方案 A），各占一张卡同时跑
bash run_main_experiment.sh main                      # 3. 主实验：主模型 + 全部基线，预训练模型载入权重后微调
bash run_main_experiment.sh ablation_pretrain         #    消融①：三个预训练模型改为从零训练
bash run_main_experiment.sh ablation_single           #    消融②：主模型逐个癌种单独训练
bash run_main_experiment.sh pretrain_b sensitivity_b  #    敏感性分析 B：只用 2011 年前数据预训练，再微调
bash run_main_experiment.sh evaluate                  # 4. 评估全部模型
bash run_main_experiment.sh all                       #    以上全部，按顺序执行

# 单个模型
MODEL=behrt GPU=2 bash run_main_experiment.sh finetune   # 载入预训练权重
MODEL=behrt GPU=2 bash run_main_experiment.sh scratch    # 从零训练
```

可以一次写多个步骤，按顺序执行，例如 `bash run_main_experiment.sh pretrain main evaluate`。不带参数运行会打印这份清单。

## 环境变量

| 变量 | 含义 | 默认值 |
|---|---|---|
| `GPUS` | 并行任务使用的 GPU 编号 | `0,1,2,3,4,5` |
| `LABEL_MODE` | `verified`；`provisional_observed` 仅用于调试流程，结果不能报告 | `verified` |
| `PY` | Python 解释器 | `python` |
| `MODEL`、`GPU` | `finetune` / `scratch` 使用的模型和 GPU | `ehr_transformer`、`0` |
| `RESUME=1` | 预训练从 `last.pt` 接着训练（中断后，或调大 epoch 后） | 不设置 |

## 各步骤说明

| 步骤 | 做什么 | 主要输出 |
|---|---|---|
| `data` | 构建研究母体与固定划分、癌症登记、病史、自报、预测点人群与结局、模型输入、风险因素、预训练语料，并做全量校验 | `data/processed/`；汇总在 `ccfa-workfiles/checks/cancer-cohort/` |
| `pretrain` | `ehr_transformer`、BEHRT、Med-BERT 掩码代码预训练（训练组 2016 年前代码），三个任务各占一张卡并行 | `data/processed/models/pretrained/main/<模型>/`：`encoder.pt`、`training_curves.png`、`history.csv` |
| `pretrain_b` | 同上，只用 2011 年前代码（方案 B） | `.../pretrained/strict_2011/<模型>/` |
| `main` | 主模型 + 全部基线，十个癌种联合的竞争风险训练；预训练模型载入 `pretrain` 的权重后微调，缺权重时先自动预训练 | `data/processed/models/<标签模式>/<模型>/` |
| `ablation_pretrain` | 三个预训练模型不载入权重、从随机初始化训练 | `<模型>__no_pretrain` |
| `ablation_single` | 主模型为每个癌种单独训练一个模型 | `ehr_transformer__single_<癌种>`，评估时合并为 `ehr_transformer__single` |
| `sensitivity_b` | 三个预训练模型载入方案 B 的权重后微调 | `<模型>__pt_strict_2011` |
| `evaluate` | 测试集上逐模型 × 预测点 × 癌种 × 年份评估：IPCW 时间依赖 AUROC（含 95% CI）、IPCW Brier、校准 | `ccfa-workfiles/checks/cancer-cohort/models/<标签模式>/`：`metrics.csv`、`auroc_<年>y.csv`、`joint_vs_single.csv` |
| `finetune` | 单个预训练模型载入权重后微调 | `<模型>` |
| `scratch` | 单个预训练模型从零训练 | `<模型>__no_pretrain` |

日志在 `logs/`：预训练为 `logs/pretrain_<模型>_<版本>.log`，多卡训练为 `logs/train_<模型>.log`。

## 当前可以运行的步骤

登记覆盖核实前（`configs/registry_coverage.json` 为 `unverified`），没有正式标签：

- 可以正式运行：`data`、`pretrain`、`pretrain_b`（预训练不使用标签）。
- `main`、两个消融、`sensitivity_b`、`evaluate`、`finetune`、`scratch` 在默认模式下会报错退出。可以用 `LABEL_MODE=provisional_observed` 跑通流程，但结果只用于调试，不能报告。

## 直接调用各脚本

`run_main_experiment.sh` 只是按顺序调用以下脚本，需要更细的控制时可直接使用：

```bash
python source_code/01_build_data.py [--from cohort | --only validate]
python source_code/02_pretrain.py --model ehr_transformer --device cuda:0 [--pretrain-variant strict_2011] [--resume]
python source_code/03_train_models.py --model ehr_transformer --device cuda:0              # 载入预训练权重
python source_code/03_train_models.py --model ehr_transformer --device cuda:0 --no-pretrain  # 从零训练
python source_code/03_train_models.py --gpus 0,1,2,3,4,5 [--models ...] [--single-site all] [--pretrain-variant strict_2011]
python source_code/04_evaluate.py [--split validation] [--bootstrap 200]
```
