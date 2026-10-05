# EHR 预训练与癌症风险预测

独立数据管线：UK Biobank 首次医疗代码历史 → 预训练数据 → 2011、2016 两个固定预测点的首次癌症风险队列。

研究定义见 [人群与数据协议](docs/cohort_protocol.md)。

## 四个主要步骤

完整实验按顺序写在 [`run_main_experiment.sh`](run_main_experiment.sh)，每一步一个命令：

```bash
bash run_main_experiment.sh data                    # 1. 数据构建与校验
bash run_main_experiment.sh pretrain                # 2. 预训练 ehr_transformer / behrt / medbert（方案 A，各占一张卡）
bash run_main_experiment.sh main                    # 3. 主实验：主模型 + 全部基线；预训练模型载入权重后微调
bash run_main_experiment.sh ablation_pretrain       #    消融：同样三个模型不载入预训练、从零训练
bash run_main_experiment.sh ablation_single         #    消融：主模型逐癌种单独训练
bash run_main_experiment.sh pretrain_b sensitivity_b  # 敏感性 B：2011 年前语料预训练 + 微调
bash run_main_experiment.sh evaluate                # 4. 评估全部模型
MODEL=behrt GPU=2 bash run_main_experiment.sh finetune   # 单个模型：载入预训练权重
MODEL=behrt GPU=2 bash run_main_experiment.sh scratch    # 单个模型：不载入、从零训练
```

环境变量：`GPUS`（默认 `0,1,2,3,4,5`）、`LABEL_MODE`（默认 `verified`）、`PY`、`RESUME=1`（预训练续训）。日志在 `logs/`。

下面是各脚本的直接用法：

```bash
python -m pip install -e ".[models]"     # 服务器已有匹配 CUDA 的 torch 时只补装其余依赖
python source_code/01_build_data.py                        # 1. 数据：人群、标签、模型输入、预训练语料、校验
python source_code/02_pretrain.py --model all --device cuda:0   # 2. 掩码代码预训练（03 也会自动补跑）
python source_code/03_train_models.py --gpus 0,1,2,3,4,5   # 3. 主模型 + 全部基线，多卡并行，结束后自动评估
python source_code/04_evaluate.py                          # 4. 单独重做评估
python -m pytest -q
```

```
source_code/
├── 01_build_data.py      数据构建（9 个阶段，见下）
├── 02_pretrain.py        预训练
├── 03_train_models.py    训练主模型与基线（单个模型或多卡调度）
├── 04_evaluate.py        评估
└── lib/                  实现代码，不直接运行
    ├── common.py, definitions.py         路径/读写/进度条；癌种、划分、随访与事件定义
    ├── data_*.py                         数据构建的各个阶段
    ├── model_data.py, model_nets.py      模型输入与标签；全部网络结构
    ├── model_training.py, model_metrics.py
    ├── model_pretrain.py, model_classical.py, model_deep.py, model_evaluate.py
    └── scheduler.py                      多 GPU 任务调度
```

### 1. 数据构建的阶段（`01_build_data.py`）

| 阶段 | 作用 | 主要输出（`data/processed/`） |
|---|---|---|
| `participants` | 研究母体：招募、性别、出生月、死亡/失访、登记覆盖；固定 70/15/15 划分及预训练角色 | `participants` |
| `registry` | 癌症登记按 Instance 配对日期与编码，派生首次恶性肿瘤日期与癌种集合 | `cancer_events` |
| `history` | 首次发生 + 住院诊断 → 每人每个 ICD-10 三级码的最早可靠日期；住院既往癌 | `events` |
| `self_report` | 自报癌症病史分类（本地 UKB 编码字典） | `self_report_cancer` |
| `cohort` | **人群定义**：各预测点逐条排除、随访与竞争事件、逐癌种标签 | `landmark_status`、`landmark_samples` |
| `inputs` | EHR-only 模型输入序列 | `landmark_inputs` |
| `risk_factors` | 增强版本：预测点前最近一次风险因素 | `features_asof` |
| `pretrain_corpus` | 预训练语料与词表；写构建清单和 `BUILD_COMPLETE.json` | `pretrain_events`、`pretrain_participants` |
| `validate` | 全量数据一致性校验 | `validation.json`（汇总目录） |

`--from cohort` 从某阶段起重跑，`--only validate` 只跑一个阶段。人群定义需要先有登记、住院和自报的既往癌症信息，所以在 `cohort` 阶段；`participants` 只建立研究母体和固定划分（先划分参与者，再生成预测点样本）。各阶段只通过 `data/processed/` 下的 parquet 文件传递数据，改动某阶段后需重跑它及之后的阶段；任一阶段开始时都会删除 `BUILD_COMPLETE.json`，直到 `pretrain_corpus` 重新写入，模型步骤没有它会拒绝运行。

每个阶段打印标题、带累计用时的日志、大文件和逐预测点/逐癌种的进度条，结束时打印关键计数，`cohort` 阶段打印每个预测点的逐条排除流程。输出重定向到文件时（如 `nohup ... > build.log`）进度条约每分钟刷新一次。质控计数写入汇总目录 `build_summary.json`（按阶段名），隔离记录写入 `data/processed/quarantine/<阶段>.parquet`。

### 2–4. 预训练、训练与评估

| 类别 | 模型 |
|---|---|
| 主模型 | `ehr_transformer`：代码 + 连续年龄 + 距预测点时间 + 就诊序号编码，掩码预训练后微调 |
| 统计基线 | `cr_logistic`：离散时间竞争风险（多项）逻辑回归，输入代码词袋 + 背景信息 |
| 词袋基线 | 逻辑回归、随机森林、LightGBM、XGBoost（GPU），每个癌种一个五年二分类模型；MLP |
| 就诊序列基线 | GRU、LSTM（Doctor AI 式）、RETAIN、Dipole |
| Transformer 基线 | Transformer（无预训练）、BEHRT、Med-BERT（仅 MLM，无住院时长任务） |

```bash
python source_code/03_train_models.py --gpus 0,1 --models gru,ehr_transformer   # 部分模型
python source_code/03_train_models.py --model behrt --device cuda:0             # 当前进程训练单个模型
python source_code/03_train_models.py --gpus 0,1,2,3,4,5 --models classical     # 只跑 4 个传统模型
```

所有模型使用相同的候选节点、患者划分、输入（预测点前最近 64 个代码 + 年龄、性别、预测点）和结局；超参数在 [configs/models.json](configs/models.json)。

**结局与输出（竞争风险）。** 深度模型和 `cr_logistic` 共用同一个输出层：把五年分成 5 个一年的时间段，每段一个 softmax，覆盖 13 类互斥结局——本段无事件、10 个目标癌种、其他首发恶性肿瘤、死亡。删失者只贡献随访到的时间段；性别不适用的癌种概率强制为 0；同日多癌种按"其中之一"计入。输出每个癌种成为首发恶性肿瘤的 1、3、5 年累积发生率。输出层偏置初始化为训练集各时间段的经验结局分布，验证集负对数似然早停。两个预测点共享一个模型。四个传统模型保留每癌种五年二分类（`label_*_5y`，删失者无标签），作为常用参照。`03_train_models.py --gpus` 每张卡同时跑一个任务，缺预训练权重时先跑 `02_pretrain.py` 再接微调；每个任务日志在 `logs/<任务>.log`，运行中定时打印状态表，结束后自动评估（`--no-evaluate` 跳过）。

评估（`04_evaluate.py`）：测试集逐模型 × 预测点 × 癌种 × 年份（1、3、5 年；二分类基线只有 5 年）计算删失加权（IPCW）的时间依赖 AUROC（病例 = 到该年该癌种为首发恶性肿瘤；对照 = 到该年无事件或先发生其他事件）及按参与者 bootstrap 95% CI、IPCW Brier 分数、与 Aalen–Johansen 观察累积发生率比较的校准截距。模型文件和逐人预测在 `data/processed/models/<标签模式>/<模型>/`（不入库），汇总指标在 `ccfa-workfiles/checks/cancer-cohort/models/<标签模式>/metrics.csv` 和 `auroc_<年>y.csv`。

**联合训练与单癌种消融。** 主实验是 10 个癌种联合训练：一个模型、共享编码器、同一个竞争风险输出。消融实验用同样的结构为每个癌种单独训练一个模型，结局合并为"无事件 / 该癌种 / 其他首发癌（含其余 9 个目标癌种）/ 死亡"：

```bash
python source_code/03_train_models.py --gpus 0,1,2,3,4,5 --models ehr_transformer --single-site all
```

10 个单癌种结果在评估中合并为一行 `<模型>__single`，并输出 `joint_vs_single.csv`（逐预测点、逐癌种的联合与单独训练 5 年 AUROC 及差值）。传统基线本来就是每癌种一个模型，不参与该消融。

**训练过程记录。** 预训练和微调每个 epoch 都会更新模型目录下的 `history.json`、`history.csv` 和 `training_curves.png`。预训练记录掩码代码的损失、top-1/top-5 准确率、困惑度和学习率；验证集的掩码位置固定，各 epoch 可直接比较。微调记录竞争风险负对数似然、验证集 AUROC 和学习率。预训练默认最多 100 个 epoch、早停耐心 10（`configs/models.json` 的 `pretrain`）；每个 epoch 保存 `last.pt`，中断后或调大 epoch 数后可用 `02_pretrain.py --resume` 接着训练。

**预训练版本。** `--pretrain-variant main`（默认，方案 A：训练组 2016 年前代码）或 `strict_2011`（方案 B：训练组 2011 年前代码，敏感性分析），定义在 `configs/models.json` 的 `pretrain_variants`。B 只对三个预训练模型重做，权重在 `models/pretrained/<版本>/`，微调结果以 `<模型>__pt_strict_2011` 与主结果并列评估：

```bash
python source_code/03_train_models.py --gpus 0,1,2 --pretrain-variant strict_2011
```

**标签模式。** 默认 `verified` 使用数据构建得到的随访状态、随访天数和首发癌种；登记覆盖未核实时这些为空，训练会直接报错退出。`--label-mode provisional_observed` 只用登记中已观察到的事件并假设登记完整（没有行政删失），**只用于调试训练流程，结果不能报告**；两种模式的输出分目录保存，评估不会混用。

## 输入与输出

输入目录默认读取 [configs/cohort.json](configs/cohort.json) 的 `data_dirs`，也可用命令行覆盖；原始文件保持不变。

| 参数 / 配置键 | 默认目录 | 内容 |
|---|---|---|
| `--ukb-fields` / `ukb_fields` | `/data15/hd/ehr` | UKB 字段导出：随访日期、死亡、首次发生、自报病史、`Base_Information_split/` |
| `--hospital-cancer` / `hospital_cancer` | `/data/hd/WB-MRI-main/data/EHR` | `record.csv`（住院 ICD-10、出生）、`cancer.csv`（癌症登记）、UKB 编码字典 |

输出、汇总和配置路径均相对项目根目录，可在任意目录下运行。缺少输入文件时会列出缺失路径后退出。

- 配置：[configs/cohort.json](configs/cohort.json)。固定患者划分 70/15/15，seed 42。
- 私有数据：`data/processed/`，已加入 `.gitignore`。
- 汇总、病例流程及质控：`ccfa-workfiles/checks/cancer-cohort/`。
- 重新构建会覆盖当前派生表；只有 `BUILD_COMPLETE.json` 存在才表示整个构建完成。

`landmark_samples` 保存通过临床病史筛查的候选节点。是否已核实可入组见 `eligible`；能否用于监督学习见 `followup_status` 和 `label_*`。`observed_*` 是登记中已看到的病例统计，不能作为完整二分类训练标签。

癌症登记覆盖日期必须与本地导出版本相符。当前 [覆盖配置](configs/registry_coverage.json) 标记为 `unverified`，因此候选者的五年训练标签保持空值。核实覆盖后提供逐参与者的 `registry_start`、`registry_end_exclusive`、数据版本与证据，再重新构建；地区归属不能仅凭癌症病例来源推给所有对照。

`pretrain_events` 仅含训练参与者在 2016 年之前的记录，内部早停集仍属于训练参与者；词表仅从内部预训练训练组生成。下游验证、测试患者不进入预训练。旧模型 checkpoint 暂不迁入。

增强特征保存原始值、评估 Instance、测量日期、距预测点天数和缺失标志。化验时间采用评估日期，当前导出没有结果发布时刻；这是回顾性测量特征，不能声称能复原当天临床可获取的化验结果。

`landmark_inputs` 是 EHR-only 模型输入序列，与 `landmark_samples` 一一对应；`event_{site}` 与 `followup_days` 给出竞争风险事件码。各划分组的均衡性见汇总目录 `split_summary.csv`。`python -m pytest` 包含合成数据上的端到端数据构建、训练与评估测试。

字段与质控规则见 [数据表说明](docs/data_tables.md)。
