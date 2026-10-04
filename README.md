# EHR 预训练与癌症风险预测

独立数据管线：UK Biobank 首次医疗代码历史 → 预训练数据 → 2011、2016 两个固定预测点的首次癌症风险队列。

研究定义见 [人群与数据协议](docs/cohort_protocol.md)。当前阶段构建数据，不启动模型训练。

```bash
python -m pip install -e .
python source_code/run_all.py                # 依次运行 01–09
python source_code/run_all.py --from 05      # 从第 05 步起重跑
python source_code/05_landmark_cohort.py     # 单独运行某一步
python -m pytest -q
```

## 流程步骤（`source_code/`）

| 步骤 | 文件 | 作用 | 主要输出（`data/processed/`） |
|---|---|---|---|
| 01 | `01_participants.py` | 研究母体：招募、性别、出生月、死亡/失访、登记覆盖；固定 70/15/15 划分及预训练角色 | `participants` |
| 02 | `02_cancer_registry.py` | 癌症登记按 Instance 配对日期与编码，派生首次恶性肿瘤日期与癌种集合 | `cancer_events` |
| 03 | `03_medical_history.py` | 首次发生 + 住院诊断 → 每人每个 ICD-10 三级码的最早可靠日期；住院既往癌 | `events` |
| 04 | `04_self_report.py` | 自报癌症病史分类（本地 UKB 编码字典） | `self_report_cancer` |
| 05 | `05_landmark_cohort.py` | **人群定义**：各预测点逐条排除、随访与竞争事件、逐癌种标签 | `landmark_status`、`landmark_samples` |
| 06 | `06_model_inputs.py` | EHR-only 模型输入序列 | `landmark_inputs` |
| 07 | `07_risk_factors.py` | 增强版本：预测点前最近一次风险因素 | `features_asof` |
| 08 | `08_pretraining.py` | 预训练数据与词表；写构建清单和 `BUILD_COMPLETE.json` | `pretrain_events`、`pretrain_participants` |
| 09 | `09_validate.py` | 全量数据一致性校验 | `validation.json`（汇总目录） |

公共代码：`common.py`（路径、读写、进度条、步骤上下文）、`definitions.py`（癌种、划分、随访与事件定义）。

## 模型步骤（10–13）

```bash
python -m pip install -e ".[models]"            # 服务器已有匹配 CUDA 的 torch 时直接安装其余依赖
python source_code/run_models.py --gpus 0,1,2,3,4,5             # 预训练 + 全部模型 + 评估
python source_code/run_models.py --gpus 0,1 --models gru,ehr_transformer
python source_code/12_deep_models.py --model behrt --device cuda:0   # 单独训练一个模型
```

| 步骤 | 文件 | 作用 |
|---|---|---|
| 10 | `10_pretrain.py` | 掩码代码预训练（BEHRT、Med-BERT、ehr_transformer）；仅训练组 2016 年前病史，内部验证组早停 |
| 11 | `11_classical_baselines.py` | 词袋代码 + 年龄/性别/预测点：逻辑回归、随机森林、LightGBM、XGBoost（GPU），每个癌种一个模型 |
| 12 | `12_deep_models.py` | 深度模型，两个预测点共享，一次输出 10 个癌种；验证集宏平均 AUROC 早停 |
| 13 | `13_evaluate.py` | 测试集逐模型 × 预测点 × 癌种：AUROC（按参与者 bootstrap 95% CI）、AUPRC、Brier、校准截距与斜率 |

| 类别 | 模型 |
|---|---|
| 主模型 | `ehr_transformer`：代码 + 连续年龄 + 距预测点时间 + 就诊序号编码，掩码预训练后微调 |
| 词袋基线 | 逻辑回归、随机森林、LightGBM、XGBoost、MLP |
| 就诊序列基线 | GRU、LSTM（Doctor AI 式）、RETAIN、Dipole |
| Transformer 基线 | Transformer（无预训练）、BEHRT、Med-BERT（仅 MLM，无住院时长任务） |

所有模型使用相同的候选节点、患者划分、输入（预测点前最近 64 个代码 + 年龄、性别、预测点）和标签；超参数在 [configs/models.json](configs/models.json)。`run_models.py` 每张卡同时跑一个任务，预训练完成后自动开始对应的微调，每个任务的日志在 `logs/<任务>.log`，运行中定时打印状态表；已有预训练权重默认复用（`--repretrain` 重跑）。

模型文件和逐人预测保存在 `data/processed/models/<标签模式>/<模型>/`（不入库），汇总指标在 `ccfa-workfiles/checks/cancer-cohort/models/<标签模式>/metrics.csv` 和 `auroc_table.csv`。

**标签模式。** 默认 `verified` 使用 `label_*_5y`；登记覆盖未核实时这些标签为空，训练会直接报错退出。`--label-mode provisional_observed` 把登记已观察到的首癌当阳性、其余当阴性，删失者被当作阴性，**只用于调试训练流程，结果不能报告**；两种模式的输出分目录保存，评估不会混用。当前二分类标签排除了删失者；与删失/竞争风险相容的评价（如 IPCW）需在覆盖核实后补充。

人群定义需要先有登记、住院和自报的既往癌症信息，所以放在第 05 步；第 01 步只建立研究母体和固定划分（先划分参与者，再生成预测点样本）。各步之间只通过 `data/processed/` 下的 parquet 文件传递数据，因此可单独重跑；改动某一步后需重跑它及之后的步骤（`run_all.py --from NN`），任一步开始时都会删除 `BUILD_COMPLETE.json`，直到第 08 步重新写入。

每一步会打印标题、各阶段日志（带累计用时）、大文件读取和逐预测点/逐癌种的进度条，结束时打印关键计数；第 05 步打印每个预测点的逐条排除流程。输出重定向到文件时（如 `nohup ... > build.log`），进度条约每分钟刷新一次。每步的质控计数写入汇总目录 `build_summary.json` 的 `step_NN` 下，隔离记录写入 `data/processed/quarantine/NN.parquet`。

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

`landmark_inputs` 是 EHR-only 模型输入序列，与 `landmark_samples` 一一对应；`event_{site}` 与 `followup_days` 给出竞争风险事件码。各划分组的均衡性见汇总目录 `split_summary.csv`。`python -m pytest` 包含一个合成数据的端到端构建与校验测试。

字段与质控规则见 [数据表说明](docs/data_tables.md)。
