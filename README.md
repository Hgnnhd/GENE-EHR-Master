# EHR 预训练与癌症风险预测

独立数据管线：UK Biobank 首次医疗代码历史 → 预训练数据 → 2011、2016 两个固定预测点的首次癌症风险队列。

研究定义见 [人群与数据协议](docs/cohort_protocol.md)。当前阶段构建数据，不启动模型训练。

```bash
python -m pip install -e .
python -m cancer_ehr.build --ehr "$EHR_SOURCE" --legacy "$LEGACY_EHR_SOURCE"
python -m cancer_ehr.validate
python -m pytest -q
```

也可不安装，以 `PYTHONPATH=src python -m cancer_ehr.build ...` 运行。原始路径通过命令行传入，原始文件保持不变。

- 配置：[configs/cohort.json](configs/cohort.json)。固定患者划分 70/15/15，seed 42。
- 私有数据：`data/processed/`，已加入 `.gitignore`。
- 汇总、病例流程及质控：`ccfa-workfiles/checks/cancer-cohort/`。
- 重新构建覆盖当前派生表；只有 `BUILD_COMPLETE.json` 存在才表示整个构建完成。

`landmark_samples` 保存通过临床病史筛查的候选节点。是否已核实可入组见 `eligible`；能否用于监督学习见 `followup_status` 和 `label_*`。`observed_*` 是登记中已看到的病例统计，不能作为完整二分类训练标签。

癌症登记覆盖日期必须与本地导出版本相符。当前 [覆盖配置](configs/registry_coverage.json) 标记为 `unverified`，因此候选者的五年训练标签保持空值。核实覆盖后提供逐参与者的 `registry_start`、`registry_end_exclusive`、数据版本与证据，再重新构建；地区归属不能仅凭癌症病例来源推给所有对照。

`pretrain_events` 仅含训练参与者在 2016 年之前的记录，内部早停集仍属于训练参与者；词表仅从内部预训练训练组生成。下游验证、测试患者不进入预训练。旧模型 checkpoint 暂不迁入。

增强特征保存原始值、评估 Instance、测量日期、距预测点天数和缺失标志。化验时间采用评估日期，当前导出没有结果发布时刻；这是回顾性测量特征，不能声称能复原当天临床可获取的化验结果。

`landmark_inputs` 是 EHR-only 模型输入序列，与 `landmark_samples` 一一对应；`event_{site}` 与 `followup_days` 给出竞争风险事件码。各划分组的均衡性见汇总目录 `split_summary.csv`。`python -m pytest` 包含一个合成数据的端到端构建与校验测试。

字段与质控规则见 [数据表说明](docs/data_tables.md)。
