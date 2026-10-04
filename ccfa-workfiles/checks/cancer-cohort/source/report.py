"""Render the aggregate feasibility report and check count invariants."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'summary.json').read_text(encoding='utf-8'))
sel=json.loads((ROOT/'cancer_selection.json').read_text(encoding='utf-8'))
c=pd.read_csv(ROOT/'cancer_counts.csv')
h=pd.read_csv(ROOT/'landmark_history.csv',dtype={'age':str})
a=pd.read_csv(ROOT/'landmark_cancer_counts.csv',dtype={'age':str})
inv=pd.read_csv(ROOT/'inventory.csv')
extra=json.loads((ROOT/'additional_source_counts.json').read_text(encoding='utf-8'))
assert inv.missing_ids.eq(0).all()
assert c.valid_date_patients.le(c.ever_patients).all()
assert (c.prevalent_at_baseline+c.first_site_after_baseline).le(c.valid_date_patients).all()
assert c.first_any_cancer_after_baseline.le(c.first_site_after_baseline).all()
assert a.observed_5y_first_any_cancer_cases.le(a.observed_5y_first_site_cases).all()
assert a.observed_5y_first_site_cases.le(a.cancer_free_candidates).all()
assert a.first_any_cases_1_to_5y.le(a.observed_5y_first_any_cancer_cases).all()
assert h.at_least_10_codes.le(h.at_least_5_codes).all()
assert h.at_least_5_codes.le(h.cancer_free_candidates-h.no_prior_history).all()
assert s['cancer_id_overlap']['legacy_only_ids']==0

def table(d):
    # Avoid optional rendering dependencies.
    def fmt(x):
        if pd.isna(x):return '—'
        if isinstance(x,float):return f'{x:,.2f}'
        if isinstance(x,int):return f'{x:,}'
        return str(x)
    return '| '+' | '.join(d.columns)+' |\n| '+' | '.join(['---']*len(d.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(v) for v in row)+' |' for row in d.itertuples(index=False,name=None))

parts=['# EHR 与十大高死亡负担癌种：样本量及年龄预测点可行性',
       '统计日期：2026-10-04。所有数字来自全量 CSV 扫描；原始文件未修改，输出仅含汇总。',
       '## 癌种选择',
       f'暂按全球、双性别、全年龄癌症死亡人数前十选择，而非病死率。来源：[{sel["edition"]}]({sel["source"]})。保留了下载的官方 PDF 及 SHA-256，详见 cancer_selection.json。',
       '肺癌口径 C33–C34；结直肠口径 C18–C21（含肛门癌）；肝癌含肝内胆管癌；乳腺癌本轮取女性。日期与编码按登记 Instance 配对，ICD-10 缺失时使用 ICD-9 的对应癌种大类补足。',
       '## 数据规模',table(inv[['file','rows','unique_ids','columns']]),
       f'First occurrences 有 {s["first_occurrences"]["valid_events"]:,} 个有效疾病日期单元格，涉及 {s["first_occurrences"]["patients"]:,} 人；旧住院诊断表有 {s["hospital_diagnoses"]["valid_events"]:,} 条有效代码—日期配对，涉及 {s["hospital_diagnoses"]["patients"]:,} 人。',
       f'两者按人和 ICD-10 三级码取最早日期，并删除已知出生前／死亡后的日期，得到 {s["combined_history"]["valid_first_icd3_events"]:,} 条疾病首次记录，覆盖 {s["combined_history"]["patients"]:,} 人。它不是完整重复就诊序列。住院 episode 也不等于独立患者或独立住院次数。',
       table(pd.DataFrame([{'来源':k,**v} for k,v in extra.items()])),
       '## 癌症登记与死亡',
       f'登记表共有 {s["cancer_registry"]["entries"]:,} 个条目、{s["cancer_registry"]["patients"]:,} 名参与者（含原位癌等非浸润性病变）。按 C00–C97 排除 C44 并补 ICD-9，共 {s["cancer_registry"]["malignant_excluding_C44_patients"]:,} 人。十大癌种合并去重后为 {s["cancer_registry"]["top10_patients"]:,} 人、{s["cancer_registry"]["top10_registry_entries"]:,} 个登记条目。',
       table(c[['global_mortality_rank','name','icd10','ever_patients','prevalent_at_baseline','first_site_after_baseline','first_any_cancer_after_baseline','primary_cause_deaths']].rename(columns={'global_mortality_rank':'全球死亡人数排名','name':'癌种','icd10':'ICD-10','ever_patients':'历年登记患者','prevalent_at_baseline':'招募前或当日已登记','first_site_after_baseline':'招募后首次该癌种','first_any_cancer_after_baseline':'招募后首次恶性肿瘤即该癌种','primary_cause_deaths':'该癌种为主要死因人数'})),
       '各癌种患者可能重叠，不应把逐癌种人数直接当作独立患者总数。死亡统计来自死亡原因表，非对癌症登记患者的病死率估计；死亡和癌症登记来源的覆盖期不同。',
       '## 年龄节点：病例和历史',
       '以下是可行性盘点，不是最终训练队列。年龄节点以所存出生日期加年龄构造；招募后、未死亡、未记载失访、且在癌症登记观测日期范围内者为候选；再排除登记中已有的非 C44 恶性肿瘤。尚未用自报病史及住院肿瘤代码全面核验既往癌症。',
       '出生日期日位分布：'+json.dumps(s['cohort']['birth_day_distribution'],ensure_ascii=False)+'。据此生日为月份精度近似，精确日界线尚不能解释。',
       table(h.rename(columns={'age':'节点','cancer_free_candidates':'无癌候选人数','median_history_codes':'病史代码数中位数','no_prior_history':'无既往疾病记录人数'})),
       '五年内观察到的首次非 C44 恶性肿瘤，按癌种与预测年龄：',
       table(a.pivot(index='name',columns='age',values='observed_5y_first_any_cancer_cases').reindex(c.name).reset_index().rename(columns={'name':'癌种'})),
       '不同节点可由同一参与者贡献；同一病例可能出现在不同节点的五年窗口中，不能跨节点直接相加作为独立病例数。乳腺、宫颈限女性，前列腺限男性。',
       '## 随访与质量边界',
       f'癌症登记观察到的最晚诊断日期：{s["cancer_registry"]["dates"]["max"]}；死亡日期范围：{s["cohort"]["death_dates"]["min"]} 至 {s["cohort"]["death_dates"]["max"]}。癌症登记各年条目见 cancer_registry_years.csv。',
       '**最大记录日期不是行政随访截止日。** 尚未确认分地区的癌症登记覆盖截止日期，不能将候选人数解释为具有完整五年随访的人数，也不能据此直接计算五年发生率或把其余人全标成阴性。',
       '晚年龄节点尤其受记录覆盖期影响。当前阳性数只计登记确诊日期不晚于已记录死亡／失访日期的事件；最终需统一登记、死亡、失访和竞争事件规则。',
       '出生前或死亡后的诊断日期质控、日期缺失、代码配对及性别编码映射均见 summary.json。旧 record.csv 的性别数值为本文件自定义编码，已通过参与者 ID 与癌症表文字性别核对，未套用常见 0/1 假设。',
       f'First occurrences 有 {s["first_occurrences"]["invalid_dates"]:,} 个不可解析日期单元格，已不计入有效日期数；住院诊断有 {s["hospital_diagnoses"]["count_mismatch_patients"]} 人代码数与日期数不一致，{s["hospital_diagnoses"]["codes_without_date"]} 个代码缺失对应日期。统计保留按原 Array 编号能匹配的条目，正式建模前需进一步核验这些不一致个体。癌症登记有 {s["cancer_registry"]["date_without_code"]} 个有日期但无编码条目；有 {s["registry_death_inconsistency"]["registry_entries_after_death"]} 个登记日期晚于已记录死亡日期。',
       '## 对预测点的判断',
       '40 岁节点仅有 5 名候选参与者，不能支持主任务。50 岁节点的大部分稀有癌种五年病例不足 50；60/65/70 岁节点具有更多可观察事件，但必须先确认分地区的癌症登记截止日期才能形成正式风险集。',
       '优先比较乳腺、前列腺、结直肠和肺癌；其他癌种可作为共享模型的次要终点。宫颈癌招募后首次登记只有 142 人，在 50/60/70 岁五年窗口中分别仅有 18/19/15 例首次恶性肿瘤事件，不宜承诺稳定的独立年龄分层评估。',
       '50/60/70 岁无癌候选者的既往首次医疗代码数中位数分别为 5/6/9，最近一年没有新代码的比例约为 79.8%/78.9%/74.3%。因此应保留较长历史，不宜直接照搬 RAVEN 的一年窗口；这里统计的是首次代码记录，不能解读为这些人一年内没有就诊。',
       '## 复现与结果文件',
       '`python source/audit.py --ukb-fields <UKB字段导出目录> --hospital-cancer <住院诊断与癌症登记目录>`\n\n`python source/report.py`',
       '输出：inventory.csv、additional_source_counts.json、cancer_counts.csv、landmark_history.csv、landmark_cancer_counts.csv、cancer_registry_years.csv、summary.json。计数关系检查通过。']
(ROOT/'report.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
print('Aggregate invariants passed; report.md written.')
