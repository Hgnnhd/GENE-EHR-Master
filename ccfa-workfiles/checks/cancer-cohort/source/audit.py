"""Aggregate-only EHR/cancer feasibility audit; raw inputs are never modified."""
from pathlib import Path
import argparse
import json
import re
import time
import numpy as np
import pandas as pd
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
P = argparse.ArgumentParser()
P.add_argument('--ukb-fields', type=Path, required=True)
P.add_argument('--hospital-cancer', type=Path, required=True)
ARGS = P.parse_args()
EHR, OLD = ARGS.ukb_fields, ARGS.hospital_cancer
START = time.time()
RESULT = {}

# GLOBOCAN 2024 world factsheet, accessed 2026-10-04; both-sex death counts.
SITES = [
    ('lung', '肺癌', [33,34], [162], None),
    ('colorectal', '结直肠癌', [18,19,20,21], [153,154], None),
    ('liver', '肝癌及肝内胆管癌', [22], [155], None),
    ('breast', '女性乳腺癌', [50], [174], 'Female'),
    ('stomach', '胃癌', [16], [151], None),
    ('pancreas', '胰腺癌', [25], [157], None),
    ('oesophagus', '食管癌', [15], [150], None),
    ('prostate', '前列腺癌', [61], [185], 'Male'),
    ('leukaemia', '白血病', list(range(91,96)), list(range(204,209)), None),
    ('cervix', '宫颈癌', [53], [180], 'Female'),
]

def log(x):
    print(f'[{time.time()-START:.1f}s] {x}', flush=True)

def read(path, **kw):
    return pd.read_csv(path, dtype=str, encoding='utf-8-sig', **kw)

def dates(s):
    return pd.to_datetime(s, format='mixed', errors='coerce')

def summary_date(s):
    s = s.dropna()
    return {'n':len(s), 'min':str(s.min().date()) if len(s) else None,
            'max':str(s.max().date()) if len(s) else None}

def quant(s):
    return {str(k):float(v) for k,v in pd.Series(s).quantile([0,.25,.5,.75,.9,.95,1]).items()}

def out(name, d):
    d.to_csv(ROOT / name, index=False, encoding='utf-8')

def classify(c10, c9, sex):
    n10 = pd.to_numeric(c10.str.extract(r'^C(\d{2})', expand=False), errors='coerce')
    n9 = pd.to_numeric(c9.str.extract(r'^(\d{3})', expand=False), errors='coerce')
    fallback = c10.isna() | c10.eq('')
    site = pd.Series('', index=c10.index)
    for key,_,codes,oldcodes,restrict in SITES:
        m = n10.isin(codes) | (fallback & n9.isin(oldcodes))
        if restrict:
            m &= sex.eq(restrict)
        site.loc[m] = key
    malignant = (n10.between(0,97) & n10.ne(44)) | (fallback & n9.between(140,208) & n9.ne(173))
    return site, malignant, fallback & n9.notna()

log('Inventory: scanning row counts and unique participant IDs')
inventory=[]
for path in list(EHR.glob('UKB_*.csv'))+[OLD/'cancer.csv',OLD/'record.csv']:
    lf=pl.scan_csv(path,infer_schema_length=0)
    cols=lf.collect_schema().names()
    a=lf.select(pl.len().alias('rows'),pl.col('Participant ID').n_unique().alias('unique_ids'),
                pl.col('Participant ID').is_null().sum().alias('null_ids')).collect().row(0)
    inventory.append({'file':path.name,'source':'ehr' if path.parent==EHR else 'legacy',
                      'rows':a[0],'unique_ids':a[1],'missing_ids':a[2],'columns':len(cols),
                      'bytes':path.stat().st_size})
out('inventory.csv',pd.DataFrame(inventory))

additional={}
for fn,pat in [('UKB_hesin.csv','date'),('UKB_gp_registrations.csv','date'),
               ('UKB_treatment_medication_codes.csv','Treatment/medication code'),
               ('UKB_self_reported_conditions.csv','code, self-reported'),
               ('UKB_algorithmically_defined_outcomes.csv','Date')]:
    lf=pl.scan_csv(EHR/fn,infer_schema_length=0)
    cols=[x for x in lf.collect_schema().names() if pat.lower() in x.lower()]
    t=lf.select(pl.col('Participant ID'),pl.sum_horizontal([pl.col(x).is_not_null().cast(pl.Int32) for x in cols]).alias('n')).collect()
    valid=t.filter(pl.col('n')>0)
    additional[fn]={'counted_columns':len(cols),'nonempty_cells':int(t['n'].sum()),
                    'rows_with_any':valid.height,'participants_with_any':valid['Participant ID'].n_unique()}
(ROOT/'additional_source_counts.json').write_text(json.dumps(additional,ensure_ascii=False,indent=2),encoding='utf-8')

vis=read(EHR/'UKB_visit_and_followup_dates.csv').set_index('Participant ID')
assert vis.index.is_unique
cohort=pd.DataFrame(index=vis.index)
cohort['baseline']=dates(vis['Date of attending assessment centre | Instance 0'])
cohort['lost']=dates(vis['Date lost to follow-up'])
birth=read(OLD/'record.csv',usecols=['Participant ID','Date of birth','Sex']).set_index('Participant ID')
assert birth.index.is_unique
cohort['birth']=dates(birth['Date of birth']).reindex(cohort.index)
sex_lookup=read(OLD/'cancer.csv',usecols=['Participant ID','Sex']).set_index('Participant ID')['Sex']
cohort['sex']=sex_lookup.reindex(cohort.index)
dd=read(EHR/'UKB_death.csv')
dd['date']=dates(dd['Date of death'])
cohort['death']=dd.groupby('Participant ID')['date'].min().reindex(cohort.index)
RESULT['cohort']={'participants':len(cohort),'birth_overlap':int(cohort.birth.notna().sum()),
    'record_only_ids':int((~birth.index.isin(cohort.index)).sum()),
    'baseline_dates':summary_date(cohort.baseline),'birth_dates':summary_date(cohort.birth),
    'birth_day_distribution':cohort.birth.dt.day.value_counts().to_dict(),
    'sex_counts':cohort.sex.value_counts(dropna=False).to_dict(),
    'legacy_record_sex_crosswalk':pd.crosstab(birth['Sex'].reindex(cohort.index),cohort.sex).to_dict(),
    'death_dates':summary_date(cohort.death),'lost_dates':summary_date(cohort.lost)}

log('Cancer registry: pairing dates and ICD codes by Instance')
c=read(OLD/'cancer.csv').set_index('Participant ID')
assert c.index.is_unique
RESULT['cancer_id_overlap']={'source_ids':len(c),'matched_ids':int(c.index.isin(cohort.index).sum()),
                             'legacy_only_ids':int((~c.index.isin(cohort.index)).sum())}
parts=[]
for i in range(22):
    dc=f'Date of cancer diagnosis | Instance {i}'
    a=c[[dc,f'Type of cancer: ICD10 | Instance {i}']].copy()
    a.columns=['raw_date','icd10']
    a['icd9']=c.get(f'Type of cancer: ICD9 | Instance {i}',pd.Series(index=c.index,dtype=object))
    a['sex']=c.Sex
    a=a.loc[a[['raw_date','icd10','icd9']].notna().any(axis=1)].copy()
    a['instance']=i
    parts.append(a.reset_index())
reg=pd.concat(parts,ignore_index=True).rename(columns={'Participant ID':'id'})
reg['date']=dates(reg.raw_date)
reg['site'],reg['malignant'],reg['icd9_fallback']=classify(reg.icd10,reg.icd9,reg.sex)
RESULT['cancer_registry']={'entries':len(reg),'patients':reg.id.nunique(),
    'date_parse_failures':int((reg.raw_date.notna()&reg.date.isna()).sum()),
    'code_without_date':int((reg.date.isna()&(reg.icd10.notna()|reg.icd9.notna())).sum()),
    'date_without_code':int((reg.date.notna()&reg.icd10.isna()&reg.icd9.isna()).sum()),
    'icd9_fallback_entries':int(reg.icd9_fallback.sum()),'dates':summary_date(reg.date),
    'malignant_excluding_C44_patients':reg.loc[reg.malignant,'id'].nunique(),
    'malignant_excluding_C44_entries':int(reg.malignant.sum()),
    'top10_patients':reg.loc[reg.site.ne(''),'id'].nunique(),
    'top10_registry_entries':int(reg.site.ne('').sum())}
reg=reg.loc[reg.id.isin(cohort.index)].copy()
cohort['first_cancer']=reg.loc[reg.malignant].groupby('id').date.min().reindex(cohort.index)
sitefirst=reg.loc[reg.site.ne('')].groupby(['id','site']).date.min().unstack().reindex(cohort.index)
for key,*_ in SITES:
    cohort[key]=sitefirst.get(key,pd.Series(pd.NaT,index=cohort.index))

causes=read(EHR/'UKB_death_cause.csv')
causes['sex']=causes['Participant ID'].map(cohort.sex)
causes['site'],causes['malignant'],_=classify(causes['Cause of death - ICD-10'],pd.Series('',index=causes.index),causes.sex)
RESULT['death_cause_classifications']=causes['Classification of cause of death'].value_counts().to_dict()
primary=causes.loc[causes['Classification of cause of death'].eq('Primary cause of death')]
RESULT['death_cause_primary_unique_patients']=primary['Participant ID'].nunique()
RESULT['death_cause_primary_cancer_patients']=primary.loc[primary.malignant,'Participant ID'].nunique()
rows=[]
for rank,(key,name,codes,oldcodes,restrict) in enumerate(SITES,1):
    r=reg.loc[reg.site.eq(key)]
    first=cohort[key]
    after=first.gt(cohort.baseline)
    firstprimary=after & first.eq(cohort.first_cancer)
    rows.append({'global_mortality_rank':rank,'site':key,'name':name,'icd10':','.join(f'C{x:02}' for x in codes),
        'registry_entries':len(r),'ever_patients':r.id.nunique(),
        'valid_date_patients':int(first.notna().sum()),
        'prevalent_at_baseline':int(first.le(cohort.baseline).sum()),
        'first_site_after_baseline':int(after.sum()),'first_any_cancer_after_baseline':int(firstprimary.sum()),
        'baseline_5y_first_any_cancer':int((firstprimary&first.le(cohort.baseline+pd.DateOffset(years=5))).sum()),
        'primary_cause_deaths':primary.loc[primary.site.eq(key),'Participant ID'].nunique(),
        'any_mention_deaths':causes.loc[causes.site.eq(key),'Participant ID'].nunique(),
        'icd9_fallback_entries':int(r.icd9_fallback.sum()),
        'first_date':str(first.min().date()) if first.notna().any() else None,
        'last_date':str(first.max().date()) if first.notna().any() else None})
out('cancer_counts.csv',pd.DataFrame(rows))
out('cancer_registry_years.csv',reg.loc[reg.date.notna()].assign(year=lambda x:x.date.dt.year).groupby('year').agg(entries=('id','size'),patients=('id','nunique')).reset_index())

log('First occurrences: creating dated disease events in memory')
fparts=[]
nonempty=bad=0
for chunk in pd.read_csv(EHR/'UKB_First_occurrences.csv',dtype=str,chunksize=10000):
    z=chunk.set_index('Participant ID').stack().rename('raw').reset_index()
    z.columns=['id','field','raw']
    nonempty+=len(z)
    z['date']=dates(z.raw)
    bad+=int(z.date.isna().sum())
    z['code']=z.field.str.extract(r'^Date ([A-Z][0-9]{2}) ',expand=False)
    fparts.append(z[['id','code','date']].dropna())
fo=pd.concat(fparts,ignore_index=True)
RESULT['first_occurrences']={'nonempty_date_cells':nonempty,'invalid_dates':bad,
    'valid_events':len(fo),'patients':fo.id.nunique(),'codes':fo.code.nunique(),
    'dates':summary_date(fo.date)}

log('Hospital diagnoses: matching pipe-separated codes to numbered date arrays')
rparts=[]
mismatch=code_total=date_total=code_without_date=invalid_dates=0
for chunk in pd.read_csv(OLD/'record.csv',dtype=str,chunksize=10000):
    chunk=chunk.set_index('Participant ID')
    cs=chunk['Diagnoses - ICD10'].str.split('|',regex=False)
    lens=cs.str.len().fillna(0)
    ds=chunk.filter(regex=r'^Date of first in-patient diagnosis - ICD10 \| Array ')
    ds=ds.reindex(columns=sorted(ds.columns,key=lambda x:int(x.rsplit(' ',1)[1])))
    dlens=ds.notna().sum(axis=1)
    mismatch+=int(lens.ne(dlens).sum()); code_total+=int(lens.sum());date_total+=int(dlens.sum())
    codes=cs.explode().dropna().rename('raw_code').reset_index()
    codes['slot']=codes.groupby('Participant ID').cumcount()
    codes['code']=codes.raw_code.str.extract(r'^([A-Z][0-9]{2})',expand=False)
    ds.columns=range(len(ds.columns))
    dl=ds.stack().rename('raw_date').reset_index();dl.columns=['Participant ID','slot','raw_date']
    z=codes.merge(dl,on=['Participant ID','slot'],how='left',validate='one_to_one')
    code_without_date+=int(z.raw_date.isna().sum());z['date']=dates(z.raw_date)
    invalid_dates+=int((z.raw_date.notna()&z.date.isna()).sum())
    rparts.append(z.rename(columns={'Participant ID':'id'})[['id','code','date']].dropna())
hr=pd.concat(rparts,ignore_index=True)
RESULT['hospital_diagnoses']={'codes':code_total,'date_cells':date_total,'count_mismatch_patients':mismatch,
    'codes_without_date':code_without_date,'invalid_date_values':invalid_dates,'valid_events':len(hr),
    'patients':hr.id.nunique(),'icd3_codes':hr.code.nunique(),'dates':summary_date(hr.date)}

log('Combining earliest ICD-10 three-character events; calculating landmark histories')
ev=pd.concat([fo,hr],ignore_index=True)
ev=ev.loc[ev.id.isin(cohort.index)].groupby(['id','code'],as_index=False).date.min()
idx=cohort.index.get_indexer(ev.id)
birthv=cohort.birth.to_numpy()[idx]
deathv=cohort.death.to_numpy()[idx]
dv=ev.date.to_numpy()
prebirth=dv<birthv
postdeath=dv>deathv
RESULT['combined_history']={'unique_patient_icd3_events_before_date_qc':len(ev),
    'before_birth':int(prebirth.sum()),'after_death':int(postdeath.sum()),
    'unknown_birth':int(np.isnat(birthv).sum())}
# Drop events known to precede birth or follow death; preserve no-event participants.
ev=ev.loc[~prebirth&~postdeath].copy()
idx=cohort.index.get_indexer(ev.id);dv=ev.date.to_numpy()
RESULT['combined_history'].update({'valid_first_icd3_events':len(ev),'patients':ev.id.nunique(),
    'icd3_codes':ev.code.nunique(),'dates':summary_date(ev.date)})
cohort['history_total']=np.bincount(idx,minlength=len(cohort))
RESULT['combined_history']['events_per_participant']=quant(cohort.history_total)
RESULT['combined_history']['zero_history_participants']=int(cohort.history_total.eq(0).sum())

# Observed maximum registry date is only a data envelope, NOT an administrative censoring date.
envelope=reg.date.max()
RESULT['landmark_assumptions']={'registry_observed_date_envelope':str(envelope.date()),
    'administrative_registry_censoring_date':'UNKNOWN; no claim of complete 5-year follow-up',
    'all_cancer_exclusion':'registry C00-C97 excluding C44; ICD9 140-208 excluding 173 fallback',
    'history':'earliest date per participant and ICD10 three-character code across first occurrences + hospital diagnoses',
    'birth_precision':'see birth_day_distribution; birthdays treated as stored dates',
    'event_counts':'registry-observed first site diagnoses; first-any endpoints counted only if tied for first non-C44 malignant diagnosis'}
age_rows=[];hist_rows=[]
for age in [0,40,50,55,60,65,70]:
    anchor=cohort.baseline if age==0 else cohort.birth+pd.DateOffset(years=age)
    candidate=(anchor.notna()&cohort.baseline.notna()&anchor.ge(cohort.baseline)&anchor.le(envelope)
               &(cohort.death.isna()|cohort.death.gt(anchor))
               &(cohort.lost.isna()|cohort.lost.gt(anchor)))
    risk=candidate&(cohort.first_cancer.isna()|cohort.first_cancer.gt(anchor))
    av=anchor.to_numpy()[idx]
    before=dv<=av
    counts=np.bincount(idx[before],minlength=len(cohort))
    prior1=before&(dv>(anchor-pd.DateOffset(years=1)).to_numpy()[idx])
    counts1=np.bincount(idx[prior1],minlength=len(cohort))
    days=ev.loc[before,['id','date']].drop_duplicates().groupby('id').size().reindex(cohort.index,fill_value=0)
    firsthist=ev.loc[before].groupby('id').date.min().reindex(cohort.index)
    span=(anchor-firsthist).dt.days/365.25
    hist_rows.append({'age':'baseline' if age==0 else age,'candidate_alive_observed_envelope':int(candidate.sum()),
        'cancer_free_candidates':int(risk.sum()),'no_prior_history':int((risk&(counts==0)).sum()),
        'at_least_5_codes':int((risk&(counts>=5)).sum()),'at_least_10_codes':int((risk&(counts>=10)).sum()),
        'at_least_2_event_days':int((risk&days.ge(2)).sum()),
        'median_history_codes':float(np.median(counts[risk])) if risk.any() else None,
        'median_event_days':float(days[risk].median()),'median_history_span_years':float(span[risk].median()),
        'median_codes_last_year':float(np.median(counts1[risk])) if risk.any() else None,
        'zero_codes_last_year':int((risk&(counts1==0)).sum()),
        'anchor_plus_5y_within_observed_date_envelope':int((risk&(anchor+pd.DateOffset(years=5)).le(envelope)).sum())})
    for key,name,_,_,sex in SITES:
        eligible=risk if sex is None else risk&cohort.sex.eq(sex)
        t=cohort[key]
        site_event=eligible&t.gt(anchor)&t.le(anchor+pd.DateOffset(years=5))
        firstevent=site_event&t.eq(cohort.first_cancer)
        observable=(cohort.death.isna()|t.le(cohort.death))&(cohort.lost.isna()|t.le(cohort.lost))
        age_rows.append({'age':'baseline' if age==0 else age,'site':key,'name':name,
            'cancer_free_candidates':int(eligible.sum()),
            'observed_5y_first_site_cases':int((site_event&observable).sum()),
            'observed_5y_first_any_cancer_cases':int((firstevent&observable).sum()),
            'first_any_cases_1_to_5y':int((firstevent&observable&t.gt(anchor+pd.DateOffset(years=1))).sum()),
            'first_any_cases_at_least_5_history_codes':int((firstevent&observable&(counts>=5)).sum()),
            'first_any_cases_at_least_10_history_codes':int((firstevent&observable&(counts>=10)).sum())})
out('landmark_history.csv',pd.DataFrame(hist_rows));out('landmark_cancer_counts.csv',pd.DataFrame(age_rows))
RESULT['registry_death_inconsistency']={'registry_entries_after_death':int((reg.date>reg.id.map(cohort.death)).sum()),
    'registry_entries_before_birth':int((reg.date<reg.id.map(cohort.birth)).sum())}
(ROOT/'summary.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
log('Done; aggregate CSV and JSON outputs written')
print(pd.DataFrame(rows).to_string(index=False),flush=True)
print(pd.DataFrame(hist_rows).to_string(index=False),flush=True)
