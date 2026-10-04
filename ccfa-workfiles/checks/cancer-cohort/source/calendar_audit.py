"""Compare calendar landmarks; aggregate outputs only, using existing cancer mapping."""
from pathlib import Path
import ast,json,argparse
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
ns={'pd':pd}
for node in ast.parse((ROOT/'source/audit.py').read_text(encoding='utf-8')).body:
    if ((isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='SITES' for x in node.targets)) or (isinstance(node,ast.FunctionDef) and node.name=='classify')):
        exec(compile(ast.Module(body=[node],type_ignores=[]),'audit.py','exec'),ns)
SITES=ns['SITES'];classify=ns['classify']
parser=argparse.ArgumentParser()
parser.add_argument('--ukb-fields',type=Path,required=True)
parser.add_argument('--hospital-cancer',type=Path,required=True)
args=parser.parse_args();E=args.ukb_fields;L=args.hospital_cancer
def dates(s):return pd.to_datetime(s,format='mixed',errors='coerce')
c=pd.read_csv(L/'cancer.csv',dtype=str).set_index('Participant ID')
v=pd.read_csv(E/'UKB_visit_and_followup_dates.csv',dtype=str).set_index('Participant ID')
d=pd.read_csv(E/'UKB_death.csv',dtype=str)
p=pd.DataFrame(index=v.index)
p['baseline']=dates(v['Date of attending assessment centre | Instance 0']);p['lost']=dates(v['Date lost to follow-up']);p['sex']=c.Sex
p['death']=d.assign(date=dates(d['Date of death'])).groupby('Participant ID').date.min()
parts=[]
for i in range(22):
    z=pd.DataFrame({'icd10':c[f'Type of cancer: ICD10 | Instance {i}'],'icd9':c.get(f'Type of cancer: ICD9 | Instance {i}',pd.Series('',index=c.index)),'date':dates(c[f'Date of cancer diagnosis | Instance {i}']),'sex':c.Sex})
    z=z[z.date.notna()].copy();z['site'],z['malignant'],_=classify(z.icd10,z.icd9,z.sex);parts.append(z.reset_index())
r=pd.concat(parts,ignore_index=True).rename(columns={'Participant ID':'id'})
p['first_any']=r[r.malignant].groupby('id').date.min()
f=r[r.site.ne('')].groupby(['id','site']).date.min().unstack()
for key,*_ in SITES:p[key]=f[key]
landmarks=[f'{y}-01-01' for y in range(2010,2017)]
site_rows=[];summary=[];sets={};eligible_sets={}
for label in ['baseline']+landmarks:
    t=p.baseline if label=='baseline' else pd.Series(pd.Timestamp(label),index=p.index)
    end=t+pd.DateOffset(years=5)
    alive=p.baseline.le(t)&(p.death.isna()|p.death.gt(t))&(p.lost.isna()|p.lost.gt(t))
    risk=alive&(p.first_any.isna()|p.first_any.gt(t))
    unique=set();eligible_sets[label]=set(p.index[risk])
    for key,name,_,_,sex in SITES:
        pop=risk if sex is None else risk&p.sex.eq(sex)
        tf=p[key];obs=(p.death.isna()|tf.le(p.death))&(p.lost.isna()|tf.le(p.lost))
        ev=pop&tf.gt(t)&tf.le(end)&tf.eq(p.first_any)&obs
        # Alternative: target-specific cancer-free population, allowing previous other cancers.
        pop2=alive&(tf.isna()|tf.gt(t));pop2=pop2 if sex is None else pop2&p.sex.eq(sex)
        ev2=pop2&tf.gt(t)&tf.le(end)&obs
        ids=set(p.index[ev]);ids2=set(p.index[ev2]);sets[(label,key)]=ids;sets[(label,key,'target')]=ids2;unique|=ids
        site_rows.append({'landmark':label,'site':key,'name':name,'common_cancer_free_candidates':int(pop.sum()),'first_any_cases_5y':len(ids),'target_only_cancer_free_candidates':int(pop2.sum()),'first_target_cases_5y':len(ids2)})
    summary.append({'landmark':label,'recruited_alive_observed_candidates':int(alive.sum()),'common_cancer_free_candidates':int(risk.sum()),'top10_unique_first_any_cases_5y':len(unique),'not_yet_recruited':int(p.baseline.gt(t).sum())})
strategies={'2010+2015':['2010-01-01','2015-01-01'],'2011+2015':['2011-01-01','2015-01-01'],'2011+2016':['2011-01-01','2016-01-01'],'annual2010-2015':[f'{y}-01-01' for y in range(2010,2016)],'baseline+2010+2015':['baseline','2010-01-01','2015-01-01'],'baseline+2015':['baseline','2015-01-01']}
comb=[];ss=[]
for strategy,ls in strategies.items():
    union=set();events=0
    for key,name,*_ in SITES:
        ids=set.union(*(sets[(l,key)] for l in ls));ids2=set.union(*(sets[(l,key,'target')] for l in ls));n=sum(len(sets[(l,key)]) for l in ls)
        union|=ids;events+=n
        comb.append({'strategy':strategy,'site':key,'name':name,'unique_first_any_cases':len(ids),'positive_windows':n,'unique_first_target_cases':len(ids2)})
    ss.append({'strategy':strategy,'unique_first_any_cases':len(union),'positive_windows_summed_across_sites':events,'unique_cancer_free_candidates':len(set.union(*(eligible_sets[l] for l in ls)))})
pd.DataFrame(site_rows).to_csv(ROOT/'calendar_cancer_counts.csv',index=False)
pd.DataFrame(summary).to_csv(ROOT/'calendar_summary.csv',index=False)
pd.DataFrame(comb).to_csv(ROOT/'calendar_strategy_counts.csv',index=False)
pd.DataFrame(ss).to_csv(ROOT/'calendar_strategy_summary.csv',index=False)
# Every same-site positive in the two disjoint five-year windows is a distinct person.
for key,*_ in SITES:assert not (sets[('2010-01-01',key)]&sets[('2015-01-01',key)])
print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(ss).to_string(index=False));print(pd.DataFrame(comb)[pd.DataFrame(comb).strategy.eq('2010+2015')].to_string(index=False))
