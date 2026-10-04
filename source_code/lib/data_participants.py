"""Data stage · Study base population and fixed patient splits.

One row per UKB participant: recruitment, sex, birth month, death, loss to follow-up,
registry coverage, the fixed 70/15/15 train/validation/test split and the internal
pretraining roles. Landmark eligibility is applied later, in the cohort stage.
"""
import json

import numpy as np
import pandas as pd

from .common import ID, ROOT, Context, fmt, log, read
from .definitions import dates, patient_split

STAGE, TITLE = "participants", "Study base population and fixed patient splits"
NEEDS = [("ukb_fields", "UKB_visit_and_followup_dates.csv"), ("ukb_fields", "UKB_death.csv"),
         ("hospital_cancer", "record.csv"), ("hospital_cancer", "cancer.csv")]


def run(args):
    ctx = Context(args, STAGE, TITLE, NEEDS)
    log("Reading visit / follow-up dates")
    vis = ctx.visits()
    assert vis.index.is_unique and vis.index.notna().all()
    p = pd.DataFrame(index=vis.index)
    p["recruited"] = dates(vis["Date of attending assessment centre | Instance 0"])
    p["lost"] = dates(vis["Date lost to follow-up"])

    log("Reading birth month and sex (record.csv, cancer.csv)")
    birth = read(ctx.source(ctx.hosp, "record.csv"), usecols=["Participant ID", "Sex", "Date of birth"]).set_index(ID)
    cancer = read(ctx.source(ctx.hosp, "cancer.csv"), usecols=["Participant ID", "Sex"]).set_index(ID)
    assert set(p.index) == set(birth.index) == set(cancer.index)
    assert birth.index.is_unique and cancer.index.is_unique
    p["birth"] = dates(birth["Date of birth"])
    p["birth_precision"] = "month"
    p["sex"] = cancer.Sex
    p["sex_source"] = "cancer.csv:Sex(text)"
    ctx.qc["sex_crosswalk"] = pd.crosstab(birth.Sex, p.sex).to_dict()
    assert p.sex.isin(["Female", "Male"]).all()
    assert (p.birth.dt.day == 1).all()

    log("Reading deaths")
    dd = read(ctx.source(ctx.ukb, "UKB_death.csv"))
    dd["date"] = dates(dd["Date of death"])
    invalid = dd["Date of death"].notna() & dd.date.isna()
    ctx.quarantine(dd.loc[invalid, [ID, "Date of death"]], "invalid_death_date", "death")
    p["death"] = dd.groupby(ID).date.min()
    p["death_date_conflict"] = dd.groupby(ID).date.nunique().gt(1).reindex(p.index, fill_value=False)
    p["death_date_unresolved"] = p.index.isin(dd.loc[invalid, ID]) | p.death_date_conflict

    log(f"Assigning fixed patient splits (seed {ctx.config['seed']})")
    p["split"] = patient_split(p.index, ctx.config["seed"])
    train = p.index[p.split.eq("train")].sort_values().to_numpy()
    inner = np.random.default_rng(ctx.config["seed"] + 1).permutation(train)
    nval = int(len(train) * ctx.config["pretrain_internal_validation_fraction"])
    p["pretrain_role"] = "excluded"
    p.loc[train, "pretrain_role"] = "train"
    p.loc[inner[:nval], "pretrain_role"] = "validation"

    log("Attaching cancer-registry coverage")
    p["registry_start"] = pd.NaT
    p["registry_end_exclusive"] = pd.NaT
    coverage = json.loads((ROOT / ctx.config["coverage_manifest"]).read_text(encoding="utf-8"))
    if coverage["status"] == "verified":
        if not all(coverage.get(x) for x in ["source_version", "evidence", "coverage_file"]):
            raise ValueError("Verified coverage requires source_version, evidence and coverage_file")
        path = ROOT / coverage["coverage_file"]
        ctx.used.add(path)
        cov = pd.read_csv(path, dtype=str).set_index(ID)
        assert cov.index.is_unique and set(cov.index).issubset(set(p.index))
        for col in ["registry_start", "registry_end_exclusive"]:
            p[col] = dates(cov[col]).reindex(p.index)
        assert (p.registry_end_exclusive.dropna() > p.registry_start.dropna()).all()
    p["coverage_verified"] = p.registry_start.notna() & p.registry_end_exclusive.notna()
    ctx.save_participants(p)

    split = p.split.value_counts()
    ctx.qc.update({"participants": len(p), "coverage_status": coverage["status"],
                   "split_counts": split.to_dict(), "pretrain_roles": p.pretrain_role.value_counts().to_dict(),
                   "death_date_conflict_people": int(p.death_date_conflict.sum())})
    ctx.done(f"participants: {fmt(len(p))}  (female {fmt(p.sex.eq('Female').sum())}, male {fmt(p.sex.eq('Male').sum())})",
             "split: " + ", ".join(f"{k} {fmt(split.get(k, 0))}" for k in ["train", "validation", "test"]),
             f"deaths: {fmt(p.death.notna().sum())}; lost to follow-up: {fmt(p.lost.notna().sum())}",
             f"registry coverage: {coverage['status']} ({fmt(p.coverage_verified.sum())} people with verified dates)")
