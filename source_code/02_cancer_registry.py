"""Step 02 · Cancer registry: pair date and code per Instance, derive first malignancy.

Writes cancer_events.parquet and adds first_registry_cancer, first_cancer_sites and
registry_unresolved to participants.

Run: python source_code/02_cancer_registry.py
"""
from pathlib import Path

if not __package__:  # also allow `python source_code/<step>.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import pandas as pd

from .common import ID, Context, fmt, log, parser, progress, read, save
from .definitions import SITES, classify, dates

STEP, TITLE = "02", "Cancer registry and first malignancy"
NEEDS = [("hospital_cancer", "cancer.csv")]


def run(args):
    ctx = Context(args, STEP, TITLE, NEEDS)
    p = ctx.participants()
    log("Reading cancer registry (cancer.csv)")
    c = read(ctx.source(ctx.hosp, "cancer.csv")).set_index(ID)
    parts = []
    for col in progress(c.filter(regex=r"^Date of cancer diagnosis \| Instance ").columns, "registry instances", unit="inst"):
        i = int(col.rsplit(" ", 1)[1])
        z = pd.DataFrame({"raw_date": c[col], "icd10": c[f"Type of cancer: ICD10 | Instance {i}"],
                          "icd9": c.get(f"Type of cancer: ICD9 | Instance {i}", pd.Series(index=c.index, dtype=str)),
                          "sex": c.Sex})
        z = z.loc[z[["raw_date", "icd10", "icd9"]].notna().any(axis=1)].copy()
        z["instance"] = i
        parts.append(z.reset_index())
    r = pd.concat(parts, ignore_index=True)
    log("Classifying sites and checking dates")
    r["date"] = dates(r.raw_date)
    r["site"], r["malignant"], r["icd9_fallback"] = classify(r.icd10, r.icd9, r.sex)
    r["qc"] = "valid"
    validcode = r.icd10.str.match(r"^[A-Z]\d{2}", na=False) | (r.icd10.isna() & r.icd9.str.match(r"^\d{3}", na=False))
    r.loc[~validcode, "qc"] = "missing_or_invalid_code"
    r.loc[r.date.isna(), "qc"] = "missing_or_invalid_date"
    r.loc[r.date < r[ID].map(p.birth), "qc"] = "before_birth"
    r.loc[r.date > r[ID].map(p.death), "qc"] = "after_death"
    p["registry_unresolved"] = p.index.isin(r.loc[r.qc.ne("valid"), ID])
    good = r.loc[r.qc.eq("valid") & r.malignant]
    p["first_registry_cancer"] = good.groupby(ID).date.min()
    first = good.loc[good.date.eq(good[ID].map(p.first_registry_cancer))].copy()
    first["endpoint_site"] = first.site.replace("", "other")
    p["first_cancer_sites"] = first.groupby(ID).endpoint_site.agg(lambda s: "|".join(sorted(set(s)))).reindex(p.index).fillna("")
    save(r, ctx.out / "cancer_events.parquet")
    ctx.save_participants(p)

    ctx.qc["registry"] = {"entries": len(r), "qc_counts": r.qc.value_counts().to_dict(),
                          "malignant_people": int(p.first_registry_cancer.notna().sum())}
    top = good.loc[good.site.isin([s[0] for s in SITES])].groupby("site")[ID].nunique().sort_values(ascending=False)
    ctx.done(f"registry entries: {fmt(len(r))}; flagged for QC: {fmt(r.qc.ne('valid').sum())}",
             f"people with a valid non-C44 malignancy: {fmt(p.first_registry_cancer.notna().sum())}",
             "target-site people (any time): " + ", ".join(f"{k} {fmt(v)}" for k, v in top.items()))


if __name__ == "__main__":
    run(parser(__doc__).parse_args())
