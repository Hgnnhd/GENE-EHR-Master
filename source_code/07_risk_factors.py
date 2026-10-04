"""Step 07 · Enhanced inputs: latest pre-landmark risk factors.

For each candidate node, takes the most recent non-missing value assessed before the
landmark, with its assessment Instance, date, age in days and a missing flag
(features_asof.parquet, feature_catalog.csv).

Run: python source_code/07_risk_factors.py
"""
from pathlib import Path

if not __package__:  # also allow `python source_code/<step>.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import re

import pandas as pd

from .common import ID, Context, fmt, log, parser, progress, read_chunks, save
from .definitions import dates

STEP, TITLE = "07", "Pre-landmark risk factors (enhanced inputs)"
SPLIT_DIR = "Base_Information_split"
# Keep blood chemistry only; urine measures/flags are outside this v1 feature set.
SELECTIONS = {
    "01_demographics_socioeconomic.csv": ["Ethnic background", "Townsend deprivation index at recruitment"],
    "02_lifestyle.csv": ["Smoking status", "Alcohol intake frequency."],
    "04_anthropometry_physical_cardiovascular.csv": ["Body mass index (BMI)"],
    "05_blood_urine_biochemistry.csv": None,
    "06_blood_count.csv": None,
}
CODED = ["Ethnic background", "Smoking status", "Alcohol intake frequency."]
NEEDS = [("ukb_fields", "UKB_visit_and_followup_dates.csv")] + [("ukb_fields", f"{SPLIT_DIR}/{f}") for f in SELECTIONS]


def run(args):
    ctx = Context(args, STEP, TITLE, NEEDS)
    ctx.require("landmark_samples.parquet")
    vis = ctx.visits()
    nodes = pd.read_parquet(ctx.out / "landmark_samples.parquet", columns=[ID, "landmark"])
    landmarks = [pd.Timestamp(v) for v in ctx.config["landmarks"]]
    index = {t0: pd.Index(nodes.loc[nodes.landmark.eq(t0), ID], name=ID) for t0 in landmarks}
    # Columns are collected per landmark and joined once (column-by-column inserts fragment the frame).
    columns = {t0: {"landmark": pd.Series(t0, index=idx)} for t0, idx in index.items()}
    visit_dates = {}  # (instance, landmark) -> assessment dates aligned to that landmark's nodes

    def assessed(i, t0):
        if (i, t0) not in visit_dates:
            visit_dates[i, t0] = dates(vis[f"Date of attending assessment centre | Instance {i}"]).reindex(index[t0])
        return visit_dates[i, t0]
    wanted = set(nodes[ID])
    catalog = []
    for fn, bases in SELECTIONS.items():
        path = ctx.source(ctx.ukb / SPLIT_DIR, fn)
        cols = pd.read_csv(path, nrows=0).columns
        if bases is None:
            bases = sorted({c.split(" | ")[0] for c in cols if " | Instance " in c and "urine" not in c.lower()})
        chosen = [c for c in cols if c.split(" | ")[0] in bases]
        log(f"{fn}: {len(bases)} fields")
        chunks = [c.loc[c["Participant ID"].isin(wanted)]
                  for c in read_chunks(path, len(vis), fn, usecols=["Participant ID"] + chosen)]
        raw = pd.concat(chunks).rename(columns={"Participant ID": ID}).set_index(ID)
        for base in progress(bases, f"{fn} fields", unit="field", leave=False):
            key = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
            for t0, idx in index.items():
                val = pd.Series(pd.NA, index=idx, dtype="string")
                when = pd.Series(pd.NaT, index=idx, dtype="datetime64[ns]")
                inst = pd.Series(pd.NA, index=idx, dtype="Int8")
                for col in [c for c in chosen if c.split(" | ")[0] == base]:
                    match = re.search(r"Instance (\d+)$", col)
                    i = int(match.group(1)) if match else 0  # recruitment-only Townsend
                    d = assessed(i, t0)
                    v = raw[col].reindex(idx).astype("string")
                    missing = v.isna() | v.str.lower().isin(["do not know", "prefer not to answer", "not known"])
                    if base in CODED:
                        missing |= v.isin(["-1", "-3"])
                    use = d.lt(t0) & ~missing & (when.isna() | d.gt(when))
                    val.loc[use], when.loc[use], inst.loc[use] = v.loc[use], d.loc[use], i
                columns[t0].update({key: val, key + "__measured_at": when, key + "__instance": inst,
                                    key + "__age_days": (t0 - when).dt.days.astype("Int32"),
                                    key + "__missing": val.isna()})
            catalog.append({"feature": key, "source_file": fn, "source_field": base,
                            "time_basis": "assessment Instance date; assay availability date not exported"})
    features = pd.concat([pd.DataFrame(cols).reset_index() for cols in columns.values()], ignore_index=True)
    save(features, ctx.out / "features_asof.parquet")
    catalog = pd.DataFrame(catalog)
    catalog.to_csv(ctx.report / "feature_catalog.csv", index=False)

    miss = features[[c for c in features.columns if c.endswith("__missing")]].mean().sort_values()
    ctx.qc["features"] = {"rows": len(features), "fields": len(catalog), "time_basis": "assessment-date retrospective measurements"}
    ctx.done(f"{fmt(len(features))} candidate nodes x {len(catalog)} fields",
             "lowest missingness: " + ", ".join(f"{k.removesuffix('__missing')} {v:.1%}" for k, v in miss.head(3).items()),
             "highest missingness: " + ", ".join(f"{k.removesuffix('__missing')} {v:.1%}" for k, v in miss.tail(3).items()))


if __name__ == "__main__":
    run(parser(__doc__).parse_args())
