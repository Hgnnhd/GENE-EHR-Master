"""Data stage · Self-reported cancer history, classified with the local UKB coding dictionary.

Writes self_report_cancer.parquet (one row per Instance x Array report with its
assessment date). How reports affect each landmark is decided in the cohort stage.
"""
import re

import pandas as pd

from .common import ID, Context, fmt, log, progress, read, save
from .definitions import dates

STAGE, TITLE = "self_report", "Self-reported cancer history"
CODINGS = "app176660_20240512000635.dataset.codings.csv"
NEEDS = [("ukb_fields", "UKB_self_reported_conditions.csv"), ("ukb_fields", "UKB_visit_and_followup_dates.csv"),
         ("hospital_cancer", CODINGS)]
EXCLUDED = {"1060", "1061", "1062", "1073", "1072"}  # non-melanoma skin cancer branch, cervical precancer
AMBIGUOUS = {"1003", "1051", "99999"}


def run(args):
    ctx = Context(args, STAGE, TITLE, NEEDS)
    vis = ctx.visits()
    log("Loading data_coding_3 dictionary")
    dictionary = read(ctx.source(ctx.hosp, CODINGS))
    coding = dictionary.loc[dictionary.coding_name.eq("data_coding_3")].set_index("code").meaning.to_dict()
    known = set(coding.values())
    excluded_text = {coding[k] for k in EXCLUDED}
    ambiguous_text = {coding[k] for k in AMBIGUOUS}
    log("Reading self-reported cancer fields")
    path = ctx.source(ctx.ukb, "UKB_self_reported_conditions.csv")
    sr = read(path, usecols=lambda c: c == "Participant ID" or "cancer first diagnosed" in c and "non-cancer" not in c or c.startswith("Cancer code,")).set_index(ID)
    parts = []
    for col in progress(sr.filter(regex="^Cancer code,").columns, "self-report slots", unit="slot"):
        inst, slot = map(int, re.findall(r"(?:Instance|Array) (\d+)", col))
        suffix = f" | Instance {inst} | Array {slot}"
        z = pd.DataFrame({"raw_code": sr[col], "reported_year": sr.get("Interpolated Year when cancer first diagnosed" + suffix),
                          "reported_age": sr.get("Interpolated Age of participant when cancer first diagnosed" + suffix)})
        z = z.loc[z.raw_code.notna()].copy()
        z["meaning"] = z.raw_code.map(coding).fillna(z.raw_code)
        z["classification"] = "malignant_history"
        z.loc[z.meaning.isin(excluded_text), "classification"] = "excluded_c44_or_precancer"
        z.loc[z.meaning.isin(ambiguous_text) | ~z.meaning.isin(known), "classification"] = "ambiguous"
        z["assessment_date"] = dates(vis[f"Date of attending assessment centre | Instance {inst}"]).reindex(z.index)
        z["instance"], z["slot"] = inst, slot
        parts.append(z.reset_index())
    cols = [ID, "raw_code", "reported_year", "reported_age", "meaning", "classification", "assessment_date", "instance", "slot"]
    sr = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    save(sr, ctx.out / "self_report_cancer.parquet")

    counts = sr.classification.value_counts()
    ctx.qc["self_report"] = counts.to_dict()
    ctx.done(f"reports: {fmt(len(sr))} from {fmt(sr[ID].nunique())} people",
             ", ".join(f"{k} {fmt(v)}" for k, v in counts.items()))
