"""Step 05 · Landmark cohorts: eligibility, follow-up, outcomes and labels.

For each calendar landmark (configs/cohort.json) applies the same exclusion rules in
order, resolves follow-up within [t0, t0 + horizon) and derives per-site observed
cases, binary labels and competing-risk event codes. Writes landmark_status (everyone),
landmark_samples (clinical candidates), cohort_flow.csv, landmark_counts.csv and
split_summary.csv.

Run: python source_code/05_landmark_cohort.py
"""
from pathlib import Path

if not __package__:  # also allow `python source_code/<step>.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import pandas as pd
import polars as pl

from .common import ID, Context, fmt, log, parser, progress, save
from .definitions import SITES, resolve_followup_frame, site_event_code

STEP, TITLE = "05", "Landmark cohorts, follow-up and labels"


def landmark(p, sr, ev, t0, horizon, flows, counts):
    end = t0 + pd.DateOffset(years=horizon)
    z = p.copy()
    z["landmark"], z["horizon_end"] = t0, end
    # History eligibility only uses self-reports already collected by t0.
    known_sr = sr.loc[sr.assessment_date.lt(t0)]
    z["prior_self_report_cancer"] = z.index.isin(known_sr.loc[known_sr.classification.eq("malignant_history"), ID])
    z["ambiguous_self_report"] = z.index.isin(known_sr.loc[known_sr.classification.eq("ambiguous"), ID])
    unknown_sr = sr.loc[sr.assessment_date.isna() & sr.classification.ne("excluded_c44_or_precancer"), ID]
    z["undated_self_report"] = z.index.isin(unknown_sr)
    # Sensitivity flag only: reports collected at/after t0 that date a malignancy before t0.
    later = sr.loc[sr.assessment_date.ge(t0) & sr.classification.eq("malignant_history")]
    year = pd.to_numeric(later.reported_year, errors="coerce")
    before = year.gt(0) & year.lt(t0.year + (t0.dayofyear - 1) / 365.25)
    z["later_self_report_prior_cancer"] = z.index.isin(later.loc[before, ID])
    rules = [
        ("not_recruited", z.recruited.isna() | z.recruited.ge(t0)),
        ("dead_at_landmark", z.death.le(t0)),
        ("lost_at_landmark", z.lost.le(t0)),
        ("prior_registry_cancer", z.first_registry_cancer.lt(t0)),
        ("prior_hospital_cancer", z.first_hospital_cancer.lt(t0)),
        ("prior_self_report_cancer", z.prior_self_report_cancer),
        ("unresolved_history_or_dates", z.registry_unresolved | z.hospital_malignancy_unresolved | z.ambiguous_self_report | z.undated_self_report | z.death_date_unresolved),
    ]
    z["eligibility_reason"] = "candidate_pending_coverage"
    remaining = pd.Series(True, index=z.index)
    value = str(t0.date())
    for name, mask in rules:
        removed = remaining & mask
        remaining &= ~mask
        z.loc[removed, "eligibility_reason"] = name
        flows.append({"landmark": value, "step": name, "removed": int(removed.sum()), "remaining": int(remaining.sum())})
    z["clinical_candidate"] = remaining
    z["eligible"] = remaining & z.coverage_verified & z.registry_start.le(t0) & z.registry_end_exclusive.gt(t0)
    z.loc[z.eligible, "eligibility_reason"] = "eligible"
    z.loc[remaining & z.coverage_verified & ~z.eligible, "eligibility_reason"] = "not_observable_at_landmark"
    hist = ev.filter(pl.col("date") < t0).group_by(ID).agg(pl.len().alias("history_codes"), pl.col("date").min().alias("history_start"))
    hist = hist.to_pandas().set_index(ID)
    z["history_codes"] = hist.history_codes.reindex(z.index, fill_value=0)
    z["history_start"] = hist.history_start
    z["empty_history"] = z.history_codes.eq(0)
    z["age_years_approx"] = (t0 - z.birth).dt.days / 365.25
    status, stop = resolve_followup_frame(t0, end, z.first_registry_cancer, z.death, z.lost,
                                          z.registry_start, z.registry_end_exclusive)
    z["followup_status"] = status.where(remaining, "ineligible")
    z["observation_end"] = stop.where(remaining)
    z["followup_days"] = (z.observation_end - t0).dt.days.astype("Int64")
    # Descriptive known diagnoses, not training labels while coverage is unresolved.
    observed = remaining & z.first_registry_cancer.ge(t0) & z.first_registry_cancer.lt(end)
    observed &= z.death.isna() | z.first_registry_cancer.le(z.death)
    observed &= z.lost.isna() | z.first_registry_cancer.lt(z.lost)
    z["registry_observed_first_cancer_5y"] = observed
    z["registry_death_same_day"] = z.first_registry_cancer.eq(z.death) & observed
    site_sets = z.first_cancer_sites.str.split("|")
    for site, name, _, _, sex in progress(SITES, f"{value} sites", unit="site", leave=False):
        applicable = pd.Series(True, index=z.index) if sex is None else z.sex.eq(sex)
        member = site_sets.apply(lambda xs: site in xs)
        z[f"observed_{site}_5y"] = observed & member & applicable
        label = pd.Series(pd.NA, index=z.index, dtype="Int8")
        evaluable = z.eligible & z.followup_status.isin(["cancer", "death", "event_free_5y"]) & applicable
        label.loc[evaluable] = 0
        label.loc[evaluable & z.followup_status.eq("cancer") & member] = 1
        z[f"label_{site}_5y"] = label
        # Survival/competing-risk target: pair with followup_days.
        z[f"event_{site}"] = site_event_code(z.followup_status.where(z.eligible, "ineligible"), member, applicable)
        for split in ["all", "train", "validation", "test"]:
            sm = pd.Series(True, index=z.index) if split == "all" else z.split.eq(split)
            counts.append({"landmark": value, "site": site, "name": name, "split": split,
                           "clinical_candidates": int((remaining & applicable & sm).sum()),
                           "registry_observed_cases": int((z[f"observed_{site}_5y"] & sm).sum()),
                           "evaluable_binary_labels": int(label.loc[sm].notna().sum())})
    return z.reset_index()


def split_summary(p, nodes):
    """Aggregate split balance (no participant IDs)."""
    age = (p.recruited - p.birth).dt.days / 365.25

    def describe(g, ages):
        return {"people": len(g), "female_share": g.sex.eq("Female").mean(),
                "age_q1": ages.quantile(.25), "age_median": ages.median(), "age_q3": ages.quantile(.75)}

    rows = []
    for split, g in p.groupby("split"):
        rows.append({"landmark": "all_participants", "split": split, **describe(g, age.loc[g.index]),
                     "share": len(g) / len(p), "died": g.death.notna().mean(), "lost": g.lost.notna().mean(),
                     "registry_malignancy_ever": g.first_registry_cancer.notna().mean()})
    cand = nodes.loc[nodes.clinical_candidate]
    target = cand[[f"observed_{s[0]}_5y" for s in SITES]].any(axis=1)
    for (t0, split), g in cand.groupby(["landmark", "split"]):
        rows.append({"landmark": str(t0.date()), "split": split, **describe(g, g.age_years_approx),
                     "share": len(g) / int(cand.landmark.eq(t0).sum()),
                     "empty_history": g.empty_history.mean(), "history_codes_median": g.history_codes.median(),
                     "observed_top10_share": target.loc[g.index].mean()})
    return pd.DataFrame(rows).round(4)


def run(args):
    ctx = Context(args, STEP, TITLE)
    ctx.require("participants.parquet", "events.parquet", "self_report_cancer.parquet")
    p = ctx.participants()
    sr = pd.read_parquet(ctx.out / "self_report_cancer.parquet")
    ev = pl.read_parquet(ctx.out / "events.parquet")
    rows, flows, counts = [], [], []
    for value in progress(ctx.config["landmarks"], "landmarks", unit="landmark"):
        log(f"Landmark {value}: eligibility rules, follow-up and labels")
        rows.append(landmark(p, sr, ev, pd.Timestamp(value), ctx.config["horizon_years"], flows, counts))
    nodes = pd.concat(rows, ignore_index=True)
    log("Writing landmark tables and aggregate reports")
    save(nodes, ctx.out / "landmark_status.parquet")
    save(nodes.loc[nodes.clinical_candidate], ctx.out / "landmark_samples.parquet")
    flow = pd.DataFrame(flows)
    flow.to_csv(ctx.report / "cohort_flow.csv", index=False)
    pd.DataFrame(counts).to_csv(ctx.report / "landmark_counts.csv", index=False)
    split_summary(p, nodes).to_csv(ctx.report / "split_summary.csv", index=False)

    ctx.qc["landmarks"], lines = {}, []
    for t0, g in nodes.groupby("landmark"):
        anytarget = g[[f"observed_{s[0]}_5y" for s in SITES]].any(axis=1)
        key = str(t0.date())
        ctx.qc["landmarks"][key] = {
            "clinical_candidates": int(g.clinical_candidate.sum()), "eligible_with_verified_coverage": int(g.eligible.sum()),
            "observed_top10_people": int(anytarget.sum()),
            "empty_history_candidates": int((g.clinical_candidate & g.empty_history).sum()),
            "followup_status": g.followup_status.value_counts().to_dict()}
        table = "\n".join(f"      - {r.step:<30}{fmt(r.removed):>10}  -> {fmt(r.remaining):>10}"
                          for r in flow.loc[flow.landmark.eq(key)].itertuples())
        lines.append(f"landmark {key}: {fmt(len(g))} participants\n{table}\n"
                     f"      = clinical candidates {fmt(g.clinical_candidate.sum())} | eligible (verified coverage) "
                     f"{fmt(g.eligible.sum())} | observed top-10 cases {fmt(anytarget.sum())}")
    anytarget = nodes[[f"observed_{s[0]}_5y" for s in SITES]].any(axis=1)
    ctx.qc["unique_observed_top10_people"] = int(nodes.loc[anytarget, ID].nunique())
    ctx.qc["split_fractions"] = p.split.value_counts(normalize=True).round(4).to_dict()
    ctx.done(*lines, f"unique observed top-10 people across landmarks: {fmt(ctx.qc['unique_observed_top10_people'])}")


if __name__ == "__main__":
    run(parser(__doc__).parse_args())
