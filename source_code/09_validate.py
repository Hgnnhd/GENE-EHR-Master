"""Step 09 · Validate full-data invariants: chronology, risk sets, labels and patient leakage.

Reads the outputs of steps 01-08 and fails on the first violated check.

Run: python source_code/09_validate.py
"""
from pathlib import Path

if not __package__:  # also allow `python source_code/<step>.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import argparse
from collections import Counter
import json
import re

import polars as pl

from .common import ROOT, log, say
from .definitions import SITES


def validate(root, config_path=ROOT / "configs/cohort.json"):
    if not (root / "BUILD_COMPLETE.json").exists():
        raise RuntimeError("Build did not complete: run steps 01-08 (python source_code/run_all.py)")
    checks = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        checks.append(name)

    p = pl.read_parquet(root / "participants.parquet")
    check("unique_participants", p["participant_id"].n_unique() == p.height)
    frac = dict(p["split"].value_counts(normalize=True).iter_rows())
    check("split_fractions_70_15_15", all(abs(frac.get(k, 0) - v) < 0.01 for k, v in {"train": .70, "validation": .15, "test": .15}.items()))
    check("pretrain_roles_only_train", p.filter((pl.col("pretrain_role") != "excluded") != (pl.col("split") == "train")).is_empty())
    n = pl.read_parquet(root / "landmark_status.parquet")
    s = pl.read_parquet(root / "landmark_samples.parquet")
    check("unique_patient_landmark", n.unique(["participant_id", "landmark"]).height == n.height)
    check("candidate_table_exact", s.height == n.filter(pl.col("clinical_candidate")).height)
    check("one_split_per_person_across_landmarks", n.group_by("participant_id").agg(pl.col("split").n_unique()).filter(pl.col("split") != 1).is_empty())
    check("candidate_recruited_before_landmark", s.filter(pl.col("recruited") >= pl.col("landmark")).is_empty())
    for col in ["death", "lost"]:
        check("candidate_no_prior_" + col, s.filter(pl.col(col) <= pl.col("landmark")).is_empty())
    for col in ["first_registry_cancer", "first_hospital_cancer"]:
        check("candidate_no_prior_" + col, s.filter(pl.col(col) < pl.col("landmark")).is_empty())
    for site, _, _, _, sex in SITES:
        lab = f"label_{site}_5y"
        check("unverified_labels_null_" + site, n.filter(~pl.col("coverage_verified") & pl.col(lab).is_not_null()).is_empty())
        check("observed_window_" + site, n.filter(pl.col(f"observed_{site}_5y") & ((pl.col("first_registry_cancer") < pl.col("landmark")) | (pl.col("first_registry_cancer") >= pl.col("horizon_end")) | ~pl.col("clinical_candidate"))).is_empty())
        ev_col = f"event_{site}"
        check("event_code_values_" + site, n.filter(pl.col(ev_col).is_not_null() & ~pl.col(ev_col).is_in([0, 1, 2])).is_empty())
        check("event_matches_label_" + site, n.filter(pl.col(lab).is_not_null() & (pl.col(ev_col).is_null() | ((pl.col(ev_col) == 1) != (pl.col(lab) == 1)))).is_empty())
        check("event_requires_eligible_" + site, n.filter(pl.col(ev_col).is_not_null() & ~pl.col("eligible")).is_empty())
        if sex:
            check("sex_applicability_" + site, n.filter((pl.col("sex") != sex) & (pl.col(lab).is_not_null() | pl.col(f"observed_{site}_5y"))).is_empty())
    ev = pl.read_parquet(root / "events.parquet")
    check("unique_patient_code", ev.unique(["participant_id", "code"]).height == ev.height)
    check("earliest_source_date", ev.filter(pl.col("date") != pl.min_horizontal("hospital_date", "first_occurrences_date")).is_empty())
    e = ev.join(p.select("participant_id", "birth", "death"), on="participant_id", how="left")
    check("event_not_before_birth", e.filter(pl.col("date") < pl.col("birth")).is_empty())
    check("event_not_after_death", e.filter(pl.col("date") > pl.col("death")).is_empty())
    for t0 in s["landmark"].unique().to_list():
        h = ev.filter(pl.col("date") < t0).group_by("participant_id").len()
        a = s.filter(pl.col("landmark") == t0).join(h, on="participant_id", how="left")
        check("history_count_" + str(t0.date()), a.filter(pl.col("history_codes") != pl.col("len").fill_null(0)).is_empty())
    li = pl.read_parquet(root / "landmark_inputs.parquet")
    check("inputs_unique_patient_landmark", li.unique(["participant_id", "landmark"]).height == li.height == s.height)
    a = li.join(s.select("participant_id", pl.col("landmark").cast(li.schema["landmark"]), "history_codes"), on=["participant_id", "landmark"], how="left")
    check("inputs_same_cohort_and_length", a.filter(pl.col("history_codes").is_null() | (pl.col("n_codes") != pl.col("history_codes"))).is_empty())
    check("inputs_strictly_before_landmark", li.filter((pl.col("dates").list.max() >= pl.col("landmark")) | (pl.col("days_before").list.min() <= 0)).is_empty())
    pe = pl.read_parquet(root / "pretrain_events.parquet")
    pp = pl.read_parquet(root / "pretrain_participants.parquet")
    check("pretrain_only_train_people", pe.join(p.select("participant_id", "split"), on="participant_id", suffix="_master").filter(pl.col("split_master") != "train").is_empty())
    check("pretrain_inner_roles_disjoint", pp["participant_id"].n_unique() == pp.height)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    from datetime import datetime
    cutoff = datetime.fromisoformat(config["pretrain_before"])
    check("pretrain_time_cutoff", pe.filter(pl.col("date") >= cutoff).is_empty())
    f = pl.read_parquet(root / "features_asof.parquet")
    check("features_unique_patient_landmark", f.unique(["participant_id", "landmark"]).height == f.height)
    check("features_same_cohort", f.select("participant_id", "landmark").join(s.select("participant_id", "landmark"), on=["participant_id", "landmark"], how="anti").is_empty() and f.height == s.height)
    for col in f.columns:
        if col.endswith("__measured_at"):
            key = col.removesuffix("__measured_at")
            check("feature_before_landmark_" + key, f.filter(pl.col(col) >= pl.col("landmark")).is_empty())
            check("feature_missing_consistency_" + key, f.filter(pl.col(key).is_null() != pl.col(key + "__missing")).is_empty())
            check("feature_date_present_" + key, f.filter(pl.col(key).is_not_null() & pl.col(col).is_null()).is_empty())
    return {"passed": len(checks), "checks": checks, "participants": p.height, "candidate_nodes": s.height, "events": ev.height}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/cohort.json")
    ap.add_argument("--report", type=Path, default=ROOT / "ccfa-workfiles/checks/cancer-cohort/validation.json")
    run(ap.parse_args())


def run(args):
    print(f"\n{'=' * 64}\n  Step 09  Validate full-data invariants\n{'=' * 64}", flush=True)
    result = validate(Path(args.data), Path(args.config))
    # Group per-site / per-landmark / per-feature checks under one line each.
    sites = "|".join(s[0] for s in SITES)
    groups = Counter()
    for name in result["checks"]:
        base = re.sub(rf"_({sites}|\d{{4}}-\d{{2}}-\d{{2}})$", "", name)
        base = re.sub(r"^(feature_(before_landmark|missing_consistency|date_present))_.*", r"\1", base)
        groups[base] += 1
    for base, n in groups.items():
        say(f"  ok  {base}" + (f"  (x{n})" if n > 1 else ""))
    Path(args.report).write_text(json.dumps(result, indent=2), encoding="utf-8")
    log(f"Passed {result['passed']} full-data checks; {result['candidate_nodes']:,} candidate nodes")
    return result


if __name__ == "__main__":
    main()
