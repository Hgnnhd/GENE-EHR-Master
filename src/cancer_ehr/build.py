"""Build local research tables without changing source files.

Run: PYTHONPATH=src python -m cancer_ehr.build --ehr ... --legacy ...
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import polars as pl

from .definitions import SITES, classify, dates, patient_split, resolve_followup_frame, site_event_code

ID = "participant_id"
START = time.time()


def log(message):
    print(f"[{time.time()-START:.1f}s] {message}", flush=True)


def read(path, **kwargs):
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", **kwargs).rename(columns={"Participant ID": ID})


def save(frame, path):
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def dump(obj, path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


class Builder:
    def __init__(self, args):
        self.ehr, self.old, self.out = args.ehr, args.legacy, args.output
        self.report = args.report
        self.config = json.loads(args.config.read_text(encoding="utf-8"))
        self.out.mkdir(parents=True, exist_ok=True)
        self.report.mkdir(parents=True, exist_ok=True)
        self.qc = {}
        self.quarantines = []
        self.used = set()

    def source(self, root, name):
        path = root / name
        self.used.add(path)
        return path

    def quarantine(self, frame, reason, source):
        if len(frame):
            frame = frame.copy()
            frame["reason"], frame["source"] = reason, source
            self.quarantines.append(frame)

    def participants(self):
        log("Building participants and fixed patient splits")
        self.vis = read(self.source(self.ehr, "UKB_visit_and_followup_dates.csv")).set_index(ID)
        assert self.vis.index.is_unique and self.vis.index.notna().all()
        p = pd.DataFrame(index=self.vis.index)
        p["recruited"] = dates(self.vis["Date of attending assessment centre | Instance 0"])
        p["lost"] = dates(self.vis["Date lost to follow-up"])
        birth = read(self.source(self.old, "record.csv"), usecols=["Participant ID", "Sex", "Date of birth"]).set_index(ID)
        cancer = read(self.source(self.old, "cancer.csv"), usecols=["Participant ID", "Sex"]).set_index(ID)
        assert set(p.index) == set(birth.index) == set(cancer.index)
        assert birth.index.is_unique and cancer.index.is_unique
        p["birth"] = dates(birth["Date of birth"])
        p["birth_precision"] = "month"
        p["sex"] = cancer.Sex
        p["sex_source"] = "cancer.csv:Sex(text)"
        self.qc["sex_crosswalk"] = pd.crosstab(birth.Sex, p.sex).to_dict()
        assert p.sex.isin(["Female", "Male"]).all()
        assert (p.birth.dt.day == 1).all()
        dd = read(self.source(self.ehr, "UKB_death.csv"))
        dd["date"] = dates(dd["Date of death"])
        invalid = dd["Date of death"].notna() & dd.date.isna()
        self.quarantine(dd.loc[invalid, [ID, "Date of death"]], "invalid_death_date", "death")
        p["death"] = dd.groupby(ID).date.min()
        p["death_date_conflict"] = dd.groupby(ID).date.nunique().gt(1).reindex(p.index, fill_value=False)
        p["death_date_unresolved"] = p.index.isin(dd.loc[invalid, ID]) | p.death_date_conflict
        p["split"] = patient_split(p.index, self.config["seed"])
        train = p.index[p.split.eq("train")].sort_values().to_numpy()
        inner = np.random.default_rng(self.config["seed"] + 1).permutation(train)
        nval = int(len(train) * self.config["pretrain_internal_validation_fraction"])
        p["pretrain_role"] = "excluded"
        p.loc[train, "pretrain_role"] = "train"
        p.loc[inner[:nval], "pretrain_role"] = "validation"
        p["registry_start"] = pd.NaT
        p["registry_end_exclusive"] = pd.NaT
        coverage = json.loads(Path(self.config["coverage_manifest"]).read_text(encoding="utf-8"))
        self.coverage = coverage
        if coverage["status"] == "verified":
            if not all(coverage.get(x) for x in ["source_version", "evidence", "coverage_file"]):
                raise ValueError("Verified coverage requires source_version, evidence and coverage_file")
            path = Path(coverage["coverage_file"])
            self.used.add(path)
            cov = pd.read_csv(path, dtype=str).set_index(ID)
            assert cov.index.is_unique and set(cov.index).issubset(set(p.index))
            for col in ["registry_start", "registry_end_exclusive"]:
                p[col] = dates(cov[col]).reindex(p.index)
            assert (p.registry_end_exclusive.dropna() > p.registry_start.dropna()).all()
        p["coverage_verified"] = p.registry_start.notna() & p.registry_end_exclusive.notna()
        self.p = p
        self.qc["participants"] = len(p)
        self.qc["death_date_conflict_people"] = int(p.death_date_conflict.sum())

    def registry(self):
        log("Pairing cancer registry by Instance")
        c = read(self.source(self.old, "cancer.csv")).set_index(ID)
        parts = []
        for col in c.filter(regex=r"^Date of cancer diagnosis \| Instance ").columns:
            i = int(col.rsplit(" ", 1)[1])
            z = pd.DataFrame({"raw_date": c[col], "icd10": c[f"Type of cancer: ICD10 | Instance {i}"],
                              "icd9": c.get(f"Type of cancer: ICD9 | Instance {i}", pd.Series(index=c.index, dtype=str)),
                              "sex": c.Sex})
            z = z.loc[z[["raw_date", "icd10", "icd9"]].notna().any(axis=1)].copy()
            z["instance"] = i
            parts.append(z.reset_index())
        r = pd.concat(parts, ignore_index=True)
        r["date"] = dates(r.raw_date)
        r["site"], r["malignant"], r["icd9_fallback"] = classify(r.icd10, r.icd9, r.sex)
        r["qc"] = "valid"
        validcode = r.icd10.str.match(r"^[A-Z]\d{2}", na=False) | (r.icd10.isna() & r.icd9.str.match(r"^\d{3}", na=False))
        r.loc[~validcode, "qc"] = "missing_or_invalid_code"
        r.loc[r.date.isna(), "qc"] = "missing_or_invalid_date"
        r.loc[r.date < r[ID].map(self.p.birth), "qc"] = "before_birth"
        r.loc[r.date > r[ID].map(self.p.death), "qc"] = "after_death"
        self.qc["registry"] = {"entries": len(r), "qc_counts": r.qc.value_counts().to_dict()}
        self.p["registry_unresolved"] = self.p.index.isin(r.loc[r.qc.ne("valid"), ID])
        good = r.loc[r.qc.eq("valid") & r.malignant]
        self.p["first_registry_cancer"] = good.groupby(ID).date.min()
        first = good.loc[good.date.eq(good[ID].map(self.p.first_registry_cancer))].copy()
        first["endpoint_site"] = first.site.replace("", "other")
        self.p["first_cancer_sites"] = first.groupby(ID).endpoint_site.agg(lambda s: "|".join(sorted(set(s)))).reindex(self.p.index).fillna("")
        save(r, self.out / "cancer_events.parquet")

    def histories(self):
        log("Extracting first-occurrence histories and quarantining invalid dates")
        parts, bad_values = [], Counter()
        fpath = self.source(self.ehr, "UKB_First_occurrences.csv")
        for chunk in pd.read_csv(fpath, dtype=str, chunksize=10000):
            z = chunk.set_index("Participant ID").stack().rename("raw_date").reset_index()
            z.columns = [ID, "field", "raw_date"]
            z["date"] = dates(z.raw_date)
            z["code"] = z.field.str.extract(r"^Date ([A-Z][0-9]{2}) ", expand=False)
            bad = z.date.isna() | z.code.isna()
            bad_values.update(z.loc[bad, "raw_date"])
            self.quarantine(z.loc[bad], "invalid_date_or_code", "first_occurrences")
            parts.append(pl.from_pandas(z.loc[~bad, [ID, "code", "date"]]))
        self.qc["first_occurrence_invalid_values"] = dict(bad_values)
        fo = pl.concat(parts)
        del parts
        log("Pairing hospital arrays; quarantining entire mismatched rows")
        parts, mismatches, undated_malignant = [], set(), set()
        for chunk in pd.read_csv(self.source(self.old, "record.csv"), dtype=str, chunksize=10000):
            chunk = chunk.set_index("Participant ID")
            cs = chunk["Diagnoses - ICD10"].str.split("|", regex=False)
            ds = chunk.filter(regex=r"^Date of first in-patient diagnosis - ICD10 \| Array ")
            ds = ds[sorted(ds.columns, key=lambda x: int(x.rsplit(" ", 1)[1]))]
            ds.columns = range(len(ds.columns))
            mismatch = cs.str.len().fillna(0).ne(ds.notna().sum(axis=1))
            mismatches.update(chunk.index[mismatch])
            z = cs.explode().dropna().rename("raw_code").reset_index().rename(columns={"Participant ID": ID})
            z["slot"] = z.groupby(ID).cumcount()
            z["code"] = z.raw_code.str.extract(r"^([A-Z][0-9]{2})", expand=False)
            dl = ds.stack().rename("raw_date").reset_index()
            dl.columns = [ID, "slot", "raw_date"]
            z = z.merge(dl, on=[ID, "slot"], how="left", validate="one_to_one")
            z["date"] = dates(z.raw_date)
            z["malignant"] = z.code.str.match(r"^C\d{2}$", na=False) & z.code.ne("C44") & z.code.le("C97")
            bad = z[ID].isin(mismatches) | z.date.isna() | z.code.isna()
            self.quarantine(z.loc[bad], "array_mismatch_or_invalid_pair", "hospital")
            undated_malignant.update(z.loc[bad & z.malignant, ID])
            parts.append(pl.from_pandas(z.loc[~bad, [ID, "code", "date"]]))
        hr = pl.concat(parts)
        del parts
        self.p["hospital_array_mismatch"] = self.p.index.isin(mismatches)
        self.p["hospital_malignancy_unresolved"] = self.p.index.isin(undated_malignant)
        self.qc["hospital_array_mismatch_people"] = len(mismatches)
        self.qc["hospital_unresolved_malignancy_people"] = len(undated_malignant)
        p = pl.from_pandas(self.p[["birth", "death"]].reset_index())
        cleaned = []
        for source, frame in [("first_occurrences", fo), ("hospital", hr)]:
            frame = frame.join(p, on=ID, how="left")
            # Null-safe separate comparisons: an absent death must not mask pre-birth dates.
            invalid = (pl.col("date") < pl.col("birth")).fill_null(False) | (pl.col("date") > pl.col("death")).fill_null(False)
            bad = frame.filter(invalid)
            self.quarantine(bad.to_pandas(), "before_birth_or_after_death", source)
            if source == "hospital":
                badc = bad.filter(pl.col("code").str.contains(r"^C\d{2}$") & (pl.col("code") != "C44") & (pl.col("code") <= "C97"))
                self.p.loc[self.p.index.isin(badc[ID].to_list()), "hospital_malignancy_unresolved"] = True
            valid = frame.filter(~invalid).select(ID, "code", "date").group_by(ID, "code").agg(pl.col("date").min())
            cleaned.append(valid.rename({"date": source + "_date"}))
        ev = cleaned[0].join(cleaned[1], on=[ID, "code"], how="full", coalesce=True)
        ev = ev.with_columns(pl.min_horizontal("first_occurrences_date", "hospital_date").alias("date"))
        ev = ev.with_columns(
            pl.when(pl.col("first_occurrences_date").is_not_null() & pl.col("hospital_date").is_not_null()).then(pl.lit("both"))
              .when(pl.col("hospital_date").is_not_null()).then(pl.lit("hospital")).otherwise(pl.lit("first_occurrences")).alias("source"),
            pl.lit("day_as_recorded").alias("date_precision"))
        ev = ev.sort([ID, "date", "code"])
        ev.write_parquet(self.out / "events.parquet", compression="zstd")
        hosp = cleaned[1].filter(pl.col("code").str.contains(r"^C\d{2}$") & (pl.col("code") != "C44") & (pl.col("code") <= "C97"))
        hd = hosp.group_by(ID).agg(pl.col("hospital_date").min()).to_pandas().set_index(ID)
        self.p["first_hospital_cancer"] = hd.hospital_date
        self.ev = ev
        self.qc["events"] = {"rows": ev.height, "people": ev[ID].n_unique(), "codes": ev["code"].n_unique()}

    def self_reports(self):
        log("Classifying self-reported cancer history using local UKB coding dictionary")
        dictionary = read(self.source(self.old, "app176660_20240512000635.dataset.codings.csv"))
        coding = dictionary.loc[dictionary.coding_name.eq("data_coding_3")].set_index("code").meaning.to_dict()
        excluded = {"1060", "1061", "1062", "1073", "1072"}
        ambiguous = {"1003", "1051", "99999"}
        known = set(coding.values())
        excluded_text = {coding[k] for k in excluded}
        ambiguous_text = {coding[k] for k in ambiguous}
        path = self.source(self.ehr, "UKB_self_reported_conditions.csv")
        sr = read(path, usecols=lambda c: c == "Participant ID" or "cancer first diagnosed" in c and "non-cancer" not in c or c.startswith("Cancer code," )).set_index(ID)
        parts = []
        for col in sr.filter(regex="^Cancer code,").columns:
            inst, slot = map(int, re.findall(r"(?:Instance|Array) (\d+)", col))
            suffix = f" | Instance {inst} | Array {slot}"
            z = pd.DataFrame({"raw_code": sr[col], "reported_year": sr.get("Interpolated Year when cancer first diagnosed" + suffix),
                              "reported_age": sr.get("Interpolated Age of participant when cancer first diagnosed" + suffix)})
            z = z.loc[z.raw_code.notna()].copy()
            z["meaning"] = z.raw_code.map(coding).fillna(z.raw_code)
            z["classification"] = "malignant_history"
            z.loc[z.meaning.isin(excluded_text), "classification"] = "excluded_c44_or_precancer"
            z.loc[z.meaning.isin(ambiguous_text) | ~z.meaning.isin(known), "classification"] = "ambiguous"
            z["assessment_date"] = dates(self.vis[f"Date of attending assessment centre | Instance {inst}"]).reindex(z.index)
            z["instance"], z["slot"] = inst, slot
            parts.append(z.reset_index())
        sr = pd.concat(parts, ignore_index=True)
        save(sr, self.out / "self_report_cancer.parquet")
        self.sr = sr
        self.qc["self_report"] = sr.classification.value_counts().to_dict()

    def landmarks(self):
        log("Applying landmark eligibility and competing-event definitions")
        p = self.p
        rows, flows, counts = [], [], []
        for value in self.config["landmarks"]:
            t0 = pd.Timestamp(value)
            end = t0 + pd.DateOffset(years=self.config["horizon_years"])
            z = p.copy()
            z["landmark"], z["horizon_end"] = t0, end
            # History eligibility only uses self-reports already collected by t0.
            known_sr = self.sr.loc[self.sr.assessment_date.lt(t0)]
            z["prior_self_report_cancer"] = z.index.isin(known_sr.loc[known_sr.classification.eq("malignant_history"), ID])
            z["ambiguous_self_report"] = z.index.isin(known_sr.loc[known_sr.classification.eq("ambiguous"), ID])
            unknown_sr = self.sr.loc[self.sr.assessment_date.isna() & self.sr.classification.ne("excluded_c44_or_precancer"), ID]
            z["undated_self_report"] = z.index.isin(unknown_sr)
            # Sensitivity flag only: reports collected at/after t0 that date a malignancy before t0.
            later = self.sr.loc[self.sr.assessment_date.ge(t0) & self.sr.classification.eq("malignant_history")]
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
            for name, mask in rules:
                removed = remaining & mask
                remaining &= ~mask
                z.loc[removed, "eligibility_reason"] = name
                flows.append({"landmark": value, "step": name, "removed": int(removed.sum()), "remaining": int(remaining.sum())})
            z["clinical_candidate"] = remaining
            z["eligible"] = remaining & z.coverage_verified & z.registry_start.le(t0) & z.registry_end_exclusive.gt(t0)
            z.loc[z.eligible, "eligibility_reason"] = "eligible"
            z.loc[remaining & z.coverage_verified & ~z.eligible, "eligibility_reason"] = "not_observable_at_landmark"
            hist = self.ev.filter(pl.col("date") < t0).group_by(ID).agg(pl.len().alias("history_codes"), pl.col("date").min().alias("history_start"))
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
            for site, name, _, _, sex in SITES:
                applicable = pd.Series(True, index=z.index) if sex is None else z.sex.eq(sex)
                member = z.first_cancer_sites.str.split("|").apply(lambda xs: site in xs)
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
            rows.append(z.reset_index())
        nodes = pd.concat(rows, ignore_index=True)
        save(nodes, self.out / "landmark_status.parquet")
        save(nodes.loc[nodes.clinical_candidate], self.out / "landmark_samples.parquet")
        save(p.reset_index(), self.out / "participants.parquet")
        pd.DataFrame(flows).to_csv(self.report / "cohort_flow.csv", index=False)
        pd.DataFrame(counts).to_csv(self.report / "landmark_counts.csv", index=False)
        self.nodes = nodes
        self.qc["landmarks"] = {}
        for landmark, g in nodes.groupby("landmark"):
            anytarget = g[[f"observed_{s[0]}_5y" for s in SITES]].any(axis=1)
            self.qc["landmarks"][str(landmark.date())] = {
                "clinical_candidates": int(g.clinical_candidate.sum()), "eligible_with_verified_coverage": int(g.eligible.sum()),
                "observed_top10_people": int(anytarget.sum()),
                "empty_history_candidates": int((g.clinical_candidate & g.empty_history).sum()),
                "followup_status": g.followup_status.value_counts().to_dict()}
        anytarget = nodes[[f"observed_{s[0]}_5y" for s in SITES]].any(axis=1)
        self.qc["unique_observed_top10_people"] = int(nodes.loc[anytarget, ID].nunique())

    def split_summary(self):
        log("Summarizing patient-split balance (aggregate only)")
        p = self.p
        age = (p.recruited - p.birth).dt.days / 365.25

        def describe(g, ages):
            return {"people": len(g), "female_share": g.sex.eq("Female").mean(),
                    "age_q1": ages.quantile(.25), "age_median": ages.median(), "age_q3": ages.quantile(.75)}

        rows = []
        for split, g in p.groupby("split"):
            rows.append({"landmark": "all_participants", "split": split, **describe(g, age.loc[g.index]),
                         "share": len(g) / len(p), "died": g.death.notna().mean(), "lost": g.lost.notna().mean(),
                         "registry_malignancy_ever": g.first_registry_cancer.notna().mean()})
        cand = self.nodes.loc[self.nodes.clinical_candidate]
        target = cand[[f"observed_{s[0]}_5y" for s in SITES]].any(axis=1)
        for (landmark, split), g in cand.groupby(["landmark", "split"]):
            rows.append({"landmark": str(landmark.date()), "split": split, **describe(g, g.age_years_approx),
                         "share": len(g) / int(cand.landmark.eq(landmark).sum()),
                         "empty_history": g.empty_history.mean(), "history_codes_median": g.history_codes.median(),
                         "observed_top10_share": target.loc[g.index].mean()})
        out = pd.DataFrame(rows).round(4)
        out.to_csv(self.report / "split_summary.csv", index=False)
        self.qc["split_fractions"] = p.split.value_counts(normalize=True).round(4).to_dict()

    def inputs(self):
        log("Materializing pre-landmark code sequences for clinical candidates")
        us = pl.Datetime("us")
        nodes = pl.from_pandas(self.nodes.loc[self.nodes.clinical_candidate, [ID, "landmark", "split"]]).with_columns(pl.col("landmark").cast(us))
        ev = self.ev.select(ID, "code", pl.col("date").cast(us), "source")
        x = nodes.join(ev, on=ID, how="inner").filter(pl.col("date") < pl.col("landmark"))
        x = x.sort([ID, "landmark", "date", "code"]).with_columns(
            (pl.col("landmark") - pl.col("date")).dt.total_days().alias("days_before"),
            (pl.col("date").rank("dense").over([ID, "landmark"]) - 1).cast(pl.Int32).alias("day_index"))
        seq = x.group_by([ID, "landmark"], maintain_order=True).agg(
            pl.col("code").alias("codes"), pl.col("date").alias("dates"), "days_before", "day_index",
            pl.col("source").alias("sources"))
        seq = nodes.join(seq, on=[ID, "landmark"], how="left")
        seq = seq.with_columns([pl.col(c).fill_null(pl.lit([], dtype=seq.schema[c])) for c in ["codes", "dates", "days_before", "day_index", "sources"]])
        seq = seq.with_columns(pl.col("codes").list.len().alias("n_codes"),
                               (pl.col("day_index").list.max().fill_null(-1) + 1).alias("n_days"))
        seq.sort([ID, "landmark"]).write_parquet(self.out / "landmark_inputs.parquet", compression="zstd")
        self.qc["landmark_inputs"] = {"rows": seq.height, "codes": int(seq["n_codes"].sum()),
                                      "empty_history": int((seq["n_codes"] == 0).sum())}

    def features(self):
        log("Extracting latest pre-landmark risk factors with assessment Instance and feature age")
        selections = {
            "01_demographics_socioeconomic.csv": ["Ethnic background", "Townsend deprivation index at recruitment"],
            "02_lifestyle.csv": ["Smoking status", "Alcohol intake frequency."],
            "04_anthropometry_physical_cardiovascular.csv": ["Body mass index (BMI)"],
            "05_blood_urine_biochemistry.csv": None,
            "06_blood_count.csv": None,
        }
        # Keep blood chemistry only; urine measures/flags are outside this v1 feature set.
        catalog, frames = [], []
        for value in self.config["landmarks"]:
            t0 = pd.Timestamp(value)
            ids = self.nodes.loc[self.nodes.landmark.eq(t0) & self.nodes.clinical_candidate, ID]
            out = pd.DataFrame(index=pd.Index(ids, name=ID))
            out["landmark"] = t0
            for fn, bases in selections.items():
                path = self.source(self.ehr / "Base_Information_split", fn)
                cols = pd.read_csv(path, nrows=0).columns
                if bases is None:
                    bases = sorted({c.split(" | ")[0] for c in cols if " | Instance " in c and "urine" not in c.lower()})
                chosen = [c for c in cols if c.split(" | ")[0] in bases]
                raw = read(path, usecols=["Participant ID"] + chosen).set_index(ID).reindex(out.index)
                for base in bases:
                    key = re.sub(r"[^a-z0-9]+", "_", base.lower()).strip("_")
                    val = pd.Series(pd.NA, index=out.index, dtype="string")
                    when = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
                    inst = pd.Series(pd.NA, index=out.index, dtype="Int8")
                    for col in [c for c in chosen if c.split(" | ")[0] == base]:
                        match = re.search(r"Instance (\d+)$", col)
                        i = int(match.group(1)) if match else 0  # recruitment-only Townsend
                        d = dates(self.vis[f"Date of attending assessment centre | Instance {i}"]).reindex(out.index)
                        v = raw[col].astype("string")
                        missing = v.isna() | v.str.lower().isin(["do not know", "prefer not to answer", "not known"])
                        if base in ["Ethnic background", "Smoking status", "Alcohol intake frequency."]:
                            missing |= v.isin(["-1", "-3"])
                        use = d.lt(t0) & ~missing & (when.isna() | d.gt(when))
                        val.loc[use], when.loc[use], inst.loc[use] = v.loc[use], d.loc[use], i
                    out[key] = val
                    out[key + "__measured_at"] = when
                    out[key + "__instance"] = inst
                    out[key + "__age_days"] = (t0 - when).dt.days.astype("Int32")
                    out[key + "__missing"] = val.isna()
                    if value == self.config["landmarks"][0]:
                        catalog.append({"feature": key, "source_file": fn, "source_field": base,
                                        "time_basis": "assessment Instance date; assay availability date not exported"})
            frames.append(out.reset_index())
        features = pd.concat(frames, ignore_index=True)
        save(features, self.out / "features_asof.parquet")
        pd.DataFrame(catalog).to_csv(self.report / "feature_catalog.csv", index=False)
        self.qc["features"] = {"rows": len(features), "fields": len(catalog), "time_basis": "assessment-date retrospective measurements"}

    def pretraining(self):
        log("Materializing train-participant histories before pretraining cutoff")
        p = pl.from_pandas(self.p[["split", "pretrain_role"]].reset_index())
        ev = self.ev.join(p, on=ID).filter((pl.col("split") == "train") & (pl.col("date") < pd.Timestamp(self.config["pretrain_before"])))
        ev.write_parquet(self.out / "pretrain_events.parquet", compression="zstd")
        manifest = self.p.loc[self.p.split.eq("train"), ["split", "pretrain_role"]].reset_index()
        n = ev.group_by(ID).len().to_pandas().set_index(ID)["len"]
        manifest["history_codes"] = manifest[ID].map(n).fillna(0).astype(int)
        save(manifest, self.out / "pretrain_participants.parquet")
        vocab = ev.filter(pl.col("pretrain_role") == "train").group_by("code").len().sort("code").to_pandas()
        vocab["token_id"] = np.arange(len(vocab)) + 5
        vocab.to_csv(self.report / "vocabulary.csv", index=False)
        self.qc["pretraining"] = {"participants": len(manifest), "events": ev.height, "vocabulary_codes": len(vocab),
                                  "internal_roles": manifest.pretrain_role.value_counts().to_dict(),
                                  "special_tokens": {"PAD": 0, "UNK": 1, "MASK": 2, "CLS": 3, "EMPTY": 4}}

    def finish(self):
        if self.quarantines:
            q = pd.concat(self.quarantines, ignore_index=True)
            # Heterogeneous raw-source fields are retained in the private QC table.
            save(q, self.out / "quarantine.parquet")
            self.qc["quarantine"] = q.groupby(["source", "reason"]).size().to_dict()
            self.qc["quarantine"] = {"/".join(k): int(v) for k, v in self.qc["quarantine"].items()}
        self.qc["coverage_status"] = self.coverage["status"]
        self.qc["elapsed_seconds"] = round(time.time() - START, 1)
        dump(self.qc, self.report / "build_summary.json")
        inventory = [{"file": p.name, "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sorted(self.used)]
        code_hash = hashlib.sha256(Path(__file__).read_bytes() + Path(__file__).with_name("definitions.py").read_bytes()).hexdigest()
        dump({"config": self.config, "coverage": self.coverage, "sources": inventory, "pipeline_sha256": code_hash,
              "source_fingerprints": "size and mtime, not cryptographic raw-file hashes"}, self.report / "build_manifest.json")
        (self.out / "BUILD_COMPLETE.json").write_text(json.dumps({"pipeline_sha256": code_hash, "coverage_status": self.coverage["status"]}), encoding="utf-8")
        log("Research tables complete; aggregate summary written")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ehr", type=Path, required=True)
    ap.add_argument("--legacy", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=Path("configs/cohort.json"))
    ap.add_argument("--output", type=Path, default=Path("data/processed"))
    ap.add_argument("--report", type=Path, default=Path("ccfa-workfiles/checks/cancer-cohort"))
    run(ap.parse_args())


def run(args):
    b = Builder(args)
    (b.out / "BUILD_COMPLETE.json").unlink(missing_ok=True)
    b.participants()
    b.registry()
    b.histories()
    b.self_reports()
    b.landmarks()
    b.split_summary()
    b.inputs()
    b.features()
    b.pretraining()
    b.finish()
    return b


if __name__ == "__main__":
    main()
