"""End-to-end build + validate on a tiny synthetic UKB-shaped export."""
import argparse
import json

import pandas as pd
import polars as pl
import pytest

from cancer_ehr.build import run
from cancer_ehr.validate import validate

ID = "Participant ID"
A0, A2 = "Date of attending assessment centre | Instance 0", "Date of attending assessment centre | Instance 2"
# id: sex, birth, recruited, instance-2 visit, lost, death
PEOPLE = {
    "1": ("Female", "1950-01-01", "2008-05-01", None, None, None),        # event-free both landmarks
    "2": ("Male", "1945-03-01", "2007-04-01", None, None, None),          # prostate 2013 -> A case, B prior
    "3": ("Female", "1955-06-01", "2009-02-01", None, None, None),        # breast 2009 -> prior cancer
    "4": ("Male", "1948-07-01", "2008-01-01", None, None, "2014-02-02"),  # death in A, dead at B
    "5": ("Female", "1952-01-01", "2010-06-01", None, "2012-01-01", None),  # lost in A
    "6": ("Male", "1949-01-01", "2007-01-01", None, None, None),          # prior hospital C34
    "7": ("Female", "1951-01-01", "2008-01-01", None, None, None),        # prior self-reported cancer
    "8": ("Male", "1953-01-01", "2011-06-01", None, None, None),          # recruited after A; colorectal in B
    "9": ("Female", "1954-01-01", "2009-01-01", None, None, None),        # cervix via ICD-9 2017
    "10": ("Male", "1946-01-01", "2008-01-01", None, None, None),         # secondary C78 = competing
    "11": ("Female", "1947-01-01", "2008-01-01", None, None, None),       # registry date without code
    "12": ("Male", "1950-01-01", "2008-01-01", "2014-05-01", None, None),  # histories; coverage ends 2018
    "13": ("Female", "1950-01-01", "2008-01-01", "2014-05-01", None, None),  # later self-report of 2009 cancer
}
REGISTRY = {"2": ("2013-06-01", "C61 prostate", None), "3": ("2009-01-01", "C50.9 breast", None),
            "8": ("2018-01-01", "C18.7 colon", None), "9": ("2017-03-01", None, "1809"),
            "10": ("2012-05-05", "C78.0 secondary lung", None), "11": ("2012-01-01", None, None),
            "12": ("2019-02-01", "C25.0 pancreas", None)}


def write(path, rows, columns=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


@pytest.fixture()
def built(tmp_path, monkeypatch):
    ehr, old = tmp_path / "ehr", tmp_path / "legacy"
    ids = list(PEOPLE)
    write(ehr / "UKB_visit_and_followup_dates.csv",
          [{ID: i, A0: v[2], "Date of attending assessment centre | Instance 1": None, A2: v[3],
            "Date of attending assessment centre | Instance 3": None, "Date lost to follow-up": v[4]} for i, v in PEOPLE.items()])
    write(ehr / "UKB_death.csv", [{ID: i, "Date of death": v[5]} for i, v in PEOPLE.items() if v[5]])
    fo = {i: {ID: i} for i in ids}
    fo["12"].update({"Date I10 first reported (essential hypertension)": "2005-03-01",
                     "Date E11 first reported (type 2 diabetes)": "2012-07-01"})
    fo["1"]["Date I10 first reported (essential hypertension)"] = "Code has event date matching participant's date of birth"
    write(ehr / "UKB_First_occurrences.csv", list(fo.values()))
    sr = {i: {ID: i} for i in ids}
    sr["7"].update({"Cancer code, self-reported | Instance 0 | Array 0": "1002",
                    "Interpolated Year when cancer first diagnosed | Instance 0 | Array 0": "2004.5"})
    sr["13"].update({"Cancer code, self-reported | Instance 2 | Array 0": "1002",
                     "Interpolated Year when cancer first diagnosed | Instance 2 | Array 0": "2009.5"})
    sr["1"].update({"Cancer code, self-reported | Instance 0 | Array 0": "1060"})
    write(ehr / "UKB_self_reported_conditions.csv", list(sr.values()))
    split = ehr / "Base_Information_split"
    write(split / "01_demographics_socioeconomic.csv",
          [{ID: i, "Ethnic background | Instance 0": "British", "Townsend deprivation index at recruitment": "-2.1"} for i in ids])
    write(split / "02_lifestyle.csv", [{ID: i, "Smoking status | Instance 0": "Never", "Smoking status | Instance 2": "Previous",
                                        "Alcohol intake frequency. | Instance 0": "-3"} for i in ids])
    write(split / "04_anthropometry_physical_cardiovascular.csv",
          [{ID: i, "Body mass index (BMI) | Instance 0": "25.1", "Body mass index (BMI) | Instance 2": "26.0"} for i in ids])
    write(split / "05_blood_urine_biochemistry.csv",
          [{ID: i, "Albumin | Instance 0": "45", "Microalbumin in urine | Instance 0": "6"} for i in ids])
    write(split / "06_blood_count.csv", [{ID: i, "Haemoglobin concentration | Instance 0": "14"} for i in ids])

    rec = []
    for i, v in PEOPLE.items():
        row = {ID: i, "Sex": "1" if v[0] == "Female" else "0", "Date of birth": v[1], "Diagnoses - ICD10": None}
        row.update({f"Date of first in-patient diagnosis - ICD10 | Array {k}": None for k in range(3)})
        rec.append(row)
    rec[ids.index("6")].update({"Diagnoses - ICD10": "C34.1 lung|I10 hypertension",
                                "Date of first in-patient diagnosis - ICD10 | Array 0": "2010-01-01",
                                "Date of first in-patient diagnosis - ICD10 | Array 1": "2009-01-01"})
    rec[ids.index("12")].update({"Diagnoses - ICD10": "I10 hypertension|K80 gallstones|J45 asthma",
                                 "Date of first in-patient diagnosis - ICD10 | Array 0": "2004-01-01",
                                 "Date of first in-patient diagnosis - ICD10 | Array 1": "2005-03-01",
                                 "Date of first in-patient diagnosis - ICD10 | Array 2": "2013-09-09"})
    write(old / "record.csv", rec)
    write(old / "cancer.csv", [{ID: i, "Sex": v[0],
                                "Date of cancer diagnosis | Instance 0": REGISTRY.get(i, (None,))[0],
                                "Type of cancer: ICD10 | Instance 0": REGISTRY.get(i, (None, None))[1],
                                "Type of cancer: ICD9 | Instance 0": REGISTRY.get(i, (None, None, None))[2]}
                               for i, v in PEOPLE.items()])
    codes = {"1002": "breast cancer", "1060": "non-melanoma skin cancer", "1061": "basal cell carcinoma",
             "1062": "squamous cell carcinoma", "1073": "rodent ulcer", "1072": "cervical intra-epithelial neoplasia",
             "1003": "skin cancer", "1051": "myelofibrosis", "99999": "unclassifiable"}
    write(old / "app176660_20240512000635.dataset.codings.csv",
          [{"coding_name": "data_coding_3", "code": k, "meaning": v} for k, v in codes.items()])

    cov = tmp_path / "coverage.csv"
    write(cov, [{"participant_id": i, "registry_start": "1990-01-01",
                 "registry_end_exclusive": "2018-01-01" if i == "12" else "2021-01-01"} for i in ids])
    manifest = tmp_path / "registry_coverage.json"
    manifest.write_text(json.dumps({"status": "verified", "source_version": "synthetic", "evidence": "test",
                                    "coverage_file": str(cov)}))
    config = tmp_path / "cohort.json"
    config.write_text(json.dumps({"landmarks": ["2011-01-01", "2016-01-01"], "horizon_years": 5, "seed": 42,
                                  "pretrain_before": "2016-01-01", "pretrain_internal_validation_fraction": 0.2,
                                  "coverage_manifest": str(manifest)}))
    args = argparse.Namespace(ehr=ehr, legacy=old, config=config, output=tmp_path / "out", report=tmp_path / "report")
    run(args)
    return args


def node(nodes, pid, landmark):
    return nodes.loc[(nodes.participant_id == pid) & (nodes.landmark == pd.Timestamp(landmark))].iloc[0]


def test_synthetic_build_cohorts_and_labels(built):
    nodes = pd.read_parquet(built.output / "landmark_status.parquet")
    a, b = "2011-01-01", "2016-01-01"
    assert node(nodes, "1", a).followup_status == "event_free_5y"
    r = node(nodes, "2", a)
    assert (r.followup_status, r.label_prostate_5y, r.event_prostate, r.event_lung) == ("cancer", 1, 1, 2)
    assert node(nodes, "2", b).eligibility_reason == "prior_registry_cancer"
    assert node(nodes, "3", a).eligibility_reason == "prior_registry_cancer"
    assert pd.isna(node(nodes, "3", a).event_breast)
    r = node(nodes, "4", a)
    assert (r.followup_status, r.event_lung, r.label_lung_5y) == ("death", 2, 0)
    assert node(nodes, "4", b).eligibility_reason == "dead_at_landmark"
    r = node(nodes, "5", a)
    assert (r.followup_status, r.event_breast, r.followup_days) == ("censored", 0, 365)
    assert pd.isna(r.label_breast_5y)
    assert node(nodes, "6", a).eligibility_reason == "prior_hospital_cancer"
    assert node(nodes, "7", a).eligibility_reason == "prior_self_report_cancer"
    assert node(nodes, "8", a).eligibility_reason == "not_recruited"
    assert node(nodes, "8", b).label_colorectal_5y == 1
    r = node(nodes, "9", b)
    assert (r.label_cervix_5y, r.first_cancer_sites) == (1, "cervix")
    assert node(nodes, "9", a).followup_status == "event_free_5y"
    r = node(nodes, "10", a)
    assert (r.first_cancer_sites, r.event_lung, r.label_lung_5y) == ("secondary_or_unknown_primary", 2, 0)
    assert pd.isna(r.event_prostate) is False and pd.isna(r.event_breast)  # sex-restricted site not applicable
    assert node(nodes, "11", a).eligibility_reason == "unresolved_history_or_dates"
    r = node(nodes, "12", b)
    assert (r.followup_status, r.event_pancreas, r.history_codes) == ("censored", 0, 4)
    assert node(nodes, "13", a).later_self_report_prior_cancer and node(nodes, "13", a).clinical_candidate
    assert node(nodes, "13", b).eligibility_reason == "prior_self_report_cancer"


def test_synthetic_inputs_and_validation(built):
    li = pl.read_parquet(built.output / "landmark_inputs.parquet")
    r = li.filter((pl.col("participant_id") == "12") & (pl.col("landmark").dt.year() == 2016)).row(0, named=True)
    assert r["codes"] == ["I10", "K80", "E11", "J45"]
    assert r["day_index"] == [0, 1, 2, 3] and r["n_days"] == 4 and all(d > 0 for d in r["days_before"])
    r = li.filter((pl.col("participant_id") == "12") & (pl.col("landmark").dt.year() == 2011)).row(0, named=True)
    assert r["codes"] == ["I10", "K80"]
    assert li.filter(pl.col("participant_id") == "1")["n_codes"].to_list() == [0, 0]
    result = validate(built.output, built.config)
    assert result["passed"] > 50
    summary = pd.read_csv(built.report / "split_summary.csv")
    assert set(summary.split) <= {"train", "validation", "test"} and "participant_id" not in summary.columns
