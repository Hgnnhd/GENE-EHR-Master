import pandas as pd
from cancer_ehr.definitions import classify, patient_split, resolve_followup

D = pd.Timestamp
T0, END = D("2011-01-01"), D("2016-01-01")


def outcome(cancer=pd.NaT, death=pd.NaT, lost=pd.NaT, start=D("2000-01-01"), stop=D("2021-01-01")):
    return resolve_followup(T0, END, cancer, death, lost, start, stop)


def test_half_open_cancer_boundaries():
    assert outcome(cancer=T0) == ("cancer", T0)
    assert outcome(cancer=END) == ("event_free_5y", END)
    assert outcome(cancer=D("2015-12-31"))[0] == "cancer"


def test_competing_events_and_censoring():
    assert outcome(cancer=D("2013-01-01"), death=D("2012-01-01"))[0] == "death"
    assert outcome(cancer=D("2013-01-01"), lost=D("2012-01-01"))[0] == "censored"
    assert outcome(stop=D("2014-01-01")) == ("censored", D("2014-01-01"))
    assert outcome(cancer=D("2014-01-01"), stop=D("2014-01-01"))[0] == "censored"
    assert outcome(start=D("2012-01-01"))[0] == "not_observable_at_landmark"


def test_unknown_coverage_never_negative():
    assert outcome(stop=pd.NaT)[0] == "coverage_unverified"
    assert outcome(cancer=D("2012-01-01"), stop=pd.NaT)[0] == "coverage_unverified"


def test_date_ties():
    event = D("2012-01-01")
    assert outcome(cancer=event, death=event)[0] == "cancer"
    assert outcome(cancer=event, lost=event)[0] == "event_loss_tie_unresolved"


def test_registry_fallback_and_sex_rules():
    c10 = pd.Series(["C44.9 skin", "D05 breast", "C50.9 breast", "C50.9 breast", None, "D00", "C78.0 secondary", "C21.0 anus"])
    c9 = pd.Series([None, None, None, None, "1749 breast", "162 lung", None, None])
    sex = pd.Series(["Female", "Female", "Female", "Male", "Female", "Male", "Male", "Male"])
    site, malignancy, fallback = classify(c10, c9, sex)
    assert site.tolist() == ["", "", "breast", "", "breast", "", "", "colorectal"]
    assert malignancy.tolist() == [False, False, True, True, True, False, True, True]
    assert fallback.sum() == 1


def test_splits_are_stable_across_input_order():
    ids = [str(i) for i in range(100)]
    a, b = patient_split(ids), patient_split(ids[::-1])
    pd.testing.assert_series_equal(a, b)
    assert a.value_counts().to_dict() == {"train": 70, "validation": 15, "test": 15}
