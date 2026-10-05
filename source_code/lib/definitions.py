import numpy as np
import pandas as pd

SITES = [
    ("lung", "肺癌", [33, 34], [162], None),
    ("colorectal", "结直肠癌（含肛门）", [18, 19, 20, 21], [153, 154], None),
    ("liver", "肝癌及肝内胆管癌", [22], [155], None),
    ("breast", "女性乳腺癌", [50], [174], "Female"),
    ("stomach", "胃癌", [16], [151], None),
    ("pancreas", "胰腺癌", [25], [157], None),
    ("oesophagus", "食管癌", [15], [150], None),
    ("prostate", "前列腺癌", [61], [185], "Male"),
    ("leukaemia", "白血病", list(range(91, 96)), list(range(204, 209)), None),
    ("cervix", "宫颈癌", [53], [180], "Female"),
]
# C76-C80 / ICD-9 195-199: ill-defined, secondary or unknown primary; never a target site.
UNSPECIFIED_SITE = "secondary_or_unknown_primary"
FOLLOWUP_EVALUABLE = ["cancer", "death", "censored", "event_free_5y"]


def dates(s):
    return pd.to_datetime(s, format="mixed", errors="coerce")


def classify(c10, c9, sex):
    n10 = pd.to_numeric(c10.str.extract(r"^C(\d{2})", expand=False), errors="coerce")
    n9 = pd.to_numeric(c9.str.extract(r"^(\d{3})", expand=False), errors="coerce")
    fallback = c10.isna() | c10.eq("")
    malignant = (n10.between(0, 97) & n10.ne(44)) | (fallback & n9.between(140, 208) & n9.ne(173))
    site = pd.Series("", index=c10.index)
    for key, _, codes, oldcodes, restrict in SITES:
        m = n10.isin(codes) | (fallback & n9.isin(oldcodes))
        if restrict:
            m &= sex.eq(restrict)
        site.loc[m] = key
    unspecified = n10.between(76, 80) | (fallback & n9.between(195, 199))
    site.loc[site.eq("") & malignant & unspecified] = UNSPECIFIED_SITE
    return site, malignant, fallback & n9.notna()


def patient_split(ids, seed=42):
    """Exact 70/15/remainder split independent of input row order."""
    ids = np.sort(np.asarray(ids, dtype=str))
    order = np.random.default_rng(seed).permutation(len(ids))
    a, b = int(len(ids) * .70), int(len(ids) * .85)
    values = np.full(len(ids), "test", dtype=object)
    values[order[:a]] = "train"
    values[order[a:b]] = "validation"
    return pd.Series(values, index=ids, name="split")


def resolve_followup(t0, end, cancer, death, lost, start, stop):
    """Coverage is [start, stop). Event/loss ties remain unresolved.

    Observed cancer/death ties are cancer events (diagnosis on death date).
    Unknown administrative coverage never produces an evaluable endpoint.
    """
    if pd.isna(start) or pd.isna(stop):
        return "coverage_unverified", pd.NaT
    if start > t0 or stop <= t0:
        return "not_observable_at_landmark", pd.NaT
    censor = min([end, stop] + ([] if pd.isna(lost) else [lost]))
    event_date = min([x for x in [cancer, death] if pd.notna(x)], default=pd.NaT)
    if pd.notna(event_date) and event_date < t0:
        return "pre_landmark_event", pd.NaT
    if pd.notna(event_date) and event_date == lost and lost < end and lost < stop:
        return "event_loss_tie_unresolved", pd.NaT
    if pd.notna(event_date) and event_date < censor:
        return ("cancer" if cancer == event_date else "death"), event_date
    return ("event_free_5y" if censor == end else "censored"), censor


def resolve_followup_frame(t0, end, cancer, death, lost, start, stop):
    """Vectorized resolve_followup over aligned Series; returns (status, observation_end)."""
    idx = cancer.index
    unverified = start.isna() | stop.isna()
    unobservable = start.gt(t0) | stop.le(t0)
    censor = pd.concat([pd.Series(end, index=idx), stop, lost], axis=1).min(axis=1)
    event = pd.concat([cancer, death], axis=1).min(axis=1)
    pre = event.lt(t0)
    tie = event.notna() & event.eq(lost) & lost.lt(end) & lost.lt(stop)
    hit = event.lt(censor)
    conditions = [unverified, unobservable, pre, tie, hit & cancer.eq(event), hit, censor.eq(end)]
    labels = ["coverage_unverified", "not_observable_at_landmark", "pre_landmark_event",
              "event_loss_tie_unresolved", "cancer", "death", "event_free_5y"]
    status = pd.Series(np.select(conditions, labels, "censored"), index=idx, dtype=object)
    observed = event.where(hit, censor).where(~(unverified | unobservable | pre | tie))
    return status, observed


def site_event_code(status, member, applicable):
    """Competing-risk coding for one target site: 0 censored/event-free, 1 target first cancer,
    2 competing event (death or another first malignancy); NA when not evaluable."""
    code = pd.Series(pd.NA, index=status.index, dtype="Int8")
    defined = status.isin(FOLLOWUP_EVALUABLE) & applicable
    code.loc[defined] = 0
    code.loc[defined & status.isin(["cancer", "death"])] = 2
    code.loc[defined & status.eq("cancer") & member] = 1
    return code
