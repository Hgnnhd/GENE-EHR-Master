"""Competing-risk targets: what happened first in [t0, t0 + horizon) and when.

Outcome classes (one softmax per time bin in the deep models):
  0            no event in this bin
  1 .. 10      first malignancy is this target site (SITE_KEYS order)
  11           first malignancy is another / secondary / unknown-primary cancer
  12           death without a prior malignancy
A first diagnosis can involve several sites on the same day; it is kept as a set of causes.
Censored people have no cause and contribute the bins they were followed through.
"""
import numpy as np
import pandas as pd

from .definitions import SITES

SITE_KEYS = [s[0] for s in SITES]
CAUSES = SITE_KEYS + ["other_cancer", "death"]   # class index = cause index + 1
N_CLASSES = 1 + len(CAUSES)
OTHER, DEATH = len(SITE_KEYS), len(SITE_KEYS) + 1
COLUMNS = ["landmark", "sex", "eligible", "followup_status", "followup_days", "first_cancer_sites",
           "first_registry_cancer", "death", "lost"]


def horizon_days(years):
    return years * 365.25


def applicable_classes(sex):
    """(N, N_CLASSES) bool: sex-restricted sites are impossible outcomes for the other sex."""
    sex = np.asarray(sex)
    out = np.ones((len(sex), N_CLASSES), dtype=bool)
    for j, (_, _, _, _, restrict) in enumerate(SITES):
        if restrict is not None:
            out[:, 1 + j] = sex == restrict
    return out


def _cause_matrix(sites, mask):
    """Multi-hot causes from first_cancer_sites ("lung|other", ...) for rows in mask."""
    out = np.zeros((len(sites), len(CAUSES)), dtype=bool)
    index = {s: j for j, s in enumerate(SITE_KEYS)}
    for i in np.flatnonzero(mask):
        for site in str(sites[i]).split("|"):
            if site:
                out[i, index.get(site, OTHER)] = True
    return out


def competing_targets(samples, mode, years):
    """time (days, censoring or event; event-free = horizon), cause (N, 12 bool), known (N bool).

    verified: follow-up from the cohort stage (requires verified registry coverage).
    provisional_observed: registry-observed events only, no administrative censoring
    (pipeline debugging; treats the registry as complete).
    """
    n, h = len(samples), horizon_days(years)
    t0 = pd.to_datetime(samples.landmark).to_numpy()
    sites = samples.first_cancer_sites.fillna("").to_numpy()
    if mode == "verified":
        status = samples.followup_status.to_numpy()
        known = samples.eligible.fillna(False).to_numpy(bool) & np.isin(status, ["cancer", "death", "censored", "event_free_5y"])
        time = samples.followup_days.astype("Float64").to_numpy(dtype=float, na_value=np.nan)
        cancer, death = known & (status == "cancer"), known & (status == "death")
        free = known & (status == "event_free_5y")
    elif mode == "provisional_observed":
        days = lambda col: (pd.to_datetime(samples[col]).to_numpy() - t0) / np.timedelta64(1, "D")
        fc, dd, lost = days("first_registry_cancer"), days("death"), days("lost")
        nan_ok = lambda a, cond: np.isnan(a) | cond
        cancer = (fc >= 0) & (fc < h) & nan_ok(dd, fc <= dd) & nan_ok(lost, fc < lost)
        death = ~cancer & (dd > 0) & (dd < h) & nan_ok(lost, dd < lost)
        censored = ~cancer & ~death & (lost > 0) & (lost < h)
        free = ~cancer & ~death & ~censored
        time = np.where(cancer, fc, np.where(death, dd, np.where(censored, lost, h)))
        known = np.ones(n, dtype=bool)
    else:
        raise ValueError(mode)
    cause = _cause_matrix(sites, cancer)
    cause[death, DEATH] = True
    time = np.where(free, h, time)
    return {"time": time.astype(np.float64), "cause": cause, "known": known,
            "applicable": applicable_classes(samples.sex)}


def single_site_targets(targets, site):
    """Targets for a single-site model (ablation of joint training).

    Outcomes collapse to: no event | `site` | other first cancer (any other target site,
    other or unknown primary) | death. The other target-site classes are made inapplicable,
    so the same 13-class head trains as a 4-class competing-risk model.
    """
    j = SITE_KEYS.index(site)
    cause = targets["cause"]
    collapsed = np.zeros_like(cause)
    collapsed[:, j] = cause[:, j]
    others = [k for k in range(len(SITE_KEYS)) if k != j] + [OTHER]
    collapsed[:, OTHER] = cause[:, others].any(1)
    collapsed[:, DEATH] = cause[:, DEATH]
    applicable = targets["applicable"].copy()
    applicable[:, [1 + k for k in range(len(SITE_KEYS)) if k != j]] = False
    return {**targets, "cause": collapsed, "applicable": applicable}


def single_name(name, site):
    return f"{name}__single_{site}"


def log_prior(targets, event_bin, survived, n_bins, rows):
    """(bins, classes) log of the empirical outcome distribution per time bin among those entering it.

    Used to initialise the output bias so training starts from the observed baseline risk
    (events are rare: about 1% a year), instead of spending early epochs learning it.
    """
    cause, known = targets["cause"][rows], targets["known"][rows]
    eb, sv = event_bin[rows], survived[rows]
    out = np.zeros((n_bins, N_CLASSES))
    for j in range(n_bins):
        event = known & (eb == j)
        counts = np.concatenate([[(known & (sv > j)).sum()], cause[event].sum(0)]) + 0.5  # add-half smoothing
        out[j] = np.log(counts / counts.sum())
    return out


def bin_targets(targets, years, n_bins):
    """Discrete-time view: event bin (-1 if none) and number of bins survived."""
    width = horizon_days(years) / n_bins
    time, cause, known = targets["time"], targets["cause"], targets["known"]
    has_event = known & cause.any(1)
    index = np.clip(np.floor(np.nan_to_num(time, nan=0.0) / width), 0, n_bins).astype(np.int64)
    event_bin = np.where(has_event, np.minimum(index, n_bins - 1), -1)
    survived = np.where(has_event, event_bin, np.where(known, index, 0))
    return event_bin.astype(np.int64), survived.astype(np.int64)
