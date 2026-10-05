"""Discrimination and calibration metrics with participant-level bootstrap."""
import numpy as np


def auroc(y, p):
    """Mann-Whitney AUROC with tie correction (average ranks)."""
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    pos = y == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    sp = p[order]
    ranks = np.empty(len(p))
    # average ranks for ties
    boundaries = np.flatnonzero(np.diff(sp)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(p)]])
    avg = (starts + ends + 1) / 2.0
    ranks[order] = np.repeat(avg, ends - starts)
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auprc(y, p):
    """Average precision (step-wise area under the precision-recall curve)."""
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    n1 = int((y == 1).sum())
    if n1 == 0:
        return float("nan")
    order = np.argsort(-p, kind="mergesort")
    sp, hits = p[order], (y[order] == 1).astype(np.float64)
    last = np.r_[np.flatnonzero(np.diff(sp)), len(sp) - 1]  # tied scores form one threshold
    tp = np.cumsum(hits)[last]
    precision, recall = tp / (last + 1), tp / n1
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def calibration(y, p):
    """Calibration-in-the-large (mean predicted - observed) and calibration slope (logistic fit on logit p)."""
    y, p = np.asarray(y, dtype=np.float64), np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    x = np.log(p / (1 - p))
    X = np.stack([np.ones_like(x), x], axis=1)
    beta = np.array([0.0, 1.0])
    for _ in range(50):
        mu = 1 / (1 + np.exp(-(X @ beta)))
        w = mu * (1 - mu)
        hess = X.T @ (X * w[:, None]) + 1e-9 * np.eye(2)
        step = np.linalg.solve(hess, X.T @ (y - mu))
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return float(p.mean() - y.mean()), float(beta[1])


# ---------------------------------------------------------------------------
# Competing risks with right censoring (time in days; cause = first-event type).

def censoring_survival(time, censored):
    """Kaplan-Meier estimate of the censoring survival G(t) = P(C > t); returns G(t-) evaluator."""
    time = np.asarray(time, dtype=np.float64)
    censored = np.asarray(censored, dtype=bool)
    grid = np.unique(time[censored])
    if not len(grid):
        return lambda t: np.ones_like(np.asarray(t, dtype=np.float64))
    order = np.sort(time)
    at_risk = len(time) - np.searchsorted(order, grid, side="left")
    drops = np.searchsorted(np.sort(time[censored]), grid, side="right") - np.searchsorted(np.sort(time[censored]), grid, side="left")
    surv = np.cumprod(1 - drops / at_risk)

    def before(t):
        """G(t-): product over censoring times strictly before t."""
        idx = np.searchsorted(grid, np.asarray(t, dtype=np.float64), side="left")
        return np.where(idx == 0, 1.0, surv[np.maximum(idx - 1, 0)])
    return before


def weighted_auroc(case_p, case_w, ctrl_p, ctrl_w):
    """P(score_case > score_control) with case and control weights (ties count half)."""
    if not len(case_p) or not len(ctrl_p) or case_w.sum() <= 0 or ctrl_w.sum() <= 0:
        return float("nan")
    order = np.argsort(ctrl_p, kind="mergesort")
    sp, cw = ctrl_p[order], np.cumsum(ctrl_w[order])
    lo = np.searchsorted(sp, case_p, side="left")
    hi = np.searchsorted(sp, case_p, side="right")
    below = np.where(lo > 0, cw[np.maximum(lo - 1, 0)], 0.0)
    upto = np.where(hi > 0, cw[np.maximum(hi - 1, 0)], 0.0)
    return float((case_w * (below + 0.5 * (upto - below))).sum() / (case_w.sum() * ctrl_w.sum()))


def aalen_johansen(time, cause_k, any_event, t):
    """Observed cumulative incidence of cause k by time t (Aalen-Johansen), censoring-aware."""
    time = np.asarray(time, dtype=np.float64)
    event_times = time[any_event & (time <= t)]
    if not len(event_times):
        return 0.0
    grid, d_any = np.unique(event_times, return_counts=True)
    d_k = np.zeros(len(grid))
    kt, kc = np.unique(time[cause_k & (time <= t)], return_counts=True)
    d_k[np.searchsorted(grid, kt)] = kc
    n = len(time) - np.searchsorted(np.sort(time), grid, side="left")
    surv_before = np.concatenate([[1.0], np.cumprod(1 - d_any / n)[:-1]])
    return float((surv_before * d_k / n).sum())


def competing_metrics(time, cause_k, any_event, p, t, horizon, bootstrap=0, seed=0):
    """Time-dependent metrics at t for cause k (cumulative cases / dynamic controls, IPCW).

    cases     cause k observed by t                       weight 1/G(T-)
    controls  no event by t (followed to t)               weight 1/G(t-)
              other cause observed by t                   weight 1/G(T-)
    censored before t without an event: weight 0 (their information enters through G).
    Administrative end of follow-up at the horizon is not censoring before the horizon.
    """
    time, p = np.asarray(time, dtype=np.float64), np.asarray(p, dtype=np.float64)
    cause_k, any_event = np.asarray(cause_k, bool), np.asarray(any_event, bool)

    def compute(idx):
        tm, ck, ev, pr = time[idx], cause_k[idx], any_event[idx], p[idx]
        G = censoring_survival(tm, ~ev & (tm < horizon))
        case = ck & (tm <= t)
        other = ev & ~ck & (tm <= t)
        free = (tm >= t) & ~(ev & (tm <= t))
        w = np.zeros(len(tm))
        w[case | other] = 1 / np.maximum(G(tm[case | other]), 1e-12)
        w[free] = 1 / np.maximum(G(np.full(free.sum(), t)), 1e-12)
        ctrl = other | free
        out = {"auroc": weighted_auroc(pr[case], w[case], pr[ctrl], w[ctrl]),
               "brier": float((w * (case.astype(float) - pr) ** 2).sum() / len(tm))}
        return out, case, ctrl

    everyone = np.arange(len(time))
    stats, case, ctrl = compute(everyone)
    observed = aalen_johansen(time, cause_k, any_event, t)
    out = {"n": int(len(time)), "cases": int(case.sum()), "controls": int(ctrl.sum()),
           "censored_before_t": int((~case & ~ctrl).sum()), **stats,
           "observed_cif": observed, "mean_predicted": float(p.mean()),
           "calibration_in_the_large": float(p.mean() - observed)}
    if bootstrap and out["cases"]:
        rng = np.random.default_rng(seed)
        draws = [compute(rng.integers(0, len(time), len(time)))[0]["auroc"] for _ in range(bootstrap)]
        draws = np.array([d for d in draws if not np.isnan(d)])
        if len(draws):
            out["auroc_lo"], out["auroc_hi"] = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    return out
