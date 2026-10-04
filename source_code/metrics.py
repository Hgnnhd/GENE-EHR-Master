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


def summarize(y, p, bootstrap=0, seed=0):
    y, p = np.asarray(y), np.asarray(p)
    out = {"n": int(len(y)), "cases": int((y == 1).sum())}
    if out["cases"] == 0 or out["cases"] == out["n"]:
        return out
    out["auroc"], out["auprc"] = auroc(y, p), auprc(y, p)
    out["brier"] = float(np.mean((p - y) ** 2))
    out["calibration_in_the_large"], out["calibration_slope"] = calibration(y, p)
    if bootstrap:
        rng = np.random.default_rng(seed)
        draws = []
        for _ in range(bootstrap):
            idx = rng.integers(0, len(y), len(y))
            draws.append(auroc(y[idx], p[idx]))
        draws = np.array([d for d in draws if not np.isnan(d)])
        if len(draws):
            out["auroc_lo"], out["auroc_hi"] = float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))
    return out
