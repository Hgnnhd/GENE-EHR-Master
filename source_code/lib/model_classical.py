"""Classical baselines on bag-of-codes: logistic regression, random forest, LightGBM, XGBoost.

Features: presence of each vocabulary code before the landmark plus the shared static
context (age at landmark, sex, landmark). One model per cancer site; gradient-boosting
models stop early on the validation split. XGBoost uses the GPU when one is visible.
"""
import time

import numpy as np
from scipy import sparse

from .common import banner, log, progress
from .model_metrics import auroc
from .model_data import SITE_KEYS, SPECIAL, NodeData, check_build, load_config, model_dir, require_labels, write_predictions


def design_matrix(data):
    """CSR: one column per vocabulary token (presence) followed by the 3 static features."""
    lengths = np.array([len(t) for t in data.tokens])
    rows = np.repeat(np.arange(len(lengths)), lengths)
    cols = np.concatenate(data.tokens).astype(np.int64)
    keep = cols >= len(SPECIAL)
    codes = sparse.csr_matrix((np.ones(keep.sum(), np.float32), (rows[keep], cols[keep])), shape=(len(lengths), data.vocab_size))
    codes.data[:] = 1.0  # duplicates collapse to presence
    return sparse.hstack([codes, sparse.csr_matrix(data.static)], format="csr", dtype=np.float32)


def make(name, params, gpu):
    params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(**params)
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_jobs=-1, random_state=0, **params)
    if name == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_jobs=-1, random_state=0, verbose=-1, **params)
    if name == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(tree_method="hist", device="cuda" if gpu else "cpu", eval_metric="logloss",
                             random_state=0, **params)
    raise ValueError(name)


def fit(name, model, params, X, y, Xv, yv):
    stop = params.get("early_stopping_rounds")
    if name == "lightgbm" and stop and len(yv):
        from lightgbm import early_stopping, log_evaluation
        model.fit(X, y, eval_set=[(Xv, yv)], callbacks=[early_stopping(stop, verbose=False), log_evaluation(0)])
    elif name == "xgboost" and stop and len(yv):
        model.set_params(early_stopping_rounds=stop)
        model.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
    else:
        model.fit(X, y)
    return model


def run(args):
    cfg = load_config(args.model_config)
    mode = args.label_mode or cfg["label_mode"]
    names = args.models.split(",") if args.models else list(cfg["classical_models"])
    check_build(args.output)
    banner(f"Training classical baselines: {', '.join(names)} (labels: {mode})")
    data = NodeData(args.output, args.report, mode, cfg["max_len"])
    require_labels(data)
    log("Building bag-of-codes design matrix")
    X = design_matrix(data)
    train, val = data.rows("train"), data.rows("validation")
    eval_rows = np.concatenate([val, data.rows("test")])
    gpu = (args.device or "cuda").startswith("cuda") and _cuda()
    log(f"{X.shape[0]:,} nodes x {X.shape[1]:,} features ({X.nnz:,} non-zeros); xgboost on {'GPU' if gpu else 'CPU'}")
    for name in names:
        params = cfg["classical_models"][name]
        t0 = time.time()
        probs = np.full((len(eval_rows), len(SITE_KEYS)), np.nan, np.float32)
        info = {}
        for j, site in enumerate(progress(SITE_KEYS, f"{name} sites", unit="site")):
            tr = train[~np.isnan(data.labels[train, j])]
            va = val[~np.isnan(data.labels[val, j])]
            y, yv = data.labels[tr, j], data.labels[va, j]
            if y.sum() == 0:
                log(f"{name}/{site}: no training positives, skipped")
                continue
            model = fit(name, make(name, params, gpu), params, X[tr], y, X[va], yv)
            probs[:, j] = model.predict_proba(X[eval_rows])[:, 1]
            val_auc = auroc(yv, model.predict_proba(X[va])[:, 1]) if 0 < yv.sum() < len(yv) else float("nan")
            info[site] = {"train_n": int(len(y)), "train_cases": int(y.sum()), "val_auroc": val_auc,
                          "best_iteration": getattr(model, "best_iteration_", getattr(model, "best_iteration", None))}
            log(f"{name}/{site}: train {len(y):,} ({int(y.sum()):,} cases); validation AUROC {val_auc:.4f}")
        write_predictions(model_dir(args.output, mode, name), data, eval_rows, probs,
                          {"model": name, "label_mode": mode, "config": params, "sites": info,
                           "minutes": round((time.time() - t0) / 60, 1)})
        log(f"{name} done in {(time.time() - t0) / 60:.1f} min")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
