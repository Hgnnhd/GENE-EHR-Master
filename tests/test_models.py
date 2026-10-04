"""Model steps 10-13 on small synthetic processed tables (CPU, tiny models)."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from source_code.lib.definitions import SITES
from source_code.lib.model_metrics import auprc, auroc, calibration
from source_code.lib.model_data import SPECIAL, NodeData, PretrainData, collate, label_matrix, mlm_collate
from source_code.lib.model_nets import build_model

ID = "participant_id"
ROOT = Path(__file__).resolve().parents[1]
SITE_KEYS = [s[0] for s in SITES]
CODES = [f"K{i:02d}" for i in range(40)] + ["R99"]  # R99 carries the risk signal
DEEP = ["mlp", "gru", "lstm", "retain", "dipole", "transformer", "behrt", "medbert", "ehr_transformer"]


def step(name):
    return importlib.import_module(f"source_code.{name}")


def history(rng, signal):
    n = int(rng.integers(0, 12))
    codes = list(rng.choice(CODES[:-1], n, replace=False)) + (["R99"] if signal else [])
    days = sorted(rng.integers(30, 4000, len(codes)).tolist(), reverse=True)
    order = np.argsort(-np.array(days)) if days else []
    codes = [codes[i] for i in order]
    days = [days[i] for i in order]
    visits = pd.Series(days).rank(method="dense", ascending=False).astype(int).sub(1).tolist() if days else []
    return codes, days, visits


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    out, report = root / "processed", root / "report"
    out.mkdir(), report.mkdir()
    rng = np.random.default_rng(0)
    n_people = 600
    ids = [f"{i:05d}" for i in range(n_people)]
    split = rng.choice(["train", "validation", "test"], n_people, p=[0.6, 0.2, 0.2])
    sex = rng.choice(["Female", "Male"], n_people)
    birth = pd.to_datetime([f"{y}-01-01" for y in rng.integers(1940, 1965, n_people)])
    people = pd.DataFrame({ID: ids, "split": split, "sex": sex, "birth": birth,
                           "pretrain_role": np.where(split == "train", np.where(rng.random(n_people) < 0.2, "validation", "train"), "excluded")})
    people.to_parquet(out / "participants.parquet", index=False)
    rows, seqs = [], []
    for landmark in [pd.Timestamp("2011-01-01"), pd.Timestamp("2016-01-01")]:
        for i, pid in enumerate(ids):
            signal = rng.random() < 0.3
            codes, days, visits = history(rng, signal)
            row = {ID: pid, "landmark": landmark, "split": split[i], "sex": sex[i],
                   "age_years_approx": (landmark - birth[i]).days / 365.25}
            for site, _, _, _, restrict in SITES:
                applicable = restrict is None or restrict == sex[i]
                y = int(rng.random() < (0.5 if signal else 0.05))
                row[f"label_{site}_5y"] = y if applicable else pd.NA
                row[f"observed_{site}_5y"] = bool(y) and applicable
            rows.append(row)
            seqs.append({ID: pid, "landmark": landmark, "codes": codes, "days_before": days, "day_index": visits})
    samples = pd.DataFrame(rows)
    for site in SITE_KEYS:
        samples[f"label_{site}_5y"] = samples[f"label_{site}_5y"].astype("Int8")
    samples.to_parquet(out / "landmark_samples.parquet", index=False)
    pl.DataFrame(seqs, schema={ID: pl.String, "landmark": pl.Datetime("us"), "codes": pl.List(pl.String),
                               "days_before": pl.List(pl.Int64), "day_index": pl.List(pl.Int32)}).write_parquet(out / "landmark_inputs.parquet")
    ev = [{ID: pid, "code": c, "date": pd.Timestamp("2016-01-01") - pd.Timedelta(days=int(d))}
          for s in seqs[n_people:] for pid, c, d in zip([s[ID]] * len(s["codes"]), s["codes"], s["days_before"])]
    ev = pd.DataFrame(ev).merge(people[[ID, "split", "pretrain_role"]], on=ID)
    ev.loc[ev.split.eq("train")].to_parquet(out / "pretrain_events.parquet", index=False)
    pd.DataFrame({"code": CODES, "len": 1, "token_id": np.arange(len(CODES)) + len(SPECIAL)}).to_csv(report / "vocabulary.csv", index=False)
    (out / "BUILD_COMPLETE.json").write_text("{}")
    cohort = root / "cohort.json"
    cohort.write_text(json.dumps({"pretrain_before": "2016-01-01"}))
    cfg = json.loads((ROOT / "configs/models.json").read_text())
    cfg.update({"max_len": 16, "pretrain_max_len": 24, "num_workers": 0})
    cfg["pretrain"].update({"epochs": 2, "batch_size": 64, "patience": 2})
    cfg["train"].update({"epochs": 20, "batch_size": 64, "patience": 20, "lr": 3e-3, "finetune_lr": 3e-3})
    for m in cfg["deep_models"].values():
        m.update({k: v for k, v in {"d_model": 32, "layers": 1, "heads": 2, "hidden": [32]}.items() if k in m})
    cfg["classical_models"]["random_forest"]["n_estimators"] = 20
    for m in ["lightgbm", "xgboost"]:
        cfg["classical_models"][m]["n_estimators"] = 50
    cfg["evaluate"]["bootstrap"] = 10
    model_cfg = root / "models.json"
    model_cfg.write_text(json.dumps(cfg))
    return argparse.Namespace(output=out, report=report, config=cohort, model_config=model_cfg, device="cpu", label_mode=None)


def args(base, **kw):
    return argparse.Namespace(**{**vars(base), **kw})


def test_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.round(rng.random(500) * 0.5 + y * 0.3, 2)  # ties included
    assert auroc(y, p) == pytest.approx(roc_auc_score(y, p))
    assert auprc(y, p) == pytest.approx(average_precision_score(y, p))
    q = 1 / (1 + np.exp(-(rng.normal(size=20000))))
    yy = (rng.random(20000) < q).astype(int)
    citl, slope = calibration(yy, q)
    assert abs(citl) < 0.02 and slope == pytest.approx(1.0, abs=0.1)


def test_labels_and_masking(processed):
    data = NodeData(processed.output, processed.report, "verified", 16)
    j = SITE_KEYS.index("prostate")
    assert np.isnan(data.labels[data.static[:, 1] == 1, j]).all()  # women: prostate not applicable
    assert all(len(t) <= 15 for t in data.tokens)
    b = collate([data.dataset(range(8))[i] for i in range(8)])
    assert (b["tokens"][:, 0] == SPECIAL["CLS"]).all() and b["static"].shape == (8, 3)
    pre = PretrainData(processed.output, processed.report, "2016-01-01", 24)
    m = mlm_collate(0.15, pre.vocab_size, torch.Generator().manual_seed(0))([pre.dataset("train")[i] for i in range(16)])
    picked = m["mlm_labels"].ne(-100)
    assert picked.any(1).all() and (m["mlm_labels"][picked] >= len(SPECIAL)).all()


@pytest.mark.parametrize("name", DEEP)
def test_forward_shapes(processed, name):
    cfg = json.loads(processed.model_config.read_text())
    data = NodeData(processed.output, processed.report, "verified", 16)
    model = build_model(name, cfg["deep_models"][name], data.vocab_size, 24)
    out = model(collate([data.dataset(range(5))[i] for i in range(5)]))
    assert out.shape == (5, len(SITE_KEYS)) and torch.isfinite(out).all()


def test_pipeline_pretrain_train_evaluate(processed):
    step("02_pretrain").run(args(processed, model="ehr_transformer"))
    assert (processed.output / "models/pretrained/ehr_transformer/encoder.pt").exists()
    for name in ["ehr_transformer", "gru", "retain", "logistic_regression", "random_forest", "lightgbm", "xgboost"]:
        step("03_train_models").run(args(processed, model=name, no_evaluate=True))
    step("lib.model_evaluate").run(args(processed, split=None, bootstrap=None))
    metrics = pd.read_csv(processed.report / "models/verified/metrics.csv")
    assert set(metrics.model) == {"ehr_transformer", "gru", "retain", "logistic_regression", "random_forest", "lightgbm", "xgboost"}
    assert set(metrics.landmark) == {"2011-01-01", "2016-01-01"}
    # R99 drives risk: every model should beat chance clearly on the common sites.
    best = metrics.groupby("model").auroc.mean()
    assert (best > 0.6).all(), best


def test_unverified_labels_refuse_training(processed, tmp_path):
    samples = pd.read_parquet(processed.output / "landmark_samples.parquet")
    for site in SITE_KEYS:
        samples[f"label_{site}_5y"] = pd.array([pd.NA] * len(samples), dtype="Int8")
    out = tmp_path / "processed"
    out.mkdir()
    for f in processed.output.glob("*.parquet"):
        (out / f.name).write_bytes(f.read_bytes())
    samples.to_parquet(out / "landmark_samples.parquet", index=False)
    (out / "BUILD_COMPLETE.json").write_text("{}")
    with pytest.raises(SystemExit, match="No verified 5-year labels"):
        step("03_train_models").run(args(processed, output=out, model="gru", no_evaluate=True))
    y = label_matrix(samples, "provisional_observed")
    assert np.nanmax(y) == 1


def test_run_models_scheduler(processed, tmp_path):
    step("03_train_models").run(args(processed, model=None, gpus=None, models="mlp,lstm", repretrain=False, bootstrap=0,
                                status_every=3600, label_mode="provisional_observed", log_dir=tmp_path))
    assert (tmp_path / "train_mlp.log").exists()
    metrics = pd.read_csv(processed.report / "models/provisional_observed/metrics.csv")
    assert set(metrics.model) == {"mlp", "lstm"}
