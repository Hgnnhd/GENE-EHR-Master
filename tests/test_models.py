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
from source_code.lib.model_metrics import aalen_johansen, competing_metrics
from source_code.lib.model_nets import build_model
from source_code.lib.model_targets import CAUSES, N_CLASSES, applicable_classes, bin_targets, competing_targets, log_prior
from source_code.lib.model_training import competing_risk_nll, cumulative_incidence

ID = "participant_id"
ROOT = Path(__file__).resolve().parents[1]
SITE_KEYS = [s[0] for s in SITES]
CODES = [f"K{i:02d}" for i in range(40)] + ["R99"]  # R99 carries the risk signal
DEEP = ["cr_logistic", "mlp", "gru", "lstm", "retain", "dipole", "transformer", "behrt", "medbert", "ehr_transformer"]
HORIZON_DAYS = 1826


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


def outcome(rng, signal, sex, landmark):
    """First event in [t0, t0 + 5y): a target site (sometimes two the same day), another cancer, death,
    censoring, or none. Columns mirror landmark_samples with verified coverage."""
    applicable = [s for s, _, _, _, restrict in SITES if restrict in (None, sex)]
    day = int(rng.integers(1, HORIZON_DAYS))
    at = landmark + pd.Timedelta(days=day)
    row = {"eligible": True, "first_cancer_sites": "", "first_registry_cancer": pd.NaT, "death": pd.NaT, "lost": pd.NaT}
    if rng.random() < (0.5 if signal else 0.06):
        sites = {"other"} if rng.random() < 0.15 else {str(rng.choice(applicable))}
        if rng.random() < 0.05:
            sites.add(str(rng.choice(applicable)))
        row.update(followup_status="cancer", followup_days=day, first_cancer_sites="|".join(sorted(sites)), first_registry_cancer=at)
    elif rng.random() < 0.05:
        sites = set()
        row.update(followup_status="death", followup_days=day, death=at)
    elif rng.random() < 0.2:
        sites = set()
        row.update(followup_status="censored", followup_days=day, lost=at)
    else:
        sites = set()
        row.update(followup_status="event_free_5y", followup_days=HORIZON_DAYS)
    for site, _, _, _, restrict in SITES:
        ok = restrict in (None, sex)
        label = pd.NA if not ok or row["followup_status"] == "censored" else int(site in sites)
        row[f"label_{site}_5y"] = label
        row[f"observed_{site}_5y"] = ok and site in sites
    return row


@pytest.fixture(scope="module")
def processed(tmp_path_factory):
    root = tmp_path_factory.mktemp("models")
    out, report = root / "processed", root / "report"
    out.mkdir(), report.mkdir()
    rng = np.random.default_rng(0)
    n_people = 1500
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
            row.update(outcome(rng, signal, sex[i], landmark))
            rows.append(row)
            seqs.append({ID: pid, "landmark": landmark, "codes": codes, "days_before": days, "day_index": visits})
    samples = pd.DataFrame(rows)
    for site in SITE_KEYS:
        samples[f"label_{site}_5y"] = samples[f"label_{site}_5y"].astype("Int8")
    samples["followup_days"] = samples.followup_days.astype("Int64")
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
    cohort.write_text(json.dumps({"pretrain_before": "2016-01-01", "horizon_years": 5}))
    cfg = json.loads((ROOT / "configs/models.json").read_text())
    cfg.update({"max_len": 16, "pretrain_max_len": 24, "num_workers": 0})
    cfg["pretrain"].update({"epochs": 2, "batch_size": 64, "patience": 2})
    cfg["train"].update({"epochs": 20, "batch_size": 64, "patience": 20, "lr": 3e-3, "finetune_lr": 3e-3})
    for m in cfg["deep_models"].values():
        m.update({k: v for k, v in {"d_model": 32, "layers": 1, "heads": 2, "hidden": [32]}.items() if k in m})
    cfg["classical_models"]["random_forest"]["n_estimators"] = 20
    for m in ["lightgbm", "xgboost"]:
        cfg["classical_models"][m].update({"n_estimators": 100, "learning_rate": 0.2})
    cfg["classical_models"]["xgboost"]["min_child_weight"] = 1  # tiny synthetic data; real-data default stays 10
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
    assert out.shape == (5, 5, len(SITE_KEYS) + 3) and torch.isfinite(out).all()


def test_pipeline_pretrain_train_evaluate(processed):
    step("02_pretrain").run(args(processed, model="ehr_transformer"))
    assert (processed.output / "models/pretrained/main/ehr_transformer/encoder.pt").exists()
    deep = ["ehr_transformer", "cr_logistic", "gru", "retain"]
    classical = ["logistic_regression", "random_forest", "lightgbm", "xgboost"]
    for name in deep + classical:
        step("03_train_models").run(args(processed, model=name, no_evaluate=True))
    pred = pd.read_parquet(processed.output / "models/verified/ehr_transformer/predictions.parquet")
    for site in SITE_KEYS:  # cumulative incidence is non-decreasing over years and ends at prob_<site>
        assert (pred[f"cif_{site}_1y"] <= pred[f"cif_{site}_3y"] + 1e-6).all()
        assert np.allclose(pred[f"cif_{site}_5y"], pred[f"prob_{site}"])
    step("lib.model_evaluate").run(args(processed, split=None, bootstrap=None))
    metrics = pd.read_csv(processed.report / "models/verified/metrics.csv")
    assert set(metrics.model) == set(deep + classical)
    assert set(metrics.landmark) == {"2011-01-01", "2016-01-01"}
    years = metrics.groupby("model").year.unique().map(lambda y: sorted(y))
    assert all(years[m] == [1, 3, 5] for m in deep) and all(years[m] == [5] for m in classical)
    assert (metrics.censored_before_t > 0).any()  # censoring present and handled by IPCW
    # R99 drives risk: every model should beat chance clearly at the horizon.
    best = metrics.loc[metrics.year.eq(5)].groupby("model").auroc.mean()
    assert (best > 0.6).all(), best


def test_unverified_labels_refuse_training(processed, tmp_path):
    samples = pd.read_parquet(processed.output / "landmark_samples.parquet")
    for site in SITE_KEYS:  # what the cohort stage writes while registry coverage is unverified
        samples[f"label_{site}_5y"] = pd.array([pd.NA] * len(samples), dtype="Int8")
    samples["followup_status"], samples["eligible"] = "coverage_unverified", False
    samples["followup_days"] = pd.array([pd.NA] * len(samples), dtype="Int64")
    out = tmp_path / "processed"
    out.mkdir()
    for f in processed.output.glob("*.parquet"):
        (out / f.name).write_bytes(f.read_bytes())
    samples.to_parquet(out / "landmark_samples.parquet", index=False)
    (out / "BUILD_COMPLETE.json").write_text("{}")
    for name in ["gru", "lightgbm"]:
        with pytest.raises(SystemExit, match="No verified outcomes"):
            step("03_train_models").run(args(processed, output=out, model=name, no_evaluate=True))
    y = label_matrix(samples, "provisional_observed")
    assert np.nanmax(y) == 1


def test_run_models_scheduler(processed, tmp_path):
    step("03_train_models").run(args(processed, model=None, gpus=None, models="mlp,lstm", repretrain=False, bootstrap=0,
                                status_every=3600, label_mode="provisional_observed", log_dir=tmp_path))
    assert (tmp_path / "train_mlp.log").exists()
    metrics = pd.read_csv(processed.report / "models/provisional_observed/metrics.csv")
    assert set(metrics.model) == {"mlp", "lstm"}


def test_pretrain_variant_strict_2011(processed):
    """Sensitivity B: pretraining sees only codes before 2011 and fine-tuned outputs are kept apart."""
    step("02_pretrain").run(args(processed, model="ehr_transformer"))
    step("02_pretrain").run(args(processed, model="ehr_transformer", pretrain_variant="strict_2011"))
    main = json.loads((processed.output / "models/pretrained/main/ehr_transformer/meta.json").read_text())
    strict = json.loads((processed.output / "models/pretrained/strict_2011/ehr_transformer/meta.json").read_text())
    assert strict["cutoff"] == "2011-01-01" and 0 < strict["events"] < main["events"]
    corpus = pd.read_parquet(processed.output / "pretrain_events.parquet")
    assert strict["events"] == int((corpus.date < pd.Timestamp("2011-01-01")).sum())
    step("03_train_models").run(args(processed, model="ehr_transformer", pretrain_variant="strict_2011", no_evaluate=True))
    meta = json.loads((processed.output / "models/verified/ehr_transformer__pt_strict_2011/meta.json").read_text())
    assert meta["pretrain_variant"] == "strict_2011"
    with pytest.raises(SystemExit, match="only applies to pretrained models"):
        step("03_train_models").run(args(processed, model="gru", pretrain_variant="strict_2011", no_evaluate=True))
    with pytest.raises(SystemExit, match="unknown pretrain variant"):
        step("02_pretrain").run(args(processed, model="ehr_transformer", pretrain_variant="2020"))


def test_competing_targets(processed):
    samples = pd.read_parquet(processed.output / "landmark_samples.parquet")
    t = competing_targets(samples, "verified", 5)
    status = samples.followup_status.to_numpy()
    assert t["known"].all()
    assert (t["cause"][status == "cancer"].sum(1) >= 1).all() and not t["cause"][np.isin(status, ["censored", "event_free_5y"])].any()
    assert t["cause"][status == "death", CAUSES.index("death")].all()
    multi = samples.first_cancer_sites.str.contains("|", regex=False).to_numpy()
    assert multi.any() and (t["cause"][multi].sum(1) == 2).all() | (samples.first_cancer_sites[multi].str.contains("other")).all()
    assert np.allclose(t["time"][status == "event_free_5y"], 5 * 365.25)
    event_bin, survived = bin_targets(t, 5, 5)
    assert (survived[status == "event_free_5y"] == 5).all() and (event_bin[status == "censored"] == -1).all()
    cancer = status == "cancer"
    assert (event_bin[cancer] == np.minimum(t["time"][cancer] // 365.25, 4)).all() and (survived[cancer] == event_bin[cancer]).all()
    men = samples.sex.eq("Male").to_numpy()
    assert not t["applicable"][men, 1 + SITE_KEYS.index("breast")].any() and t["applicable"][:, 0].all()
    prior = np.exp(log_prior(t, event_bin, survived, 5, np.arange(len(samples))))
    assert np.allclose(prior.sum(1), 1) and (prior[:, 0] > 0.8).all()
    provisional = competing_targets(samples, "provisional_observed", 5)
    assert provisional["known"].all() and (provisional["cause"].any(1) == np.isin(status, ["cancer", "death"])).all()


def test_competing_risk_loss_and_cif():
    torch.manual_seed(0)
    b, k = 4, 5
    logits = torch.randn(b, k, N_CLASSES, requires_grad=True)
    applicable = torch.as_tensor(applicable_classes(np.array(["Male", "Female", "Female", "Male"])))
    cause = torch.zeros(b, len(CAUSES), dtype=torch.bool)
    cause[0, 0] = cause[0, 1] = True        # two sites, same day, in bin 2
    cause[3, CAUSES.index("death")] = True  # death in bin 0
    batch = {"applicable": applicable, "cause": cause, "known": torch.tensor([True, True, True, True]),
             "event_bin": torch.tensor([2, -1, -1, 0]), "survived": torch.tensor([2, 5, 3, 0])}
    loss = competing_risk_nll(logits, batch)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    logp = logits.detach().masked_fill(~applicable[:, None, :], float("-inf")).log_softmax(-1)
    manual = [logp[0, :2, 0].sum() + torch.logsumexp(logp[0, 2, 1:3], 0), logp[1, :, 0].sum(),
              logp[2, :3, 0].sum(), logp[3, 0, N_CLASSES - 1]]
    assert torch.isclose(loss, -torch.stack(manual).mean(), atol=1e-5)
    cif = cumulative_incidence(logits.detach(), applicable)
    alive = torch.cumprod(logp.exp()[..., 0], 1)
    assert torch.allclose(cif.sum(-1) + alive, torch.ones(b, k), atol=1e-5)   # every outcome accounted for
    assert (cif[:, 1:] >= cif[:, :-1] - 1e-7).all()
    assert (cif[0, :, SITE_KEYS.index("breast")] == 0).all()                  # impossible for men


def test_ipcw_metrics_recover_truth_under_censoring():
    rng = np.random.default_rng(0)
    n, h = 20000, 5 * 365.25
    x = rng.normal(size=n)
    tk, td = rng.exponential(h * 8 * np.exp(-0.8 * x)), rng.exponential(h * 10, n)
    T, first = np.minimum(tk, td), np.where(tk < td, 0, 1)
    p = 1 - np.exp(-np.exp(0.8 * x) / 8)
    full = competing_metrics(np.minimum(T, h), (T <= h) & (first == 0), T <= h, p, h, h)
    assert full["auroc"] == pytest.approx(auroc(((T <= h) & (first == 0)).astype(int), p))
    assert full["observed_cif"] == pytest.approx(((T <= h) & (first == 0)).mean())
    C = rng.uniform(0, 1.6 * h, n)
    time, event = np.minimum(np.minimum(T, C), h), (T <= C) & (T <= h)
    cens = competing_metrics(time, event & (first == 0), event, p, h, h)
    assert cens["censored_before_t"] > 0.3 * n
    assert cens["auroc"] == pytest.approx(full["auroc"], abs=0.01)
    assert cens["observed_cif"] == pytest.approx(full["observed_cif"], abs=0.01)
    assert aalen_johansen(time, event & (first == 0), event, h) > (event & (first == 0)).mean() + 0.02  # naive is biased


def test_pretraining_curves_and_resume(processed, tmp_path):
    """Every epoch writes history.csv / training_curves.png; --resume continues from last.pt."""
    import shutil
    base = args(processed, model_config=tmp_path / "models.json")
    cfg = json.loads(processed.model_config.read_text())
    cfg["pretrain"].update({"epochs": 2, "patience": 10})
    base.model_config.write_text(json.dumps(cfg))
    out = processed.output / "models/pretrained/main/medbert"
    shutil.rmtree(out, ignore_errors=True)
    step("02_pretrain").run(args(base, model="medbert"))
    first = json.loads((out / "history.json").read_text())
    assert [h["epoch"] for h in first] == [1, 2]
    assert {"val_mlm_top5", "val_perplexity", "lr"} <= set(first[0]) and (out / "training_curves.png").stat().st_size > 10_000
    assert len((out / "history.csv").read_text().strip().splitlines()) == 3
    cfg["pretrain"]["epochs"] = 4
    base.model_config.write_text(json.dumps(cfg))
    step("02_pretrain").run(args(base, model="medbert", resume=True))
    resumed = json.loads((out / "history.json").read_text())
    assert [h["epoch"] for h in resumed] == [1, 2, 3, 4] and resumed[:2] == first
