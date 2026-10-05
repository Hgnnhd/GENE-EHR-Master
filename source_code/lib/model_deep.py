"""Train a deep model on the landmark cohorts (one model per run).

One shared model for both landmarks with one output per cancer site. Pretrained models
(BEHRT, Med-BERT, ehr_transformer) start from the 02_pretrain.py weights. Early stopping uses the
validation split; predictions are written for validation and test nodes.
"""
import time

import numpy as np
import torch

from .common import banner, log
from .model_data import (SITE_KEYS, NodeData, check_build, collate, embedding_len, load_config, model_dir, pretrained_dir,
                         require_labels, run_name, survival_setup, variant_name, write_predictions,
                         year_bin)
from .model_nets import build_model, init_output_bias
from .model_targets import log_prior, single_name, single_site_targets
from .model_training import fit_competing_risk, loader, macro_auroc, pick_device, predict, seed_everything


def run(args):
    cfg = load_config(args.model_config)
    mcfg = cfg["deep_models"][args.model]
    mode = args.label_mode or cfg["label_mode"]
    variant = variant_name(cfg, getattr(args, "pretrain_variant", None))
    if variant != "main" and not mcfg.get("pretrain"):
        raise SystemExit(f"--pretrain-variant {variant} only applies to pretrained models, not {args.model}")
    scratch = getattr(args, "no_pretrain", False)
    if scratch and (not mcfg.get("pretrain") or variant != "main"):
        raise SystemExit("--no-pretrain applies to pretrained models (behrt, medbert, ehr_transformer) without --pretrain-variant")
    pretrained = bool(mcfg.get("pretrain")) and not scratch
    name = run_name(args.model, variant, scratch)
    site = getattr(args, "single_site", None)
    if site:
        if site not in SITE_KEYS:
            raise SystemExit(f"unknown site {site}; choose from {', '.join(SITE_KEYS)}")
        name = single_name(name, site)
    check_build(args.output)
    banner(f"Training {name} (labels: {mode})")
    device = pick_device(args.device)  # before any CUDA call
    seed_everything(cfg["seed"])
    horizon, n_bins = survival_setup(cfg, args.config)
    data = NodeData(args.output, args.report, mode, cfg["max_len"], horizon, n_bins)
    require_labels(data)
    labels = data.labels
    if site:  # ablation: this site against all other first events, same encoder and training
        data.targets = single_site_targets(data.targets, site)
        labels = np.full_like(data.labels, np.nan)
        labels[:, SITE_KEYS.index(site)] = data.labels[:, SITE_KEYS.index(site)]
        log(f"single-site model for {site}: outcomes = no event | {site} | other first cancer | death")
    tcfg = cfg["train"]
    known = data.targets["known"]
    train_rows = data.rows("train")[known[data.rows("train")]]
    val_rows = data.rows("validation")[known[data.rows("validation")]]
    events = data.targets["cause"][train_rows].any(1)
    log(f"device {device}; train {len(train_rows):,} nodes with known follow-up ({int(events.sum()):,} with an event, "
        f"{int((data.event_bin[train_rows] < 0).sum()):,} censored or event-free) / validation {len(val_rows):,}; "
        f"{n_bins} time bins over {horizon} years")
    model = build_model(args.model, mcfg, data.vocab_size, embedding_len(cfg), n_bins)
    init_output_bias(model, log_prior(data.targets, data.event_bin, data.survived, n_bins, train_rows))
    lr = mcfg.get("lr", tcfg["lr"])
    if scratch:
        log("--no-pretrain: encoder starts from random initialisation (pretraining ablation)")
    if pretrained:
        weights = pretrained_dir(args.output, variant, args.model) / "encoder.pt"
        if not weights.exists():
            raise SystemExit(f"{weights} missing: run python source_code/02_pretrain.py --model {args.model} "
                             f"--pretrain-variant {variant}")
        state = {k: v for k, v in torch.load(weights, map_location="cpu").items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert all(k.startswith("head.") for k in missing) and not unexpected, (missing, unexpected)
        lr = mcfg.get("finetune_lr", tcfg["finetune_lr"])
        log(f"loaded pretrained encoder {weights}")
    log(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters; lr {lr}")
    amp = mcfg["family"] in ("transformer", "bag")
    out = model_dir(args.output, mode, name)
    train = loader(data.dataset(train_rows), tcfg["batch_size"], True, cfg["num_workers"], collate, cfg["seed"])
    val = loader(data.dataset(val_rows), tcfg["batch_size"], False, cfg["num_workers"], collate)
    t0 = time.time()
    history = fit_competing_risk(model, train, val, labels[val_rows], tcfg, lr, device, amp, out / "best.pt", out, f"Training {name} (labels: {mode})")
    eval_rows = np.concatenate([data.rows("validation"), data.rows("test")])
    cif, _ = predict(model, loader(data.dataset(eval_rows), tcfg["batch_size"], False, cfg["num_workers"], collate), device, amp)
    sites = len(SITE_KEYS)
    if site:  # only this site's cumulative incidence is a prediction; leave the others empty
        cif[:, :, [k for k in range(sites) if SITE_KEYS[k] != site]] = np.nan
    by_year = {year: cif[:, year_bin(year, horizon, n_bins), :sites] for year in cfg["survival"]["report_years"]}
    test = data.split[eval_rows] == "test"
    test_auc = macro_auroc(np.nan_to_num(cif[test, -1, :sites]), labels[eval_rows][test])
    write_predictions(out, data, eval_rows, cif[:, -1, :sites],
                      {"model": name, "label_mode": mode, "pretrain_variant": variant if pretrained else None, "no_pretrain": scratch,
                       "single_site": site,
                       "output": "competing_risk", "horizon_years": horizon, "n_bins": n_bins, "config": mcfg,
                       "train": tcfg, "history": history, "minutes": round((time.time() - t0) / 60, 1)}, cif=by_year)
    log(f"{name}: test macro AUROC (complete follow-up, horizon) {test_auc:.4f}; predictions -> {out / 'predictions.parquet'}")
