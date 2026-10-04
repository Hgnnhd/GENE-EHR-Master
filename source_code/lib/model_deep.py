"""Train a deep model on the landmark cohorts (one model per run).

One shared model for both landmarks with one output per cancer site. Pretrained models
(BEHRT, Med-BERT, ehr_transformer) start from the 02_pretrain.py weights. Early stopping uses the
validation split; predictions are written for validation and test nodes.
"""
import time

import numpy as np
import torch

from .common import banner, log
from .model_data import (NodeData, check_build, collate, embedding_len, load_config, model_dir, require_labels,
                         write_predictions)
from .model_nets import build_model
from .model_training import fit_classifier, loader, macro_auroc, pick_device, predict, seed_everything


def run(args):
    cfg = load_config(args.model_config)
    mcfg = cfg["deep_models"][args.model]
    mode = args.label_mode or cfg["label_mode"]
    check_build(args.output)
    banner(f"Training {args.model} (labels: {mode})")
    seed_everything(cfg["seed"])
    device = pick_device(args.device)
    data = NodeData(args.output, args.report, mode, cfg["max_len"])
    require_labels(data)
    tcfg = cfg["train"]
    labelled = ~np.isnan(data.labels).all(1)
    train_rows = data.rows("train")[labelled[data.rows("train")]]
    val_rows = data.rows("validation")[labelled[data.rows("validation")]]
    log(f"device {device}; train {len(train_rows):,} / validation {len(val_rows):,} labelled nodes")
    model = build_model(args.model, mcfg, data.vocab_size, embedding_len(cfg))
    lr = tcfg["lr"]
    if mcfg.get("pretrain"):
        weights = model_dir(args.output, "pretrained", args.model) / "encoder.pt"
        if not weights.exists():
            raise SystemExit(f"{weights} missing: run python source_code/02_pretrain.py --model {args.model}")
        state = {k: v for k, v in torch.load(weights, map_location="cpu").items() if not k.startswith("head.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert all(k.startswith("head.") for k in missing) and not unexpected, (missing, unexpected)
        lr = tcfg["finetune_lr"]
        log(f"loaded pretrained encoder {weights}")
    log(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters; lr {lr}")
    amp = mcfg["family"] in ("transformer", "bag")
    out = model_dir(args.output, mode, args.model)
    train = loader(data.dataset(train_rows), tcfg["batch_size"], True, cfg["num_workers"], collate, cfg["seed"])
    val = loader(data.dataset(val_rows), tcfg["batch_size"], False, cfg["num_workers"], collate)
    t0 = time.time()
    history = fit_classifier(model, train, val, data.labels[val_rows], tcfg, lr, device, amp, out / "best.pt", out / "history.json")
    eval_rows = np.concatenate([data.rows("validation"), data.rows("test")])
    probs = predict(model, loader(data.dataset(eval_rows), tcfg["batch_size"], False, cfg["num_workers"], collate), device, amp)
    test = data.split[eval_rows] == "test"
    test_auc = macro_auroc(probs[test], data.labels[eval_rows][test])
    write_predictions(out, data, eval_rows, probs, {"model": args.model, "label_mode": mode, "config": mcfg,
                                                    "train": tcfg, "history": history, "minutes": round((time.time() - t0) / 60, 1)})
    log(f"{args.model}: test macro AUROC {test_auc:.4f}; predictions -> {out / 'predictions.parquet'}")
