"""Step 12 · Train a deep model on the landmark cohorts (one model per run).

One shared model for both landmarks with one output per cancer site. Pretrained models
(BEHRT, Med-BERT, ehr_transformer) start from step 10 weights. Early stopping uses the
validation split; predictions are written for validation and test nodes.

Run: python source_code/12_deep_models.py --model gru [--device cuda:0] [--label-mode ...]
"""
from pathlib import Path

if not __package__:  # also allow `python source_code/<step>.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import time

import numpy as np
import torch

from .common import log, say
from .model_data import (NodeData, check_build, collate, embedding_len, load_config, model_dir, model_parser,
                         require_labels, write_predictions)
from .models import build_model
from .training import fit_classifier, loader, macro_auroc, pick_device, predict, seed_everything


def run(args):
    cfg = load_config(args.model_config)
    mcfg = cfg["deep_models"][args.model]
    mode = args.label_mode or cfg["label_mode"]
    check_build(args.output)
    say(f"\n{'=' * 64}\n  Step 12  Training {args.model} (labels: {mode})\n{'=' * 64}")
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
            raise SystemExit(f"{weights} missing: run python source_code/10_pretrain.py --model {args.model}")
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


if __name__ == "__main__":
    ap = model_parser(__doc__)
    ap.add_argument("--model", required=True, help="mlp | gru | lstm | retain | dipole | transformer | behrt | medbert | ehr_transformer")
    run(ap.parse_args())
