"""Masked-code pretraining for the pretrained transformers (BEHRT, Med-BERT, ehr_transformer).

Uses only train-split participants' codes before the pretraining cutoff (pretrain_corpus stage); the
internal pretraining-validation participants drive early stopping. Labels are not used.
"""
from pathlib import Path

import json
import time


from .common import banner, log
from .model_data import PretrainData, check_build, embedding_len, load_config, mlm_collate, model_dir
from .model_nets import build_model
from .model_training import fit_mlm, loader, pick_device, seed_everything


def run(args):
    cfg = load_config(args.model_config)
    mcfg = cfg["deep_models"][args.model]
    if not mcfg.get("pretrain"):
        raise SystemExit(f"{args.model} is not a pretrained model in {args.model_config}")
    check_build(args.output)
    banner(f"Pretraining {args.model} (masked codes)")
    seed_everything(cfg["seed"])
    device = pick_device(args.device)
    cohort = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data = PretrainData(args.output, args.report, cohort["pretrain_before"], cfg["pretrain_max_len"])
    pcfg = cfg["pretrain"]
    collate = mlm_collate(pcfg["mask_prob"], data.vocab_size)
    train_ds, val_ds = data.dataset("train"), data.dataset("validation")
    log(f"device {device}; train {len(train_ds):,} / validation {len(val_ds):,} sequences with history; vocabulary {data.vocab_size:,}")
    train = loader(train_ds, pcfg["batch_size"], True, cfg["num_workers"], collate, cfg["seed"])
    val = loader(val_ds, pcfg["batch_size"], False, cfg["num_workers"], collate, cfg["seed"])
    model = build_model(args.model, mcfg, data.vocab_size, embedding_len(cfg))
    log(f"{args.model}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    out = model_dir(args.output, "pretrained", args.model)
    t0 = time.time()
    history = fit_mlm(model, train, val, pcfg, device, True, out / "encoder.pt", out / "history.json")
    best = min(history, key=lambda h: h["val_mlm_loss"])
    (out / "meta.json").write_text(json.dumps({"model": args.model, "config": mcfg, "pretrain": pcfg,
                                               "vocab_size": data.vocab_size, "best_epoch": best["epoch"],
                                               "minutes": round((time.time() - t0) / 60, 1)}, indent=2))
    log(f"best epoch {best['epoch']}: val MLM loss {best['val_mlm_loss']:.3f}, acc {best['val_mlm_acc']:.3f}; saved {out / 'encoder.pt'}")
