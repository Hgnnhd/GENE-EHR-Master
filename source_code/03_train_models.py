"""Step 3 · Train the main model and every baseline, then evaluate.

Models (configs/models.json):
  main          ehr_transformer   pretrained transformer: codes + continuous age + time to landmark
  bag-of-codes  logistic_regression, random_forest, lightgbm, xgboost, mlp
  visit RNNs    gru, lstm, retain, dipole
  transformers  transformer (no pretraining), behrt, medbert

  python source_code/03_train_models.py --gpus 0,1,2,3,4,5                # everything, one job per GPU
  python source_code/03_train_models.py --gpus 0,1 --models gru,ehr_transformer
  python source_code/03_train_models.py --model behrt --device cuda:0      # one model in this process
  python source_code/03_train_models.py --gpus 0,1,2,3,4,5 --label-mode provisional_observed  # debugging only
  python source_code/03_train_models.py --gpus 0,1,2 --pretrain-variant strict_2011   # sensitivity B (pretrained models only)
  python source_code/03_train_models.py --gpus 0,1,2,3,4,5 --models ehr_transformer --single-site all   # joint-training ablation
  python source_code/03_train_models.py --gpus 0,1,2 --models ehr_transformer,behrt,medbert --no-pretrain  # pretraining ablation

Pretrained models (behrt, medbert, ehr_transformer) load data/processed/models/pretrained/<variant>/<model>/encoder.pt
and fine-tune with finetune_lr; --no-pretrain trains them from scratch with lr instead.
See run_main_experiment.sh for the whole experiment in order.

With --gpus (or no --model) jobs run in parallel as subprocesses, pretraining first where
weights are missing; logs go to logs/<job>.log and a status table is printed. Afterwards
04_evaluate runs on everything trained (skip with --no-evaluate).
"""
from pathlib import Path

if not __package__:  # allow `python source_code/03_train_models.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import argparse

from .lib.common import ROOT, say
from .lib.model_data import SITE_KEYS, load_config, model_parser, variant_name


def model_names(cfg, text):
    every = list(cfg["deep_models"]) + list(cfg["classical_models"])
    if not text or text == "all":
        return every
    names = []
    for item in text.split(","):
        names += list(cfg["classical_models"]) if item == "classical" else [item]
    unknown = [n for n in names if n not in every]
    if unknown:
        raise SystemExit(f"unknown model(s) {unknown}; choose from {', '.join(every)} or 'classical'")
    return names


def site_names(text):
    """--single-site: None (joint model), 'all', or comma-separated site keys."""
    if not text:
        return None
    sites = SITE_KEYS if text == "all" else text.split(",")
    unknown = [s for s in sites if s not in SITE_KEYS]
    if unknown:
        raise SystemExit(f"unknown site(s) {unknown}; choose from {', '.join(SITE_KEYS)} or 'all'")
    return sites


def run(args):
    cfg = load_config(args.model_config)
    sites = site_names(getattr(args, "single_site", None))
    if sites and getattr(args, "model", None) and args.model in cfg["classical_models"]:
        raise SystemExit("--single-site applies to deep / competing-risk models; classical baselines are already per site")
    if getattr(args, "model", None) and sites and len(sites) > 1:
        names, failed = [], []
        for site in sites:  # several single-site models, one after another in this process
            run(argparse.Namespace(**{**vars(args), "single_site": site, "no_evaluate": True}))
            names.append(site)
    elif getattr(args, "model", None):
        names, failed = [args.model], []
        if args.model in cfg["classical_models"]:
            from .lib.model_classical import run as train_classical
            train_classical(argparse.Namespace(**{**vars(args), "models": args.model}))
        elif args.model in cfg["deep_models"]:
            from .lib.model_deep import run as train_deep
            train_deep(args)
        else:
            raise SystemExit(f"unknown model {args.model}")
    else:
        from .lib.scheduler import schedule
        names = model_names(cfg, getattr(args, "models", None))
        args.sites = sites
        if sites:
            skipped = [n for n in names if n in cfg["classical_models"]]
            names = [n for n in names if n not in skipped]
            if skipped:
                say(f"single-site ablation: classical baselines are already per site; skipping {', '.join(skipped)}")
        if getattr(args, "no_pretrain", False):
            skipped = [n for n in names if not cfg["deep_models"].get(n, {}).get("pretrain")]
            names = [n for n in names if n not in skipped]
            if skipped:
                say(f"--no-pretrain: only models that are normally pretrained apply; skipping {', '.join(skipped)}")
        if variant_name(cfg, getattr(args, "pretrain_variant", None)) != "main":
            skipped = [n for n in names if not cfg["deep_models"].get(n, {}).get("pretrain")]
            names = [n for n in names if n not in skipped]
            if skipped:
                say(f"pretrain variant {args.pretrain_variant}: only pretrained models apply; skipping {', '.join(skipped)}")
        names, failed = schedule(args, names)
    if names and not getattr(args, "no_evaluate", False):
        from .lib.model_evaluate import run as evaluate
        evaluate(argparse.Namespace(**{**vars(args), "split": None, "bootstrap": getattr(args, "bootstrap", None)}))
    if failed:
        raise SystemExit(f"jobs not completed: {', '.join(failed)}")


def main():
    ap = model_parser(__doc__)
    ap.add_argument("--model", help="train one model in this process (deep or classical)")
    ap.add_argument("--models", help="subset for the parallel run: names, 'classical' or 'all' (default all)")
    ap.add_argument("--gpus", help="GPU ids for the parallel run, e.g. 0,1,2,3,4,5 (default: CPU, one job at a time)")
    ap.add_argument("--repretrain", action="store_true", help="rerun 02_pretrain.py even if weights exist")
    ap.add_argument("--no-pretrain", action="store_true",
                    help="pretraining ablation: train behrt / medbert / ehr_transformer from random initialisation")
    ap.add_argument("--single-site", help="ablation of joint training: one model per site ('all' or comma-separated sites)")
    ap.add_argument("--no-evaluate", action="store_true", help="skip 04_evaluate afterwards")
    ap.add_argument("--bootstrap", type=int, help="bootstrap replicates for evaluation")
    ap.add_argument("--status-every", type=int, default=60, help="seconds between status tables")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "logs", help="per-job log files")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
