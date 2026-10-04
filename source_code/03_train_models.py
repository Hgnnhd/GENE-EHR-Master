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

from .lib.common import ROOT
from .lib.model_data import load_config, model_parser


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


def run(args):
    cfg = load_config(args.model_config)
    if getattr(args, "model", None):
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
        names, failed = schedule(args, model_names(cfg, getattr(args, "models", None)))
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
    ap.add_argument("--no-evaluate", action="store_true", help="skip 04_evaluate afterwards")
    ap.add_argument("--bootstrap", type=int, help="bootstrap replicates for evaluation")
    ap.add_argument("--status-every", type=int, default=60, help="seconds between status tables")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "logs", help="per-job log files")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
