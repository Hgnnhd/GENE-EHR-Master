"""Step 2 · Masked-code pretraining of the pretrained transformers (no labels used).

Pretrains BEHRT, Med-BERT and the main ehr_transformer on train-split participants' codes
before the variant's cutoff (configs/models.json pretrain_variants):
  main         codes before 2016-01-01 (analysis A, default)
  strict_2011  codes before 2011-01-01 (sensitivity B: no information after the first landmark)
Internal pretraining-validation participants drive early stopping. Weights go to
data/processed/models/pretrained/<variant>/<model>/encoder.pt; 03_train_models.py runs this
automatically when weights are missing.

  python source_code/02_pretrain.py --model ehr_transformer --device cuda:0
  python source_code/02_pretrain.py --model all --device cuda:0      # the three, one after another
  python source_code/02_pretrain.py --model ehr_transformer --pretrain-variant strict_2011
  python source_code/02_pretrain.py --model behrt --device cuda:0 --resume    # continue after an interruption

Every epoch rewrites history.json / history.csv / training_curves.png (loss, top-1 and top-5
accuracy of masked codes, learning rate) next to the weights.
"""
from pathlib import Path

if not __package__:  # allow `python source_code/02_pretrain.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import argparse

from .lib.model_data import load_config, model_parser
from .lib.model_pretrain import run as pretrain


def run(args):
    cfg = load_config(args.model_config)
    pretrained = [k for k, v in cfg["deep_models"].items() if v.get("pretrain")]
    names = pretrained if args.model == "all" else args.model.split(",")
    for name in names:
        if name not in pretrained:
            raise SystemExit(f"{name} is not a pretrained model; choose from {', '.join(pretrained)}")
        pretrain(argparse.Namespace(**{**vars(args), "model": name}))


if __name__ == "__main__":
    ap = model_parser(__doc__)
    ap.add_argument("--model", required=True, help="behrt | medbert | ehr_transformer | all (comma-separated allowed)")
    ap.add_argument("--resume", action="store_true",
                    help="continue from last.pt: an interrupted run, or a finished one after raising epochs/patience")
    run(ap.parse_args())
