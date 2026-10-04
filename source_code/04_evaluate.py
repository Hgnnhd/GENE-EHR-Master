"""Step 4 · Evaluate every trained model on the same nodes and labels.

Per model x landmark x cancer site on the test split: AUROC with participant-level
bootstrap 95% CI, AUPRC, Brier score, calibration-in-the-large and slope.
Writes metrics.csv and auroc_table.csv under <report>/models/<label_mode>/.

  python source_code/04_evaluate.py [--label-mode ...] [--split validation] [--bootstrap 200]
"""
from pathlib import Path

if not __package__:  # allow `python source_code/04_evaluate.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

from .lib.model_data import model_parser
from .lib.model_evaluate import run

if __name__ == "__main__":
    ap = model_parser(__doc__)
    ap.add_argument("--split", choices=["validation", "test"])
    ap.add_argument("--bootstrap", type=int, help="bootstrap replicates for AUROC CIs (0 = off)")
    run(ap.parse_args())
