"""Run the numbered pipeline steps in order (01 ... 09).

  python source_code/run_all.py                 # everything
  python source_code/run_all.py --from 05       # rerun step 05 and everything after it
  python source_code/run_all.py --only 03       # a single step

Each step reads the parquet outputs of the earlier steps from --output, so after
changing one step rerun it and every later step (BUILD_COMPLETE.json is only
rewritten by step 08).
"""
from pathlib import Path

if not __package__:  # also allow `python source_code/run_all.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import argparse
import importlib
import time

from .common import log, parser

STEPS = {
    "01": ("01_participants", "Study base population and fixed patient splits"),
    "02": ("02_cancer_registry", "Cancer registry and first malignancy"),
    "03": ("03_medical_history", "Medical history codes (first occurrences + inpatient)"),
    "04": ("04_self_report", "Self-reported cancer history"),
    "05": ("05_landmark_cohort", "Landmark cohorts, follow-up and labels"),
    "06": ("06_model_inputs", "EHR-only model input sequences"),
    "07": ("07_risk_factors", "Pre-landmark risk factors (enhanced inputs)"),
    "08": ("08_pretraining", "Pretraining data, vocabulary and build manifest"),
    "09": ("09_validate", "Validate full-data invariants"),
}


def step_module(key):
    return importlib.import_module(f"{__package__}.{STEPS[key][0]}")


def run(args):
    keys = list(STEPS)
    first, last = getattr(args, "start", None) or keys[0], getattr(args, "stop", None) or keys[-1]
    if getattr(args, "only", None):
        first = last = args.only
    chosen = [k for k in keys if first <= k <= last]
    print("Pipeline plan:\n" + "\n".join(f"  {'>>' if k in chosen else '  '} {k}  {STEPS[k][1]}" for k in keys), flush=True)
    timings = []
    for n, key in enumerate(chosen, 1):
        print(f"\n### [{n}/{len(chosen)}] step {key}", flush=True)
        t = time.time()
        module = step_module(key)
        if key == "09":
            module.run(argparse.Namespace(data=args.output, config=args.config, report=Path(args.report) / "validation.json"))
        else:
            module.run(args)
        timings.append((key, time.time() - t))
    print("\nStep timings:\n" + "\n".join(f"  {k}  {STEPS[k][1]:<55} {s:7.1f}s" for k, s in timings), flush=True)
    log(f"Finished steps {chosen[0]}-{chosen[-1]}")


def main():
    ap = parser(__doc__)
    choices = list(STEPS)
    ap.add_argument("--from", dest="start", choices=choices, help="first step to run (default 01)")
    ap.add_argument("--to", dest="stop", choices=choices, help="last step to run (default 09)")
    ap.add_argument("--only", choices=choices, help="run a single step")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
