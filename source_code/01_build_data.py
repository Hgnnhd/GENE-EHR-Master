"""Step 1 · Build the research data: cohorts, labels, model inputs, pretraining corpus, validation.

Stages (implementation in source_code/lib/data_*.py), in order:
  participants    study base population, birth/sex/death/loss, registry coverage, fixed 70/15/15 split
  registry        cancer registry: pair date and code per Instance, first malignancy and its sites
  history         first-occurrence + inpatient ICD-10 codes -> earliest reliable date per code
  self_report     self-reported cancer history classified with the UKB coding dictionary
  cohort          landmark cohorts: ordered exclusions, follow-up, competing events, labels
  inputs          EHR-only input sequences (codes before each landmark)
  risk_factors    enhanced inputs: latest pre-landmark risk factors
  pretrain_corpus train-split codes before the pretraining cutoff, vocabulary, build manifest
  validate        full-data invariants (chronology, risk sets, labels, leakage)

  python source_code/01_build_data.py                    # all stages
  python source_code/01_build_data.py --from cohort      # rerun cohort and every later stage
  python source_code/01_build_data.py --only validate    # one stage

Stages exchange data through parquet files in --output, so a single stage can be rerun once
the earlier ones exist; after changing one, rerun it and the stages after it.
"""
from pathlib import Path

if not __package__:  # allow `python source_code/01_build_data.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import argparse
import importlib
import time

from .lib.common import STAGES, log, parser, say


def run(args):
    first, last = getattr(args, "start", None) or STAGES[0], getattr(args, "stop", None) or STAGES[-1]
    if getattr(args, "only", None):
        first = last = args.only
    chosen = STAGES[STAGES.index(first):STAGES.index(last) + 1]
    say("Data build plan:\n" + "\n".join(f"  {'>>' if s in chosen else '  '} {i + 1}. {s}" for i, s in enumerate(STAGES)))
    timings = []
    for stage in chosen:
        t = time.time()
        module = importlib.import_module(f"{__package__}.lib.data_{stage}")
        if stage == "validate":
            module.run(argparse.Namespace(data=args.output, config=args.config, report=Path(args.report) / "validation.json"))
        else:
            module.run(args)
        timings.append((stage, time.time() - t))
    say("\nStage timings:\n" + "\n".join(f"  {s:<14} {sec:7.1f}s" for s, sec in timings))
    log(f"Finished stages {chosen[0]} -> {chosen[-1]}")


def main():
    ap = parser(__doc__)
    ap.add_argument("--from", dest="start", choices=STAGES, help="first stage to run (default participants)")
    ap.add_argument("--to", dest="stop", choices=STAGES, help="last stage to run (default validate)")
    ap.add_argument("--only", choices=STAGES, help="run a single stage")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
