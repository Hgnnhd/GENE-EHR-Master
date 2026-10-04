"""Run pretraining, every baseline and the main model across GPUs, then evaluate (steps 10-13).

  python source_code/run_models.py --gpus 0,1,2,3,4,5
  python source_code/run_models.py --gpus 0,1 --models gru,ehr_transformer
  python source_code/run_models.py --label-mode provisional_observed   # pipeline debugging only

One job per GPU at a time: 10_pretrain for BEHRT / Med-BERT / ehr_transformer, then their
fine-tuning in 12_deep_models; the other deep models start immediately; 11_classical_baselines
runs as one job (XGBoost uses its GPU). Each job logs to logs/<job>.log; this script prints a
status table while they run. Existing pretrained weights are reused unless --repretrain.
"""
from pathlib import Path

if not __package__:  # also allow `python source_code/run_models.py`
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "source_code"

import argparse
import importlib
import os
import subprocess
import sys
import time

from .common import ROOT, log, say
from .model_data import load_config, model_dir, model_parser

HERE = Path(__file__).resolve().parent


class Job:
    def __init__(self, name, script, extra, after=()):
        self.name, self.script, self.extra, self.after = name, script, extra, set(after)
        self.state, self.gpu, self.proc, self.t0, self.t1, self.last = "waiting", None, None, None, None, ""

    def tail(self, log_dir):
        path = log_dir / f"{self.name}.log"
        if not path.exists():
            return ""
        lines = [x for x in path.read_text(errors="replace").replace("\r", "\n").splitlines() if x.strip()]
        return lines[-1].strip()[:90] if lines else ""


def plan(cfg, args, mode):
    deep = cfg["deep_models"]
    chosen = args.models.split(",") if args.models else list(deep) + ["classical"]
    common = ["--output", str(args.output), "--report", str(args.report), "--config", str(args.config),
              "--model-config", str(args.model_config), "--label-mode", mode]
    jobs = []
    for name in chosen:
        if name == "classical" or name in cfg["classical_models"]:
            continue
        mcfg = deep[name]
        after = []
        if mcfg.get("pretrain"):
            exists = (model_dir(args.output, "pretrained", name) / "encoder.pt").exists()
            if args.repretrain or not exists:
                jobs.append(Job(f"pretrain_{name}", "10_pretrain.py", common + ["--model", name]))
                after = [f"pretrain_{name}"]
        jobs.append(Job(f"train_{name}", "12_deep_models.py", common + ["--model", name], after))
    classical = [m for m in chosen if m in cfg["classical_models"]] or (list(cfg["classical_models"]) if "classical" in chosen else [])
    if classical:
        jobs.append(Job("classical", "11_classical_baselines.py", common + ["--models", ",".join(classical)]))
    return jobs


def status(jobs):
    now = time.time()
    lines = [f"{'job':<28}{'gpu':>4}  {'state':<9}{'time':>8}  last log line"]
    for j in jobs:
        elapsed = ((j.t1 or now) - j.t0) if j.t0 else 0
        lines.append(f"{j.name:<28}{'' if j.gpu is None else j.gpu:>4}  {j.state:<9}{elapsed / 60:7.1f}m  {j.last}")
    return "\n".join(lines)


def run(args):
    cfg = load_config(args.model_config)
    mode = args.label_mode or cfg["label_mode"]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else ["cpu"]
    log_dir = Path(getattr(args, "log_dir", None) or ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = plan(cfg, args, mode)
    say("Model plan (label mode %s, GPUs %s):\n" % (mode, ",".join(gpus))
        + "\n".join(f"  {j.name:<28} {j.script}" + (f"  after {', '.join(sorted(j.after))}" if j.after else "") for j in jobs))
    free, last_print = list(gpus), 0
    while any(j.state in ("waiting", "running") for j in jobs):
        for j in jobs:
            if j.state == "running" and j.proc.poll() is not None:
                j.state, j.t1 = ("done" if j.proc.returncode == 0 else "FAILED"), time.time()
                free.append(j.gpu)
                say(f"[{j.name}] {j.state} after {(j.t1 - j.t0) / 60:.1f} min (log: {log_dir / (j.name + '.log')})")
        for j in jobs:
            if j.state == "waiting" and any(d.state in ("FAILED", "skipped") for d in jobs if d.name in j.after):
                j.state = "skipped"
            if j.state == "waiting" and free and all(d.state == "done" for d in jobs if d.name in j.after):
                j.gpu = free.pop(0)
                env = dict(os.environ)
                device = ["--device", "cpu"] if j.gpu == "cpu" else ["--device", "cuda"]
                if j.gpu != "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = j.gpu
                handle = open(log_dir / f"{j.name}.log", "w")
                j.proc = subprocess.Popen([sys.executable, str(HERE / j.script)] + j.extra + device,
                                          stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                j.state, j.t0 = "running", time.time()
                say(f"[{j.name}] started on {'GPU ' + j.gpu if j.gpu != 'cpu' else 'CPU'}")
        for j in jobs:
            j.last = j.tail(log_dir) if j.state != "waiting" else ""
        if time.time() - last_print >= args.status_every:
            say("\n" + status(jobs) + "\n")
            last_print = time.time()
        time.sleep(2)
    say("\n" + status(jobs))
    failed = [j.name for j in jobs if j.state != "done"]
    if any(j.state == "done" for j in jobs if not j.name.startswith("pretrain_")):
        importlib.import_module(f"{__package__}.13_evaluate").run(argparse.Namespace(
            output=args.output, report=args.report, config=args.config, model_config=args.model_config,
            label_mode=mode, device=None, split=None, bootstrap=args.bootstrap))
    if failed:
        raise SystemExit(f"jobs not completed: {', '.join(failed)} (see {log_dir})")


def main():
    ap = model_parser(__doc__)
    ap.add_argument("--gpus", help="comma-separated GPU ids, e.g. 0,1,2,3,4,5 (default: run on CPU, one job at a time)")
    ap.add_argument("--models", help="subset of deep model names and/or classical model names ('classical' = all)")
    ap.add_argument("--repretrain", action="store_true", help="redo step 10 even if pretrained weights exist")
    ap.add_argument("--bootstrap", type=int, help="bootstrap replicates in step 13")
    ap.add_argument("--status-every", type=int, default=60, help="seconds between status tables")
    ap.add_argument("--log-dir", type=Path, default=ROOT / "logs", help="per-job log files")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
