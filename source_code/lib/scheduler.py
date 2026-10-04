"""Run pretraining and model training as parallel jobs, one per GPU, with a live status table.

Used by 03_train_models.py. Pretrained transformers (BEHRT, Med-BERT, ehr_transformer) get a
02_pretrain.py job unless weights already exist (or --repretrain); their training job waits
for it. Every other model, classical ones included, is its own 03_train_models.py --model job.
Each job logs to <log-dir>/<job>.log.
"""
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import ROOT, say
from .model_data import load_config, model_dir

SCRIPTS = Path(__file__).resolve().parents[1]


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


def plan(cfg, args, mode, names):
    common = ["--output", str(args.output), "--report", str(args.report), "--config", str(args.config),
              "--model-config", str(args.model_config), "--label-mode", mode, "--no-evaluate"]
    jobs = []
    for name in names:
        after = []
        if cfg["deep_models"].get(name, {}).get("pretrain"):
            exists = (model_dir(args.output, "pretrained", name) / "encoder.pt").exists()
            if args.repretrain or not exists:
                jobs.append(Job(f"pretrain_{name}", "02_pretrain.py", common[:-1] + ["--model", name]))
                after = [f"pretrain_{name}"]
        jobs.append(Job(f"train_{name}", "03_train_models.py", common + ["--model", name], after))
    return jobs


def status(jobs):
    now = time.time()
    lines = [f"{'job':<28}{'gpu':>4}  {'state':<9}{'time':>8}  last log line"]
    for j in jobs:
        elapsed = ((j.t1 or now) - j.t0) if j.t0 else 0
        lines.append(f"{j.name:<28}{'' if j.gpu is None else j.gpu:>4}  {j.state:<9}{elapsed / 60:7.1f}m  {j.last}")
    return "\n".join(lines)


def schedule(args, names):
    """Run jobs for `names` on args.gpus (CPU if none); returns the names whose training finished."""
    cfg = load_config(args.model_config)
    mode = args.label_mode or cfg["label_mode"]
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else ["cpu"]
    log_dir = Path(getattr(args, "log_dir", None) or ROOT / "logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    jobs = plan(cfg, args, mode, names)
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
                j.proc = subprocess.Popen([sys.executable, str(SCRIPTS / j.script)] + j.extra + device,
                                          stdout=handle, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
                j.state, j.t0 = "running", time.time()
                say(f"[{j.name}] started on {'GPU ' + j.gpu if j.gpu != 'cpu' else 'CPU'}")
        for j in jobs:
            j.last = j.tail(log_dir) if j.state != "waiting" else ""
        if time.time() - last_print >= getattr(args, "status_every", 60):
            say("\n" + status(jobs) + "\n")
            last_print = time.time()
        time.sleep(2)
    say("\n" + status(jobs))
    failed = [j.name for j in jobs if j.state != "done"]
    if failed:
        say(f"jobs not completed: {', '.join(failed)} (see {log_dir})")
    return [j.name[len("train_"):] for j in jobs if j.name.startswith("train_") and j.state == "done"], failed
