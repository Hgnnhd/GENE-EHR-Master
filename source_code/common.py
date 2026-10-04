"""Shared paths, I/O, step context and progress display for the numbered pipeline steps."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # progress bars are optional; plain log lines still show each stage
    tqdm = None

ID = "participant_id"
ROOT = Path(__file__).resolve().parents[1]
CHUNK = 10000
START = time.time()

INPUT_HELP = """Input directories default to configs/cohort.json "data_dirs":
  --ukb-fields       UKB field exports: UKB_visit_and_followup_dates.csv, UKB_death.csv,
                     UKB_First_occurrences.csv, UKB_self_reported_conditions.csv, Base_Information_split/
  --hospital-cancer  record.csv (inpatient ICD-10 + birth), cancer.csv (cancer registry),
                     UKB coding dictionary
"""


def say(message):
    """Print without breaking an active progress bar."""
    if tqdm is None:
        print(message, flush=True)
    else:
        tqdm.write(message, file=sys.stdout)
        sys.stdout.flush()


def log(message):
    elapsed = time.time() - START
    say(f"[{int(elapsed // 60):02d}:{elapsed % 60:04.1f}] {message}")


def progress(iterable, desc, total=None, unit="it", leave=True):
    if tqdm is None:
        return iterable
    # Redirected output (nohup, > file): refresh rarely so the log stays readable.
    interactive = sys.stderr.isatty()
    return tqdm(iterable, desc=f"  {desc}", total=total, unit=unit, leave=leave or not interactive,
                dynamic_ncols=True, mininterval=0.5 if interactive else 60)


def read(path, **kwargs):
    return pd.read_csv(path, dtype=str, encoding="utf-8-sig", **kwargs).rename(columns={"Participant ID": ID})


def read_chunks(path, rows, desc, **kwargs):
    """Yield CSV chunks (raw column names) with a progress bar; rows = expected data rows."""
    reader = pd.read_csv(path, dtype=str, encoding="utf-8-sig", chunksize=CHUNK, **kwargs)
    yield from progress(reader, desc, total=math.ceil(rows / CHUNK) if rows else None, unit="chunk")


def save(frame, path):
    temp = path.with_suffix(".tmp.parquet")
    frame.to_parquet(temp, index=False, compression="zstd")
    temp.replace(path)


def dump(obj, path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def fmt(n):
    return f"{int(n):,}"


def pipeline_hash():
    files = sorted(Path(__file__).parent.glob("*.py"))
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def parser(description):
    ap = argparse.ArgumentParser(description=description + "\n" + INPUT_HELP,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ukb-fields", type=Path, help="UKB field-export directory (default: config data_dirs.ukb_fields)")
    ap.add_argument("--hospital-cancer", type=Path, help="record.csv/cancer.csv directory (default: config data_dirs.hospital_cancer)")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/cohort.json")
    ap.add_argument("--output", type=Path, default=ROOT / "data/processed")
    ap.add_argument("--report", type=Path, default=ROOT / "ccfa-workfiles/checks/cancer-cohort")
    return ap


class Context:
    """State of one pipeline step: paths, config, QC, quarantined records and used sources.

    Steps exchange data only through parquet files in `out`, so any step can be rerun
    once the steps before it have completed.
    """

    def __init__(self, args, step, title, needs=()):
        self.step, self.title = step, title
        self.config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        dirs = self.config.get("data_dirs", {})
        self.ukb = Path(args.ukb_fields or dirs.get("ukb_fields") or "")
        self.hosp = Path(args.hospital_cancer or dirs.get("hospital_cancer") or "")
        self.out, self.report = Path(args.output), Path(args.report)
        self.out.mkdir(parents=True, exist_ok=True)
        self.report.mkdir(parents=True, exist_ok=True)
        roots = {"ukb_fields": self.ukb, "hospital_cancer": self.hosp}
        missing = [str(roots[k] / f) for k, f in needs if not (roots[k] / f).exists()]
        if missing:
            raise FileNotFoundError("Missing input files (check --ukb-fields / --hospital-cancer or "
                                    "configs data_dirs):\n  " + "\n  ".join(missing))
        self.qc, self.quarantines, self.used = {}, [], set()
        self.t0 = time.time()
        (self.out / "BUILD_COMPLETE.json").unlink(missing_ok=True)
        say(f"\n{'=' * 64}\n  Step {step}  {title}\n{'=' * 64}")

    def source(self, root, name):
        path = root / name
        self.used.add(path)
        return path

    def quarantine(self, frame, reason, source):
        if len(frame):
            frame = frame.copy()
            frame["reason"], frame["source"] = reason, source
            self.quarantines.append(frame)

    def require(self, *names):
        missing = [n for n in names if not (self.out / n).exists()]
        if missing:
            raise FileNotFoundError(f"Step {self.step} needs earlier outputs {missing} in {self.out}; "
                                    "run the earlier steps first (python source_code/run_all.py).")

    def participants(self):
        self.require("participants.parquet")
        return pd.read_parquet(self.out / "participants.parquet").set_index(ID)

    def save_participants(self, p):
        save(p.reset_index(), self.out / "participants.parquet")

    def visits(self):
        return read(self.source(self.ukb, "UKB_visit_and_followup_dates.csv")).set_index(ID)

    def done(self, *lines):
        """Write this step's quarantine and QC, then print a short summary."""
        qdir = self.out / "quarantine"
        qdir.mkdir(exist_ok=True)
        qpath = qdir / f"{self.step}.parquet"
        if self.quarantines:
            q = pd.concat(self.quarantines, ignore_index=True)
            # Heterogeneous raw-source fields are retained in the private QC table.
            save(q.astype({c: "string" for c in q.columns if q[c].dtype == object}), qpath)
            self.qc["quarantine"] = {"/".join(k): int(v) for k, v in q.groupby(["source", "reason"]).size().items()}
        else:
            qpath.unlink(missing_ok=True)
        summary_path = self.report / "build_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        summary = {k: v for k, v in summary.items() if k.startswith("step_")}  # drop pre-step-layout keys
        self.qc["elapsed_seconds"] = round(time.time() - self.t0, 1)
        self.qc["sources"] = sorted(str(p) for p in self.used)
        summary[f"step_{self.step}"] = {"title": self.title, **self.qc}
        dump(dict(sorted(summary.items())), summary_path)
        for line in lines:
            say(f"  - {line}")
        log(f"Step {self.step} done in {self.qc['elapsed_seconds']:.0f}s")
