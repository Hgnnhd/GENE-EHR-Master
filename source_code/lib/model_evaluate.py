"""Evaluate every trained model on the same nodes and the same competing-risk outcomes.

For each model x landmark x cancer site x year (configs/models.json survival.report_years)
on the evaluation split (default test): time-dependent AUROC with inverse-probability-of-
censoring weights (cases = site is the first malignancy by year t; controls = no event by t
or another first event by t) and a participant-level bootstrap CI, IPCW Brier score, and
calibration-in-the-large against the Aalen-Johansen observed cumulative incidence.
Competing-risk models report every year; per-site binary baselines only the horizon.
Each landmark node is one participant, so per-landmark bootstrap resamples participants.
Writes metrics.csv and auroc_<year>y.csv under <report>/models/<label_mode>/.
"""
import json
from pathlib import Path

import pandas as pd

from .common import ID, banner, log, progress, say
from .definitions import SITES
from .model_data import PROVISIONAL_WARNING, SITE_KEYS, load_config, results_dir, survival_setup
from .model_metrics import competing_metrics
from .model_targets import COLUMNS, competing_targets, horizon_days


def outcome_table(data_dir, mode, horizon):
    samples = pd.read_parquet(Path(data_dir) / "landmark_samples.parquet", columns=[ID] + COLUMNS)
    t = competing_targets(samples, mode, horizon)
    out = pd.DataFrame({ID: samples[ID].to_numpy(), "landmark": samples.landmark.to_numpy(), "time": t["time"],
                        "known": t["known"], "any_event": t["cause"].any(1)})
    for j, site in enumerate(SITE_KEYS):
        out[f"cause_{site}"] = t["cause"][:, j]
        out[f"applies_{site}"] = t["applicable"][:, 1 + j]
    return out


def run(args):
    cfg = load_config(args.model_config)
    mode = args.label_mode or cfg["label_mode"]
    split = args.split or cfg["evaluate"]["split"]
    bootstrap = cfg["evaluate"]["bootstrap"] if args.bootstrap is None else args.bootstrap
    horizon, _ = survival_setup(cfg, args.config)
    years = cfg["survival"]["report_years"]
    banner(f"Evaluate models on {split} (labels: {mode}; years {', '.join(map(str, years))})")
    if mode == "provisional_observed":
        log("WARNING " + PROVISIONAL_WARNING)
    root = Path(args.output) / "models" / mode
    runs = sorted(p.parent for p in root.glob("*/predictions.parquet"))
    if not runs:
        raise SystemExit(f"no predictions under {root}: run 03_train_models.py first")
    outcomes = outcome_table(args.output, mode, horizon)
    names = dict((s[0], s[1]) for s in SITES)
    rows = []
    for run_dir in progress(runs, "models", unit="model"):
        meta = json.loads((run_dir / "meta.json").read_text())
        if meta.get("label_mode") != mode:
            raise SystemExit(f"{run_dir} was trained with label mode {meta.get('label_mode')}, not {mode}")
        pred = pd.read_parquet(run_dir / "predictions.parquet")
        pred = pred.loc[pred.split.eq(split)].merge(outcomes, on=[ID, "landmark"], how="left", validate="one_to_one")
        pred = pred.loc[pred.known]
        for landmark, g in pred.groupby("landmark"):
            for site in SITE_KEYS:
                g_site = g.loc[g[f"applies_{site}"]]
                for year in years:
                    col = f"cif_{site}_{year}y" if f"cif_{site}_{year}y" in g_site else (f"prob_{site}" if year == horizon else None)
                    if col is None or g_site[col].isna().all():
                        continue
                    stats = competing_metrics(g_site.time.to_numpy(), g_site[f"cause_{site}"].to_numpy(),
                                              g_site.any_event.to_numpy(), g_site[col].to_numpy(),
                                              horizon_days(year), horizon_days(horizon), bootstrap, seed=0)
                    rows.append({"model": run_dir.name, "landmark": str(pd.Timestamp(landmark).date()), "site": site,
                                 "name": names[site], "year": year, "output": meta.get("output", "binary"), **stats})
    metrics = pd.DataFrame(rows)
    out = results_dir(args.report, mode)
    metrics.round(5).to_csv(out / "metrics.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        for year in years:
            m = metrics.loc[metrics.year.eq(year)]
            if m.empty:
                continue
            table = m.pivot_table(index=["landmark", "model"], columns="site", values="auroc")
            table = table[[s for s in SITE_KEYS if s in table]]
            table["macro"] = table.mean(axis=1)
            table = table.sort_values(["landmark", "macro"], ascending=[True, False])
            table.round(4).to_csv(out / f"auroc_{year}y.csv")
            for landmark, t in table.groupby(level="landmark"):
                say(f"\nIPCW AUROC at {year} year(s), {split} split, landmark {landmark} (sorted by macro mean):")
                say(t.droplevel("landmark").round(3).to_string())
    stale = out / "auroc_table.csv"
    stale.unlink(missing_ok=True)  # pre-competing-risk layout
    log(f"{len(runs)} models evaluated; metrics -> {out / 'metrics.csv'}")
    if mode == "provisional_observed":
        log("Reminder: provisional labels — do not report these numbers.")
    return metrics
