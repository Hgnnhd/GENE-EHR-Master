"""Evaluate every trained model on the same nodes and labels.

Per model x landmark x cancer site on the evaluation split (default test): AUROC with
participant-level bootstrap CI, AUPRC, Brier score, calibration-in-the-large and slope.
Each landmark node is one participant, so per-landmark bootstrap resamples participants.
Writes metrics.csv and auroc_table.csv under <report>/models/<label_mode>/.
"""
from pathlib import Path

import json

import pandas as pd

from .common import ID, banner, log, progress, say
from .definitions import SITES
from .model_metrics import summarize
from .model_data import PROVISIONAL_WARNING, SITE_KEYS, label_matrix, load_config, results_dir


def run(args):
    cfg = load_config(args.model_config)
    mode = args.label_mode or cfg["label_mode"]
    split = args.split or cfg["evaluate"]["split"]
    bootstrap = cfg["evaluate"]["bootstrap"] if args.bootstrap is None else args.bootstrap
    banner(f"Evaluate models on {split} (labels: {mode})")
    if mode == "provisional_observed":
        log("WARNING " + PROVISIONAL_WARNING)
    root = Path(args.output) / "models" / mode
    runs = sorted(p.parent for p in root.glob("*/predictions.parquet"))
    if not runs:
        raise SystemExit(f"no predictions under {root}: run 03_train_models.py first")
    cols = [ID, "landmark", "sex"] + [f"label_{s}_5y" for s in SITE_KEYS] + [f"observed_{s}_5y" for s in SITE_KEYS]
    samples = pd.read_parquet(Path(args.output) / "landmark_samples.parquet", columns=cols)
    labels = pd.DataFrame(label_matrix(samples, mode), columns=SITE_KEYS)
    labels[ID], labels["landmark"] = samples[ID].to_numpy(), samples.landmark.to_numpy()
    names = dict((s[0], s[1]) for s in SITES)
    rows = []
    for run_dir in progress(runs, "models", unit="model"):
        meta = json.loads((run_dir / "meta.json").read_text())
        if meta.get("label_mode") != mode:
            raise SystemExit(f"{run_dir} was trained with label mode {meta.get('label_mode')}, not {mode}")
        pred = pd.read_parquet(run_dir / "predictions.parquet")
        pred = pred.loc[pred.split.eq(split)].merge(labels, on=[ID, "landmark"], how="left", validate="one_to_one")
        for landmark, g in pred.groupby("landmark"):
            for site in SITE_KEYS:
                keep = g[site].notna() & g[f"prob_{site}"].notna()
                if not keep.any():
                    continue
                stats = summarize(g.loc[keep, site].to_numpy(), g.loc[keep, f"prob_{site}"].to_numpy(), bootstrap, seed=0)
                rows.append({"model": run_dir.name, "landmark": str(pd.Timestamp(landmark).date()), "site": site,
                             "name": names[site], **stats})
    metrics = pd.DataFrame(rows)
    out = results_dir(args.report, mode)
    metrics.round(5).to_csv(out / "metrics.csv", index=False)
    table = metrics.pivot_table(index=["landmark", "model"], columns="site", values="auroc")[SITE_KEYS]
    table["macro"] = table.mean(axis=1)
    table = table.sort_values(["landmark", "macro"], ascending=[True, False])
    table.round(4).to_csv(out / "auroc_table.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        for landmark, t in table.groupby(level="landmark"):
            say(f"\nAUROC, {split} split, landmark {landmark} (sorted by macro mean):")
            say(t.droplevel("landmark").round(3).to_string())
    log(f"{len(runs)} models evaluated; metrics -> {out / 'metrics.csv'}")
    if mode == "provisional_observed":
        log("Reminder: provisional labels — do not report these numbers.")
