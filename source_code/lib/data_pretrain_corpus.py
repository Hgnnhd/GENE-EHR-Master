"""Data stage · Pretraining data and vocabulary; closes the build.

Keeps only train-split participants' history codes before the pretraining cutoff,
builds the vocabulary from the internal pretraining-train role, then writes
build_manifest.json and BUILD_COMPLETE.json once every earlier output is present.
"""
from pathlib import Path

import json

import numpy as np
import pandas as pd
import polars as pl

from .common import ID, ROOT, Context, dump, fmt, log, pipeline_hash, save

STAGE, TITLE = "pretrain_corpus", "Pretraining data, vocabulary and build manifest"
OUTPUTS = ["participants.parquet", "cancer_events.parquet", "events.parquet", "self_report_cancer.parquet",
           "landmark_status.parquet", "landmark_samples.parquet", "landmark_inputs.parquet", "features_asof.parquet"]
SPECIAL = {"PAD": 0, "UNK": 1, "MASK": 2, "CLS": 3, "EMPTY": 4}


def run(args):
    ctx = Context(args, STAGE, TITLE)
    ctx.require(*OUTPUTS)
    p = ctx.participants()
    log("Selecting train-split history before the pretraining cutoff")
    roles = pl.from_pandas(p[["split", "pretrain_role"]].reset_index())
    ev = pl.read_parquet(ctx.out / "events.parquet").join(roles, on=ID).filter(
        (pl.col("split") == "train") & (pl.col("date") < pd.Timestamp(ctx.config["pretrain_before"])))
    ev.write_parquet(ctx.out / "pretrain_events.parquet", compression="zstd")
    manifest = p.loc[p.split.eq("train"), ["split", "pretrain_role"]].reset_index()
    n = ev.group_by(ID).len().to_pandas().set_index(ID)["len"]
    manifest["history_codes"] = manifest[ID].map(n).fillna(0).astype(int)
    save(manifest, ctx.out / "pretrain_participants.parquet")
    log("Building vocabulary from the internal pretraining-train role")
    vocab = ev.filter(pl.col("pretrain_role") == "train").group_by("code").len().sort("code").to_pandas()
    vocab["token_id"] = np.arange(len(vocab)) + len(SPECIAL)
    vocab.to_csv(ctx.report / "vocabulary.csv", index=False)
    ctx.qc["pretraining"] = {"participants": len(manifest), "events": ev.height, "vocabulary_codes": len(vocab),
                             "internal_roles": manifest.pretrain_role.value_counts().to_dict(), "special_tokens": SPECIAL}

    log("Writing build manifest")
    summary = json.loads((ctx.report / "build_summary.json").read_text(encoding="utf-8")) if (ctx.report / "build_summary.json").exists() else {}
    sources = sorted({Path(s) for step in summary.values() for s in step.get("sources", [])} | ctx.used)
    inventory = [{"file": str(s), "bytes": s.stat().st_size, "mtime_ns": s.stat().st_mtime_ns} for s in sources if s.exists()]
    coverage = json.loads((ROOT / ctx.config["coverage_manifest"]).read_text(encoding="utf-8"))
    code_hash = pipeline_hash()
    dump({"config": ctx.config, "input_dirs": {"ukb_fields": str(ctx.ukb), "hospital_cancer": str(ctx.hosp)},
          "coverage": coverage, "sources": inventory, "pipeline_sha256": code_hash,
          "source_fingerprints": "size and mtime, not cryptographic raw-file hashes"}, ctx.report / "build_manifest.json")
    ctx.done(f"pretraining participants: {fmt(len(manifest))} "
             f"({', '.join(f'{k} {fmt(v)}' for k, v in manifest.pretrain_role.value_counts().items())})",
             f"pretraining events: {fmt(ev.height)}; vocabulary: {fmt(len(vocab))} codes + {len(SPECIAL)} special tokens")
    (ctx.out / "BUILD_COMPLETE.json").write_text(json.dumps({"pipeline_sha256": code_hash, "coverage_status": coverage["status"]}), encoding="utf-8")
    log("Build complete: all research tables written")
