"""Data stage · EHR-only model inputs: pre-landmark code sequences per candidate node.

Writes landmark_inputs.parquet, one row per landmark_samples row with equal-length lists
codes / dates / days_before / day_index / sources, sorted by date then code.
"""
import polars as pl

from .common import ID, Context, fmt, log

STAGE, TITLE = "inputs", "EHR-only model input sequences"
LISTS = ["codes", "dates", "days_before", "day_index", "sources"]


def run(args):
    ctx = Context(args, STAGE, TITLE)
    ctx.require("landmark_samples.parquet", "events.parquet")
    us = pl.Datetime("us")
    log("Joining candidate nodes with history codes before each landmark")
    nodes = pl.read_parquet(ctx.out / "landmark_samples.parquet", columns=[ID, "landmark", "split"]).with_columns(pl.col("landmark").cast(us))
    ev = pl.read_parquet(ctx.out / "events.parquet", columns=[ID, "code", "date", "source"]).with_columns(pl.col("date").cast(us))
    x = nodes.join(ev, on=ID, how="inner").filter(pl.col("date") < pl.col("landmark"))
    log("Ordering codes and grouping same-day events")
    x = x.sort([ID, "landmark", "date", "code"]).with_columns(
        (pl.col("landmark") - pl.col("date")).dt.total_days().alias("days_before"),
        (pl.col("date").rank("dense").over([ID, "landmark"]) - 1).cast(pl.Int32).alias("day_index"))
    seq = x.group_by([ID, "landmark"], maintain_order=True).agg(
        pl.col("code").alias("codes"), pl.col("date").alias("dates"), "days_before", "day_index",
        pl.col("source").alias("sources"))
    seq = nodes.join(seq, on=[ID, "landmark"], how="left")
    seq = seq.with_columns([pl.col(c).fill_null(pl.lit([], dtype=seq.schema[c])) for c in LISTS])
    seq = seq.with_columns(pl.col("codes").list.len().alias("n_codes"),
                           (pl.col("day_index").list.max().fill_null(-1) + 1).alias("n_days"))
    log("Writing landmark_inputs.parquet")
    seq.sort([ID, "landmark"]).write_parquet(ctx.out / "landmark_inputs.parquet", compression="zstd")

    stats = seq.group_by("landmark").agg(pl.len().alias("rows"), pl.col("n_codes").median().alias("median"),
                                         pl.col("n_codes").quantile(0.95).alias("p95"),
                                         (pl.col("n_codes") == 0).sum().alias("empty")).sort("landmark")
    ctx.qc["landmark_inputs"] = {"rows": seq.height, "codes": int(seq["n_codes"].sum()),
                                 "empty_history": int((seq["n_codes"] == 0).sum())}
    ctx.done(*[f"{r['landmark'].date()}: {fmt(r['rows'])} sequences; codes median {r['median']:.0f}, "
               f"p95 {r['p95']:.0f}; empty {fmt(r['empty'])}" for r in stats.iter_rows(named=True)])
