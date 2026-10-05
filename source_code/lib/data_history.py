"""Data stage · Medical history: first-occurrence and inpatient ICD-10 codes.

Combines UKB first occurrences and inpatient diagnoses (record.csv) into one earliest
reliable date per participant x ICD-10 3-character code (events.parquet). Adds
first_hospital_cancer and the hospital quality flags to participants.
"""
from collections import Counter

import polars as pl

from .common import ID, Context, fmt, log, progress, read_chunks
from .definitions import dates

STAGE, TITLE = "history", "Medical history codes (first occurrences + inpatient)"
NEEDS = [("ukb_fields", "UKB_First_occurrences.csv"), ("hospital_cancer", "record.csv")]
MALIGNANT = pl.col("code").str.contains(r"^C\d{2}$") & (pl.col("code") != "C44") & (pl.col("code") <= "C97")


def run(args):
    ctx = Context(args, STAGE, TITLE, NEEDS)
    p = ctx.participants()

    log("Reading first occurrences")
    parts, bad_values = [], Counter()
    for chunk in read_chunks(ctx.source(ctx.ukb, "UKB_First_occurrences.csv"), len(p), "first occurrences"):
        z = chunk.set_index("Participant ID").stack().rename("raw_date").reset_index()
        z.columns = [ID, "field", "raw_date"]
        z["date"] = dates(z.raw_date)
        z["code"] = z.field.str.extract(r"^Date ([A-Z][0-9]{2}) ", expand=False)
        bad = z.date.isna() | z.code.isna()
        bad_values.update(z.loc[bad, "raw_date"])
        ctx.quarantine(z.loc[bad], "invalid_date_or_code", "first_occurrences")
        parts.append(pl.from_pandas(z.loc[~bad, [ID, "code", "date"]]))
    ctx.qc["first_occurrence_invalid_values"] = dict(bad_values)
    fo = pl.concat(parts)
    del parts

    log("Pairing inpatient code/date arrays; quarantining entire mismatched rows")
    parts, mismatches, undated_malignant = [], set(), set()
    for chunk in read_chunks(ctx.source(ctx.hosp, "record.csv"), len(p), "inpatient diagnoses"):
        chunk = chunk.set_index("Participant ID")
        cs = chunk["Diagnoses - ICD10"].str.split("|", regex=False)
        ds = chunk.filter(regex=r"^Date of first in-patient diagnosis - ICD10 \| Array ")
        ds = ds[sorted(ds.columns, key=lambda x: int(x.rsplit(" ", 1)[1]))]
        ds.columns = range(len(ds.columns))
        mismatch = cs.str.len().fillna(0).ne(ds.notna().sum(axis=1))
        mismatches.update(chunk.index[mismatch])
        z = cs.explode().dropna().rename("raw_code").reset_index().rename(columns={"Participant ID": ID})
        z["slot"] = z.groupby(ID).cumcount()
        z["code"] = z.raw_code.str.extract(r"^([A-Z][0-9]{2})", expand=False)
        dl = ds.stack().rename("raw_date").reset_index()
        dl.columns = [ID, "slot", "raw_date"]
        z = z.merge(dl, on=[ID, "slot"], how="left", validate="one_to_one")
        z["date"] = dates(z.raw_date)
        z["malignant"] = z.code.str.match(r"^C\d{2}$", na=False) & z.code.ne("C44") & z.code.le("C97")
        bad = z[ID].isin(mismatches) | z.date.isna() | z.code.isna()
        ctx.quarantine(z.loc[bad], "array_mismatch_or_invalid_pair", "hospital")
        undated_malignant.update(z.loc[bad & z.malignant, ID])
        parts.append(pl.from_pandas(z.loc[~bad, [ID, "code", "date"]]))
    hr = pl.concat(parts)
    del parts
    p["hospital_array_mismatch"] = p.index.isin(mismatches)
    p["hospital_malignancy_unresolved"] = p.index.isin(undated_malignant)

    log("Removing dates before birth / after death; keeping earliest date per code")
    life = pl.from_pandas(p[["birth", "death"]].reset_index())
    cleaned = []
    for source, frame in progress([("first_occurrences", fo), ("hospital", hr)], "date QC per source", unit="source"):
        frame = frame.join(life, on=ID, how="left")
        # Null-safe separate comparisons: an absent death must not mask pre-birth dates.
        invalid = (pl.col("date") < pl.col("birth")).fill_null(False) | (pl.col("date") > pl.col("death")).fill_null(False)
        bad = frame.filter(invalid)
        ctx.quarantine(bad.to_pandas(), "before_birth_or_after_death", source)
        if source == "hospital":
            badc = bad.filter(MALIGNANT)
            p.loc[p.index.isin(badc[ID].to_list()), "hospital_malignancy_unresolved"] = True
        valid = frame.filter(~invalid).select(ID, "code", "date").group_by(ID, "code").agg(pl.col("date").min())
        cleaned.append(valid.rename({"date": source + "_date"}))
    ev = cleaned[0].join(cleaned[1], on=[ID, "code"], how="full", coalesce=True)
    ev = ev.with_columns(pl.min_horizontal("first_occurrences_date", "hospital_date").alias("date"))
    ev = ev.with_columns(
        pl.when(pl.col("first_occurrences_date").is_not_null() & pl.col("hospital_date").is_not_null()).then(pl.lit("both"))
          .when(pl.col("hospital_date").is_not_null()).then(pl.lit("hospital")).otherwise(pl.lit("first_occurrences")).alias("source"),
        pl.lit("day_as_recorded").alias("date_precision"))
    ev = ev.sort([ID, "date", "code"])
    ev.write_parquet(ctx.out / "events.parquet", compression="zstd")
    hd = cleaned[1].filter(MALIGNANT).group_by(ID).agg(pl.col("hospital_date").min()).to_pandas().set_index(ID)
    p["first_hospital_cancer"] = hd.hospital_date
    ctx.save_participants(p)

    sources = dict(ev.group_by("source").len().iter_rows())
    ctx.qc.update({"hospital_array_mismatch_people": len(mismatches),
                   "hospital_unresolved_malignancy_people": int(p.hospital_malignancy_unresolved.sum()),
                   "events": {"rows": ev.height, "people": ev[ID].n_unique(), "codes": ev["code"].n_unique(), "by_source": sources}})
    ctx.done(f"history codes: {fmt(ev.height)} for {fmt(ev[ID].n_unique())} people, {fmt(ev['code'].n_unique())} distinct ICD-10 codes",
             "by source: " + ", ".join(f"{k} {fmt(v)}" for k, v in sorted(sources.items())),
             f"inpatient array mismatches: {fmt(len(mismatches))} people; unresolved inpatient malignancy: "
             f"{fmt(p.hospital_malignancy_unresolved.sum())} people")
