"""Shared data for the model steps (10-13): token sequences, static context, labels and batches.

Every model sees the same candidate nodes, splits, inputs (codes before the landmark,
age at landmark, sex, landmark) and labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch

from .common import ID, ROOT, log
from .definitions import SITES

SPECIAL = {"PAD": 0, "UNK": 1, "MASK": 2, "CLS": 3, "EMPTY": 4}
SITE_KEYS = [s[0] for s in SITES]
LABEL_MODES = ("verified", "provisional_observed")
PROVISIONAL_WARNING = (
    "LABEL MODE provisional_observed: positives are registry-observed first cancers, everyone else counts as 0. "
    "Censored people are treated as negatives, so these results are for pipeline debugging only and must not "
    "be reported (docs/cohort_protocol.md section 5).")


def load_config(path=None):
    return json.loads(Path(path or ROOT / "configs/models.json").read_text(encoding="utf-8"))


def model_parser(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROOT / "data/processed", help="processed data directory (steps 01-08)")
    ap.add_argument("--report", type=Path, default=ROOT / "ccfa-workfiles/checks/cancer-cohort", help="aggregate report directory")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/cohort.json")
    ap.add_argument("--model-config", type=Path, default=ROOT / "configs/models.json")
    ap.add_argument("--label-mode", choices=LABEL_MODES, help="default: configs/models.json label_mode")
    ap.add_argument("--device", help="e.g. cuda, cuda:0, cpu (default: cuda if available)")
    return ap


def check_build(data_dir):
    if not (Path(data_dir) / "BUILD_COMPLETE.json").exists():
        raise SystemExit(f"{data_dir}/BUILD_COMPLETE.json missing: run the data steps first (python source_code/run_all.py)")


def embedding_len(cfg):
    """Position/visit tables sized for the longer of pretraining and fine-tuning sequences."""
    return max(cfg["max_len"], cfg["pretrain_max_len"])


def results_dir(report_dir, label_mode):
    path = Path(report_dir) / "models" / label_mode
    path.mkdir(parents=True, exist_ok=True)
    return path


def model_dir(data_dir, label_mode, name):
    path = Path(data_dir) / "models" / label_mode / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_vocab(report_dir):
    vocab = pd.read_csv(Path(report_dir) / "vocabulary.csv")
    return dict(zip(vocab.code, vocab.token_id.astype(int))), int(vocab.token_id.max()) + 1


def label_matrix(samples, mode):
    """(N, sites) float32 with NaN where the label is unavailable or the site does not apply."""
    if mode not in LABEL_MODES:
        raise ValueError(f"label_mode must be one of {LABEL_MODES}")
    out = np.full((len(samples), len(SITE_KEYS)), np.nan, dtype=np.float32)
    for j, (site, _, _, _, sex) in enumerate(SITES):
        if mode == "verified":
            col = samples[f"label_{site}_5y"]
            out[:, j] = col.astype("Float32").to_numpy(dtype=np.float32, na_value=np.nan)
        else:
            applicable = np.ones(len(samples), bool) if sex is None else samples.sex.eq(sex).to_numpy()
            out[applicable, j] = samples.loc[applicable, f"observed_{site}_5y"].to_numpy(dtype=np.float32)
    return out


def static_features(age, sex, landmark):
    """Shared context for every model: scaled age at landmark, female indicator, landmark offset."""
    years = pd.to_datetime(pd.Series(landmark)).dt.year.to_numpy()
    return np.stack([(np.asarray(age, dtype=np.float32) - 60) / 10,
                     (np.asarray(sex) == "Female").astype(np.float32),
                     ((years - 2011) / 5).astype(np.float32)], axis=1).astype(np.float32)


def encode_sequences(frame, vocab, max_len):
    """frame: polars with list columns codes, days_before, day_index (oldest first).

    Keeps the most recent max_len - 1 codes (one slot is reserved for CLS); an empty history
    becomes a single EMPTY token. Returns lists of int32 tokens, float32 days, int32 visit index.
    """
    keep = max_len - 1
    frame = frame.select(
        pl.col("codes").list.eval(pl.element().replace_strict(vocab, default=SPECIAL["UNK"], return_dtype=pl.Int32)).list.tail(keep).alias("tok"),
        pl.col("days_before").list.tail(keep).alias("days"),
        pl.col("day_index").list.tail(keep).alias("visit"))
    frame = frame.with_columns((pl.col("visit") - pl.col("visit").list.min().fill_null(0)).alias("visit"))
    tokens, days, visits = [], [], []
    for tok, d, v in frame.iter_rows():
        if not tok:
            tok, d, v = [SPECIAL["EMPTY"]], [0], [0]
        tokens.append(np.asarray(tok, dtype=np.int32))
        days.append(np.asarray(d, dtype=np.float32))
        visits.append(np.asarray(v, dtype=np.int32))
    return tokens, days, visits


class NodeData:
    """Candidate landmark nodes with sequences, static context and labels (aligned row order)."""

    def __init__(self, data_dir, report_dir, label_mode, max_len):
        data_dir = Path(data_dir)
        cols = [ID, "landmark", "split", "sex", "age_years_approx"]
        cols += [f"label_{s}_5y" for s in SITE_KEYS] + [f"observed_{s}_5y" for s in SITE_KEYS]
        samples = pd.read_parquet(data_dir / "landmark_samples.parquet", columns=cols)
        samples = samples.sort_values([ID, "landmark"]).reset_index(drop=True)
        inputs = pl.read_parquet(data_dir / "landmark_inputs.parquet", columns=[ID, "landmark", "codes", "days_before", "day_index"])
        inputs = inputs.with_columns(pl.col("landmark").cast(pl.Datetime("us"))).sort([ID, "landmark"])
        keys = samples[[ID, "landmark"]].astype({"landmark": "datetime64[us]"})
        if not (inputs[ID].to_numpy() == keys[ID].to_numpy()).all() or not (inputs["landmark"].to_numpy() == keys.landmark.to_numpy()).all():
            raise ValueError("landmark_inputs and landmark_samples rows do not match; rerun steps 05-06")
        self.vocab, self.vocab_size = load_vocab(report_dir)
        log(f"Encoding {len(samples):,} node sequences (max_len {max_len})")
        self.tokens, self.days, self.visits = encode_sequences(inputs, self.vocab, max_len)
        self.ids = samples[ID].to_numpy()
        self.landmark = samples.landmark.to_numpy()
        self.split = samples.split.to_numpy()
        self.age = samples.age_years_approx.to_numpy(dtype=np.float32)
        self.static = static_features(self.age, samples.sex, samples.landmark)
        self.labels = label_matrix(samples, label_mode)
        self.label_mode = label_mode

    def rows(self, split):
        return np.flatnonzero(self.split == split)

    def dataset(self, rows):
        return SequenceDataset(self, rows)


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, rows):
        self.data, self.rows = data, np.asarray(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, d = self.rows[i], self.data
        return d.tokens[r], d.days[r], d.visits[r], d.age[r], d.static[r], d.labels[r]


def pad_batch(tokens, days, visits, ages):
    """Prepend CLS and right-pad. Token ages = age at reference date - days before it."""
    n = len(tokens)
    length = 1 + max(len(t) for t in tokens)
    tok = np.zeros((n, length), np.int64)
    day = np.zeros((n, length), np.float32)
    vis = np.zeros((n, length), np.int64)
    for i, (t, d, v) in enumerate(zip(tokens, days, visits)):
        tok[i, 0] = SPECIAL["CLS"]
        tok[i, 1:len(t) + 1], day[i, 1:len(t) + 1], vis[i, 1:len(t) + 1] = t, d, v
    tok, day, vis = torch.from_numpy(tok), torch.from_numpy(day), torch.from_numpy(vis)
    age = torch.as_tensor(np.asarray(ages, np.float32))[:, None] - day / 365.25
    return {"tokens": tok, "days": day, "visits": vis, "ages": age, "mask": tok.ne(SPECIAL["PAD"])}


def collate(batch):
    tokens, days, visits, ages, static, labels = zip(*batch)
    out = pad_batch(tokens, days, visits, ages)
    out["static"] = torch.as_tensor(np.stack(static))
    out["labels"] = torch.as_tensor(np.stack(labels))
    return out


class PretrainData:
    """Train-split participants' histories before the pretraining cutoff (step 08 output)."""

    def __init__(self, data_dir, report_dir, cutoff, max_len):
        data_dir = Path(data_dir)
        self.vocab, self.vocab_size = load_vocab(report_dir)
        cutoff = pd.Timestamp(cutoff)
        people = pd.read_parquet(data_dir / "participants.parquet", columns=[ID, "birth", "pretrain_role"])
        people = people.loc[people.pretrain_role.isin(["train", "validation"])].sort_values(ID).reset_index(drop=True)
        ev = pl.read_parquet(data_dir / "pretrain_events.parquet", columns=[ID, "code", "date"])
        ev = ev.with_columns(pl.col("date").cast(pl.Datetime("us"))).sort([ID, "date", "code"]).with_columns(
            (pl.lit(cutoff).cast(pl.Datetime("us")) - pl.col("date")).dt.total_days().alias("days_before"),
            (pl.col("date").rank("dense").over(ID) - 1).cast(pl.Int32).alias("day_index"))
        seq = ev.group_by(ID, maintain_order=True).agg(pl.col("code").alias("codes"), "days_before", "day_index")
        seq = pl.DataFrame({ID: people[ID].to_numpy()}).join(seq, on=ID, how="left")
        seq = seq.with_columns([pl.col(c).fill_null(pl.lit([], dtype=seq.schema[c])) for c in ["codes", "days_before", "day_index"]])
        log(f"Encoding {seq.height:,} pretraining sequences (max_len {max_len})")
        self.tokens, self.days, self.visits = encode_sequences(seq, self.vocab, max_len)
        self.role = people.pretrain_role.to_numpy()
        self.age = ((cutoff - people.birth).dt.days / 365.25).to_numpy(dtype=np.float32)

    def dataset(self, role):
        rows = np.flatnonzero((self.role == role) & np.array([t[0] != SPECIAL["EMPTY"] for t in self.tokens]))
        return MaskedDataset(self, rows)


class MaskedDataset(torch.utils.data.Dataset):
    """Unmasked pretraining sequences; masking happens per batch in mlm_collate."""

    def __init__(self, data, rows):
        self.data, self.rows = data, rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r, d = self.rows[i], self.data
        return d.tokens[r], d.days[r], d.visits[r], d.age[r]


def mlm_collate(mask_prob, vocab_size, generator=None):
    """BERT-style masking of code tokens: 80% MASK, 10% random code, 10% unchanged."""
    def fn(batch):
        tokens, days, visits, ages = zip(*batch)
        out = pad_batch(tokens, days, visits, ages)
        tok = out["tokens"]
        candidates = tok.ge(len(SPECIAL))
        pick = (torch.rand(tok.shape, generator=generator) < mask_prob) & candidates
        # Guarantee at least one masked code per sequence.
        none = ~pick.any(1)
        if none.any():
            first = candidates.float().argmax(1)
            pick[none, first[none]] = candidates[none, first[none]]
        labels = torch.where(pick, tok, torch.full_like(tok, -100))
        roll = torch.rand(tok.shape, generator=generator)
        tok = torch.where(pick & (roll < 0.8), torch.full_like(tok, SPECIAL["MASK"]), tok)
        random_tok = torch.randint(len(SPECIAL), vocab_size, tok.shape, generator=generator)
        tok = torch.where(pick & (roll >= 0.8) & (roll < 0.9), random_tok, tok)
        out["tokens"], out["mlm_labels"] = tok, labels
        return out
    return fn


def write_predictions(directory, data, rows, probs, meta):
    """Patient-level predictions (validation/test rows) plus a small metadata file."""
    frame = pd.DataFrame({ID: data.ids[rows], "landmark": data.landmark[rows], "split": data.split[rows]})
    for j, site in enumerate(SITE_KEYS):
        frame[f"prob_{site}"] = probs[:, j].astype(np.float32)
    frame.to_parquet(Path(directory) / "predictions.parquet", index=False)
    (Path(directory) / "meta.json").write_text(json.dumps(meta, indent=2, default=str))


def require_labels(data):
    available = (~np.isnan(data.labels[data.rows("train")])).sum()
    if data.label_mode == "verified" and available == 0:
        raise SystemExit(
            "No verified 5-year labels: configs/registry_coverage.json is 'unverified', so label_*_5y are empty.\n"
            "Verify registry coverage and rebuild (steps 01-09), or run with --label-mode provisional_observed\n"
            "to debug the training pipeline only (results must not be reported).")
    if data.label_mode == "provisional_observed":
        log("WARNING " + PROVISIONAL_WARNING)
