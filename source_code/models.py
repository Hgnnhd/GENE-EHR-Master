"""Deep models: shared static context and multi-site head; baselines and the main model.

Families (configs/models.json "deep_models"):
  bag          MLP on multi-hot codes                                     (non-sequential baseline)
  rnn          GRU / LSTM over visits (codes summed per day)              (Choi et al. 2016, Doctor AI)
  retain       reverse-time two-level attention                           (Choi et al. 2016, RETAIN)
  dipole       bidirectional GRU with location attention                  (Ma et al. 2017, Dipole)
  transformer  encoder over codes with a CLS token; embedding variants:
                 basic    code + position                                 (Transformer, no pretraining)
                 behrt    code + age + visit position + visit segment      (Li et al. 2020, BEHRT)
                 medbert  code + serialization position + visit            (Rasmy et al. 2021, Med-BERT; MLM only)
                 time     code + continuous age + time-to-landmark + visit (main model, ehr_transformer)
"""
import math

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .model_data import SITE_KEYS, SPECIAL

N_STATIC = 3
N_SITES = len(SITE_KEYS)


class Head(nn.Module):
    """Pooled representation + static context -> one logit per site."""

    def __init__(self, d, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + N_STATIC, d), nn.GELU(), nn.Dropout(dropout), nn.Linear(d, N_SITES))

    def forward(self, pooled, static):
        return self.net(torch.cat([pooled, static], dim=-1))


class Sinusoid(nn.Module):
    """Continuous scalar -> d via fixed sinusoids and a linear map."""

    def __init__(self, d, n_freq=16, max_period=1e4):
        super().__init__()
        self.register_buffer("freq", torch.exp(-math.log(max_period) * torch.arange(n_freq) / n_freq))
        self.proj = nn.Linear(2 * n_freq, d)

    def forward(self, x):
        z = x[..., None] * self.freq
        return self.proj(torch.cat([z.sin(), z.cos()], dim=-1))


class SeqEmbedding(nn.Module):
    def __init__(self, vocab_size, d, kind, max_len, dropout):
        super().__init__()
        self.kind = kind
        self.code = nn.Embedding(vocab_size, d, padding_idx=SPECIAL["PAD"])
        if kind in ("basic", "medbert"):
            self.position = nn.Embedding(max_len + 1, d)
        if kind in ("behrt", "medbert", "time"):
            self.visit = nn.Embedding(max_len + 1, d)
        if kind == "behrt":
            self.age = nn.Embedding(121, d)
            self.segment = nn.Embedding(2, d)
        if kind == "time":
            self.age = Sinusoid(d)
            self.time = Sinusoid(d)
        if kind not in ("basic", "behrt", "medbert", "time"):
            raise ValueError(f"unknown embedding {kind}")
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, b):
        tok = b["tokens"]
        x = self.code(tok)
        # Visit index: CLS is 0, real visits start at 1.
        visit = torch.where(tok.eq(SPECIAL["CLS"]), 0, b["visits"] + 1).clamp(max=self.visit.num_embeddings - 1) if hasattr(self, "visit") else None
        if self.kind in ("basic", "medbert"):
            pos = torch.arange(tok.shape[1], device=tok.device).clamp(max=self.position.num_embeddings - 1)
            x = x + self.position(pos)[None]
        if self.kind == "behrt":
            x = x + self.age(b["ages"].floor().long().clamp(0, 120)) + self.visit(visit) + self.segment(visit % 2)
        if self.kind == "medbert":
            x = x + self.visit(visit)
        if self.kind == "time":
            x = x + self.visit(visit) + self.age((b["ages"] - 60) / 10) + self.time(torch.log1p(b["days"]))
        return self.drop(self.norm(x))


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, cfg, max_len):
        super().__init__()
        d = cfg["d_model"]
        self.embed = SeqEmbedding(vocab_size, d, cfg["embedding"], max_len, cfg["dropout"])
        layer = nn.TransformerEncoderLayer(d, cfg["heads"], 4 * d, cfg["dropout"], activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg["layers"], enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(d)
        self.head = Head(d, cfg["dropout"])
        self.mlm_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.LayerNorm(d), nn.Linear(d, vocab_size))

    def encode(self, b):
        return self.final_norm(self.encoder(self.embed(b), src_key_padding_mask=~b["mask"]))

    def forward(self, b):
        return self.head(self.encode(b)[:, 0], b["static"])

    def mlm_logits(self, b):
        return self.mlm_head(self.encode(b))


class VisitEncoder(nn.Module):
    """Sum code embeddings per visit (same day) and add a time-to-landmark feature."""

    def __init__(self, vocab_size, d):
        super().__init__()
        self.code = nn.Embedding(vocab_size, d, padding_idx=SPECIAL["PAD"])
        self.time = nn.Linear(1, d)

    def forward(self, b):
        tok, visit, mask, days = b["tokens"][:, 1:], b["visits"][:, 1:], b["mask"][:, 1:], b["days"][:, 1:]
        n, d = tok.shape[0], self.code.embedding_dim
        n_visits = int(visit.max().item()) + 1
        index = (torch.arange(n, device=tok.device)[:, None] * n_visits + visit).reshape(-1)
        emb = self.code(tok) * mask[..., None]
        x = torch.zeros(n * n_visits, d, device=tok.device, dtype=emb.dtype).index_add_(0, index, emb.reshape(-1, d))
        vday = torch.zeros(n * n_visits, device=tok.device).scatter_(0, index, (days * mask).reshape(-1))
        vmask = torch.zeros(n * n_visits, dtype=torch.bool, device=tok.device).scatter_(0, index[mask.reshape(-1)], True)
        x = x.view(n, n_visits, d) + self.time(torch.log1p(vday).view(n, n_visits, 1) / 10)
        vmask = vmask.view(n, n_visits)
        return x * vmask[..., None], vmask, vmask.sum(1)


class RNNModel(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.visits = VisitEncoder(vocab_size, d)
        rnn = nn.GRU if cfg["cell"] == "gru" else nn.LSTM
        self.rnn = rnn(d, d, cfg["layers"], batch_first=True, dropout=cfg["dropout"] if cfg["layers"] > 1 else 0)
        self.head = Head(d, cfg["dropout"])

    def forward(self, b):
        x, _, lengths = self.visits(b)
        out, _ = pad_packed_sequence(self.rnn(pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False))[0], batch_first=True)
        last = out[torch.arange(len(lengths), device=x.device), lengths - 1]
        return self.head(last, b["static"])


def reverse_valid(x, lengths):
    """Reverse each sequence within its valid length (padding stays at the end)."""
    steps = torch.arange(x.shape[1], device=x.device)[None]
    idx = (lengths[:, None] - 1 - steps).clamp(min=0)
    return x.gather(1, idx[..., None].expand_as(x))


class RETAIN(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.visits = VisitEncoder(vocab_size, d)
        self.gru_alpha, self.gru_beta = nn.GRU(d, d, batch_first=True), nn.GRU(d, d, batch_first=True)
        self.w_alpha, self.w_beta = nn.Linear(d, 1), nn.Linear(d, d)
        self.drop = nn.Dropout(cfg["dropout"])
        self.head = Head(d, cfg["dropout"])

    def forward(self, b):
        v, vmask, lengths = self.visits(b)
        rev = self.drop(reverse_valid(v, lengths))
        packed = pack_padded_sequence(rev, lengths.cpu(), batch_first=True, enforce_sorted=False)
        g, _ = pad_packed_sequence(self.gru_alpha(packed)[0], batch_first=True, total_length=v.shape[1])
        h, _ = pad_packed_sequence(self.gru_beta(packed)[0], batch_first=True, total_length=v.shape[1])
        alpha = self.w_alpha(g).squeeze(-1).masked_fill(~vmask, float("-inf")).softmax(1)
        beta = torch.tanh(self.w_beta(h))
        context = (alpha[..., None] * beta * rev).sum(1)
        return self.head(context, b["static"])


class Dipole(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.visits = VisitEncoder(vocab_size, d)
        self.rnn = nn.GRU(d, d, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(2 * d, 1)
        self.combine = nn.Linear(4 * d, d)
        self.head = Head(d, cfg["dropout"])

    def forward(self, b):
        v, vmask, lengths = self.visits(b)
        packed = pack_padded_sequence(v, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.rnn(packed)[0], batch_first=True, total_length=v.shape[1])
        alpha = self.attn(h).squeeze(-1).masked_fill(~vmask, float("-inf")).softmax(1)
        context = (alpha[..., None] * h).sum(1)
        last = h[torch.arange(len(lengths), device=v.device), lengths - 1]
        return self.head(torch.tanh(self.combine(torch.cat([context, last], -1))), b["static"])


class BagMLP(nn.Module):
    def __init__(self, vocab_size, cfg):
        super().__init__()
        layers, width = [], vocab_size
        for h in cfg["hidden"]:
            layers += [nn.Linear(width, h), nn.GELU(), nn.Dropout(cfg["dropout"])]
            width = h
        self.vocab_size, self.net = vocab_size, nn.Sequential(*layers)
        self.head = Head(width, cfg["dropout"])

    def forward(self, b):
        tok = b["tokens"]
        bag = torch.zeros(tok.shape[0], self.vocab_size, device=tok.device).scatter_(1, tok, 1.0)
        bag[:, :len(SPECIAL)] = 0
        return self.head(self.net(bag), b["static"])


def build_model(name, cfg, vocab_size, max_len):
    family = cfg["family"]
    if family == "transformer":
        return TransformerModel(vocab_size, cfg, max_len)
    if family == "rnn":
        return RNNModel(vocab_size, cfg)
    if family == "retain":
        return RETAIN(vocab_size, cfg)
    if family == "dipole":
        return Dipole(vocab_size, cfg)
    if family == "bag":
        return BagMLP(vocab_size, cfg)
    raise ValueError(f"{name}: unknown family {family}")
