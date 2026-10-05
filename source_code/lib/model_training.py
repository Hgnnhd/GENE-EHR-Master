"""Training loops: optimiser, schedule, competing-risk loss and prediction, MLM pretraining."""
import json
import math
import os
from pathlib import Path
import re
import time

import numpy as np
import torch
from torch import nn

from .common import log, progress
from .model_data import SITE_KEYS
from .model_targets import N_CLASSES
from .model_metrics import auroc
from .plots import FINETUNE_PANELS, PRETRAIN_PANELS, plot_history


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name=None):
    """Resolve --device. "cuda:N" restricts this process to GPU N via CUDA_VISIBLE_DEVICES (set before
    CUDA starts), so PyTorch never probes the other GPUs; probing can fail when the driver lists a GPU
    the CUDA runtime cannot use ("device >= 0 && device < num_gpus INTERNAL ASSERT FAILED")."""
    match = re.fullmatch(r"cuda:(\d+)", name or "")
    if match and "CUDA_VISIBLE_DEVICES" not in os.environ and not torch.cuda.is_initialized():
        os.environ["CUDA_VISIBLE_DEVICES"] = match.group(1)
        log(f"CUDA_VISIBLE_DEVICES={match.group(1)} (this process uses only that GPU, as cuda:0)")
        return torch.device("cuda:0")
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loader(dataset, batch_size, shuffle, workers, collate_fn, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                                       collate_fn=collate_fn, pin_memory=torch.cuda.is_available(),
                                       persistent_workers=workers > 0, generator=g)


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def optimizer_and_schedule(model, lr, weight_decay, total_steps, warmup_frac):
    decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim > 1]
    no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=lr)
    warmup = max(1, int(total_steps * warmup_frac))

    def factor(step):
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total_steps - warmup))))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, factor)


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


def masked_log_probs(logits, applicable):
    """log softmax over outcome classes per time bin; classes impossible for the person (sex) are excluded."""
    return logits.float().masked_fill(~applicable[:, None, :], float("-inf")).log_softmax(-1)


def competing_risk_nll(logits, batch):
    """Discrete-time competing-risk negative log-likelihood, mean over nodes with known follow-up.

    Bins survived contribute log P(no event); the event bin contributes log P(any of the
    diagnosed causes) (several sites can be first diagnosed on the same day). Censored nodes
    contribute only the bins they were followed through.
    """
    logp = masked_log_probs(logits, batch["applicable"])
    known = batch["known"]
    if not known.any():
        return logits.sum() * 0
    bins = torch.arange(logp.shape[1], device=logp.device)
    ll = (logp[..., 0] * (bins[None] < batch["survived"][:, None])).sum(1)
    has = (batch["event_bin"] >= 0) & known
    if has.any():
        rows = has.nonzero(as_tuple=True)[0]
        event = logp[rows, batch["event_bin"][rows], 1:].masked_fill(~batch["cause"][rows], float("-inf")).logsumexp(-1)
        ll = ll.index_add(0, rows, event)
    return -ll[known].mean()


def cumulative_incidence(logits, applicable):
    """(batch, bins, causes): probability that each cause is the first event by the end of each bin."""
    p = masked_log_probs(logits, applicable).exp()
    alive = torch.cumprod(p[..., 0], 1)
    alive_before = torch.cat([torch.ones_like(alive[:, :1]), alive[:, :-1]], 1)
    return torch.cumsum(alive_before[..., None] * p[..., 1:], 1)


@torch.no_grad()
def predict(model, data_loader, device, amp, desc="predict"):
    """Cumulative incidence (N, bins, causes) and mean NLL over nodes with known follow-up."""
    model.eval()
    out, total, count = [], 0.0, 0
    for batch in progress(data_loader, desc, unit="batch", leave=False):
        batch = to_device(batch, device)
        with autocast(device, amp):
            logits = model(batch)
        n = int(batch["known"].sum())
        if n:
            total, count = total + competing_risk_nll(logits, batch).item() * n, count + n
        out.append(cumulative_incidence(logits, batch["applicable"]).cpu().numpy())
    cif = np.concatenate(out) if out else np.zeros((0, 1, N_CLASSES - 1), np.float32)
    return cif, (total / count if count else float("nan"))


def macro_auroc(probs, labels):
    values = []
    for j in range(labels.shape[1]):
        keep = ~np.isnan(labels[:, j])
        y = labels[keep, j]
        if y.size and 0 < y.sum() < y.size:
            values.append(auroc(y, probs[keep, j]))
    return float(np.mean(values)) if values else float("nan")


def fit_competing_risk(model, train_loader, val_loader, val_labels, cfg, lr, device, amp, ckpt, out_dir, title):
    """Competing-risk NLL; early stopping on validation NLL. The validation macro AUROC of the
    horizon cumulative incidence against complete-follow-up binary labels is logged for monitoring."""
    model.to(device)
    steps = cfg["epochs"] * len(train_loader)
    opt, sched = optimizer_and_schedule(model, lr, cfg["weight_decay"], steps, cfg["warmup_frac"])
    best, bad, history = float("inf"), 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        bar = progress(train_loader, f"epoch {epoch}/{cfg['epochs']}", unit="batch", leave=False)
        for batch in bar:
            batch = to_device(batch, device)
            with autocast(device, amp):
                loss = competing_risk_nll(model(batch), batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
            if hasattr(bar, "set_postfix") and n % 50 == 0:
                bar.set_postfix(loss=f"{total / n:.4f}")
        cif, val_loss = predict(model, val_loader, device, amp, "validate")
        val_auc = macro_auroc(cif[:, -1, :len(SITE_KEYS)], val_labels)
        history.append({"epoch": epoch, "train_nll": total / max(n, 1), "val_nll": val_loss,
                        "val_macro_auroc_horizon": val_auc, "lr": sched.get_last_lr()[0],
                        "seconds": round(time.time() - t0, 1)})
        improved = val_loss < best
        if improved:
            best, bad = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        log(f"epoch {epoch}: train NLL {total / max(n, 1):.4f} | val NLL {val_loss:.4f} | val macro AUROC (horizon) {val_auc:.4f}"
            + ("  * best" if improved else f"  (no gain {bad}/{cfg['patience']})"))
        save_history(history, out_dir, FINETUNE_PANELS, title, min(history, key=lambda h: h["val_nll"])["epoch"])
        if bad >= cfg["patience"]:
            log("early stopping")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return history


def save_history(history, directory, panels, title, best_epoch):
    """history.json + history.csv + training_curves.png, rewritten after every epoch."""
    directory = Path(directory)
    (directory / "history.json").write_text(json.dumps(history, indent=2))
    keys = list(dict.fromkeys(k for h in history for k in h))
    lines = [",".join(keys)] + [",".join("" if h.get(k) is None else str(h[k]) for k in keys) for h in history]
    (directory / "history.csv").write_text("\n".join(lines) + "\n")
    plot_history(history, panels, directory / "training_curves.png", title, best_epoch)


def fit_mlm(model, train_loader, val_batches, cfg, device, amp, out_dir, title, resume=False):
    """Masked-code pretraining; early stopping on validation MLM loss.

    val_batches: a fixed list of masked validation batches (same masks every epoch, so the
    validation loss is comparable across epochs). Writes encoder.pt (best), last.pt (resume
    state) and the per-epoch history/curves into out_dir.
    """
    model.to(device)
    steps = cfg["epochs"] * len(train_loader)
    opt, sched = optimizer_and_schedule(model, cfg["lr"], cfg["weight_decay"], steps, cfg["warmup_frac"])
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    best, bad, history, first = float("inf"), 0, [], 1
    last = Path(out_dir) / "last.pt"
    if resume and last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        sched.load_state_dict(state["scheduler"])
        best, bad, history, first = state["best"], state["bad"], state["history"], state["epoch"] + 1
        log(f"resumed from {last} after epoch {state['epoch']} (best val loss {best:.4f})")

    def run(batches, train, desc, total_batches=None):
        model.train(train)
        total, n, count, top1, top5 = 0.0, 0, 0, 0, 0
        bar = progress(batches, desc, unit="batch", leave=False, total=total_batches)
        for batch in bar:
            batch = to_device(batch, device)
            with torch.set_grad_enabled(train), autocast(device, amp):
                logits = model.mlm_logits(batch)
                loss = lossf(logits.float().reshape(-1, logits.shape[-1]), batch["mlm_labels"].reshape(-1))
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
            with torch.no_grad():
                keep = batch["mlm_labels"].ne(-100)
                target = batch["mlm_labels"][keep]
                ranked = logits[keep].topk(5, dim=-1).indices
                top1 += (ranked[:, 0] == target).sum().item()
                top5 += (ranked == target[:, None]).any(1).sum().item()
                count += int(keep.sum())
            total, n = total + loss.item(), n + 1
            if hasattr(bar, "set_postfix") and n % 50 == 0:
                bar.set_postfix(loss=f"{total / n:.3f}", acc=f"{top1 / max(count, 1):.3f}")
        return total / max(n, 1), top1 / max(count, 1), top5 / max(count, 1)

    for epoch in range(first, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc, train_top5 = run(train_loader, True, f"epoch {epoch}/{cfg['epochs']}")
        val_loss, val_acc, val_top5 = run(val_batches, False, "validate", len(val_batches))
        improved = val_loss < best
        if improved:
            best, bad = val_loss, 0
            torch.save(model.state_dict(), Path(out_dir) / "encoder.pt")
        else:
            bad += 1
        history.append({"epoch": epoch, "train_mlm_loss": train_loss, "val_mlm_loss": val_loss,
                        "train_mlm_acc": train_acc, "val_mlm_acc": val_acc,
                        "train_mlm_top5": train_top5, "val_mlm_top5": val_top5,
                        "val_perplexity": math.exp(min(val_loss, 50)), "lr": sched.get_last_lr()[0],
                        "best": improved, "seconds": round(time.time() - t0, 1)})
        best_epoch = min(history, key=lambda h: h["val_mlm_loss"])["epoch"]
        save_history(history, out_dir, PRETRAIN_PANELS, title, best_epoch)
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                    "epoch": epoch, "best": best, "bad": bad, "history": history}, last)
        log(f"epoch {epoch}: loss {train_loss:.3f}/{val_loss:.3f} | top-1 {train_acc:.3f}/{val_acc:.3f} | "
            f"top-5 {train_top5:.3f}/{val_top5:.3f} (train/val) | lr {history[-1]['lr']:.2e} | {history[-1]['seconds']:.0f}s"
            + ("  * best" if improved else f"  (no gain {bad}/{cfg['patience']})"))
        if bad >= cfg["patience"]:
            log("early stopping")
            break
    return history
