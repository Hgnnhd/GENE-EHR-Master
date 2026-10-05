"""Training loops shared by steps 10 and 12: optimiser, schedule, early stopping, prediction."""
import json
import math
import os
import re
import time

import numpy as np
import torch
from torch import nn

from .common import log, progress
from .model_data import SITE_KEYS
from .model_metrics import auroc


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


def masked_bce(logits, labels):
    keep = ~torch.isnan(labels)
    if not keep.any():
        return logits.sum() * 0
    return nn.functional.binary_cross_entropy_with_logits(logits[keep].float(), labels[keep])


def autocast(device, enabled):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled and device.type == "cuda")


@torch.no_grad()
def predict(model, data_loader, device, amp, desc="predict"):
    model.eval()
    out = []
    for batch in progress(data_loader, desc, unit="batch", leave=False):
        with autocast(device, amp):
            out.append(torch.sigmoid(model(to_device(batch, device)).float()).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(SITE_KEYS)), np.float32)


def macro_auroc(probs, labels):
    values = []
    for j in range(labels.shape[1]):
        keep = ~np.isnan(labels[:, j])
        y = labels[keep, j]
        if y.size and 0 < y.sum() < y.size:
            values.append(auroc(y, probs[keep, j]))
    return float(np.mean(values)) if values else float("nan")


def fit_classifier(model, train_loader, val_loader, val_labels, cfg, lr, device, amp, ckpt, history_path):
    """Masked multi-site BCE; early stopping on validation macro AUROC (validation loss if undefined)."""
    model.to(device)
    steps = cfg["epochs"] * len(train_loader)
    opt, sched = optimizer_and_schedule(model, lr, cfg["weight_decay"], steps, cfg["warmup_frac"])
    best, bad, history = -float("inf"), 0, []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        bar = progress(train_loader, f"epoch {epoch}/{cfg['epochs']}", unit="batch", leave=False)
        for batch in bar:
            batch = to_device(batch, device)
            with autocast(device, amp):
                loss = masked_bce(model(batch), batch["labels"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
            if hasattr(bar, "set_postfix") and n % 50 == 0:
                bar.set_postfix(loss=f"{total / n:.4f}")
        probs = predict(model, val_loader, device, amp, "validate")
        val_auc = macro_auroc(probs, val_labels)
        with torch.no_grad():
            p = np.clip(probs, 1e-7, 1 - 1e-7)
            keep = ~np.isnan(val_labels)
            val_loss = float(-(val_labels[keep] * np.log(p[keep]) + (1 - val_labels[keep]) * np.log(1 - p[keep])).mean()) if keep.any() else float("nan")
        score = val_auc if not math.isnan(val_auc) else -val_loss
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_loss": val_loss,
                        "val_macro_auroc": val_auc, "seconds": round(time.time() - t0, 1)})
        improved = score > best
        if improved:
            best, bad = score, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        log(f"epoch {epoch}: train loss {total / max(n, 1):.4f} | val loss {val_loss:.4f} | val macro AUROC {val_auc:.4f}"
            + ("  * best" if improved else f"  (no gain {bad}/{cfg['patience']})"))
        history_path.write_text(json.dumps(history, indent=2))
        if bad >= cfg["patience"]:
            log("early stopping")
            break
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return history


def fit_mlm(model, train_loader, val_loader, cfg, device, amp, ckpt, history_path):
    """Masked-code pretraining; early stopping on validation MLM loss."""
    model.to(device)
    steps = cfg["epochs"] * len(train_loader)
    opt, sched = optimizer_and_schedule(model, cfg["lr"], cfg["weight_decay"], steps, cfg["warmup_frac"])
    lossf = nn.CrossEntropyLoss(ignore_index=-100)
    best, bad, history = float("inf"), 0, []

    def run(data_loader, train, desc):
        model.train(train)
        total, n, correct, count = 0.0, 0, 0, 0
        bar = progress(data_loader, desc, unit="batch", leave=False)
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
            keep = batch["mlm_labels"].ne(-100)
            correct += (logits.argmax(-1)[keep] == batch["mlm_labels"][keep]).sum().item()
            count += keep.sum().item()
            total, n = total + loss.item(), n + 1
            if hasattr(bar, "set_postfix") and n % 50 == 0:
                bar.set_postfix(loss=f"{total / n:.3f}")
        return total / max(n, 1), correct / max(count, 1)

    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc = run(train_loader, True, f"epoch {epoch}/{cfg['epochs']}")
        val_loss, val_acc = run(val_loader, False, "validate")
        history.append({"epoch": epoch, "train_mlm_loss": train_loss, "train_mlm_acc": train_acc,
                        "val_mlm_loss": val_loss, "val_mlm_acc": val_acc, "seconds": round(time.time() - t0, 1)})
        improved = val_loss < best
        if improved:
            best, bad = val_loss, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad += 1
        log(f"epoch {epoch}: train MLM loss {train_loss:.3f} acc {train_acc:.3f} | val loss {val_loss:.3f} acc {val_acc:.3f}"
            + ("  * best" if improved else f"  (no gain {bad}/{cfg['patience']})"))
        history_path.write_text(json.dumps(history, indent=2))
        if bad >= cfg["patience"]:
            log("early stopping")
            break
    return history
