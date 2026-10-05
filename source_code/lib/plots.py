"""Training-curve figures (PNG), redrawn after every epoch.

Small multiples on a shared epoch axis, one measure per panel (never two y-scales).
Train and validation keep fixed colours in every panel; the best epoch is marked.
Colours: validated categorical slots 1-2 (blue, orange) on a light chart surface.
"""
from .common import log

SERIES = {"train": "#2a78d6", "validation": "#eb6834"}
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
_warned = []


def _pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        if not _warned:
            log("matplotlib not installed: training curves are saved as history.csv only (pip install matplotlib)")
            _warned.append(True)
        return None


def plot_history(history, panels, path, title, best_epoch=None):
    """history: list of per-epoch dicts. panels: [(panel title, [(key, series)], value format)]
    where series is "train" or "validation"; a panel with one series needs no legend."""
    plt = _pyplot()
    if plt is None or not history:
        return
    epochs = [h["epoch"] for h in history]
    cols = 2 if len(panels) > 2 else len(panels)
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), squeeze=False, facecolor=SURFACE)
    markers = len(epochs) <= 30
    for ax, (name, lines, fmt) in zip(axes.flat, panels):
        ax.set_facecolor(SURFACE)
        ends = []
        for key, series in lines:
            values = [h.get(key) for h in history]
            points = [(e, v) for e, v in zip(epochs, values) if v is not None and v == v]
            if not points:
                continue
            xs, ys = zip(*points)
            ax.plot(xs, ys, color=SERIES[series], linewidth=2, label=series, solid_capstyle="round",
                    marker="o" if markers else None, markersize=4, markeredgecolor=SURFACE, markeredgewidth=1)
            ends.append([xs[-1], ys[-1], f"{series} {format(ys[-1], fmt)}"])
        if len(ends) > 1:  # direct labels at the line ends, in text ink, pushed apart if they would collide
            low, high = ax.get_ylim()
            gap = 0.08 * (high - low)
            ends.sort(key=lambda e: e[1])
            for i in range(1, len(ends)):
                ends[i][1] = max(ends[i][1], ends[i - 1][1] + gap)
            for x, y, text in ends:
                ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", fontsize=8, color=INK_2)
        if best_epoch is not None:
            ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
            ax.annotate(f"best {best_epoch}", (best_epoch, 1), xycoords=("data", "axes fraction"), xytext=(3, -10),
                        textcoords="offset points", fontsize=8, color=MUTED)
        ax.set_title(name, loc="left", fontsize=10, color=INK, pad=8)
        ax.set_xlabel("epoch", fontsize=8, color=MUTED)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.margins(x=0.12 if len(lines) > 1 else 0.04)
        if len(lines) > 1:
            ax.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="best")
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


PRETRAIN_PANELS = [
    ("Masked-code loss (cross-entropy)", [("train_mlm_loss", "train"), ("val_mlm_loss", "validation")], ".3f"),
    ("Top-1 accuracy of masked codes", [("train_mlm_acc", "train"), ("val_mlm_acc", "validation")], ".3f"),
    ("Top-5 accuracy of masked codes", [("train_mlm_top5", "train"), ("val_mlm_top5", "validation")], ".3f"),
    ("Learning rate (end of epoch)", [("lr", "train")], ".1e"),
]
FINETUNE_PANELS = [
    ("Competing-risk NLL", [("train_nll", "train"), ("val_nll", "validation")], ".4f"),
    ("Validation macro AUROC at the horizon", [("val_macro_auroc_horizon", "validation")], ".3f"),
    ("Learning rate (end of epoch)", [("lr", "train")], ".1e"),
]
