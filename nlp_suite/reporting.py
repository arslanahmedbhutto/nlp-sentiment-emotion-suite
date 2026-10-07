"""Shared plotting and report helpers."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3, fixed order
MUTED = "#b9b8b2"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "font.size": 10.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def bar_comparison(scores: dict[str, float], highlight: str, title: str, xlabel: str,
                   path: Path, xlim=None):
    """Horizontal bars, best model highlighted, values labelled directly."""
    items = sorted(scores.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(items) + 1.2))
    colors = [SERIES[0] if n == highlight else MUTED for n in names]
    ax.barh(names, vals, color=colors, height=0.6)
    lo = xlim[0] if xlim else 0
    for i, v in enumerate(vals):
        ax.text(v + (ax.get_xlim()[1] - lo) * 0.004, i, f"{v:.3f}", va="center", color=INK, fontsize=10)
    if xlim:
        ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    _save(fig, path)


def grouped_bars(categories: list[str], series: dict[str, list[float]], title: str,
                 ylabel: str, path: Path):
    """Grouped vertical bars with legend, for up to 3 series."""
    n = len(series)
    width = 0.8 / n
    x = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (name, vals) in enumerate(series.items()):
        pos = x - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width=width - 0.03, color=SERIES[i], label=name)
        for p, v in zip(pos, vals):
            ax.text(p, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5, color=INK_2)
    ax.set_xticks(x, categories)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=30)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", ncols=n, bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    _save(fig, path)


def confusion_matrices(mats: dict[str, np.ndarray], labels: list[str], path: Path):
    """Row-normalised confusion matrices side by side (sequential blue)."""
    k = len(mats)
    fig, axes = plt.subplots(1, k, figsize=(4.6 * k, 4.2), squeeze=False)
    for ax, (name, cm) in zip(axes[0], mats.items()):
        pct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap=BLUES, vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=9,
                        color="white" if pct[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(labels)), labels, rotation=45 if len(labels) > 2 else 0, ha="right" if len(labels) > 2 else "center")
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name, fontsize=11)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    _save(fig, path)


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def markdown_table(rows: list[dict], columns: list[str], bold_row: str | None = None) -> str:
    out = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        cells = []
        for c in columns:
            v = r[c]
            s = f"{v:.3f}" if isinstance(v, float) else str(v)
            if bold_row is not None and r[columns[0]] == bold_row and c == columns[0]:
                s = f"**{s}**"
            cells.append(s)
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)
