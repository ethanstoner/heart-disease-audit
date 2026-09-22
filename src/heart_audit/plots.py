"""Figures. Colours are the validated reference palette (slots 1-2) on its light surface."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8b8a85"
GRID = "#e6e5e1"
BAND = "#f0efec"
SERIES_1 = "#2a78d6"
SERIES_2 = "#eb6834"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=TEXT_2, length=0, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def control_distribution(control_acc, survey_acc, survey_median: float, band: tuple[float, float],
                         n_test: int, path: Path) -> Path:
    """Histogram of the control's accuracy over split seeds, over a strip of published results."""
    control_acc, survey_acc = np.asarray(control_acc), np.asarray(survey_acc)
    fig, (top, strip) = plt.subplots(
        2, 1, figsize=(8, 4.6), sharex=True, gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
        facecolor=SURFACE,
    )
    for ax in (top, strip):
        _style(ax)
        ax.axvspan(*band, color=BAND, zorder=0)

    # Accuracy on a fixed test set takes values k / n_test: one bin per attainable value.
    k = np.round(control_acc * n_test).astype(int)
    edges = (np.arange(k.min(), k.max() + 2) - 0.5) / n_test
    top.hist(control_acc, bins=edges, color=SERIES_1, edgecolor=SURFACE, linewidth=1, zorder=2)
    top.set_ylabel("split seeds", color=TEXT_2, fontsize=9)

    rng = np.random.default_rng(0)
    strip.scatter(survey_acc, rng.uniform(-0.3, 0.3, len(survey_acc)), s=36, color=SERIES_2,
                  edgecolor=SURFACE, linewidth=1.5, zorder=3)
    strip.set_ylim(-0.6, 0.6)
    strip.set_yticks([])
    strip.grid(False)

    for ax in (top, strip):
        ax.axvline(survey_median, color=TEXT, linewidth=1.2, linestyle=(0, (4, 3)), zorder=4)
    ymax = top.get_ylim()[1]
    top.text(survey_median, ymax * 0.97, f" median published result {survey_median:.1%}",
             color=TEXT, fontsize=9, va="top")
    top.text(band[0], ymax * 0.97, f"central 95%: {band[0]:.1%} to {band[1]:.1%} ",
             color=TEXT_2, fontsize=9, va="top", ha="right")
    strip.set_ylabel(f"{len(survey_acc)} published\nnotebooks", color=TEXT_2, fontsize=9,
                     rotation=0, ha="right", va="center")

    strip.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    strip.set_xlabel("test-set accuracy", color=TEXT_2, fontsize=9)
    spread = round((band[1] - band[0]) * 100)
    fig.suptitle(f"Same pipeline, same data: the split alone moves accuracy by {spread} points",
                 x=0.13, ha="left", fontsize=12, color=TEXT, fontweight="bold")
    fig.text(0.13, 0.87, f"Survey's modal pipeline, random forest, {len(control_acc):,} random 80/20 splits "
             "(central 95% shaded).\n"
             f"Dots: the headline accuracies of the {len(survey_acc)} surveyed notebooks.",
             fontsize=9, color=TEXT_2, linespacing=1.4)
    fig.subplots_adjust(left=0.13, right=0.97, top=0.82, bottom=0.12)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def survey_practices(labels: list[str], counts: list[int], n: int, path: Path) -> Path:
    """Horizontal bars: how many of the n surveyed notebooks follow each practice."""
    order = np.argsort(counts)
    labels, counts = [labels[i] for i in order], [counts[i] for i in order]
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(labels) + 1.1), facecolor=SURFACE)
    _style(ax)
    ax.grid(False)
    ax.spines["bottom"].set_visible(False)
    y = np.arange(len(labels))
    ax.barh(y, [n] * len(labels), height=0.5, color=BAND, zorder=1)
    ax.barh(y, counts, height=0.5, color=SERIES_1, zorder=2)
    for yi, c in zip(y, counts):
        ax.text(n + 0.4, yi, f"{c} of {n}", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, labels, fontsize=9, color=TEXT)
    ax.set_xticks([])
    ax.set_xlim(0, n * 1.15)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.suptitle(f"What {n} published notebooks do", x=0.02, ha="left", fontsize=12, color=TEXT,
                 fontweight="bold")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path
