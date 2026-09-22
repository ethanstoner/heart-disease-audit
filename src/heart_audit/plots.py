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


def _save(fig, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fill_rule(panels: list[dict], path: Path) -> Path:
    """Disease rate by value, filled vs observed. Each panel: title, values, filled, observed, n_filled, n_observed."""
    fig, axes = plt.subplots(1, len(panels), figsize=(8, 3.6), facecolor=SURFACE, sharey=True)
    for ax, p in zip(np.atleast_1d(axes), panels):
        _style(ax)
        x = np.arange(len(p["values"]))
        w = 0.36
        ax.bar(x - w / 2 - 0.02, p["filled"], w, color=SERIES_2, zorder=2)
        ax.bar(x + w / 2 + 0.02, p["observed"], w, color=SERIES_1, zorder=2)
        for xi, f, o, nf, no in zip(x, p["filled"], p["observed"], p["n_filled"], p["n_observed"]):
            ax.text(xi - w / 2 - 0.02, f + 0.02, f"{f:.0%}\nn={nf}", ha="center", va="bottom", fontsize=8, color=TEXT)
            ax.text(xi + w / 2 + 0.02, o + 0.02, f"{o:.0%}\nn={no}", ha="center", va="bottom", fontsize=8, color=TEXT)
        ax.set_xticks(x, [str(v) for v in p["values"]], fontsize=9, color=TEXT)
        ax.set_ylim(0, 1.3)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(p["title"], loc="left", fontsize=10, color=TEXT)
    np.atleast_1d(axes)[0].set_ylabel("share with heart disease", color=TEXT_2, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES_2), plt.Rectangle((0, 0), 1, 1, color=SERIES_1)]
    fig.legend(handles, ["value filled in by the dataset author", "value recorded in the UCI source"],
               loc="upper right", bbox_to_anchor=(0.98, 0.9), frameon=False, fontsize=9, ncol=2)
    fig.suptitle("The filled-in values were decided by the diagnosis", x=0.02, ha="left",
                 fontsize=12, color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, path)


def effect_summary(rows: list[dict], path: Path) -> Path:
    """Dot + interval per practice: effect on RF accuracy in percentage points.
    Each row: label, median, lo, hi, note (optional)."""
    rows = sorted(rows, key=lambda r: r["median"])
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.4), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.axvline(0, color=MUTED, linewidth=1, zorder=1)
    y = np.arange(len(rows))
    for yi, r in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=SERIES_1, linewidth=2, solid_capstyle="round", zorder=2)
        ax.scatter([r["median"]], [yi], s=60, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(max(r["hi"], r["median"]) + 0.4, yi, f"{r['median']:+.1f}", va="center", fontsize=9, color=TEXT)
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=9, color=TEXT)
    ax.set_xlabel("change in random-forest test accuracy, percentage points (median, central 95% over splits)",
                  color=TEXT_2, fontsize=8.5)
    lo = min(r["lo"] for r in rows)
    hi = max(r["hi"] for r in rows)
    ax.set_xlim(min(lo, 0) - 1, hi + 2.5)
    fig.suptitle("What each practice adds to the published accuracy", x=0.02, ha="left", fontsize=12,
                 color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return _save(fig, path)


def paired_difference(diffs, label: str, title: str, path: Path) -> Path:
    """Histogram of a per-seed paired accuracy difference, one bin per attainable step."""
    d = np.asarray(diffs)
    values = np.unique(np.round(d, 9))          # float noise must not create a tiny step
    step = np.min(np.diff(values)) if len(values) > 1 else 0.01
    edges = np.arange(d.min() - step / 2, d.max() + step, step)
    fig, ax = plt.subplots(figsize=(8, 3.4), facecolor=SURFACE)
    _style(ax)
    ax.hist(d, bins=edges, color=SERIES_1, edgecolor=SURFACE, linewidth=1, zorder=2)
    ax.axvline(0, color=TEXT, linewidth=1.2, linestyle=(0, (4, 3)), zorder=3)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel(label, color=TEXT_2, fontsize=9)
    ax.set_ylabel("split seeds", color=TEXT_2, fontsize=9)
    ahead, behind = (d > 0).mean(), (d < 0).mean()
    ax.text(0.99, 0.95, f"above 0 on {ahead:.0%} of seeds\nbelow 0 on {behind:.0%}\nequal on {1 - ahead - behind:.0%}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9, color=TEXT)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, path)


def site_dumbbell(sites: list[str], left: list[float], right: list[float], left_label: str, right_label: str,
                  title: str, path: Path, notes: list[str] | None = None) -> Path:
    """Per-site AUC under two schemes, joined by a line."""
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(sites) + 1.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(sites))[::-1]
    for yi, a, b in zip(y, left, right):
        ax.plot([a, b], [yi, yi], color=GRID, linewidth=3, zorder=1)
    ax.scatter(left, y, s=60, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3, label=left_label)
    ax.scatter(right, y, s=60, color=SERIES_2, edgecolor=SURFACE, linewidth=2, zorder=3, label=right_label)
    ax.set_ylim(-0.7, len(sites) - 0.3)
    labels = [s if not notes else f"{s}\n{n}" for s, n in zip(sites, notes or [""] * len(sites))]
    ax.set_yticks(y, labels, fontsize=9, color=TEXT)
    ax.set_xlabel("AUC", color=TEXT_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.suptitle(title, x=0.02, ha="left", fontsize=12, color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)


def honest_estimates(rows: list[dict], path: Path) -> Path:
    """Per model/arm: LOSO within-site AUC with its interval, against random-CV pooled AUC.
    Each row: label, loso, lo, hi, cv."""
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rows) + 1.8), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["lo"], r["hi"]], [yi, yi], color=SERIES_1, linewidth=2, solid_capstyle="round", zorder=2)
        ax.plot([r["loso"], r["cv"]], [yi, yi], color=GRID, linewidth=3, zorder=1)
    ax.scatter([r["loso"] for r in rows], y, s=60, color=SERIES_1, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="hospital held out (within-site AUC, 95% CI)")
    ax.scatter([r["cv"] for r in rows], y, s=60, color=SERIES_2, edgecolor=SURFACE, linewidth=2, zorder=3,
               label="random 10-fold x 5 (pooled AUC)")
    ax.set_yticks(y, [r["label"] for r in rows], fontsize=9, color=TEXT)
    ax.set_xlabel("AUC", color=TEXT_2, fontsize=9)
    ax.legend(frameon=False, fontsize=9, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2)
    fig.suptitle("What the models are worth on a hospital they have not seen", x=0.02, ha="left",
                 fontsize=12, color=TEXT, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return _save(fig, path)
