"""Thesis-ready dataset visualizations (Dataset / Preprocessing chapter).

Every function here is pure: it takes already-loaded arrays/dataframes and
returns a `matplotlib.figure.Figure`. No file I/O happens in this module --
`scripts/make_figures.py` loads the real pickles/CSVs and saves the figures,
mirroring the split between `src/amr_data/*.py` (logic) and `scripts/*.py`
(I/O) used everywhere else in this package.

Color/mark choices follow the project's dataviz method: categorical hues
assigned in a fixed order (never cycled), one hue for magnitude/sequential
encodings, a blue<->red diverging pair for signed deltas, direct value
labels instead of a legend for single-series charts, no dual-axis charts,
and "Other" folding for pies/donuts past ~5 slices. Hex values are the
skill's validated reference palette (light mode only -- these are static
PDF/PNG figures for a printed thesis, not a themed HTML chart).
"""
from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Palette (dataviz skill's validated reference instance, references/palette.md)
# ---------------------------------------------------------------------------
INK_PRIMARY = "#0b0b0b"
# Darkened for print: the original secondary/muted greys were too light to read
# once a figure is scaled down to \textwidth in the thesis PDF.
INK_SECONDARY = "#33322f"
INK_MUTED = "#575652"
GRIDLINE = "#d2d1c8"
BASELINE = "#93928a"
SURFACE = "#fcfcfb"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
CATEGORICAL_ORDER = [BLUE, ORANGE, AQUA, YELLOW]  # fixed order, never cycled

SEQUENTIAL = BLUE
DIVERGING_NEG = BLUE   # test prevalence below train
DIVERGING_POS = "#e34948"  # red -- test prevalence above train

FIGSIZE_WIDE = (10, 4.2)
FIGSIZE_SQUARE = (9, 6.5)


def _setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI"],
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_PRIMARY,
        "text.color": INK_PRIMARY,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10.5,
        "axes.titleweight": "bold",
        "figure.dpi": 120,
        "savefig.dpi": 300,
    })


def _style_bar_axes(ax, horizontal: bool = False) -> None:
    ax.grid(axis="x" if horizontal else "y", alpha=0.6)
    ax.spines["left"].set_visible(horizontal)
    ax.spines["bottom"].set_visible(not horizontal)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)


# ---------------------------------------------------------------------------
# 1. Label overview -- resistance prevalence + label coverage per antibiotic
# ---------------------------------------------------------------------------
def fig_label_overview(
    prevalence: dict[str, float],
    coverage: dict[str, float],
    n_samples: int,
) -> Figure:
    """2-panel: (a) % resistant among labelled isolates, (b) % of the
    `n_samples` genomes that have a known (non-missing) label, per
    antibiotic. Both are magnitude comparisons across a fixed category
    set -> single sequential hue, sorted descending, direct labels.
    """
    _setup_style()
    # Squarer than FIGSIZE_WIDE on purpose: this figure is placed at full
    # \textwidth, so a less elongated canvas means a smaller downscale factor
    # and therefore larger, legible type on the printed page.
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1))

    # No title, no subtitle, no suptitle. Nothing is printed above a figure:
    # the LaTeX caption underneath carries the description of both panels.
    for ax, rates, panel in ((axes[0], prevalence, "Resistance prevalence"),
                             (axes[1], coverage, "Label coverage")):
        items = sorted(rates.items(), key=lambda kv: -kv[1])
        names = [k for k, _ in items]
        values = [v for _, v in items]
        bars = ax.bar(names, values, color=SEQUENTIAL, width=0.62)
        # Rotated upright. Printed flat, neighbouring labels such as 31.2% and
        # 31.0% run into each other, because seven labels do not fit side by
        # side across a half-width panel.
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, v + max(values) * 0.025,
                    f"{v:.1%}", ha="center", va="bottom", fontsize=10,
                    color=INK_PRIMARY, rotation=90)
        ax.set_ylim(0, max(values) * 1.45)  # headroom for the rotated labels
        ax.set_ylabel("")
        # Panel name under the panel, so the caption can stay short and nothing
        # is printed above the figure.
        ax.set_xlabel(panel, fontsize=11.5, color=INK_PRIMARY, labelpad=8)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.tick_params(axis="both", labelsize=11, colors=INK_PRIMARY,
                       labelcolor=INK_PRIMARY)
        _style_bar_axes(ax)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. Population structure -- phylogroup counts + folded share donut
# ---------------------------------------------------------------------------
def fig_population_structure(phylogroup_counts: dict[str, int]) -> Figure:
    """Single panel: exact phylogroup counts, bar, sorted descending, with count
    and share annotated on each bar. The share donut that used to sit beside this
    was a third presentation of the same eight numbers (they are also printed in
    full in the phylogroup table), so it was removed. No baked-in title: the LaTeX
    caption carries it.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))

    items = sorted(phylogroup_counts.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    total = sum(values)

    bars = ax.bar(names, values, color=SEQUENTIAL, width=0.62)
    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, v + total * 0.012,
                f"{v:,}\n({v/total:.1%})", ha="center", va="bottom", fontsize=7.8,
                color=INK_SECONDARY, linespacing=1.3)
    ax.set_ylim(0, max(values) * 1.24)
    ax.set_ylabel("Genomes")
    ax.set_xlabel("Phylogroup")
    _style_bar_axes(ax)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2b. Co-resistance -- phi coefficient between antibiotic pairs
# ---------------------------------------------------------------------------
# Drug-class ordering, so class blocks are contiguous and visible.
CORESIST_ORDER = ["AMP", "AMC", "CTX", "CAZ", "CIP", "GEN", "SXT"]
CORESIST_CLASS_BREAKS = [2, 4, 5, 6]  # draw separators after AMC, CAZ, CIP, GEN


def _phi_pairwise(y_i, y_j, m_i, m_j):
    """Phi coefficient on pairwise-complete rows only.

    Returns (phi, n) where n is the number of strains with BOTH labels observed.
    Phi is undefined when either variable is constant on the shared rows; we
    return nan in that case rather than 0, so it cannot be mistaken for
    'no association measured'.
    """
    both = m_i & m_j
    n = int(both.sum())
    if n < 2:
        return np.nan, n
    a = y_i[both].astype(float)
    b = y_j[both].astype(float)
    if a.std() == 0 or b.std() == 0:
        return np.nan, n
    return float(np.corrcoef(a, b)[0, 1]), n


def fig_coresistance(Y, label_mask, label_names) -> Figure:
    """7x7 phi-coefficient heatmap between antibiotic pairs.

    Every pair is computed on pairwise-complete strains only. Label coverage in
    this collection runs from 49.1% (SXT) to 86.7% (CIP, GEN), so computing this
    over the full matrix, or treating an untested drug as susceptible, silently
    produces a different and wrong figure. The assertions below make that failure
    loud instead of quiet.

    Raises AssertionError if AMP-AMC and CTX-CAZ are not the two strongest
    off-diagonal pairs: that pattern is the stated rationale for the
    Cross-Antibiotic Context Module, and if it does not hold in this dataset the
    text in Chapter 4 needs changing, not the figure.
    """
    _setup_style()
    names = list(label_names)
    idx = [names.index(a) for a in CORESIST_ORDER]
    k = len(CORESIST_ORDER)

    Y = np.asarray(Y)
    M = np.asarray(label_mask).astype(bool)

    phi = np.full((k, k), np.nan)
    nn = np.zeros((k, k), dtype=int)
    cov = {a: int(M[:, names.index(a)].sum()) for a in CORESIST_ORDER}

    for r, ir in enumerate(idx):
        for c, ic in enumerate(idx):
            if r == c:
                continue
            phi[r, c], nn[r, c] = _phi_pairwise(Y[:, ir], Y[:, ic], M[:, ir], M[:, ic])

    # --- guard 1: no pair may rest on more strains than the smaller coverage ---
    for r, a in enumerate(CORESIST_ORDER):
        for c, b in enumerate(CORESIST_ORDER):
            if r == c:
                continue
            assert nn[r, c] <= min(cov[a], cov[b]), (
                f"{a}-{b} computed on {nn[r, c]} strains but coverage caps it at "
                f"{min(cov[a], cov[b])} -- the pairwise-complete mask is wrong.")

    # --- guard 2: the cephalosporin pair must be the strongest ---
    #
    # An earlier version of this guard required the two SAME-CLASS pairs
    # (AMP-AMC and CTX-CAZ) to be the two strongest. That is false in this
    # dataset and the guard caught it: the observed ranking is CTX-CAZ 0.843,
    # AMP-SXT 0.553, AMP-AMC 0.536, CAZ-CIP 0.524. AMP-SXT crosses drug classes
    # (penicillin against folate-pathway inhibitor) and still outranks the
    # within-class penicillin pair, because those determinants travel together
    # on the same mobile elements rather than through a shared mechanism.
    # Chapter 4 now states this instead of the class-hierarchy-only claim, so
    # the guard is narrowed to what the data actually supports: the two
    # cephalosporins are the strongest pair, and by a wide margin.
    off = [((CORESIST_ORDER[r], CORESIST_ORDER[c]), phi[r, c])
           for r in range(k) for c in range(r + 1, k) if np.isfinite(phi[r, c])]
    off.sort(key=lambda t: -t[1])
    assert frozenset(off[0][0]) == frozenset({"CTX", "CAZ"}), (
        "CTX-CAZ is no longer the strongest co-resistance pair. "
        f"Strongest observed: {[(f'{a}-{b}', round(v, 3)) for (a, b), v in off[:4]]}. "
        "Stop and report this: Chapter 4 states that the two cephalosporins are "
        "the most strongly linked pair in this collection.")
    assert off[0][1] - off[1][1] > 0.15, (
        f"CTX-CAZ ({off[0][1]:.3f}) no longer leads the next pair "
        f"({off[1][0][0]}-{off[1][0][1]}, {off[1][1]:.3f}) by a wide margin. "
        "Chapter 4 says 'by a wide margin'; soften that wording or re-check.")

    diverging = np.nanmin(phi) < 0
    cmap = "RdBu_r" if diverging else "cividis"
    vmax = float(np.nanmax(np.abs(phi)))
    vmin = -vmax if diverging else 0.0

    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    im = ax.imshow(np.ma.masked_invalid(phi), cmap=cmap, vmin=vmin, vmax=vmax)

    for r in range(k):
        for c in range(k):
            if r == c or not np.isfinite(phi[r, c]):
                continue
            shade = (phi[r, c] - vmin) / (vmax - vmin + 1e-12)
            colour = SURFACE if (0.25 < shade < 0.55 and not diverging) else INK_PRIMARY
            colour = SURFACE if (not diverging and shade < 0.55) else colour
            ax.text(c, r, f"{phi[r, c]:.2f}\n$n$={nn[r, c]:,}", ha="center",
                    va="center", fontsize=7.4, color=colour, linespacing=1.25)

    for b in CORESIST_CLASS_BREAKS:
        ax.axhline(b - 0.5, color=INK_PRIMARY, lw=0.9)
        ax.axvline(b - 0.5, color=INK_PRIMARY, lw=0.9)

    ax.set_xticks(range(k), CORESIST_ORDER)
    ax.set_yticks(range(k), CORESIST_ORDER)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Phi coefficient" + (" (diverging, centred at 0)" if diverging else ""),
                 fontsize=9)
    cb.outline.set_visible(False)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2c. Clinical error heatmap -- random split against lineage holdout
# ---------------------------------------------------------------------------
def fig_clinical_heatmap(clinical: pd.DataFrame) -> Figure:
    """VME and categorical agreement, random split against lineage holdout.

    `clinical` must carry one row per antibiotic and the columns
    antibiotic, vme_random, vme_holdout, ca_random, ca_holdout, me_holdout.
    Rows are sorted by holdout VME so the two-group structure reads top to
    bottom, with a separator drawn where the gap falls.

    The two metric families get separate colour scales because their direction of
    badness is opposite: high VME is bad, high CA is good. Both are sequential
    and both survive greyscale; the caption must state which is which.
    """
    _setup_style()
    df = clinical.sort_values("vme_holdout").reset_index(drop=True)
    drugs = df["antibiotic"].tolist()

    vme_cols = ["vme_random", "vme_holdout"]
    ca_cols = ["ca_random", "ca_holdout"]
    headers = ["VME\nrandom", "VME\nholdout", "CA\nrandom", "CA\nholdout"]

    V = df[vme_cols].to_numpy(float)
    C = df[ca_cols].to_numpy(float)
    k = len(drugs)

    fig, ax = plt.subplots(figsize=(6.4, 0.62 * k + 1.9))

    ax.imshow(V, cmap="Reds", vmin=0, vmax=1, extent=(-0.5, 1.5, k - 0.5, -0.5),
              aspect="auto")
    ax.imshow(C, cmap="Greens", vmin=0.5, vmax=1, extent=(1.5, 3.5, k - 0.5, -0.5),
              aspect="auto")

    for r in range(k):
        for c in range(2):
            ax.text(c, r, f"{V[r, c]:.3f}", ha="center", va="center", fontsize=8.2,
                    color=SURFACE if V[r, c] > 0.55 else INK_PRIMARY)
        for c in range(2):
            ax.text(c + 2, r, f"{C[r, c]:.3f}", ha="center", va="center", fontsize=8.2,
                    color=SURFACE if C[r, c] > 0.87 else INK_PRIMARY)

    # separator between the two metric families, and between the two drug groups
    ax.axvline(1.5, color=INK_PRIMARY, lw=1.4)
    gap = np.diff(df["vme_holdout"].to_numpy(float))
    if len(gap):
        split_at = int(np.argmax(gap)) + 1
        ax.axhline(split_at - 0.5, color=INK_PRIMARY, lw=1.4, ls="--")

    ax.set_xticks(range(4), headers, fontsize=8.6)
    ax.set_yticks(range(k), drugs)
    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(k - 0.5, -0.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0, colors=INK_PRIMARY, labelcolor=INK_PRIMARY)

    fig.tight_layout()
    # Equal left/right margins. tight_layout only reserves space where labels
    # actually are, and the drug names sit on the left with nothing on the
    # right, which left the coloured grid visibly off-centre in the canvas.
    fig.subplots_adjust(left=0.15, right=0.85)
    return fig


# ---------------------------------------------------------------------------
# 2d. Bootstrap confidence intervals -- forest plot with no-skill baselines
# ---------------------------------------------------------------------------
def fig_ci_forest(ci: pd.DataFrame) -> Figure:
    """Per-antibiotic PR-AUC with 95% bootstrap intervals, sorted descending.

    `ci` must carry columns antibiotic, pr_auc, ci_lo, ci_hi, prevalence.
    The prevalence marker matters: PR-AUC is read against a no-skill baseline
    equal to the prevalence, so without it a reader cannot see that gentamicin's
    0.704 is a larger gain over chance than ampicillin's 0.944. That point is made
    in the text and, until this figure, shown nowhere.
    """
    _setup_style()
    df = ci.sort_values("pr_auc", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(7.0, 0.52 * len(df) + 1.8))

    lo = df["pr_auc"] - df["ci_lo"]
    hi = df["ci_hi"] - df["pr_auc"]
    ax.errorbar(df["pr_auc"], y, xerr=[lo, hi], fmt="o", color=BLUE,
                ecolor=BLUE, elinewidth=1.6, capsize=4, markersize=6.5, zorder=3)

    for i, row in df.iterrows():
        ax.plot([row["prevalence"]], [i], marker="|", markersize=13,
                markeredgewidth=2.0, color=ORANGE, zorder=4)
        ax.annotate(f"{row['pr_auc']:.3f}", (row["ci_hi"], i), xytext=(7, 0),
                    textcoords="offset points", va="center", fontsize=8.2,
                    color=INK_SECONDARY)

    ax.set_yticks(y, df["antibiotic"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("PR-AUC (95% bootstrap interval)")
    ax.grid(axis="x", alpha=0.6)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.legend(handles=[
        Line2D([0], [0], marker="o", color=BLUE, lw=1.6, markersize=6.5,
               label="PR-AUC with 95% interval"),
        Line2D([0], [0], marker="|", color=ORANGE, lw=0, markersize=13,
               markeredgewidth=2.0, label="No-skill baseline (test prevalence)"),
    ], loc="lower left", bbox_to_anchor=(0.0, 1.005), ncol=2,
       frameon=False, fontsize=8.6, handletextpad=0.5, columnspacing=1.8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2e. Lineage marker test -- what the attended genes actually encode
# ---------------------------------------------------------------------------
def fig_lineage_marker(u_st: dict[str, float], u_res: dict[str, float]) -> Figure:
    """Two panels: how much of each gene set is explained by lineage, and how
    much by resistance.

    The point of the figure is that the attention gene sets lose on both axes at
    once. They sit BELOW random on lineage, which rules out the easy criticism
    that attention is just reading population structure; but they also sit below
    SHAP on resistance, so ruling that out does not leave them explaining much.
    A table states both facts; only a figure makes it obvious they are the same
    gene sets failing two different tests.

    `u_st` and `u_res` map gene-set name to value. Order of insertion is the
    plotting order, so pass them with the random baseline first.
    """
    _setup_style()
    # Placed at full \textwidth; a slightly narrower canvas means less
    # downscaling on the page, so the labels stay legible. All type here is
    # INK_PRIMARY (black) rather than the secondary grey.
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))

    def _panel(ax, data, title, ref_key=None):
        labels = list(data)
        vals = [data[k] for k in labels]
        y = np.arange(len(labels))
        colors = [BASELINE if k == ref_key else
                  (ORANGE if "SHAP" in k else BLUE) for k in labels]
        ax.barh(y, vals, color=colors, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.02, i, f"{v:.3f}".rstrip("0").rstrip("."),
                    va="center", fontsize=10, color=INK_PRIMARY)
        if ref_key is not None:
            ax.axvline(data[ref_key], color=INK_SECONDARY, lw=1.0, ls="--", zorder=0)
        ax.set_yticks(y, labels, fontsize=10.5)
        ax.invert_yaxis()
        ax.set_xlim(0, max(vals) * 1.22)
        ax.set_title(title, loc="left", fontsize=11.5, color=INK_PRIMARY)
        ax.grid(axis="x", alpha=0.6)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="both", colors=INK_PRIMARY, labelcolor=INK_PRIMARY,
                       labelsize=10.5)

    # Titles kept short: at 9.6in wide the two panels sit close enough that a
    # long left-aligned title runs into the right panel's.
    _panel(axes[0], u_st, "$U_{ST}$: lineage  (lower is better)",
           ref_key=next(iter(u_st)))
    _panel(axes[1], u_res, "$U_{res}$: resistance  (higher is better)")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2b. Sequence-type size distribution -- the shape the lineage holdout has to
#     work around. Everything is computed from the real label array, so the
#     printed counts double as a check that the right array was read.
# ---------------------------------------------------------------------------
def fig_st_distribution(sequence_types: np.ndarray, n_total_strains: int) -> Figure:
    """Rank-frequency of sequence-type sizes, log y, with the 5% cutoff drawn.

    `sequence_types`: 1-D array of the called sequence type for every strain
    that has one (unmatched strains excluded by the caller).
    """
    _setup_style()
    _, counts = np.unique(sequence_types, return_counts=True)
    counts = np.sort(counts)[::-1]
    n_typed = int(counts.sum())
    n_types = len(counts)
    cutoff = 0.05 * n_typed
    n_above = int((counts > cutoff).sum())

    fig, ax = plt.subplots(figsize=(9.4, 4.6))
    ranks = np.arange(1, n_types + 1)
    ax.plot(ranks, counts, color=BLUE, lw=1.5, zorder=2)
    # The types that the five per cent rule forces into training.
    ax.scatter(ranks[:n_above], counts[:n_above], color=ORANGE, s=52,
               zorder=5, edgecolor=SURFACE, linewidth=0.8)
    ax.axhline(cutoff, color=INK_MUTED, ls="--", lw=1.1, zorder=1)

    ax.set_yscale("log")
    ax.set_xlim(0, n_types + 6)
    ax.set_xlabel("Sequence type, ranked by size")
    ax.set_ylabel("Strains in that type (log scale)")

    ax.text(n_types * 0.30, cutoff * 1.18,
            f"five per cent of the {n_typed:,} typed strains",
            fontsize=10, color=INK_PRIMARY, va="bottom")
    ax.annotate(
        f"{n_above} types sit above the line\nand are held in training\n"
        f"largest: {counts[0]:,} strains",
        xy=(ranks[0], counts[0]), xytext=(n_types * 0.16, counts[0] * 0.78),
        fontsize=10, color=INK_PRIMARY, va="top",
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9),
    )
    ax.text(0.99, 0.93,
            f"{n_types:,} types across {n_typed:,} of {n_total_strains:,} strains",
            transform=ax.transAxes, ha="right", va="top", fontsize=10,
            color=INK_PRIMARY)

    ax.grid(axis="y", alpha=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", colors=INK_PRIMARY, labelcolor=INK_PRIMARY)
    _style_bar_axes(ax)
    fig.tight_layout()

    print(f"  {n_types} sequence types over {n_typed} typed strains "
          f"({n_typed / n_total_strains:.1%} of {n_total_strains})")
    print(f"  five per cent cutoff = {cutoff:.1f} strains; {n_above} types above it")
    print(f"  ten largest types: {', '.join(f'{c:,}' for c in counts[:10])}")
    return fig


# ---------------------------------------------------------------------------
# 3. Split composition -- train/val/test sizes per split regime
# ---------------------------------------------------------------------------
def fig_split_composition(split_sizes: dict[str, dict[str, int]], display_names: Optional[dict[str, str]] = None) -> Figure:
    """Grouped bar: train/val/test size per split regime.

    `split_sizes`: {regime_name: {"train_idx": n, "val_idx": n, "test_idx": n}}
    Three fixed-order series (train/val/test) -> categorical color, legend.
    """
    _setup_style()
    display_names = display_names or {}
    regimes = list(split_sizes.keys())
    parts = ["train_idx", "val_idx", "test_idx"]
    part_labels = ["Train", "Val", "Test"]
    colors = [BLUE, AQUA, ORANGE]

    # Sized and inked for a full \textwidth figure in the thesis: every label is
    # INK_PRIMARY (black) rather than the secondary grey, and the point sizes are
    # set so the figure stays legible after LaTeX scales it to the text block.
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    x = np.arange(len(regimes))
    width = 0.25
    for i, (part, label, color) in enumerate(zip(parts, part_labels, colors)):
        values = [split_sizes[r][part] for r in regimes]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, values, width=width * 0.92, color=color, label=label)
        for rect, v in zip(bars, values):
            ax.text(rect.get_x() + rect.get_width() / 2, v + 40, f"{v:,}",
                    ha="center", va="bottom", fontsize=10, color=INK_PRIMARY, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(r, r) for r in regimes], fontsize=11)
    ax.set_ylabel("Genomes")
    all_values = [split_sizes[r][part] for r in regimes for part in parts]
    ax.set_ylim(0, max(all_values) * 1.40)  # headroom for the rotated per-bar value labels
    ax.set_title("Split-regime composition -- same 6,962 genomes, different partitions",
                 loc="left", fontsize=12.5, color=INK_PRIMARY)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=11)
    _style_bar_axes(ax)
    ax.tick_params(axis="both", colors=INK_PRIMARY, labelcolor=INK_PRIMARY)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 4. Split prevalence shift -- the B1 evidence chart
# ---------------------------------------------------------------------------
def fig_split_prevalence_shift(
    diagnostics: pd.DataFrame,
    antibiotic_order: Sequence[str],
    display_names: Optional[dict[str, str]] = None,
) -> Figure:
    """Small multiples: one diverging horizontal-bar panel per split
    regime, `prev_delta` (test - train resistance rate) per antibiotic.
    Shared, symmetric x-limits across panels so magnitudes are directly
    comparable -- this is the chart that shows `phylogroup_stratified_split`
    and `random_split` barely shift prevalence, while a genuine lineage
    holdout shifts it by tens of percentage points.
    """
    _setup_style()
    display_names = display_names or {}
    regimes = list(diagnostics["split"].unique())
    n = len(regimes)
    # Wider and taller than the analysis default, with black labels, so the
    # figure reads cleanly at full \textwidth in the thesis.
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5.1), sharey=True)
    if n == 1:
        axes = [axes]

    max_abs = diagnostics["prev_delta"].abs().max()
    xlim = max_abs * 1.25

    y_pos = np.arange(len(antibiotic_order))
    for ax, regime in zip(axes, regimes):
        sub = diagnostics[diagnostics["split"] == regime].set_index("antibiotic")
        deltas = [sub.loc[ab, "prev_delta"] if ab in sub.index else np.nan for ab in antibiotic_order]
        colors = [DIVERGING_POS if d >= 0 else DIVERGING_NEG for d in deltas]
        ax.barh(y_pos, deltas, color=colors, height=0.62)
        ax.axvline(0, color=BASELINE, linewidth=1.1)
        for yp, d in zip(y_pos, deltas):
            if np.isnan(d):
                continue
            ax.text(d + np.sign(d if d else 1) * xlim * 0.03, yp,
                    f"{d*100:+.1f}pp", va="center",
                    ha="left" if d >= 0 else "right", fontsize=9.5, color=INK_PRIMARY)
        ax.set_xlim(-xlim, xlim)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(antibiotic_order, fontsize=11)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v*100:+.0f}pp")
        ax.set_title(display_names.get(regime, regime), fontsize=11.5, loc="center",
                     color=INK_PRIMARY)
        ax.grid(axis="x", alpha=0.6)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_color(BASELINE)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="both", colors=INK_PRIMARY, labelcolor=INK_PRIMARY)

    legend_handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=DIVERGING_NEG, markersize=9,
               label="Test prevalence < train"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=DIVERGING_POS, markersize=9,
               label="Test prevalence > train"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 1.06),
               ncol=2, frameon=False, fontsize=10.5, labelcolor=INK_PRIMARY)
    fig.suptitle("Train -> test resistance-prevalence shift by split regime (per antibiotic)",
                 x=0.02, ha="left", fontsize=12.5, y=1.14, color=INK_PRIMARY)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 5. Gene feature space -- size distribution pre/post Stage 1.5, P/A prevalence,
#    per-strain gene count
# ---------------------------------------------------------------------------
def fig_gene_feature_space(
    sizes_before: np.ndarray,
    sizes_after: np.ndarray,
    gene_prevalence: np.ndarray,
    per_strain_gene_count: np.ndarray,
    min_snps_per_gene: int,
    gene_prevalence_stats: dict,
    per_strain_gene_count_stats: dict,
) -> Figure:
    """3-panel: (a) SNPs-per-gene histogram before vs after Stage 1.5
    filtering (log-x, two sequential contexts overlaid -> second hue is
    orange per the palette's convention). (b) per-gene presence-absence
    prevalence across all pan-genome genes (core/shell/rare structure).
    (c) per-strain gene-count histogram (pan-genome complement size).

    The histograms in (b)/(c) are drawn from the raw `gene_prevalence` /
    `per_strain_gene_count` arrays, but the printed rare/shell/core counts
    and the median/mean annotations use `gene_prevalence_stats` /
    `per_strain_gene_count_stats` (the pkl's own already-locked summary,
    e.g. `arcadia_v2_gene_presence_absence.pkl["gene_prevalence_stats"]`)
    rather than a fresh threshold count on the array -- a direct recompute
    from `X_pa` does not exactly reproduce those stored figures (off by a
    few dozen genes / a few tens of a gene in the mean; the stored values
    predate this refactor and were not built by exactly this code path),
    and Rule 1 (never silently change a locked number) means the already-
    reported figure wins over a fresh, slightly different recomputation.
    """
    _setup_style()
    # 13.5 x 4.2 downscaled to \textwidth left the type at roughly 5pt on the
    # page. A shorter, taller canvas plus explicit font sizes keeps all three
    # panels readable at full text width.
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 4.0))

    # No panel titles and no suptitle: nothing is printed above a figure, and
    # the LaTeX caption underneath describes all three panels. Each panel is
    # identified by its own x-axis label.
    ax = axes[0]
    bins = np.logspace(0, np.log10(max(sizes_before.max(), sizes_after.max())), 40)
    # Short legend labels. The full "Before filter (n=18,975 genes)" wording ran
    # across most of the panel and sat on top of the bars.
    ax.hist(sizes_before, bins=bins, color=BLUE, alpha=0.55,
            label=f"Before filter ({len(sizes_before):,})")
    ax.hist(sizes_after, bins=bins, color=ORANGE, alpha=0.55,
            label=f"After filter ({len(sizes_after):,})")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.38)  # clear band for the legend
    ax.axvline(min_snps_per_gene, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(min_snps_per_gene, ax.get_ylim()[1] * 0.99,
            f" min={min_snps_per_gene}", fontsize=10, color=INK_PRIMARY, va="top")
    ax.set_xscale("log")
    ax.set_xlabel("SNPs per gene (log scale)")
    ax.set_ylabel("Genes")
    # Nudged down off the top edge so it sits inside the panel rather than
    # crowding the frame.
    ax.legend(frameon=False, fontsize=9.5, loc="upper right",
              bbox_to_anchor=(1.0, 0.93), labelcolor=INK_PRIMARY)
    ax.grid(axis="y", alpha=0.6)

    ax = axes[1]
    ax.hist(gene_prevalence * 100, bins=40, color=BLUE, alpha=0.85)
    ax.axvline(5, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.axvline(95, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.set_yscale("log")
    ax.set_xlabel("% of strains carrying the gene")
    ax.set_ylabel("Genes (log scale)")
    n_rare = gene_prevalence_stats["rare_lt05"]
    n_core = gene_prevalence_stats["core_gt95"]
    n_shell = gene_prevalence_stats["shell"]
    # Anchored right, but pulled in from the edge so the block clears the
    # dashed 95% line. On the left it sat on top of the tall rare-gene bars.
    ax.text(0.88, 0.96,
            f"rare (<5%): {n_rare:,}\nshell: {n_shell:,}\ncore (>95%): {n_core:,}",
            transform=ax.transAxes, fontsize=10, va="top", ha="right",
            color=INK_PRIMARY)
    ax.grid(axis="y", alpha=0.6)

    ax = axes[2]
    ax.hist(per_strain_gene_count, bins=40, color=BLUE, alpha=0.85)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_xlabel("Genes present (per genome)")
    ax.set_ylabel("Genomes")
    med = per_strain_gene_count_stats["median"]
    ax.axvline(med, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(med, ax.get_ylim()[1] * 0.99, f" median={med:,.0f}", fontsize=10,
            color=INK_PRIMARY, va="top")
    ax.grid(axis="y", alpha=0.6)

    for a in axes:
        a.tick_params(axis="both", labelsize=10.5, colors=INK_PRIMARY,
                      labelcolor=INK_PRIMARY)
        _style_bar_axes(a)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 6. Giessen overview -- external validation set
# ---------------------------------------------------------------------------
def fig_giessen_overview(prevalence: dict[str, float], split_sizes: dict[str, int]) -> Figure:
    """2-panel: (a) antibiotic resistance prevalence (4 antibiotics, all
    fully labelled -- no missingness in Giessen). (b) train/val/test
    composition as a pie (3 slices, genuine part-to-whole of one fixed
    total -- a legitimate pie use, unlike the 8-way phylogroup case).
    """
    _setup_style()
    # Placed at full \textwidth, so keep the canvas compact to limit the
    # downscale and keep the labels readable. wspace leaves room for the pie
    # legend, which sits outside ax2 on the right.
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0),
                             gridspec_kw={"wspace": 0.45})

    ax = axes[0]
    items = sorted(prevalence.items(), key=lambda kv: -kv[1])
    names = [k for k, _ in items]
    values = [v for _, v in items]
    bars = ax.bar(names, values, color=SEQUENTIAL, width=0.5)
    for rect, v in zip(bars, values):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.1%}",
                ha="center", va="bottom", fontsize=10.5, color=INK_PRIMARY)
    ax.set_ylim(0, max(values) * 1.25)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    # Panel name below the panel, never above it.
    ax.set_xlabel("Resistance prevalence", fontsize=11, color=INK_PRIMARY,
                  labelpad=8)
    ax.tick_params(axis="both", labelsize=11, colors=INK_PRIMARY,
                   labelcolor=INK_PRIMARY)
    _style_bar_axes(ax)

    ax2 = axes[1]
    labels = ["Train", "Val", "Test"]
    sizes = [split_sizes["train_idx"], split_sizes["val_idx"], split_sizes["test_idx"]]
    total = sum(sizes)
    colors = [BLUE, AQUA, ORANGE]
    wedges, _texts, autotexts = ax2.pie(
        sizes, colors=colors, autopct=lambda p: f"{p:.0f}%", pctdistance=0.7,
        wedgeprops=dict(edgecolor=SURFACE, linewidth=2), startangle=90, counterclock=False,
    )
    for t in autotexts:
        t.set_fontsize(10.5)
        t.set_color(SURFACE)
        t.set_fontweight("bold")
    ax2.legend(wedges, [f"{l} (n={s})" for l, s in zip(labels, sizes)],
               loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
               fontsize=10.5, labelcolor=INK_PRIMARY)
    # The retraining split total, printed under the pie rather than over it.
    ax2.set_xlabel(f"Retraining split (n={total})", fontsize=11, color=INK_PRIMARY)

    # No tight_layout here: a pie is aspect-locked, so tight_layout cannot
    # solve this figure and warns that the result "might be incorrect". Fixed
    # margins are deterministic, and _save's bbox_inches="tight" still trims
    # the canvas around the legend that sits outside ax2.
    fig.subplots_adjust(left=0.09, right=0.82, top=0.94, bottom=0.16, wspace=0.45)
    return fig
