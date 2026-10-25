#!/usr/bin/env python
"""
Regenerate every data figure in the thesis from one place, with one palette.

    python make_thesis_figures.py

Writes .pdf (+ .png) straight into images/, overwriting the current files.

------------------------------------------------------------------------------
WHY THIS SCRIPT EXISTS
------------------------------------------------------------------------------
thesis_audit/make_figures.py is the provenance script: it rebuilds figures from
the processed .pkl artifacts via src.amr_data.viz. Keep it. It is the audit
trail. But it covers only 6 figures and it cannot run outside the project repo.

This script is the *presentation* layer. Every number below is a locked, final
value already reported in a thesis table, so the figures are reproducible from
the document itself. Each constant is tagged with the table it comes from. If a
table changes, change it here too.

------------------------------------------------------------------------------
FIGURES THIS SCRIPT CANNOT MAKE  (they need the real arrays, not summary stats)
------------------------------------------------------------------------------
    fig_pr_curves              full precision-recall curves
    fig_gene_feature_space     histograms over 18,975 / 32,441 genes
    fig_st_distribution        rank curve over 458 sequence types
    fig_attention_concentration  per-rank attention weights
    fig_threshold_sweep        VME/ME across the whole threshold range
Regenerate those from the project repo, but import PALETTE and apply_style()
from here so they match.
------------------------------------------------------------------------------
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Figures go straight into the thesis LaTeX folder: TriLens-AMR/thesis/images
OUT = Path(__file__).resolve().parents[3] / "thesis" / "images"
OUT.mkdir(exist_ok=True)

# =============================================================================
# ONE PALETTE FOR THE WHOLE THESIS
# =============================================================================
# Okabe-Ito: colourblind-safe (deuteranopia/protanopia/tritanopia) AND
# lightness-separated, so the figures survive a black-and-white printer.
# Never pair RED with GREEN to carry meaning. Use BLUE vs VERMILLION.

# Tokens mirror the \definecolor block in main.tex exactly, so the matplotlib
# figures and the TikZ diagrams are one system. Change both together.
TOKENS = {
    "primary":  "#1F6F8B",   # our model / random split / first series
    "accent":   "#D1601A",   # contrast condition / lineage holdout
    "support1": "#4A9D7C",   # third series (validation, prevalence-matched)
    "support2": "#C9A227",   # fourth series only
    "neutral":  "#7F8C8D",   # baselines, external methods, random genes
    "ink":      "#1A1A1A",   # text, arrows, borders
    "rule":     "#C8CFD3",   # gridlines
}

# --- semantic roles: a colour means the SAME THING in every figure -----------
PALETTE = {
    "random":       TOKENS["primary"],   # random split
    "holdout":      TOKENS["accent"],    # sequence-type lineage holdout
    "matched":      TOKENS["support1"],  # prevalence-matched holdout
    "phylogroup":   TOKENS["support1"],  # phylogroup-stratified split
    "proposed":     TOKENS["primary"],   # TriLens-MoE-Stack, always emphasised
    # the audit maps every baseline to neutral, but TabNet and the PCA baseline
    # appear as separate series in one figure, so the project's own baseline
    # takes the fourth-series colour and only external methods are neutral
    "baseline":     TOKENS["support2"],
    "external":     TOKENS["neutral"],   # TabNet, null model
    "lensA":        TOKENS["support2"],
    "lensB":        TOKENS["primary"],
    "lensC":        TOKENS["support1"],
    "good":         TOKENS["primary"],
    "bad":          TOKENS["accent"],
    "neutral":      TOKENS["neutral"],
    "met":          TOKENS["primary"],
    "partial":      TOKENS["support2"],
    "unmet":        TOKENS["accent"],
}

# ---------------------------------------------------------------------------
# FILL RICHNESS -- the one knob for how saturated every figure looks.
# Pale fills wash out on glossy paper, so all fills are pulled back toward the
# pure colour by this factor. 1.0 = the old pastel look, 0.0 = fully saturated.
FILL_SCALE = 0.28
# ---------------------------------------------------------------------------


def light(hex_colour: str, amount: float = 0.72) -> str:
    """Blend a colour toward white. amount=0 -> unchanged, 1 -> white.

    The requested amount is scaled by FILL_SCALE, so the richness of the whole
    document can be tuned from a single constant above.
    """
    amount = amount * FILL_SCALE
    r, g, b = mcolors.to_rgb(hex_colour)
    r = r + (1.0 - r) * amount
    g = g + (1.0 - g) * amount
    b = b + (1.0 - b) * amount
    return mcolors.to_hex((r, g, b))


def dark(hex_colour: str, amount: float = 0.32) -> str:
    """Blend a colour toward black, for outlines that read against a rich fill."""
    r, g, b = mcolors.to_rgb(hex_colour)
    return mcolors.to_hex((r * (1 - amount), g * (1 - amount), b * (1 - amount)))


def apply_style() -> None:
    """House style. Serif to match the thesis body text."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Segoe UI"],
        # 9 pt base, so nothing falls under the 8 pt print floor
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": TOKENS["ink"],
        "axes.labelcolor": TOKENS["ink"],
        "text.color": TOKENS["ink"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": TOKENS["rule"],
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "xtick.color": TOKENS["ink"],
        "ytick.color": TOKENS["ink"],
    })


def save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print(f"  wrote images/{name}.pdf (+ .png)")


# =============================================================================
# LOCKED DATA  (every block tagged with its source table)
# =============================================================================
AB = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]

# Table 4.4 -- labels after cleaning
PREVALENCE = {"AMP": .501, "CIP": .228, "AMC": .312, "SXT": .310,
              "GEN": .118, "CTX": .152, "CAZ": .131}
COVERAGE = {"AMP": .699, "CIP": .867, "AMC": .543, "SXT": .491,
            "GEN": .867, "CTX": .777, "CAZ": .858}

# Table 4.5 -- phylogroup composition
PHYLOGROUPS = {"B2": 3542, "Unknown": 1697, "D": 723, "A": 317,
               "F": 307, "B1": 258, "C": 114, "E": 4}

# Table 4.7 -- the three evaluation splits
SPLIT_SIZES = {
    "Random":                (4872, 1045, 1045),
    "Phylogroup-stratified": (4872, 1045, 1045),
    "Sequence-type holdout": (4836, 1081, 1045),
}

# Figure 4.x -- test minus train prevalence, percentage points
PREV_SHIFT = {
    "Random":                {"AMP": -0.0, "CIP": -1.7, "SXT": +0.5, "CTX": -2.5,
                              "CAZ": -1.1, "AMC": -0.5, "GEN": -1.8},
    "Phylogroup-stratified": {"AMP": -1.5, "CIP": -1.1, "SXT": -1.0, "CTX": +0.7,
                              "CAZ": +1.3, "AMC": +0.4, "GEN": -0.3},
    "Sequence-type holdout": {"AMP": -8.2, "CIP": -12.1, "SXT": -2.5, "CTX": -8.8,
                              "CAZ": -10.6, "AMC": -5.9, "GEN": -10.4},
}

# Table 5.1 -- TriLens on the random split
PR_RANDOM = {"CIP": .9773, "AMP": .9440, "CTX": .9019, "SXT": .8352,
             "CAZ": .8299, "AMC": .7482, "GEN": .7037}

# Table 5.5 -- bootstrap confidence intervals
CI = {"AMP": (.919, .965), "CIP": (.961, .990), "AMC": (.680, .809),
      "SXT": (.773, .891), "GEN": (.612, .786), "CTX": (.851, .943),
      "CAZ": (.762, .882)}
# test-set no-skill baselines, read off fig_ci_forest
NOSKILL = {"AMP": .50, "CIP": .22, "AMC": .31, "SXT": .31,
           "GEN": .10, "CTX": .13, "CAZ": .12}

# Table 5.3 -- mean gate weights (Lens A, Lens B, Lens C)
GATE = {"AMP": (.070, .852, .078), "CIP": (.035, .590, .374),
        "AMC": (.095, .742, .163), "SXT": (.052, .900, .047),
        "GEN": (.098, .731, .171), "CTX": (.064, .815, .121),
        "CAZ": (.044, .835, .122)}

# Table 5.2 / 5.12 -- clinical error rates
CLIN_RANDOM = {"CIP": (.086, .015, .969), "AMP": (.109, .104, .893),
               "CTX": (.190, .026, .953), "SXT": (.236, .080, .871),
               "CAZ": (.291, .033, .936), "AMC": (.394, .108, .804),
               "GEN": (.436, .023, .934)}
CLIN_HOLDOUT = {"CIP": (.114, .006, .978), "AMP": (.160, .095, .877),
                "SXT": (.310, .072, .863), "AMC": (.710, .076, .754),
                "GEN": (.868, .002, .962), "CAZ": (.837, .001, .959),
                "CTX": (.915, .001, .918)}

# Table 5.10 -- prevalence-matched holdout. (random, holdout, matched, lo, hi, gap)
PREVINV = {
    "AMP": (.9440, .9043, .9268, .918, .938, -.017),
    "CIP": (.9773, .9118, .9471, .926, .967, -.030),
    "SXT": (.8352, .7872, .8046, .792, .823, -.031),
    "CTX": (.9019, .6073, .7260, .687, .766, -.176),
    "CAZ": (.8299, .4930, .6431, .600, .679, -.187),
    "AMC": (.7482, .4865, .5513, .529, .584, -.197),
    "GEN": (.7037, .2457, .4594, .398, .511, -.244),
}

# Table 5.9 -- per-antibiotic PR-AUC on all three splits
THREE_SPLITS = {
    "AMP": (.9440, .9338, .9043), "SXT": (.8352, .8373, .7872),
    "CIP": (.9773, .9617, .9118), "AMC": (.7482, .7963, .4865),
    "CTX": (.9019, .8807, .6073), "CAZ": (.8299, .8481, .4930),
    "GEN": (.7037, .7136, .2457),
}

# Table 5.6 -- baseline ladder. (label, random, holdout or None, kind)
BASELINES = [
    ("Phylogroup label only",       .2914, None,  "external"),
    ("TabNet",                      .6951, .3255, "external"),
    ("XGBoost on 1,024 PCA comps",  .8033, None,  "baseline"),
    ("Random Forest, genes",        .7809, .4931, "component"),
    ("Logistic Regression, genes",  .7923, .4408, "component"),
    ("LightGBM, genes",             .8338, .5516, "component"),
    ("XGBoost, genes",              .8375, .5507, "component"),
    ("XGBoost, chi-sq SNPs",        .8400, .6299, "component"),
    ("TriLens-MoE-Stack",           .8486, .6337, "proposed"),
]

# Table 5.13 -- project progression
PROGRESSION = [
    ("XGBoost\non PCA",       .8033, "baseline"),
    ("XGBoost on\ngene feats", .8297, "baseline"),
    ("GeneAttention\nNet v1",  .7430, "external"),
    ("GeneAttention\nNet v2",  .7689, "external"),
    ("Deep meta-\nlearner",    .8253, "external"),
    ("TriLens-\nMoE-Stack",    .8486, "proposed"),
]

# Table 5.4 -- component ablation
ABLATION = {
    "Stage 1: lenses only": {"AMP": .8920, "CIP": .9248, "AMC": .6249, "SXT": .7217,
                             "GEN": .5168, "CTX": .8103, "CAZ": .7267},
    "Stage 2: + MoE gate":  {"AMP": .8970, "CIP": .9489, "AMC": .6347, "SXT": .7453,
                             "GEN": .5835, "CTX": .8659, "CAZ": .7893},
    "Stage 3: + stacking":  {"AMP": .9440, "CIP": .9773, "AMC": .7482, "SXT": .8352,
                             "GEN": .7037, "CTX": .9019, "CAZ": .8299},
}

# Table 5.15 -- Giessen rebuild. (arcadia, giessen, retention %)
GIESSEN = {"CIP": (.9773, .9868, 101.0), "CAZ": (.8299, .7775, 93.7),
           "CTX": (.9019, .7876, 87.3), "GEN": (.7037, .5480, 77.9)}
GIESSEN_PREV = {"CIP": .452, "CTX": .443, "CAZ": .341, "GEN": .232}
GIESSEN_SPLIT = (647, 81, 81)

# Figure 5.x -- TabNet comparison. (tabnet, pca baseline, trilens)
TABNET = {"AMP": (.862, .903, .9440), "CIP": (.902, .948, .9773),
          "AMC": (.620, .748, .7482), "SXT": (.774, .710, .8352),
          "GEN": (.3751, .655, .7037), "CTX": (.710, .877, .9019),
          "CAZ": (.618, .782, .8299)}

# Table 5.17 -- lineage vs resistance information
UMARK = {
    "U_ST":  [("Random genes", .539), ("SHAP, classical", .375),
              ("Attention, standard", .312), ("Attention, strain-cond.", .357)],
    "U_res": [("SHAP, classical", .0473), ("Attention, standard", .0192),
              ("Attention, strain-cond.", .0419)],
}

# Section 3.1.6 -- requirements compliance
FR_STATUS = {"FR1": "partial", "FR2": "met", "FR3": "met", "FR4": "met",
             "FR5": "met", "FR6": "met", "FR7": "met", "FR8": "partial",
             "FR9": "met", "FR10": "unmet"}
NFR_STATUS = {"NFR1": "met", "NFR2": "partial", "NFR3": "met", "NFR4": "met",
              "NFR5": "met", "NFR6": "met", "NFR7": "met", "NFR8": "met",
              "NFR9": "met", "NFR10": "met", "NFR11": "unmet"}

# Figure 4.x -- pairwise co-resistance, phi coefficient and n
CORES_ORDER = ["AMP", "AMC", "CTX", "CAZ", "CIP", "GEN", "SXT"]
CORES_PHI = {
    ("AMP", "AMC"): (.54, 1688), ("AMP", "CTX"): (.38, 3393), ("AMP", "CAZ"): (.36, 3880),
    ("AMP", "CIP"): (.43, 3941), ("AMP", "GEN"): (.27, 3943), ("AMP", "SXT"): (.55, 3403),
    ("AMC", "CTX"): (.23, 2619), ("AMC", "CAZ"): (.28, 2806), ("AMC", "CIP"): (.28, 2853),
    ("AMC", "GEN"): (.32, 2855), ("AMC", "SXT"): (.04, 235),
    ("CTX", "CAZ"): (.84, 5394), ("CTX", "CIP"): (.51, 5401), ("CTX", "GEN"): (.39, 5405),
    ("CTX", "SXT"): (.26, 2861),
    ("CAZ", "CIP"): (.52, 5966), ("CAZ", "GEN"): (.36, 5972), ("CAZ", "SXT"): (.23, 3350),
    ("CIP", "GEN"): (.40, 6030), ("CIP", "SXT"): (.34, 3410),
    ("GEN", "SXT"): (.21, 3414),
}


# =============================================================================
# EXISTING FIGURES, RECOLOURED
# =============================================================================
def fig_label_overview() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, data, panel, title, colour in [
        (axes[0], PREVALENCE, "(a)", "Resistance prevalence", PALETTE["holdout"]),
        (axes[1], COVERAGE, "(b)", "Label coverage", PALETTE["random"]),
    ]:
        items = sorted(data.items(), key=lambda kv: -kv[1])
        names = [k for k, _ in items]
        vals = [v * 100 for _, v in items]
        bars = ax.bar(names, vals, color=light(colour, .55), edgecolor=dark(colour), linewidth=1.0)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        # both panels share 0-100 so the two quantities are comparable
        ax.set_ylim(0, 100)
        ax.set_title(f"{panel} {title}", fontsize=10, loc="left")
        ax.set_xlabel("Antibiotic")
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Percentage of the 6,962 strains (\\%)".replace("\\", ""))
    fig.tight_layout()
    save(fig, "fig_label_overview")


def fig_population_structure() -> None:
    fig, ax = plt.subplots(figsize=(8, 3.6))
    names = list(PHYLOGROUPS)
    vals = list(PHYLOGROUPS.values())
    total = sum(vals)
    # B2 and Unknown are the two limitations discussed in the text: highlight them.
    colours = [PALETTE["holdout"] if n in ("B2", "Unknown") else PALETTE["random"]
               for n in names]
    bars = ax.bar(names, vals,
                  color=[light(c, .55) for c in colours],
                  edgecolor=[dark(c) for c in colours], linewidth=1.0)
    ax.bar_label(bars, labels=[f"{v:,}\n({v / total * 100:.1f}%)" for v in vals],
                 padding=3, fontsize=8)
    ax.set_ylim(0, max(vals) * 1.28)
    ax.set_ylabel("Strains")
    ax.set_xlabel("Phylogroup")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[
        Patch(facecolor=light(PALETTE["holdout"], .55), edgecolor=dark(PALETTE["holdout"]),
              label="Dominant or unassigned (limits per-group analysis)"),
        Patch(facecolor=light(PALETTE["random"], .55), edgecolor=dark(PALETTE["random"]),
              label="Remaining phylogroups"),
    ], loc="upper right")
    fig.tight_layout()
    save(fig, "fig_population_structure")


def fig_split_composition() -> None:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(SPLIT_SIZES))
    w = 0.26
    parts = [("Train", 0, PALETTE["random"]),
             ("Validation", 1, PALETTE["matched"]),
             ("Test", 2, PALETTE["holdout"])]
    for label, i, colour in parts:
        vals = [v[i] for v in SPLIT_SIZES.values()]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=label,
                      color=light(colour, .5), edgecolor=dark(colour), linewidth=1.0)
        # labels=... rather than a callable fmt, which needs matplotlib >= 3.7
        ax.bar_label(bars, labels=[f"{v:,}" for v in vals], padding=3, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(SPLIT_SIZES.keys())
    ax.set_ylabel("Genomes")
    ax.set_ylim(0, 5600)
    ax.legend(ncol=3, loc="upper center")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_split_composition")


def fig_split_prevalence_shift() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    order = ["AMP", "CIP", "SXT", "CTX", "CAZ", "AMC", "GEN"]
    for ax, (split, data) in zip(axes, PREV_SHIFT.items()):
        y = np.arange(len(order))[::-1]
        vals = [data[a] for a in order]
        # direction, not identity: down = lower in test. Blue vs vermillion.
        colours = [PALETTE["holdout"] if v < 0 else PALETTE["random"] for v in vals]
        ax.barh(y, vals, color=[light(c, .45) for c in colours],
                edgecolor=[dark(c) for c in colours], linewidth=1.0, height=.62)
        for yi, v in zip(y, vals):
            ax.text(v + (-0.45 if v < 0 else 0.45), yi, f"{v:+.1f}pp",
                    va="center", ha="right" if v < 0 else "left", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(order)
        ax.axvline(0, color="#333333", linewidth=0.9)
        ax.set_xlim(-16, 8)
        ax.set_title(split)
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
    axes[1].set_xlabel("Test prevalence minus train prevalence (percentage points)")
    fig.legend(handles=[
        Patch(facecolor=light(PALETTE["holdout"], .45), edgecolor=dark(PALETTE["holdout"]),
              label="Test prevalence lower than train"),
        Patch(facecolor=light(PALETTE["random"], .45), edgecolor=dark(PALETTE["random"]),
              label="Test prevalence higher than train"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(.5, -.06))
    fig.tight_layout()
    save(fig, "fig_split_prevalence_shift")


def fig_gate_weights() -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    order = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
    a = np.array([GATE[k][0] for k in order])
    b = np.array([GATE[k][1] for k in order])
    c = np.array([GATE[k][2] for k in order])
    for vals, base, label, colour in [
        (a, np.zeros_like(a), "Lens A (SNP motifs)", PALETTE["lensA"]),
        (b, a, "Lens B (presence/absence)", PALETTE["lensB"]),
        (c, a + b, "Lens C (cross-antibiotic)", PALETTE["lensC"]),
    ]:
        ax.bar(order, vals, bottom=base, label=label,
               color=light(colour, .35), edgecolor=dark(colour), linewidth=1.0)
    ax.set_ylabel("Mean gate weight")
    ax.set_ylim(0, 1.0)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.14))
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_gate_weights")


def fig_ci_forest() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.4))
    order = sorted(PR_RANDOM, key=lambda k: PR_RANDOM[k])
    y = np.arange(len(order))
    for yi, ab in zip(y, order):
        lo, hi = CI[ab]
        v = PR_RANDOM[ab]
        ax.plot([lo, hi], [yi, yi], color=PALETTE["random"], linewidth=1.8,
                solid_capstyle="butt")
        ax.plot([lo, lo], [yi - .12, yi + .12], color=PALETTE["random"], linewidth=1.8)
        ax.plot([hi, hi], [yi - .12, yi + .12], color=PALETTE["random"], linewidth=1.8)
        ax.plot(v, yi, "o", color=PALETTE["random"], markersize=6.5, zorder=3)
        ax.text(hi + .015, yi, f"{v:.3f}", va="center", fontsize=8.5)
        ax.plot([NOSKILL[ab]] * 2, [yi - .22, yi + .22],
                color=PALETTE["holdout"], linewidth=3.0, solid_capstyle="butt")
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlim(0, 1.06)
    ax.set_xlabel("PR-AUC (95% bootstrap interval)")
    # above the axes: "lower right" sat on top of the GEN row
    ax.legend(handles=[
        plt.Line2D([], [], color=PALETTE["random"], marker="o",
                   label="PR-AUC with 95% interval"),
        plt.Line2D([], [], color=PALETTE["holdout"], linewidth=3,
                   label="No-skill baseline (test prevalence)"),
    ], loc="lower center", ncol=2, bbox_to_anchor=(.5, 1.01))
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_ci_forest")


def fig_clinical_heatmap() -> None:
    """VME and CA, random vs holdout.

    Deliberately NOT red-vs-green. Both blocks use a single sequential ramp and
    are separated by position and by an explicit 'lower/higher is better' note,
    so nothing depends on hue discrimination.
    """
    order = sorted(CLIN_HOLDOUT, key=lambda k: CLIN_HOLDOUT[k][0])
    mat = np.array([[CLIN_RANDOM[a][0], CLIN_HOLDOUT[a][0],
                     1 - CLIN_RANDOM[a][2], 1 - CLIN_HOLDOUT[a][2]] for a in order])
    disp = np.array([[CLIN_RANDOM[a][0], CLIN_HOLDOUT[a][0],
                      CLIN_RANDOM[a][2], CLIN_HOLDOUT[a][2]] for a in order])
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "vermillion", ["#FDF2EA", light(PALETTE["holdout"], 1.4), PALETTE["holdout"],
                       "#7A3600"])
    fig, ax = plt.subplots(figsize=(7, 5.2))
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(len(order)):
        for j in range(4):
            ax.text(j, i, f"{disp[i, j]:.3f}", ha="center", va="center",
                    fontsize=9.5, color="#111111" if mat[i, j] < .42 else "white")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["VME\nrandom", "VME\nholdout", "CA\nrandom", "CA\nholdout"])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    # uniform cell borders, same reasoning as the co-resistance matrix
    ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(order), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", length=0)
    ax.axvline(1.5, color="#2B2B2B", linewidth=1.6)
    ax.axhline(2.5, color="#2B2B2B", linewidth=1.6, linestyle="--")
    ax.set_title("Darker = worse in both blocks\n"
                 "(VME: lower is better · CA: higher is better)", fontsize=9.5)
    ax.grid(which="major", visible=False)  # keep the minor cell borders
    fig.tight_layout()
    save(fig, "fig_clinical_heatmap")


def fig_giessen_overview() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 5),
                             gridspec_kw={"height_ratios": [3, 1]})
    names = list(GIESSEN_PREV)
    vals = [v * 100 for v in GIESSEN_PREV.values()]
    bars = axes[0].bar(names, vals, color=light(PALETTE["holdout"], .55),
                       edgecolor=dark(PALETTE["holdout"]), linewidth=1.0)
    axes[0].bar_label(bars, fmt="%.1f", padding=3, fontsize=8.5)
    axes[0].set_ylim(0, 55)
    axes[0].set_ylabel("Resistance prevalence (%)")
    axes[0].set_xlabel("Antibiotic")
    axes[0].set_title("(a) Resistance prevalence, four shared antibiotics",
                      fontsize=10, loc="left")
    axes[0].set_axisbelow(True)
    axes[0].grid(axis="x", visible=False)

    # a horizontal stacked bar, not a pie: the audit is right that a 3-slice
    # pie of an already-stated split adds nothing
    tr, va, te = GIESSEN_SPLIT
    total = sum(GIESSEN_SPLIT)
    left = 0
    for n, label, colour in ((tr, "Train", PALETTE["random"]),
                             (va, "Validation", PALETTE["matched"]),
                             (te, "Test", PALETTE["holdout"])):
        axes[1].barh(0, n, left=left, height=.55, color=light(colour, .45),
                     edgecolor=dark(colour), linewidth=1.0)
        axes[1].text(left + n / 2, 0, f"{label}\nn={n} ({n / total * 100:.0f}%)",
                     ha="center", va="center", fontsize=8.5)
        left += n
    axes[1].set_xlim(0, total)
    axes[1].set_ylim(-.5, .5)
    axes[1].set_yticks([])
    axes[1].set_xlabel("Isolates")
    axes[1].set_title(f"(b) Retraining split (n = {total})", fontsize=10, loc="left")
    axes[1].grid(visible=False)
    for s in ("left", "top", "right"):
        axes[1].spines[s].set_visible(False)
    fig.tight_layout()
    save(fig, "fig_giessen_overview")


def fig_giessen_retention() -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    order = sorted(GIESSEN, key=lambda k: -GIESSEN[k][2])
    x = np.arange(len(order))
    w = .36
    arc = [GIESSEN[a][0] for a in order]
    gie = [GIESSEN[a][1] for a in order]
    b1 = ax.bar(x - w / 2, arc, w, label="Arcadia",
                color=light(PALETTE["random"], .5), edgecolor=dark(PALETTE["random"]), linewidth=1.0)
    b2 = ax.bar(x + w / 2, gie, w, label="Giessen (rebuilt)",
                color=light(PALETTE["holdout"], .5), edgecolor=dark(PALETTE["holdout"]), linewidth=1.0)
    ax.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.3f", padding=2, fontsize=8)
    for xi, a in zip(x, order):
        ax.text(xi, 1.06, f"retention {GIESSEN[a][2]:.1f}%", ha="center", fontsize=8.5,
                color=PALETTE["holdout"])
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("PR-AUC")
    # above the axes: "lower left" printed inside the CIP bars
    ax.legend(loc="lower center", ncol=2, bbox_to_anchor=(.5, 1.05))
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_giessen_retention")


def fig_lineage_marker() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, key, title, better in [
        (axes[0], "U_ST", r"$U_{ST}$: lineage  (lower is better)", "lower"),
        (axes[1], "U_res", r"$U_{res}$: resistance  (higher is better)", "higher"),
    ]:
        rows = UMARK[key]
        names = [r[0] for r in rows][::-1]
        vals = [r[1] for r in rows][::-1]
        colours = [PALETTE["neutral"] if "Random" in n else
                   PALETTE["baseline"] if "SHAP" in n else PALETTE["random"]
                   for n in names]
        bars = ax.barh(names, vals, color=[light(c, .5) for c in colours],
                       edgecolor=[dark(c) for c in colours], linewidth=1.0, height=.62)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(vals) * 1.25)
        ax.set_axisbelow(True)
        ax.grid(axis="y", visible=False)
        if key == "U_ST":
            ax.axvline(.539, color="#333333", linestyle="--", linewidth=1.0)
    fig.tight_layout()
    save(fig, "fig_lineage_marker")


def fig_coresistance() -> None:
    n = len(CORES_ORDER)
    mat = np.full((n, n), np.nan)
    lab = np.empty((n, n), dtype=object)
    for i, a in enumerate(CORES_ORDER):
        for j, b in enumerate(CORES_ORDER):
            if i == j:
                continue
            key = (a, b) if (a, b) in CORES_PHI else (b, a)
            phi, cnt = CORES_PHI[key]
            mat[i, j] = phi
            lab[i, j] = f"{phi:.2f}\nn={cnt:,}"
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "phi", ["#EFF5FA", light(PALETTE["random"], 1.9), PALETTE["random"], "#00314F"])
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat, cmap=cmap, vmin=0, vmax=.85)
    # the empty diagonal would otherwise render white, which is the colormap's
    # own 0.0 value -- hatch it so it cannot be read as "phi = 0"
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - .5, i - .5), 1, 1, facecolor="#E8E8E8",
                                   edgecolor="white", linewidth=1.8,
                                   hatch="///", zorder=2))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ax.text(j, i, lab[i, j], ha="center", va="center", fontsize=7.5,
                    color="#111111" if mat[i, j] < .40 else "white")
    ax.set_xticks(range(n)); ax.set_xticklabels(CORES_ORDER)
    ax.set_yticks(range(n)); ax.set_yticklabels(CORES_ORDER)
    # A border on EVERY cell. Previously only cells beside a class divider
    # looked boxed and the rest had none, which read as an inconsistency.
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.8)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="major", visible=False)
    # Class boundaries must be unmistakably heavier than the 1.8pt white cell
    # borders, and drawn on top of them.
    for edge in (1.5, 3.5, 4.5, 5.5):  # penicillins | cephalosporins | CIP | GEN | SXT
        ax.axvline(edge, color="#1A1A1A", linewidth=3.0, zorder=6)
        ax.axhline(edge, color="#1A1A1A", linewidth=3.0, zorder=6)
    fig.colorbar(im, ax=ax, label="Phi coefficient", shrink=.82)
    fig.tight_layout()
    save(fig, "fig_coresistance")


def fig_tabnet_vs_trilens() -> None:
    fig, ax = plt.subplots(figsize=(10, 4.4))
    order = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
    x = np.arange(len(order))
    w = .26
    series = [("TabNet", 0, PALETTE["external"]),
              ("XGBoost on PCA components (baseline)", 1, PALETTE["baseline"]),
              ("TriLens-MoE-Stack", 2, PALETTE["proposed"])]
    for label, i, colour in series:
        vals = [TABNET[a][i] for a in order]
        ax.bar(x + (i - 1) * w, vals, w, label=label,
               color=light(colour, .45), edgecolor=dark(colour), linewidth=1.0)
    for xi, a in zip(x, order):
        gain = TABNET[a][2] - TABNET[a][1]
        ax.text(xi + w, TABNET[a][2] + .02, f"{gain:+.3f}", ha="center", fontsize=8.5)
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("PR-AUC (random split)")
    ax.set_ylim(0, 1.14)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.16))
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_tabnet_vs_trilens")


# =============================================================================
# NEW FIGURES
# =============================================================================
def fig_baseline_ladder() -> None:
    """Tier 1 #4. Every model on both splits.

    Shows the proposed model on top for the random split, and the whole field
    compressing under the holdout. The compression is the finding.
    """
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    rows = list(reversed(BASELINES))
    y = np.arange(len(rows))
    for yi, (label, rnd, hold, kind) in zip(y, rows):
        # the proposed model must be visually distinct: "proposed" and "random"
        # are both blue, so every other row is drawn neutral instead
        c_r = PALETTE["proposed"] if kind == "proposed" else PALETTE["neutral"]
        ax.plot([0, rnd], [yi + .16] * 2, color=light(c_r, .6), linewidth=5,
                solid_capstyle="butt", zorder=1)
        ax.plot(rnd, yi + .16, "o", color=c_r, markersize=8, zorder=3)
        ax.text(rnd + .012, yi + .16, f"{rnd:.4f}", va="center", fontsize=8.5)
        if hold is not None:
            ax.plot([0, hold], [yi - .16] * 2, color=light(PALETTE["holdout"], .6),
                    linewidth=5, solid_capstyle="butt", zorder=1)
            ax.plot(hold, yi - .16, "s", color=PALETTE["holdout"], markersize=7, zorder=3)
            ax.text(hold + .012, yi - .16, f"{hold:.4f}", va="center", fontsize=8.5)
        else:
            ax.text(.012, yi - .16, "not run under the holdout", va="center",
                    fontsize=8, style="italic", color="#888888")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    for yi, (_, _, _, kind) in zip(y, rows):
        if kind == "proposed":
            ax.get_yticklabels()[yi].set_fontweight("bold")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Mean PR-AUC across the seven antibiotics")
    ax.axhline(5.5, color="#BBBBBB", linewidth=.8, linestyle=":")
    ax.axhline(0.5, color="#BBBBBB", linewidth=.8, linestyle=":")
    # Three keys, not two: the random-split rows are drawn in two colours
    # (proposed in blue, baselines in neutral), so a single blue "Random split"
    # key contradicted eight of the nine bars.
    ax.legend(handles=[
        plt.Line2D([], [], color=PALETTE["neutral"], marker="o", linestyle="",
                   label="Random split, baselines"),
        plt.Line2D([], [], color=PALETTE["proposed"], marker="o", linestyle="",
                   label="Random split, proposed"),
        plt.Line2D([], [], color=PALETTE["holdout"], marker="s", linestyle="",
                   label="Sequence-type lineage holdout"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(.5, 1.01))
    ax.set_axisbelow(True)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "fig_baseline_ladder")


def fig_progression() -> None:
    """Tier 2 #5. The project's real trajectory, dips included."""
    fig, ax = plt.subplots(figsize=(9, 4.4))
    labels = [p[0] for p in PROGRESSION]
    vals = [p[1] for p in PROGRESSION]
    kinds = [p[2] for p in PROGRESSION]
    x = np.arange(len(vals))
    colours = [PALETTE[k] if k in PALETTE else PALETTE["neutral"] for k in kinds]
    bars = ax.bar(x, vals, .58, color=[light(c, .5) for c in colours],
                  edgecolor=[dark(c) for c in colours], linewidth=1.2, zorder=2)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8.5)
    ax.axhline(.8033, color=PALETTE["baseline"], linestyle="--", linewidth=1.0)
    # parked in the gap above the two GeneAttentionNet bars: on the right it
    # printed over the TriLens bar, on the left over the first bar's own label
    ax.text(2.5, .8033 + .004, "starting baseline 0.8033",
            fontsize=8, color=PALETTE["baseline"], ha="center")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(.68, .90)
    ax.set_ylabel("Mean PR-AUC")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_progression")


def fig_layer_ablation() -> None:
    """Tier 2 #6. Stage 1 -> 2 -> 3 per antibiotic."""
    fig, ax = plt.subplots(figsize=(10, 4.4))
    order = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
    x = np.arange(len(order))
    w = .26
    # explicit sequential ramp: FILL_SCALE would otherwise squeeze three shades
    # of one hue together until they were hard to tell apart
    ramp = ["#9FC8E3", "#3C8DC0", "#04456E"]
    for i, (stage, data) in enumerate(ABLATION.items()):
        vals = [data[a] for a in order]
        ax.bar(x + (i - 1) * w, vals, w, label=stage,
               color=ramp[i], edgecolor=dark(ramp[i], .28), linewidth=.9)
    # only the Stage 3 mean is drawn: the other two lines were unlabelled and
    # read as stray gridlines
    means = [np.mean(list(d.values())) for d in ABLATION.values()]
    ax.axhline(means[-1], color=PALETTE["random"], linewidth=.8,
               linestyle=(0, (4, 3)), alpha=.6)
    ax.text(len(order) - .5, means[-1] + .008, f"Stage 3 mean {means[-1]:.4f}",
            fontsize=8, ha="right", color=PALETTE["random"])
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylabel("PR-AUC (random split)")
    ax.set_ylim(.45, 1.06)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.15))
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_layer_ablation")


def fig_requirements_status() -> None:
    """Tier 2 #7. Compliance at a glance -- Chapter 3 currently has no figure."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 3.4),
                             gridspec_kw={"height_ratios": [1, 1]})
    for ax, statuses, title in [
        (axes[0], FR_STATUS, "Functional requirements"),
        (axes[1], NFR_STATUS, "Non-functional requirements"),
    ]:
        keys = list(statuses)
        for i, k in enumerate(keys):
            s = statuses[k]
            # these chips carry text inside, so they stay lighter than the
            # bar fills elsewhere; punch comes from a saturated border instead
            ax.add_patch(plt.Rectangle((i, 0), .88, 1,
                                       facecolor=light(PALETTE[s], 1.55),
                                       edgecolor=PALETTE[s], linewidth=1.8))
            ax.text(i + .44, .5, k, ha="center", va="center", fontsize=8.5,
                    color=dark(PALETTE[s], .45), fontweight="bold")
        ax.set_xlim(-.2, len(keys) + .1)
        ax.set_ylim(-.1, 1.1)
        ax.set_title(title, fontsize=10, loc="left")
        ax.axis("off")
    fig.legend(handles=[
        Patch(facecolor=light(PALETTE["met"], 1.55), edgecolor=PALETTE["met"],
              linewidth=1.8, label="Fully met"),
        Patch(facecolor=light(PALETTE["partial"], 1.55), edgecolor=PALETTE["partial"],
              linewidth=1.8, label="Partly met"),
        Patch(facecolor=light(PALETTE["unmet"], 1.55), edgecolor=PALETTE["unmet"],
              linewidth=1.8, label="Not met"),
    ], loc="lower center", ncol=3, bbox_to_anchor=(.5, -.10))
    fig.tight_layout()
    save(fig, "fig_requirements_status")


def fig_mechanism_gap() -> None:
    """Tier 3 #10. The closing figure: two problems, not one.

    Prevalence-corrected gap against the resistance mechanism, so the
    two-group result and its biological explanation sit in one picture.
    """
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    order = sorted(PREVINV, key=lambda k: PREVINV[k][5], reverse=True)
    gaps = [abs(PREVINV[a][5]) for a in order]
    x = np.arange(len(order))
    colours = [PALETTE["random"] if g < .1 else PALETTE["holdout"] for g in gaps]
    bars = ax.bar(x, gaps, .58, color=[light(c, .45) for c in colours],
                  edgecolor=[dark(c) for c in colours], linewidth=1.2)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.axvline(2.5, color="#333333", linestyle="--", linewidth=1.1)
    # Neutral group labels. The mechanistic reading is argued in the prose,
    # where it can be qualified per drug; it is not asserted on the figure.
    ax.text(1.0, .265, "small corrected gap",
            ha="center", fontsize=9, style="italic", color=PALETTE["random"])
    ax.text(4.5, .265, "large corrected gap",
            ha="center", fontsize=9, style="italic", color=PALETTE["holdout"])
    ax.set_xticks(x); ax.set_xticklabels(order)
    ax.set_ylim(0, .30)
    ax.set_ylabel("Prevalence-corrected generalisation gap")
    ax.set_xlabel("Antibiotic")
    # was sitting in the x-label slot, colliding with the tick labels
    ax.annotate("nothing falls between\n0.031 and 0.176",
                xy=(2.5, .105), xytext=(2.5, .105), ha="center", va="center",
                fontsize=8.5, style="italic", color="#2B2B2B",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#C8CFD3", linewidth=.8))
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "fig_mechanism_gap")


# =============================================================================
def main() -> None:
    apply_style()
    print("Regenerating thesis figures into images/ ...\n")
    print("Existing figures, recoloured:")
    for fn in (fig_label_overview, fig_population_structure, fig_split_composition,
               fig_split_prevalence_shift, fig_gate_weights, fig_ci_forest,
               fig_clinical_heatmap, fig_giessen_overview, fig_giessen_retention,
               fig_lineage_marker, fig_coresistance, fig_tabnet_vs_trilens):
        fn()
    print("\nNew figures:")
    for fn in (fig_baseline_ladder, fig_progression, fig_layer_ablation,
               fig_requirements_status, fig_mechanism_gap):
        fn()
    print("\nDone.")
    print("NOT regenerated here (need the real arrays, run from the project repo):")
    print("  fig_pr_curves, fig_gene_feature_space, fig_st_distribution,")
    print("  fig_attention_concentration, fig_threshold_sweep")
    print("Import PALETTE and apply_style() from this file so they match.")


if __name__ == "__main__":
    main()
