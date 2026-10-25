"""
make_results_figures.py
=======================
Post-hoc thesis figures (PR curves, calibration, gate weights, threshold sweep,
attention concentration, TabNet comparison). For the full figure set run
make_all_figures.py instead; this file is kept because fig_pr_curves and
fig_attention_concentration in the thesis were last drawn by it. Reads only saved .npz / .csv outputs -- never trains,
never touches a results directory except to read. Safe to run repeatedly.

Writes PDFs straight into the LaTeX images folder.

Every figure is independent and wrapped: a missing input skips that figure and
prints why, rather than killing the rest.

Usage:
    cd code
    python scripts/figures/make_results_figures.py
"""
from __future__ import annotations

import pickle
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[2]   # the code/ folder
RESULTS = ROOT / "results"
IMAGES = ROOT.parent / "thesis" / "images"

AB = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]

# Match the TikZ colours already defined in main.tex
BLUE = "#3465A4"
RED = "#CC0000"
ORANGE = "#E07C23"
# Darkened from #888888: the old grey was unreadable once these PDFs were
# scaled down to \textwidth in the thesis.
GREY = "#5f5e59"
# Pure black for every piece of figure text. The old near-black (#33322f) went
# muddy once these PDFs were scaled down into the thesis text block.
INK = "#000000"

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 9.5,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.edgecolor": "#93928a",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})


def _labels(pkl_name: str):
    p = ROOT / "data" / "processed" / pkl_name
    with open(p, "rb") as f:
        d = pickle.load(f)
    return d["Y"], d["label_mask"], list(d["label_names"])


def _load_run(run_dir: Path, pkl_name: str):
    """Return (probs, Y_test, mask_test, label_names) for a saved run."""
    npz = np.load(run_dir / "predictions_saved.npz", allow_pickle=True)
    key = "test_ens_probs" if "test_ens_probs" in npz else "test_probs"
    idx_key = "test_indices" if "test_indices" in npz else "test_idx"
    probs = npz[key]
    idx = npz[idx_key]
    Y, mask, names = _labels(pkl_name)
    return probs, Y[idx], mask[idx], names


def _clinical(y, p, thr):
    """VME = resistant called susceptible; ME = susceptible called resistant."""
    pred = (p >= thr).astype(int)
    R, S = (y == 1), (y == 0)
    vme = float((pred[R] == 0).sum() / max(R.sum(), 1))
    me = float((pred[S] == 1).sum() / max(S.sum(), 1))
    ca = float((pred == y).mean())
    return vme, me, ca


# ─────────────────────────────────────────────────────────────────────────────
def fig_pr_curves():
    """Precision-recall curves, random split vs MLST lineage holdout."""
    rnd = _load_run(RESULTS / "trilens_attnfix", "arcadia_v2_gene_filtered.pkl")
    hol = _load_run(RESULTS / "pillar_trilens_mlst",
                    "arcadia_v2_gene_filtered__mlst_holdout.pkl")

    # Taller panels and larger type: this figure is now placed at full
    # \textwidth, and at 9.5x4.6 the eight cells came out short and cramped.
    fig, axes = plt.subplots(2, 4, figsize=(9.0, 5.3), sharex=True, sharey=True)
    for k, ab in enumerate(AB):
        ax = axes.flat[k]
        for (probs, Yt, Mt, names), colour, lab in (
            (rnd, BLUE, "Random split"),
            (hol, RED, "Lineage holdout"),
        ):
            if ab not in names:
                continue
            i = names.index(ab)
            v = Mt[:, i].astype(bool)
            if v.sum() < 5 or Yt[v, i].sum() < 2:
                continue
            pr, rc, _ = precision_recall_curve(Yt[v, i], probs[v, i])
            ax.plot(rc, pr, color=colour, lw=1.6, label=lab)
            ax.axhline(Yt[v, i].mean(), color=colour, lw=0.7, ls=":", alpha=0.8)
        ax.set_title(ab, fontsize=12, fontweight="bold")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.tick_params(labelsize=10)
    # The eighth cell is empty (seven antibiotics in a 2x4 grid), so the legend
    # goes there instead of sitting on top of the AMP curve.
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False,
                         fontsize=11.5, handlelength=1.8)
    fig.supxlabel("Recall", fontsize=12)
    fig.supylabel("Precision", fontsize=12)
    fig.tight_layout()
    fig.savefig(IMAGES / "fig_pr_curves.pdf")
    plt.close(fig)
    print("  dotted lines = prevalence floor for that split")


# ─────────────────────────────────────────────────────────────────────────────
def fig_calibration():
    """Reliability diagram: are the predicted probabilities honest?"""
    probs, Yt, Mt, names = _load_run(RESULTS / "trilens_attnfix",
                                     "arcadia_v2_gene_filtered.pkl")
    edges = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.8, ls="--", label="Perfect")
    for i, ab in enumerate(names):
        if ab not in AB:
            continue
        v = Mt[:, i].astype(bool)
        p, y = probs[v, i], Yt[v, i]
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (p >= lo) & (p < hi)
            if m.sum() >= 10:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        if xs:
            ax.plot(xs, ys, marker="o", ms=2.5, lw=1.0, label=ab)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed resistant fraction")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    fig.savefig(IMAGES / "fig_calibration.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig_gate_weights():
    """How the mixture-of-experts gate splits its weight across the 3 lenses."""
    g = np.load(RESULTS / "trilens_attnfix" / "gate_weights_test.npz")["gate"]
    if g.ndim != 3:
        raise ValueError(f"unexpected gate shape {g.shape}")
    mean = g.mean(axis=0)                      # (7, 3)
    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    x = np.arange(len(AB))
    bot = np.zeros(len(AB))
    for j, (lens, colour) in enumerate(
        [("Lens A (SNP motifs)", BLUE),
         ("Lens B (presence/absence)", RED),
         ("Lens C (cross-antibiotic)", GREY)]
    ):
        ax.bar(x, mean[:, j], bottom=bot, width=0.62, label=lens,
               color=colour, edgecolor="white", linewidth=0.5)
        bot += mean[:, j]
    ax.set_xticks(x); ax.set_xticklabels(AB)
    ax.set_ylabel("Mean gate weight")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc="upper center",
              bbox_to_anchor=(0.5, 1.22), ncol=3)
    fig.savefig(IMAGES / "fig_gate_weights.pdf")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
def fig_threshold_sweep():
    """VME and ME against decision threshold, with the CLSI 3% line."""
    probs, Yt, Mt, names = _load_run(RESULTS / "trilens_attnfix",
                                     "arcadia_v2_gene_filtered.pkl")
    thrs = np.linspace(0.02, 0.98, 49)
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 4.4), sharex=True, sharey=True)
    for k, ab in enumerate(AB):
        ax = axes.flat[k]
        if ab not in names:
            continue
        i = names.index(ab)
        v = Mt[:, i].astype(bool)
        y, p = Yt[v, i], probs[v, i]
        vs = [_clinical(y, p, t) for t in thrs]
        ax.plot(thrs, [a[0] for a in vs], color=RED, lw=1.2, label="VME")
        ax.plot(thrs, [a[1] for a in vs], color=BLUE, lw=1.2, label="ME")
        ax.axhline(0.03, color=GREY, lw=0.7, ls="--")
        ax.axvline(0.5, color=GREY, lw=0.5, ls=":")
        ax.set_title(ab, fontsize=9)
        ax.set_ylim(0, 1)
    axes.flat[-1].axis("off")
    axes.flat[0].legend(frameon=False, fontsize=7)
    fig.supxlabel("Decision threshold", fontsize=9)
    fig.supylabel("Error rate", fontsize=9)
    fig.tight_layout()
    fig.savefig(IMAGES / "fig_threshold_sweep.pdf")
    plt.close(fig)
    print("  dashed line = CLSI 3% limit; dotted line = the fixed 0.5 threshold")


# ─────────────────────────────────────────────────────────────────────────────
def fig_attention_concentration():
    """Lens B attention against rank, before and after the strain-conditioning fix."""
    # Standard (un-conditioned) Lens B: the full re-run puts that ablation model in
    # results/standard/; the original run kept it in results/trilens_v10/.
    old_dir = next((d for d in (RESULTS / "standard" / "trilens_attnfix", RESULTS / "trilens_v10")
                    if (d / "attention_B_topgenes.csv").exists()), RESULTS / "trilens_v10")
    old = pd.read_csv(old_dir / "attention_B_topgenes.csv")
    new = pd.read_csv(RESULTS / "trilens_attnfix" / "attention_B_topgenes.csv")
    # Sized for a full \textwidth figure in the thesis, with thicker lines and
    # larger markers so nothing disappears once LaTeX scales the PDF down.
    fig, ax = plt.subplots(figsize=(8.6, 4.3))
    for df, colour, lab in ((old, GREY, "Original gated attention"),
                            (new, RED, "Strain-conditioned attention")):
        piv = df.pivot_table(index="rank", columns="ab", values="attn_weight")
        ax.plot(piv.index, piv.mean(axis=1), color=colour, lw=1.9, marker="o",
                ms=4.0, label=lab)
        ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1),
                        color=colour, alpha=0.16, linewidth=0)
    ax.axhline(1.0 / 32441, color=BLUE, lw=1.1, ls="--",
               label="Uniform weight (1/32,441)")
    ax.set_xlabel("Gene rank")
    ax.set_ylabel("Mean attention weight")
    # No title: nothing goes above a figure, the LaTeX caption carries it.
    ax.legend(frameon=False, fontsize=10.5, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(IMAGES / "fig_attention_concentration.pdf")
    plt.close(fig)
    print("  band = range across the seven antibiotics")


# ─────────────────────────────────────────────────────────────────────────────
def fig_tabnet_vs_trilens():
    """Figure 5.4. TabNet baseline against the proposed model, per antibiotic,
    random split.

    Rebuilt because the original generator was lost with the other PC. Both
    columns are read from result files, nothing is hard-coded:
      TabNet   results/tabnet_baseline/tabnet_results.csv
      TriLens  results/trilens_attnfix/ablation_results.csv (T6, random_split)

    Note this uses the DELIVERED attnfix TriLens run (average 0.8486), which is
    what Chapter 5 quotes. The PDF shipped before this rebuild was drawn from
    the older pre-attnfix column (average 0.8462), so the average gap moves from
    0.151 to 0.154 -- matching the text.
    """
    tab = pd.read_csv(RESULTS / "tabnet_baseline" / "tabnet_results.csv")
    tabnet = dict(zip(tab["antibiotic"], tab["pr_auc"]))

    # The gradient-boosting baseline this project started from: XGBoost on 1,024
    # PCA components, the `no_phylogroup` variant (n_features = 1024). Its seven
    # values average 0.80331, which is the 0.8033 quoted throughout Chapter 5.
    # The `with_phylogroup` rows are a different, 1,032-feature run and must not
    # be mixed in here.
    p1 = pd.read_csv(RESULTS / "pillar1" / "pillar1_combined.csv")
    p1 = p1[(p1["source"] == "baseline_xgb") & (p1["variant"] == "no_phylogroup")]
    xgb = dict(zip(p1["antibiotic"], p1["test_pr_auc"]))

    abl = pd.read_csv(RESULTS / "trilens_attnfix" / "ablation_results.csv")
    abl = abl[(abl["ablation"] == "T6") & (abl["split_key"] == "random_split")]
    trilens = dict(zip(abl["ab"], abl["prauc"]))

    order = [a for a in AB if a in tabnet and a in xgb and a in trilens]
    t_vals = [tabnet[a] for a in order]
    b_vals = [xgb[a] for a in order]
    x_vals = [trilens[a] for a in order]

    x = np.arange(len(order))
    w = 0.27
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.bar(x - w, t_vals, width=w * 0.94, color=GREY, label="TabNet",
           edgecolor="white", linewidth=0.5)
    ax.bar(x, b_vals, width=w * 0.94, color=ORANGE,
           label="XGBoost on PCA components (baseline)",
           edgecolor="white", linewidth=0.5)
    ax.bar(x + w, x_vals, width=w * 0.94, color=BLUE, label="TriLens-MoE-Stack",
           edgecolor="white", linewidth=0.5)

    # Gain over the gradient-boosting baseline, which is the comparison the
    # thesis leads with (0.849 against 0.803), not the gain over TabNet.
    for xi, (bv, xv) in enumerate(zip(b_vals, x_vals)):
        ax.text(xi + w, xv + 0.018, f"{xv - bv:+.3f}", ha="center",
                va="bottom", fontsize=9.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("PR-AUC (random split)")
    ax.set_ylim(0, 1.30)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", color="#d2d1c8", linewidth=0.8)
    ax.set_axisbelow(True)
    # One horizontal row inside the axes. The extra headroom above 1.0 keeps it
    # clear of the tallest bar (CIP) and of the printed gains.
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.0),
              ncol=3, handlelength=1.3, columnspacing=1.4, fontsize=10,
              labelcolor=INK)
    fig.tight_layout()
    fig.savefig(IMAGES / "fig_tabnet_vs_trilens.pdf")
    plt.close(fig)
    print(f"  TabNet avg {np.mean(t_vals):.4f} | XGB-PCA avg {np.mean(b_vals):.4f} "
          f"| TriLens avg {np.mean(x_vals):.4f}")
    print(f"  gain over XGB-PCA {np.mean(x_vals) - np.mean(b_vals):+.4f} | "
          f"over TabNet {np.mean(x_vals) - np.mean(t_vals):+.4f}")


# ─────────────────────────────────────────────────────────────────────────────
FIGURES = [
    ("fig_pr_curves",              fig_pr_curves),
    ("fig_calibration",            fig_calibration),
    ("fig_gate_weights",           fig_gate_weights),
    ("fig_threshold_sweep",        fig_threshold_sweep),
    ("fig_attention_concentration", fig_attention_concentration),
    ("fig_tabnet_vs_trilens",      fig_tabnet_vs_trilens),
]


def main() -> None:
    import sys
    IMAGES.mkdir(parents=True, exist_ok=True)
    print(f"Writing to {IMAGES}\n")
    wanted = set(sys.argv[1:])          # optional: only these figure names
    todo = [(n, f) for n, f in FIGURES if not wanted or n in wanted]
    ok = 0
    for name, fn in todo:
        try:
            fn()
            print(f"[ok]      {name}.pdf")
            ok += 1
        except FileNotFoundError as exc:
            print(f"[skipped] {name} -- missing input: {exc}")
        except Exception:
            print(f"[FAILED]  {name}")
            traceback.print_exc(limit=3)
    print(f"\n{ok}/{len(todo)} figures written.")


if __name__ == "__main__":
    main()
