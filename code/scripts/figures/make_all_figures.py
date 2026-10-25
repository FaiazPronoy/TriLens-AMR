#!/usr/bin/env python
r"""
ONE COMMAND -> every thesis figure, in one palette, straight into images/.

    cd code
    python scripts/figures/make_all_figures.py

This is the only script you need to run. It does two things:

  1. Calls make_thesis_figures.main(), which rebuilds the 17 figures whose data
     is already locked in a thesis table.
  2. Rebuilds the remaining figures from the real arrays in code/data and
     code/results, using the same PALETTE and apply_style().

Nothing else has to be run, and nothing has to be copied by hand.

--------------------------------------------------------------------------
MEMORY NOTE
--------------------------------------------------------------------------
fig_gene_feature_space loads arcadia_v2_snp_matrix.pkl, which contains a
6,962 x 535,439 int8 array (~3.7 GB). It is deleted immediately after the SNP
positions are read. If the machine is short on RAM, run with
    --skip-genespace
and that one figure keeps its existing file.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# the presentation layer: palette, style, save(), and the table-derived figures
import make_thesis_figures as base
from make_thesis_figures import PALETTE, apply_style, dark, light, save

# --------------------------------------------------------------------------
THESIS_REPO = Path(__file__).resolve().parents[2]   # the code/ folder
# --------------------------------------------------------------------------

PROC = THESIS_REPO / "data" / "processed"
RES = THESIS_REPO / "results"

# Delivered model, random split (strain-conditioned -- avg 0.8486).
PRED_RANDOM = RES / "trilens_attnfix" / "predictions_saved.npz"
# Sequence-type lineage holdout. See the note in fig_pr_curves().
PRED_HOLDOUT = RES / "pillar_mlst_final_comparison" / "predictions_saved.npz"

MASTER = PROC / "arcadia_v2_master_features.pkl"
SNPMAT = PROC / "arcadia_v2_snp_matrix.pkl"
GENEPA = PROC / "arcadia_v2_gene_presence_absence.pkl"
GENEFILT = PROC / "arcadia_v2_gene_filtered.pkl"
MLSTSPLIT = PROC / "mlst_lineage_holdout_split.pkl"

AB_ORDER = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]


def load_pickle(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _labels():
    """Y, label_mask and the antibiotic names, in the file's own order."""
    mf = load_pickle(MASTER)
    return (np.asarray(mf["Y"]),
            np.asarray(mf["label_mask"]),
            [str(x) for x in mf["label_names"]])


# ===========================================================================
# fig_pr_curves
# ===========================================================================
def fig_pr_curves() -> None:
    """Precision-recall curves, random split against the lineage holdout.

    NOT RUN BY DEFAULT. Pass --rebuild-prcurves to force it.

    The blue (random-split) curves are fine: trilens_attnfix is present here.
    The red (holdout) curves are the problem. The delivered strain-conditioned
    holdout run lives on another machine, so the only holdout predictions in
    THIS repo are pillar_mlst_final_comparison, which is the standard
    formulation and averages 0.6300 rather than the 0.6337 reported in
    Table 5.9. The figure already in images/ was built on the other machine
    from the correct arrays, so rebuilding here would replace a correct figure
    with one drawn from a different run. Leave it alone unless you have copied
    results/trilens_attnfix's MLST holdout directory across, in which case
    point PRED_HOLDOUT at it and pass the flag.
    """
    from sklearn.metrics import precision_recall_curve

    Y, mask, names = _labels()
    out = {}
    for tag, path in (("random", PRED_RANDOM), ("holdout", PRED_HOLDOUT)):
        with np.load(path, allow_pickle=True) as z:
            out[tag] = (np.asarray(z["test_ens_probs"], dtype=float),
                        np.asarray(z["test_indices"], dtype=int))

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.4), sharex=True, sharey=True)
    axes = axes.ravel()
    for k, ab in enumerate(AB_ORDER):
        ax = axes[k]
        j = names.index(ab)
        for tag, colour in (("random", PALETTE["random"]),
                            ("holdout", PALETTE["holdout"])):
            probs, idx = out[tag]
            m = mask[idx, j]
            y = Y[idx, j][m].astype(int)
            p = probs[m, j]
            if y.sum() == 0:
                continue
            prec, rec, _ = precision_recall_curve(y, p)
            ax.plot(rec, prec, color=colour, linewidth=1.7,
                    label="Random split" if tag == "random" else "Lineage holdout")
            ax.axhline(y.mean(), color=colour, linestyle=":", linewidth=1.1)
        ax.set_title(ab, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.set_axisbelow(True)
    axes[7].axis("off")
    axes[7].legend(*axes[0].get_legend_handles_labels(), loc="center", frameon=False)
    fig.supxlabel("Recall")
    fig.supylabel("Precision")
    fig.tight_layout()
    save(fig, "fig_pr_curves")


# ===========================================================================
# fig_threshold_sweep
# ===========================================================================
def fig_threshold_sweep() -> None:
    """Mean VME and ME against the decision threshold, with the CLSI limit.

    Shows directly that no threshold puts both error rates inside 0.03: the
    two curves cross well above the limit.
    """
    Y, mask, names = _labels()
    with np.load(PRED_RANDOM, allow_pickle=True) as z:
        probs = np.asarray(z["test_ens_probs"], dtype=float)
        idx = np.asarray(z["test_indices"], dtype=int)

    ts = np.linspace(0.02, 0.98, 97)
    vme, me = [], []
    for t in ts:
        v, m_ = [], []
        for ab in AB_ORDER:
            j = names.index(ab)
            sel = mask[idx, j]
            y = Y[idx, j][sel].astype(int)
            pred = (probs[sel, j] > t).astype(int)
            if y.sum():
                v.append(((pred == 0) & (y == 1)).sum() / y.sum())
            if (y == 0).sum():
                m_.append(((pred == 1) & (y == 0)).sum() / (y == 0).sum())
        vme.append(np.mean(v))
        me.append(np.mean(m_))
    vme, me = np.array(vme), np.array(me)

    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.plot(ts, vme, color=PALETTE["holdout"], linewidth=2.2,
            label="Very major error (resistant called susceptible)")
    ax.plot(ts, me, color=PALETTE["random"], linewidth=2.2,
            label="Major error (susceptible called resistant)")
    ax.axhline(0.03, color="#2B2B2B", linestyle="--", linewidth=1.3)
    ax.text(0.985, 0.045, "CLSI limit 0.03", ha="right", fontsize=9, color="#2B2B2B")
    ax.axvline(0.5, color=PALETTE["neutral"], linestyle=":", linewidth=1.3)
    i50 = int(np.argmin(np.abs(ts - 0.5)))
    ax.plot(0.5, vme[i50], "o", color=dark(PALETTE["holdout"]), markersize=7, zorder=5)
    ax.plot(0.5, me[i50], "o", color=dark(PALETTE["random"]), markersize=7, zorder=5)
    ax.text(0.5, vme[i50] + .045, f"  operating point\n  VME {vme[i50]:.3f}",
            fontsize=8.5, color=dark(PALETTE["holdout"]))
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Mean error rate across the seven antibiotics")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.15), ncol=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_threshold_sweep")


# ===========================================================================
# fig_st_distribution
# ===========================================================================
def fig_st_distribution() -> None:
    sp = load_pickle(MLSTSPLIT)
    st = np.asarray(sp["st_per_genome"], dtype=object)
    typed = np.array([s is not None and str(s) not in ("", "None", "nan") for s in st])
    vals, counts = np.unique(np.array([str(s) for s in st[typed]]), return_counts=True)
    counts = np.sort(counts)[::-1]

    diag = sp.get("diag_meta", {}) or {}
    thr = diag.get("big_st_threshold")
    big = diag.get("big_sts", []) or []
    n_typed, n_types = int(typed.sum()), len(vals)
    # NOTE: the pipeline's own threshold is 5% of ALL genomes (348 of 6,962),
    # not 5% of the typed subset (324 of 6,487). The label below must say so.
    if thr is None:
        thr = 0.05 * len(st)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    x = np.arange(1, len(counts) + 1)
    ax.plot(x, counts, color=PALETTE["random"], linewidth=2.0)
    ax.set_yscale("log")
    above = counts[counts > thr]
    ax.plot(np.arange(1, len(above) + 1), above, "o",
            color=PALETTE["holdout"], markersize=7, zorder=4)
    ax.axhline(thr, color="#2B2B2B", linestyle="--", linewidth=1.2)
    # right-aligned at the far edge; centred it collided with the stats block
    ax.text(len(counts) * .99, thr * 1.22,
            f"five per cent of the collection ({int(round(thr))} genomes)",
            fontsize=9, color="#2B2B2B", ha="right")
    # parked well below the rule, where the curve has already decayed
    ax.text(len(counts) * .52, thr * .30,
            f"{n_types} sequence types\n"
            f"{n_typed:,} of {len(st):,} strains typed\n"
            f"largest holds {counts.max():,}\n"
            f"{len(above)} above the line, held in training"
            + (f"\n({', '.join('ST' + str(b) for b in big)})" if big else ""),
            fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#C8CFD3", linewidth=.8))
    ax.set_xlabel("Sequence type, ranked by size")
    ax.set_ylabel("Strains in that type (log scale)")
    ax.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_st_distribution")
    if big:
        print(f"      largest types held in training: {', '.join(map(str, big))}")


# ===========================================================================
# fig_gene_feature_space
# ===========================================================================
def _genes_from_positions(pos) -> np.ndarray:
    """'LMHECDEF_00004:12:A>T' -> 'LMHECDEF_00004'."""
    return np.array([str(p).split(":", 1)[0] for p in pos])


def _sizes_after() -> np.ndarray | None:
    """SNPs per gene after the two filters. Several fallbacks; the pickle has
    been seen to fail to load on some machines."""
    try:
        gf = load_pickle(GENEFILT)
        if "gene_to_range" in gf:
            out = np.array([e - s for s, e in gf["gene_to_range"].values()])
            del gf
            return out
    except Exception as exc:
        print(f"      gene_filtered.pkl unreadable ({exc}); trying fallbacks")
    try:
        gi = load_pickle(RES / "stage1_5" / "gene_to_snp_indices.pkl")
        if isinstance(gi, dict):
            return np.array([len(v) for v in gi.values()])
    except Exception:
        pass
    try:
        js = json.loads((RES / "trilens_attnfix" / "pergene_snp_columns.json")
                        .read_text(encoding="utf-8"))
        if isinstance(js, dict):
            return np.array([len(v) for v in js.values()])
    except Exception:
        pass
    return None


def fig_gene_feature_space() -> None:
    print("      loading the 3.7 GB SNP matrix (positions only are kept) ...")
    snp = load_pickle(SNPMAT)
    genes = _genes_from_positions(snp["snp_positions"])
    del snp
    _, sizes_before = np.unique(genes, return_counts=True)
    del genes
    sizes_after = _sizes_after()

    pa = load_pickle(GENEPA)
    X = pa["X_pa"]
    prevalence = np.asarray(X).mean(axis=0, dtype=np.float64)
    per_strain = np.asarray(X).sum(axis=1, dtype=np.int64)
    # The pkl's OWN stored summaries, not a fresh recompute. Recomputing from
    # X_pa drifts by ~0.1-0.3% (17,802 vs 17,762 rare; median 3,331 vs 3,349)
    # and would silently disagree with the caption in Chapter 4. The original
    # pipeline cited the stored values for exactly this reason.
    gstats = pa.get("gene_prevalence_stats", {}) or {}
    sstats = pa.get("per_strain_gene_count_stats", {}) or {}
    del pa, X

    # RESOLVED. Chapter 4 originally mixed two populations: median 3,349, min
    # 2,141 and max 8,333 from this stored block (which excludes the 13 strains
    # with no PLINK record), but mean 3,709 from an all-rows recompute that
    # counts those 13 as zero. The text now quotes 3,727 so all four describe
    # the same 6,949 strains. Guard against silent drift:
    assert int(sstats.get("median", 0)) == 3349, "stored median moved"
    assert int(sstats.get("min", 0)) == 2141, "stored min moved"
    print(f"      per-strain gene count (stored, {len(per_strain) - 13:,} strains): "
          f"median {int(sstats['median']):,}, mean {sstats['mean']:.0f}, "
          f"range {int(sstats['min']):,}-{int(sstats['max']):,}  [matches Chapter 4]")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    bins = np.logspace(0, np.log10(max(sizes_before.max(), 2)), 46)
    axes[0].hist(sizes_before, bins=bins, color=light(PALETTE["random"], .8),
                 edgecolor=dark(PALETTE["random"]), linewidth=.5,
                 label=f"Before filter ({len(sizes_before):,})")
    if sizes_after is not None:
        axes[0].hist(sizes_after, bins=bins, color=light(PALETTE["holdout"], .5),
                     edgecolor=dark(PALETTE["holdout"]), linewidth=.5,
                     label=f"After filter ({len(sizes_after):,})")
    axes[0].axvline(5, color="#2B2B2B", linestyle="--", linewidth=1.2)
    axes[0].text(5.4, axes[0].get_ylim()[1] * .93, "min=5", fontsize=9)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("SNPs per gene (log scale)")
    axes[0].set_ylabel("Genes")
    axes[0].legend(fontsize=8.5)

    axes[1].hist(prevalence * 100, bins=60, color=light(PALETTE["random"], .8),
                 edgecolor=dark(PALETTE["random"]), linewidth=.4)
    axes[1].set_yscale("log")
    rare = int(gstats.get("rare_lt05", (prevalence < .05).sum()))
    core = int(gstats.get("core_gt95", (prevalence > .95).sum()))
    shell = int(gstats.get("shell", len(prevalence) - rare - core))
    axes[1].axvline(5, color="#2B2B2B", linestyle="--", linewidth=1.1)
    axes[1].axvline(95, color="#2B2B2B", linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("% of strains carrying the gene")
    axes[1].set_ylabel("Genes (log scale)")
    axes[1].set_title(f"rare (<5%): {rare:,}   shell: {shell:,}"
                      f"   core (>95%): {core:,}", fontsize=9)
    assert rare + shell + core == 32441, "gene classes must sum to 32,441"

    axes[2].hist(per_strain, bins=60, color=light(PALETTE["random"], .8),
                 edgecolor=dark(PALETTE["random"]), linewidth=.4)
    # Median, mean, min and max all come from the pickle's stored block, which
    # excludes the 13 strains with no gene-presence record. Chapter 4 now quotes
    # the same block throughout (mean 3,727, not the all-rows 3,709, which was
    # inconsistent with the 2,141 lower bound printed beside it).
    med = float(sstats.get("median", np.median(per_strain)))
    mean = float(sstats.get("mean", np.mean(per_strain)))
    axes[2].axvline(med, color="#2B2B2B", linestyle="--", linewidth=1.2)
    axes[2].text(med * 1.04, axes[2].get_ylim()[1] * .93,
                 f"median={int(round(med)):,}\nmean={int(round(mean)):,}", fontsize=9)
    axes[2].set_xlabel("Genes present (per genome)")
    axes[2].set_ylabel("Genomes")

    for a in axes:
        a.set_axisbelow(True)
    fig.tight_layout()
    save(fig, "fig_gene_feature_space")


# ===========================================================================
def main() -> None:
    skip_genespace = "--skip-genespace" in sys.argv

    if not THESIS_REPO.exists():
        sys.exit(f"THESIS_REPO not found: {THESIS_REPO}\nEdit the constant at the top.")

    # 1. everything that comes from a thesis table
    base.main()

    # 2. everything that needs the real arrays
    apply_style()
    print("\nFrom the project repo:")
    jobs = [("fig_threshold_sweep", fig_threshold_sweep),
            ("fig_st_distribution", fig_st_distribution)]
    if not skip_genespace:
        jobs.append(("fig_gene_feature_space", fig_gene_feature_space))
    # opt-in only -- see the docstring of fig_pr_curves for why
    if "--rebuild-prcurves" in sys.argv:
        jobs.insert(0, ("fig_pr_curves", fig_pr_curves))

    failed = []
    for name, fn in jobs:
        try:
            fn()
        except Exception as exc:  # keep going; one bad file must not stop the rest
            failed.append((name, exc))
            print(f"  SKIPPED {name}: {exc}")

    print("\n" + "=" * 70)
    if failed:
        print("These kept their existing file:")
        for name, exc in failed:
            print(f"  {name}: {exc}")
    print("KEPT AS THEY ARE (both already correct, built from the attnfix run")
    print("on the other machine -- do not overwrite them from this repo):")
    print("  fig_pr_curves               holdout predictions here are the")
    print("                              standard formulation (0.6300), not the")
    print("                              delivered 0.6337 your tables report")
    print("  fig_attention_concentration the pre-fix attention array needed for")
    print("                              its comparison curve is not in this repo")
    print()
    print("These two stay on the older, paler palette. To bring them into line,")
    print("copy from the other machine into this repo and re-run:")
    print("  results/<attnfix MLST holdout dir>/predictions_saved.npz")
    print("  the pre-fix Lens B attention array (a_B over 32,441 genes)")
    print("then run again with --rebuild-prcurves.")
    print("=" * 70)


if __name__ == "__main__":
    main()
