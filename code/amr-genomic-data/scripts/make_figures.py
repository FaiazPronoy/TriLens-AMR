#!/usr/bin/env python
"""Generate every thesis Dataset/Preprocessing figure from real, already-
processed artifacts (no raw VCF/PLINK re-processing here -- see
`scripts/build_snp_matrix.py` etc. for that).

Running this end-to-end also doubles as a pipeline smoke test: every figure
depends on `src.amr_data.config`/`io`/`splits`/`gene_filter`/`giessen`
successfully loading and processing the real production pkl/csv files, so a
clean run here confirms that code path still works, not just that the
numbers are right.

Usage:
    python scripts/make_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.amr_data import viz  # noqa: E402
from src.amr_data.config import load_paths  # noqa: E402
from src.amr_data.gene_filter import build_gene_to_snp_ranges, get_gene_size_stats  # noqa: E402
from src.amr_data.giessen import build_giessen_split, load_giessen_raw  # noqa: E402
from src.amr_data.io import load_pickle  # noqa: E402
from src.amr_data.splits import (  # noqa: E402
    groups_from_phylogroup,
    lineage_holdout_split,
    phylogroup_stratified_split,
    split_diagnostics,
)

OUT_DIR_NAME = "figures"

SPLIT_DISPLAY_NAMES = {
    "random_split": "Random split\n(no holdout)",
    "phylogroup_stratified_split": "Phylogroup-stratified\n(no holdout)",
    "lineage_holdout_split_8phylogroup": "Lineage holdout\n(8 phylogroups)",
    "lineage_holdout_split_30cluster_REAL": "Lineage holdout\n(30 real clusters)",
}


def _save(fig, out_dir: Path, name: str) -> None:
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    print(f"  wrote {pdf_path.relative_to(out_dir.parent.parent)} (+ .png)")


def main() -> None:
    paths = load_paths()
    out_dir = paths.results_dir / OUT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading arcadia_v2_master_features.pkl ...")
    mf = load_pickle(paths.processed("master_features_pkl"))
    Y = mf["Y"].astype(float)
    label_mask = mf["label_mask"]
    label_names = list(mf["label_names"])
    cluster_labels = mf["cluster_labels"]
    phylogroup_to_int = mf["phylogroup_to_int"]
    int_to_phylogroup = {v: k for k, v in phylogroup_to_int.items()}
    n_samples = mf["n_samples"]

    prevalence = {
        ab: float(Y[label_mask[:, j], j].mean())
        for j, ab in enumerate(label_names)
    }
    coverage = {ab: float(label_mask[:, j].mean()) for j, ab in enumerate(label_names)}

    print("[1/9] fig_label_overview ...")
    fig1 = viz.fig_label_overview(prevalence, coverage, n_samples)
    _save(fig1, out_dir, "fig_label_overview")

    phylogroup_counts: dict[str, int] = {}
    for code in np.unique(cluster_labels):
        phylogroup_counts[int_to_phylogroup[int(code)]] = int((cluster_labels == code).sum())

    print("[2/9] fig_population_structure ...")
    fig2 = viz.fig_population_structure(phylogroup_counts)
    _save(fig2, out_dir, "fig_population_structure")

    print("[3/9] fig_coresistance ...")
    fig2b = viz.fig_coresistance(Y, label_mask, label_names)
    _save(fig2b, out_dir, "fig_coresistance")

    print("Recomputing split regimes (random_split read from the locked pkl; "
          "phylogroup_stratified_split and both lineage_holdout variants "
          "recomputed fresh via src.amr_data.splits to exercise that code "
          "for real) ...")
    random_split = mf["random_split"]
    phylo_split = phylogroup_stratified_split(cluster_labels)
    holdout_8 = lineage_holdout_split(groups_from_phylogroup(cluster_labels))

    ported = load_pickle(paths.processed("ported_cluster_labels_pkl"))
    assert bool(ported["matched_mask"].all()), (
        "ported_cluster_labels has unmatched (-1) entries -- this script assumes "
        "100% match (true as of docs/CHANGELOG_NUMBERS.md); filter to matched_mask "
        "before calling lineage_holdout_split if that ever changes."
    )
    holdout_30 = lineage_holdout_split(ported["ported_cluster_labels"])

    # NOTE: fig_st_distribution is deliberately NOT built here. This array is
    # the 30-cluster grouping behind lineage_holdout_split_30cluster_REAL, not
    # the multi-locus sequence types the thesis reports (458 types, 6,487 typed,
    # largest 884). The figure is built by Thesis/make_st_figure.py from
    # data/processed/mlst_lineage_holdout_split.pkl instead.

    splits = {
        "random_split": random_split,
        "phylogroup_stratified_split": phylo_split,
        "lineage_holdout_split_8phylogroup": holdout_8,
        "lineage_holdout_split_30cluster_REAL": holdout_30,
    }
    split_sizes = {name: {k: len(v) for k, v in s.items()} for name, s in splits.items()}

    # The thesis reports three splits in Table 4.6 (tab:splits): random,
    # phylogroup-stratified and the sequence-type lineage holdout.
    # `lineage_holdout_split_8phylogroup` is an earlier variant the thesis does
    # not report, so it is left out of the two split figures to keep them in
    # lockstep with that table. It stays in `split_diagnostics_recomputed.csv`.
    REPORTED_SPLITS = [
        "random_split",
        "phylogroup_stratified_split",
        "lineage_holdout_split_30cluster_REAL",
    ]
    split_sizes_reported = {k: split_sizes[k] for k in REPORTED_SPLITS}
    print("  sizes going into fig_split_composition "
          "(these must match Table 4.6 in the thesis):")
    for k in REPORTED_SPLITS:
        s = split_sizes_reported[k]
        print(f"    {k}: train={s['train_idx']:,} "
              f"val={s['val_idx']:,} test={s['test_idx']:,}")

    print("[4/9] fig_split_composition ...")
    fig3 = viz.fig_split_composition(split_sizes_reported, SPLIT_DISPLAY_NAMES)
    _save(fig3, out_dir, "fig_split_composition")

    diagnostics = split_diagnostics(splits, Y, label_mask, label_names)
    diag_out = paths.results_dir / "splits" / "split_diagnostics_recomputed.csv"
    diag_out.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.to_csv(diag_out, index=False)
    print(f"  wrote {diag_out} (cross-check against results/splits/split_diagnostics.csv)")

    print("[5/9] fig_split_prevalence_shift ...")
    # Same three reported splits as the composition figure; the CSV above keeps
    # all four regimes for cross-checking.
    diagnostics_reported = diagnostics[diagnostics["split"].isin(REPORTED_SPLITS)]
    fig4 = viz.fig_split_prevalence_shift(diagnostics_reported, label_names, SPLIT_DISPLAY_NAMES)
    _save(fig4, out_dir, "fig_split_prevalence_shift")

    print("Loading arcadia_v2_snp_matrix.pkl (full, pre-Stage-1.5) ...")
    snp = load_pickle(paths.processed("snp_matrix_pkl"))
    gene_to_indices_before = build_gene_to_snp_ranges(snp["snp_positions"])
    sizes_before = np.array([len(v) for v in gene_to_indices_before.values()])
    stats_before = get_gene_size_stats(gene_to_indices_before)
    print(f"  before filter: {stats_before['n_genes']} genes, "
          f"{stats_before['n_with_at_least_5']} pass the >=5-SNP variability filter")
    del snp  # 6962 x 535439 int8 -- free before loading the next big pkl

    print("Loading arcadia_v2_gene_filtered.pkl (post-Stage-1.5) ...")
    gfl = load_pickle(paths.processed("gene_filtered_legacy_pkl"))
    sizes_after = np.array([end - start for start, end in gfl["gene_to_range"].values()])
    assert sizes_after.sum() == gfl["n_snps_filtered"], (
        "recomputed post-filter gene sizes do not sum to n_snps_filtered -- "
        "gene_to_range's tuple convention may have changed."
    )
    min_snps_per_gene = gfl["filter_metadata"]["min_snps_per_gene"]
    del gfl

    print("Loading arcadia_v2_gene_presence_absence.pkl ...")
    gene_pa = load_pickle(paths.processed("gene_pa_pkl"))
    gene_prevalence = gene_pa["X_pa"].mean(axis=0).astype(np.float64)
    per_strain_gene_count = gene_pa["X_pa"].sum(axis=1).astype(np.int64)
    # NOTE: gene_prevalence_stats / per_strain_gene_count_stats below are the
    # pkl's OWN pre-stored summary (predates this refactor) -- a fresh
    # recompute from X_pa above does not reproduce them exactly (checked:
    # off by ~0.1-0.3%, even after restricting to matched_mask), so the
    # figure's printed annotations cite the locked stored values directly
    # rather than silently shipping a slightly different number (Rule 1).
    gene_prevalence_stats = gene_pa["gene_prevalence_stats"]
    per_strain_gene_count_stats = gene_pa["per_strain_gene_count_stats"]
    del gene_pa

    print("[6/9] fig_gene_feature_space ...")
    fig5 = viz.fig_gene_feature_space(
        sizes_before, sizes_after, gene_prevalence, per_strain_gene_count, min_snps_per_gene,
        gene_prevalence_stats, per_strain_gene_count_stats,
    )
    _save(fig5, out_dir, "fig_gene_feature_space")

    print("Loading Giessen external validation set ...")
    giessen = load_giessen_raw(paths)
    giessen_prevalence = {
        ab: float(np.nanmean(giessen["Y"][:, i]))
        for i, ab in enumerate(giessen["label_names"])
    }
    # build_giessen_split() is this refactor's own 70/15/15 stratified split and
    # returns 566/121/122. That is NOT the split behind the reported Giessen
    # numbers: results/pillar4_giessen_transfer/SUMMARY.md records n_test = 81
    # per antibiotic, i.e. the 647/81/81 split, and Table 5.16 and the
    # confidence intervals in Chapter 5 all come from that run. The figure must
    # show the split that produced the reported results, so the sizes are taken
    # from the run rather than recomputed here.
    giessen_split = build_giessen_split(giessen)
    _refactor_sizes = {k: len(giessen_split[k]) for k in ("train_idx", "val_idx", "test_idx")}
    giessen_split_sizes = {"train_idx": 647, "val_idx": 81, "test_idx": 81}
    if _refactor_sizes != giessen_split_sizes:
        print(f"  note: figure uses the reported pillar4 split {giessen_split_sizes}, "
              f"not build_giessen_split()'s {_refactor_sizes}")

    print("[7/9] fig_giessen_overview ...")
    fig6 = viz.fig_giessen_overview(giessen_prevalence, giessen_split_sizes)
    _save(fig6, out_dir, "fig_giessen_overview")

    # ------------------------------------------------------------------
    # Results-side figures.
    #
    # The numbers below are transcribed from the two locked tables in
    # Chapter 5 -- tab:holdoutclin (clinical errors, random split against
    # lineage holdout) and tab:ci (bootstrap intervals on the random split).
    # They are hardcoded rather than re-read from a CSV on purpose: the point
    # of these two figures is to display exactly what the thesis tables state,
    # so a silent drift between a regenerated CSV and the typeset table would
    # be worse than no figure at all. If either table is edited, edit here too.
    # The no-skill baselines are NOT hardcoded -- they are recomputed from the
    # locked pkl below, because that is the one quantity available here.
    # ------------------------------------------------------------------
    clinical = pd.DataFrame([
        # antibiotic, VME random, VME holdout, CA random, CA holdout, ME holdout
        ("CIP", 0.086, 0.114, 0.969, 0.978, 0.006),
        ("AMP", 0.109, 0.160, 0.893, 0.877, 0.095),
        ("SXT", 0.236, 0.310, 0.871, 0.863, 0.072),
        ("AMC", 0.394, 0.710, 0.804, 0.754, 0.076),
        ("GEN", 0.436, 0.868, 0.934, 0.962, 0.002),
        ("CAZ", 0.291, 0.837, 0.936, 0.959, 0.001),
        ("CTX", 0.190, 0.915, 0.953, 0.918, 0.001),
    ], columns=["antibiotic", "vme_random", "vme_holdout",
                "ca_random", "ca_holdout", "me_holdout"])
    assert abs(clinical["vme_random"].mean() - 0.249) < 5e-4, (
        "hardcoded VME random column no longer averages to the 0.249 printed in "
        "tab:holdoutclin -- the table and this script have drifted apart.")
    assert abs(clinical["vme_holdout"].mean() - 0.559) < 5e-4, (
        "hardcoded VME holdout column no longer averages to the 0.559 printed in "
        "tab:holdoutclin -- the table and this script have drifted apart.")

    print("[8/9] fig_clinical_heatmap ...")
    fig7 = viz.fig_clinical_heatmap(clinical)
    _save(fig7, out_dir, "fig_clinical_heatmap")

    test_idx = random_split.get("test_idx", random_split.get("test"))
    assert test_idx is not None, (
        f"random_split has keys {sorted(random_split)} -- neither 'test_idx' nor "
        "'test'; cannot recompute the no-skill baselines for fig_ci_forest.")
    test_prevalence = {}
    for j, ab in enumerate(label_names):
        m = label_mask[test_idx, j]
        test_prevalence[ab] = float(Y[test_idx, j][m].mean())

    ci = pd.DataFrame([
        # antibiotic, PR-AUC, CI low, CI high  (tab:ci)
        ("AMP", 0.9440, 0.919, 0.965),
        ("CIP", 0.9773, 0.961, 0.990),
        ("AMC", 0.7482, 0.680, 0.809),
        ("SXT", 0.8352, 0.773, 0.891),
        ("GEN", 0.7037, 0.612, 0.786),
        ("CTX", 0.9019, 0.851, 0.943),
        ("CAZ", 0.8299, 0.762, 0.882),
    ], columns=["antibiotic", "pr_auc", "ci_lo", "ci_hi"])
    assert abs(ci["pr_auc"].mean() - 0.8486) < 5e-5, (
        "hardcoded PR-AUC column no longer averages to the 0.8486 printed in "
        "tab:ci -- the table and this script have drifted apart.")
    missing = set(ci["antibiotic"]) - set(test_prevalence)
    assert not missing, (
        f"no test-set prevalence available for {sorted(missing)}; label_names are "
        f"{label_names} -- the abbreviations in this script do not match the pkl.")
    ci["prevalence"] = ci["antibiotic"].map(test_prevalence)
    assert (ci["pr_auc"] > ci["prevalence"]).all(), (
        "at least one antibiotic scores at or below its own no-skill baseline; "
        "check that test_idx is the RANDOM split and not a holdout before "
        "publishing this figure.")

    print("[9/10] fig_ci_forest ...")
    fig8 = viz.fig_ci_forest(ci)
    _save(fig8, out_dir, "fig_ci_forest")

    # Transcribed from tab:umark in Chapter 5, same reasoning as the two tables
    # above: the figure must show exactly what the table states. The two
    # attention rows are the same lens before and after strain conditioning.
    # U_res has no entry for the random baseline -- the table does not report
    # one, so neither does this.
    u_st = {
        "Random genes": 0.539,
        "SHAP, classical": 0.375,
        "Attention, standard": 0.312,
        "Attention, strain-cond.": 0.357,
    }
    u_res = {
        "SHAP, classical": 0.0473,
        "Attention, standard": 0.0192,
        "Attention, strain-cond.": 0.0419,
    }
    assert u_st["Attention, standard"] < u_st["Random genes"], (
        "the attended genes are no longer LESS lineage-determined than random; "
        "Chapter 5 argues they are, so re-check tab:umark before publishing.")
    assert u_res["Attention, standard"] < u_res["SHAP, classical"], (
        "attention now carries MORE resistance information than SHAP; that "
        "reverses the finding in Chapter 5.")

    print("[10/10] fig_lineage_marker ...")
    fig9 = viz.fig_lineage_marker(u_st, u_res)
    _save(fig9, out_dir, "fig_lineage_marker")

    print(f"\nAll figures written to {out_dir}")
    print("Pipeline smoke test: every real artifact above (master_features, "
          "ported_cluster_labels, snp_matrix, gene_filtered_legacy, gene_pa, "
          "Giessen) loaded and processed without error.")


if __name__ == "__main__":
    main()
