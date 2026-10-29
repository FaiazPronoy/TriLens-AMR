from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from trilens.data.gene_selection_rules import (
    build_filtered_snp_matrix,
    build_gene_to_snp_ranges,
    filter_genes,
    get_gene_size_stats,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
IMPORTANCE_DIR = ROOT / "results" / "gene_ranking_phylogroup"
OUT_DIR = IMPORTANCE_DIR
OUTPUT_PKL = PROCESSED_DIR / "arcadia_v2_gene_filtered__phylogeny_clean.pkl"

MIN_SNPS_PER_GENE = 5
FINAL_GENE_CAP = 5000


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 1.5b (phylogeny_split, clean) -- Build Filtered Gene Dataset")
    print("=" * 80)

    print("\n[1/4] Loading Arcadia data...")
    with open(PROCESSED_DIR / "arcadia_v2_snp_matrix.pkl", "rb") as f:
        snp_data = pickle.load(f)
    X_snp = snp_data["X_snp"]
    snp_positions = np.array(snp_data["snp_positions"])
    print(f"  X_snp: {X_snp.shape}, dtype={X_snp.dtype}")

    with open(PROCESSED_DIR / "arcadia_v2_master_features.pkl", "rb") as f:
        master = pickle.load(f)

    importance_csv = IMPORTANCE_DIR / "xgb_gene_importance.csv"
    if not importance_csv.exists():
        raise FileNotFoundError(
            f"{importance_csv} not found -- run trilens.experiments.gene_importance_ranking_phylogroup first."
        )
    print(f"\n[2/4] Loading gene importance (phylogeny_split-ranked) from {importance_csv}...")
    imp_df = pd.read_csv(importance_csv)
    gene_importance = dict(zip(imp_df["gene_id"], imp_df["imp_sum"]))
    print(f"  Loaded importance for {len(gene_importance):,} genes")

    print("\n[3/4] Building gene -> SNP mapping...")
    gene_to_snp_indices, _ = build_gene_to_snp_ranges(snp_positions)
    stats_full = get_gene_size_stats(gene_to_snp_indices)
    print(f"  Total genes: {stats_full['n_genes']:,}  Genes with >=5 SNPs: {stats_full['n_with_at_least_5']:,}")

    selected_genes = filter_genes(
        gene_to_snp_indices=gene_to_snp_indices,
        gene_importance=gene_importance,
        min_snps_per_gene=MIN_SNPS_PER_GENE,
        top_k_by_importance=FINAL_GENE_CAP,
        keep_genes=None,
    )
    print(f"  Selected: {len(selected_genes):,} genes")

    print("\n[4/4] Building filtered SNP matrix...")
    X_filtered, positions_filtered, gene_to_range = build_filtered_snp_matrix(
        X_snp=X_snp,
        snp_positions=snp_positions,
        gene_to_snp_indices=gene_to_snp_indices,
        selected_genes=selected_genes,
    )
    print(f"  X_filtered: {X_filtered.shape}, dtype={X_filtered.dtype}")

    filtered_sizes = np.array([end - start for start, end in gene_to_range.values()])
    print(
        f"  Filtered gene size stats: min={filtered_sizes.min()} "
        f"median={int(np.median(filtered_sizes))} mean={filtered_sizes.mean():.1f} "
        f"max={filtered_sizes.max()} total={filtered_sizes.sum():,}"
    )

    overlap_with_locked = None
    locked_pkl = PROCESSED_DIR / "arcadia_v2_gene_filtered.pkl"
    if locked_pkl.exists():
        with open(locked_pkl, "rb") as f:
            locked = pickle.load(f)
        locked_genes = set(str(g) for g in locked["selected_gene_ids"])
        new_genes = set(selected_genes)
        overlap_with_locked = len(locked_genes & new_genes)
        print(
            f"  Overlap with locked (random_split-ranked) gene set: "
            f"{overlap_with_locked:,}/{len(new_genes):,} "
            f"({100 * overlap_with_locked / len(new_genes):.1f}%)"
        )

    output = {
        "X_snp_filtered": X_filtered,
        "snp_positions_filtered": positions_filtered,
        "selected_gene_ids": np.array(selected_genes),
        "gene_to_range": gene_to_range,
        "Y": master["Y"],
        "label_mask": master["label_mask"],
        "label_names": master["label_names"],
        "antibiotic_to_index": master.get("antibiotic_to_index"),
        "random_split": master["random_split"],
        "phylogeny_split": master["phylogeny_split"],
        "phylogroup_onehot": master["phylogroup_onehot"],
        "phylogroup_names": master.get("phylogroup_names"),
        "genome_names": master["genome_names"],
        "sra_ids": master["sra_ids"],
        "n_samples": int(X_filtered.shape[0]),
        "n_genes": len(selected_genes),
        "n_snps_filtered": int(X_filtered.shape[1]),
        "n_snps_original": int(X_snp.shape[1]),
        "filter_metadata": {
            "min_snps_per_gene": MIN_SNPS_PER_GENE,
            "final_gene_cap": FINAL_GENE_CAP,
            "gene_filter_formula": "(variability_pass ∩ importance_top_k)",
            "importance_source": "xgb_gain_summed_across_antibiotics__RANKED_ON_PHYLOGENY_SPLIT_TRAIN",
            "n_genes_original": stats_full["n_genes"],
            "n_genes_after_variability": stats_full["n_with_at_least_5"],
            "n_genes_selected": len(selected_genes),
            "overlap_with_locked_random_split_gene_set": overlap_with_locked,
            "stats_filtered_sizes": {
                "min": int(filtered_sizes.min()),
                "median": int(np.median(filtered_sizes)),
                "mean": float(filtered_sizes.mean()),
                "max": int(filtered_sizes.max()),
                "p95": int(np.percentile(filtered_sizes, 95)),
            },
        },
        "stage": "1.5_filtered_phylogeny_clean",
        "version": "v1_phylogeny_clean",
    }

    OUTPUT_PKL.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(output, f, protocol=4)
    print(f"\n  Saved: {OUTPUT_PKL} ({OUTPUT_PKL.stat().st_size / 1e6:.1f} MB)")

    summary = {
        "input_n_snps": int(X_snp.shape[1]),
        "input_n_genes": stats_full["n_genes"],
        "output_n_snps": int(X_filtered.shape[1]),
        "output_n_genes": len(selected_genes),
        "overlap_with_locked_random_split_gene_set": overlap_with_locked,
        "filter_metadata": output["filter_metadata"],
    }
    with open(OUT_DIR / "SUMMARY_build.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n  STAGE 1.5b (phylogeny_split, clean) COMPLETE")


if __name__ == "__main__":
    main()
