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
IMPORTANCE_DIR = ROOT / "results" / "gene_ranking_lineage"
SPLIT_PKL = PROCESSED_DIR / "mlst_lineage_holdout_split.pkl"
OUTPUT_PKL = PROCESSED_DIR / "arcadia_v2_gene_filtered__mlst_holdout.pkl"
SPLIT_KEY = "mlst_lineage_holdout_split"

MIN_SNPS_PER_GENE = 5
FINAL_GENE_CAP = 5000


def main() -> None:
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 2b -- Build filtered gene dataset (MLST lineage holdout ranking)")
    print("=" * 80)

    print("\n[1/4] Loading Arcadia data...")
    with open(PROCESSED_DIR / "arcadia_v2_snp_matrix.pkl", "rb") as f:
        snp_data = pickle.load(f)
    X_snp = snp_data["X_snp"]
    snp_positions = np.array(snp_data["snp_positions"])
    print(f"  X_snp: {X_snp.shape}, dtype={X_snp.dtype}")

    with open(PROCESSED_DIR / "arcadia_v2_master_features.pkl", "rb") as f:
        master = pickle.load(f)
    with open(SPLIT_PKL, "rb") as f:
        sp = pickle.load(f)

    importance_csv = IMPORTANCE_DIR / "xgb_gene_importance.csv"
    if not importance_csv.exists():
        raise FileNotFoundError(
            f"{importance_csv} not found -- run trilens.experiments.gene_importance_ranking_lineage first."
        )
    print("\n[2/4] Loading MLST-holdout-ranked gene importance...")
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

    filtered_sizes = np.array([e - s for s, e in gene_to_range.values()])
    print(
        f"  gene size: min={filtered_sizes.min()} "
        f"median={int(np.median(filtered_sizes))} mean={filtered_sizes.mean():.1f} "
        f"max={filtered_sizes.max()} total={filtered_sizes.sum():,}"
    )

    overlaps = {}
    for label, path in [
        ("locked_random_split", PROCESSED_DIR / "arcadia_v2_gene_filtered.pkl"),
        ("phylogeny_clean", PROCESSED_DIR / "arcadia_v2_gene_filtered__phylogeny_clean.pkl"),
    ]:
        if path.exists():
            with open(path, "rb") as f:
                other = pickle.load(f)
            other_genes = set(str(g) for g in other["selected_gene_ids"])
            ov = len(other_genes & set(selected_genes))
            overlaps[label] = ov
            print(
                f"  Overlap with {label} gene set: {ov:,}/{len(selected_genes):,} "
                f"({100 * ov / len(selected_genes):.1f}%)"
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
        SPLIT_KEY: {
            "train_idx": np.asarray(sp["train_idx"]),
            "val_idx": np.asarray(sp["val_idx"]),
            "test_idx": np.asarray(sp["test_idx"]),
        },
        "st_per_genome": sp["st_per_genome"],
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
            "importance_source": "xgb_gain_summed_across_antibiotics__RANKED_ON_MLST_LINEAGE_HOLDOUT_TRAIN",
            "n_genes_original": stats_full["n_genes"],
            "n_genes_after_variability": stats_full["n_with_at_least_5"],
            "n_genes_selected": len(selected_genes),
            "gene_set_overlaps": overlaps,
            "stats_filtered_sizes": {
                "min": int(filtered_sizes.min()),
                "median": int(np.median(filtered_sizes)),
                "mean": float(filtered_sizes.mean()),
                "max": int(filtered_sizes.max()),
                "p95": int(np.percentile(filtered_sizes, 95)),
            },
        },
        "stage": "1.5_filtered_mlst_holdout",
        "version": "v1_mlst_holdout",
    }

    with open(OUTPUT_PKL, "wb") as f:
        pickle.dump(output, f, protocol=4)
    print(f"\n  Saved: {OUTPUT_PKL} ({OUTPUT_PKL.stat().st_size / 1e6:.1f} MB)")

    with open(IMPORTANCE_DIR / "SUMMARY_build.json", "w") as f:
        json.dump(
            {
                "input_n_snps": int(X_snp.shape[1]),
                "input_n_genes": stats_full["n_genes"],
                "output_n_snps": int(X_filtered.shape[1]),
                "output_n_genes": len(selected_genes),
                "gene_set_overlaps": overlaps,
                "filter_metadata": output["filter_metadata"],
            },
            f,
            indent=2,
        )
    print("\n  STEP 2b COMPLETE")


if __name__ == "__main__":
    main()
