from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from trilens.data.gene_selection_rules import (
    build_filtered_snp_matrix,
    build_gene_to_snp_ranges,
    filter_genes,
    get_gene_size_stats,
)


def run(cfg: DictConfig) -> None:
    exp_cfg = cfg.experiment
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Build the filtered gene dataset")
    print("=" * 80)

    print("\n[1/4] Loading Arcadia data...")
    with open(cfg.paths.arcadia_snp_matrix_pkl, "rb") as f:
        snp_data = pickle.load(f)
    X_snp = snp_data["X_snp"]
    snp_positions = np.array(snp_data["snp_positions"])
    print(f"  X_snp: {X_snp.shape}, dtype={X_snp.dtype}")

    with open(cfg.paths.arcadia_master_pkl, "rb") as f:
        master = pickle.load(f)
    print("  Master loaded")

    importance_csv = out_dir / "xgb_gene_importance.csv"
    if not importance_csv.exists():
        raise FileNotFoundError(f"Importance CSV not found: {importance_csv}\nRun xgb_gene_importance first.")
    print(f"\n[2/4] Loading gene importance from {importance_csv}...")
    imp_df = pd.read_csv(importance_csv)
    gene_importance = dict(zip(imp_df["gene_id"], imp_df["imp_sum"]))
    print(f"  Loaded importance for {len(gene_importance):,} genes")

    print("\n[3/4] Building gene → SNP mapping...")
    gene_to_snp_indices, _ = build_gene_to_snp_ranges(snp_positions)
    stats_full = get_gene_size_stats(gene_to_snp_indices)
    print(f"  Total genes: {stats_full['n_genes']:,}")
    print(f"  Genes with ≥5 SNPs: {stats_full['n_with_at_least_5']:,}")

    filt_cfg = exp_cfg.filtering
    print("\n  Filtering parameters:")
    print(f"    min_snps_per_gene: {filt_cfg.min_snps_per_gene}")
    print(f"    final_gene_cap: {filt_cfg.final_gene_cap}")

    selected_genes = filter_genes(
        gene_to_snp_indices=gene_to_snp_indices,
        gene_importance=gene_importance,
        min_snps_per_gene=filt_cfg.min_snps_per_gene,
        top_k_by_importance=filt_cfg.final_gene_cap,
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
    print(f"  Memory: {X_filtered.nbytes / 1e9:.2f} GB (was {X_snp.nbytes / 1e9:.2f} GB)")

    filtered_sizes = np.array([end - start for start, end in gene_to_range.values()])
    print("\n  Filtered gene size stats:")
    print(f"    Min: {filtered_sizes.min()}")
    print(f"    Median: {int(np.median(filtered_sizes))}")
    print(f"    Mean: {filtered_sizes.mean():.1f}")
    print(f"    Max: {filtered_sizes.max()}")
    print(f"    Total SNPs kept: {filtered_sizes.sum():,}")
    print(f"    Reduction: {X_filtered.shape[1] / X_snp.shape[1] * 100:.1f}% of original")

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
            "min_snps_per_gene": filt_cfg.min_snps_per_gene,
            "final_gene_cap": filt_cfg.final_gene_cap,
            "gene_filter_formula": "(variability_pass ∩ importance_top_k)",
            "importance_source": "xgb_gain_summed_across_antibiotics",
            "n_genes_original": stats_full["n_genes"],
            "n_genes_after_variability": stats_full["n_with_at_least_5"],
            "n_genes_selected": len(selected_genes),
            "stats_filtered_sizes": {
                "min": int(filtered_sizes.min()),
                "median": int(np.median(filtered_sizes)),
                "mean": float(filtered_sizes.mean()),
                "max": int(filtered_sizes.max()),
                "p95": int(np.percentile(filtered_sizes, 95)),
            },
        },
        "stage": "1.5_filtered",
        "version": "v1",
    }

    out_path = Path(exp_cfg.output_pkl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(output, f, protocol=4)

    file_size_mb = out_path.stat().st_size / 1e6
    print(f"\n  Saved: {out_path}")
    print(f"  File size: {file_size_mb:.1f} MB")

    summary = {
        "input_n_snps": int(X_snp.shape[1]),
        "input_n_genes": stats_full["n_genes"],
        "output_n_snps": int(X_filtered.shape[1]),
        "output_n_genes": len(selected_genes),
        "filter_metadata": output["filter_metadata"],
    }
    with open(out_dir / "SUMMARY.json", "w") as f:
        json.dump(summary, f, indent=2)

    md = f"""# Gene Filtering Summary

## Input
- SNPs: {X_snp.shape[1]:,}
- Genes: {stats_full["n_genes"]:,}

## Filtering Applied
- Variability filter: ≥{filt_cfg.min_snps_per_gene} SNPs per gene
- Importance filter: top-{filt_cfg.final_gene_cap} by XGBoost gain
- Formula: `(variability_pass ∩ importance_top_k)`

## Output
- Genes selected: **{len(selected_genes):,}**
- SNPs retained: **{X_filtered.shape[1]:,}** ({X_filtered.shape[1] / X_snp.shape[1] * 100:.1f}% of original)
- File: `{out_path}`
- File size: {file_size_mb:.1f} MB

## Gene Size Distribution
- Min SNPs/gene: {filtered_sizes.min()}
- Median: {int(np.median(filtered_sizes))}
- Mean: {filtered_sizes.mean():.1f}
- Max: {filtered_sizes.max()}
- p95: {int(np.percentile(filtered_sizes, 95))}

"""
    with open(out_dir / "SUMMARY.md", "w", encoding="utf-8") as f:
        f.write(md)
    print(f"\n  Summary → {out_dir / 'SUMMARY.md'}")
    print(f"\n{'=' * 80}")
    print("  Filtered gene dataset complete")
    print(f"{'=' * 80}")
