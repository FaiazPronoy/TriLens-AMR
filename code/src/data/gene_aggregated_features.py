# src/data/gene_aggregated_features.py
"""Aggregate per-gene SNP statistics into tabular features for tree-based models.

Takes the Stage 1.5 filtered.pkl (5000 genes, ~260K SNPs) and produces a
compact tabular matrix:

    For each (sample, gene), compute:
        mean         — mean allele state (allele-frequency proxy)
        variance     — within-gene SNP variance (genetic heterogeneity proxy)
        n_nonref     — count of SNPs where sample != 0 (mutation load proxy)
        max          — max allele state (rare-allele indicator)

    → final shape (n_samples, n_genes * 4) ≈ (6962, 20000)

This format is ideal for XGBoost / LightGBM / CascadeStack — much smaller
than raw 535K SNPs, interpretable (each feature attached to a named gene),
and respects gene boundaries (no spurious cross-gene patterns).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


# Statistic names emitted per gene (controls column order in output)
GENE_STAT_NAMES: tuple[str, ...] = ("mean", "var", "n_nonref", "max")


def compute_gene_features(
    X_snp: np.ndarray,                  # (n_samples, n_snps_filtered) int8
    gene_to_range: dict[str, tuple[int, int]],
    selected_genes: list[str],
    stats: tuple[str, ...] = GENE_STAT_NAMES,
    chunk_size: int = 100,              # genes per chunk; keeps memory low
) -> tuple[np.ndarray, list[str]]:
    """Compute per-gene aggregated features.

    Args:
        X_snp: filtered SNP matrix.
        gene_to_range: gene_id -> (start_idx, end_idx) in X_snp.
        selected_genes: ordered list of gene IDs to process.
        stats: which statistics to compute per gene.
        chunk_size: number of genes to process per chunk.

    Returns:
        features: float32 array of shape (n_samples, n_genes * len(stats)).
        feature_names: human-readable column names like
            "LMHECDEF_04343__mean", "LMHECDEF_04343__var", ...
    """
    n_samples = X_snp.shape[0]
    n_genes = len(selected_genes)
    n_stats = len(stats)
    out = np.zeros((n_samples, n_genes * n_stats), dtype=np.float32)
    feature_names: list[str] = []

    for g_idx, gene_id in enumerate(selected_genes):
        start, end = gene_to_range[gene_id]
        # cast small slice to float32 for stable arithmetic
        gene_block = X_snp[:, start:end].astype(np.float32)  # (n_samples, gene_len)

        col_base = g_idx * n_stats
        for s_idx, stat in enumerate(stats):
            col = col_base + s_idx
            if stat == "mean":
                out[:, col] = gene_block.mean(axis=1)
            elif stat == "var":
                out[:, col] = gene_block.var(axis=1)
            elif stat == "n_nonref":
                out[:, col] = (gene_block > 0).sum(axis=1)
            elif stat == "max":
                out[:, col] = gene_block.max(axis=1)
            elif stat == "sum":
                out[:, col] = gene_block.sum(axis=1)
            else:
                raise ValueError(f"Unknown stat: {stat}")

        for s in stats:
            feature_names.append(f"{gene_id}__{s}")

        if (g_idx + 1) % 500 == 0:
            print(f"      computed features for {g_idx + 1}/{n_genes} genes")

    return out, feature_names


def build_gene_feature_matrix_from_pkl(
    gene_filtered_pkl_path: str | Path,
    stats: tuple[str, ...] = GENE_STAT_NAMES,
) -> dict:
    """End-to-end builder: load Stage 1.5 pkl, compute features, return dict.

    Returns dict with keys:
        X_gene_features    — (n_samples, n_genes*n_stats) float32
        feature_names      — list of feature column names
        gene_ids           — list of source gene IDs (in order)
        stats_used         — tuple of stat names
        Y, label_mask, label_names, random_split, phylogeny_split,
        phylogroup_onehot — passthrough
    """
    print(f"  Loading {gene_filtered_pkl_path}...")
    with open(gene_filtered_pkl_path, "rb") as f:
        d = pickle.load(f)

    X_snp_filtered = d["X_snp_filtered"]
    gene_to_range = d["gene_to_range"]
    selected_gene_ids = list(d["selected_gene_ids"])
    print(f"  X_snp_filtered: {X_snp_filtered.shape}, dtype={X_snp_filtered.dtype}")
    print(f"  n_genes selected: {len(selected_gene_ids)}")
    print(f"  Stats per gene: {stats}")
    print(f"  Expected feature count: {len(selected_gene_ids) * len(stats):,}")

    X_features, feature_names = compute_gene_features(
        X_snp=X_snp_filtered,
        gene_to_range=gene_to_range,
        selected_genes=selected_gene_ids,
        stats=stats,
    )

    print(f"  X_gene_features: {X_features.shape}, "
          f"memory={X_features.nbytes / 1e9:.2f} GB")

    return {
        "X_gene_features": X_features,
        "feature_names": feature_names,
        "gene_ids": selected_gene_ids,
        "stats_used": stats,
        # passthroughs needed for training
        "Y": d["Y"],
        "label_mask": d["label_mask"],
        "label_names": d["label_names"],
        "random_split": d["random_split"],
        "phylogeny_split": d.get("phylogeny_split"),
        "phylogroup_onehot": d.get("phylogroup_onehot"),
        "genome_names": d.get("genome_names"),
    }
