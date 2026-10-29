from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

GENE_STAT_NAMES: tuple[str, ...] = ("mean", "var", "n_nonref", "max")


def compute_gene_features(
    X_snp: np.ndarray,
    gene_to_range: dict[str, tuple[int, int]],
    selected_genes: list[str],
    stats: tuple[str, ...] = GENE_STAT_NAMES,
    chunk_size: int = 100,
) -> tuple[np.ndarray, list[str]]:

    n_samples = X_snp.shape[0]
    n_genes = len(selected_genes)
    n_stats = len(stats)
    out = np.zeros((n_samples, n_genes * n_stats), dtype=np.float32)
    feature_names: list[str] = []

    for g_idx, gene_id in enumerate(selected_genes):
        start, end = gene_to_range[gene_id]

        gene_block = X_snp[:, start:end].astype(np.float32)

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

    print(f"  X_gene_features: {X_features.shape}, memory={X_features.nbytes / 1e9:.2f} GB")

    return {
        "X_gene_features": X_features,
        "feature_names": feature_names,
        "gene_ids": selected_gene_ids,
        "stats_used": stats,
        "Y": d["Y"],
        "label_mask": d["label_mask"],
        "label_names": d["label_names"],
        "random_split": d["random_split"],
        "phylogeny_split": d.get("phylogeny_split"),
        "phylogroup_onehot": d.get("phylogroup_onehot"),
        "genome_names": d.get("genome_names"),
    }
