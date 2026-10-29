from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

import numpy as np

_GENE_ID_PATTERN = re.compile(r"^([^:]+):(\d+):")


def parse_gene_id(pos_str: str) -> tuple[str, int]:
    m = _GENE_ID_PATTERN.match(pos_str)
    if m is None:
        raise ValueError(f"Cannot parse SNP position: {pos_str}")
    return m.group(1), int(m.group(2))


def build_gene_to_snp_ranges(
    snp_positions: np.ndarray,
) -> tuple[dict[str, list[int]], np.ndarray]:

    gene_to_indices: dict[str, list[int]] = defaultdict(list)

    for snp_idx, pos_str in enumerate(snp_positions):
        gene_id, _ = parse_gene_id(str(pos_str))
        gene_to_indices[gene_id].append(snp_idx)

    gene_id_list = list(gene_to_indices.keys())
    gene_to_int = {g: i for i, g in enumerate(gene_id_list)}

    gene_id_per_snp = np.zeros(len(snp_positions), dtype=np.int32)
    for snp_idx, pos_str in enumerate(snp_positions):
        gene_id, _ = parse_gene_id(str(pos_str))
        gene_id_per_snp[snp_idx] = gene_to_int[gene_id]

    return dict(gene_to_indices), gene_id_per_snp


def get_gene_size_stats(
    gene_to_snp_indices: dict[str, list[int]],
) -> dict[str, float]:

    sizes = np.array([len(v) for v in gene_to_snp_indices.values()])
    return {
        "n_genes": len(sizes),
        "total_snps": int(sizes.sum()),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "median_size": float(np.median(sizes)),
        "mean_size": float(sizes.mean()),
        "p25": float(np.percentile(sizes, 25)),
        "p75": float(np.percentile(sizes, 75)),
        "p95": float(np.percentile(sizes, 95)),
        "n_with_at_least_5": int((sizes >= 5).sum()),
        "n_with_at_least_10": int((sizes >= 10).sum()),
    }


def filter_genes(
    gene_to_snp_indices: dict[str, list[int]],
    gene_importance: dict[str, float],
    min_snps_per_gene: int = 5,
    top_k_by_importance: int = 5000,
    keep_genes: Optional[set[str]] = None,
) -> list[str]:

    variability_pass = {g for g, idxs in gene_to_snp_indices.items() if len(idxs) >= min_snps_per_gene}

    valid_for_importance = variability_pass & set(gene_importance.keys())
    sorted_by_imp = sorted(
        valid_for_importance,
        key=lambda g: gene_importance.get(g, 0.0),
        reverse=True,
    )
    importance_top_k = set(sorted_by_imp[:top_k_by_importance])

    selected = variability_pass & importance_top_k

    if keep_genes is not None:
        selected = selected | (keep_genes & set(gene_to_snp_indices.keys()))

    return sorted(selected)


def build_filtered_snp_matrix(
    X_snp: np.ndarray,
    snp_positions: np.ndarray,
    gene_to_snp_indices: dict[str, list[int]],
    selected_genes: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:

    new_indices: list[int] = []
    gene_to_range: dict[str, tuple[int, int]] = {}

    for gene_id in selected_genes:
        indices = gene_to_snp_indices[gene_id]
        start = len(new_indices)
        new_indices.extend(indices)
        end = len(new_indices)
        gene_to_range[gene_id] = (start, end)

    indices_array = np.array(new_indices, dtype=np.int64)
    X_filtered = X_snp[:, indices_array]
    positions_filtered = snp_positions[indices_array]

    return X_filtered, positions_filtered, gene_to_range
