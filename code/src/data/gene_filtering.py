"""Gene filtering and grouping utilities for Stage 1.5.

Takes Arcadia v2 SNP positions (LMHECDEF_XXXXX:POS:REF>ALT format) and
groups SNPs by gene_id (the prefix). Applies filtering criteria to reduce
~18,975 genes down to ~3000-5000 for Pillar 2 architecture.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


_GENE_ID_PATTERN = re.compile(r"^([^:]+):(\d+):")


def parse_gene_id(pos_str: str) -> tuple[str, int]:
    """Parse single SNP position string.
    
    Args:
        pos_str: e.g. 'LMHECDEF_00004:12:A>T' or 'Cluster_1284_+_+_...'
    
    Returns:
        (gene_id, position) — gene_id is everything before first colon.
    """
    m = _GENE_ID_PATTERN.match(pos_str)
    if m is None:
        raise ValueError(f"Cannot parse SNP position: {pos_str}")
    return m.group(1), int(m.group(2))


def build_gene_to_snp_ranges(
    snp_positions: np.ndarray,
) -> tuple[dict[str, list[int]], np.ndarray]:
    """Group SNP indices by gene_id (preserves SNP order within gene).
    
    Args:
        snp_positions: array of position strings (length n_snps).
    
    Returns:
        gene_to_snp_indices: dict mapping gene_id -> list of SNP indices.
        gene_id_per_snp: int array (length n_snps), gene index per SNP.
    """
    gene_to_indices: dict[str, list[int]] = defaultdict(list)
    
    for snp_idx, pos_str in enumerate(snp_positions):
        gene_id, _ = parse_gene_id(str(pos_str))
        gene_to_indices[gene_id].append(snp_idx)
    
    # Build int array of gene index per SNP
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
    """Compute summary statistics of gene sizes (SNPs per gene).
    
    Args:
        gene_to_snp_indices: gene_id -> list of SNP indices.
    
    Returns:
        Dictionary with min, max, median, mean, p25, p75, p95, total stats.
    """
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
    """Apply locked filtering formula.
    
    Filter: (variability_pass AND importance_top_k) UNION keep_genes
    
    Args:
        gene_to_snp_indices: All genes mapped to their SNP indices.
        gene_importance: Aggregated importance score per gene.
        min_snps_per_gene: Minimum SNPs for variability filter.
        top_k_by_importance: Top-K genes by aggregated importance.
        keep_genes: Always-keep set (e.g., CARD-known AMR genes).
    
    Returns:
        Sorted list of gene IDs to keep.
    """
    # Variability filter
    variability_pass = {
        g for g, idxs in gene_to_snp_indices.items()
        if len(idxs) >= min_snps_per_gene
    }
    
    # Importance filter: top-K genes
    valid_for_importance = variability_pass & set(gene_importance.keys())
    sorted_by_imp = sorted(
        valid_for_importance,
        key=lambda g: gene_importance.get(g, 0.0),
        reverse=True,
    )
    importance_top_k = set(sorted_by_imp[:top_k_by_importance])
    
    # Combine
    selected = variability_pass & importance_top_k
    
    if keep_genes is not None:
        # Add keep_genes regardless of filters
        selected = selected | (keep_genes & set(gene_to_snp_indices.keys()))
    
    return sorted(selected)


def build_filtered_snp_matrix(
    X_snp: np.ndarray,
    snp_positions: np.ndarray,
    gene_to_snp_indices: dict[str, list[int]],
    selected_genes: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, tuple[int, int]]]:
    """Build filtered SNP matrix containing only SNPs from selected genes.
    
    Args:
        X_snp: Full SNP matrix (n_samples, n_snps_total) int8.
        snp_positions: Full position array.
        gene_to_snp_indices: All genes mapped to SNP indices.
        selected_genes: Gene IDs to keep.
    
    Returns:
        X_filtered: (n_samples, n_snps_filtered) int8.
        positions_filtered: array of position strings.
        gene_to_range: dict[gene_id -> (start_idx, end_idx)] in filtered matrix.
    """
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