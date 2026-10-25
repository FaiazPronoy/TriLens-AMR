"""Three-view assembly: per-gene SNP blocks (Lens A), gene-aggregated tabular
features (Lens C), gene presence/absence (Lens B).

The gene-aggregated feature computation is ported from the existing,
untouched `src/data/gene_aggregated_features.py` (one directory up) --
logic unchanged, just re-homed here so this package doesn't depend on
importing across the two repositories for something this small.

`assemble_pillar2_bundle()` is the one function that ties everything
together into a single dict Pillar 2 can consume, and it is where the B3 fix
actually gets enforced structurally: it takes one `gene_filter_result` PER
split regime (see `gene_filter.select_genes_for_split`), so it is not
possible to accidentally hand Pillar 2 a bundle built from a gene selection
that doesn't match the split it's about to be evaluated on.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

GENE_STAT_NAMES: tuple[str, ...] = ("mean", "var", "n_nonref", "max")


def compute_gene_aggregated_features(
    X_snp: np.ndarray,
    gene_to_range: dict[str, tuple[int, int]],
    selected_genes: list[str],
    stats: tuple[str, ...] = GENE_STAT_NAMES,
) -> tuple[np.ndarray, list[str]]:
    """Per-(sample, gene) summary statistics over that gene's SNP columns.

    Same four statistics as the existing pipeline's Expert C features:
    mean (allele-frequency proxy), variance (within-gene heterogeneity),
    n_nonref (mutation load), max (rare-allele indicator).
    """
    n_samples = X_snp.shape[0]
    n_genes = len(selected_genes)
    n_stats = len(stats)
    out = np.zeros((n_samples, n_genes * n_stats), dtype=np.float32)
    feature_names: list[str] = []

    for g_idx, gene_id in enumerate(selected_genes):
        start, end = gene_to_range[gene_id]
        cols = X_snp[:, start:end]
        base = g_idx * n_stats
        for s_idx, stat in enumerate(stats):
            if stat == "mean":
                out[:, base + s_idx] = cols.mean(axis=1)
            elif stat == "var":
                out[:, base + s_idx] = cols.var(axis=1)
            elif stat == "n_nonref":
                out[:, base + s_idx] = (cols > 0).sum(axis=1)
            elif stat == "max":
                out[:, base + s_idx] = cols.max(axis=1)
            else:
                raise ValueError(f"Unknown gene stat: {stat!r}")
            feature_names.append(f"{gene_id}__{stat}")

    return out, feature_names


def assemble_pillar2_bundle(
    gene_filter_result: dict,
    gene_pa_result: dict,
    Y: np.ndarray,
    label_mask: np.ndarray,
    label_names: list[str],
    phylogroup_onehot: np.ndarray,
    genome_names: np.ndarray,
    sra_ids: np.ndarray,
    split: dict[str, np.ndarray],
    *,
    gene_stats: tuple[str, ...] = GENE_STAT_NAMES,
) -> dict:
    """Bundle Lens A (per-gene SNP blocks), Lens B (P/A), Lens C
    (gene-aggregated) into one dict for a single split regime.

    `gene_filter_result` and `split` MUST correspond to the same split
    regime (i.e. `gene_filter_result["split_name"]` should describe the same
    partition as `split`) -- this function does not itself verify that
    (it has no way to know split identity beyond the name string), so the
    caller is responsible for not mixing a `random_split`-selected gene set
    with a `lineage_holdout_split` partition. See gene_filter.py's module
    docstring (B3) for why that specific mistake is the one this whole
    module exists to make harder.
    """
    X_snp_filtered = gene_filter_result["X_snp_filtered"]
    gene_to_range = gene_filter_result["gene_to_range"]
    selected_genes = list(gene_filter_result["selected_gene_ids"])

    X_agg, agg_feature_names = compute_gene_aggregated_features(
        X_snp_filtered, gene_to_range, selected_genes, stats=gene_stats
    )

    return {
        "split_name": gene_filter_result.get("split_name", "unknown"),
        # Lens A input
        "X_snp_filtered": X_snp_filtered,
        "gene_to_range": gene_to_range,
        "selected_gene_ids": np.array(selected_genes),
        # Lens B input
        "X_pa": gene_pa_result["X_pa"],
        "pa_gene_ids": gene_pa_result["gene_ids"],
        "pa_matched_mask": gene_pa_result["matched_mask"],
        # Lens C input
        "X_gene_agg": X_agg,
        "gene_agg_feature_names": agg_feature_names,
        "gene_agg_stats_used": gene_stats,
        # Labels / metadata
        "Y": Y,
        "label_mask": label_mask,
        "label_names": np.array(label_names),
        "phylogroup_onehot": phylogroup_onehot,
        "genome_names": genome_names,
        "sra_ids": sra_ids,
        # Split
        "split": split,
        "n_samples": int(X_snp_filtered.shape[0]),
        "n_genes": len(selected_genes),
    }
