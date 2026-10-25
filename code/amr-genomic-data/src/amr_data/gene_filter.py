"""Stage 1.5 gene filtering — train-only, per-split-regime (B3).

**Not reimplemented from a notebook** -- none of the five source notebooks
contain Stage 1.5's own code (`arcadia-v2-phase-c.ipynb` only ever *loads*
`arcadia_v2_gene_filtered.pkl`, referencing a placeholder path
`<your-stage-1.5-notebook>`). The real source was found in this working
directory's existing, separate pipeline:
`src/experiments/run_xgb_gene_importance.py` and
`src/experiments/build_gene_filtered_dataset.py` (one directory up from this
package), plus the grouping utilities in `src/data/gene_filtering.py`.

**Audit verdict (see docs/AUDIT_FINDINGS.md, B3), reported prominently as
instructed:**

1. The existing XGBoost gain-ranking IS fit train-only, correctly --
   `_train_xgb_per_antibiotic()` in `run_xgb_gene_importance.py` masks to
   `train_idx` (further intersected with each antibiotic's label
   availability) before the variance filter, the chi² pre-filter, AND
   `xgb.train()` itself. `test_idx` is never referenced in that function.
   `random_split`'s own numbers are not leaked by this step.

2. **However**, the resulting 5,000-gene feature set is a SINGLE shared
   artifact, applied to both `random_split` and `phylogeny_split` alike.
   Verified directly (loading `arcadia_v2_gene_filtered.pkl`'s own two
   split dicts): **73.1% of `phylogroup_stratified_split`'s test set (764 /
   1045 genomes) was part of `random_split`'s training set** whose labels
   picked the genes. Every "phylogeny_split" / T7 number in the thesis
   therefore carries a real, quantified gene-selection-leakage caveat on
   top of B1's stratification-not-holdout issue. This is *worse* once
   `lineage_holdout_split` is used, because that split's test genomes are
   chosen specifically to be dissimilar to train -- reusing a
   `random_split`-selected gene set there would be even less justified.

This module is written so gene selection is **required** to declare which
split's `train_idx` it was fit on, and is **re-run once per split regime**
-- `random_split` gets its own filtered feature set, `phylogroup_stratified_
split` gets its own, `lineage_holdout_split` gets its own. Sharing one
feature set across split regimes is exactly the "leakage through the back
door" the refactor spec warned about, and is no longer possible to do by
accident: `select_genes_for_split()` takes `train_idx` as a required
positional argument and raises `TypeError` if called without one.

**Measured runtime (this reimplementation, CPU, not the original GPU-locked
pipeline):** `compute_gene_importance_for_antibiotic()` run against the real
`arcadia_v2_snp_matrix.pkl` (535,439 SNPs, ~4,872-row `random_split` train
partition) for a single antibiotic (AMP) took **182 seconds** on CPU and
produced non-degenerate output (1,857/18,975 genes with nonzero importance).
This module deliberately defaults to CPU (`tree_method: hist`, no
`device: cuda`) for portability -- add `xgb_params={"device": "cuda", ...}`
per call if a GPU is available and speed matters; a full 7-antibiotic x
3-split-regime run at this CPU speed is on the order of tens of minutes, not
seconds, and was NOT re-run for all antibiotics/splits as part of this
refactor (compute task, not a refactor task -- see README.md).
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import chi2

from .config import load_params

_params = load_params()["gene_filter"]

_GENE_ID_PATTERN = re.compile(r"^([^:]+):(\d+):")


# ---------------------------------------------------------------------------
# Gene <-> SNP grouping (split-independent -- this part is fine to share)
# ---------------------------------------------------------------------------
def parse_gene_id(pos_str: str) -> tuple[str, int]:
    """'LMHECDEF_00004:12:A>T' -> ('LMHECDEF_00004', 12).

    The prefix before the first colon is treated as the gene ID. This was
    independently cross-checked during the audit against
    `arcadia_v2_gene_presence_absence.pkl`'s Prokka-locus-tag-based gene IDs
    (same LMHECDEF_XXXXX namespace, same per-gene granularity) and against
    successful eggNOG functional annotation of these exact IDs elsewhere in
    the project -- both are independent evidence that this prefix is a
    real per-gene identifier, not an arbitrary contig fragment.
    """
    m = _GENE_ID_PATTERN.match(pos_str)
    if m is None:
        raise ValueError(f"Cannot parse SNP position: {pos_str}")
    return m.group(1), int(m.group(2))


def build_gene_to_snp_ranges(snp_positions: np.ndarray) -> dict[str, list[int]]:
    """Group SNP indices by gene_id, preserving order within each gene."""
    gene_to_indices: dict[str, list[int]] = defaultdict(list)
    for snp_idx, pos_str in enumerate(snp_positions):
        gene_id, _ = parse_gene_id(str(pos_str))
        gene_to_indices[gene_id].append(snp_idx)
    return dict(gene_to_indices)


def get_gene_size_stats(gene_to_snp_indices: dict[str, list[int]]) -> dict[str, float]:
    sizes = np.array([len(v) for v in gene_to_snp_indices.values()])
    return {
        "n_genes": len(sizes),
        "total_snps": int(sizes.sum()),
        "min_size": int(sizes.min()),
        "max_size": int(sizes.max()),
        "median_size": float(np.median(sizes)),
        "mean_size": float(sizes.mean()),
        "p95": float(np.percentile(sizes, 95)),
        "n_with_at_least_5": int((sizes >= 5).sum()),
    }


# ---------------------------------------------------------------------------
# Gene importance -- REQUIRES train_idx (this is the B3 fix)
# ---------------------------------------------------------------------------
def _variance_chunked(X_int8: np.ndarray, chunk_size: int = 10_000) -> np.ndarray:
    """Per-feature variance in column chunks (B8-style memory safety: a full
    float32 cast of e.g. 4872 x 535439 int8 is ~10 GB)."""
    n_feat = X_int8.shape[1]
    out = np.zeros(n_feat, dtype=np.float32)
    for start in range(0, n_feat, chunk_size):
        end = min(start + chunk_size, n_feat)
        out[start:end] = X_int8[:, start:end].astype(np.float32).var(axis=0)
    return out


def _chi2_chunked(X_int8: np.ndarray, y: np.ndarray, chunk_size: int = 10_000) -> np.ndarray:
    """Per-feature chi² statistic vs. binary y, column-chunked. chi2 is
    computed independently per feature, so chunking is exact."""
    n_feat = X_int8.shape[1]
    stats = np.zeros(n_feat, dtype=np.float32)
    for start in range(0, n_feat, chunk_size):
        end = min(start + chunk_size, n_feat)
        chi2_stats, _ = chi2(X_int8[:, start:end].astype(np.float32), y)
        stats[start:end] = np.nan_to_num(chi2_stats, nan=0.0, posinf=0.0, neginf=0.0)
    return stats


def compute_gene_importance_for_antibiotic(
    X_snp: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    gene_to_snp_indices: dict[str, list[int]],
    *,
    prefilter_top_k: Optional[int] = None,
    min_variance: Optional[float] = None,
    xgb_params: Optional[dict] = None,
) -> dict[str, float]:
    """Per-gene XGBoost-gain importance, fit strictly on `train_idx`.

    `train_idx` is a REQUIRED positional argument (not keyword-optional with
    a "use everything" default) -- this function cannot be called against
    the full sample set by accident. Every array indexing operation below
    only ever touches rows in `train_idx`; `y` is expected already aligned
    to `X_snp`'s row order (mask externally for missing labels before
    calling, same contract as the existing pipeline).
    """
    if train_idx is None:
        raise TypeError(
            "compute_gene_importance_for_antibiotic() requires train_idx "
            "explicitly -- see module docstring (B3) for why fitting gene "
            "importance on anything but the training partition is a "
            "feature-selection leakage bug."
        )
    import xgboost as xgb  # local import, keeps this module importable without it

    prefilter_top_k = prefilter_top_k or _params["prefilter_chi2_top_k"]
    min_variance = min_variance if min_variance is not None else _params["prefilter_min_variance"]
    xgb_params = dict(xgb_params or _params["xgb_params"])

    y_tr = y[train_idx]
    if len(np.unique(y_tr)) < 2:
        return {}

    # Column-chunked, same reasoning as B8: casting all ~535K int8 columns
    # to float32 for ~4-5K training rows at once is a multi-GB spike
    # (n_train * n_snps * 4 bytes). Chunking keeps this bounded regardless
    # of how many SNPs are passed in.
    X_tr_int8 = X_snp[train_idx]
    var = _variance_chunked(X_tr_int8)
    keep_var = np.where(var > min_variance)[0]
    if len(keep_var) == 0:
        return {}

    chi2_stats = _chi2_chunked(X_tr_int8[:, keep_var], y_tr)

    k = min(prefilter_top_k, len(keep_var))
    top_in_var = np.argpartition(chi2_stats, -k)[-k:] if k < len(keep_var) else np.arange(len(keep_var))
    top_orig_idx = keep_var[top_in_var]

    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    params = dict(xgb_params)
    params["scale_pos_weight"] = n_neg / max(n_pos, 1)
    early_stop = params.pop("early_stopping_rounds", 30)
    n_estimators = params.pop("n_estimators", 500)

    # Only the final top-K columns (bounded by prefilter_top_k, default 50K)
    # are ever materialized as float32 -- this is the size XGBoost actually
    # needs, not the full SNP matrix.
    dtrain = xgb.DMatrix(X_tr_int8[:, top_orig_idx].astype(np.float32), label=y_tr)
    booster = xgb.train(
        params, dtrain, num_boost_round=n_estimators,
        evals=[(dtrain, "train")], early_stopping_rounds=early_stop, verbose_eval=False,
    )

    importances_full = np.zeros(X_snp.shape[1], dtype=np.float32)
    for feat_str, score in booster.get_score(importance_type="gain").items():
        pos = int(feat_str[1:])
        importances_full[top_orig_idx[pos]] = float(score)

    gene_importance = {}
    for gene_id, idxs in gene_to_snp_indices.items():
        gene_importance[gene_id] = float(importances_full[idxs].sum())
    return gene_importance


def filter_genes(
    gene_to_snp_indices: dict[str, list[int]],
    gene_importance: dict[str, float],
    *,
    min_snps_per_gene: Optional[int] = None,
    top_k_by_importance: Optional[int] = None,
    keep_genes: Optional[set[str]] = None,
) -> list[str]:
    """`(variability_pass ∩ importance_top_k) ∪ keep_genes`.

    `keep_genes` is where a CARD-derived "always keep known AMR genes" set
    would plug in -- NOT populated by this refactor (CARD integration was
    never built in the original pipeline either; see docs/OPEN_QUESTIONS.md).
    Passing it explicitly is how a future caller adds that safety net
    without this module needing to know anything about CARD itself.
    """
    min_snps_per_gene = min_snps_per_gene or _params["min_snps_per_gene"]
    top_k_by_importance = top_k_by_importance or _params["final_gene_cap"]

    variability_pass = {g for g, idxs in gene_to_snp_indices.items() if len(idxs) >= min_snps_per_gene}
    valid_for_importance = variability_pass & set(gene_importance.keys())
    sorted_by_imp = sorted(valid_for_importance, key=lambda g: gene_importance.get(g, 0.0), reverse=True)
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
        gene_to_range[gene_id] = (start, len(new_indices))
    idx_arr = np.array(new_indices, dtype=np.int64)
    return X_snp[:, idx_arr], snp_positions[idx_arr], gene_to_range


def select_genes_for_split(
    X_snp: np.ndarray,
    snp_positions: np.ndarray,
    Y: np.ndarray,
    label_mask: np.ndarray,
    label_names: list[str],
    train_idx: np.ndarray,
    split_name: str,
    *,
    keep_genes: Optional[set[str]] = None,
) -> dict:
    """Full Stage 1.5 pipeline for ONE split regime, gene selection fit
    strictly on that split's `train_idx`. `split_name` is recorded in the
    output so two different splits' filtered artifacts are never
    confusable (and, per config/paths.yaml, are saved to different files:
    `arcadia_v2_gene_filtered__{split_name}.pkl`).
    """
    if train_idx is None:
        raise TypeError("select_genes_for_split() requires train_idx (B3) -- see module docstring.")

    gene_to_snp_indices = build_gene_to_snp_ranges(snp_positions)
    stats_full = get_gene_size_stats(gene_to_snp_indices)

    gene_importance_combined: dict[str, float] = {g: 0.0 for g in gene_to_snp_indices}
    per_ab_status = {}
    for j, ab in enumerate(label_names):
        m = label_mask[:, j]
        tr_use = train_idx[m[train_idx]]
        if len(np.unique(Y[tr_use, j])) < 2:
            per_ab_status[ab] = "skipped_no_variation"
            continue
        gi = compute_gene_importance_for_antibiotic(
            X_snp, Y[:, j], tr_use, gene_to_snp_indices,
        )
        for g, score in gi.items():
            gene_importance_combined[g] += score
        per_ab_status[ab] = "completed"

    selected_genes = filter_genes(gene_to_snp_indices, gene_importance_combined, keep_genes=keep_genes)
    X_filtered, positions_filtered, gene_to_range = build_filtered_snp_matrix(
        X_snp, snp_positions, gene_to_snp_indices, selected_genes
    )

    filtered_sizes = np.array([e - s for s, e in gene_to_range.values()])

    return {
        "split_name": split_name,
        "X_snp_filtered": X_filtered,
        "snp_positions_filtered": positions_filtered,
        "selected_gene_ids": np.array(selected_genes),
        "gene_to_range": gene_to_range,
        "n_genes": len(selected_genes),
        "n_snps_filtered": int(X_filtered.shape[1]),
        "n_snps_original": int(X_snp.shape[1]),
        "per_antibiotic_status": per_ab_status,
        "filter_metadata": {
            "min_snps_per_gene": _params["min_snps_per_gene"],
            "final_gene_cap": _params["final_gene_cap"],
            "gene_filter_formula": "(variability_pass ∩ importance_top_k) ∪ keep_genes",
            "fit_on": f"train_idx of split '{split_name}' ONLY (B3 fix)",
            "n_genes_original": stats_full["n_genes"],
            "n_genes_after_variability": stats_full["n_with_at_least_5"],
            "n_genes_selected": len(selected_genes),
            "stats_filtered_sizes": {
                "min": int(filtered_sizes.min()) if len(filtered_sizes) else None,
                "median": float(np.median(filtered_sizes)) if len(filtered_sizes) else None,
                "mean": float(filtered_sizes.mean()) if len(filtered_sizes) else None,
                "max": int(filtered_sizes.max()) if len(filtered_sizes) else None,
            },
        },
        "stage": "1.5_filtered_per_split_regime",
        "version": "v2_refactor",
    }
