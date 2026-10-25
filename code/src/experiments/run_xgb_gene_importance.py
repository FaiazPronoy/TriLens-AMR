"""Stage 1.5 — Compute XGBoost gene importance per antibiotic.

Pre-filters 535K SNPs → top-50K by chi-squared statistic → trains XGBoost
on the pre-filtered subset → remaps feature importances back to full SNP
space → aggregates SNP importance to gene level.

Why pre-filter?
    535K features × ~3-4K samples is too high-dimensional: XGBoost with
    colsample_bytree=0.5 still samples 267K features per tree and finds no
    splits with positive gain → all importances zero, val PR-AUC at base
    rate. Chi-squared screening reduces dimensionality 10x while keeping
    biologically meaningful signal (it ranks each SNP's marginal association
    with the binary phenotype, which is the standard GWAS first-pass).

Output: results/stage1_5/xgb_gene_importance.csv with per-gene aggregated
scores across all antibiotics.

Runtime estimate: 5-15 min per antibiotic on RTX 4080 Super = 45 min - 2 h total.
"""
from __future__ import annotations

import gc
import json
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig
from sklearn.feature_selection import chi2
from sklearn.metrics import average_precision_score

from src.data.gene_filtering import (
    build_gene_to_snp_ranges,
    get_gene_size_stats,
)


# ───────────────────────────────────────────────────────────────────────────────
# Data loading
# ───────────────────────────────────────────────────────────────────────────────
def _load_arcadia_data(
    snp_pkl_path: str,
    master_pkl_path: str,
) -> dict:
    """Load Arcadia SNP matrix + labels + splits."""
    print(f"  Loading SNP matrix from {snp_pkl_path}...")
    t0 = time.time()
    with open(snp_pkl_path, "rb") as f:
        snp_data = pickle.load(f)
    print(f"    Loaded in {time.time() - t0:.1f}s")

    print(f"  Loading master pkl from {master_pkl_path}...")
    with open(master_pkl_path, "rb") as f:
        master = pickle.load(f)

    X_snp = snp_data["X_snp"]
    snp_positions = np.array(snp_data["snp_positions"])

    rs = master["random_split"]
    train_idx = np.array(rs["train_idx"])
    val_idx = np.array(rs["val_idx"])
    test_idx = np.array(rs["test_idx"])

    Y = master["Y"].astype(np.int64)
    label_mask = master["label_mask"].astype(bool)
    label_names = [str(x) for x in master["label_names"]]

    return {
        "X_snp": X_snp,
        "snp_positions": snp_positions,
        "Y": Y,
        "label_mask": label_mask,
        "label_names": label_names,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


# ───────────────────────────────────────────────────────────────────────────────
# Chunked variance + chi-squared (avoid 8 GB float32 cast of full SNP matrix)
# ───────────────────────────────────────────────────────────────────────────────
def _variance_chunked(
    X_int8: np.ndarray, chunk_size: int = 10_000
) -> np.ndarray:
    """Per-feature variance, computed in column chunks.

    Cost analysis for 4K samples × 535K int8 SNPs:
        Full float32 cast:     8.4 GB   (OOM on most lab PCs)
        Chunk of 10K columns:  160 MB   (always safe)
    """
    n_feat = X_int8.shape[1]
    out = np.zeros(n_feat, dtype=np.float32)
    for start in range(0, n_feat, chunk_size):
        end = min(start + chunk_size, n_feat)
        chunk = X_int8[:, start:end].astype(np.float32)
        out[start:end] = chunk.var(axis=0)
    return out


def _chi2_chunked(
    X_int8: np.ndarray, y: np.ndarray, chunk_size: int = 10_000
) -> np.ndarray:
    """Per-feature chi-squared statistic vs binary y, column-chunked.

    chi2 is computed independently per feature, so chunking is exact (not an
    approximation). Values must be non-negative — SNPs in {0,1,2,3} satisfy
    that natively.
    """
    n_feat = X_int8.shape[1]
    stats = np.zeros(n_feat, dtype=np.float32)
    for start in range(0, n_feat, chunk_size):
        end = min(start + chunk_size, n_feat)
        chunk = X_int8[:, start:end].astype(np.float32)
        chi2_stats, _ = chi2(chunk, y)
        stats[start:end] = np.nan_to_num(
            chi2_stats, nan=0.0, posinf=0.0, neginf=0.0
        )
    return stats


# ───────────────────────────────────────────────────────────────────────────────
# Main: prefilter → train XGB → remap importance to full SNP space
# ───────────────────────────────────────────────────────────────────────────────
def _train_xgb_per_antibiotic(
    X_snp: np.ndarray,          # full (n_samples, n_snps_total) int8
    Y: np.ndarray,
    label_mask: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    ab_idx: int,
    xgb_params: dict,
    *,
    prefilter_top_k: int = 50_000,
    min_variance: float = 1e-6,
) -> tuple[Optional[np.ndarray], dict]:
    """Train XGBoost on chi²-prefiltered SNPs; return importances on full space.

    Pipeline:
        1. Sample masking      — keep only rows with label for this antibiotic.
        2. Variance filter     — drop SNPs that are constant on the train set.
        3. Chi-squared rank    — score each remaining SNP's association with y.
        4. Top-K selection     — keep the prefilter_top_k highest-chi² SNPs.
        5. XGBoost training    — fit on the (n_train, K) matrix.
        6. Importance remap    — write XGB gains back into the full SNP index.

    Returns:
        importances: float32 array of shape (n_snps_total,).
            Zero everywhere except for SNPs that XGBoost actually used.
        metadata: dict with training stats, prefilter params, and val PR-AUC.
    """
    n_features_total = X_snp.shape[1]

    # ── Step 1: sample masking ────────────────────────────────────────────────
    m = label_mask[:, ab_idx]
    tr_m = m[train_idx]
    va_m = m[val_idx]
    tr_use = train_idx[tr_m]
    va_use = val_idx[va_m]
    y_tr = Y[tr_use, ab_idx]
    y_va = Y[va_use, ab_idx]

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
        return None, {"status": "skipped_no_variation"}

    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    pos_weight = n_neg / max(n_pos, 1)

    print(f"      Train: {len(tr_use)} samples ({n_pos} pos), "
          f"Val: {len(va_use)}")
    print(f"      n_features_total: {n_features_total:,}, "
          f"pos_weight: {pos_weight:.2f}")

    # ── Step 2: variance filter on train set ─────────────────────────────────
    t0 = time.time()
    X_tr_int8 = X_snp[tr_use]               # int8 (n_tr, n_features_total)
    var = _variance_chunked(X_tr_int8)
    keep_var_mask = var > min_variance
    n_after_var = int(keep_var_mask.sum())
    print(f"      [filter 1] variance > {min_variance}: "
          f"{n_after_var:,} / {n_features_total:,} kept "
          f"({time.time() - t0:.1f}s)")
    if n_after_var == 0:
        return None, {"status": "skipped_no_variance"}

    keep_var_indices = np.where(keep_var_mask)[0]   # indices into full space

    # ── Step 3: chi-squared rank on variance-passing features ────────────────
    t0 = time.time()
    X_tr_var = X_tr_int8[:, keep_var_indices]       # still int8
    chi2_stats = _chi2_chunked(X_tr_var, y_tr)
    print(f"      [filter 2] chi² over {n_after_var:,} features "
          f"({time.time() - t0:.1f}s, max={chi2_stats.max():.2f}, "
          f"median={float(np.median(chi2_stats)):.4f})")

    # ── Step 4: top-K selection ───────────────────────────────────────────────
    actual_k = int(min(prefilter_top_k, n_after_var))
    if actual_k < n_after_var:
        top_k_in_var_space = np.argpartition(chi2_stats, -actual_k)[-actual_k:]
    else:
        top_k_in_var_space = np.arange(n_after_var)
    # Sort descending by chi² for clean logging (XGB doesn't care about order)
    top_k_in_var_space = top_k_in_var_space[
        np.argsort(chi2_stats[top_k_in_var_space])[::-1]
    ]
    top_k_orig_idx = keep_var_indices[top_k_in_var_space]
    chi2_cutoff = float(chi2_stats[top_k_in_var_space[-1]])
    print(f"      [filter 3] top-{actual_k:,} retained "
          f"(chi² cutoff = {chi2_cutoff:.2f})")

    # Free large intermediates before XGB allocates GPU memory
    del X_tr_int8, X_tr_var, var, chi2_stats
    gc.collect()

    # ── Step 5: build prefiltered matrices and train XGBoost ─────────────────
    X_tr_top = X_snp[tr_use][:, top_k_orig_idx].astype(np.float32)
    X_va_top = X_snp[va_use][:, top_k_orig_idx].astype(np.float32)

    params = dict(xgb_params)
    params["scale_pos_weight"] = pos_weight
    params["objective"] = "binary:logistic"
    # Now that we're at ~50K features (not 535K), let XGB consider more per tree:
    params.setdefault("colsample_bytree", 1.0)
    params.setdefault("colsample_bynode", 0.5)   # per-split diversity instead
    params.setdefault("min_child_weight", 1)
    early_stop = params.pop("early_stopping_rounds", 30)
    n_estimators = params.pop("n_estimators", 500)
    # n_jobs is ignored when device=cuda; pop to silence XGB warning
    params.pop("n_jobs", None)

    dtrain = xgb.DMatrix(X_tr_top, label=y_tr)
    dval = xgb.DMatrix(X_va_top, label=y_va)

    t0 = time.time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=early_stop,
        verbose_eval=100,
    )
    train_time = time.time() - t0

    # ── Step 6: remap importance back to full SNP space ──────────────────────
    importance_dict = booster.get_score(importance_type="gain")
    importances_full = np.zeros(n_features_total, dtype=np.float32)
    for feat_str, score in importance_dict.items():
        if feat_str.startswith("f"):
            top_k_pos = int(feat_str[1:])              # position in 0..K-1
            orig_pos = int(top_k_orig_idx[top_k_pos])  # position in 0..n_total
            importances_full[orig_pos] = float(score)

    y_pred = booster.predict(dval)
    val_pr_auc = float(average_precision_score(y_va, y_pred))

    metadata = {
        "status": "completed",
        "train_time_sec": train_time,
        "val_pr_auc": val_pr_auc,
        "n_features_used": int((importances_full > 0).sum()),
        "best_iteration": booster.best_iteration,
        "n_train": int(len(tr_use)),
        "n_val": int(len(va_use)),
        "prefilter": {
            "n_total": int(n_features_total),
            "n_after_variance_filter": int(n_after_var),
            "n_after_chi2_topk": int(actual_k),
            "chi2_cutoff": chi2_cutoff,
            "min_variance": float(min_variance),
            "prefilter_top_k": int(prefilter_top_k),
        },
    }

    # Cleanup before next antibiotic
    del X_tr_top, X_va_top, dtrain, dval, booster
    gc.collect()

    return importances_full, metadata


# ───────────────────────────────────────────────────────────────────────────────
# Per-gene aggregation (unchanged)
# ───────────────────────────────────────────────────────────────────────────────
def _aggregate_snp_importance_to_gene(
    snp_importances: np.ndarray,
    gene_to_snp_indices: dict[str, list[int]],
    aggregation: str = "sum",
) -> dict[str, float]:
    """Aggregate per-SNP importance scores to per-gene scores."""
    result = {}
    for gene_id, indices in gene_to_snp_indices.items():
        scores = snp_importances[indices]
        if aggregation == "sum":
            result[gene_id] = float(scores.sum())
        elif aggregation == "mean":
            result[gene_id] = float(scores.mean())
        elif aggregation == "max":
            result[gene_id] = float(scores.max())
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return result


# ───────────────────────────────────────────────────────────────────────────────
# Hydra entry-point
# ───────────────────────────────────────────────────────────────────────────────
def run(cfg: DictConfig) -> None:
    """Main entry point for XGB gene importance experiment."""
    exp_cfg = cfg.experiment
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 1.5 — XGBoost Gene Importance per Antibiotic")
    print("=" * 80)

    # Prefilter parameters (config-driven; safe defaults)
    prefilter_cfg = exp_cfg.get("prefilter", {}) or {}
    prefilter_top_k = int(prefilter_cfg.get("top_k", 50_000))
    min_variance = float(prefilter_cfg.get("min_variance", 1e-6))
    print(f"  Prefilter: top_k={prefilter_top_k:,}, "
          f"min_variance={min_variance}")

    # Load data once
    print("\n[1/3] Loading Arcadia data...")
    data = _load_arcadia_data(
        snp_pkl_path=cfg.paths.arcadia_snp_matrix_pkl,
        master_pkl_path=cfg.paths.arcadia_master_pkl,
    )
    X_snp = data["X_snp"]
    snp_positions = data["snp_positions"]

    n_samples, n_snps = X_snp.shape
    print(f"  Data shape: ({n_samples:,}, {n_snps:,})")
    print(f"  Memory: {X_snp.nbytes / 1e9:.1f} GB")

    # Build gene mapping
    print("\n[2/3] Building gene -> SNP index mapping...")
    t0 = time.time()
    gene_to_snp_indices, _ = build_gene_to_snp_ranges(snp_positions)
    stats = get_gene_size_stats(gene_to_snp_indices)
    print(f"  Built in {time.time() - t0:.1f}s")
    print(f"  Total genes: {stats['n_genes']:,}")
    print(f"  Genes with ≥5 SNPs: {stats['n_with_at_least_5']:,}")
    print(f"  Median SNPs/gene: {stats['median_size']:.1f}")
    print(f"  Max SNPs in a gene: {stats['max_size']}")

    mapping_path = out_dir / "gene_to_snp_indices.pkl"
    with open(mapping_path, "wb") as f:
        pickle.dump({
            "gene_to_snp_indices": gene_to_snp_indices,
            "stats": stats,
        }, f)
    print(f"  Saved: {mapping_path}")

    # XGB per antibiotic
    print("\n[3/3] Training XGBoost per antibiotic...")

    antibiotics = list(exp_cfg.antibiotics)
    label_names = data["label_names"]

    per_ab_results = {}
    snp_importance_table = np.zeros((n_snps, len(antibiotics)), dtype=np.float32)

    xgb_params = dict(exp_cfg.xgb_params)

    for k, ab in enumerate(antibiotics):
        print(f"\n  [{k+1}/{len(antibiotics)}] Training for {ab}...")

        if ab not in label_names:
            print(f"    WARNING: {ab} not in label_names, skipping")
            continue
        ab_idx = label_names.index(ab)

        importances, meta = _train_xgb_per_antibiotic(
            X_snp=X_snp,
            Y=data["Y"],
            label_mask=data["label_mask"],
            train_idx=data["train_idx"],
            val_idx=data["val_idx"],
            ab_idx=ab_idx,
            xgb_params=xgb_params,
            prefilter_top_k=prefilter_top_k,
            min_variance=min_variance,
        )

        if importances is None:
            per_ab_results[ab] = meta
            continue

        per_ab_results[ab] = meta
        snp_importance_table[:, k] = importances

        gene_importance = _aggregate_snp_importance_to_gene(
            importances, gene_to_snp_indices, aggregation="sum"
        )

        ab_path = out_dir / f"xgb_importance_{ab}.pkl"
        with open(ab_path, "wb") as f:
            pickle.dump({
                "antibiotic": ab,
                "snp_importance": importances,
                "gene_importance": gene_importance,
                "metadata": meta,
            }, f)
        print(f"    Saved: {ab_path}")
        print(f"    Val PR-AUC: {meta['val_pr_auc']:.4f}")
        print(f"    Non-zero features: {meta['n_features_used']:,}")

    # Build combined gene importance table
    print("\n[Aggregating] Building combined gene importance...")

    gene_importance_combined = {}
    for gene_id in gene_to_snp_indices.keys():
        gene_importance_combined[gene_id] = {
            "n_snps": len(gene_to_snp_indices[gene_id]),
        }

    for ab in antibiotics:
        ab_path = out_dir / f"xgb_importance_{ab}.pkl"
        if not ab_path.exists():
            continue
        with open(ab_path, "rb") as f:
            ab_data = pickle.load(f)
        for gene_id, score in ab_data["gene_importance"].items():
            gene_importance_combined[gene_id][f"imp_{ab}"] = score

    for gene_id in gene_importance_combined:
        scores = [
            gene_importance_combined[gene_id].get(f"imp_{ab}", 0.0)
            for ab in antibiotics
        ]
        gene_importance_combined[gene_id]["imp_max"] = max(scores)
        gene_importance_combined[gene_id]["imp_sum"] = sum(scores)
        gene_importance_combined[gene_id]["imp_mean"] = sum(scores) / len(scores)

    rows = [{"gene_id": g, **info} for g, info in gene_importance_combined.items()]
    df = pd.DataFrame(rows).sort_values("imp_sum", ascending=False)
    csv_path = out_dir / "xgb_gene_importance.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")
    print(f"  Top-10 genes by sum-importance:")
    print(df.head(10)[["gene_id", "n_snps", "imp_sum", "imp_max"]]
          .to_string(index=False))

    summary = {
        "n_genes": stats["n_genes"],
        "n_genes_filter_5snps": stats["n_with_at_least_5"],
        "per_antibiotic_metadata": per_ab_results,
        "gene_size_stats": stats,
        "prefilter": {
            "top_k": prefilter_top_k,
            "min_variance": min_variance,
        },
    }
    with open(out_dir / "xgb_importance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✅ Stage 1.5 XGB importance COMPLETE")
    print(f"  Outputs → {out_dir}")
