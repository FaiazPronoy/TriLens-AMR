from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trilens.data.gene_selection_rules import build_gene_to_snp_ranges, get_gene_size_stats
from trilens.experiments.gene_importance_ranking import (
    _aggregate_snp_importance_to_gene,
    _train_xgb_per_antibiotic,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "results" / "gene_ranking_phylogroup"

ANTIBIOTICS = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
PREFILTER_TOP_K = 50_000
MIN_VARIANCE = 1e-6
XGB_PARAMS = dict(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=1.0,
    colsample_bynode=0.5,
    min_child_weight=1,
    tree_method="hist",
    device="cuda",
    early_stopping_rounds=30,
    eval_metric="aucpr",
)


def _load_data_phylogeny_split() -> dict:
    print(f"  Loading SNP matrix from {PROCESSED_DIR / 'arcadia_v2_snp_matrix.pkl'}...")
    t0 = time.time()
    with open(PROCESSED_DIR / "arcadia_v2_snp_matrix.pkl", "rb") as f:
        snp_data = pickle.load(f)
    print(f"    Loaded in {time.time() - t0:.1f}s")

    with open(PROCESSED_DIR / "arcadia_v2_master_features.pkl", "rb") as f:
        master = pickle.load(f)

    X_snp = snp_data["X_snp"]
    snp_positions = np.array(snp_data["snp_positions"])

    ps = master["phylogeny_split"]
    train_idx = np.array(ps["train_idx"])
    val_idx = np.array(ps["val_idx"])
    test_idx = np.array(ps["test_idx"])

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


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STAGE 1.5a (phylogeny_split, clean) -- XGBoost Gene Importance per Antibiotic")
    print("=" * 80)
    print(f"  Prefilter: top_k={PREFILTER_TOP_K:,}, min_variance={MIN_VARIANCE}")

    print("\n[1/3] Loading Arcadia data (phylogeny_split train/val)...")
    data = _load_data_phylogeny_split()
    X_snp = data["X_snp"]
    snp_positions = data["snp_positions"]
    n_samples, n_snps = X_snp.shape
    print(f"  Data shape: ({n_samples:,}, {n_snps:,})")

    print("\n[2/3] Building gene -> SNP index mapping...")
    t0 = time.time()
    gene_to_snp_indices, _ = build_gene_to_snp_ranges(snp_positions)
    stats = get_gene_size_stats(gene_to_snp_indices)
    print(f"  Built in {time.time() - t0:.1f}s")
    print(f"  Total genes: {stats['n_genes']:,}  Genes with >=5 SNPs: {stats['n_with_at_least_5']:,}")

    with open(OUT_DIR / "gene_to_snp_indices.pkl", "wb") as f:
        pickle.dump({"gene_to_snp_indices": gene_to_snp_indices, "stats": stats}, f)

    print("\n[3/3] Training XGBoost per antibiotic (phylogeny_split train/val)...")
    label_names = data["label_names"]
    per_ab_results = {}
    snp_importance_table = np.zeros((n_snps, len(ANTIBIOTICS)), dtype=np.float32)

    for k, ab in enumerate(ANTIBIOTICS):
        print(f"\n  [{k + 1}/{len(ANTIBIOTICS)}] Training for {ab}...")
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
            xgb_params=XGB_PARAMS,
            prefilter_top_k=PREFILTER_TOP_K,
            min_variance=MIN_VARIANCE,
        )
        if importances is None:
            per_ab_results[ab] = meta
            continue
        per_ab_results[ab] = meta
        snp_importance_table[:, k] = importances

        gene_importance = _aggregate_snp_importance_to_gene(importances, gene_to_snp_indices, aggregation="sum")
        with open(OUT_DIR / f"xgb_importance_{ab}.pkl", "wb") as f:
            pickle.dump(
                {"antibiotic": ab, "snp_importance": importances, "gene_importance": gene_importance, "metadata": meta},
                f,
            )
        print(f"    Val PR-AUC: {meta['val_pr_auc']:.4f}  Non-zero features: {meta['n_features_used']:,}")

    print("\n[Aggregating] Building combined gene importance...")
    gene_importance_combined = {g: {"n_snps": len(idxs)} for g, idxs in gene_to_snp_indices.items()}
    for ab in ANTIBIOTICS:
        ab_path = OUT_DIR / f"xgb_importance_{ab}.pkl"
        if not ab_path.exists():
            continue
        with open(ab_path, "rb") as f:
            ab_data = pickle.load(f)
        for gene_id, score in ab_data["gene_importance"].items():
            gene_importance_combined[gene_id][f"imp_{ab}"] = score

    for gene_id in gene_importance_combined:
        scores = [gene_importance_combined[gene_id].get(f"imp_{ab}", 0.0) for ab in ANTIBIOTICS]
        gene_importance_combined[gene_id]["imp_max"] = max(scores)
        gene_importance_combined[gene_id]["imp_sum"] = sum(scores)
        gene_importance_combined[gene_id]["imp_mean"] = sum(scores) / len(scores)

    rows = [{"gene_id": g, **info} for g, info in gene_importance_combined.items()]
    df = pd.DataFrame(rows).sort_values("imp_sum", ascending=False)
    csv_path = OUT_DIR / "xgb_gene_importance.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    summary = {
        "split_used_for_importance": "phylogeny_split",
        "n_genes": stats["n_genes"],
        "n_genes_filter_5snps": stats["n_with_at_least_5"],
        "per_antibiotic_metadata": per_ab_results,
        "gene_size_stats": stats,
        "prefilter": {"top_k": PREFILTER_TOP_K, "min_variance": MIN_VARIANCE},
    }
    with open(OUT_DIR / "xgb_importance_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  STAGE 1.5a (phylogeny_split, clean) COMPLETE -- outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
