"""
src/experiments/run_xgb_gene_importance_mlst.py
=================================================
STEP 2a: Stage 1.5 gene-importance ranking using the MLST lineage holdout
split's OWN train/val partition (from build_mlst_lineage_holdout_split.py).

Identical logic to run_xgb_gene_importance.py / _phylogeny.py -- the same
unmodified `_train_xgb_per_antibiotic` and `_aggregate_snp_importance_to_gene`
are imported directly. The ONLY difference is which split supplies train_idx
and val_idx.

Locked hyperparameters are byte-identical to the other two arms (taken from
the original run's saved Hydra config,
experiments/2026-05-23_01-26-01_stage1_5_gene_filtering/.hydra/config.yaml),
because the comparison table is only meaningful if the gene-filtering method
is held constant and ONLY the split changes.

Output: results/stage1_5_mlst_clean/  (new folder; nothing existing touched)

Usage:
    conda activate amr
    python -m src.experiments.run_xgb_gene_importance_mlst
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.gene_filtering import build_gene_to_snp_ranges, get_gene_size_stats
from src.experiments.run_xgb_gene_importance import (
    _aggregate_snp_importance_to_gene,
    _train_xgb_per_antibiotic,
)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_PKL = PROCESSED_DIR / "mlst_lineage_holdout_split.pkl"
OUT_DIR = ROOT / "results" / "stage1_5_mlst_clean"

ANTIBIOTICS = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
PREFILTER_TOP_K = 50_000
MIN_VARIANCE = 1e-6
XGB_PARAMS = dict(
    n_estimators=500, max_depth=4, learning_rate=0.1, subsample=0.8,
    colsample_bytree=1.0, colsample_bynode=0.5, min_child_weight=1,
    tree_method="hist", device="cuda", early_stopping_rounds=30,
    eval_metric="aucpr",
)


def _load_data_mlst_split() -> dict:
    if not SPLIT_PKL.exists():
        raise FileNotFoundError(
            f"{SPLIT_PKL} not found -- run "
            f"src.experiments.build_mlst_lineage_holdout_split first."
        )
    with open(SPLIT_PKL, "rb") as f:
        sp = pickle.load(f)

    print(f"  Loading SNP matrix from {PROCESSED_DIR / 'arcadia_v2_snp_matrix.pkl'}...")
    t0 = time.time()
    with open(PROCESSED_DIR / "arcadia_v2_snp_matrix.pkl", "rb") as f:
        snp_data = pickle.load(f)
    print(f"    Loaded in {time.time() - t0:.1f}s")

    with open(PROCESSED_DIR / "arcadia_v2_master_features.pkl", "rb") as f:
        master = pickle.load(f)

    train_idx = np.asarray(sp["train_idx"])
    val_idx = np.asarray(sp["val_idx"])
    test_idx = np.asarray(sp["test_idx"])

    # Defensive: the ranking must never see val/test rows.
    assert not (set(train_idx.tolist()) & set(test_idx.tolist())), "train/test overlap"
    assert not (set(train_idx.tolist()) & set(val_idx.tolist())), "train/val overlap"

    return {
        "X_snp": snp_data["X_snp"],
        "snp_positions": np.array(snp_data["snp_positions"]),
        "Y": master["Y"].astype(np.int64),
        "label_mask": master["label_mask"].astype(bool),
        "label_names": [str(x) for x in master["label_names"]],
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 2a -- XGBoost gene importance on the MLST LINEAGE HOLDOUT train split")
    print("=" * 80)

    data = _load_data_mlst_split()
    X_snp = data["X_snp"]
    n_samples, n_snps = X_snp.shape
    print(f"  Data shape: ({n_samples:,}, {n_snps:,})")
    print(f"  train={len(data['train_idx'])}  val={len(data['val_idx'])}  "
          f"test={len(data['test_idx'])}  (test never used here)")

    print("\n  Building gene -> SNP index mapping...")
    gene_to_snp_indices, _ = build_gene_to_snp_ranges(data["snp_positions"])
    stats = get_gene_size_stats(gene_to_snp_indices)
    print(f"  Total genes: {stats['n_genes']:,}  "
          f"Genes with >=5 SNPs: {stats['n_with_at_least_5']:,}")

    with open(OUT_DIR / "gene_to_snp_indices.pkl", "wb") as f:
        pickle.dump({"gene_to_snp_indices": gene_to_snp_indices, "stats": stats}, f)

    label_names = data["label_names"]
    per_ab_results = {}

    for k, ab in enumerate(ANTIBIOTICS):
        print(f"\n  [{k+1}/{len(ANTIBIOTICS)}] Training for {ab}...")
        if ab not in label_names:
            print(f"    WARNING: {ab} not in label_names, skipping")
            continue
        ab_idx = label_names.index(ab)

        importances, meta = _train_xgb_per_antibiotic(
            X_snp=X_snp, Y=data["Y"], label_mask=data["label_mask"],
            train_idx=data["train_idx"], val_idx=data["val_idx"],
            ab_idx=ab_idx, xgb_params=XGB_PARAMS,
            prefilter_top_k=PREFILTER_TOP_K, min_variance=MIN_VARIANCE,
        )
        if importances is None:
            per_ab_results[ab] = meta
            continue
        per_ab_results[ab] = meta

        gene_importance = _aggregate_snp_importance_to_gene(
            importances, gene_to_snp_indices, aggregation="sum"
        )
        with open(OUT_DIR / f"xgb_importance_{ab}.pkl", "wb") as f:
            pickle.dump({"antibiotic": ab, "snp_importance": importances,
                         "gene_importance": gene_importance, "metadata": meta}, f)
        print(f"    Val PR-AUC: {meta['val_pr_auc']:.4f}  "
              f"Non-zero features: {meta['n_features_used']:,}")

    print("\n  [Aggregating] Building combined gene importance...")
    combined = {g: {"n_snps": len(idxs)} for g, idxs in gene_to_snp_indices.items()}
    for ab in ANTIBIOTICS:
        p = OUT_DIR / f"xgb_importance_{ab}.pkl"
        if not p.exists():
            continue
        with open(p, "rb") as f:
            ab_data = pickle.load(f)
        for gene_id, score in ab_data["gene_importance"].items():
            combined[gene_id][f"imp_{ab}"] = score

    for gene_id in combined:
        scores = [combined[gene_id].get(f"imp_{ab}", 0.0) for ab in ANTIBIOTICS]
        combined[gene_id]["imp_max"] = max(scores)
        combined[gene_id]["imp_sum"] = sum(scores)
        combined[gene_id]["imp_mean"] = sum(scores) / len(scores)

    rows = [{"gene_id": g, **info} for g, info in combined.items()]
    df = pd.DataFrame(rows).sort_values("imp_sum", ascending=False)
    df.to_csv(OUT_DIR / "xgb_gene_importance.csv", index=False)
    print(f"  Saved: {OUT_DIR / 'xgb_gene_importance.csv'}")

    with open(OUT_DIR / "xgb_importance_summary.json", "w") as f:
        json.dump({
            "split_used_for_importance": SPLIT_PKL.stem.removesuffix("_split") + "_split",
            "n_genes": stats["n_genes"],
            "n_genes_filter_5snps": stats["n_with_at_least_5"],
            "per_antibiotic_metadata": per_ab_results,
            "gene_size_stats": stats,
            "prefilter": {"top_k": PREFILTER_TOP_K, "min_variance": MIN_VARIANCE},
        }, f, indent=2)

    print(f"\n  STEP 2a COMPLETE -- outputs -> {OUT_DIR}")


if __name__ == "__main__":
    main()
