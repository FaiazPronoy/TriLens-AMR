# src/experiments/run_stage2_gene_xgb.py
"""Stage 2 lite — XGBoost on gene-aggregated features.

This is the simpler alternative to the full AMR-PhyloGene-MTL neural
architecture. It uses the same Stage 1.5 gene selection but represents
each gene as a small bundle of summary statistics (mean, variance,
n_nonref, max). The result is a ~20K-feature tabular matrix that
tree-based models can train on directly.

Outputs:
    {output_dir}/stage2_gene_xgb_results.csv     — per-AB metrics + CIs
    {output_dir}/stage2_gene_xgb_predictions_{AB}.npz
    {output_dir}/SUMMARY.md
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score, matthews_corrcoef

from src.data.gene_aggregated_features import (
    GENE_STAT_NAMES, build_gene_feature_matrix_from_pkl,
)
from src.experiments._shared import bootstrap_mcc, bootstrap_pr_auc, compute_metrics
from src.utils.seeds import set_global_seed


def _train_one_ab(
    *,
    X: np.ndarray,
    Y: np.ndarray,
    label_mask: np.ndarray,
    splits: dict,
    ab_idx: int,
    ab_name: str,
    xgb_params: dict,
    use_phylogroup: bool,
    phylogroup_oh: np.ndarray | None,
    eval_cfg: DictConfig,
) -> dict:
    """Train + evaluate XGBoost for one antibiotic."""
    tr_idx = np.asarray(splits["train_idx"])
    va_idx = np.asarray(splits["val_idx"])
    te_idx = np.asarray(splits["test_idx"])

    m = label_mask[:, ab_idx]
    tr = tr_idx[m[tr_idx]]
    va = va_idx[m[va_idx]]
    te = te_idx[m[te_idx]]

    y_tr = Y[tr, ab_idx].astype(np.int32)
    y_va = Y[va, ab_idx].astype(np.int32)
    y_te = Y[te, ab_idx].astype(np.int32)

    X_tr = X[tr]
    X_va = X[va]
    X_te = X[te]

    if use_phylogroup and phylogroup_oh is not None:
        X_tr = np.concatenate([X_tr, phylogroup_oh[tr]], axis=1).astype(np.float32)
        X_va = np.concatenate([X_va, phylogroup_oh[va]], axis=1).astype(np.float32)
        X_te = np.concatenate([X_te, phylogroup_oh[te]], axis=1).astype(np.float32)

    n_pos = int(y_tr.sum())
    n_neg = int(len(y_tr) - n_pos)
    pos_weight = n_neg / max(n_pos, 1)

    params = dict(xgb_params)
    params["scale_pos_weight"] = pos_weight
    params["objective"] = "binary:logistic"
    early_stop = params.pop("early_stopping_rounds", 200)
    eval_metric = params.pop("eval_metric", "aucpr")

    print(f"      Train={len(tr)} ({n_pos} pos)  Val={len(va)}  Test={len(te)}  "
          f"n_features={X_tr.shape[1]:,}  pos_weight={pos_weight:.2f}")

    try:
        clf = xgb.XGBClassifier(
            **params,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stop,
            random_state=42,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    except Exception as ex:
        print(f"      GPU XGB failed ({ex!r}); falling back to CPU.")
        cpu_params = dict(params)
        cpu_params["device"] = "cpu"
        clf = xgb.XGBClassifier(
            **cpu_params,
            eval_metric=eval_metric,
            early_stopping_rounds=early_stop,
            random_state=42,
            verbosity=0,
        )
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

    # Test predictions
    y_prob = clf.predict_proba(X_te)[:, 1]
    metrics = compute_metrics(y_te, y_prob, threshold=0.5)
    pr_lo, pr_hi = bootstrap_pr_auc(
        y_te, y_prob,
        n_boot=eval_cfg.bootstrap_n, seed=eval_cfg.bootstrap_seed,
    )
    mcc_lo, mcc_hi = bootstrap_mcc(
        y_te, y_prob,
        n_boot=eval_cfg.bootstrap_n, seed=eval_cfg.bootstrap_seed,
    )

    return {
        "y_true": y_te, "y_prob": y_prob,
        "metrics": metrics, "pr_auc_ci": (pr_lo, pr_hi), "mcc_ci": (mcc_lo, mcc_hi),
        "n_test": int(len(te)),
    }


def run(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    exp_cfg = cfg.experiment
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config_used.yaml")

    print("=" * 80)
    print("  STAGE 2 — XGBoost on gene-aggregated features")
    print("=" * 80)

    # Step 1: build gene-aggregated features
    print("\n[1/3] Building gene-aggregated features...")
    t0 = time.time()
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=cfg.paths.processed_dir + "/arcadia_v2_gene_filtered.pkl"
        if isinstance(cfg.paths.processed_dir, str)
        else str(cfg.paths.processed_dir) + "/arcadia_v2_gene_filtered.pkl",
        stats=tuple(exp_cfg.gene_stats),
    )
    print(f"  built in {time.time() - t0:.1f}s")

    X = bundle["X_gene_features"]
    Y = bundle["Y"].astype(np.int32)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]
    splits = bundle["random_split"]
    phylogroup_oh = bundle.get("phylogroup_onehot")
    if phylogroup_oh is not None:
        phylogroup_oh = phylogroup_oh.astype(np.float32)

    # Step 2: XGBoost per antibiotic (with and without phylogroup)
    print("\n[2/3] Training XGBoost per antibiotic...")
    rows: list[dict] = []
    xgb_params = dict(exp_cfg.xgb_params)
    use_pg_variants = list(exp_cfg.phylogroup_variants)  # e.g. [false, true]

    for ab in exp_cfg.antibiotics:
        if ab not in label_names:
            print(f"\n  skipping {ab} (not in label_names)")
            continue
        ab_idx = label_names.index(ab)

        for use_pg in use_pg_variants:
            tag = "with_pg" if use_pg else "no_pg"
            print(f"\n  -- {ab} × {tag} --")
            t0 = time.time()
            res = _train_one_ab(
                X=X, Y=Y, label_mask=label_mask, splits=splits,
                ab_idx=ab_idx, ab_name=ab,
                xgb_params=xgb_params,
                use_phylogroup=bool(use_pg),
                phylogroup_oh=phylogroup_oh,
                eval_cfg=exp_cfg.evaluation,
            )
            elapsed = time.time() - t0

            m = res["metrics"]
            row = {
                "antibiotic": ab,
                "variant": tag,
                "test_pr_auc": m["pr_auc"],
                "test_pr_auc_lo": res["pr_auc_ci"][0],
                "test_pr_auc_hi": res["pr_auc_ci"][1],
                "test_mcc": m["mcc"],
                "test_mcc_lo": res["mcc_ci"][0],
                "test_mcc_hi": res["mcc_ci"][1],
                "test_roc_auc": m["roc_auc"],
                "test_f1": m["f1"],
                "test_f2": m["f2"],
                "test_precision": m["precision"],
                "test_recall": m["recall"],
                "test_sens_at_90_spec": m["sens_at_90_spec"],
                "test_brier": m["brier"],
                "n_test": res["n_test"],
                "n_features": X.shape[1] + (phylogroup_oh.shape[1] if use_pg and phylogroup_oh is not None else 0),
                "fit_time_s": elapsed,
            }
            rows.append(row)
            print(f"    PR-AUC: {m['pr_auc']:.4f} "
                  f"[{res['pr_auc_ci'][0]:.4f}, {res['pr_auc_ci'][1]:.4f}]")
            print(f"    MCC:    {m['mcc']:.4f} "
                  f"[{res['mcc_ci'][0]:.4f}, {res['mcc_ci'][1]:.4f}]")

            np.savez_compressed(
                out_dir / f"stage2_gene_xgb_predictions_{ab}_{tag}.npz",
                y_true=res["y_true"], y_prob=res["y_prob"],
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage2_gene_xgb_results.csv", index=False)

    # Step 3: summary
    print("\n[3/3] Writing summary...")
    summary_lines = ["# Stage 2 — XGBoost on Gene-Aggregated Features\n"]
    summary_lines.append(f"Features per gene: {tuple(exp_cfg.gene_stats)}\n")
    summary_lines.append(f"Total feature dimension: {X.shape[1]:,}\n\n")
    summary_lines.append("## Per-antibiotic results\n")
    summary_lines.append("| AB | variant | PR-AUC | [lo, hi] | MCC | F2 |")
    summary_lines.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        summary_lines.append(
            f"| {r['antibiotic']} | {r['variant']} | {r['test_pr_auc']:.4f} | "
            f"[{r['test_pr_auc_lo']:.4f}, {r['test_pr_auc_hi']:.4f}] | "
            f"{r['test_mcc']:.4f} | {r['test_f2']:.4f} |"
        )
    summary_lines.append("")

    # Average PR-AUC across antibiotics
    for v in df["variant"].unique():
        sub = df[df["variant"] == v]
        summary_lines.append(f"\n**Avg PR-AUC ({v}):** {sub['test_pr_auc'].mean():.4f}")
        summary_lines.append(f"**Avg MCC ({v}):** {sub['test_mcc'].mean():.4f}")
    summary_lines.append("")

    (out_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("  STAGE 2 COMPLETE")
    print("=" * 80)
    for v in df["variant"].unique():
        sub = df[df["variant"] == v]
        print(f"  {v:<10}  avg PR-AUC = {sub['test_pr_auc'].mean():.4f}  "
              f"avg MCC = {sub['test_mcc'].mean():.4f}")
    print(f"\n  CSV     → {out_dir / 'stage2_gene_xgb_results.csv'}")
    print(f"  SUMMARY → {out_dir / 'SUMMARY.md'}")
