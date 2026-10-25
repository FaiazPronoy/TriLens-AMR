# src/experiments/run_stacking_ensemble.py
"""Three-model stacking ensemble for Stage 2.

Base learners (per antibiotic):
    1.  XGB on gene-aggregated features (20K features, 5K genes × 4 stats)
    2.  GeneAttentionNet (loaded from saved checkpoint)
    3.  XGB on raw chi²-prefiltered SNPs (top-50K per AB; same as Stage 1.5)

Meta-learner:
    Per-antibiotic Logistic Regression with three inputs (base-model
    probabilities). Trained on validation predictions, evaluated on test.

The three base learners use materially different feature spaces — gene-level
summaries, neural attention over those summaries, and raw SNP genotypes —
which is what makes stacking actually buy something here.

Outputs:
    {output_dir}/ensemble_results.csv            — per-AB metrics + CIs
    {output_dir}/ensemble_predictions.npz        — test probabilities
    {output_dir}/SUMMARY.md                      — markdown comparison table
"""
from __future__ import annotations

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

from src.data.gene_aggregated_features import build_gene_feature_matrix_from_pkl
from src.data.gene_attention_dataset import (
    GeneAttentionDataset, normalize_gene_features_per_stat, reshape_gene_features,
)
from src.experiments._shared import (
    bootstrap_mcc, bootstrap_pr_auc, compute_metrics,
)
from src.models.gene_attention_net import GeneAttentionNet
from src.utils.seeds import set_global_seed


# ───────────────────────────────────────────────────────────────────────────────
# Chunked variance + chi² (reused from Stage 1.5)
# ───────────────────────────────────────────────────────────────────────────────
def _variance_chunked(X_int8: np.ndarray, chunk: int = 10_000) -> np.ndarray:
    n = X_int8.shape[1]
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = X_int8[:, s:e].astype(np.float32).var(axis=0)
    return out


def _chi2_chunked(X_int8: np.ndarray, y: np.ndarray, chunk: int = 10_000) -> np.ndarray:
    n = X_int8.shape[1]
    out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        c, _ = chi2(X_int8[:, s:e].astype(np.float32), y)
        out[s:e] = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Base learner 1: XGB on gene-aggregated features
# ───────────────────────────────────────────────────────────────────────────────
def _xgb_gene_agg(
    X_gene_flat: np.ndarray,    # (n, 20000) float32 (already standardised? no — raw)
    y: np.ndarray, mask: np.ndarray,
    tr: np.ndarray, va: np.ndarray, te: np.ndarray,
    ab_idx: int, ab_name: str, params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (val_probs, test_probs) aligned to va / te indices."""
    m = mask[:, ab_idx]
    tr_use = tr[m[tr]]; va_use = va[m[va]]; te_use = te[m[te]]
    y_tr = y[tr_use, ab_idx]; y_va = y[va_use, ab_idx]; y_te = y[te_use, ab_idx]
    if len(np.unique(y_tr)) < 2:
        return np.array([]), np.array([])

    n_pos = int(y_tr.sum()); n_neg = int(len(y_tr) - n_pos)
    pw = n_neg / max(n_pos, 1)

    p = dict(params); p["scale_pos_weight"] = pw; p["objective"] = "binary:logistic"
    early = p.pop("early_stopping_rounds", 200)
    n_round = p.pop("n_estimators", 4000)
    p.pop("n_jobs", None)

    print(f"      [base1 xgb-gene-agg] {ab_name}  tr={len(tr_use)} va={len(va_use)} te={len(te_use)}")
    dtr = xgb.DMatrix(X_gene_flat[tr_use], label=y_tr)
    dva = xgb.DMatrix(X_gene_flat[va_use], label=y_va)
    dte = xgb.DMatrix(X_gene_flat[te_use], label=y_te)
    b = xgb.train(p, dtr, num_boost_round=n_round,
                  evals=[(dva, "val")], early_stopping_rounds=early, verbose_eval=False)
    val_probs = b.predict(dva)
    test_probs = b.predict(dte)
    del b, dtr, dva, dte; gc.collect()
    return val_probs.astype(np.float32), test_probs.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────────
# Base learner 2: GeneAttentionNet from checkpoint
# ───────────────────────────────────────────────────────────────────────────────
def _nn_predictions(
    X_t: torch.Tensor, y_arr: np.ndarray, mask_arr: np.ndarray,
    indices_per_split: dict, ab_indices: list[int], antibiotics: list[str],
    ckpt_path: Path, n_genes: int, n_stats: int, batch_size: int,
) -> dict:
    """Returns dict {ab: {'val_probs': np.array, 'test_probs': np.array}}."""
    print(f"      [base2 GeneAttentionNet] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})

    model = GeneAttentionNet(
        n_genes=n_genes, n_stats=n_stats, n_antibiotics=len(antibiotics),
        antibiotic_names=antibiotics,
        hidden_dim=int(hp.get("hidden_dim", 128)),
        n_heads=int(hp.get("n_heads", 4)),
        n_layers=int(hp.get("n_layers", 2)),
        ffn_mult=int(hp.get("ffn_mult", 2)),
        dropout=float(hp.get("dropout", 0.3)),
        pos_weights=[1.0] * len(antibiotics),
        learning_rate=1e-3, weight_decay=1e-4, scheduler_t_max=50,
    )
    model.load_state_dict(ckpt["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    Y_sub = y_arr[:, ab_indices].astype(np.float32)
    mask_sub = mask_arr[:, ab_indices]

    out = {ab: {} for ab in antibiotics}
    for split_name, idxs in indices_per_split.items():
        ds = GeneAttentionDataset(
            X_t, torch.from_numpy(Y_sub), torch.from_numpy(mask_sub), idxs
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=torch.cuda.is_available())
        all_probs = []
        with torch.no_grad():
            for xb, _, _ in loader:
                logits = model(xb.to(device, non_blocking=True))
                p = torch.sigmoid(logits.float()).detach().cpu().numpy()
                all_probs.append(p)
        probs_full = np.concatenate(all_probs, axis=0)        # (len(idxs), n_ab)
        # Index back into original sample order with mask per AB
        for k, ab in enumerate(antibiotics):
            m = mask_sub[idxs, k]
            probs_masked = probs_full[m, k].astype(np.float32)
            out[ab][f"{split_name}_probs"] = probs_masked

    del model; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# ───────────────────────────────────────────────────────────────────────────────
# Base learner 3: XGB on raw chi²-prefiltered SNPs
# ───────────────────────────────────────────────────────────────────────────────
def _xgb_raw_chi2(
    X_snp_filtered: np.ndarray,            # (n, n_snps_filtered) int8
    y: np.ndarray, mask: np.ndarray,
    tr: np.ndarray, va: np.ndarray, te: np.ndarray,
    ab_idx: int, ab_name: str, params: dict,
    prefilter_k: int = 50_000, min_var: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    m = mask[:, ab_idx]
    tr_use = tr[m[tr]]; va_use = va[m[va]]; te_use = te[m[te]]
    y_tr = y[tr_use, ab_idx]; y_va = y[va_use, ab_idx]; y_te = y[te_use, ab_idx]
    if len(np.unique(y_tr)) < 2:
        return np.array([]), np.array([])

    n_pos = int(y_tr.sum()); n_neg = int(len(y_tr) - n_pos)
    pw = n_neg / max(n_pos, 1)

    print(f"      [base3 xgb-raw-chi²] {ab_name}  tr={len(tr_use)} va={len(va_use)} te={len(te_use)}")

    X_tr_int = X_snp_filtered[tr_use]
    var = _variance_chunked(X_tr_int)
    keep_var = np.where(var > min_var)[0]
    X_tr_var = X_tr_int[:, keep_var]
    chi2_stats = _chi2_chunked(X_tr_var, y_tr)
    k = int(min(prefilter_k, len(keep_var)))
    top_local = np.argpartition(chi2_stats, -k)[-k:]
    top_idx = keep_var[top_local]
    del X_tr_int, X_tr_var, var, chi2_stats; gc.collect()

    X_tr = X_snp_filtered[tr_use][:, top_idx].astype(np.float32)
    X_va = X_snp_filtered[va_use][:, top_idx].astype(np.float32)
    X_te = X_snp_filtered[te_use][:, top_idx].astype(np.float32)

    p = dict(params); p["scale_pos_weight"] = pw; p["objective"] = "binary:logistic"
    p.setdefault("colsample_bytree", 1.0); p.setdefault("colsample_bynode", 0.5)
    early = p.pop("early_stopping_rounds", 200)
    n_round = p.pop("n_estimators", 4000)
    p.pop("n_jobs", None)

    dtr = xgb.DMatrix(X_tr, label=y_tr); dva = xgb.DMatrix(X_va, label=y_va)
    dte = xgb.DMatrix(X_te, label=y_te)
    b = xgb.train(p, dtr, num_boost_round=n_round,
                  evals=[(dva, "val")], early_stopping_rounds=early, verbose_eval=False)
    val_probs = b.predict(dva)
    test_probs = b.predict(dte)
    del b, dtr, dva, dte, X_tr, X_va, X_te; gc.collect()
    return val_probs.astype(np.float32), test_probs.astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────────
# Meta-learner: per-AB LogReg on val predictions
# ───────────────────────────────────────────────────────────────────────────────
def _fit_meta_learner(
    val_probs_stack: np.ndarray,    # (n_val, 3) — columns are 3 base models
    y_val: np.ndarray,
) -> LogisticRegression:
    """Mild L2 regularisation; 3-feature model can barely overfit."""
    clf = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    clf.fit(val_probs_stack, y_val)
    return clf


# ───────────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────────
def run(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    exp_cfg = cfg.experiment
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config_used.yaml")

    print("=" * 80)
    print("  STACKING ENSEMBLE — 3 base models + per-AB LogReg meta-learner")
    print("=" * 80)

    # ── Load Stage 1.5 filtered SNP matrix (for base learner 3) ──────────────
    print("\n[1/6] Loading Stage 1.5 filtered SNP matrix...")
    pkl_path = (str(cfg.paths.processed_dir).rstrip("/\\") +
                "/arcadia_v2_gene_filtered.pkl")
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    X_snp_filtered = d["X_snp_filtered"]
    print(f"  X_snp_filtered: {X_snp_filtered.shape}, dtype={X_snp_filtered.dtype}")

    # ── Build gene-aggregated features (for base 1 + base 2) ─────────────────
    print("\n[2/6] Building gene-aggregated features...")
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=pkl_path,
        stats=tuple(exp_cfg.gene_stats),
    )
    X_flat = bundle["X_gene_features"]
    n_genes = len(bundle["gene_ids"])
    n_stats = len(bundle["stats_used"])
    Y = bundle["Y"].astype(np.float32)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]
    splits = bundle["random_split"]
    tr = np.asarray(splits["train_idx"])
    va = np.asarray(splits["val_idx"])
    te = np.asarray(splits["test_idx"])

    # Tensor input for the NN (uses train-stat standardisation)
    X_3d = reshape_gene_features(X_flat, n_genes, n_stats)
    Xn_tr, Xn_va, Xn_te, _ = normalize_gene_features_per_stat(
        X_3d[tr], X_3d[va], X_3d[te]
    )
    X_norm = np.zeros_like(X_3d, dtype=np.float32)
    X_norm[tr] = Xn_tr; X_norm[va] = Xn_va; X_norm[te] = Xn_te
    X_t = torch.from_numpy(X_norm)

    antibiotics = list(exp_cfg.antibiotics)
    ab_indices = [label_names.index(a) for a in antibiotics]

    # ── Base 1: XGB on gene-aggregated features ──────────────────────────────
    print("\n[3/6] Training base 1: XGB on gene-aggregated features...")
    xgb_agg_params = dict(exp_cfg.xgb_agg_params)
    base1_preds: dict[str, dict] = {ab: {} for ab in antibiotics}
    t0 = time.time()
    for ab, ab_idx in zip(antibiotics, ab_indices):
        vp, tp = _xgb_gene_agg(
            X_flat, Y, label_mask, tr, va, te, ab_idx, ab, xgb_agg_params
        )
        base1_preds[ab]["val_probs"] = vp
        base1_preds[ab]["test_probs"] = tp
    print(f"  base 1 done in {(time.time() - t0)/60:.1f} min")

    # ── Base 2: GeneAttentionNet inference ───────────────────────────────────
    print("\n[4/6] Running base 2: GeneAttentionNet inference...")
    ckpt_path = Path(exp_cfg.gene_attention_ckpt)
    t0 = time.time()
    base2_preds = _nn_predictions(
        X_t, Y, label_mask,
        indices_per_split={"val": va, "test": te},
        ab_indices=ab_indices, antibiotics=antibiotics,
        ckpt_path=ckpt_path, n_genes=n_genes, n_stats=n_stats,
        batch_size=int(exp_cfg.nn_inference_batch_size),
    )
    print(f"  base 2 done in {(time.time() - t0)/60:.1f} min")

    # ── Base 3: XGB on raw chi²-prefiltered SNPs ─────────────────────────────
    print("\n[5/6] Training base 3: XGB on raw chi²-prefiltered SNPs...")
    xgb_raw_params = dict(exp_cfg.xgb_raw_params)
    base3_preds: dict[str, dict] = {ab: {} for ab in antibiotics}
    t0 = time.time()
    for ab, ab_idx in zip(antibiotics, ab_indices):
        vp, tp = _xgb_raw_chi2(
            X_snp_filtered, Y, label_mask, tr, va, te, ab_idx, ab,
            xgb_raw_params,
            prefilter_k=int(exp_cfg.prefilter_top_k),
            min_var=float(exp_cfg.prefilter_min_variance),
        )
        base3_preds[ab]["val_probs"] = vp
        base3_preds[ab]["test_probs"] = tp
    print(f"  base 3 done in {(time.time() - t0)/60:.1f} min")

    # Free SNP matrix after base 3
    del X_snp_filtered, d; gc.collect()

    # ── Meta-learner + evaluation ────────────────────────────────────────────
    print("\n[6/6] Stacking + evaluation...")
    boot_n = int(exp_cfg.evaluation.bootstrap_n)
    boot_seed = int(exp_cfg.evaluation.bootstrap_seed)
    rows = []
    test_probs_by_ab: dict[str, np.ndarray] = {}

    for ab, ab_idx in zip(antibiotics, ab_indices):
        m_va = label_mask[va, ab_idx]
        m_te = label_mask[te, ab_idx]
        y_va = Y[va[m_va], ab_idx].astype(np.int32)
        y_te = Y[te[m_te], ab_idx].astype(np.int32)

        v1 = base1_preds[ab]["val_probs"]
        v2 = base2_preds[ab]["val_probs"]
        v3 = base3_preds[ab]["val_probs"]
        t1 = base1_preds[ab]["test_probs"]
        t2 = base2_preds[ab]["test_probs"]
        t3 = base3_preds[ab]["test_probs"]
        # Sanity: lengths align with y_va/y_te
        if not (len(v1) == len(v2) == len(v3) == len(y_va)):
            print(f"    WARN {ab} val length mismatch: "
                  f"{len(v1)} {len(v2)} {len(v3)} vs y={len(y_va)}")
        if not (len(t1) == len(t2) == len(t3) == len(y_te)):
            print(f"    WARN {ab} test length mismatch: "
                  f"{len(t1)} {len(t2)} {len(t3)} vs y={len(y_te)}")

        # Solo metrics
        solo = {
            "xgb_gene_agg": float(average_precision_score(y_te, t1)),
            "gene_attn_net": float(average_precision_score(y_te, t2)),
            "xgb_raw_chi2": float(average_precision_score(y_te, t3)),
        }

        # Ensemble: meta-learner trained on val
        X_meta_val = np.stack([v1, v2, v3], axis=1)
        X_meta_test = np.stack([t1, t2, t3], axis=1)
        meta = _fit_meta_learner(X_meta_val, y_va)
        ens_probs = meta.predict_proba(X_meta_test)[:, 1].astype(np.float32)

        # Also compute simple mean ensemble
        mean_probs = X_meta_test.mean(axis=1).astype(np.float32)

        # Metrics for the meta ensemble
        metrics = compute_metrics(y_te, ens_probs, threshold=0.5)
        pr_lo, pr_hi = bootstrap_pr_auc(y_te, ens_probs, n_boot=boot_n, seed=boot_seed)
        mcc_lo, mcc_hi = bootstrap_mcc(y_te, ens_probs, n_boot=boot_n, seed=boot_seed)

        meta_coef = meta.coef_.ravel().tolist()
        rows.append({
            "antibiotic": ab,
            "solo_xgb_gene_agg": solo["xgb_gene_agg"],
            "solo_gene_attn_net": solo["gene_attn_net"],
            "solo_xgb_raw_chi2": solo["xgb_raw_chi2"],
            "ensemble_mean": float(average_precision_score(y_te, mean_probs)),
            "ensemble_meta_pr_auc": metrics["pr_auc"],
            "ensemble_meta_pr_auc_lo": pr_lo,
            "ensemble_meta_pr_auc_hi": pr_hi,
            "ensemble_meta_mcc": metrics["mcc"],
            "ensemble_meta_mcc_lo": mcc_lo,
            "ensemble_meta_mcc_hi": mcc_hi,
            "ensemble_meta_f2": metrics["f2"],
            "ensemble_meta_roc_auc": metrics["roc_auc"],
            "ensemble_meta_sens_at_90_spec": metrics["sens_at_90_spec"],
            "n_test_valid": int(m_te.sum()),
            "n_pos_test": int(y_te.sum()),
            "meta_w_xgb_gene_agg": meta_coef[0],
            "meta_w_gene_attn_net": meta_coef[1],
            "meta_w_xgb_raw_chi2": meta_coef[2],
            "meta_intercept": float(meta.intercept_[0]),
        })
        test_probs_by_ab[ab] = ens_probs
        print(f"    {ab:<5} "
              f"solo xgb-gene={solo['xgb_gene_agg']:.4f}  "
              f"nn={solo['gene_attn_net']:.4f}  "
              f"xgb-raw={solo['xgb_raw_chi2']:.4f}  "
              f"→ MEAN={float(average_precision_score(y_te, mean_probs)):.4f}  "
              f"META={metrics['pr_auc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "ensemble_results.csv", index=False)

    # ── Summary ──────────────────────────────────────────────────────────────
    avg_xgb_agg = float(df["solo_xgb_gene_agg"].mean())
    avg_nn      = float(df["solo_gene_attn_net"].mean())
    avg_xgb_raw = float(df["solo_xgb_raw_chi2"].mean())
    avg_mean    = float(df["ensemble_mean"].mean())
    avg_meta    = float(df["ensemble_meta_pr_auc"].mean())
    delta_vs_best_solo = avg_meta - max(avg_xgb_agg, avg_xgb_raw)

    np.savez_compressed(
        out_dir / "ensemble_predictions.npz",
        **{f"probs_{ab}": p for ab, p in test_probs_by_ab.items()},
        antibiotics=np.array(antibiotics),
    )

    lines = [
        "# Stacking Ensemble — 3 base models + LogReg meta\n",
        "## Average test PR-AUC across antibiotics\n",
        f"- solo XGB gene-agg:    **{avg_xgb_agg:.4f}**",
        f"- solo GeneAttentionNet: **{avg_nn:.4f}**",
        f"- solo XGB raw+chi²:    **{avg_xgb_raw:.4f}**",
        f"- ensemble (simple mean): **{avg_mean:.4f}**",
        f"- ensemble (LogReg meta): **{avg_meta:.4f}**",
        f"- Δ meta vs best solo:   **{delta_vs_best_solo:+.4f}**\n",
        "## Per-antibiotic test PR-AUC\n",
        "| AB | XGB-gene | NN | XGB-raw | Mean | **Meta** | [lo, hi] |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['solo_xgb_gene_agg']:.4f} | "
            f"{r['solo_gene_attn_net']:.4f} | {r['solo_xgb_raw_chi2']:.4f} | "
            f"{r['ensemble_mean']:.4f} | **{r['ensemble_meta_pr_auc']:.4f}** | "
            f"[{r['ensemble_meta_pr_auc_lo']:.4f}, {r['ensemble_meta_pr_auc_hi']:.4f}] |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"  Avg solo XGB-gene-agg : {avg_xgb_agg:.4f}")
    print(f"  Avg solo NN           : {avg_nn:.4f}")
    print(f"  Avg solo XGB-raw-chi² : {avg_xgb_raw:.4f}")
    print(f"  Avg ensemble (mean)   : {avg_mean:.4f}")
    print(f"  Avg ensemble (meta)   : {avg_meta:.4f}")
    print(f"  Δ meta vs best solo   : {delta_vs_best_solo:+.4f}")
    print(f"  Outputs → {out_dir}")
    print("=" * 80)
