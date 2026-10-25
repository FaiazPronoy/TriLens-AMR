# src/experiments/run_cascadestack_v9.py
"""
================================================================================
CascadeStack v9 — v8 + GeneAttentionNet (neural) as 6th base learner
================================================================================

v9 adds the neural component the CSE-thesis defence needs, without bringing
back v7's overfitting:

  - 5 classical base learners (identical to v8): XGB / LGBM / RF / LogReg on
    gene-aggregated features + XGB on raw chi²-prefiltered SNPs.
  - 6th base: a multi-task GeneAttentionNet, trained leak-free via 5-fold OOF.
    The multi-task design means ONE GAN training per fold covers all 7
    antibiotics, so K-fold cost is only 5 retrainings, not 5 × 7 = 35.
  - LogReg meta-learner with mild L2 — same as v8, robust to correlated bases.

Pipeline order:
    1.  Build gene-aggregated tabular features (5000 genes × 4 stats).
    2.  Standardise 3-D tensor per-stat (using train statistics only).
    3.  K-fold multi-task GeneAttentionNet → OOF / val / test prediction
        matrices of shape (n, n_antibiotics).
    4.  Per-antibiotic K-fold stacking with 6 bases.
    5.  Per-AB LogReg meta-learner; threshold tuned on val.

Targets:
    - Defence-grade: >= 0.85 avg test PR-AUC across 7 antibiotics
    - Beat best XGB solo by >= +1%
    - Total runtime ~ 55-70 min on RTX 4070 Ti
================================================================================
"""
from __future__ import annotations

import gc
import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.gene_aggregated_features import build_gene_feature_matrix_from_pkl
from src.experiments._shared import (
    bootstrap_mcc, bootstrap_pr_auc, compute_metrics,
)
from src.models.gene_attention_net import GeneAttentionNet
from src.utils.seeds import set_global_seed

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

warnings.filterwarnings("ignore")


# ───────────────────────────────────────────────────────────────────────────────
# Small helpers
# ───────────────────────────────────────────────────────────────────────────────
def _spw(y: np.ndarray) -> float:
    pos = float((y == 1).sum()); neg = float((y == 0).sum())
    return max(1.0, neg / max(pos, 1.0))


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


def _best_threshold_mcc(y, p):
    thresholds = np.linspace(0.05, 0.95, 91)
    best_t, best_m = 0.5, -1.0
    for t in thresholds:
        pred = (p >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        m = matthews_corrcoef(y, pred)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)


# ───────────────────────────────────────────────────────────────────────────────
# Classical base learners (same as v8)
# ───────────────────────────────────────────────────────────────────────────────
def _train_xgb_gene(X_fit, y_fit, X_es, y_es, params):
    p = dict(params); p["scale_pos_weight"] = _spw(y_fit)
    p["objective"] = "binary:logistic"; p["eval_metric"] = "aucpr"
    early = p.pop("early_stopping_rounds", 150)
    n_round = p.pop("n_estimators", 1500)
    p.pop("n_jobs", None)
    dtr = xgb.DMatrix(X_fit, label=y_fit); des = xgb.DMatrix(X_es, label=y_es)
    return xgb.train(p, dtr, num_boost_round=n_round,
                     evals=[(des, "es")], early_stopping_rounds=early,
                     verbose_eval=False)


def _train_lgbm_gene(X_fit, y_fit, X_es, y_es, params):
    p = dict(params); p["scale_pos_weight"] = _spw(y_fit)
    p["random_state"] = 42; p["verbose"] = -1; p["n_jobs"] = -1
    early = p.pop("early_stopping_rounds", 150)
    n_round = p.pop("n_estimators", 1500)
    clf = lgb.LGBMClassifier(n_estimators=n_round, **p)
    clf.fit(X_fit, y_fit, eval_set=[(X_es, y_es)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(early, verbose=False),
                       lgb.log_evaluation(-1)])
    return clf


def _train_rf_gene(X_fit, y_fit, params):
    return RandomForestClassifier(
        n_estimators=int(params.get("n_estimators", 200)),
        max_depth=int(params.get("max_depth", 12)),
        min_samples_leaf=int(params.get("min_samples_leaf", 10)),
        max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=42,
    ).fit(X_fit, y_fit)


def _train_logreg_gene(X_fit, y_fit):
    sc = StandardScaler().fit(X_fit)
    clf = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs",
        max_iter=3000, class_weight="balanced", random_state=42,
    ).fit(sc.transform(X_fit), y_fit)
    return clf, sc


def _train_xgb_raw_chi2(X_fit_snp, y_fit, X_es_snp, y_es,
                         prefilter_k, min_var, params):
    var = _variance_chunked(X_fit_snp)
    keep_var = np.where(var > min_var)[0]
    X_var = X_fit_snp[:, keep_var]
    chi2_stats = _chi2_chunked(X_var, y_fit)
    k = int(min(prefilter_k, len(keep_var)))
    top_local = np.argpartition(chi2_stats, -k)[-k:]
    top_idx = keep_var[top_local]
    del X_var, chi2_stats, var; gc.collect()

    X_fit_top = X_fit_snp[:, top_idx].astype(np.float32)
    X_es_top = X_es_snp[:, top_idx].astype(np.float32)

    p = dict(params); p["scale_pos_weight"] = _spw(y_fit)
    p["objective"] = "binary:logistic"; p["eval_metric"] = "aucpr"
    p.setdefault("colsample_bytree", 1.0); p.setdefault("colsample_bynode", 0.5)
    early = p.pop("early_stopping_rounds", 150)
    n_round = p.pop("n_estimators", 1500)
    p.pop("n_jobs", None)
    dtr = xgb.DMatrix(X_fit_top, label=y_fit)
    des = xgb.DMatrix(X_es_top, label=y_es)
    booster = xgb.train(p, dtr, num_boost_round=n_round,
                        evals=[(des, "es")], early_stopping_rounds=early,
                        verbose_eval=False)
    del dtr, des; gc.collect()
    return booster, top_idx


# ───────────────────────────────────────────────────────────────────────────────
# Multi-task GeneAttentionNet — K-fold OOF helper
# ───────────────────────────────────────────────────────────────────────────────
def _train_gan_one_fold(
    X_3d_fit: np.ndarray, Y_fit: np.ndarray, mask_fit: np.ndarray,
    X_3d_es: np.ndarray, Y_es: np.ndarray, mask_es: np.ndarray,
    n_genes: int, n_stats: int, antibiotic_names: list[str],
    max_epochs: int, batch_size: int, learning_rate: float,
    hidden_dim: int, n_heads: int, n_layers: int, ffn_mult: int, dropout: float,
    device: str = "cuda",
) -> GeneAttentionNet:
    """Train one multi-task GeneAttentionNet on fit partition with ES early stop."""
    # Per-AB pos_weights from FIT only (leak-free)
    pos_weights = []
    for k in range(Y_fit.shape[1]):
        if mask_fit[:, k].sum() < 5:
            pos_weights.append(1.0); continue
        y_k = Y_fit[mask_fit[:, k], k]
        if len(np.unique(y_k)) < 2:
            pos_weights.append(1.0); continue
        n_pos = float((y_k == 1).sum()); n_neg = float((y_k == 0).sum())
        pos_weights.append(max(1.0, n_neg / max(n_pos, 1.0)))

    model = GeneAttentionNet(
        n_genes=n_genes, n_stats=n_stats, n_antibiotics=Y_fit.shape[1],
        antibiotic_names=antibiotic_names,
        hidden_dim=hidden_dim, n_heads=n_heads, n_layers=n_layers,
        ffn_mult=ffn_mult, dropout=dropout,
        pos_weights=pos_weights,
        learning_rate=learning_rate, weight_decay=1e-4,
        scheduler_t_max=max_epochs,
    ).to(device)

    fit_ds = TensorDataset(
        torch.from_numpy(X_3d_fit).float(),
        torch.from_numpy(Y_fit).float(),
        torch.from_numpy(mask_fit).bool(),
    )
    fit_dl = DataLoader(fit_ds, batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=False)

    X_es_t = torch.from_numpy(X_3d_es).float().to(device)
    Y_es_np = Y_es.astype(np.int32)
    mask_es_np = mask_es.astype(bool)

    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)

    best_es_pr = -1.0
    best_state = None
    patience_left = 4
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    for ep in range(max_epochs):
        t_ep = time.time()
        model.train()
        for xb, yb, mb in fit_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(xb)
                pw = model.pos_weights.view(1, -1).expand_as(logits)
                bce = F.binary_cross_entropy_with_logits(
                    logits, yb, pos_weight=pw, reduction="none"
                )
                denom = mb.float().sum().clamp(min=1.0)
                loss = (bce * mb.float()).sum() / denom
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()
        print(f"        epoch {ep+1:2d}/{max_epochs}  ({time.time() - t_ep:.0f}s)",
              flush=True)
        # Multi-task val PR-AUC (mean across antibiotics with valid labels)
        model.eval()
        with torch.no_grad():
            logits_es = model(X_es_t).cpu().numpy()
        probs_es = 1.0 / (1.0 + np.exp(-logits_es.astype(np.float64)))
        prs = []
        for k in range(Y_es_np.shape[1]):
            mk = mask_es_np[:, k]
            if mk.sum() < 5: continue
            yk = Y_es_np[mk, k]
            if len(np.unique(yk)) < 2: continue
            prs.append(float(average_precision_score(yk, probs_es[mk, k])))
        es_pr = float(np.mean(prs)) if prs else 0.0

        if es_pr > best_es_pr + 1e-4:
            best_es_pr = es_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 4
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _predict_gan_batched(model: GeneAttentionNet, X_3d: np.ndarray,
                          device: str = "cuda", batch_size: int = 32) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X_3d), batch_size):
        xb = torch.from_numpy(X_3d[i:i + batch_size]).float().to(device)
        logits = model(xb).cpu().numpy()
        out.append(1.0 / (1.0 + np.exp(-logits.astype(np.float64))))
    return np.concatenate(out, axis=0).astype(np.float32)


def _gan_kfold_oof_multitask(
    X_3d_norm: np.ndarray, Y_all: np.ndarray, mask_all: np.ndarray,
    train_idx: np.ndarray, val_idx: np.ndarray, test_idx: np.ndarray,
    n_genes: int, n_stats: int, label_names: list[str],
    n_folds: int, gan_cfg: dict, device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run K-fold multi-task GeneAttentionNet → per-AB OOF / val / test probs.

    Returns:
        oof_train: (len(train_idx), n_antibiotics)  — leak-free OOF on train
        val_avg  : (len(val_idx),   n_antibiotics)  — averaged across folds
        test_avg : (len(test_idx),  n_antibiotics)  — averaged across folds
    """
    print(f"  [GAN K-fold] {n_folds} folds, max_epochs={gan_cfg['max_epochs']}, "
          f"batch_size={gan_cfg['batch_size']}, hidden_dim={gan_cfg['hidden_dim']}")

    n_train = len(train_idx); n_ab = Y_all.shape[1]
    oof_train = np.zeros((n_train, n_ab), dtype=np.float32)
    val_avg = np.zeros((len(val_idx), n_ab), dtype=np.float32)
    test_avg = np.zeros((len(test_idx), n_ab), dtype=np.float32)

    X_val = X_3d_norm[val_idx]
    X_test = X_3d_norm[test_idx]

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (fit_rel, oof_rel) in enumerate(kf.split(np.arange(n_train))):
        t_fold = time.time()
        # 90/10 split within fit for early stopping
        rng = np.random.RandomState(42 + fold)
        perm = rng.permutation(len(fit_rel))
        es_size = max(50, int(0.1 * len(fit_rel)))
        es_rel = fit_rel[perm[:es_size]]
        train_only_rel = fit_rel[perm[es_size:]]

        fit_abs = train_idx[train_only_rel]
        es_abs = train_idx[es_rel]
        oof_abs = train_idx[oof_rel]

        gan = _train_gan_one_fold(
            X_3d_fit=X_3d_norm[fit_abs],
            Y_fit=Y_all[fit_abs].astype(np.float32),
            mask_fit=mask_all[fit_abs].astype(bool),
            X_3d_es=X_3d_norm[es_abs],
            Y_es=Y_all[es_abs].astype(np.float32),
            mask_es=mask_all[es_abs].astype(bool),
            n_genes=n_genes, n_stats=n_stats,
            antibiotic_names=label_names,
            max_epochs=int(gan_cfg["max_epochs"]),
            batch_size=int(gan_cfg["batch_size"]),
            learning_rate=float(gan_cfg["learning_rate"]),
            hidden_dim=int(gan_cfg["hidden_dim"]),
            n_heads=int(gan_cfg["n_heads"]),
            n_layers=int(gan_cfg["n_layers"]),
            ffn_mult=int(gan_cfg["ffn_mult"]),
            dropout=float(gan_cfg["dropout"]),
            device=device,
        )

        oof_train[oof_rel] = _predict_gan_batched(gan, X_3d_norm[oof_abs], device=device)
        val_avg += _predict_gan_batched(gan, X_val, device=device) / n_folds
        test_avg += _predict_gan_batched(gan, X_test, device=device) / n_folds

        print(f"    Fold {fold + 1}/{n_folds}  fit={len(fit_abs)} es={len(es_abs)} "
              f"oof={len(oof_abs)}  ({time.time() - t_fold:.0f}s)")

        del gan; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return oof_train, val_avg, test_avg


# ───────────────────────────────────────────────────────────────────────────────
# Per-antibiotic K-fold stacking (6 base learners)
# ───────────────────────────────────────────────────────────────────────────────
def _run_one_antibiotic(
    X_flat, X_snp_int8, Y, label_mask,
    gan_oof_train_all, gan_val_all, gan_test_all,
    train_idx, val_idx, test_idx,
    ab_idx, ab_name, exp_cfg,
):
    """K-fold stacking for one antibiotic with 6 base learners.

    gan_oof_train_all : (len(train_idx), n_antibiotics) — precomputed K-fold OOF
    gan_val_all       : (len(val_idx),   n_antibiotics) — fold-averaged
    gan_test_all      : (len(test_idx),  n_antibiotics) — fold-averaged
    """
    n_folds = int(exp_cfg.n_folds)
    use_lgbm = bool(exp_cfg.use_lgbm) and HAS_LGB

    m = label_mask[:, ab_idx]
    tr_pos = np.where(m[train_idx])[0]      # positions within train_idx
    va_pos = np.where(m[val_idx])[0]
    te_pos = np.where(m[test_idx])[0]
    tr = train_idx[tr_pos]
    va = val_idx[va_pos]
    te = test_idx[te_pos]

    y_tr = Y[tr, ab_idx].astype(np.int32)
    y_va = Y[va, ab_idx].astype(np.int32)
    y_te = Y[te, ab_idx].astype(np.int32)

    X_tr_flat = X_flat[tr]; X_va_flat = X_flat[va]; X_te_flat = X_flat[te]
    X_tr_snp = X_snp_int8[tr]; X_va_snp = X_snp_int8[va]; X_te_snp = X_snp_int8[te]

    # GAN OOF / val / test predictions for this antibiotic (slice ab_idx)
    gan_oof_ab = gan_oof_train_all[tr_pos, ab_idx]
    gan_val_ab = gan_val_all[va_pos, ab_idx]
    gan_test_ab = gan_test_all[te_pos, ab_idx]

    print(f"\n  -- {ab_name} --")
    print(f"      Train={len(y_tr)} (pos={int(y_tr.sum())}, "
          f"{100*y_tr.mean():.1f}%)  Val={len(y_va)}  Test={len(y_te)}")

    base_names = ["XGB-gene"]
    if use_lgbm:
        base_names.append("LGBM-gene")
    base_names += ["RF-gene", "LogReg-gene", "XGB-raw-chi2", "GeneAttentionNet"]
    n_base = len(base_names)

    oof_preds = np.zeros((len(y_tr), n_base), dtype=np.float32)
    val_acc = np.zeros((len(y_va), n_base), dtype=np.float32)
    test_acc = np.zeros((len(y_te), n_base), dtype=np.float32)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (fit_rel, oof_rel) in enumerate(skf.split(X_tr_flat, y_tr)):
        t_fold = time.time()
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42 + fold)
        rel_fit, rel_es = next(sss.split(X_tr_flat[fit_rel], y_tr[fit_rel]))
        fit_idx_local = fit_rel[rel_fit]; es_idx_local = fit_rel[rel_es]
        X_fit, y_fit = X_tr_flat[fit_idx_local], y_tr[fit_idx_local]
        X_es, y_es = X_tr_flat[es_idx_local], y_tr[es_idx_local]
        X_fit_snp = X_tr_snp[fit_idx_local]; X_es_snp = X_tr_snp[es_idx_local]

        col = 0

        # 1. XGB-gene
        b = _train_xgb_gene(X_fit, y_fit, X_es, y_es, dict(exp_cfg.xgb_params))
        oof_preds[oof_rel, col] = b.predict(xgb.DMatrix(X_tr_flat[oof_rel]))
        val_acc[:, col] += b.predict(xgb.DMatrix(X_va_flat)) / n_folds
        test_acc[:, col] += b.predict(xgb.DMatrix(X_te_flat)) / n_folds
        del b; gc.collect()
        col += 1

        # 2. LGBM-gene
        if use_lgbm:
            clf = _train_lgbm_gene(X_fit, y_fit, X_es, y_es, dict(exp_cfg.lgbm_params))
            oof_preds[oof_rel, col] = clf.predict_proba(X_tr_flat[oof_rel])[:, 1]
            val_acc[:, col] += clf.predict_proba(X_va_flat)[:, 1] / n_folds
            test_acc[:, col] += clf.predict_proba(X_te_flat)[:, 1] / n_folds
            del clf; gc.collect()
            col += 1

        # 3. RF-gene
        clf = _train_rf_gene(X_fit, y_fit, dict(exp_cfg.rf_params))
        oof_preds[oof_rel, col] = clf.predict_proba(X_tr_flat[oof_rel])[:, 1]
        val_acc[:, col] += clf.predict_proba(X_va_flat)[:, 1] / n_folds
        test_acc[:, col] += clf.predict_proba(X_te_flat)[:, 1] / n_folds
        del clf; gc.collect()
        col += 1

        # 4. LogReg-gene
        clf, sc = _train_logreg_gene(X_fit, y_fit)
        oof_preds[oof_rel, col] = clf.predict_proba(sc.transform(X_tr_flat[oof_rel]))[:, 1]
        val_acc[:, col] += clf.predict_proba(sc.transform(X_va_flat))[:, 1] / n_folds
        test_acc[:, col] += clf.predict_proba(sc.transform(X_te_flat))[:, 1] / n_folds
        del clf, sc; gc.collect()
        col += 1

        # 5. XGB-raw-chi²
        b, top_idx = _train_xgb_raw_chi2(
            X_fit_snp, y_fit, X_es_snp, y_es,
            prefilter_k=int(exp_cfg.prefilter_top_k),
            min_var=float(exp_cfg.prefilter_min_variance),
            params=dict(exp_cfg.xgb_raw_params),
        )
        X_oof_top = X_tr_snp[oof_rel][:, top_idx].astype(np.float32)
        X_va_top = X_va_snp[:, top_idx].astype(np.float32)
        X_te_top = X_te_snp[:, top_idx].astype(np.float32)
        oof_preds[oof_rel, col] = b.predict(xgb.DMatrix(X_oof_top))
        val_acc[:, col] += b.predict(xgb.DMatrix(X_va_top)) / n_folds
        test_acc[:, col] += b.predict(xgb.DMatrix(X_te_top)) / n_folds
        del b, X_oof_top, X_va_top, X_te_top; gc.collect()
        col += 1

        print(f"      Fold {fold + 1}/{n_folds}  fit={len(fit_idx_local)} "
              f"es={len(es_idx_local)} oof={len(oof_rel)}  "
              f"({time.time() - t_fold:.0f}s)")

    # 6. GeneAttentionNet — precomputed K-fold OOF (does not vary by inner fold)
    oof_preds[:, -1] = gan_oof_ab
    val_acc[:, -1] = gan_val_ab
    test_acc[:, -1] = gan_test_ab

    # Diagnostics
    print(f"\n      Base diagnostics  (OOF / Val / Test):")
    for i, bn in enumerate(base_names):
        oof_pr = float(average_precision_score(y_tr, oof_preds[:, i]))
        val_pr = float(average_precision_score(y_va, val_acc[:, i]))
        te_pr = float(average_precision_score(y_te, test_acc[:, i]))
        print(f"        {bn:<18}  OOF={oof_pr:.4f}  Val={val_pr:.4f}  Test={te_pr:.4f}")

    # Meta-learner
    meta_C = float(exp_cfg.meta_C)
    print(f"\n      Meta-learner: LogReg C={meta_C}")
    meta = LogisticRegression(
        C=meta_C, penalty="l2", solver="lbfgs",
        max_iter=2000, class_weight="balanced", random_state=42,
    ).fit(oof_preds, y_tr)
    meta_coefs = meta.coef_.ravel().tolist()
    print(f"        weights: " +
          ", ".join(f"{bn}={w:+.3f}" for bn, w in zip(base_names, meta_coefs)))

    val_meta = meta.predict_proba(val_acc)[:, 1]
    test_meta = meta.predict_proba(test_acc)[:, 1]

    thr, _ = _best_threshold_mcc(y_va, val_meta)
    metrics = compute_metrics(y_te, test_meta, threshold=thr)
    boot_n = int(exp_cfg.evaluation.bootstrap_n)
    boot_seed = int(exp_cfg.evaluation.bootstrap_seed)
    pr_lo, pr_hi = bootstrap_pr_auc(y_te, test_meta, n_boot=boot_n, seed=boot_seed)
    mcc_lo, mcc_hi = bootstrap_mcc(y_te, test_meta, n_boot=boot_n, seed=boot_seed)

    solo_test_prs = [float(average_precision_score(y_te, test_acc[:, i]))
                     for i in range(n_base)]
    best_solo_te = max(solo_test_prs)
    delta = metrics["pr_auc"] - best_solo_te
    print(f"\n      -- TEST RESULT --")
    print(f"      Ensemble PR-AUC: {metrics['pr_auc']:.4f} "
          f"[{pr_lo:.4f}, {pr_hi:.4f}]")
    print(f"      Best solo      : {best_solo_te:.4f}  "
          f"(delta_vs_best_solo: {delta:+.4f})")
    print(f"      MCC: {metrics['mcc']:.4f}  F2: {metrics['f2']:.4f}  "
          f"Sens@90: {metrics['sens_at_90_spec']:.4f}")

    return {
        "antibiotic": ab_name,
        "ensemble_pr_auc": metrics["pr_auc"],
        "ensemble_pr_auc_lo": pr_lo, "ensemble_pr_auc_hi": pr_hi,
        "ensemble_mcc": metrics["mcc"],
        "ensemble_mcc_lo": mcc_lo, "ensemble_mcc_hi": mcc_hi,
        "ensemble_f2": metrics["f2"],
        "ensemble_roc_auc": metrics["roc_auc"],
        "ensemble_sens_at_90_spec": metrics["sens_at_90_spec"],
        "best_solo_pr_auc": best_solo_te,
        "delta_vs_best_solo": delta,
        "threshold": thr,
        "n_test": len(y_te),
        **{f"solo_{bn}_test_pr_auc": v for bn, v in zip(base_names, solo_test_prs)},
        **{f"meta_w_{bn}": w for bn, w in zip(base_names, meta_coefs)},
    }, test_meta, base_names, oof_preds, val_acc, test_acc, tr_pos, va_pos, te_pos

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
    print("  CascadeStack v9 — 5 classical bases + GeneAttentionNet (6th)")
    print("=" * 80)
    print(f"  LightGBM: {'YES' if HAS_LGB else 'NO'}")
    print(f"  GPU     : {'YES' if torch.cuda.is_available() else 'NO'}")
    print(f"  Output  : {out_dir}")

    print("\n[1/4] Loading data...")
    pkl_path = (str(cfg.paths.processed_dir).rstrip("/\\") +
                "/arcadia_v2_gene_filtered.pkl")
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=pkl_path,
        stats=tuple(exp_cfg.gene_stats),
    )
    X_flat = bundle["X_gene_features"].astype(np.float32)
    n_genes = len(bundle["gene_ids"]); n_stats = len(bundle["stats_used"])
    Y = bundle["Y"].astype(np.int32)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]
    _split_key = str(exp_cfg.get("split_key", "random_split"))
    splits = bundle.get(_split_key) or bundle["random_split"]
    tr_idx = np.asarray(splits["train_idx"])
    va_idx = np.asarray(splits["val_idx"])
    te_idx = np.asarray(splits["test_idx"])
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    X_snp_int8 = d["X_snp_filtered"]
    print(f"  X_flat: {X_flat.shape}, X_snp_int8: {X_snp_int8.shape}")

    # ── 3-D tensor for GAN (per-stat standardised using train stats only)
    print("\n[2/4] Standardising 3-D tensor for GeneAttentionNet input...")
    X_3d = X_flat.reshape(X_flat.shape[0], n_genes, n_stats)
    train_3d = X_3d[tr_idx]
    means = train_3d.reshape(-1, n_stats).mean(axis=0)
    stds = train_3d.reshape(-1, n_stats).std(axis=0) + 1e-7
    X_3d_norm = ((X_3d - means[None, None, :]) / stds[None, None, :]).astype(np.float32)
    print(f"  per-stat means: {[round(float(v), 3) for v in means]}")
    print(f"  per-stat stds : {[round(float(v), 3) for v in stds]}")

    # ── K-fold multi-task GeneAttentionNet OOF predictions
    print("\n[3/4] Computing K-fold multi-task GeneAttentionNet OOF...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t_gan = time.time()
    gan_oof_train_all, gan_val_all, gan_test_all = _gan_kfold_oof_multitask(
        X_3d_norm=X_3d_norm,
        Y_all=Y, mask_all=label_mask,
        train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx,
        n_genes=n_genes, n_stats=n_stats, label_names=label_names,
        n_folds=int(exp_cfg.n_folds),
        gan_cfg=dict(exp_cfg.gan),
        device=device,
    )
    print(f"  GAN K-fold total: {(time.time() - t_gan) / 60:.1f} min")

    # ── Per-AB stacking
    print("\n[4/4] Per-antibiotic K-fold stacking with 6 bases...")
    all_rows = []
    all_test_preds: dict[str, np.ndarray] = {}
    base_names_final: list[str] = []
    _oof_list, _val_list, _test_list = {}, {}, {}
    _tr_pos, _va_pos, _te_pos = {}, {}, {}
    for ab in exp_cfg.antibiotics:
        if ab not in label_names:
            print(f"  SKIP {ab}")
            continue
        ab_idx = label_names.index(ab)
        row, test_meta, base_names_final, oof_p, val_p, test_p, tr_pos, va_pos, te_pos = _run_one_antibiotic(
            X_flat, X_snp_int8, Y, label_mask,
            gan_oof_train_all, gan_val_all, gan_test_all,
            tr_idx, va_idx, te_idx, ab_idx, ab, exp_cfg,
        )
        all_rows.append(row)
        all_test_preds[ab] = test_meta
        _oof_list[ab] = oof_p       # (n_tr_valid, n_bases)
        _val_list[ab] = val_p       # (n_va_valid, n_bases)
        _test_list[ab] = test_p     # (n_te_valid, n_bases)
        _tr_pos[ab]  = tr_pos
        _va_pos[ab]  = va_pos
        _te_pos[ab]  = te_pos

    # ── Save predictions_saved.npz for posthoc analysis ──────────────────────
    # test_acc / val_acc / oof_preds are per-AB local arrays; we need global
    # stacked versions. Reconstruct from all_test_preds + gan arrays.
    # Shape convention: (n_samples, n_bases, n_abs) — bases axis=1, ABs axis=2
    ab_list = list(all_test_preds.keys())
    n_abs = len(ab_list)
    # ── Save predictions_saved.npz — CORRECT FORMAT for TriLens v10 ──────────
    if base_names_final:
        ab_list = list(all_test_preds.keys())
        n_abs   = len(ab_list)
        n_base  = len(base_names_final)

        # Build global (n_samples, n_bases, n_abs) arrays (zeros for masked entries)
        oof_base  = np.zeros((len(tr_idx), n_base, n_abs), dtype=np.float32)
        val_base  = np.zeros((len(va_idx), n_base, n_abs), dtype=np.float32)
        test_base = np.zeros((len(te_idx), n_base, n_abs), dtype=np.float32)

        for i, ab in enumerate(ab_list):
            if ab not in _oof_list: continue
            oof_base[_tr_pos[ab],  :, i] = _oof_list[ab]
            val_base[_va_pos[ab],  :, i] = _val_list[ab]
            test_base[_te_pos[ab], :, i] = _test_list[ab]

        _test_ens = np.zeros((len(te_idx), n_abs), dtype=np.float32)
        for _i, _ab in enumerate(ab_list):
            if _ab in _te_pos:
                _test_ens[_te_pos[_ab], _i] = all_test_preds[_ab]
        np.savez_compressed(
            str(out_dir / "predictions_saved.npz"),
            oof_base_probs   = oof_base,        # (4872, n_bases, 7)  ← TriLens uses this
            test_base_probs  = test_base,       # (1045, n_bases, 7)  ← TriLens uses this
            val_base_probs   = val_base,        # (1045, n_bases, 7)
            test_ens_probs   = _test_ens,       # (1045, 7) ensemble
            gan_test_probs   = gan_test_all.astype(np.float32),
            gan_val_probs    = gan_val_all.astype(np.float32),
            gan_oof_probs    = gan_oof_train_all.astype(np.float32),
            test_indices     = te_idx.astype(np.int64),
            val_indices      = va_idx.astype(np.int64),
            train_indices    = tr_idx.astype(np.int64),
            base_names       = np.array(base_names_final),
            antibiotic_names = np.array(ab_list),
        )
        print(f"[posthoc] Saved predictions_saved.npz")
        print(f"  oof_base_probs : {oof_base.shape}")
        print(f"  test_base_probs: {test_base.shape}")
    # ─────────────────────────────────────────────────────────────────────────

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / "v9_results.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        out_dir / "v9_test_predictions.npz",
        **{f"probs_{ab}": p for ab, p in all_test_preds.items()},
        antibiotics=np.array(list(all_test_preds.keys())),
    )
    np.savez_compressed(
        out_dir / "v9_gan_oof_predictions.npz",
        gan_oof_train=gan_oof_train_all,
        gan_val=gan_val_all,
        gan_test=gan_test_all,
        label_names=np.array(label_names),
    )
    
    avg_ens = float(df["ensemble_pr_auc"].mean()) if len(df) else 0.0
    avg_solo = float(df["best_solo_pr_auc"].mean()) if len(df) else 0.0
    avg_delta = float(df["delta_vs_best_solo"].mean()) if len(df) else 0.0
    avg_mcc = float(df["ensemble_mcc"].mean()) if len(df) else 0.0

    lines = [
        "# CascadeStack v9 - 6 base learners (5 classical + GeneAttentionNet)\n",
        "## Average across antibiotics\n",
        f"- Ensemble PR-AUC : **{avg_ens:.4f}**",
        f"- Best solo PR-AUC: {avg_solo:.4f}",
        f"- Delta (ens-solo): **{avg_delta:+.4f}**",
        f"- Avg MCC         : {avg_mcc:.4f}\n",
        "## Per-antibiotic\n",
        "| AB | Ensemble PR-AUC | 95% CI | Best Solo | Delta | MCC | F2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['ensemble_pr_auc']:.4f} "
            f"| [{r['ensemble_pr_auc_lo']:.4f}, {r['ensemble_pr_auc_hi']:.4f}] "
            f"| {r['best_solo_pr_auc']:.4f} "
            f"| {r['delta_vs_best_solo']:+.4f} "
            f"| {r['ensemble_mcc']:.4f} | {r['ensemble_f2']:.4f} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"  Avg ensemble PR-AUC : {avg_ens:.4f}")
    print(f"  Avg best solo       : {avg_solo:.4f}")
    print(f"  Avg delta           : {avg_delta:+.4f}")
    print(f"  Outputs -> {out_dir}")
    print("=" * 80)