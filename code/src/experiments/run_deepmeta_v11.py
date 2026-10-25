# src/experiments/run_deepmeta_v11.py
"""
================================================================================
DeepMetaStack v11 — 3-level hybrid neural-classical ensemble
================================================================================

Level 0 — Five classical base learners (K-fold OOF):
              XGB-gene, LGBM-gene, RF-gene, LogReg-gene, XGB-raw-chi2.
Level 1 — Frozen GeneAttentionNet (K-fold OOF, AMP-accelerated):
              outputs logits (7) + pooled embedding (128) per sample.
Level 2 — DeepMetaV11 neural meta-learner with cross-antibiotic attention,
              trained on (level 0 OOF + level 1 OOF) meta-features.

This is the FINAL ARCHITECTURE for defence — synthesizes consensus advice
from six external AI consultations. Three layers of novelty:
    (a) gene-aware attention (GeneAttentionNet, motivated by Pillar 1),
    (b) hybrid neural-classical fusion via deep meta-learner,
    (c) cross-AB attention learning co-resistance structure.

Key fixes versus v9:
    - mixed-precision (autocast + GradScaler) on GAN training loop
    - pin_memory=False (Windows DataLoader safety)
    - per-stat standardisation kept identical to Stage 2 GAN training
    - GAN forward pass exposes 128-dim pooled embedding (not just logits)
    - DeepMetaV11 replaces v8/v9's per-AB LogReg meta with a multi-task
      attention-based meta-learner

Targets:
    - Avg test PR-AUC ≥ 0.85 (matching or beating v8's 0.847)
    - Total runtime ≈ 60-90 min on RTX 4070 Ti
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
from src.models.deep_meta_v11 import DeepMetaV11
from src.models.gene_attention_net import GeneAttentionNet
from src.utils.seeds import set_global_seed

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")  # tensor cores


# ───────────────────────────────────────────────────────────────────────────────
# Small helpers (identical to v8)
# ───────────────────────────────────────────────────────────────────────────────
def _spw(y):
    pos = float((y == 1).sum()); neg = float((y == 0).sum())
    return max(1.0, neg / max(pos, 1.0))


def _variance_chunked(X, chunk=10_000):
    n = X.shape[1]; out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = X[:, s:e].astype(np.float32).var(axis=0)
    return out


def _chi2_chunked(X, y, chunk=10_000):
    n = X.shape[1]; out = np.zeros(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        c, _ = chi2(X[:, s:e].astype(np.float32), y)
        out[s:e] = np.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _best_threshold_mcc(y, p):
    best_t, best_m = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        pred = (p >= t).astype(int)
        if pred.sum() in (0, len(pred)):
            continue
        m = matthews_corrcoef(y, pred)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)


# ───────────────────────────────────────────────────────────────────────────────
# Classical base trainers (v8)
# ───────────────────────────────────────────────────────────────────────────────
def _train_xgb_gene(X_fit, y_fit, X_es, y_es, params):
    p = dict(params); p["scale_pos_weight"] = _spw(y_fit)
    p["objective"] = "binary:logistic"; p["eval_metric"] = "aucpr"
    early = p.pop("early_stopping_rounds", 150)
    n_round = p.pop("n_estimators", 1500); p.pop("n_jobs", None)
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
    chi2_stats = _chi2_chunked(X_fit_snp[:, keep_var], y_fit)
    k = int(min(prefilter_k, len(keep_var)))
    top_local = np.argpartition(chi2_stats, -k)[-k:]
    top_idx = keep_var[top_local]
    del var, chi2_stats; gc.collect()

    X_fit_top = X_fit_snp[:, top_idx].astype(np.float32)
    X_es_top = X_es_snp[:, top_idx].astype(np.float32)

    p = dict(params); p["scale_pos_weight"] = _spw(y_fit)
    p["objective"] = "binary:logistic"; p["eval_metric"] = "aucpr"
    p.setdefault("colsample_bytree", 1.0); p.setdefault("colsample_bynode", 0.5)
    early = p.pop("early_stopping_rounds", 150)
    n_round = p.pop("n_estimators", 1500); p.pop("n_jobs", None)

    dtr = xgb.DMatrix(X_fit_top, label=y_fit)
    des = xgb.DMatrix(X_es_top, label=y_es)
    booster = xgb.train(p, dtr, num_boost_round=n_round,
                        evals=[(des, "es")], early_stopping_rounds=early,
                        verbose_eval=False)
    return booster, top_idx


# ───────────────────────────────────────────────────────────────────────────────
# GeneAttentionNet training with AMP + embedding extraction
# ───────────────────────────────────────────────────────────────────────────────
def _train_gan_one_fold_amp(
    X_3d_fit, Y_fit, mask_fit, X_3d_es, Y_es, mask_es,
    n_genes, n_stats, antibiotic_names,
    max_epochs, batch_size, learning_rate,
    hidden_dim, n_heads, n_layers, ffn_mult, dropout,
    device,
):
    """AMP-patched single-fold multi-task GAN training. Returns trained model."""
    # Per-AB pos_weights from fit
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
    opt = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

    best_es_pr = -1.0
    best_state = None
    patience_left = 4
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
            scaler.step(opt); scaler.update()
        sched.step()

        # ES PR-AUC (multi-task mean) — batched inference to avoid OOM
        model.eval()
        probs_es = np.zeros_like(Y_es, dtype=np.float64)
        with torch.no_grad():
            for s in range(0, X_es_t.size(0), 32):
                logits = model(X_es_t[s:s + 32]).cpu().numpy()
                probs_es[s:s + 32] = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        prs = []
        for k in range(Y_es.shape[1]):
            mk = mask_es[:, k]
            if mk.sum() < 5: continue
            yk = Y_es[mk, k].astype(np.int32)
            if len(np.unique(yk)) < 2: continue
            prs.append(float(average_precision_score(yk, probs_es[mk, k])))
        es_pr = float(np.mean(prs)) if prs else 0.0

        print(f"        ep {ep + 1:2d}/{max_epochs}  es_pr={es_pr:.4f}  "
              f"({time.time() - t_ep:.0f}s)", flush=True)

        if es_pr > best_es_pr + 1e-4:
            best_es_pr = es_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 4
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"        early-stop at ep {ep + 1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def _gan_predict_with_embeddings(
    model: GeneAttentionNet, X_3d: np.ndarray,
    device: str = "cuda", batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run GAN forward pass; return both logits (B, n_ab) and pooled embedding (B, hidden).

    Mirrors the model's forward() but exposes the pooled vector.
    """
    model.eval()
    logits_out, emb_out = [], []
    for i in range(0, len(X_3d), batch_size):
        xb = torch.from_numpy(X_3d[i:i + batch_size]).float().to(device)
        B = xb.size(0)
        h = model.gene_encoder(xb)
        gene_ids = torch.arange(model.n_genes, device=device)
        pos = model.gene_pos_emb(gene_ids)
        h = h + pos.unsqueeze(0)
        for blk in model.blocks:
            h = blk(h)
        pooled, _ = model.pool(h)               # (B, hidden_dim)
        logits = torch.cat([head(pooled) for head in model.heads], dim=1)
        logits_out.append(logits.cpu().numpy().astype(np.float32))
        emb_out.append(pooled.cpu().numpy().astype(np.float32))
    return np.concatenate(logits_out, axis=0), np.concatenate(emb_out, axis=0)


def _run_gan_kfold(
    X_3d_norm, Y_all, mask_all, train_idx, val_idx, test_idx,
    n_genes, n_stats, label_names, n_folds, gan_cfg, device,
):
    """K-fold OOF for multi-task GAN. Returns logits + embeddings on all splits."""
    print(f"\n  [GAN K-fold with AMP] {n_folds} folds, max_epochs={gan_cfg['max_epochs']}, "
          f"batch_size={gan_cfg['batch_size']}, hidden_dim={gan_cfg['hidden_dim']}")

    n_train = len(train_idx); n_ab = Y_all.shape[1]; hid = int(gan_cfg["hidden_dim"])
    gan_oof_logits = np.zeros((n_train, n_ab), dtype=np.float32)
    gan_oof_emb = np.zeros((n_train, hid), dtype=np.float32)
    gan_val_logits = np.zeros((len(val_idx), n_ab), dtype=np.float32)
    gan_val_emb = np.zeros((len(val_idx), hid), dtype=np.float32)
    gan_test_logits = np.zeros((len(test_idx), n_ab), dtype=np.float32)
    gan_test_emb = np.zeros((len(test_idx), hid), dtype=np.float32)

    X_val = X_3d_norm[val_idx]; X_test = X_3d_norm[test_idx]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    for fold, (fit_rel, oof_rel) in enumerate(kf.split(np.arange(n_train))):
        t_fold = time.time()
        rng = np.random.RandomState(42 + fold)
        perm = rng.permutation(len(fit_rel))
        es_size = max(50, int(0.1 * len(fit_rel)))
        es_rel = fit_rel[perm[:es_size]]
        train_only_rel = fit_rel[perm[es_size:]]

        fit_abs = train_idx[train_only_rel]
        es_abs = train_idx[es_rel]
        oof_abs = train_idx[oof_rel]

        gan = _train_gan_one_fold_amp(
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

        # OOF
        oof_l, oof_e = _gan_predict_with_embeddings(gan, X_3d_norm[oof_abs], device=device)
        gan_oof_logits[oof_rel] = oof_l
        gan_oof_emb[oof_rel] = oof_e
        # Val + test (averaged across folds)
        v_l, v_e = _gan_predict_with_embeddings(gan, X_val, device=device)
        t_l, t_e = _gan_predict_with_embeddings(gan, X_test, device=device)
        gan_val_logits += v_l / n_folds
        gan_val_emb += v_e / n_folds
        gan_test_logits += t_l / n_folds
        gan_test_emb += t_e / n_folds

        print(f"    Fold {fold + 1}/{n_folds}  fit={len(fit_abs)} es={len(es_abs)} "
              f"oof={len(oof_abs)}  ({time.time() - t_fold:.0f}s)")

        del gan; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return (gan_oof_logits, gan_oof_emb,
            gan_val_logits, gan_val_emb,
            gan_test_logits, gan_test_emb)


# ───────────────────────────────────────────────────────────────────────────────
# Classical 5-base K-fold OOF — returns dense (n, n_ab, 5) matrices with 0.5
# filled where label_mask is False so meta features stay aligned across ABs.
# ───────────────────────────────────────────────────────────────────────────────
def _run_classical_kfold(
    X_flat, X_snp_int8, Y, label_mask,
    train_idx, val_idx, test_idx,
    antibiotics, label_names, exp_cfg,
):
    n_folds = int(exp_cfg.n_folds)
    use_lgbm = bool(exp_cfg.use_lgbm) and HAS_LGB
    n_train = len(train_idx); n_val = len(val_idx); n_test = len(test_idx)
    n_ab_full = Y.shape[1]
    n_base = 5

    oof_train = np.full((n_train, n_ab_full, n_base), 0.5, dtype=np.float32)
    val_avg = np.full((n_val, n_ab_full, n_base), 0.5, dtype=np.float32)
    test_avg = np.full((n_test, n_ab_full, n_base), 0.5, dtype=np.float32)

    ab_indices = [label_names.index(a) for a in antibiotics if a in label_names]

    for ab_name in antibiotics:
        if ab_name not in label_names:
            print(f"      SKIP {ab_name} (not in labels)"); continue
        ab_idx = label_names.index(ab_name)
        m = label_mask[:, ab_idx]
        tr_pos = np.where(m[train_idx])[0]
        va_pos = np.where(m[val_idx])[0]
        te_pos = np.where(m[test_idx])[0]
        tr = train_idx[tr_pos]; va = val_idx[va_pos]; te = test_idx[te_pos]
        y_tr = Y[tr, ab_idx].astype(np.int32)
        if len(np.unique(y_tr)) < 2:
            continue

        X_tr_flat = X_flat[tr]; X_va_flat = X_flat[va]; X_te_flat = X_flat[te]
        X_tr_snp = X_snp_int8[tr]; X_va_snp = X_snp_int8[va]; X_te_snp = X_snp_int8[te]

        n_pos = int(y_tr.sum())
        print(f"\n  -- classical bases for {ab_name} --  "
              f"train={len(y_tr)} (pos={n_pos}) val={len(va)} test={len(te)}")

        oof_local = np.zeros((len(tr), n_base), dtype=np.float32)
        val_local = np.zeros((len(va), n_base), dtype=np.float32)
        test_local = np.zeros((len(te), n_base), dtype=np.float32)

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        for fold, (fit_rel, oof_rel) in enumerate(skf.split(X_tr_flat, y_tr)):
            t_fold = time.time()
            sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10,
                                         random_state=42 + fold)
            rel_fit, rel_es = next(sss.split(X_tr_flat[fit_rel], y_tr[fit_rel]))
            fit_idx_local = fit_rel[rel_fit]; es_idx_local = fit_rel[rel_es]
            X_fit, y_fit = X_tr_flat[fit_idx_local], y_tr[fit_idx_local]
            X_es, y_es = X_tr_flat[es_idx_local], y_tr[es_idx_local]
            X_fit_snp = X_tr_snp[fit_idx_local]; X_es_snp = X_tr_snp[es_idx_local]

            col = 0
            # 1 — XGB gene-agg
            b = _train_xgb_gene(X_fit, y_fit, X_es, y_es, dict(exp_cfg.xgb_params))
            oof_local[oof_rel, col] = b.predict(xgb.DMatrix(X_tr_flat[oof_rel]))
            val_local[:, col] += b.predict(xgb.DMatrix(X_va_flat)) / n_folds
            test_local[:, col] += b.predict(xgb.DMatrix(X_te_flat)) / n_folds
            del b; gc.collect()
            col += 1

            # 2 — LGBM gene-agg (skip if lightgbm unavailable; col stays at 0.5)
            if use_lgbm:
                clf = _train_lgbm_gene(X_fit, y_fit, X_es, y_es, dict(exp_cfg.lgbm_params))
                oof_local[oof_rel, col] = clf.predict_proba(X_tr_flat[oof_rel])[:, 1]
                val_local[:, col] += clf.predict_proba(X_va_flat)[:, 1] / n_folds
                test_local[:, col] += clf.predict_proba(X_te_flat)[:, 1] / n_folds
                del clf; gc.collect()
            col += 1

            # 3 — RF gene-agg
            clf = _train_rf_gene(X_fit, y_fit, dict(exp_cfg.rf_params))
            oof_local[oof_rel, col] = clf.predict_proba(X_tr_flat[oof_rel])[:, 1]
            val_local[:, col] += clf.predict_proba(X_va_flat)[:, 1] / n_folds
            test_local[:, col] += clf.predict_proba(X_te_flat)[:, 1] / n_folds
            del clf; gc.collect()
            col += 1

            # 4 — LogReg gene-agg
            clf, sc = _train_logreg_gene(X_fit, y_fit)
            oof_local[oof_rel, col] = clf.predict_proba(sc.transform(X_tr_flat[oof_rel]))[:, 1]
            val_local[:, col] += clf.predict_proba(sc.transform(X_va_flat))[:, 1] / n_folds
            test_local[:, col] += clf.predict_proba(sc.transform(X_te_flat))[:, 1] / n_folds
            del clf, sc; gc.collect()
            col += 1

            # 5 — XGB raw chi²
            b, top_idx = _train_xgb_raw_chi2(
                X_fit_snp, y_fit, X_es_snp, y_es,
                prefilter_k=int(exp_cfg.prefilter_top_k),
                min_var=float(exp_cfg.prefilter_min_variance),
                params=dict(exp_cfg.xgb_raw_params),
            )
            X_oof_top = X_tr_snp[oof_rel][:, top_idx].astype(np.float32)
            X_va_top = X_va_snp[:, top_idx].astype(np.float32)
            X_te_top = X_te_snp[:, top_idx].astype(np.float32)
            oof_local[oof_rel, col] = b.predict(xgb.DMatrix(X_oof_top))
            val_local[:, col] += b.predict(xgb.DMatrix(X_va_top)) / n_folds
            test_local[:, col] += b.predict(xgb.DMatrix(X_te_top)) / n_folds
            del b, X_oof_top, X_va_top, X_te_top; gc.collect()
            col += 1

            print(f"      fold {fold + 1}/{n_folds}  ({time.time() - t_fold:.0f}s)")

        # Place into dense per-AB matrix
        oof_train[tr_pos, ab_idx, :] = oof_local
        val_avg[va_pos, ab_idx, :] = val_local
        test_avg[te_pos, ab_idx, :] = test_local

    return oof_train, val_avg, test_avg


# ───────────────────────────────────────────────────────────────────────────────
# DeepMetaV11 training with snapshot averaging
# ───────────────────────────────────────────────────────────────────────────────
def _build_meta_features(v8_oof, gan_logits, gan_emb):
    """Concatenate v8 OOF (B, n_ab, 5) + GAN logits (B, n_ab) + GAN emb (B, hid)
    into (B, n_ab*5 + n_ab + hid) flat meta-features."""
    B = v8_oof.shape[0]
    v8_flat = v8_oof.reshape(B, -1)                  # (B, n_ab * 5)
    return np.concatenate([v8_flat, gan_logits, gan_emb], axis=1).astype(np.float32)


def _train_deepmeta(
    meta_train, Y_train, mask_train,
    meta_val, Y_val, mask_val,
    n_antibiotics, cfg, device,
):
    """Train DeepMetaV11 with masked multi-task BCE + snapshot averaging."""
    n_meta = meta_train.shape[1]
    # Pos weights from train (per AB)
    pos_w = []
    for k in range(n_antibiotics):
        if mask_train[:, k].sum() < 5:
            pos_w.append(1.0); continue
        y_k = Y_train[mask_train[:, k], k]
        if len(np.unique(y_k)) < 2:
            pos_w.append(1.0); continue
        n_pos = float((y_k == 1).sum()); n_neg = float((y_k == 0).sum())
        pos_w.append(max(1.0, n_neg / max(n_pos, 1.0)))

    # Standardise meta features (train stats)
    scaler = StandardScaler().fit(meta_train)
    M_tr = scaler.transform(meta_train).astype(np.float32)
    M_va = scaler.transform(meta_val).astype(np.float32)

    model = DeepMetaV11(
        n_meta_features=n_meta,
        n_antibiotics=n_antibiotics,
        token_dim=int(cfg.token_dim),
        n_heads=int(cfg.n_heads),
        dropout=float(cfg.dropout),
        attn_dropout=float(cfg.attn_dropout),
    ).to(device)

    pw_t = torch.tensor(pos_w, dtype=torch.float32, device=device)

    tr_ds = TensorDataset(
        torch.from_numpy(M_tr),
        torch.from_numpy(Y_train.astype(np.float32)),
        torch.from_numpy(mask_train.astype(bool)),
    )
    tr_dl = DataLoader(tr_ds, batch_size=int(cfg.batch_size), shuffle=True,
                       num_workers=0, pin_memory=False)
    M_va_t = torch.from_numpy(M_va).to(device)
    Y_va_t = Y_val.astype(np.int32); mask_va_t = mask_val.astype(bool)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.lr), weight_decay=float(cfg.weight_decay),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=int(cfg.max_epochs), eta_min=1e-5,
    )

    snapshots: list[tuple[float, dict]] = []
    snapshots_k = int(cfg.snapshots_k)
    best = -1.0; patience = 0; patience_max = int(cfg.patience)

    print(f"  [meta] training DeepMetaV11 — n_meta={n_meta}, "
          f"params={sum(p.numel() for p in model.parameters())/1e3:.1f}k")

    for ep in range(int(cfg.max_epochs)):
        t_ep = time.time()
        model.train()
        for mb_in, mb_y, mb_mask in tr_dl:
            mb_in = mb_in.to(device, non_blocking=True)
            mb_y = mb_y.to(device, non_blocking=True)
            mb_mask = mb_mask.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(mb_in)
            pw_b = pw_t.view(1, -1).expand_as(logits)
            bce = F.binary_cross_entropy_with_logits(
                logits, mb_y, pos_weight=pw_b, reduction="none"
            )
            denom = mb_mask.float().sum().clamp(min=1.0)
            loss = (bce * mb_mask.float()).sum() / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Val PR-AUC (multi-task mean)
        model.eval()
        with torch.no_grad():
            val_logits = model(M_va_t).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits.astype(np.float64)))
        prs = []
        for k in range(n_antibiotics):
            mk = mask_va_t[:, k]
            if mk.sum() < 5: continue
            yk = Y_va_t[mk, k]
            if len(np.unique(yk)) < 2: continue
            prs.append(float(average_precision_score(yk, val_probs[mk, k])))
        val_pr = float(np.mean(prs)) if prs else 0.0

        state_snapshot = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        snapshots.append((val_pr, state_snapshot))
        snapshots.sort(key=lambda z: -z[0])
        snapshots = snapshots[:snapshots_k]

        improved = val_pr > best + 1e-4
        if improved:
            best = val_pr; patience = 0
        else:
            patience += 1

        print(f"    ep {ep + 1:3d}/{cfg.max_epochs}  val_pr_auc={val_pr:.4f}  "
              f"best={best:.4f}  ({time.time() - t_ep:.0f}s){' *' if improved else ''}",
              flush=True)
        if patience >= patience_max:
            print(f"    early-stop at ep {ep + 1}", flush=True)
            break

    return scaler, [s for _, s in snapshots], best


@torch.no_grad()
def _predict_deepmeta(scaler, snapshot_states, n_meta, n_antibiotics, cfg,
                      meta_X, device):
    """Predict using averaged snapshot states. Returns probabilities (B, n_ab)."""
    M = scaler.transform(meta_X).astype(np.float32)
    M_t = torch.from_numpy(M).to(device)
    probs_sum = None
    for state in snapshot_states:
        model = DeepMetaV11(
            n_meta_features=n_meta, n_antibiotics=n_antibiotics,
            token_dim=int(cfg.token_dim), n_heads=int(cfg.n_heads),
            dropout=float(cfg.dropout), attn_dropout=float(cfg.attn_dropout),
        ).to(device)
        model.load_state_dict(state); model.eval()
        out_chunks = []
        for s in range(0, M_t.size(0), 512):
            logits = model(M_t[s:s + 512]).cpu().numpy()
            out_chunks.append(1.0 / (1.0 + np.exp(-logits.astype(np.float64))))
        probs = np.concatenate(out_chunks, axis=0)
        probs_sum = probs if probs_sum is None else probs_sum + probs
    return (probs_sum / len(snapshot_states)).astype(np.float32)


# ───────────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────────
def run(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    exp_cfg = cfg.experiment
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config_used.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 80)
    print("  DeepMetaStack v11 — final hybrid neural-classical defence architecture")
    print("=" * 80)
    print(f"  LightGBM: {'YES' if HAS_LGB else 'NO'}")
    print(f"  GPU     : {'YES' if device == 'cuda' else 'NO'}  ({device})")
    print(f"  Output  : {out_dir}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
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
    splits = bundle["random_split"]
    tr_idx = np.asarray(splits["train_idx"])
    va_idx = np.asarray(splits["val_idx"])
    te_idx = np.asarray(splits["test_idx"])
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    X_snp_int8 = d["X_snp_filtered"]
    print(f"  X_flat: {X_flat.shape}, X_snp_int8: {X_snp_int8.shape}")

    antibiotics = list(exp_cfg.antibiotics)
    n_ab_full = Y.shape[1]

    # ── Standardise 3-D tensor for GAN ───────────────────────────────────────
    print("\n[2/6] Standardising 3-D tensor for GeneAttentionNet...")
    X_3d = X_flat.reshape(X_flat.shape[0], n_genes, n_stats)
    train_3d = X_3d[tr_idx]
    means = train_3d.reshape(-1, n_stats).mean(axis=0)
    stds = train_3d.reshape(-1, n_stats).std(axis=0) + 1e-7
    X_3d_norm = ((X_3d - means[None, None, :]) / stds[None, None, :]).astype(np.float32)

    # ── Phase 1: classical K-fold OOF (5 base learners) ──────────────────────
    print("\n[3/6] Classical K-fold OOF (5 bases per antibiotic)...")
    t0 = time.time()
    v8_oof_train, v8_val_avg, v8_test_avg = _run_classical_kfold(
        X_flat, X_snp_int8, Y, label_mask, tr_idx, va_idx, te_idx,
        antibiotics, label_names, exp_cfg,
    )
    print(f"  Phase 1 done in {(time.time() - t0) / 60:.1f} min")

    # ── Phase 2: GAN K-fold OOF (logits + embedding) with AMP ─────────────────
    print("\n[4/6] GAN K-fold OOF (multi-task, AMP-accelerated)...")
    t0 = time.time()
    (gan_oof_logits, gan_oof_emb,
     gan_val_logits, gan_val_emb,
     gan_test_logits, gan_test_emb) = _run_gan_kfold(
        X_3d_norm=X_3d_norm,
        Y_all=Y, mask_all=label_mask,
        train_idx=tr_idx, val_idx=va_idx, test_idx=te_idx,
        n_genes=n_genes, n_stats=n_stats, label_names=label_names,
        n_folds=int(exp_cfg.n_folds),
        gan_cfg=dict(exp_cfg.gan), device=device,
    )
    print(f"  Phase 2 done in {(time.time() - t0) / 60:.1f} min")

    # ── Phase 3: build meta features ─────────────────────────────────────────
    print("\n[5/6] Building 170-dim meta features and training DeepMetaV11...")
    meta_train = _build_meta_features(v8_oof_train, gan_oof_logits, gan_oof_emb)
    meta_val   = _build_meta_features(v8_val_avg,   gan_val_logits,  gan_val_emb)
    meta_test  = _build_meta_features(v8_test_avg,  gan_test_logits, gan_test_emb)
    print(f"  meta dim: train={meta_train.shape}, val={meta_val.shape}, "
          f"test={meta_test.shape}")

    Y_train = Y[tr_idx]; mask_train = label_mask[tr_idx]
    Y_val = Y[va_idx]; mask_val = label_mask[va_idx]
    Y_test = Y[te_idx]; mask_test = label_mask[te_idx]

    scaler, snap_states, best_val_pr = _train_deepmeta(
        meta_train, Y_train, mask_train,
        meta_val, Y_val, mask_val,
        n_antibiotics=n_ab_full,
        cfg=exp_cfg.meta, device=device,
    )
    print(f"  best val avg PR-AUC during training: {best_val_pr:.4f}")

    # ── Phase 4: predict val + test, evaluate per AB ─────────────────────────
    print("\n[6/6] Predicting + evaluating per antibiotic...")
    n_meta = meta_train.shape[1]
    val_probs = _predict_deepmeta(scaler, snap_states, n_meta, n_ab_full,
                                   exp_cfg.meta, meta_val, device=device)
    test_probs = _predict_deepmeta(scaler, snap_states, n_meta, n_ab_full,
                                    exp_cfg.meta, meta_test, device=device)

    rows = []; boot_n = int(exp_cfg.evaluation.bootstrap_n)
    boot_seed = int(exp_cfg.evaluation.bootstrap_seed)
    test_probs_by_ab: dict[str, np.ndarray] = {}

    for ab in antibiotics:
        if ab not in label_names:
            continue
        ab_idx = label_names.index(ab)
        m_va = mask_val[:, ab_idx]; m_te = mask_test[:, ab_idx]
        if m_te.sum() == 0:
            continue
        y_te = Y_test[m_te, ab_idx].astype(np.int32)
        p_te = test_probs[m_te, ab_idx]
        y_va = Y_val[m_va, ab_idx].astype(np.int32) if m_va.sum() else np.array([])
        p_va = val_probs[m_va, ab_idx] if m_va.sum() else np.array([])

        thr, _ = _best_threshold_mcc(y_va, p_va) if len(p_va) else (0.5, 0.0)
        metrics = compute_metrics(y_te, p_te, threshold=thr)
        pr_lo, pr_hi = bootstrap_pr_auc(y_te, p_te, n_boot=boot_n, seed=boot_seed)
        mcc_lo, mcc_hi = bootstrap_mcc(y_te, p_te, n_boot=boot_n, seed=boot_seed)

        # Also v8-equivalent: best of the 5 classical bases on test
        v8_test_for_ab = v8_test_avg[m_te, ab_idx, :]
        best_v8_solo = max(float(average_precision_score(y_te, v8_test_for_ab[:, b]))
                           for b in range(v8_test_for_ab.shape[1]))

        delta = metrics["pr_auc"] - best_v8_solo

        rows.append({
            "antibiotic": ab,
            "deepmeta_pr_auc": metrics["pr_auc"],
            "deepmeta_pr_auc_lo": pr_lo, "deepmeta_pr_auc_hi": pr_hi,
            "deepmeta_mcc": metrics["mcc"],
            "deepmeta_mcc_lo": mcc_lo, "deepmeta_mcc_hi": mcc_hi,
            "deepmeta_f2": metrics["f2"],
            "deepmeta_roc_auc": metrics["roc_auc"],
            "deepmeta_sens_at_90_spec": metrics["sens_at_90_spec"],
            "best_v8_solo_pr_auc": best_v8_solo,
            "delta_vs_best_v8_solo": delta,
            "threshold": thr,
            "n_test": int(m_te.sum()),
        })
        test_probs_by_ab[ab] = p_te
        print(f"  {ab:<5}  PR-AUC={metrics['pr_auc']:.4f} "
              f"[{pr_lo:.4f},{pr_hi:.4f}]  MCC={metrics['mcc']:.4f}  "
              f"Δvs best v8 solo={delta:+.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "v11_results.csv", index=False, encoding="utf-8")
    np.savez_compressed(
        out_dir / "v11_test_predictions.npz",
        **{f"probs_{ab}": p for ab, p in test_probs_by_ab.items()},
        antibiotics=np.array(list(test_probs_by_ab.keys())),
    )
    np.savez_compressed(
        out_dir / "v11_meta_features.npz",
        meta_train=meta_train, meta_val=meta_val, meta_test=meta_test,
    )

    avg_pr = float(df["deepmeta_pr_auc"].mean()) if len(df) else 0.0
    avg_solo = float(df["best_v8_solo_pr_auc"].mean()) if len(df) else 0.0
    avg_delta = float(df["delta_vs_best_v8_solo"].mean()) if len(df) else 0.0
    avg_mcc = float(df["deepmeta_mcc"].mean()) if len(df) else 0.0

    lines = [
        "# DeepMetaStack v11 - Hybrid Neural-Classical Final Architecture\n",
        "## Average across antibiotics\n",
        f"- DeepMeta PR-AUC : **{avg_pr:.4f}**",
        f"- Best v8 solo   : {avg_solo:.4f}",
        f"- Delta          : **{avg_delta:+.4f}**",
        f"- Avg MCC        : {avg_mcc:.4f}\n",
        "## Per-antibiotic\n",
        "| AB | DeepMeta PR-AUC | 95% CI | Best v8 solo | Delta | MCC | F2 |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['deepmeta_pr_auc']:.4f} "
            f"| [{r['deepmeta_pr_auc_lo']:.4f}, {r['deepmeta_pr_auc_hi']:.4f}] "
            f"| {r['best_v8_solo_pr_auc']:.4f} "
            f"| {r['delta_vs_best_v8_solo']:+.4f} "
            f"| {r['deepmeta_mcc']:.4f} | {r['deepmeta_f2']:.4f} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"  DeepMetaStack v11 COMPLETE")
    print(f"  Avg PR-AUC (deepmeta) : {avg_pr:.4f}")
    print(f"  Avg best v8 solo      : {avg_solo:.4f}")
    print(f"  Avg delta             : {avg_delta:+.4f}")
    print(f"  Avg MCC               : {avg_mcc:.4f}")
    print(f"  Outputs -> {out_dir}")
    print("=" * 80)
