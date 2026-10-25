# src/experiments/run_cascadestack_v7.py
"""
================================================================================
CascadeStack v7 — Defense-grade ensemble for AMR prediction
================================================================================

UPGRADES over v6.1:
  1. Uses gene-aggregated features (20K) instead of PCA-1024 (1024)
     → Stronger base features (XGB on gene-agg = 0.830 vs PCA = 0.799)
  2. Adds GeneAttentionNet as 6th base learner (per-fold retrained, leak-free)
     → Neural model contribution + ensemble diversity
  3. Preserves all v6.1 tunings:
     - Adaptive base-learner gating per antibiotic
     - Train-size-aware meta MLP capacity
     - Dual-track threshold (MCC vs F2) selection
     - Top-K snapshot averaging for meta MLP
     - Improved TabNet config (entmax, larger batch)

LEAK-FREE GUARANTEES:
  ✅ 3-way fold split (fit / ES / cal) — OOF rows never seen during training
  ✅ Per-fold GAN training (no train→val leak)
  ✅ Threshold selected on val_holdout (40% of val) — separate from val_meta
  ✅ Test set never used for any hyperparameter, threshold, or gating
  ✅ All randomness fixed (SEED=42)

TARGET: Beat XGBoost gene-aggregated baseline (0.830 avg PR-AUC) by ≥1%
================================================================================
"""
from __future__ import annotations

import pickle
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    accuracy_score,
    precision_recall_curve,
    roc_auc_score,
    brier_score_loss,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier
from scipy.stats import wilcoxon

import lightning as L
from omegaconf import DictConfig

try:
    import lightgbm as lgb
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    HAS_TABNET = True
except Exception:
    HAS_TABNET = False

from src.data.gene_aggregated_features import build_gene_feature_matrix_from_pkl
from src.models.gene_attention_net import GeneAttentionNet
from src.utils.seeds import set_global_seed

warnings.filterwarnings("ignore")


# ===========================================================================
# UTILITIES
# ===========================================================================
def safe_pr(y, p) -> float:
    p = np.nan_to_num(np.clip(np.asarray(p, dtype=float), 0.0, 1.0), nan=0.5)
    y = np.asarray(y)
    return float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else 0.0


def safe_auc(y, p) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")


def best_thr_mcc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_m = 0.5, -1.0
    for t in thresholds:
        pred = (p >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred):
            continue
        m = matthews_corrcoef(y, pred)
        if m > best_m:
            best_m, best_t = m, t
    return float(best_t), float(best_m)


def best_thr_f2(y, p):
    y, p = np.asarray(y), np.clip(np.asarray(p), 1e-9, 1 - 1e-9)
    prec, rec, thr = precision_recall_curve(y, p)
    if thr.size == 0:
        return 0.5, 0.0
    f2_arr = 5 * (prec[1:] * rec[1:]) / (4 * prec[1:] + rec[1:] + 1e-12)
    best_i = int(np.nanargmax(f2_arr))
    return float(thr[best_i]), float(f2_arr[best_i])


def sensitivity_at_specificity(y_true, y_score, target_spec=0.90):
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    valid = spec >= target_spec
    return float(tpr[valid].max()) if valid.any() else float("nan")


def compute_metrics(y, p, thr):
    y, p = np.asarray(y), np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    pred = (p >= thr).astype(int)
    sens90 = sensitivity_at_specificity(y, p, 0.90)
    return dict(
        auc=safe_auc(y, p),
        prauc=safe_pr(y, p),
        f1=float(f1_score(y, pred, zero_division=0)),
        f2=float(fbeta_score(y, pred, beta=2, zero_division=0)),
        rec=float(recall_score(y, pred, zero_division=0)),
        prec=float(precision_score(y, pred, zero_division=0)),
        mcc=float(matthews_corrcoef(y, pred)),
        brier=float(brier_score_loss(y, p)),
        acc=float(accuracy_score(y, pred)),
        sens90=float(sens90) if not np.isnan(sens90) else float("nan"),
        thr=float(thr),
    )


def bootstrap_ci(y_true, y_score, metric_fn, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    scores, n = [], len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_score[idx]))
        except Exception:
            continue
    if len(scores) < 50:
        return (float("nan"), float("nan"))
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def spw(y) -> float:
    y = np.asarray(y)
    pos = float(np.sum(y == 1))
    neg = float(len(y) - pos)
    return max(1.0, neg / max(1.0, pos))


def isotonic_calibrate(p_cal, y_cal, p_target, min_samples=30):
    y_cal, p_cal, p_target = np.asarray(y_cal, dtype=float), np.asarray(p_cal), np.asarray(p_target)
    if len(np.unique(y_cal)) < 2 or len(y_cal) < min_samples:
        # Fallback: Platt scaling
        try:
            lo = np.log(np.clip(p_cal, 1e-7, 1 - 1e-7) / (1 - np.clip(p_cal, 1e-7, 1 - 1e-7))).reshape(-1, 1)
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
            lr.fit(lo, y_cal)
            lo_t = np.log(np.clip(p_target, 1e-7, 1 - 1e-7) / (1 - np.clip(p_target, 1e-7, 1 - 1e-7))).reshape(-1, 1)
            return lr.predict_proba(lo_t)[:, 1].astype(np.float32)
        except Exception:
            return p_target.astype(np.float32)
    try:
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(p_cal, y_cal)
        return ir.predict(p_target).astype(np.float32)
    except Exception:
        return p_target.astype(np.float32)


# ===========================================================================
# BASE MODEL TRAINERS
# ===========================================================================
def train_xgb(Xfit, yfit, Xes, yes, params, device="cuda"):
    p = dict(params)
    p["scale_pos_weight"] = spw(yfit)
    p["eval_metric"] = "aucpr"
    p["random_state"] = 42
    p["verbosity"] = 0
    if device == "cuda":
        p["device"] = "cuda"
    clf = XGBClassifier(**p)
    clf.fit(Xfit, yfit, eval_set=[(Xes, yes)], verbose=False)
    return clf


def train_lgbm(Xfit, yfit, Xes, yes, params):
    p = dict(params)
    p["scale_pos_weight"] = spw(yfit)
    p["random_state"] = 42
    p["verbose"] = -1
    p["n_jobs"] = -1
    early_stop = p.pop("early_stopping_rounds", 150)
    clf = lgb.LGBMClassifier(**p)
    cbs = [lgb.early_stopping(early_stop, verbose=False), lgb.log_evaluation(-1)]
    clf.fit(Xfit, yfit, eval_set=[(Xes, yes)], eval_metric="average_precision", callbacks=cbs)
    return clf


def train_rf(Xfit, yfit):
    return RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=10,
        class_weight="balanced_subsample", n_jobs=-1, random_state=42
    ).fit(Xfit, yfit)


def train_logreg(Xfit, yfit):
    sc = StandardScaler().fit(Xfit)
    clf = LogisticRegression(
        C=0.1, penalty="l2", solver="lbfgs",
        max_iter=3000, class_weight="balanced", random_state=42,
    )
    clf.fit(sc.transform(Xfit), yfit)
    return clf, sc


def train_tabnet(Xfit, yfit, Xes, yes, params, device="cuda"):
    p = dict(params)
    max_epochs = p.pop("max_epochs", 200)
    patience = p.pop("patience", 30)
    batch_size = p.pop("batch_size", 512)
    vb_size = p.pop("virtual_batch_size", 128)
    lr = p.pop("learning_rate", 2e-2)
    
    clf = TabNetClassifier(
        n_d=p.get("n_d", 24), n_a=p.get("n_a", 24), n_steps=p.get("n_steps", 5),
        gamma=p.get("gamma", 1.5), n_independent=2, n_shared=2,
        momentum=p.get("momentum", 0.3), mask_type=p.get("mask_type", "entmax"),
        seed=42, verbose=0, device_name=device,
        optimizer_fn=torch.optim.AdamW,
        optimizer_params=dict(lr=lr, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params=dict(T_max=80, eta_min=1e-5),
    )
    clf.fit(
        Xfit.astype(np.float32), yfit.astype(np.int64),
        eval_set=[(Xes.astype(np.float32), yes.astype(np.int64))],
        eval_metric=["auc"], max_epochs=max_epochs, patience=patience,
        batch_size=batch_size, virtual_batch_size=vb_size,
        weights={0: 1.0, 1: spw(yfit)}, drop_last=False,
    )
    return clf


def train_gan_per_fold(
    X_3d_fit, y_multi_fit, mask_multi_fit,
    X_3d_es, y_multi_es, mask_multi_es,
    n_genes, n_stats, antibiotic_names, ab_idx, gan_cfg, device="cuda",
):
    """Train GeneAttentionNet on fold's fit partition, predict in inference.
    
    Returns model trained on fit-only data (no val leak).
    Multi-task training (all antibiotics) — single model for all targets.
    """
    pos_weights = [spw(y_multi_fit[mask_multi_fit[:, i], i]) for i in range(y_multi_fit.shape[1])]
    
    model = GeneAttentionNet(
        n_genes=n_genes, n_stats=n_stats,
        n_antibiotics=y_multi_fit.shape[1],
        antibiotic_names=antibiotic_names,
        hidden_dim=gan_cfg.get("hidden_dim", 128),
        n_heads=gan_cfg.get("n_heads", 4),
        n_layers=gan_cfg.get("n_layers", 2),
        ffn_mult=gan_cfg.get("ffn_mult", 2),
        dropout=gan_cfg.get("dropout", 0.3),
        pos_weights=pos_weights,
        learning_rate=gan_cfg.get("learning_rate", 1e-3),
        weight_decay=1e-4,
        scheduler_t_max=gan_cfg.get("max_epochs", 15),
    ).to(device)
    
    # Build dataloaders
    X_fit_t = torch.from_numpy(X_3d_fit).float()
    y_fit_t = torch.from_numpy(y_multi_fit).float()
    m_fit_t = torch.from_numpy(mask_multi_fit).bool()
    
    X_es_t = torch.from_numpy(X_3d_es).float()
    y_es_t = torch.from_numpy(y_multi_es).float()
    m_es_t = torch.from_numpy(mask_multi_es).bool()
    
    train_ds = TensorDataset(X_fit_t, y_fit_t, m_fit_t)
    train_dl = DataLoader(train_ds, batch_size=gan_cfg.get("batch_size", 16), shuffle=True, drop_last=False)
    
    opt = torch.optim.AdamW(model.parameters(), lr=gan_cfg.get("learning_rate", 1e-3), weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=gan_cfg.get("max_epochs", 15), eta_min=1e-6)
    
    best_es_pr = 0.0
    best_state = None
    patience_left = 5
    max_epochs = gan_cfg.get("max_epochs", 15)
    
    for epoch in range(max_epochs):
        model.train()
        for xb, yb, mb in train_dl:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            logits = model(xb)
            pw = model.pos_weights.view(1, -1).expand_as(logits)
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, yb, pos_weight=pw, reduction="none"
            )
            denom = mb.float().sum().clamp(min=1.0)
            loss = (bce * mb.float()).sum() / denom
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        # Validate on ES set (ab_idx only — for early stop)
        model.eval()
        with torch.no_grad():
            X_es_dev = X_es_t.to(device)
            logits_es = model(X_es_dev).cpu().numpy()
        probs_es = 1.0 / (1.0 + np.exp(-logits_es))
        m_es_ab = mask_multi_es[:, ab_idx]
        if m_es_ab.sum() < 10 or len(np.unique(y_multi_es[m_es_ab, ab_idx])) < 2:
            continue
        es_pr = safe_pr(y_multi_es[m_es_ab, ab_idx], probs_es[m_es_ab, ab_idx])
        
        if es_pr > best_es_pr + 0.001:
            best_es_pr = es_pr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = 5
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_gan(model, X_3d, device="cuda"):
    """Multi-task prediction. Returns (n_samples, n_antibiotics) probabilities."""
    model.eval()
    X_t = torch.from_numpy(X_3d).float()
    probs_all = []
    bs = 32
    with torch.no_grad():
        for i in range(0, len(X_t), bs):
            xb = X_t[i:i+bs].to(device)
            logits = model(xb).cpu().numpy()
            probs = 1.0 / (1.0 + np.exp(-logits))
            probs_all.append(probs)
    return np.concatenate(probs_all, axis=0)


# ===========================================================================
# META FEATURES
# ===========================================================================
def make_meta_features(probs, active_mask=None):
    if active_mask is not None:
        probs = probs[:, active_mask]
    probs = np.nan_to_num(np.clip(probs.astype(float), 0.0, 1.0), nan=0.5)
    n_m = max(probs.shape[1], 1)
    if probs.shape[1] == 0:
        return np.zeros((0, 0))
    feats = [probs]
    
    p_max = probs.max(axis=1, keepdims=True)
    p_min = probs.min(axis=1, keepdims=True)
    p_mean = probs.mean(axis=1, keepdims=True)
    p_std = probs.std(axis=1, keepdims=True)
    p_rng = p_max - p_min
    feats += [p_max, p_min, p_mean, p_std, p_rng]
    
    p_clip = np.clip(probs, 1e-8, 1 - 1e-8)
    entropy = -(p_clip * np.log(p_clip) + (1 - p_clip) * np.log(1 - p_clip)).mean(axis=1, keepdims=True)
    feats.append(entropy)
    
    ranks = np.argsort(np.argsort(probs, axis=1), axis=1).astype(float)
    feats.append(ranks.mean(axis=1, keepdims=True) / max(n_m - 1, 1))
    
    votes = (probs > 0.5).sum(axis=1, keepdims=True).astype(float) / n_m
    feats.append(votes)
    
    mean_clipped = np.clip(p_mean, 1e-7, 1 - 1e-7)
    log_odds = np.log(mean_clipped / (1 - mean_clipped))
    feats.append(log_odds)
    
    feats.append(p_max * p_min)
    
    return np.hstack(feats)


# ===========================================================================
# META MLP
# ===========================================================================
class MetaMLPSmall(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(32, 16), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class MetaMLPLarge(nn.Module):
    def __init__(self, n_input):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_input, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.25),
            nn.Linear(64, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(32, 16), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_meta_mlp(X_meta_tr, y_tr, X_meta_val, y_val, meta_cfg, device="cuda"):
    sc = StandardScaler().fit(X_meta_tr)
    Xtr_s = sc.transform(X_meta_tr).astype("float32")
    Xva_s = sc.transform(X_meta_val).astype("float32")
    
    small_thr = meta_cfg.get("small_train_threshold", 3000)
    Model = MetaMLPSmall if len(y_tr) < small_thr else MetaMLPLarge
    capacity_tag = "small" if len(y_tr) < small_thr else "large"
    n_input = X_meta_tr.shape[1]
    
    model = Model(n_input).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=meta_cfg.get("lr", 3e-4), weight_decay=meta_cfg.get("weight_decay", 1e-3))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=meta_cfg.get("max_epochs", 150), eta_min=1e-5)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([spw(y_tr)], dtype=torch.float32, device=device),
    )
    
    tr_dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr_s), torch.from_numpy(y_tr.astype("float32"))),
        batch_size=meta_cfg.get("batch_size", 128), shuffle=True, drop_last=False,
    )
    
    snapshots_k = meta_cfg.get("snapshots_k", 3)
    max_epochs = meta_cfg.get("max_epochs", 150)
    patience_max = meta_cfg.get("patience", 25)
    
    snapshots = []
    wait, last_best = 0, 0.0
    for ep in range(max_epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            p_va = torch.sigmoid(model(torch.from_numpy(Xva_s).to(device))).cpu().numpy()
        pr = safe_pr(y_val, p_va)
        
        snap_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        snapshots.append((pr, snap_state))
        snapshots.sort(key=lambda z: -z[0])
        snapshots = snapshots[:snapshots_k]
        
        if snapshots[0][0] > last_best + 0.0005:
            last_best = snapshots[0][0]
            wait = 0
        else:
            wait += 1
            if wait >= patience_max:
                break
    
    return [s for _, s in snapshots], sc, last_best, capacity_tag, Model, n_input


def pred_meta_mlp(snapshot_states, Model, n_input, X_meta, sc, device="cuda"):
    X_s = torch.from_numpy(sc.transform(X_meta).astype("float32"))
    preds = []
    for state in snapshot_states:
        model = Model(n_input).to(device)
        model.load_state_dict(state)
        model.eval()
        out = []
        with torch.no_grad():
            for (xb,) in DataLoader(TensorDataset(X_s), batch_size=512):
                out.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        preds.append(np.concatenate(out))
    return np.mean(preds, axis=0)


# ===========================================================================
# OOF ENGINE
# ===========================================================================
def run_oof_for_antibiotic(
    X_flat, X_3d, y_full, mask_full,
    train_idx_ab, val_idx_ab, test_idx_ab,
    n_genes, n_stats, antibiotic_names, ab_idx,
    cfg, exp_cfg, device="cuda",
):
    """Run K-fold OOF for one antibiotic with all base learners.
    
    Returns oof_p, val_p, te_p arrays with shape (n_samples, n_active_bases).
    """
    n_folds = exp_cfg.ensemble.n_folds
    oof_es_frac = exp_cfg.ensemble.oof_es_frac
    oof_cal_frac = exp_cfg.ensemble.oof_cal_frac
    use_lgbm = exp_cfg.base_models.use_lgbm and HAS_LGB
    use_tabnet = exp_cfg.base_models.use_tabnet and HAS_TABNET
    use_gan = exp_cfg.use_gan_base and exp_cfg.gan_per_fold.enabled
    
    base_names = ["XGBoost"]
    if use_lgbm: base_names.append("LightGBM")
    base_names += ["RF", "LogReg"]
    if use_tabnet: base_names.append("TabNet")
    if use_gan: base_names.append("GeneAttentionNet")
    n_base = len(base_names)
    
    # Get per-AB labels (filtered to valid samples)
    y_tr = y_full[train_idx_ab, ab_idx].astype(np.int32)
    y_val = y_full[val_idx_ab, ab_idx].astype(np.int32)
    y_te = y_full[test_idx_ab, ab_idx].astype(np.int32)
    
    X_tr_flat = X_flat[train_idx_ab]
    X_val_flat = X_flat[val_idx_ab]
    X_te_flat = X_flat[test_idx_ab]
    
    X_tr_3d = X_3d[train_idx_ab]
    X_val_3d = X_3d[val_idx_ab]
    X_te_3d = X_3d[test_idx_ab]
    
    # For GAN multi-task training, need full Y + mask matrices
    Y_tr_multi = y_full[train_idx_ab].astype(np.float32)
    mask_tr_multi = mask_full[train_idx_ab].astype(bool)
    Y_val_multi = y_full[val_idx_ab].astype(np.float32)
    mask_val_multi = mask_full[val_idx_ab].astype(bool)
    Y_te_multi = y_full[test_idx_ab].astype(np.float32)
    mask_te_multi = mask_full[test_idx_ab].astype(bool)
    
    oof_p = np.zeros((len(y_tr), n_base), dtype=np.float32)
    val_p = np.zeros((len(y_val), n_base), dtype=np.float32)
    te_p = np.zeros((len(y_te), n_base), dtype=np.float32)
    oof_pr_per_fold = {n: [] for n in base_names}
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    for fold, (fold_tr_idx, fold_va_idx) in enumerate(skf.split(X_tr_flat, y_tr)):
        Xf_tr_flat, yf_tr = X_tr_flat[fold_tr_idx], y_tr[fold_tr_idx]
        Xf_va_flat, yf_va = X_tr_flat[fold_va_idx], y_tr[fold_va_idx]
        Xf_tr_3d = X_tr_3d[fold_tr_idx]
        Xf_va_3d = X_tr_3d[fold_va_idx]
        
        # Multi-task Y/mask for fold (for GAN)
        Yf_tr_multi = Y_tr_multi[fold_tr_idx]
        maskf_tr_multi = mask_tr_multi[fold_tr_idx]
        
        # 3-way split of fold-train
        rest_frac = oof_es_frac + oof_cal_frac
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=rest_frac, random_state=42 + fold)
        fit_rel, rest_rel = next(sss1.split(Xf_tr_flat, yf_tr))
        Xfit_flat, yfit = Xf_tr_flat[fit_rel], yf_tr[fit_rel]
        Xrest_flat, yrest = Xf_tr_flat[rest_rel], yf_tr[rest_rel]
        Xfit_3d = Xf_tr_3d[fit_rel]
        Xrest_3d = Xf_tr_3d[rest_rel]
        Yfit_multi = Yf_tr_multi[fit_rel]
        maskfit_multi = maskf_tr_multi[fit_rel]
        
        cal_share = oof_cal_frac / rest_frac
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=cal_share, random_state=42 + fold + 100)
        es_rel, cal_rel = next(sss2.split(Xrest_flat, yrest))
        Xes_flat, yes = Xrest_flat[es_rel], yrest[es_rel]
        Xcal_flat, ycal = Xrest_flat[cal_rel], yrest[cal_rel]
        Xes_3d = Xrest_3d[es_rel]
        Xcal_3d = Xrest_3d[cal_rel]
        Yes_multi = Yf_tr_multi[rest_rel][es_rel]
        maskes_multi = maskf_tr_multi[rest_rel][es_rel]
        
        n_fit, n_es, n_cal = len(yfit), len(yes), len(ycal)
        print(f"      Fold {fold+1}/{n_folds}  [fit={n_fit} | ES={n_es} | cal={n_cal} | OOF={len(yf_va)}]", end=" ", flush=True)
        t_fold = time.time()
        
        col = 0
        # XGBoost
        m = train_xgb(Xfit_flat, yfit, Xes_flat, yes, dict(exp_cfg.xgb_params), device=device)
        p_c = m.predict_proba(Xcal_flat)[:, 1]
        oof_p[fold_va_idx, col] = isotonic_calibrate(p_c, ycal, m.predict_proba(Xf_va_flat)[:, 1])
        val_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(X_val_flat)[:, 1]) / n_folds
        te_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(X_te_flat)[:, 1]) / n_folds
        oof_pr_per_fold["XGBoost"].append(safe_pr(yf_va, oof_p[fold_va_idx, col]))
        col += 1
        
        # LightGBM
        if use_lgbm:
            m = train_lgbm(Xfit_flat, yfit, Xes_flat, yes, dict(exp_cfg.lgbm_params))
            p_c = m.predict_proba(Xcal_flat)[:, 1]
            oof_p[fold_va_idx, col] = isotonic_calibrate(p_c, ycal, m.predict_proba(Xf_va_flat)[:, 1])
            val_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(X_val_flat)[:, 1]) / n_folds
            te_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(X_te_flat)[:, 1]) / n_folds
            oof_pr_per_fold["LightGBM"].append(safe_pr(yf_va, oof_p[fold_va_idx, col]))
            col += 1
        
        # RF
        m = train_rf(Xfit_flat, yfit)
        p_c = m.predict_proba(Xcal_flat)[:, 1]
        oof_p[fold_va_idx, col] = isotonic_calibrate(p_c, ycal, m.predict_proba(Xf_va_flat)[:, 1])
        val_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(X_val_flat)[:, 1]) / n_folds
        te_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(X_te_flat)[:, 1]) / n_folds
        oof_pr_per_fold["RF"].append(safe_pr(yf_va, oof_p[fold_va_idx, col]))
        col += 1
        
        # LogReg
        m, sc_lr = train_logreg(Xfit_flat, yfit)
        p_c = m.predict_proba(sc_lr.transform(Xcal_flat))[:, 1]
        oof_p[fold_va_idx, col] = isotonic_calibrate(p_c, ycal, m.predict_proba(sc_lr.transform(Xf_va_flat))[:, 1])
        val_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(sc_lr.transform(X_val_flat))[:, 1]) / n_folds
        te_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(sc_lr.transform(X_te_flat))[:, 1]) / n_folds
        oof_pr_per_fold["LogReg"].append(safe_pr(yf_va, oof_p[fold_va_idx, col]))
        col += 1
        
        # TabNet
        if use_tabnet:
            m = train_tabnet(Xfit_flat, yfit, Xes_flat, yes, dict(exp_cfg.tabnet_params), device=device)
            p_c = m.predict_proba(Xcal_flat.astype(np.float32))[:, 1]
            oof_p[fold_va_idx, col] = isotonic_calibrate(p_c, ycal, m.predict_proba(Xf_va_flat.astype(np.float32))[:, 1])
            val_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(X_val_flat.astype(np.float32))[:, 1]) / n_folds
            te_p[:, col] += isotonic_calibrate(p_c, ycal, m.predict_proba(X_te_flat.astype(np.float32))[:, 1]) / n_folds
            oof_pr_per_fold["TabNet"].append(safe_pr(yf_va, oof_p[fold_va_idx, col]))
            col += 1
        
        # GeneAttentionNet (per-fold trained, multi-task)
        if use_gan:
            gan_model = train_gan_per_fold(
                Xfit_3d, Yfit_multi, maskfit_multi,
                Xes_3d, Yes_multi, maskes_multi,
                n_genes, n_stats, antibiotic_names, ab_idx,
                dict(exp_cfg.gan_per_fold), device=device,
            )
            # Predict on fold OOF, outer val, outer test
            p_oof_gan_multi = predict_gan(gan_model, Xf_va_3d, device=device)
            p_val_gan_multi = predict_gan(gan_model, X_val_3d, device=device)
            p_te_gan_multi = predict_gan(gan_model, X_te_3d, device=device)
            p_cal_gan_multi = predict_gan(gan_model, Xcal_3d, device=device)
            
            # Use only ab_idx column
            p_oof_gan = p_oof_gan_multi[:, ab_idx]
            p_val_gan = p_val_gan_multi[:, ab_idx]
            p_te_gan = p_te_gan_multi[:, ab_idx]
            p_cal_gan = p_cal_gan_multi[:, ab_idx]
            
            oof_p[fold_va_idx, col] = isotonic_calibrate(p_cal_gan, ycal, p_oof_gan)
            val_p[:, col] += isotonic_calibrate(p_cal_gan, ycal, p_val_gan) / n_folds
            te_p[:, col] += isotonic_calibrate(p_cal_gan, ycal, p_te_gan) / n_folds
            oof_pr_per_fold["GeneAttentionNet"].append(safe_pr(yf_va, oof_p[fold_va_idx, col]))
            col += 1
            
            del gan_model
            torch.cuda.empty_cache()
        
        print(f"({time.time() - t_fold:.0f}s)")
    
    return oof_p, val_p, te_p, base_names, oof_pr_per_fold, (y_tr, y_val, y_te)


# ===========================================================================
# MAIN
# ===========================================================================
def run(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    exp_cfg = cfg.experiment
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("  CascadeStack v7 — Defense-grade ensemble on gene-aggregated features")
    print("=" * 80)
    print(f"  Device : {device}")
    print(f"  LightGBM: {'✅' if HAS_LGB else '❌'}")
    print(f"  TabNet  : {'✅' if HAS_TABNET else '❌'}")
    print(f"  Output  : {out_dir}")
    
   # ----- Load gene-aggregated features -----
    print("\n[1/4] Loading gene-aggregated features...")
    t0 = time.time()
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=str(exp_cfg.filtered_pkl),
        stats=tuple(exp_cfg.gene_stats),
    )
    X_flat = bundle["X_gene_features"].astype(np.float32)
    Y = bundle["Y"].astype(np.int64)
    label_mask = bundle["label_mask"].astype(bool)
    label_names = [str(x) for x in bundle["label_names"]]
    ab_to_idx = {name: i for i, name in enumerate(label_names)}
    splits = bundle["random_split"]
    train_idx_full = np.array(splits["train_idx"])
    val_idx_full = np.array(splits["val_idx"])
    test_idx_full = np.array(splits["test_idx"])
    n_genes = len(bundle["gene_ids"])
    n_stats = len(exp_cfg.gene_stats)
    
    print(f"  X_flat: {X_flat.shape}, dtype={X_flat.dtype}")
    print(f"  Labels: {label_names}")
    print(f"  Loaded in {time.time() - t0:.1f}s")
    
    # Reshape for GAN: (n, n_genes * n_stats) → (n, n_genes, n_stats)
    # Then standardize per-stat across all samples (same as Stage 2)
    print("\n[2/4] Standardising per-stat for GAN input...")
    X_3d = X_flat.reshape(X_flat.shape[0], n_genes, n_stats)
    
    # Compute stats from train portion only
    X_train_3d = X_3d[train_idx_full]
    means = X_train_3d.reshape(-1, n_stats).mean(axis=0)
    stds = X_train_3d.reshape(-1, n_stats).std(axis=0) + 1e-7
    X_3d_std = ((X_3d - means[None, None, :]) / stds[None, None, :]).astype(np.float32)
    print(f"  X_3d_std: {X_3d_std.shape}, per-stat means: {means.round(3).tolist()}")
    
    # Also standardize flat for non-GAN models — fit on train only
    sc_flat = StandardScaler().fit(X_flat[train_idx_full])
    X_flat_std = sc_flat.transform(X_flat).astype(np.float32)
    
    # ----- Per-antibiotic CascadeStack -----
    print("\n[3/4] Running CascadeStack per antibiotic...")
    
    all_rows = []
    decision_log = {}
    gating_log = {}
    base_summary = {}
    
    antibiotics = list(exp_cfg.antibiotics)
    
    for ab in antibiotics:
        if ab not in ab_to_idx:
            print(f"\n  WARNING: {ab} not in dataset, skipping")
            continue
        ab_idx = ab_to_idx[ab]
        
        print(f"\n{'─' * 80}")
        print(f"  ANTIBIOTIC: {ab} (idx={ab_idx})")
        print(f"{'─' * 80}")
        
        # Per-AB valid samples
        train_idx_ab = train_idx_full[label_mask[train_idx_full, ab_idx]]
        val_idx_ab = val_idx_full[label_mask[val_idx_full, ab_idx]]
        test_idx_ab = test_idx_full[label_mask[test_idx_full, ab_idx]]
        
        y_tr_ab = Y[train_idx_ab, ab_idx]
        y_val_ab = Y[val_idx_ab, ab_idx]
        y_te_ab = Y[test_idx_ab, ab_idx]
        
        print(f"  Train={len(y_tr_ab)} (pos={int(y_tr_ab.sum())}, {100*y_tr_ab.mean():.1f}%)")
        print(f"  Val={len(y_val_ab)} (pos={int(y_val_ab.sum())})")
        print(f"  Test={len(y_te_ab)} (pos={int(y_te_ab.sum())})")
        
        # Val split: meta-ES + threshold-holdout
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=(1 - exp_cfg.ensemble.val_meta_frac), random_state=42)
        meta_va_idx, holdout_va_idx = next(sss_val.split(X_flat_std[val_idx_ab], y_val_ab))
        
        # OOF for this antibiotic
        oof_p, val_p, te_p, base_names, oof_pr_per_fold, (y_tr, y_val, y_te) = run_oof_for_antibiotic(
            X_flat_std, X_3d_std, Y, label_mask,
            train_idx_ab, val_idx_ab, test_idx_ab,
            n_genes, n_stats, label_names, ab_idx,
            cfg, exp_cfg, device=device,
        )
        
        # Diagnostics
        print(f"\n      OOF PR-AUC per fold:")
        for n_b in base_names:
            folds_pr = oof_pr_per_fold[n_b]
            if folds_pr:
                print(f"        {n_b:<18} folds={[f'{v:.3f}' for v in folds_pr]} mean={np.mean(folds_pr):.3f}")
        
        # ----- TUNE 1: Adaptive gating -----
        base_oof_means = [float(np.mean(oof_pr_per_fold[n])) if oof_pr_per_fold[n] else 0.0 for n in base_names]
        base_val_prs = [safe_pr(y_val, val_p[:, i]) for i in range(len(base_names))]
        
        active_mask = np.ones(len(base_names), dtype=bool)
        gate_decisions = []
        gap_thr = exp_cfg.gating.gap_threshold
        pr_thr = exp_cfg.gating.pr_threshold
        val_pr_gate_thr = exp_cfg.gating.val_pr_gate_threshold
        min_base = exp_cfg.gating.min_base_after_gate
        
        for i, bn in enumerate(base_names):
            oof_m = base_oof_means[i]
            val_pr = base_val_prs[i]
            gap = abs(val_pr - oof_m)
            reasons = []
            if gap > gap_thr and val_pr < val_pr_gate_thr:
                reasons.append(f"gap={gap:.3f}>{gap_thr} AND val={val_pr:.3f}<{val_pr_gate_thr}")
            if oof_m < pr_thr:
                reasons.append(f"OOF_PR={oof_m:.3f}<{pr_thr}")
            if reasons:
                active_mask[i] = False
                gate_decisions.append((bn, "GATE_OUT", " AND ".join(reasons)))
            else:
                gate_decisions.append((bn, "KEEP", f"OOF={oof_m:.3f} val={val_pr:.3f}"))
        
        if active_mask.sum() < min_base:
            order = np.argsort(-np.array(base_oof_means))
            for i in order:
                if not active_mask[i]:
                    active_mask[i] = True
                    gate_decisions[i] = (base_names[i], "RE_ENABLED", "safety floor")
                if active_mask.sum() >= min_base:
                    break
        
        print(f"\n      Gating:")
        for bn, d, r in gate_decisions:
            print(f"        {bn:<18} [{d:<11}] {r}")
        n_active = int(active_mask.sum())
        gating_log[ab] = [{"name": n, "decision": d, "reason": r} for n, d, r in gate_decisions]
        
        # Build meta features with gating
        meta_tr = make_meta_features(oof_p, active_mask)
        meta_val = make_meta_features(val_p, active_mask)
        meta_te = make_meta_features(te_p, active_mask)
        print(f"      Meta feature dim (after gating): {meta_tr.shape[1]}")
        
        # Base reference (full val)
        base_pr_full = [safe_pr(y_val, val_p[:, i]) for i in range(len(base_names))]
        print(f"\n      Base val PR-AUC (full val):")
        for n_b, v in zip(base_names, base_pr_full):
            print(f"        {n_b:<18} {v:.4f}")
        best_base_val = max(base_pr_full)
        best_base_idx = int(np.argmax(base_pr_full))
        
        # Per-base test metrics (for paper comparison)
        per_base_test = {}
        for i, bn in enumerate(base_names):
            thr_b, _ = best_thr_mcc(y_val, val_p[:, i])
            per_base_test[bn] = compute_metrics(y_te, te_p[:, i], thr_b)
        
        # ----- Train Meta MLP -----
        meta_val_meta = meta_val[meta_va_idx]
        meta_val_hold = meta_val[holdout_va_idx]
        val_base_meta = val_p[meta_va_idx]
        val_base_hold = val_p[holdout_va_idx]
        y_val_meta = y_val[meta_va_idx]
        y_val_hold = y_val[holdout_va_idx]
        
        print(f"\n      Training meta MLP (snapshot averaging, capacity-aware)...")
        snap_states, meta_sc, meta_best_pr, capacity_tag, ModelCls, n_in = train_meta_mlp(
            meta_tr, y_tr, meta_val_meta, y_val_meta, dict(exp_cfg.meta_mlp), device=device,
        )
        print(f"      capacity={capacity_tag}  best_val_pr={meta_best_pr:.4f}  snapshots={len(snap_states)}")
        
        p_meta_val_meta = pred_meta_mlp(snap_states, ModelCls, n_in, meta_val_meta, meta_sc, device=device)
        p_meta_val_hold = pred_meta_mlp(snap_states, ModelCls, n_in, meta_val_hold, meta_sc, device=device)
        p_meta_val_full = pred_meta_mlp(snap_states, ModelCls, n_in, meta_val, meta_sc, device=device)
        p_meta_te = pred_meta_mlp(snap_states, ModelCls, n_in, meta_te, meta_sc, device=device)
        
        best_base_meta_pr = max(safe_pr(y_val_meta, val_base_meta[:, i]) for i in range(len(base_names)))
        meta_pr_on_meta = safe_pr(y_val_meta, p_meta_val_meta)
        delta_meta = meta_pr_on_meta - best_base_meta_pr
        print(f"      val_meta PR-AUC: meta={meta_pr_on_meta:.4f}  best_base={best_base_meta_pr:.4f}  Δ={delta_meta:+.4f}")
        
        # Fallback to weighted avg if meta loses on val_meta
        use_meta = meta_pr_on_meta >= best_base_meta_pr
        if not use_meta:
            w = np.array(base_pr_full)
            w = w / (w.sum() + 1e-12)
            p_meta_te = (te_p * w).sum(axis=1)
            p_meta_val_hold = (val_base_hold * w).sum(axis=1)
            p_meta_val_full = (val_p * w).sum(axis=1)
            print(f"      ⚠ Fallback to weighted avg: val_full PR-AUC={safe_pr(y_val, p_meta_val_full):.4f}")
        
        decision_log[ab] = {
            "used": "meta_mlp" if use_meta else "weighted_avg",
            "delta_on_val_meta": round(delta_meta, 4),
            "meta_capacity": capacity_tag,
            "n_active_bases": n_active,
            "n_snapshots_avg": len(snap_states),
            "best_base_name": base_names[best_base_idx],
            "best_base_val_pr": round(best_base_val, 4),
        }
        
        # ----- TUNE 3: Dual-track threshold -----
        thr_mcc, mcc_at_mcc = best_thr_mcc(y_val_hold, p_meta_val_hold)
        thr_f2, f2_at_f2 = best_thr_f2(y_val_hold, p_meta_val_hold)
        pred_at_mcc = (p_meta_val_hold >= thr_mcc).astype(int)
        pred_at_f2 = (p_meta_val_hold >= thr_f2).astype(int)
        f2_at_mcc = float(fbeta_score(y_val_hold, pred_at_mcc, beta=2, zero_division=0))
        mcc_at_f2 = float(matthews_corrcoef(y_val_hold, pred_at_f2)) if pred_at_f2.sum() > 0 else 0.0
        score_mcc = float(np.sqrt(max(mcc_at_mcc, 0.0) * max(f2_at_mcc, 0.0)))
        score_f2 = float(np.sqrt(max(mcc_at_f2, 0.0) * max(f2_at_f2, 0.0)))
        if score_mcc >= score_f2:
            thr, thr_choice = thr_mcc, "MCC"
        else:
            thr, thr_choice = thr_f2, "F2"
        print(f"\n      Threshold: [{thr_choice}] thr={thr:.3f}")
        
        # ----- Test eval -----
        m_out = compute_metrics(y_te, p_meta_te, thr)
        ci_prauc = bootstrap_ci(y_te, p_meta_te, average_precision_score, n_boot=exp_cfg.evaluation.bootstrap_n)
        ci_mcc = bootstrap_ci(y_te, p_meta_te, lambda y, p: matthews_corrcoef(y, (p >= thr).astype(int)), n_boot=exp_cfg.evaluation.bootstrap_n)
        m_out.update({
            "prauc_ci_lo": ci_prauc[0], "prauc_ci_hi": ci_prauc[1],
            "mcc_ci_lo": ci_mcc[0], "mcc_ci_hi": ci_mcc[1],
            "thr_choice": thr_choice,
        })
        
        # Best base test
        best_te_pr = max(safe_pr(y_te, te_p[:, i]) for i in range(len(base_names)))
        d_pr = m_out["prauc"] - best_te_pr
        d_mcc = m_out["mcc"] - max(per_base_test[bn]["mcc"] for bn in base_names)
        
        print(f"\n      ── TEST RESULT ──")
        print(f"      PR-AUC: {m_out['prauc']:.4f} [{ci_prauc[0]:.4f}, {ci_prauc[1]:.4f}]  Δ_best={d_pr:+.4f}")
        print(f"      MCC   : {m_out['mcc']:.4f} [{ci_mcc[0]:.4f}, {ci_mcc[1]:.4f}]  Δ_best={d_mcc:+.4f}")
        print(f"      F1={m_out['f1']:.4f} F2={m_out['f2']:.4f} Sens@90={m_out['sens90']:.4f}")
        
        all_rows.append({
            "antibiotic": ab,
            **{f"test_{k}": v for k, v in m_out.items()},
            "best_base_test_prauc": best_te_pr,
            "delta_vs_best_base": d_pr,
            **{f"base_{bn}_test_prauc": per_base_test[bn]["prauc"] for bn in base_names},
        })
        base_summary[ab] = {bn: per_base_test[bn] for bn in base_names}
    
    # ----- Summary -----
    print(f"\n\n{'=' * 80}")
    print("  CascadeStack v7 — FINAL SUMMARY")
    print(f"{'=' * 80}")
    
    df = pd.DataFrame(all_rows)
    print(f"\n  {'Antibiotic':<10} {'PR-AUC':>7} {'CI':>20} {'MCC':>6} {'Δbest':>7} {'F2':>6} {'Sens90':>7}")
    print("  " + "-" * 70)
    for _, r in df.iterrows():
        ci_str = f"[{r['test_prauc_ci_lo']:.3f}, {r['test_prauc_ci_hi']:.3f}]"
        print(f"  {r['antibiotic']:<10} {r['test_prauc']:>7.4f} {ci_str:>20} {r['test_mcc']:>6.3f} {r['delta_vs_best_base']:>+7.4f} {r['test_f2']:>6.3f} {r['test_sens90']:>7.3f}")
    
    avg_prauc = df["test_prauc"].mean()
    avg_mcc = df["test_mcc"].mean()
    avg_delta = df["delta_vs_best_base"].mean()
    print(f"\n  AVERAGE   PR-AUC = {avg_prauc:.4f}  MCC = {avg_mcc:.4f}  Δ_best = {avg_delta:+.4f}")
    
    # Wilcoxon vs best base per antibiotic
    cas_pr = df["test_prauc"].values
    best_pr = df["best_base_test_prauc"].values
    if len(cas_pr) >= 5:
        try:
            _, p_w = wilcoxon(cas_pr, best_pr)
            print(f"  Wilcoxon (CascadeStack vs best base, per-AB PR-AUC): p={p_w:.4f}")
        except Exception:
            pass
    
    # Save outputs
    df.to_csv(out_dir / "v7_results.csv", index=False)
    with open(out_dir / "v7_decision_log.json", "w") as f:
        json.dump(decision_log, f, indent=2)
    with open(out_dir / "v7_gating_log.json", "w") as f:
        json.dump(gating_log, f, indent=2)
    with open(out_dir / "v7_base_summary.json", "w") as f:
        json.dump(base_summary, f, indent=2, default=str)
    
    # Markdown summary
    md_lines = [
        f"# CascadeStack v7 — Defense Results",
        f"",
        f"## Average PR-AUC: **{avg_prauc:.4f}**",
        f"",
        f"## Per-antibiotic test results",
        f"",
        f"| AB | PR-AUC | 95% CI | MCC | F2 | Sens@90 | Δ vs best base |",
        f"|----|--------|--------|-----|----|---------|----------------|",
    ]
    for _, r in df.iterrows():
        md_lines.append(
            f"| {r['antibiotic']} | {r['test_prauc']:.4f} | "
            f"[{r['test_prauc_ci_lo']:.3f}, {r['test_prauc_ci_hi']:.3f}] | "
            f"{r['test_mcc']:.4f} | {r['test_f2']:.4f} | {r['test_sens90']:.4f} | "
            f"{r['delta_vs_best_base']:+.4f} |"
        )
    md_lines += [
        f"",
        f"## Comparison to baselines",
        f"- XGBoost gene-aggregated baseline: 0.830 avg PR-AUC",
        f"- **CascadeStack v7: {avg_prauc:.4f} avg PR-AUC**",
        f"- Improvement: {avg_prauc - 0.830:+.4f}",
        f"",
        f"## Configuration",
        f"- Features: gene-aggregated (5000 genes × 4 stats = 20K features)",
        f"- Base learners: {len(base_names)} ({', '.join(base_names)})",
        f"- Meta MLP: snapshot averaging (top-{exp_cfg.meta_mlp.snapshots_k})",
        f"- Threshold: dual-track (MCC vs F2 on val_holdout)",
        f"- Leak-free: 3-way fold split + per-fold GAN retraining",
    ]
    with open(out_dir / "SUMMARY.md", "w") as f:
        f.write("\n".join(md_lines))
    
    print(f"\n  Outputs → {out_dir}")
    print(f"    • v7_results.csv")
    print(f"    • v7_decision_log.json")
    print(f"    • v7_gating_log.json")
    print(f"    • v7_base_summary.json")
    print(f"    • SUMMARY.md")
    print(f"\n{'=' * 80}")
    print("  CascadeStack v7 COMPLETE")
    print(f"{'=' * 80}")