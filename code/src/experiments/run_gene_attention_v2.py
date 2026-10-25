"""
src/experiments/run_gene_attention_v2.py
=========================================
Runner for the GeneAttentionNet v2 ablation (E0-E6).

Entry point (called by src/run_experiment.py dispatcher):
    run(cfg: DictConfig) -> None

Experiment matrix
-----------------
E0: softmax pool, multi-task,  2 transformer layers, 5000 genes  (v1 control)
E1: gated   pool, multi-task,  2 transformer layers, 5000 genes  (H2: gated MT)
E2: entmax  pool, multi-task,  2 transformer layers, 5000 genes  (H2: sparse MT)
E3: softmax pool, single-task, 2 transformer layers, 5000 genes  (H1: single-task)
E4: gated   pool, single-task, 2 transformer layers, 5000 genes  (H1+H2: primary bet)
E5: gated   pool, single-task, 0 transformer layers, 5000 genes  (lean attn-MIL)
E6: entmax  pool, single-task, 0 transformer layers, 1000 genes  (sparse + lean)
"""
from __future__ import annotations

import gc
import logging
import os
import pickle
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.feature_selection import f_classif
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.gene_attention_net_v2 import GeneAttentionNetV2

# ─── Optional dependencies ──────────────────────────────────────────────────

try:
    import wandb as _wandb_mod
    _WANDB = True
except ImportError:
    _WANDB = False

try:
    import entmax as _entmax_mod  # noqa: F401
    _ENTMAX = True
except ImportError:
    _ENTMAX = False

log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────

AB_NAMES = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]

# Reference values (locked from previous sessions — do not change)
V1_PRAUC = dict(zip(AB_NAMES,
    [0.896, 0.913, 0.601, 0.758, 0.498, 0.820, 0.713]))
XGB_PRAUC = dict(zip(AB_NAMES,
    [0.9432, 0.9770, 0.7490, 0.8183, 0.7128, 0.8846, 0.8158]))

# Full experiment matrix (spec Table 1)
EXPERIMENT_MATRIX: List[Dict[str, Any]] = [
    {"id": "E0", "pool_type": "softmax", "multi_task": True,
     "n_layers": 2, "n_genes": 5000, "purpose": "v1 control (reproduce ~0.74)"},
    {"id": "E1", "pool_type": "gated",   "multi_task": True,
     "n_layers": 2, "n_genes": 5000, "purpose": "H2: gated attention, multi-task"},
    {"id": "E2", "pool_type": "entmax",  "multi_task": True,
     "n_layers": 2, "n_genes": 5000, "purpose": "H2: sparse attention, multi-task"},
    {"id": "E3", "pool_type": "softmax", "multi_task": False,
     "n_layers": 2, "n_genes": 5000, "purpose": "H1: single-task per-AB"},
    {"id": "E4", "pool_type": "gated",   "multi_task": False,
     "n_layers": 2, "n_genes": 5000, "purpose": "H1+H2: gated single-task (primary bet)"},
    {"id": "E5", "pool_type": "gated",   "multi_task": False,
     "n_layers": 0, "n_genes": 5000, "purpose": "lean attention-MIL (no transformer)"},
    {"id": "E6", "pool_type": "entmax",  "multi_task": False,
     "n_layers": 0, "n_genes": 1000, "purpose": "sparse + lean + top-1000 genes"},
]


# ─── Reproducibility ────────────────────────────────────────────────────────

def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── AMP compatibility shims ────────────────────────────────────────────────

class _NoOpScaler:
    """CPU / AMP-disabled no-op that mirrors GradScaler's interface."""
    def scale(self, loss):       return loss
    def unscale_(self, opt):     pass
    def step(self, opt):         opt.step()
    def update(self):            pass


def _make_autocast(device_type: str, enabled: bool):
    """Return the appropriate AMP autocast context for PyTorch ≥ 1.8."""
    if not enabled or device_type == "cpu":
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(enabled=True)


def _make_scaler(device_type: str, enabled: bool):
    """Return a GradScaler compatible with PyTorch ≥ 1.8."""
    if not enabled or device_type == "cpu":
        return _NoOpScaler()
    try:
        return torch.amp.GradScaler(device_type)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler()


# ─── Data loading ────────────────────────────────────────────────────────────

def _build_gene_features_inline(
    X_snp:          np.ndarray,
    gene_to_range:  dict,
    selected_genes: list,
) -> np.ndarray:
    """
    Build (N, n_genes, 4) feature tensor from (N, n_snps) int8 SNP matrix.
    Stats per gene: [mean, var, n_nonref, max]
    Called only if the cached .npy file doesn't exist.
    """
    N       = X_snp.shape[0]
    n_genes = len(selected_genes)
    out     = np.zeros((N, n_genes, 4), dtype=np.float32)
    X       = X_snp.astype(np.float32)   # cast once

    for g_idx, gene_id in enumerate(selected_genes):
        start, end = gene_to_range[gene_id]
        if end <= start:
            continue
        cols             = X[:, start:end]          # (N, k)
        out[:, g_idx, 0] = cols.mean(axis=1)        # mean
        out[:, g_idx, 1] = cols.var(axis=1)         # var
        out[:, g_idx, 2] = (cols > 0).sum(axis=1)   # n_nonref (count of non-ref alleles)
        out[:, g_idx, 3] = cols.max(axis=1)         # max

    return out   # (N, n_genes, 4)  float32


def load_data(pkl_path: str, cache_dir: str) -> Dict:
    """
    Load AMR data from pkl and build / load cached (N, n_genes, 4) features.

    Returns
    -------
    dict with keys:
        X_feat        : (N, n_genes, 4) float32  numpy array
        Y             : (N, 7)           float32  numpy array
        label_mask    : (N, 7)           bool     numpy array
        splits        : {'train': [...], 'val': [...], 'test': [...]}
        selected_genes: list of n_genes gene IDs (for documentation)
    """
    pkl_path  = Path(pkl_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Loading data from {pkl_path}")
    with open(pkl_path, "rb") as f:
        raw = pickle.load(f)

    X_snp          = raw["X_snp_filtered"]                  # (N, n_snps) int8
    gene_to_range  = raw["gene_to_range"]                   # {gene_id: (start, end)}
    selected_genes = raw["selected_gene_ids"]               # list[gene_id], len=n_genes
    Y              = raw["Y"].astype(np.float32)            # (N, 7)
    label_mask     = raw["label_mask"].astype(bool)         # (N, 7) bool
    splits         = raw["random_split"]                    # {'train': idx, ...}

    n_genes    = len(selected_genes)
    cache_path = cache_dir / f"gene_features_{n_genes}x4.npy"

    if cache_path.exists():
        X_feat = np.load(str(cache_path))
        expected = (Y.shape[0], n_genes, 4)
        if X_feat.shape != expected:
            log.warning(
                f"Cache shape {X_feat.shape} != expected {expected}, rebuilding"
            )
            cache_path.unlink()
            X_feat = None
        else:
            log.info(f"Loaded cached features: {cache_path}  shape={X_feat.shape}")
    else:
        X_feat = None

    if X_feat is None:
        # Try the existing project feature builder first
        built = False
        try:
            from src.data.gene_aggregated_features import (
                build_gene_feature_matrix_from_pkl,
            )
            log.info("Building features via src.data.gene_aggregated_features …")
            result = build_gene_feature_matrix_from_pkl(
                str(pkl_path), stats=["mean", "var", "n_nonref", "max"]
            )
            X_feat = (
                result.numpy()
                if hasattr(result, "numpy")
                else np.asarray(result, dtype=np.float32)
            )
            built = True
            log.info(f"  Feature builder → shape={X_feat.shape}")
        except Exception as exc:
            log.warning(f"Feature builder unavailable ({exc}); building inline …")

        if not built:
            t0 = time.time()
            log.info(
                f"Building gene features: {n_genes} genes × {X_snp.shape[0]} samples …"
            )
            X_feat = _build_gene_features_inline(X_snp, gene_to_range, selected_genes)
            log.info(f"  Done in {time.time() - t0:.1f}s  shape={X_feat.shape}")

        np.save(str(cache_path), X_feat)
        log.info(f"  Cached to {cache_path}")

    return {
        "X_feat":         X_feat,
        "Y":              Y,
        "label_mask":     label_mask,
        "splits":         splits,
        "selected_genes": selected_genes,
    }


# ─── Feature normalisation ──────────────────────────────────────────────────

def standardize_features(
    X:          np.ndarray,
    train_idx:  np.ndarray,
    val_idx:    np.ndarray,
    test_idx:   np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-stat z-score using TRAIN statistics only (no leakage).
    Computes mean/std over all (sample, gene) pairs for each of the 4 stats.

    X: (N, n_genes, 4) float32
    Returns: X_train_norm, X_val_norm, X_test_norm  (same dtype, same shape slices)
    """
    X_tr  = X[train_idx]                                      # (n_tr, G, 4)
    flat  = X_tr.reshape(-1, X_tr.shape[-1])                  # (n_tr*G, 4)
    mean  = flat.mean(axis=0).astype(np.float32)              # (4,)
    std   = flat.std(axis=0).astype(np.float32).clip(min=1e-8) # (4,)

    log.info(
        f"Feature normalisation — "
        f"mean={mean.round(4).tolist()}  std={std.round(4).tolist()}"
    )

    def _norm(arr: np.ndarray) -> np.ndarray:
        return ((arr - mean) / std).astype(np.float32)

    return _norm(X[train_idx]), _norm(X[val_idx]), _norm(X[test_idx])


# ─── Gene selection for E6 ──────────────────────────────────────────────────

def select_top_k_genes(
    X_train:    np.ndarray,    # (n_tr, n_genes, 4) — standardized
    Y_train:    np.ndarray,    # (n_tr, 7)
    mask_train: np.ndarray,    # (n_tr, 7) bool
    top_k:      int = 1000,
) -> np.ndarray:
    """
    Rank genes by mean ANOVA F-score (f_classif) across all valid ABs.
    Uses stat-0 (mean allele value) as the gene-level feature for ranking.

    Returns sorted array of top_k gene indices (sorted for determinism).
    """
    n_genes     = X_train.shape[1]
    f_sum       = np.zeros(n_genes, dtype=np.float64)
    n_valid_abs = 0

    for ab in range(Y_train.shape[1]):
        valid = mask_train[:, ab]
        if valid.sum() < 20:
            continue
        y_ab = Y_train[valid, ab]
        if len(np.unique(y_ab)) < 2:
            continue
        X_ab = X_train[valid, :, 0]   # mean-allele feature: (n_valid, n_genes)
        try:
            f_vals, _ = f_classif(X_ab, y_ab)
            f_sum    += np.nan_to_num(f_vals, nan=0.0, posinf=0.0, neginf=0.0)
            n_valid_abs += 1
        except Exception as exc:
            log.warning(f"f_classif failed for AB {ab}: {exc}")

    if n_valid_abs == 0:
        log.warning("Gene selection: no valid ABs found; returning first top_k genes")
        return np.arange(top_k, dtype=np.int64)

    avg_f      = f_sum / n_valid_abs
    top_idx    = np.argsort(avg_f)[::-1][:top_k].astype(np.int64)
    return np.sort(top_idx)    # sort for reproducibility


# ─── Training utilities ─────────────────────────────────────────────────────

def compute_pos_weights(
    Y:      np.ndarray,     # (N, 7)
    mask:   np.ndarray,     # (N, 7) bool
    device: torch.device,
) -> torch.Tensor:
    """n_neg / n_pos per AB, computed on training data only (for BCEWithLogits)."""
    pw = []
    for ab in range(Y.shape[1]):
        valid = mask[:, ab]
        y_ab  = Y[valid, ab]
        n_pos = float(y_ab.sum())
        n_neg = float(len(y_ab) - n_pos)
        pw.append(n_neg / max(n_pos, 1.0))
    return torch.tensor(pw, dtype=torch.float32, device=device)


def _masked_multitask_bce(
    logits:      torch.Tensor,   # (B, 7)  may be fp16 inside autocast
    Y:           torch.Tensor,   # (B, 7)  float32
    mask:        torch.Tensor,   # (B, 7)  bool
    pos_weights: torch.Tensor,   # (7,)    float32
) -> torch.Tensor:
    """
    Masked multi-task BCE averaged over ABs that have valid labels in this batch.
    Returns scalar tensor; gradient flows through all contributing ABs.
    """
    losses = []
    for ab in range(Y.shape[1]):
        valid = mask[:, ab]
        if valid.sum() == 0:
            continue
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits[valid, ab],
                Y[valid, ab],
                pos_weight=pos_weights[ab],
            )
        )
    if not losses:
        # Entire batch has no valid label — return zero-grad placeholder
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def _train_one_epoch(
    model:        nn.Module,
    loader:       DataLoader,
    optimizer:    torch.optim.Optimizer,
    scaler,
    autocast_ctx,
    pos_weights:  torch.Tensor,
    device:       torch.device,
    multi_task:   bool,
    ab_idx:       int = 0,
) -> float:
    """
    One training epoch.

    For single-task, all samples in the loader have a valid label for ab_idx.
    The DataLoader yields (X_b, Y_b_full7, M_b_dummy) for API uniformity.

    Returns mean batch loss (float).
    """
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for X_b, Y_b, M_b in loader:
        X_b = X_b.to(device, non_blocking=True)
        Y_b = Y_b.to(device, non_blocking=True)
        M_b = M_b.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx:
            logits = model(X_b)

            if multi_task:
                loss = _masked_multitask_bce(logits, Y_b, M_b, pos_weights)
            else:
                # Single-task: logits (B,1); all samples valid for this AB
                loss = F.binary_cross_entropy_with_logits(
                    logits.squeeze(-1),     # (B,)
                    Y_b[:, ab_idx],         # (B,)  correct label column
                    pos_weight=pos_weights[ab_idx],
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


# ─── Inference utilities ────────────────────────────────────────────────────

@torch.no_grad()
def _collect_predictions(
    model:      nn.Module,
    X_tensor:   torch.Tensor,   # CPU tensor
    device:     torch.device,
    batch_size: int,
    multi_task: bool,
) -> np.ndarray:
    """
    Run model over X_tensor in batches, return sigmoid probabilities.
    Returns: (N, 7) float32 if multi_task else (N, 1) float32
    """
    model.eval()
    parts = []
    for i in range(0, len(X_tensor), batch_size):
        X_b = X_tensor[i: i + batch_size].to(device)
        p   = torch.sigmoid(model(X_b))
        parts.append(p.cpu().float())
    return torch.cat(parts, dim=0).numpy()


@torch.no_grad()
def _collect_attention(
    model:      nn.Module,
    X_tensor:   torch.Tensor,   # CPU tensor
    device:     torch.device,
    batch_size: int,
) -> np.ndarray:
    """
    Collect (B, n_genes) attention weights via model(x, return_attention=True).
    Returns (N, n_genes) float32.  Returns a zero array on any failure.
    """
    model.eval()
    parts = []
    try:
        for i in range(0, len(X_tensor), batch_size):
            X_b = X_tensor[i: i + batch_size].to(device)
            _, a = model(X_b, return_attention=True)
            parts.append(a.cpu().float())
        return torch.cat(parts, dim=0).numpy()
    except Exception as exc:
        log.warning(f"Attention collection failed ({exc}); returning zeros")
        return np.zeros((len(X_tensor), model.n_genes), dtype=np.float32)


def _evaluate_prauc(
    model:      nn.Module,
    X_tensor:   torch.Tensor,
    Y_np:       np.ndarray,
    mask_np:    np.ndarray,
    device:     torch.device,
    batch_size: int,
    multi_task: bool,
    ab_idx:     int = 0,
) -> Dict[str, float]:
    """
    Compute PR-AUC per AB.  For single-task models returns NaN for all ABs
    except ab_idx (we only have a meaningful predictor for that AB).
    """
    preds = _collect_predictions(model, X_tensor, device, batch_size, multi_task)
    # preds: (N, 7) or (N, 1)

    result: Dict[str, float] = {}
    for ab, ab_name in enumerate(AB_NAMES):
        if not multi_task and ab != ab_idx:
            result[ab_name] = np.nan
            continue

        col   = ab if multi_task else 0
        valid = mask_np[:, ab]
        if valid.sum() < 2:
            result[ab_name] = np.nan
            continue

        y_true  = Y_np[valid, ab]
        y_score = preds[valid, col]

        if y_true.sum() == 0 or y_true.sum() == valid.sum():
            result[ab_name] = np.nan
            continue

        result[ab_name] = float(average_precision_score(y_true, y_score))

    return result


# ─── Bootstrap CI ───────────────────────────────────────────────────────────

def _bootstrap_ci(
    y_true:      np.ndarray,
    y_score:     np.ndarray,
    n_bootstrap: int   = 1000,
    seed:        int   = 42,
    alpha:       float = 0.05,
) -> Tuple[float, float, float]:
    """
    Bootstrap 95% CI for PR-AUC.
    Returns (point_estimate, lower_ci, upper_ci).
    Returns (nan, nan, nan) for degenerate label distributions.
    """
    y_true  = np.asarray(y_true,  dtype=np.float32)
    y_score = np.asarray(y_score, dtype=np.float32)

    n = len(y_true)
    if n < 2 or y_true.sum() == 0 or y_true.sum() == n:
        return np.nan, np.nan, np.nan

    point = float(average_precision_score(y_true, y_score))
    rng   = np.random.default_rng(seed)
    boot  = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        try:
            boot.append(float(average_precision_score(yt, ys)))
        except Exception:
            continue

    if len(boot) < 50:    # not enough valid bootstrap iterations
        return point, np.nan, np.nan

    arr = np.array(boot)
    lo  = float(np.percentile(arr, 100.0 * alpha / 2))
    hi  = float(np.percentile(arr, 100.0 * (1.0 - alpha / 2)))
    return point, lo, hi


# ─── Core training function ─────────────────────────────────────────────────

def train_variant(
    exp:       Dict[str, Any],
    X_tr:      np.ndarray,     # (n_tr, n_genes, 4) — already standardized
    X_va:      np.ndarray,
    X_te:      np.ndarray,
    Y_tr:      np.ndarray,     # (n_tr, 7) float32
    Y_va:      np.ndarray,
    Y_te:      np.ndarray,
    mask_tr:   np.ndarray,     # (n_tr, 7) bool
    mask_va:   np.ndarray,
    mask_te:   np.ndarray,
    cfg:       DictConfig,
    device:    torch.device,
    wb_logger,                 # wandb module or None
) -> Dict:
    """
    Train one experiment variant (multi-task OR single-task AB loop).

    Returns a results dict with per-AB PR-AUC, 95% CIs, and attention weights.
    """
    exp_id     = exp["id"]
    pool_type  = exp["pool_type"]
    multi_task = exp["multi_task"]
    n_layers   = exp["n_layers"]
    n_genes    = exp["n_genes"]

    # Build CPU tensors once; move to GPU per-batch inside DataLoader
    Xtr_t = torch.tensor(X_tr, dtype=torch.float32)
    Xva_t = torch.tensor(X_va, dtype=torch.float32)
    Xte_t = torch.tensor(X_te, dtype=torch.float32)
    Ytr_t = torch.tensor(Y_tr, dtype=torch.float32)
    Mtr_t = torch.tensor(mask_tr, dtype=torch.bool)

    pos_weights  = compute_pos_weights(Y_tr, mask_tr, device)
    autocast_ctx = _make_autocast(device.type, cfg.use_amp)

    results_per_ab:  Dict[str, Dict] = {}
    attn_weights_te: Dict[str, np.ndarray] = {}

    # ── MULTI-TASK path ──────────────────────────────────────────────────────
    if multi_task:
        model = GeneAttentionNetV2(
            n_genes=n_genes, n_stats=4, hidden_dim=cfg.hidden_dim,
            n_heads=cfg.n_heads, n_layers=n_layers, ffn_mult=cfg.ffn_mult,
            dropout=cfg.dropout, pool_type=pool_type, attn_dim=cfg.attn_dim,
            multi_task=True, n_antibiotics=len(AB_NAMES),
        ).to(device)
        log.info(f"[{exp_id}] Multi-task model | pool={pool_type} | "
                 f"params={model.count_parameters():,}")

        train_loader = DataLoader(
            TensorDataset(Xtr_t, Ytr_t, Mtr_t),
            batch_size=cfg.batch_size, shuffle=True,
            num_workers=0, pin_memory=(device.type == "cuda"),
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        scaler = _make_scaler(device.type, cfg.use_amp)

        # Initialise best_state to model weights at epoch 0
        best_val     = -np.inf
        best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
        patience_ctr = 0

        for epoch in range(cfg.max_epochs):
            tr_loss = _train_one_epoch(
                model, train_loader, optimizer, scaler, autocast_ctx,
                pos_weights, device, multi_task=True,
            )

            if np.isnan(tr_loss):
                log.error(f"[{exp_id}] NaN training loss at epoch {epoch+1} — stopping")
                break

            val_dict  = _evaluate_prauc(
                model, Xva_t, Y_va, mask_va, device,
                cfg.batch_size, multi_task=True,
            )
            val_prauc = float(np.nanmean(list(val_dict.values())))
            log.info(
                f"[{exp_id}] ep {epoch+1:2d}/{cfg.max_epochs}  "
                f"loss={tr_loss:.4f}  val_avg={val_prauc:.4f}"
            )
            if wb_logger:
                wb_logger.log({
                    f"{exp_id}/epoch":        epoch + 1,
                    f"{exp_id}/train_loss":   tr_loss,
                    f"{exp_id}/val_prauc_avg": val_prauc,
                })

            if val_prauc > best_val + 1e-5:
                best_val     = val_prauc
                best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= cfg.patience:
                    log.info(f"[{exp_id}] Early stop at epoch {epoch+1}")
                    break

        # Load best checkpoint
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

        # Test evaluation
        preds_te = _collect_predictions(
            model, Xte_t, device, cfg.batch_size, multi_task=True
        )   # (N_te, 7)
        attn_all = _collect_attention(model, Xte_t, device, cfg.batch_size)

        for ab_idx, ab_name in enumerate(AB_NAMES):
            valid = mask_te[:, ab_idx]
            if valid.sum() < 2:
                results_per_ab[ab_name] = {
                    "prauc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan
                }
                continue
            y_true  = Y_te[valid, ab_idx]
            y_score = preds_te[valid, ab_idx]
            pt, lo, hi = _bootstrap_ci(
                y_true, y_score, cfg.n_bootstrap, cfg.bootstrap_seed
            )
            results_per_ab[ab_name]  = {"prauc": pt, "ci_lo": lo, "ci_hi": hi}
            attn_weights_te[ab_name] = attn_all   # shared pool → same attn for all ABs

        del model, train_loader
        torch.cuda.empty_cache()
        gc.collect()

    # ── SINGLE-TASK path (one model per AB, independent training loops) ─────
    else:
        for ab_idx, ab_name in enumerate(AB_NAMES):
            log.info(f"\n[{exp_id}/{ab_name}]")

            valid_tr = mask_tr[:, ab_idx]
            valid_va = mask_va[:, ab_idx]
            valid_te = mask_te[:, ab_idx]

            if valid_tr.sum() < 10:
                log.warning(
                    f"  Only {valid_tr.sum()} valid train samples — skipping {ab_name}"
                )
                results_per_ab[ab_name] = {
                    "prauc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan
                }
                continue

            log.info(
                f"  n_train={valid_tr.sum()}  n_val={valid_va.sum()}  "
                f"n_test={valid_te.sum()}"
            )

            # Filter training data to valid-label samples for this AB
            Xtr_ab       = Xtr_t[valid_tr]          # (n_tr_ab, n_genes, 4)
            Ytr_ab_full  = Ytr_t[valid_tr]          # (n_tr_ab, 7) — keep all 7 cols
            # Dummy all-True mask (not used in single-task loss, just for API)
            Mtr_ab = torch.ones(Xtr_ab.shape[0], len(AB_NAMES), dtype=torch.bool)

            model = GeneAttentionNetV2(
                n_genes=n_genes, n_stats=4, hidden_dim=cfg.hidden_dim,
                n_heads=cfg.n_heads, n_layers=n_layers, ffn_mult=cfg.ffn_mult,
                dropout=cfg.dropout, pool_type=pool_type, attn_dim=cfg.attn_dim,
                multi_task=False, n_antibiotics=1,
            ).to(device)

            if ab_idx == 0:
                log.info(f"  Single-task model: {model.count_parameters():,} params")

            train_loader = DataLoader(
                TensorDataset(Xtr_ab, Ytr_ab_full, Mtr_ab),
                batch_size=cfg.batch_size, shuffle=True,
                num_workers=0, pin_memory=(device.type == "cuda"),
            )
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
            )
            scaler = _make_scaler(device.type, cfg.use_amp)

            best_val     = -np.inf
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0

            # Val mask scoped to this AB (only its valid samples count)
            mask_va_ab             = np.zeros_like(mask_va)
            mask_va_ab[:, ab_idx]  = mask_va[:, ab_idx]

            for epoch in range(cfg.max_epochs):
                tr_loss = _train_one_epoch(
                    model, train_loader, optimizer, scaler, autocast_ctx,
                    pos_weights, device, multi_task=False, ab_idx=ab_idx,
                )

                if np.isnan(tr_loss):
                    log.error(
                        f"  [{ab_name}] NaN loss at epoch {epoch+1} — stopping"
                    )
                    break

                val_dict  = _evaluate_prauc(
                    model, Xva_t, Y_va, mask_va_ab, device,
                    cfg.batch_size, multi_task=False, ab_idx=ab_idx,
                )
                val_prauc = val_dict.get(ab_name, np.nan)
                if np.isnan(val_prauc):
                    val_prauc = -np.inf

                if epoch % 5 == 0 or epoch < 2:
                    log.info(
                        f"  ep {epoch+1:2d}/{cfg.max_epochs}  "
                        f"loss={tr_loss:.4f}  val_{ab_name}={val_prauc:.4f}"
                    )

                if val_prauc > best_val + 1e-5:
                    best_val     = val_prauc
                    best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= cfg.patience:
                        log.info(f"  Early stop at epoch {epoch+1}")
                        break

            # Load best checkpoint
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            model.eval()

            # Test evaluation: run on FULL test set, filter by mask for metrics
            preds_te = _collect_predictions(
                model, Xte_t, device, cfg.batch_size, multi_task=False
            )   # (N_te, 1)

            if valid_te.sum() < 2:
                results_per_ab[ab_name] = {
                    "prauc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan
                }
            else:
                y_true  = Y_te[valid_te, ab_idx]
                y_score = preds_te[valid_te, 0]
                pt, lo, hi = _bootstrap_ci(
                    y_true, y_score, cfg.n_bootstrap, cfg.bootstrap_seed
                )
                results_per_ab[ab_name] = {"prauc": pt, "ci_lo": lo, "ci_hi": hi}

            # Attention for valid test samples only (used for sparsity analysis)
            Xte_ab  = Xte_t[valid_te]
            attn_ab = _collect_attention(model, Xte_ab, device, cfg.batch_size)
            attn_weights_te[ab_name] = attn_ab   # (n_te_valid, n_genes)

            if wb_logger:
                wb_logger.log({
                    f"{exp_id}/{ab_name}/test_prauc":
                        results_per_ab[ab_name]["prauc"]
                })

            del model, train_loader
            torch.cuda.empty_cache()
            gc.collect()

    avg_prauc = float(
        np.nanmean([v["prauc"] for v in results_per_ab.values()])
    )
    log.info(
        f"\n[{exp_id}] DONE  avg_test_prauc={avg_prauc:.4f}  "
        + "  ".join(
            f"{ab}={results_per_ab[ab]['prauc']:.3f}"
            for ab in AB_NAMES
        )
    )

    return {
        "exp_id":     exp_id,
        "pool_type":  pool_type,
        "multi_task": multi_task,
        "n_layers":   n_layers,
        "n_genes":    n_genes,
        "purpose":    exp["purpose"],
        "per_ab":     results_per_ab,
        "avg_prauc":  avg_prauc,
        "attn":       attn_weights_te,
    }


# ─── Experiment sweep ────────────────────────────────────────────────────────

def run_all_variants(cfg: DictConfig, data: Dict) -> List[Dict]:
    """Sweep the full E0–E6 experiment matrix."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    splits    = data["splits"]
    train_idx = np.array(splits.get("train") or splits.get("train_idx"))
    val_idx   = np.array(splits.get("val")   or splits.get("val_idx"))
    test_idx  = np.array(splits.get("test")  or splits.get("test_idx"))
    log.info(
        f"Split sizes — train:{len(train_idx)}  "
        f"val:{len(val_idx)}  test:{len(test_idx)}"
    )

    X_feat     = data["X_feat"]
    Y          = data["Y"]
    label_mask = data["label_mask"]

    # Standardise using full 5000-gene train statistics
    X_tr, X_va, X_te = standardize_features(X_feat, train_idx, val_idx, test_idx)
    Y_tr,  Y_va,  Y_te         = Y[train_idx],          Y[val_idx],          Y[test_idx]
    mask_tr, mask_va, mask_te  = (
        label_mask[train_idx],
        label_mask[val_idx],
        label_mask[test_idx],
    )

    # Pre-compute top-1000 gene indices for E6 (once, from train split)
    log.info("Pre-computing top-1000 gene indices for E6 (ANOVA F-score on train) …")
    top1000_idx = select_top_k_genes(X_tr, Y_tr, mask_tr, top_k=1000)
    log.info(
        f"  top-1000 gene index range: [{top1000_idx.min()}, {top1000_idx.max()}]  "
        f"first-10: {top1000_idx[:10].tolist()}"
    )

    # WandB (optional, graceful fallback)
    wb_logger = None
    if cfg.get("use_wandb", False):
        if _WANDB:
            import wandb
            wandb.init(
                project=cfg.get("wandb_project", "amr_gan_v2"),
                entity=cfg.get("wandb_entity", None) or None,
                config=OmegaConf.to_container(cfg, resolve=True),
                reinit=True,
            )
            wb_logger = wandb
            log.info("WandB run initialised")
        else:
            log.warning("use_wandb=True but wandb not installed; skipping WandB")

    # Determine which experiments to run (allow partial runs for debugging)
    try:
        exps_to_run = set(list(cfg.experiments_to_run))
    except Exception:
        exps_to_run = set(e["id"] for e in EXPERIMENT_MATRIX)
    log.info(f"Experiments to run: {sorted(exps_to_run)}")

    all_results: List[Dict] = []

    for exp in EXPERIMENT_MATRIX:
        if exp["id"] not in exps_to_run:
            log.info(f"Skipping {exp['id']} (not in experiments_to_run)")
            continue

        # Guard: entmax/sparsemax require the entmax package
        if exp["pool_type"] in ("entmax", "sparsemax") and not _ENTMAX:
            log.error(
                f"[{exp['id']}] SKIPPED — pool_type={exp['pool_type']} requires "
                f"`pip install entmax`"
            )
            all_results.append({
                "exp_id":     exp["id"],
                "pool_type":  exp["pool_type"],
                "multi_task": exp["multi_task"],
                "n_layers":   exp["n_layers"],
                "n_genes":    exp["n_genes"],
                "purpose":    exp["purpose"],
                "per_ab":     {ab: {"prauc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
                               for ab in AB_NAMES},
                "avg_prauc":  np.nan,
                "attn":       {},
                "error":      "entmax not installed (pip install entmax)",
            })
            continue

        # Gene subset: E6 uses top-1000; all others use all 5000
        if exp["n_genes"] < X_tr.shape[1]:
            # Slice AFTER standardisation: per-stat z-score is global, not per-gene,
            # so slicing a subset of genes does not invalidate the normalisation.
            X_tr_e = X_tr[:, top1000_idx, :]
            X_va_e = X_va[:, top1000_idx, :]
            X_te_e = X_te[:, top1000_idx, :]
        else:
            X_tr_e, X_va_e, X_te_e = X_tr, X_va, X_te

        # Re-seed identically before each experiment for fair comparison
        set_seeds(cfg.get("seed", 42))

        t_start = time.time()
        log.info(f"\n{'='*66}\n[{exp['id']}]  {exp['purpose']}\n{'='*66}")

        try:
            result = train_variant(
                exp, X_tr_e, X_va_e, X_te_e,
                Y_tr, Y_va, Y_te,
                mask_tr, mask_va, mask_te,
                cfg, device, wb_logger,
            )
            result["runtime_min"] = round((time.time() - t_start) / 60, 2)

        except Exception as exc:
            log.exception(f"[{exp['id']}] FAILED with exception: {exc}")
            result = {
                "exp_id":     exp["id"],
                "pool_type":  exp["pool_type"],
                "multi_task": exp["multi_task"],
                "n_layers":   exp["n_layers"],
                "n_genes":    exp["n_genes"],
                "purpose":    exp["purpose"],
                "per_ab":     {ab: {"prauc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
                               for ab in AB_NAMES},
                "avg_prauc":  np.nan,
                "attn":       {},
                "error":      str(exc),
            }

        all_results.append(result)
        torch.cuda.empty_cache()
        gc.collect()

    if wb_logger and _WANDB:
        import wandb
        wandb.finish()

    return all_results


# ─── Output saving ──────────────────────────────────────────────────────────

def _compute_sparsity(attn: np.ndarray) -> Dict:
    """
    Compute sparsity metrics from (N_samples, n_genes) attention matrix.
    Averages over test samples to get a per-gene mean attention weight.
    """
    avg = np.maximum(attn.mean(axis=0), 0.0)    # (n_genes,)
    tot = avg.sum()
    if tot == 0:
        return {
            "n_above_1pct": 0, "n_for_90pct": 0,
            "max_attn": 0.0, "entropy": 0.0,
            "top20_idx": np.arange(min(20, len(avg))),
        }

    avg_norm = avg / tot                          # sum-normalised
    sorted_d = np.sort(avg_norm)[::-1]
    cumsum   = np.cumsum(sorted_d)
    n_for_90 = int(np.searchsorted(cumsum, 0.9) + 1)

    return {
        "n_above_1pct": int((avg_norm > 0.01).sum()),
        "n_for_90pct":  n_for_90,
        "max_attn":     float(avg_norm.max()),
        "entropy":      float(-np.sum(avg_norm * np.log(avg_norm + 1e-12))),
        "top20_idx":    np.argsort(avg_norm)[::-1][:20].astype(int),
    }


def save_results(all_results: List[Dict], output_dir: Path) -> None:
    """Write all output CSVs and the best-variant attention .npz."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. v2_variant_matrix.csv — all E0-E6 × per-AB PR-AUC + CI
    rows = []
    for r in all_results:
        for ab_name in AB_NAMES:
            ab_r = r.get("per_ab", {}).get(ab_name, {})
            rows.append({
                "exp_id":     r["exp_id"],
                "pool_type":  r.get("pool_type",  ""),
                "multi_task": r.get("multi_task", ""),
                "n_layers":   r.get("n_layers",   ""),
                "n_genes":    r.get("n_genes",    ""),
                "purpose":    r.get("purpose",    ""),
                "ab":         ab_name,
                "prauc":      ab_r.get("prauc", np.nan),
                "ci_lo":      ab_r.get("ci_lo", np.nan),
                "ci_hi":      ab_r.get("ci_hi", np.nan),
                "runtime_min": r.get("runtime_min", np.nan),
                "error":      r.get("error", ""),
            })
    pd.DataFrame(rows).to_csv(output_dir / "v2_variant_matrix.csv", index=False)
    log.info("Saved v2_variant_matrix.csv")

    # Identify best valid result
    valid_results = [
        r for r in all_results
        if "error" not in r and not np.isnan(r.get("avg_prauc", np.nan))
    ]

    if not valid_results:
        log.warning("No valid results to write comparison CSV or attention npz")
        return

    best = max(valid_results, key=lambda r: r["avg_prauc"])
    log.info(f"Best variant: {best['exp_id']}  avg_prauc={best['avg_prauc']:.4f}")

    # 2. v2_vs_v1_vs_xgb.csv — head-to-head comparison with reference models
    cmp_rows = []
    for ab_name in AB_NAMES:
        ab_r     = best["per_ab"].get(ab_name, {})
        v2_pr    = ab_r.get("prauc", np.nan)
        v1_pr    = V1_PRAUC[ab_name]
        xgb_pr   = XGB_PRAUC[ab_name]
        cmp_rows.append({
            "ab":           ab_name,
            "best_exp_id":  best["exp_id"],
            "v2_prauc":     v2_pr,
            "v2_ci_lo":     ab_r.get("ci_lo", np.nan),
            "v2_ci_hi":     ab_r.get("ci_hi", np.nan),
            "v1_prauc":     v1_pr,
            "xgb_prauc":    xgb_pr,
            "delta_vs_v1":  (v2_pr - v1_pr)   if not np.isnan(v2_pr) else np.nan,
            "delta_vs_xgb": (v2_pr - xgb_pr)  if not np.isnan(v2_pr) else np.nan,
            "beats_xgb":    bool(not np.isnan(v2_pr) and v2_pr >= xgb_pr),
        })
    pd.DataFrame(cmp_rows).to_csv(output_dir / "v2_vs_v1_vs_xgb.csv", index=False)
    log.info("Saved v2_vs_v1_vs_xgb.csv")

    # 3. attention_sparsity.csv — per variant / per AB sparsity metrics
    sp_rows = []
    for r in all_results:
        if "error" in r:
            continue
        for ab_name, attn_arr in r.get("attn", {}).items():
            if attn_arr is None or attn_arr.size == 0:
                continue
            sp = _compute_sparsity(attn_arr)
            sp_rows.append({
                "exp_id":             r["exp_id"],
                "pool_type":          r.get("pool_type", ""),
                "multi_task":         r.get("multi_task", ""),
                "n_genes":            r.get("n_genes", ""),
                "ab":                 ab_name,
                "n_genes_above_1pct": sp["n_above_1pct"],
                "n_genes_for_90pct":  sp["n_for_90pct"],
                "max_attn":           round(sp["max_attn"], 7),
                "entropy":            round(sp["entropy"], 4),
                "top20_gene_indices": sp["top20_idx"].tolist(),
            })
    if sp_rows:
        pd.DataFrame(sp_rows).to_csv(
            output_dir / "attention_sparsity.csv", index=False
        )
        log.info("Saved attention_sparsity.csv")

    # 4. best_variant_attention.npz — attention weights of winning variant
    attn_save = {
        ab: arr.astype(np.float32)
        for ab, arr in best.get("attn", {}).items()
        if arr is not None and arr.size > 0
    }
    if attn_save:
        np.savez_compressed(
            str(output_dir / "best_variant_attention.npz"), **attn_save
        )
        log.info(
            f"Saved best_variant_attention.npz  "
            f"(variant={best['exp_id']}, ABs={list(attn_save.keys())})"
        )


def write_summary(all_results: List[Dict], output_dir: Path) -> None:
    """Write SUMMARY.md — hypothesis verdict + recommendation."""
    valid = [
        r for r in all_results
        if "error" not in r and not np.isnan(r.get("avg_prauc", np.nan))
    ]

    if not valid:
        text = (
            "# GeneAttentionNetV2 — SUMMARY\n\n"
            "All experiments failed or were skipped. "
            "Check logs for details.\n"
        )
        (output_dir / "SUMMARY.md").write_text(text, encoding="utf-8")
        return

    best = max(valid, key=lambda r: r["avg_prauc"])

    # Evaluate success criteria
    arch_win      = best["avg_prauc"] > 0.74         # criterion 1
    n_beats_xgb   = 0
    beats_xgb_abs = []
    for ab in AB_NAMES:
        v2_pr = best["per_ab"].get(ab, {}).get("prauc", np.nan)
        if not np.isnan(v2_pr) and v2_pr >= XGB_PRAUC[ab]:
            n_beats_xgb += 1
            beats_xgb_abs.append(ab)
    xgb_win = n_beats_xgb >= 2                       # criterion 2

    # Sparsity check (criterion 3)
    sparse_win    = False
    sparsity_note = "Attention data not available."
    for ab_name, attn_arr in best.get("attn", {}).items():
        if attn_arr is not None and attn_arr.size > 0:
            sp = _compute_sparsity(attn_arr)
            sparse_win = sp["n_for_90pct"] < 50
            sparsity_note = (
                f"{best['exp_id']} ({ab_name}): {sp['n_for_90pct']} genes carry "
                f"90% of attention weight  "
                f"(n_above_1pct={sp['n_above_1pct']}, "
                f"entropy={sp['entropy']:.2f}, max_w={sp['max_attn']:.5f})"
            )
            break

    lines = [
        "# GeneAttentionNetV2 — Experiment Summary",
        "",
        f"**Best variant:** `{best['exp_id']}` — "
        f"pool=`{best['pool_type']}`, "
        f"multi_task={best['multi_task']}, "
        f"n_layers={best['n_layers']}, "
        f"n_genes={best['n_genes']}",
        "",
        "---",
        "",
        "## Results Table (sorted by avg PR-AUC)",
        "",
        "| Exp | Pool | MT | L | n_genes | Avg PR-AUC | Runtime |",
        "|-----|------|----|---|---------|------------|---------|",
    ]
    for r in sorted(valid, key=lambda x: x.get("avg_prauc", 0), reverse=True):
        marker = "  ← **best**" if r["exp_id"] == best["exp_id"] else ""
        rt     = f"{r.get('runtime_min', 0):.1f} min" if r.get("runtime_min") else "—"
        lines.append(
            f"| {r['exp_id']} | {r['pool_type']} "
            f"| {r['multi_task']} | {r['n_layers']} | {r['n_genes']} "
            f"| **{r['avg_prauc']:.4f}**{marker} | {rt} |"
        )
    for r in all_results:
        if "error" in r:
            lines.append(
                f"| {r['exp_id']} | {r.get('pool_type','')} | — | — | — "
                f"| FAILED | — |"
            )

    lines += [
        "",
        "## Per-AB: Best v2 vs v1 vs XGB",
        "",
        "| AB | v2 PR-AUC | 95% CI | v1 | XGB | Δ vs v1 | Δ vs XGB |",
        "|----|-----------|--------|----|-----|---------|----------|",
    ]
    for ab in AB_NAMES:
        ab_r = best["per_ab"].get(ab, {})
        v2   = ab_r.get("prauc", np.nan)
        lo   = ab_r.get("ci_lo", np.nan)
        hi   = ab_r.get("ci_hi", np.nan)
        v1   = V1_PRAUC[ab]
        xgb  = XGB_PRAUC[ab]
        ci_s = (
            f"[{lo:.4f},{hi:.4f}]"
            if not (np.isnan(lo) or np.isnan(hi))
            else "—"
        )
        d1 = f"{v2 - v1:+.4f}"  if not np.isnan(v2) else "n/a"
        d2 = f"{v2 - xgb:+.4f}" if not np.isnan(v2) else "n/a"
        v2_s = f"{v2:.4f}" if not np.isnan(v2) else "n/a"
        lines.append(
            f"| {ab} | {v2_s} | {ci_s} | {v1:.4f} | {xgb:.4f} | {d1} | {d2} |"
        )

    lines += [
        "",
        "## Success Criteria",
        "",
        f"1. **Architecture win** (avg v2 > 0.74 = v1 standalone): "
        f"{'✅ YES' if arch_win else '❌ NO'}  "
        f"(best avg = {best['avg_prauc']:.4f})",
        f"2. **XGB competitiveness** (v2 ≥ XGB on ≥ 2 ABs): "
        f"{'✅ YES' if xgb_win else '❌ NO'}  "
        f"({n_beats_xgb}/7 ABs"
        + (f": {', '.join(beats_xgb_abs)}" if beats_xgb_abs else "")
        + ")",
        f"3. **Interpretability** (< 50 genes carry 90% weight): "
        f"{'✅ YES' if sparse_win else '❌ NO'}  "
        f"— {sparsity_note}",
        "",
        "## Hypothesis Assessment",
        "",
    ]

    # Honest verdict based on combination of criteria
    if arch_win and xgb_win and sparse_win:
        verdict = (
            "**Strong success.** The v2 architecture (single-task + gated/sparse "
            "attention) achieves meaningful accuracy gains over v1 standalone, "
            "matches or exceeds XGBoost on multiple antibiotics, AND produces "
            "genuinely sparse attention weights.  The neural component now earns "
            "its place in both the ensemble architecture story (Pillar 2) and the "
            "gene interpretability story (Pillar 3)."
        )
    elif arch_win and (xgb_win or sparse_win):
        verdict = (
            "**Partial success.** v2 clearly improves over v1 standalone, confirming "
            "at least one hypothesis (H1: single-task reduces negative transfer; "
            "H2: gated/sparse attention improves gene selection).  The XGB gap "
            "narrows but does not fully close.  "
            "**Recommendation:** retain GAN-v2 as a base learner in the ensemble "
            "for diversity; use the sparse attention weights for Pillar 3 gene "
            "importance analysis in the paper."
        )
    elif arch_win:
        verdict = (
            "**Modest improvement.** v2 beats v1 standalone but does not match "
            "XGBoost or produce clearly sparse attention.  "
            "**Recommendation:** frame the neural component as an ensemble "
            "base-learner for signal diversity and as an interpretability tool, "
            "not as the primary standalone predictor.  Focus the paper narrative "
            "on the stacking architecture and SHAP analysis."
        )
    else:
        verdict = (
            "**Negative / null result.** No v2 variant surpasses the v1 standalone "
            "baseline of 0.74 avg PR-AUC.  Neural architectures do not outperform "
            "gradient-boosted trees on this gene-aggregated tabular representation.  \n\n"
            "**This is a publishable, intellectually honest finding**, not a failure.  "
            "It tells us that gene-level summary statistics naturally favour tree-based "
            "models on this dataset size (~5K samples), and motivates future work on "
            "raw-sequence encoders or graph neural networks.  "
            "Document clearly in §4.3 (Limitations and Future Work) and reframe the "
            "neural component as an interpretability tool / auxiliary base learner."
        )

    lines.append(verdict)
    lines += [
        "",
        "## Recommendation for v9.2 Ensemble",
        "",
        f"- **Best v2 variant:** `{best['exp_id']}` (avg PR-AUC = "
        f"{best['avg_prauc']:.4f}).  Replace v1 GAN only if this avg exceeds 0.78.",
        f"- **If H1 confirmed** (single-task wins): update CascadeStack to load "
        "7 per-AB GAN-v2 models at inference time.",
        f"- **If H2 confirmed** (gated/sparse pool wins): feed attention weights "
        "from `best_variant_attention.npz` into Pillar 3 gene importance ranking.",
        f"- **If negative overall:** keep v1 GAN as base learner; document the "
        "negative result; focus paper on ensemble + SHAP.",
        "",
        "---",
        f"*Generated automatically by run_gene_attention_v2.py — "
        f"{len(valid)}/{len(all_results)} experiments succeeded*",
    ]

    (output_dir / "SUMMARY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    log.info("Saved SUMMARY.md")


# ─── Entry point ────────────────────────────────────────────────────────────

def run(cfg: DictConfig) -> None:
    if OmegaConf.select(cfg, "data_pkl") is None:
        try:
            cfg = cfg.experiment
        except Exception:
            pass
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    """
    Hydra entry point.  Called by src/run_experiment.py dispatch.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    set_seeds(cfg.get("seed", 42))

    output_dir = Path(OmegaConf.select(cfg, "output_dir", default="results/gene_attention_v2"))
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {output_dir.resolve()}")

    # Persist the config for full reproducibility
    try:
        (output_dir / "config_used.yaml").write_text(
            OmegaConf.to_yaml(cfg), encoding="utf-8"
        )
    except Exception:
        pass

    # Data loading
    cache_dir = str(output_dir / "cache")
    data = load_data(cfg.data_pkl, cache_dir)

    for split_name, idx in data["splits"].items():
        valid_counts = data["label_mask"][idx].sum(axis=0).astype(int)
        log.info(
            f"  {split_name:5s}: n={len(idx)}  "
            f"valid per AB = {dict(zip(AB_NAMES, valid_counts.tolist()))}"
        )

    # Run all experiments
    all_results = run_all_variants(cfg, data)

    # Save outputs
    save_results(all_results,  output_dir)
    write_summary(all_results, output_dir)

    # Console summary
    log.info("\n── FINAL RESULTS ─────────────────────────────────────────────────────")
    for r in sorted(
        all_results, key=lambda x: x.get("avg_prauc", -np.inf), reverse=True
    ):
        if "error" in r:
            status = f"ERROR({r['error'][:60]})"
        else:
            status = f"avg={r['avg_prauc']:.4f}"
        log.info(
            f"  {r['exp_id']:3s}  pool={r.get('pool_type',''):<8}  "
            f"mt={str(r.get('multi_task','')):<5}  "
            f"L={r.get('n_layers','')}  G={r.get('n_genes','')}  {status}"
        )
    log.info("──────────────────────────────────────────────────────────────────────")
