"""
src/experiments/run_trilens_v10.py
====================================
TriLens-MoE-Stack v10 — headline neural ensemble runner.

Entry point: run(cfg: DictConfig)
Dispatch:    "trilens_v10": "src.experiments.run_trilens_v10"

Ablation matrix:
  T1  Expert A only  (within-gene motif CNN)
  T2  Expert B only  (P/A sparse-attention) — critical GEN diagnostic
  T3  Expert C only  (structured RMRNet)
  T4  All 3 experts, uniform average fusion
  T5  All 3 experts, MoE gate (full TriLens standalone)
  T6  T5 + 5 classical OOF bases + LogReg meta  ← HEADLINE
  T7  T6 re-run on phylogeny_split  ← reviewer-proof
"""
from __future__ import annotations

import gc
import json
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from src.models.trilens import AB_NAMES, TriLens

log = logging.getLogger(__name__)

# ─── Ablation definitions ────────────────────────────────────────────────────

ABLATION_DEFS: Dict[str, Dict[str, Any]] = {
    "T1": {"use_a": True,  "use_b": False, "use_c": False, "fusion": "single",
           "desc": "Expert A only (within-gene motif CNN)"},
    "T2": {"use_a": False, "use_b": True,  "use_c": False, "fusion": "single",
           "desc": "Expert B only (P/A sparse-attn) — P/A lever + GEN diagnostic"},
    "T3": {"use_a": False, "use_b": False, "use_c": True,  "fusion": "single",
           "desc": "Expert C only (structured RMRNet cross-AB)"},
    "T4": {"use_a": True,  "use_b": True,  "use_c": True,  "fusion": "uniform",
           "desc": "All 3 experts, uniform average (no gating)"},
    "T5": {"use_a": True,  "use_b": True,  "use_c": True,  "fusion": "moe",
           "desc": "All 3 experts, MoE gate (full TriLens standalone)"},
    "T6": {"use_a": True,  "use_b": True,  "use_c": True,  "fusion": "moe",
           "desc": "v10 stack: K-fold OOF + 5 classical bases + LogReg meta"},
    "T7": {"use_a": True,  "use_b": True,  "use_c": True,  "fusion": "moe",
           "desc": "v10 stack on phylogeny_split (leakage-corrected)"},
}

# v9.1 reference for comparison
V91_AVG  = 0.8469
V91_PER_AB = {"AMP": 0.9412, "CIP": 0.9774, "AMC": 0.7569, "SXT": 0.8346,
              "GEN": 0.7037, "CTX": 0.8931, "CAZ": 0.8215}


# ─── Utility ──────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _NoOpScaler:
    def scale(self, loss):   return loss
    def unscale_(self, opt): pass
    def step(self, opt):     opt.step()
    def update(self):        pass


def _make_autocast(device_type: str, enabled: bool):
    if not enabled or device_type == "cpu":
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    except (TypeError, AttributeError):
        return torch.cuda.amp.autocast(enabled=True)


def _make_scaler(device_type: str, enabled: bool):
    if not enabled or device_type == "cpu":
        return _NoOpScaler()
    try:
        return torch.amp.GradScaler(device_type)
    except (TypeError, AttributeError):
        return torch.cuda.amp.GradScaler()


# ─── Loss Functions ────────────────────────────────────────────────────────────

def masked_bce_loss(
    logits:      torch.Tensor,   # (B, n_abs)
    Y:           torch.Tensor,   # (B, n_abs) float
    mask:        torch.Tensor,   # (B, n_abs) bool
    pos_weights: torch.Tensor,   # (n_abs,)
) -> torch.Tensor:
    """Masked multi-task BCE. Normalizes by n_valid_ABs (not 7)."""
    losses: List[torch.Tensor] = []
    for ab in range(Y.shape[1]):
        valid = mask[:, ab]
        if valid.sum() == 0:
            continue
        losses.append(
            F.binary_cross_entropy_with_logits(
                logits[valid, ab], Y[valid, ab], pos_weight=pos_weights[ab]
            )
        )
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def compute_pos_weights(
    Y: np.ndarray, mask: np.ndarray, device: torch.device
) -> torch.Tensor:
    """n_neg/n_pos per AB from training data (TRAIN ONLY)."""
    pw = []
    for ab in range(Y.shape[1]):
        valid = mask[:, ab]
        y_ab  = Y[valid, ab]
        n_pos = float(y_ab.sum())
        n_neg = float(len(y_ab) - n_pos)
        pw.append(n_neg / max(n_pos, 1.0))
    return torch.tensor(pw, dtype=torch.float32, device=device)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class TriLensDataset(Dataset):
    """
    Returns (X_blocks, X_pa, X_gene_agg, Y, label_mask) per sample.
    M_blocks (gene mask) is NOT included — it is a model buffer shared across samples.
    """

    def __init__(
        self,
        X_blocks:    np.ndarray,   # (N, G, L) int8
        X_pa:        np.ndarray,   # (N, G_pa) uint8
        X_gene_agg:  np.ndarray,   # (N, G, 4) float32
        Y:           np.ndarray,   # (N, 7) float32
        label_mask:  np.ndarray,   # (N, 7) bool
    ) -> None:
        self.X_blocks   = X_blocks
        self.X_pa       = X_pa
        self.X_gene_agg = X_gene_agg
        self.Y          = Y
        self.mask       = label_mask

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int) -> Tuple:
        return (
            torch.tensor(self.X_blocks[idx],   dtype=torch.float32),   # (G, L)
            torch.tensor(self.X_pa[idx],        dtype=torch.float32),   # (G_pa,)
            torch.from_numpy(self.X_gene_agg[idx]),                      # (G, 4)
            torch.from_numpy(self.Y[idx]),                                # (7,)
            torch.from_numpy(self.mask[idx]),                             # (7,) bool
        )


def _build_dataset(
    indices:     np.ndarray,
    X_blocks:    np.ndarray,
    X_pa:        np.ndarray,
    X_gene_agg:  np.ndarray,
    Y:           np.ndarray,
    label_mask:  np.ndarray,
) -> TriLensDataset:
    """Slice all arrays to given indices; numpy fancy indexing always copies."""
    return TriLensDataset(
        X_blocks    = X_blocks[indices],
        X_pa        = X_pa[indices],
        X_gene_agg  = X_gene_agg[indices],
        Y           = Y[indices],
        label_mask  = label_mask[indices],
    )


# ─── Data Loading ──────────────────────────────────────────────────────────────

def load_all_data(cfg: DictConfig) -> Dict[str, Any]:
    """Load gene_filtered pkl + P/A pkl; return aligned data dict."""
    log.info(f"Loading gene_filtered pkl: {cfg.gene_filtered_pkl}")
    with open(cfg.gene_filtered_pkl, "rb") as f:
        gf = pickle.load(f)

    log.info(f"Loading gene P/A pkl: {cfg.gene_pa_pkl}")
    with open(cfg.gene_pa_pkl, "rb") as f:
        pa = pickle.load(f)

    X_snp          = gf["X_snp_filtered"]          # (6962, 262225) int8
    gene_to_range  = gf["gene_to_range"]
    selected_genes = gf["selected_gene_ids"]        # list[str], 5000 locus tags
    Y              = gf["Y"].astype(np.float32)
    label_mask     = gf["label_mask"].astype(bool)
    genome_names   = gf["genome_names"]
    random_split   = gf["random_split"]
    phylo_split    = gf.get("phylogeny_split", random_split)

    X_pa    = pa["X_pa"].astype(np.uint8)            # (6962, 32441)
    pa_gids = list(pa["gene_ids"])                   # 32441 locus tags

    log.info(
        f"  X_snp={X_snp.shape}  X_pa={X_pa.shape}  "
        f"Y={Y.shape}  selected_genes={len(selected_genes)}"
    )

    # Verify median P/A count
    pa_per_strain = X_pa.sum(axis=1)
    log.info(
        f"  P/A per strain: median={np.median(pa_per_strain):.0f}  "
        f"mean={pa_per_strain.mean():.0f}  min={pa_per_strain.min()}  max={pa_per_strain.max()}"
    )

    return {
        "X_snp":           X_snp,
        "gene_to_range":   gene_to_range,
        "selected_genes":  selected_genes,
        "Y":               Y,
        "label_mask":      label_mask,
        "genome_names":    genome_names,
        "X_pa":            X_pa,
        "pa_gene_ids":     pa_gids,
        "random_split":    random_split,
        "phylogeny_split": phylo_split,
        "gene_filtered_pkl": str(cfg.gene_filtered_pkl),
    }


def _get_split_indices(
    data: Dict, split_key: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split     = data[split_key]
    train_idx = np.array(split.get("train_idx", split.get("train", [])))
    val_idx   = np.array(split.get("val_idx",   split.get("val",   [])))
    test_idx  = np.array(split.get("test_idx",  split.get("test",  [])))
    return train_idx, val_idx, test_idx


# ─── Expert A Block Builder ───────────────────────────────────────────────────

def build_expert_a_blocks(
    data:       Dict,
    cfg:        DictConfig,
    cache_dir:  Path,
    train_idx:  np.ndarray,
    split_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Build/load per-gene SNP block cache for Expert A.

    `split_name` (optional) is folded into the cache's identity fingerprint
    (see build_pergene_snp_blocks) so a cache built for one split is never
    silently reused for another.
    """
    from src.data.pergene_snp_blocks import build_pergene_snp_blocks, verify_blocks

    X_blocks, M_blocks, col_indices = build_pergene_snp_blocks(
        pkl_path   = str(cfg.gene_filtered_pkl),
        cache_dir  = str(cache_dir),
        L_max      = int(cfg.expert_a.L_max),
        train_idx  = train_idx,
        force_rebuild = False,
        split_name = split_name,
    )
    verify_blocks(X_blocks, M_blocks)
    return X_blocks, M_blocks, col_indices


# ─── Expert C Feature Builder ─────────────────────────────────────────────────

def _build_gene_feat_inline(
    X_snp: np.ndarray,
    gene_to_range: Dict,
    selected_genes: List,
) -> np.ndarray:
    """Build (N, G, 4) [mean, var, n_nonref, max] inline when no builder available."""
    N = X_snp.shape[0]
    G = len(selected_genes)
    out = np.zeros((N, G, 4), dtype=np.float32)
    t0  = time.time()
    for g, gid in enumerate(selected_genes):
        s, e = gene_to_range[gid]
        if e <= s:
            continue
        cols = X_snp[:, s:e].astype(np.float32)
        out[:, g, 0] = cols.mean(axis=1)
        out[:, g, 1] = cols.var(axis=1)
        out[:, g, 2] = (cols > 0).sum(axis=1).astype(np.float32)
        out[:, g, 3] = cols.max(axis=1)
        if (g + 1) % 1000 == 0:
            log.info(f"  gene_feat inline: {g+1}/{G}  ({time.time()-t0:.0f}s)")
    return out


def build_expert_c_features(
    data:       Dict,
    train_idx:  np.ndarray,
    cache_dir:  Path,
    split_name: str = "",
) -> Tuple[np.ndarray, Dict]:
    """
    Build/load gene-aggregated (N, G, 4) features for Expert C.
    Standardizes per-stat using TRAIN ONLY stats; caches to disk.

    Cache identity: same fingerprinting scheme as build_pergene_snp_blocks
    (hash of gene_filtered_pkl path + split_name + sorted train_idx), stored
    in a `cache_identity.json` sidecar and checked before any existing cache
    is trusted. A cache from a different split/pkl, or one written before
    this check existed (no sidecar), is rebuilt rather than silently reused.
    This guards future runs; it does not alter any already-produced cache
    or result file.
    """
    from src.data.pergene_snp_blocks import (
        _load_and_check_identity, _save_identity, compute_cache_identity_hash,
    )

    feat_p = cache_dir / "gene_features_5000x4.npy"
    mean_p = cache_dir / "gene_feat_mean.npy"
    std_p  = cache_dir / "gene_feat_std.npy"
    identity_p = cache_dir / "cache_identity.json"

    pkl_path = str(data.get("gene_filtered_pkl", ""))
    expected_hash = compute_cache_identity_hash(pkl_path, split_name, train_idx)
    cache_files_present = feat_p.exists() and mean_p.exists() and std_p.exists()
    identity_ok = cache_files_present and _load_and_check_identity(
        identity_p, expected_hash, pkl_path, split_name, train_idx
    )

    if identity_ok:
        X_feat    = np.load(str(feat_p))
        feat_mean = np.load(str(mean_p))
        feat_std  = np.load(str(std_p))
        log.info(f"Loaded Expert C features from cache: {feat_p}  "
                 f"shape={X_feat.shape}  (identity verified)")
        return X_feat, {"mean": feat_mean, "std": feat_std}

    for p in (feat_p, mean_p, std_p, identity_p):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    log.info("Building Expert C gene-aggregated features (inline)…")
    t0 = time.time()

    try:
        from src.data.gene_aggregated_features import build_gene_feature_matrix_from_pkl
        X_raw = build_gene_feature_matrix_from_pkl(
            data["gene_filtered_pkl"], stats=["mean", "var", "n_nonref", "max"]
        )
        X_raw = X_raw.numpy() if hasattr(X_raw, "numpy") else np.asarray(X_raw, np.float32)
        log.info(f"  Built via src.data.gene_aggregated_features in {time.time()-t0:.1f}s")
    except Exception as exc:
        log.warning(f"  Feature builder unavailable ({exc}); building inline…")
        X_raw = _build_gene_feat_inline(
            data["X_snp"], data["gene_to_range"], data["selected_genes"]
        )
        log.info(f"  Built inline in {time.time()-t0:.1f}s")

    X_raw = np.asarray(X_raw, dtype=np.float32)

    # Standardize per-stat on TRAIN ONLY
    flat      = X_raw[train_idx].reshape(-1, 4)        # (n_tr*G, 4)
    feat_mean = flat.mean(axis=0).astype(np.float32)
    feat_std  = flat.std(axis=0).clip(min=1e-8).astype(np.float32)
    X_feat    = ((X_raw - feat_mean) / feat_std).astype(np.float32)

    np.save(str(feat_p), X_feat)
    np.save(str(mean_p), feat_mean)
    np.save(str(std_p),  feat_std)
    _save_identity(identity_p, expected_hash, pkl_path, split_name, train_idx)
    log.info(
        f"Saved Expert C features: {feat_p}  shape={X_feat.shape}  "
        f"mean={feat_mean.round(3).tolist()}  std={feat_std.round(3).tolist()}  "
        f"(identity fingerprint saved)"
    )
    return X_feat, {"mean": feat_mean, "std": feat_std}


# ─── Training ──────────────────────────────────────────────────────────────────

def _train_epoch(
    model:       TriLens,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    scaler,
    autocast_ctx,
    pos_weights: torch.Tensor,
    device:      torch.device,
    cfg:         DictConfig,
    beta:        float,           # entropy regularization coefficient
) -> Dict[str, Any]:
    """One training epoch with AMP and gradient accumulation."""
    model.train()
    grad_accum = int(cfg.training.grad_accum)
    aux_w      = float(cfg.aux_loss_weight)

    tot_loss = tot_res = tot_aux = tot_ent = 0.0
    gate_acc: Optional[torch.Tensor] = None
    n_updates = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        X_b, X_pa, X_gc, Y, mask = [t.to(device, non_blocking=True) for t in batch]
        is_last = step == len(loader) - 1

        with autocast_ctx:
            out    = model(X_b, X_pa, X_gc)
            l_fin  = out["l_final"]
            L_res  = masked_bce_loss(l_fin, Y, mask, pos_weights)

            # Auxiliary per-expert losses (only when multiple experts active)
            L_aux = torch.tensor(0.0, device=device)
            if model.n_active > 1:
                for key in ("l_A", "l_B", "l_C"):
                    lk = out.get(key)
                    if lk is not None:
                        L_aux = L_aux + masked_bce_loss(lk, Y, mask, pos_weights)
                L_aux = aux_w * L_aux

            # MoE entropy loss
            L_ent = torch.tensor(0.0, device=device)
            if out.get("gate") is not None:
                gate = out["gate"]           # (B, 7, n_active)
                H    = -(gate * (gate + 1e-8).log()).sum(dim=-1).mean()
                L_ent = -beta * H
                with torch.no_grad():
                    g_avg = gate.float().mean(dim=(0, 1)).cpu()  # (n_active,)
                    gate_acc = g_avg if gate_acc is None else gate_acc + g_avg

            loss = (L_res + L_aux + L_ent) / grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % grad_accum == 0 or is_last:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.training.grad_clip_norm))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            n_updates += 1

        tot_loss += L_res.item() + L_aux.item() + L_ent.item()
        tot_res  += L_res.item()
        tot_aux  += L_aux.item()
        tot_ent  += L_ent.item()

    n  = max(len(loader), 1)
    return {
        "total":      tot_loss / n,
        "resistance": tot_res  / n,
        "aux":        tot_aux  / n,
        "entropy":    tot_ent  / n,
        "gate_avg":   gate_acc / n if gate_acc is not None else None,
    }


@torch.no_grad()
def _predict(
    model:        TriLens,
    dataset:      TriLensDataset,
    device:       torch.device,
    batch_size:   int,
    collect_gate: bool = False,
    collect_attn: bool = False,
) -> Tuple:
    """Inference → (probs, gates, a_A, a_B, a_C) all numpy."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_l, gate_l, aA_l, aB_l, aC_l = [], [], [], [], []

    for batch in loader:
        X_b, X_pa, X_gc, _, _ = [t.to(device, non_blocking=True) for t in batch]
        out = model(X_b, X_pa, X_gc)
        probs_l.append(torch.sigmoid(out["l_final"]).cpu().float().numpy())
        if collect_gate and out.get("gate") is not None:
            gate_l.append(out["gate"].cpu().float().numpy())
        if collect_attn:
            for src, lst in (("a_A", aA_l), ("a_B", aB_l), ("a_C", aC_l)):
                if out.get(src) is not None:
                    lst.append(out[src].cpu().float().numpy())

    probs = np.concatenate(probs_l, axis=0) if probs_l else np.zeros((0, len(AB_NAMES)))
    gates = np.concatenate(gate_l, axis=0)  if gate_l  else None
    a_A   = np.concatenate(aA_l, axis=0)    if aA_l    else None
    a_B   = np.concatenate(aB_l, axis=0)    if aB_l    else None
    a_C   = np.concatenate(aC_l, axis=0)    if aC_l    else None
    return probs, gates, a_A, a_B, a_C


def _avg_prauc(probs: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> float:
    """Mean PR-AUC across valid ABs."""
    vals = []
    for ab in range(Y.shape[1]):
        valid = mask[:, ab]
        if valid.sum() < 2:
            continue
        yt, ys = Y[valid, ab], probs[valid, ab]
        if yt.sum() == 0 or yt.sum() == valid.sum():
            continue
        try:
            vals.append(float(average_precision_score(yt, ys)))
        except Exception:
            pass
    return float(np.mean(vals)) if vals else 0.0


def _train_until_convergence(
    model:        TriLens,
    fit_ds:       TriLensDataset,
    es_ds:        TriLensDataset,
    pos_weights:  torch.Tensor,
    cfg:          DictConfig,
    device:       torch.device,
) -> Tuple[TriLens, List[Dict]]:
    """
    Train model on fit_ds; early-stop on es_ds avg PR-AUC.
    Returns best-checkpoint model + history list.
    """
    bs = int(cfg.training.batch_size)
    fit_loader = DataLoader(
        fit_ds, batch_size=bs, shuffle=True, num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    autocast_ctx = _make_autocast(device.type, bool(cfg.training.use_amp))
    scaler       = _make_scaler(device.type, bool(cfg.training.use_amp))
    optimizer    = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(cfg.training.max_epochs),
        eta_min=float(cfg.training.lr) * 0.01,
    )

    best_val     = -np.inf
    best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
    patience_ctr = 0
    beta         = float(cfg.moe.entropy_beta)
    history      = []
    eval_bs      = bs * 2

    for epoch in range(int(cfg.training.max_epochs)):
        t_ep = time.time()
        stats = _train_epoch(
            model, fit_loader, optimizer, scaler, autocast_ctx,
            pos_weights, device, cfg, beta,
        )
        scheduler.step()

        # Expert collapse guard
        gw = stats.get("gate_avg")
        if gw is not None and epoch >= 10:
            if gw.min().item() < float(cfg.moe.expert_collapse_threshold):
                log.warning(
                    f"  ep {epoch+1}: expert collapse detected! "
                    f"gate_avg={gw.numpy().round(3).tolist()} → β ↑ to "
                    f"{cfg.moe.expert_collapse_beta_increase}"
                )
                beta = float(cfg.moe.expert_collapse_beta_increase)

        # Early-stop evaluation
        es_probs, _, _, _, _ = _predict(model, es_ds, device, eval_bs)
        val_avg = _avg_prauc(es_probs, es_ds.Y, es_ds.mask)

        ep_log = {
            "epoch":    epoch + 1,
            "es_avg":   val_avg,
            "runtime":  round(time.time() - t_ep, 1),
            **{k: v.tolist() if isinstance(v, torch.Tensor) else v
               for k, v in stats.items()},
        }
        history.append(ep_log)

        log.info(
            f"  ep {epoch+1:2d}/{cfg.training.max_epochs}  "
            f"loss={stats['total']:.4f} (res={stats['resistance']:.3f} "
            f"aux={stats['aux']:.3f} ent={stats['entropy']:.4f})  "
            f"es_avg={val_avg:.4f}  "
            + (f"gate={gw.numpy().round(3)}" if gw is not None else "")
        )

        if val_avg > best_val + 1e-5:
            best_val     = val_avg
            best_state   = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= int(cfg.training.patience):
                log.info(f"  Early stop at epoch {epoch+1}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    log.info(f"  Best es_avg = {best_val:.4f}")
    return model, history


def _create_model(
    cfg:       DictConfig,
    M_blocks:  np.ndarray,
    use_a:     bool,
    use_b:     bool,
    use_c:     bool,
    fusion:    str,
    device:    torch.device,
) -> TriLens:
    gene_mask = torch.tensor(M_blocks, dtype=torch.bool)
    model     = TriLens(cfg, gene_mask, use_a=use_a, use_b=use_b, use_c=use_c, fusion=fusion)
    model     = model.to(device)
    log.info(f"  Model: {model.count_parameters():,} params  fusion={fusion}  experts={'ABC'[not use_a::1]}")
    return model


# ─── Metrics ──────────────────────────────────────────────────────────────────

def _bootstrap_ci(
    y_true: np.ndarray, y_score: np.ndarray, n: int = 1000, seed: int = 42
) -> Tuple[float, float, float]:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan, np.nan
    point = float(average_precision_score(y_true, y_score))
    rng   = np.random.default_rng(seed)
    boot  = []
    for _ in range(n):
        idx = rng.integers(0, len(y_true), size=len(y_true))
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        try:
            boot.append(float(average_precision_score(yt, ys)))
        except Exception:
            pass
    if len(boot) < 50:
        return point, np.nan, np.nan
    arr = np.array(boot)
    return point, float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def compute_metrics(
    probs: np.ndarray, Y: np.ndarray, mask: np.ndarray, cfg: DictConfig
) -> Dict[str, Dict]:
    results: Dict[str, Dict] = {}
    for ab_idx, ab in enumerate(AB_NAMES):
        valid = mask[:, ab_idx]
        if valid.sum() < 2:
            results[ab] = {"prauc": np.nan, "mcc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
            continue
        yt, ys = Y[valid, ab_idx], probs[valid, ab_idx]
        if yt.sum() == 0 or yt.sum() == valid.sum():
            results[ab] = {"prauc": np.nan, "mcc": np.nan, "ci_lo": np.nan, "ci_hi": np.nan}
            continue
        prauc = float(average_precision_score(yt, ys))
        mcc   = float(matthews_corrcoef(yt, (ys >= 0.5).astype(int)))
        n_b   = int(cfg.evaluation.bootstrap_n)
        s_b   = int(cfg.evaluation.bootstrap_seed)
        pt, lo, hi = _bootstrap_ci(yt, ys, n_b, s_b)
        results[ab] = {"prauc": prauc, "mcc": mcc, "ci_lo": lo, "ci_hi": hi}
    avg = float(np.nanmean([r["prauc"] for r in results.values()]))
    results["_avg_prauc"] = avg
    return results


# ─── Standalone Ablations T1–T5 ──────────────────────────────────────────────

def run_standalone(
    abl_id:     str,
    cfg:        DictConfig,
    data:       Dict,
    X_blocks:   np.ndarray,
    M_blocks:   np.ndarray,
    X_gene_agg: np.ndarray,
    split_key:  str,
    seed:       int,
    device:     torch.device,
) -> Dict:
    """Train on train_idx, ES on val_idx, eval on test_idx. No K-fold."""
    abl_def = ABLATION_DEFS[abl_id]
    use_a, use_b, use_c, fusion = (
        abl_def["use_a"], abl_def["use_b"], abl_def["use_c"], abl_def["fusion"]
    )
    set_seed(seed)

    train_idx, val_idx, test_idx = _get_split_indices(data, split_key)
    pw = compute_pos_weights(data["Y"][train_idx], data["label_mask"][train_idx], device)

    train_ds = _build_dataset(train_idx, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
    val_ds   = _build_dataset(val_idx,   X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
    test_ds  = _build_dataset(test_idx,  X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])

    model, history = _train_until_convergence(
        _create_model(cfg, M_blocks, use_a, use_b, use_c, fusion, device),
        train_ds, val_ds, pw, cfg, device,
    )

    bs = int(cfg.training.batch_size) * 2
    test_probs, test_gates, a_A, a_B, a_C = _predict(
        model, test_ds, device, bs, collect_gate=True, collect_attn=True
    )
    val_probs, _, _, _, _ = _predict(model, val_ds, device, bs)

    metrics = compute_metrics(test_probs, data["Y"][test_idx], data["label_mask"][test_idx], cfg)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "abl_id":     abl_id,
        "desc":       abl_def["desc"],
        "mode":       "standalone",
        "split_key":  split_key,
        "metrics":    metrics,
        "test_probs": test_probs,
        "val_probs":  val_probs,
        "gate_test":  test_gates,
        "a_A":        a_A,
        "a_B":        a_B,
        "a_C":        a_C,
        "history":    history,
        "test_idx":   test_idx,
        "val_idx":    val_idx,
        "train_idx":  train_idx,
    }


# ─── K-fold OOF + Stacking T6/T7 ──────────────────────────────────────────────

def run_kfold_oof(
    cfg:        DictConfig,
    data:       Dict,
    X_blocks:   np.ndarray,
    M_blocks:   np.ndarray,
    X_gene_agg: np.ndarray,
    split_key:  str,
    seed:       int,
    device:     torch.device,
) -> Dict:
    """5-fold OOF on train_idx; average val/test across folds."""
    train_idx, val_idx, test_idx = _get_split_indices(data, split_key)
    n_train = len(train_idx)
    n_folds = int(cfg.n_folds)
    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=int(cfg.kfold_seed))

    # Global pos_weights from full training set
    pw = compute_pos_weights(data["Y"][train_idx], data["label_mask"][train_idx], device)

    oof_preds    = np.zeros((n_train, len(AB_NAMES)), dtype=np.float32)
    val_preds_f  = []
    test_preds_f = []
    gate_test_f  = []
    histories    = []

    val_ds  = _build_dataset(val_idx,  X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
    test_ds = _build_dataset(test_idx, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])

    for fold, (fit_es_local, oof_local) in enumerate(kf.split(np.arange(n_train))):
        fold_seed = seed + fold * 100
        set_seed(fold_seed)
        log.info(f"\n{'─'*60}\n  [Fold {fold+1}/{n_folds}]  seed={fold_seed}")

        # Split fit_es into fit (90%) and early-stop (10%)
        n_fe  = len(fit_es_local)
        n_es  = max(int(n_fe * 0.1), 50)
        fit_l = fit_es_local[:n_fe - n_es]
        es_l  = fit_es_local[n_fe - n_es:]

        fit_s = train_idx[fit_l]
        es_s  = train_idx[es_l]
        oof_s = train_idx[oof_local]
        log.info(f"  fit={len(fit_s)}  es={len(es_s)}  oof={len(oof_s)}")

        fit_ds = _build_dataset(fit_s, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
        es_ds  = _build_dataset(es_s,  X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
        oof_ds = _build_dataset(oof_s, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])

        model = _create_model(cfg, M_blocks, use_a=True, use_b=True, use_c=True, fusion="moe", device=device)
        model, hist = _train_until_convergence(model, fit_ds, es_ds, pw, cfg, device)
        histories.append(hist)

        bs = int(cfg.training.batch_size) * 2
        oof_p, _, _, _, _         = _predict(model, oof_ds,  device, bs)
        val_p, _, _, _, _         = _predict(model, val_ds,  device, bs)
        tst_p, tst_g, aA, aB, aC = _predict(model, test_ds, device, bs, collect_gate=True, collect_attn=True)

        oof_preds[oof_local] = oof_p
        val_preds_f.append(val_p)
        test_preds_f.append(tst_p)
        if tst_g is not None:
            gate_test_f.append(tst_g)

        # Log fold result
        fold_metrics = compute_metrics(tst_p, data["Y"][test_idx], data["label_mask"][test_idx], cfg)
        log.info(
            f"  Fold {fold+1} test avg = {fold_metrics['_avg_prauc']:.4f}  "
            + "  ".join(f"{ab}={fold_metrics.get(ab, {}).get('prauc', 0):.3f}" for ab in AB_NAMES)
        )

        del model, fit_ds, es_ds, oof_ds
        torch.cuda.empty_cache()
        gc.collect()

    val_preds  = np.mean(val_preds_f,  axis=0)
    test_preds = np.mean(test_preds_f, axis=0)
    gate_test  = np.mean(gate_test_f,  axis=0) if gate_test_f else None

    return {
        "oof_preds":   oof_preds,         # (n_train, 7)
        "val_preds":   val_preds,          # (n_val, 7)
        "test_preds":  test_preds,         # (n_test, 7)
        "gate_test":   gate_test,          # (n_test, 7, 3) averaged across folds
        "train_idx":   train_idx,
        "val_idx":     val_idx,
        "test_idx":    test_idx,
        "histories":   histories,
        # Attention from LAST fold (for interpretability — consistent choice)
        "a_A": aA, "a_B": aB, "a_C": aC,
    }


def run_meta_stack(
    oof_trilens:    np.ndarray,   # (n_train, 7)
    test_trilens:   np.ndarray,   # (n_test, 7)
    oof_classical:  Optional[np.ndarray],   # (n_train, 5, 7) or None
    test_classical: Optional[np.ndarray],   # (n_test, 5, 7) or None
    Y_tr:           np.ndarray,
    mask_tr:        np.ndarray,
    cfg:            DictConfig,
    val_trilens:    Optional[np.ndarray] = None,   # (n_val, 7)
    val_classical:  Optional[np.ndarray] = None,   # (n_val, 5, 7)
    return_val:     bool = False,
):
    """Per-AB L2 LogReg meta-learner on up to 6 bases.

    Returns (test_meta, meta_weights), or (test_meta, meta_weights, val_meta)
    with return_val=True. val_meta is the SAME fitted meta-learner applied to the
    validation bases, so a threshold tuned on it transfers to test_meta. (Before,
    the saved validation probabilities were TriLens-only while the test
    probabilities were the stack, so tuned thresholds came from a different model.)
    """
    n_test      = len(test_trilens)
    n_abs       = len(AB_NAMES)
    test_meta   = np.full((n_test, n_abs), np.nan, dtype=np.float32)
    has_val     = val_trilens is not None and (oof_classical is None or val_classical is not None)
    val_meta    = (np.full((len(val_trilens), n_abs), np.nan, dtype=np.float32)
                   if has_val else None)
    meta_weights: Dict[str, Dict] = {}
    C           = float(cfg.meta_C)

    classical_names = ["XGB_gene", "LGBM", "RF", "LogReg", "XGB_raw"]

    for ab_idx, ab in enumerate(AB_NAMES):
        valid_tr = mask_tr[:, ab_idx]
        if valid_tr.sum() < 2:
            test_meta[:, ab_idx] = test_trilens[:, ab_idx]
            if has_val:
                val_meta[:, ab_idx] = val_trilens[:, ab_idx]
            continue

        # Build training feature matrix
        if oof_classical is not None:
            X_meta_tr = np.column_stack([
                oof_classical[valid_tr, :, ab_idx],      # (n_valid, 5)
                oof_trilens[valid_tr, ab_idx:ab_idx+1],  # (n_valid, 1)
            ])  # (n_valid, 6)
        else:
            X_meta_tr = oof_trilens[valid_tr, ab_idx:ab_idx+1]  # (n_valid, 1)

        y_meta_tr = Y_tr[valid_tr, ab_idx]

        if y_meta_tr.sum() == 0 or y_meta_tr.sum() == valid_tr.sum():
            log.warning(f"  Meta {ab}: single-class train split — using TriLens probs")
            test_meta[:, ab_idx] = test_trilens[:, ab_idx]
            if has_val:
                val_meta[:, ab_idx] = val_trilens[:, ab_idx]
            continue

        lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
        lr.fit(X_meta_tr, y_meta_tr)

        if test_classical is not None:
            X_meta_te = np.column_stack([
                test_classical[:, :, ab_idx],           # (n_test, 5)
                test_trilens[:, ab_idx:ab_idx+1],       # (n_test, 1)
            ])
        else:
            X_meta_te = test_trilens[:, ab_idx:ab_idx+1]

        test_meta[:, ab_idx] = lr.predict_proba(X_meta_te)[:, 1]
        if has_val:
            X_meta_va = (np.column_stack([val_classical[:, :, ab_idx], val_trilens[:, ab_idx:ab_idx+1]])
                         if oof_classical is not None else val_trilens[:, ab_idx:ab_idx+1])
            val_meta[:, ab_idx] = lr.predict_proba(X_meta_va)[:, 1]
        meta_weights[ab] = {
            "bases":     (classical_names + ["TriLens"]) if oof_classical is not None else ["TriLens"],
            "coef":      lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
        }
        log.info(
            f"  Meta {ab}: bases={len(lr.coef_[0])}  "
            f"coef={np.array(lr.coef_[0]).round(3).tolist()}"
        )

    if return_val:
        return test_meta, meta_weights, val_meta
    return test_meta, meta_weights


def _load_classical_arm(v9_path: Path, train_idx, val_idx, test_idx):
    """Classical base predictions (OOF, test, val or None), checked against this run's split."""
    # A file that is there but unreadable is an error, not a TriLens-only stack.
    v9 = np.load(str(v9_path), allow_pickle=False)
    # oof_base_probs: (n_train, n_bases, n_abs); drop GAN (index 5)
    oof_classical  = v9["oof_base_probs"][:, :5, :]   # (4872, 5, 7)
    test_classical = v9["test_base_probs"][:, :5, :]  # (1045, 5, 7)
    val_classical = None
    if "val_base_probs" in v9.files:
        val_classical = v9["val_base_probs"][:, :5, :]
    # The two arms must describe the same strains in the same order,
    # otherwise the meta-learner pairs one strain's classical score with
    # another strain's TriLens score.
    keys = [("train_indices", train_idx), ("test_indices", test_idx)]
    if val_classical is not None:
        keys.append(("val_indices", val_idx))
    for key, idx in keys:
        if key not in v9.files:
            raise ValueError(f"{v9_path}: no {key}, so the row order of the classical arm cannot be checked")
        if not np.array_equal(np.asarray(v9[key]).ravel(), np.asarray(idx).ravel()):
            raise ValueError(f"{v9_path}: {key} differ from this run's split - "
                             "the classical arm was built on another split or row order")
    log.info(f"Loaded v9.1 classical OOF from {v9_path} (split indices match)")
    return oof_classical, test_classical, val_classical


def run_kfold_and_stack(
    abl_id:     str,
    cfg:        DictConfig,
    data:       Dict,
    X_blocks:   np.ndarray,
    M_blocks:   np.ndarray,
    X_gene_agg: np.ndarray,
    split_key:  str,
    seed:       int,
    device:     torch.device,
) -> Dict:
    """Full K-fold OOF + meta-stacking pipeline for T6/T7."""
    log.info(f"K-fold OOF ({split_key}) …")
    kfold = run_kfold_oof(cfg, data, X_blocks, M_blocks, X_gene_agg, split_key, seed, device)

    train_idx = kfold["train_idx"]
    val_idx   = kfold["val_idx"]
    test_idx  = kfold["test_idx"]

    # Load v9.1 classical OOF predictions (5 classical bases, drop GAN at index 5)
    v9_path = Path(str(cfg.v9_predictions))
    oof_classical  = None
    test_classical = None
    val_classical  = None
    if v9_path.exists():
        oof_classical, test_classical, val_classical = _load_classical_arm(
            v9_path, train_idx, val_idx, test_idx)
    elif v9_path.name.startswith("_intentionally_absent"):
        log.info("Neural arm only (the classical arm is stacked later by the diagnose_* script).")
    else:
        # Used to be a warning, which let a missing classical arm silently turn the
        # headline into a TriLens-only number.
        raise FileNotFoundError(
            f"classical-arm predictions not found at {v9_path}. "
            "Run experiment=stage3_cascadestack_v9 first."
        )

    # Meta-stacking
    log.info("Running meta-stacking …")
    test_meta, meta_weights, val_meta = run_meta_stack(
        oof_trilens    = kfold["oof_preds"],
        test_trilens   = kfold["test_preds"],
        oof_classical  = oof_classical,
        test_classical = test_classical,
        Y_tr           = data["Y"][train_idx],
        mask_tr        = data["label_mask"][train_idx],
        cfg            = cfg,
        val_trilens    = kfold["val_preds"],
        val_classical  = val_classical,
        return_val     = True,
    )
    val_is_stack = val_meta is not None and (oof_classical is None or val_classical is not None)
    if val_meta is None:
        log.warning("No classical validation probabilities: val_probs stays TriLens-only; "
                    "do not tune thresholds on it.")
        val_meta = kfold["val_preds"]

    # Final test metrics
    metrics = compute_metrics(test_meta, data["Y"][test_idx], data["label_mask"][test_idx], cfg)
    log.info(
        f"[{abl_id}] Stack avg test PR-AUC = {metrics['_avg_prauc']:.4f}  "
        + "  ".join(f"{ab}={metrics.get(ab,{}).get('prauc',0):.3f}" for ab in AB_NAMES)
    )

    # Build base prob arrays for predictions_saved.npz
    n_base = 6 if oof_classical is not None else 1
    oof_base_probs  = np.zeros((len(train_idx), n_base, len(AB_NAMES)), np.float32)
    test_base_probs = np.zeros((len(test_idx),  n_base, len(AB_NAMES)), np.float32)
    if oof_classical is not None:
        oof_base_probs[:, :5, :]  = oof_classical
        test_base_probs[:, :5, :] = test_classical
    oof_base_probs[:, -1, :]  = kfold["oof_preds"]
    test_base_probs[:, -1, :] = kfold["test_preds"]

    base_names = (["XGB_gene", "LGBM", "RF", "LogReg", "XGB_raw", "TriLens"]
                  if oof_classical is not None else ["TriLens"])

    return {
        "abl_id":          abl_id,
        "desc":            ABLATION_DEFS[abl_id]["desc"],
        "mode":            "kfold_stack",
        "split_key":       split_key,
        "metrics":         metrics,
        "test_probs":      test_meta,
        "val_probs":       val_meta,            # the stack on validation (same model as test_probs)
        "val_trilens":     kfold["val_preds"],  # TriLens alone on validation
        "val_is_stack":    bool(val_is_stack),
        "gate_test":       kfold["gate_test"],
        "a_A":             kfold.get("a_A"),
        "a_B":             kfold.get("a_B"),
        "a_C":             kfold.get("a_C"),
        "history":         kfold["histories"],
        "meta_weights":    meta_weights,
        "base_names":      base_names,
        "oof_base_probs":  oof_base_probs,
        "test_base_probs": test_base_probs,
        "oof_trilens":     kfold["oof_preds"],
        "test_trilens":    kfold["test_preds"],
        "train_idx":       train_idx,
        "val_idx":         val_idx,
        "test_idx":        test_idx,
    }


# ─── Attention Top-Gene Analysis ──────────────────────────────────────────────

def compute_topgenes(
    attn:      np.ndarray,   # (N, G) attention weights
    Y_te:      np.ndarray,   # (N, 7)
    mask_te:   np.ndarray,   # (N, 7)
    gene_ids:  List[str],
    n_top:     int = 20,
) -> pd.DataFrame:
    rows = []
    for ab_idx, ab in enumerate(AB_NAMES):
        valid = mask_te[:, ab_idx]
        if valid.sum() < 2 or attn is None:
            continue
        y_ab  = Y_te[valid, ab_idx]
        a_ab  = attn[valid]
        res   = y_ab == 1
        avg   = a_ab[res].mean(axis=0) if res.sum() > 0 else a_ab.mean(axis=0)
        top   = np.argsort(avg)[::-1][:n_top]
        for rank, gi in enumerate(top):
            if gi < len(gene_ids):
                rows.append({
                    "ab":          ab,
                    "rank":        rank + 1,
                    "gene_id":     str(gene_ids[gi]),
                    "attn_weight": float(avg[gi]),
                    "n_resistant": int(res.sum()),
                })
    return pd.DataFrame(rows)


# ─── Output Saving ────────────────────────────────────────────────────────────

def save_all_outputs(
    all_results: Dict[str, Dict],
    data:        Dict,
    output_dir:  Path,
    cfg:         DictConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── ablation_results.csv ──────────────────────────────────────────────
    rows = []
    for abl_id, res in all_results.items():
        if "error" in res:
            continue
        for ab, ab_res in res.get("metrics", {}).items():
            if ab.startswith("_"):
                continue
            rows.append({
                "ablation":  abl_id,
                "desc":      res.get("desc", "")[:50],
                "split_key": res.get("split_key", ""),
                "ab":        ab,
                "prauc":     ab_res.get("prauc", np.nan),
                "mcc":       ab_res.get("mcc", np.nan),
                "ci_lo":     ab_res.get("ci_lo", np.nan),
                "ci_hi":     ab_res.get("ci_hi", np.nan),
                "v91_prauc": V91_PER_AB.get(ab, np.nan),
                "delta_v91": ab_res.get("prauc", np.nan) - V91_PER_AB.get(ab, np.nan),
            })
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "ablation_results.csv", index=False)
        log.info("Saved ablation_results.csv")

    # ── attention CSVs ──────────────────────────────────────────────────
    # Use best T5 or T6 result for attention
    best_abl = next(
        (k for k in ("T6", "T5", "T4", "T3", "T2", "T1") if k in all_results and "error" not in all_results[k]),
        None
    )
    if best_abl:
        res = all_results[best_abl]
        te_idx  = res.get("test_idx", np.array([]))
        Y_te    = data["Y"][te_idx] if len(te_idx) > 0 else data["Y"][:1]
        mask_te = data["label_mask"][te_idx] if len(te_idx) > 0 else data["label_mask"][:1]

        for expert_key, gene_ids, out_name in [
            ("a_A", data["selected_genes"], "attention_A_topgenes.csv"),
            ("a_B", data["pa_gene_ids"],    "attention_B_topgenes.csv"),
            ("a_C", data["selected_genes"], "attention_C_topgenes.csv"),
        ]:
            attn = res.get(expert_key)
            if attn is not None:
                df = compute_topgenes(attn, Y_te, mask_te, gene_ids)
                df.to_csv(output_dir / out_name, index=False)
                log.info(f"Saved {out_name}  ({len(df)} rows)")

    # ── gate_weights_test.npz ──────────────────────────────────────────
    gate_res = next(
        (all_results[k] for k in ("T6", "T5") if k in all_results and
         "error" not in all_results[k] and all_results[k].get("gate_test") is not None),
        None
    )
    if gate_res is not None and gate_res.get("gate_test") is not None:
        np.savez(str(output_dir / "gate_weights_test.npz"), gate=gate_res["gate_test"])
        log.info(f"Saved gate_weights_test.npz  shape={gate_res['gate_test'].shape}")

    # ── stack_meta_weights.csv ─────────────────────────────────────────
    meta_rows = []
    for abl_id in ("T6", "T7"):
        res = all_results.get(abl_id)
        if res and "error" not in res and res.get("meta_weights"):
            for ab, mw in res["meta_weights"].items():
                row = {"ablation": abl_id, "ab": ab}
                for base, coef in zip(mw.get("bases", []), mw.get("coef", [])):
                    row[base] = round(coef, 6)
                row["intercept"] = mw.get("intercept", np.nan)
                meta_rows.append(row)
    if meta_rows:
        pd.DataFrame(meta_rows).to_csv(output_dir / "stack_meta_weights.csv", index=False)
        log.info("Saved stack_meta_weights.csv")

    # ── split_comparison.csv ───────────────────────────────────────────
    split_rows = []
    for abl_id in ("T6", "T7"):
        res = all_results.get(abl_id)
        if res and "error" not in res:
            for ab, ab_res in res.get("metrics", {}).items():
                if ab.startswith("_"):
                    continue
                split_rows.append({
                    "ablation":  abl_id,
                    "split_key": res.get("split_key", ""),
                    "ab":        ab,
                    "prauc":     ab_res.get("prauc", np.nan),
                    "ci_lo":     ab_res.get("ci_lo", np.nan),
                    "ci_hi":     ab_res.get("ci_hi", np.nan),
                })
    if split_rows:
        pd.DataFrame(split_rows).to_csv(output_dir / "split_comparison.csv", index=False)
        log.info("Saved split_comparison.csv")

    # ── trilens_kfold_oof.npz ──────────────────────────────────────────
    t6 = all_results.get("T6") or all_results.get("T7")
    if t6 and "error" not in t6 and "oof_trilens" in t6:
        np.savez(
            str(output_dir / "trilens_kfold_oof.npz"),
            oof_preds   = t6["oof_trilens"],
            val_preds   = t6.get("val_trilens", t6["val_probs"]),
            test_preds  = t6["test_trilens"],
            train_idx   = t6["train_idx"],
            val_idx     = t6["val_idx"],
            test_idx    = t6["test_idx"],
        )
        log.info("Saved trilens_kfold_oof.npz")

    # ── predictions_saved.npz (v9.1-compatible) ────────────────────────
    t6_res = all_results.get("T6")
    if t6_res and "error" not in t6_res and "oof_base_probs" in t6_res:
        np.savez(
            str(output_dir / "predictions_saved.npz"),
            test_ens_probs  = t6_res["test_probs"],
            test_base_probs = t6_res["test_base_probs"],
            oof_base_probs  = t6_res["oof_base_probs"],
            val_ens_probs   = t6_res.get("val_probs", np.zeros((0, len(AB_NAMES)))),
            test_indices    = t6_res["test_idx"],
            val_indices     = t6_res["val_idx"],
            train_indices   = t6_res["train_idx"],
            base_names      = np.array(t6_res.get("base_names", ["TriLens"])),
            model_name      = np.array("TriLens-T6-attnfix"),
            # True: val_ens_probs is the same stacked model as test_ens_probs, so a
            # threshold tuned on validation is valid for test (checked by check_leakage.py)
            val_is_stack    = np.array(bool(t6_res.get("val_is_stack", False))),
        )
        log.info("Saved predictions_saved.npz  (v9.1-compatible format)")

    # ── attention_full.npz ─────────────────────────────────────────────
    # The per-gene attention for every test strain, not just the top-20
    # average. Every interpretability question in this project needed this
    # and had to be answered from the top-20 summary instead.
    attn_res = next(
        (all_results[k] for k in ("T6", "T5")
         if k in all_results and "error" not in all_results[k]
         and all_results[k].get("a_B") is not None),
        None,
    )
    if attn_res is not None:
        payload = {"test_indices": attn_res["test_idx"]}
        for key in ("a_A", "a_B", "a_C"):
            arr = attn_res.get(key)
            if arr is not None:
                payload[key] = np.asarray(arr, dtype=np.float16)
        if attn_res.get("gate_test") is not None:
            payload["gate"] = np.asarray(attn_res["gate_test"], dtype=np.float32)
        np.savez_compressed(str(output_dir / "attention_full.npz"), **payload)
        log.info(f"Saved attention_full.npz  ({', '.join(payload)})")


def write_summary(
    all_results: Dict[str, Dict],
    cfg:         DictConfig,
    output_dir:  Path,
) -> None:
    lines = [
        "# TriLens-MoE-Stack v10 — Experiment Summary",
        "",
        "## Reference: CascadeStack v9.1",
        f"| v9.1 avg test PR-AUC | {V91_AVG:.4f} |",
        "",
        "---",
        "",
        "## Ablation Results (test set)",
        "",
        "| Ablation | Description | Avg PR-AUC | Δ vs v9.1 | Notes |",
        "|----------|-------------|-----------|-----------|-------|",
    ]

    valid = {k: v for k, v in all_results.items() if "error" not in v and "metrics" in v}
    for abl_id in ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]:
        res = valid.get(abl_id)
        if res is None:
            continue
        avg   = res["metrics"].get("_avg_prauc", np.nan)
        delta = avg - V91_AVG if not np.isnan(avg) else np.nan
        sig   = "***" if abs(delta) > 0.02 else ("**" if abs(delta) > 0.01 else ("*" if abs(delta) > 0.005 else ""))
        lines.append(
            f"| {abl_id} | {res['desc'][:40]} | "
            f"{avg:.4f} | {delta:+.4f} {sig} | {res.get('split_key','')} |"
        )

    # Per-AB breakdown for T6
    t6 = valid.get("T6")
    if t6:
        lines += ["", "## Per-AB Detail: T6 (v10 Stack) vs v9.1", "",
                  "| AB | v10 PR-AUC | v9.1 | Δ | CI |",
                  "|----|-----------|------|---|---|"]
        for ab in AB_NAMES:
            ab_res  = t6["metrics"].get(ab, {})
            prauc   = ab_res.get("prauc", np.nan)
            v91     = V91_PER_AB.get(ab, np.nan)
            delta   = prauc - v91 if not np.isnan(prauc) else np.nan
            ci_lo   = ab_res.get("ci_lo", np.nan)
            ci_hi   = ab_res.get("ci_hi", np.nan)
            lines.append(
                f"| {ab} | {prauc:.4f} | {v91:.4f} | {delta:+.4f} | "
                f"[{ci_lo:.3f}, {ci_hi:.3f}] |"
            )

    # Gate analysis for T5/T6
    gate_res = valid.get("T6") or valid.get("T5")
    if gate_res and gate_res.get("gate_test") is not None:
        gate_avg = gate_res["gate_test"].mean(axis=0)  # (7, 3) per AB per expert
        lines += ["", "## Expert Routing (gate weights, T6 test set)", "",
                  "| AB | Expert A (motif CNN) | Expert B (P/A) | Expert C (RMRNet) |",
                  "|----|---------------------|----------------|------------------|"]
        for ab_idx, ab in enumerate(AB_NAMES):
            gw = gate_avg[ab_idx]
            lines.append(
                f"| {ab} | {gw[0]:.3f} | {gw[1]:.3f} | {gw[2]:.3f} |"
            )

    # P/A lever check (T2)
    t2 = valid.get("T2")
    if t2:
        gen_prauc = t2["metrics"].get("GEN", {}).get("prauc", np.nan)
        lines += [
            "", "## P/A Lever (T2: Expert B alone) — GEN Critical Check",
            f"- GEN PR-AUC (T2 Expert B): **{gen_prauc:.4f}**",
            f"- GEN v9.1: {V91_PER_AB['GEN']:.4f}",
            f"- P/A lever {'CONFIRMED ✓' if gen_prauc > V91_PER_AB['GEN'] else 'not decisive'} for GEN",
        ]

    # Top-10 Expert B genes
    lines += [
        "", "## Top Expert B Genes (attention_B_topgenes.csv)",
        "These are candidate acquired-resistance markers to be cross-referenced with CARD/RGI.",
        "",
        "*See attention_B_topgenes.csv for full per-AB tables.*",
    ]

    # Verdict
    t6_avg = valid.get("T6", {}).get("metrics", {}).get("_avg_prauc", np.nan)
    lines += ["", "## Verdict", ""]
    if np.isnan(t6_avg):
        lines.append("T6 did not complete — no final verdict available.")
    elif t6_avg > V91_AVG + 0.01:
        lines.append(
            f"**STRONG SUCCESS:** v10 stack avg PR-AUC {t6_avg:.4f} beats v9.1 {V91_AVG:.4f} "
            f"by {t6_avg - V91_AVG:.4f} (>1%)."
        )
    elif t6_avg > V91_AVG + 0.001:
        lines.append(
            f"**PARTIAL SUCCESS:** v10 stack avg PR-AUC {t6_avg:.4f} beats v9.1 by "
            f"{t6_avg - V91_AVG:.4f}. Marginal but consistent with 35-45% prior."
        )
    elif t6_avg >= V91_AVG - 0.001:
        lines.append(
            f"**TIE:** v10 stack avg PR-AUC {t6_avg:.4f} ties v9.1 {V91_AVG:.4f}. "
            "Gate analysis and per-AB lens contribution remain publishable."
        )
    else:
        lines.append(
            f"**NULL RESULT:** v10 stack avg PR-AUC {t6_avg:.4f} does not beat v9.1. "
            "Honest reporting: neural additions did not improve the ensemble."
        )

    lines += ["", f"*Split: {cfg.split_key} | Seed: {cfg.training.seed} | N_folds: {cfg.n_folds}*"]

    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved SUMMARY.md")


# ─── Main Entry Point ──────────────────────────────────────────────────────────

def run(cfg: DictConfig) -> None:
    """Hydra entry point for TriLens v10."""
    if OmegaConf.select(cfg, "training") is None:
        try:
            cfg = cfg.experiment
        except Exception:
            pass
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    seed = int(cfg.training.seed)
    set_seed(seed)

    output_dir = Path(str(cfg.output_dir))
    cache_dir  = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    log.info(f"TriLens v10 output dir: {output_dir.resolve()}")

    try:
        (output_dir / "config_used.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(0)}  "
           f"{torch.cuda.get_device_properties(0).total_memory // 1024**3}GB)"
           if device.type == "cuda" else "")
    )

    # ── Load data ─────────────────────────────────────────────────────────
    log.info("─── Loading data ────────────────────────────────────────")
    data = load_all_data(cfg)

    split_key  = str(cfg.split_key)
    train_idx, _, _ = _get_split_indices(data, split_key)

    # ── Expert A blocks ───────────────────────────────────────────────────
    log.info("─── Expert A: per-gene SNP blocks ───────────────────────")
    X_blocks, M_blocks, col_indices = build_expert_a_blocks(
        data, cfg, cache_dir, train_idx
    )

    # Save column indices to output (reproducibility)
    cols_out = output_dir / "pergene_snp_columns.json"
    if not cols_out.exists():
        with open(cols_out, "w") as f:
            json.dump(col_indices, f)
        log.info(f"Saved pergene_snp_columns.json")

    # ── Expert C features ─────────────────────────────────────────────────
    log.info("─── Expert C: gene-aggregated features ─────────────────")
    X_gene_agg, feat_stats = build_expert_c_features(data, train_idx, cache_dir)

    # ── Ablation loop ─────────────────────────────────────────────────────
    ablations   = list(cfg.ablations)
    all_results: Dict[str, Dict] = {}

    log.info(f"─── Ablations: {ablations} ─────────────────────────────────")

    for abl_id in ablations:
        if abl_id not in ABLATION_DEFS:
            log.warning(f"Unknown ablation {abl_id}, skipping")
            continue

        log.info(f"\n{'='*66}\n  Ablation {abl_id}: {ABLATION_DEFS[abl_id]['desc']}\n{'='*66}")
        t0 = time.time()

        try:
            if abl_id in ("T6", "T7"):
                sk = "phylogeny_split" if abl_id == "T7" else split_key
                result = run_kfold_and_stack(
                    abl_id, cfg, data, X_blocks, M_blocks, X_gene_agg, sk, seed, device
                )
            else:
                result = run_standalone(
                    abl_id, cfg, data, X_blocks, M_blocks, X_gene_agg,
                    split_key, seed, device
                )
            result["runtime_min"] = round((time.time() - t0) / 60, 2)
            all_results[abl_id]   = result
            avg = result["metrics"].get("_avg_prauc", np.nan)
            log.info(f"  [{abl_id}] done  avg_prauc={avg:.4f}  runtime={result['runtime_min']:.1f}min")

        except Exception as exc:
            log.exception(f"  [{abl_id}] FAILED: {exc}")
            all_results[abl_id] = {
                "abl_id": abl_id, "error": str(exc),
                "desc": ABLATION_DEFS[abl_id]["desc"]
            }
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    # ── Save all outputs ──────────────────────────────────────────────────
    log.info("─── Saving outputs ──────────────────────────────────────")
    save_all_outputs(all_results, data, output_dir, cfg)
    write_summary(all_results, cfg, output_dir)

    # ── Final console summary ─────────────────────────────────────────────
    log.info("\n── FINAL SUMMARY ──────────────────────────────────────────────")
    for abl_id in ablations:
        res = all_results.get(abl_id, {})
        if "error" in res:
            log.info(f"  {abl_id:4s}  ERROR: {str(res['error'])[:60]}")
        else:
            avg = res.get("metrics", {}).get("_avg_prauc", np.nan)
            log.info(
                f"  {abl_id:4s}  avg={avg:.4f}  "
                f"(v9.1={V91_AVG:.4f}  Δ={avg-V91_AVG:+.4f})  "
                f"{res.get('split_key','')}  {res.get('runtime_min',0):.1f}min"
            )
    log.info("─────────────────────────────────────────────────────────────────")
