from __future__ import annotations

import gc
import json
import logging
import os
import pickle
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, matthews_corrcoef
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from trilens.models.trilens_network import AB_NAMES, TriLens

log = logging.getLogger(__name__)


ABLATION_DEFS: Dict[str, Dict[str, Any]] = {
    "motif_branch_only": {
        "use_a": True,
        "use_b": False,
        "use_c": False,
        "fusion": "single",
        "desc": "Expert A only (within-gene motif CNN)",
    },
    "presence_branch_only": {
        "use_a": False,
        "use_b": True,
        "use_c": False,
        "fusion": "single",
        "desc": "Expert B only (P/A sparse-attn) - P/A lever + GEN diagnostic",
    },
    "cross_antibiotic_branch_only": {
        "use_a": False,
        "use_b": False,
        "use_c": True,
        "fusion": "single",
        "desc": "Expert C only (structured RMRNet cross-AB)",
    },
    "branches_averaged": {
        "use_a": True,
        "use_b": True,
        "use_c": True,
        "fusion": "uniform",
        "desc": "All 3 experts, uniform average (no gating)",
    },
    "branches_gated": {
        "use_a": True,
        "use_b": True,
        "use_c": True,
        "fusion": "moe",
        "desc": "All 3 experts, MoE gate (full TriLens standalone)",
    },
    "full_stack": {
        "use_a": True,
        "use_b": True,
        "use_c": True,
        "fusion": "moe",
        "desc": "full stack: K-fold OOF + 5 classical bases + LogReg meta",
    },
    "full_stack_phylogroup": {
        "use_a": True,
        "use_b": True,
        "use_c": True,
        "fusion": "moe",
        "desc": "full stack on phylogeny_split (leakage-corrected)",
    },
}


CLASSICAL_REFERENCE_AVG = 0.8469
CLASSICAL_REFERENCE_PER_AB = {
    "AMP": 0.9412,
    "CIP": 0.9774,
    "AMC": 0.7569,
    "SXT": 0.8346,
    "GEN": 0.7037,
    "CTX": 0.8931,
    "CAZ": 0.8215,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _NoOpScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, opt):
        pass

    def step(self, opt):
        opt.step()

    def update(self):
        pass


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


def masked_bce_loss(
    logits: torch.Tensor,
    Y: torch.Tensor,
    mask: torch.Tensor,
    pos_weights: torch.Tensor,
) -> torch.Tensor:

    losses: List[torch.Tensor] = []
    for ab in range(Y.shape[1]):
        valid = mask[:, ab]
        if valid.sum() == 0:
            continue
        losses.append(F.binary_cross_entropy_with_logits(logits[valid, ab], Y[valid, ab], pos_weight=pos_weights[ab]))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def compute_pos_weights(Y: np.ndarray, mask: np.ndarray, device: torch.device) -> torch.Tensor:

    pw = []
    for ab in range(Y.shape[1]):
        valid = mask[:, ab]
        y_ab = Y[valid, ab]
        n_pos = float(y_ab.sum())
        n_neg = float(len(y_ab) - n_pos)
        pw.append(n_neg / max(n_pos, 1.0))
    return torch.tensor(pw, dtype=torch.float32, device=device)


class TriLensDataset(Dataset):
    def __init__(
        self,
        X_blocks: np.ndarray,
        X_pa: np.ndarray,
        X_gene_agg: np.ndarray,
        Y: np.ndarray,
        label_mask: np.ndarray,
    ) -> None:
        self.X_blocks = X_blocks
        self.X_pa = X_pa
        self.X_gene_agg = X_gene_agg
        self.Y = Y
        self.mask = label_mask

    def __len__(self) -> int:
        return len(self.Y)

    def __getitem__(self, idx: int) -> Tuple:
        return (
            torch.tensor(self.X_blocks[idx], dtype=torch.float32),
            torch.tensor(self.X_pa[idx], dtype=torch.float32),
            torch.from_numpy(self.X_gene_agg[idx]),
            torch.from_numpy(self.Y[idx]),
            torch.from_numpy(self.mask[idx]),
        )


def _build_dataset(
    indices: np.ndarray,
    X_blocks: np.ndarray,
    X_pa: np.ndarray,
    X_gene_agg: np.ndarray,
    Y: np.ndarray,
    label_mask: np.ndarray,
) -> TriLensDataset:

    return TriLensDataset(
        X_blocks=X_blocks[indices],
        X_pa=X_pa[indices],
        X_gene_agg=X_gene_agg[indices],
        Y=Y[indices],
        label_mask=label_mask[indices],
    )


def load_all_data(cfg: DictConfig) -> Dict[str, Any]:
    log.info(f"Loading gene_filtered pkl: {cfg.gene_filtered_pkl}")
    with open(cfg.gene_filtered_pkl, "rb") as f:
        gf = pickle.load(f)

    log.info(f"Loading gene P/A pkl: {cfg.gene_pa_pkl}")
    with open(cfg.gene_pa_pkl, "rb") as f:
        pa = pickle.load(f)

    X_snp = gf["X_snp_filtered"]
    gene_to_range = gf["gene_to_range"]
    selected_genes = gf["selected_gene_ids"]
    Y = gf["Y"].astype(np.float32)
    label_mask = gf["label_mask"].astype(bool)
    genome_names = gf["genome_names"]
    random_split = gf["random_split"]
    phylo_split = gf.get("phylogeny_split", random_split)

    X_pa = pa["X_pa"].astype(np.uint8)
    pa_gids = list(pa["gene_ids"])

    log.info(f"  X_snp={X_snp.shape}  X_pa={X_pa.shape}  Y={Y.shape}  selected_genes={len(selected_genes)}")

    pa_per_strain = X_pa.sum(axis=1)
    log.info(
        f"  P/A per strain: median={np.median(pa_per_strain):.0f}  "
        f"mean={pa_per_strain.mean():.0f}  min={pa_per_strain.min()}  max={pa_per_strain.max()}"
    )

    return {
        "X_snp": X_snp,
        "gene_to_range": gene_to_range,
        "selected_genes": selected_genes,
        "Y": Y,
        "label_mask": label_mask,
        "genome_names": genome_names,
        "X_pa": X_pa,
        "pa_gene_ids": pa_gids,
        "random_split": random_split,
        "phylogeny_split": phylo_split,
        "gene_filtered_pkl": str(cfg.gene_filtered_pkl),
    }


def _get_split_indices(data: Dict, split_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    split = data[split_key]
    train_idx = np.array(split.get("train_idx", split.get("train", [])))
    val_idx = np.array(split.get("val_idx", split.get("val", [])))
    test_idx = np.array(split.get("test_idx", split.get("test", [])))
    return train_idx, val_idx, test_idx


def build_expert_a_blocks(
    data: Dict,
    cfg: DictConfig,
    cache_dir: Path,
    train_idx: np.ndarray,
    split_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, Dict]:

    from trilens.data.gene_snp_blocks import build_pergene_snp_blocks, verify_blocks

    X_blocks, M_blocks, col_indices = build_pergene_snp_blocks(
        pkl_path=str(cfg.gene_filtered_pkl),
        cache_dir=str(cache_dir),
        L_max=int(cfg.expert_a.L_max),
        train_idx=train_idx,
        force_rebuild=False,
        split_name=split_name,
    )
    verify_blocks(X_blocks, M_blocks)
    return X_blocks, M_blocks, col_indices


def _build_gene_feat_inline(
    X_snp: np.ndarray,
    gene_to_range: Dict,
    selected_genes: List,
) -> np.ndarray:

    N = X_snp.shape[0]
    G = len(selected_genes)
    out = np.zeros((N, G, 4), dtype=np.float32)
    t0 = time.time()
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
            log.info(f"  gene_feat inline: {g + 1}/{G}  ({time.time() - t0:.0f}s)")
    return out


def build_expert_c_features(
    data: Dict,
    train_idx: np.ndarray,
    cache_dir: Path,
    split_name: str = "",
) -> Tuple[np.ndarray, Dict]:

    from trilens.data.gene_snp_blocks import (
        _load_and_check_identity,
        _save_identity,
        compute_cache_identity_hash,
    )

    feat_p = cache_dir / "gene_features_5000x4.npy"
    mean_p = cache_dir / "gene_feat_mean.npy"
    std_p = cache_dir / "gene_feat_std.npy"
    identity_p = cache_dir / "cache_identity.json"

    pkl_path = str(data.get("gene_filtered_pkl", ""))
    expected_hash = compute_cache_identity_hash(pkl_path, split_name, train_idx)
    cache_files_present = feat_p.exists() and mean_p.exists() and std_p.exists()
    identity_ok = cache_files_present and _load_and_check_identity(
        identity_p, expected_hash, pkl_path, split_name, train_idx
    )

    if identity_ok:
        X_feat = np.load(str(feat_p))
        feat_mean = np.load(str(mean_p))
        feat_std = np.load(str(std_p))
        log.info(f"Loaded Expert C features from cache: {feat_p}  shape={X_feat.shape}  (identity verified)")
        return X_feat, {"mean": feat_mean, "std": feat_std}

    for p in (feat_p, mean_p, std_p, identity_p):
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass

    log.info("Building Expert C gene-aggregated features (inline)…")
    t0 = time.time()

    try:
        from trilens.data.gene_statistics import build_gene_feature_matrix_from_pkl

        X_raw = build_gene_feature_matrix_from_pkl(data["gene_filtered_pkl"], stats=["mean", "var", "n_nonref", "max"])
        X_raw = X_raw.numpy() if hasattr(X_raw, "numpy") else np.asarray(X_raw, np.float32)
        log.info(f"  Built via trilens.data.gene_statistics in {time.time() - t0:.1f}s")
    except Exception as exc:
        log.warning(f"  Feature builder unavailable ({exc}); building inline…")
        X_raw = _build_gene_feat_inline(data["X_snp"], data["gene_to_range"], data["selected_genes"])
        log.info(f"  Built inline in {time.time() - t0:.1f}s")

    X_raw = np.asarray(X_raw, dtype=np.float32)

    flat = X_raw[train_idx].reshape(-1, 4)
    feat_mean = flat.mean(axis=0).astype(np.float32)
    feat_std = flat.std(axis=0).clip(min=1e-8).astype(np.float32)
    X_feat = ((X_raw - feat_mean) / feat_std).astype(np.float32)

    np.save(str(feat_p), X_feat)
    np.save(str(mean_p), feat_mean)
    np.save(str(std_p), feat_std)
    _save_identity(identity_p, expected_hash, pkl_path, split_name, train_idx)
    log.info(
        f"Saved Expert C features: {feat_p}  shape={X_feat.shape}  "
        f"mean={feat_mean.round(3).tolist()}  std={feat_std.round(3).tolist()}  "
        f"(identity fingerprint saved)"
    )
    return X_feat, {"mean": feat_mean, "std": feat_std}


def _train_epoch(
    model: TriLens,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    autocast_ctx,
    pos_weights: torch.Tensor,
    device: torch.device,
    cfg: DictConfig,
    beta: float,
) -> Dict[str, Any]:

    model.train()
    grad_accum = int(cfg.training.grad_accum)
    aux_w = float(cfg.aux_loss_weight)

    tot_loss = tot_res = tot_aux = tot_ent = 0.0
    gate_acc: Optional[torch.Tensor] = None
    n_updates = 0

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader):
        X_b, X_pa, X_gc, Y, mask = [t.to(device, non_blocking=True) for t in batch]
        is_last = step == len(loader) - 1

        with autocast_ctx:
            out = model(X_b, X_pa, X_gc)
            l_fin = out["l_final"]
            L_res = masked_bce_loss(l_fin, Y, mask, pos_weights)

            L_aux = torch.tensor(0.0, device=device)
            if model.n_active > 1:
                for key in ("l_A", "l_B", "l_C"):
                    lk = out.get(key)
                    if lk is not None:
                        L_aux = L_aux + masked_bce_loss(lk, Y, mask, pos_weights)
                L_aux = aux_w * L_aux

            L_ent = torch.tensor(0.0, device=device)
            if out.get("gate") is not None:
                gate = out["gate"]
                H = -(gate * (gate + 1e-8).log()).sum(dim=-1).mean()
                L_ent = -beta * H
                with torch.no_grad():
                    g_avg = gate.float().mean(dim=(0, 1)).cpu()
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
        tot_res += L_res.item()
        tot_aux += L_aux.item()
        tot_ent += L_ent.item()

    n = max(len(loader), 1)
    return {
        "total": tot_loss / n,
        "resistance": tot_res / n,
        "aux": tot_aux / n,
        "entropy": tot_ent / n,
        "gate_avg": gate_acc / n if gate_acc is not None else None,
    }


REPRESENTATION_KEYS = ("z_A", "z_B", "z_C", "z_fused")


def _representations(
    model: TriLens, dataset: TriLensDataset, device: torch.device, batch_size: int
) -> Dict[str, np.ndarray]:

    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    parts: Dict[str, list] = {k: [] for k in REPRESENTATION_KEYS}
    with torch.no_grad():
        for batch in loader:
            X_b, X_pa, X_gc, _, _ = [t.to(device, non_blocking=True) for t in batch]
            out = model(X_b, X_pa, X_gc)
            for k in REPRESENTATION_KEYS:
                if out.get(k) is not None:
                    parts[k].append(out[k].cpu().float().numpy())
    return {k: np.concatenate(v, axis=0) for k, v in parts.items() if v}


def save_representations(result: Dict, path: Path) -> bool:
    reps = result.get("representations")
    if not reps:
        return False
    payload = {
        "train_indices": result["train_idx"],
        "val_indices": result["val_idx"],
        "test_indices": result["test_idx"],
        "train_rep_indices": np.concatenate([reps["fit_idx"], reps["es_idx"], reps["oof_idx"]]),
        "fit_indices": reps["fit_idx"],
        "attention": np.array(os.environ.get("TRILENS_ATTENTION", "conditioned")),
        "fold": np.array(reps["fold"]),
        "fold_seed": np.array(reps["fold_seed"]),
    }
    for state in ("trained", "init"):
        r = reps[state]
        for k in REPRESENTATION_KEYS:
            if k in r["fit"]:
                payload[f"{state}_{k}_train"] = np.concatenate([r[p][k] for p in ("fit", "es", "oof")])
                payload[f"{state}_{k}_val"] = r["val"][k]
                payload[f"{state}_{k}_test"] = r["test"][k]
    np.savez_compressed(str(path), **payload)
    log.info(f"Saved {Path(path).name}  (last-fold representations, trained and untrained)")
    return True


@torch.no_grad()
def _predict(
    model: TriLens,
    dataset: TriLensDataset,
    device: torch.device,
    batch_size: int,
    collect_gate: bool = False,
    collect_attn: bool = False,
) -> Tuple:

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
    gates = np.concatenate(gate_l, axis=0) if gate_l else None
    a_A = np.concatenate(aA_l, axis=0) if aA_l else None
    a_B = np.concatenate(aB_l, axis=0) if aB_l else None
    a_C = np.concatenate(aC_l, axis=0) if aC_l else None
    return probs, gates, a_A, a_B, a_C


def _avg_prauc(probs: np.ndarray, Y: np.ndarray, mask: np.ndarray) -> float:
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
    model: TriLens,
    fit_ds: TriLensDataset,
    es_ds: TriLensDataset,
    pos_weights: torch.Tensor,
    cfg: DictConfig,
    device: torch.device,
) -> Tuple[TriLens, List[Dict]]:

    bs = int(cfg.training.batch_size)
    fit_loader = DataLoader(fit_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    autocast_ctx = _make_autocast(device.type, bool(cfg.training.use_amp))
    scaler = _make_scaler(device.type, bool(cfg.training.use_amp))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.lr),
        weight_decay=float(cfg.training.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg.training.max_epochs),
        eta_min=float(cfg.training.lr) * 0.01,
    )

    best_val = -np.inf
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    patience_ctr = 0
    beta = float(cfg.moe.entropy_beta)
    history = []
    eval_bs = bs * 2

    for epoch in range(int(cfg.training.max_epochs)):
        t_ep = time.time()
        stats = _train_epoch(
            model,
            fit_loader,
            optimizer,
            scaler,
            autocast_ctx,
            pos_weights,
            device,
            cfg,
            beta,
        )
        scheduler.step()

        gw = stats.get("gate_avg")
        if gw is not None and epoch >= 10:
            if gw.min().item() < float(cfg.moe.expert_collapse_threshold):
                log.warning(
                    f"  ep {epoch + 1}: expert collapse detected! "
                    f"gate_avg={gw.numpy().round(3).tolist()} → β ↑ to "
                    f"{cfg.moe.expert_collapse_beta_increase}"
                )
                beta = float(cfg.moe.expert_collapse_beta_increase)

        es_probs, _, _, _, _ = _predict(model, es_ds, device, eval_bs)
        val_avg = _avg_prauc(es_probs, es_ds.Y, es_ds.mask)

        ep_log = {
            "epoch": epoch + 1,
            "es_avg": val_avg,
            "runtime": round(time.time() - t_ep, 1),
            **{k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in stats.items()},
        }
        history.append(ep_log)

        log.info(
            f"  ep {epoch + 1:2d}/{cfg.training.max_epochs}  "
            f"loss={stats['total']:.4f} (res={stats['resistance']:.3f} "
            f"aux={stats['aux']:.3f} ent={stats['entropy']:.4f})  "
            f"es_avg={val_avg:.4f}  " + (f"gate={gw.numpy().round(3)}" if gw is not None else "")
        )

        if val_avg > best_val + 1e-5:
            best_val = val_avg
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= int(cfg.training.patience):
                log.info(f"  Early stop at epoch {epoch + 1}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    log.info(f"  Best es_avg = {best_val:.4f}")
    return model, history


def _create_model(
    cfg: DictConfig,
    M_blocks: np.ndarray,
    use_a: bool,
    use_b: bool,
    use_c: bool,
    fusion: str,
    device: torch.device,
) -> TriLens:
    gene_mask = torch.tensor(M_blocks, dtype=torch.bool)
    model = TriLens(cfg, gene_mask, use_a=use_a, use_b=use_b, use_c=use_c, fusion=fusion)
    model = model.to(device)
    log.info(f"  Model: {model.count_parameters():,} params  fusion={fusion}  experts={'ABC'[not use_a :: 1]}")
    return model


def _bootstrap_ci(y_true: np.ndarray, y_score: np.ndarray, n: int = 1000, seed: int = 42) -> Tuple[float, float, float]:
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan, np.nan
    point = float(average_precision_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    boot = []
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


def compute_metrics(probs: np.ndarray, Y: np.ndarray, mask: np.ndarray, cfg: DictConfig) -> Dict[str, Dict]:
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
        mcc = float(matthews_corrcoef(yt, (ys >= 0.5).astype(int)))
        n_b = int(cfg.evaluation.bootstrap_n)
        s_b = int(cfg.evaluation.bootstrap_seed)
        pt, lo, hi = _bootstrap_ci(yt, ys, n_b, s_b)
        results[ab] = {"prauc": prauc, "mcc": mcc, "ci_lo": lo, "ci_hi": hi}
    avg = float(np.nanmean([r["prauc"] for r in results.values()]))
    results["_avg_prauc"] = avg
    return results


def run_standalone(
    abl_id: str,
    cfg: DictConfig,
    data: Dict,
    X_blocks: np.ndarray,
    M_blocks: np.ndarray,
    X_gene_agg: np.ndarray,
    split_key: str,
    seed: int,
    device: torch.device,
) -> Dict:

    abl_def = ABLATION_DEFS[abl_id]
    use_a, use_b, use_c, fusion = (abl_def["use_a"], abl_def["use_b"], abl_def["use_c"], abl_def["fusion"])
    set_seed(seed)

    train_idx, val_idx, test_idx = _get_split_indices(data, split_key)
    pw = compute_pos_weights(data["Y"][train_idx], data["label_mask"][train_idx], device)

    train_ds = _build_dataset(train_idx, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
    val_ds = _build_dataset(val_idx, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
    test_ds = _build_dataset(test_idx, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])

    model, history = _train_until_convergence(
        _create_model(cfg, M_blocks, use_a, use_b, use_c, fusion, device),
        train_ds,
        val_ds,
        pw,
        cfg,
        device,
    )

    bs = int(cfg.training.batch_size) * 2
    test_probs, test_gates, a_A, a_B, a_C = _predict(model, test_ds, device, bs, collect_gate=True, collect_attn=True)
    val_probs, _, _, _, _ = _predict(model, val_ds, device, bs)

    metrics = compute_metrics(test_probs, data["Y"][test_idx], data["label_mask"][test_idx], cfg)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "abl_id": abl_id,
        "desc": abl_def["desc"],
        "mode": "standalone",
        "split_key": split_key,
        "metrics": metrics,
        "test_probs": test_probs,
        "val_probs": val_probs,
        "gate_test": test_gates,
        "a_A": a_A,
        "a_B": a_B,
        "a_C": a_C,
        "history": history,
        "test_idx": test_idx,
        "val_idx": val_idx,
        "train_idx": train_idx,
    }


def _fold_path(fold_dir, split_key, fold):
    return Path(fold_dir) / f"{split_key}_fold{fold}.npz"


def _pa_signature(X_pa) -> str:
    import hashlib

    if X_pa is None:
        return "none"
    cnt = np.asarray(X_pa).sum(axis=0, dtype=np.int64)
    return hashlib.md5(np.ascontiguousarray(cnt).tobytes()).hexdigest()


def _save_fold(
    fold_dir,
    split_key,
    fold,
    train_idx,
    oof_local,
    fold_seed,
    run_tag,
    last,
    oof_p,
    val_p,
    tst_p,
    tst_g,
    hist,
    aA,
    aB,
    aC,
):

    if fold_dir is None:
        return
    p = _fold_path(fold_dir, split_key, fold)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        train_idx=np.asarray(train_idx),
        oof_local=np.asarray(oof_local),
        fold_seed=np.array(fold_seed),
        lensb=np.array(run_tag[0]),
        pa_sig=np.array(run_tag[1]),
        oof=oof_p,
        val=val_p,
        test=tst_p,
        history=np.array(json.dumps(hist, default=float)),
    )
    if tst_g is not None:
        arrays["gate"] = tst_g
    if last:
        for k, v in (("a_A", aA), ("a_B", aB), ("a_C", aC)):
            if v is not None:
                arrays[k] = v
    tmp = p.with_name(p.stem + ".part.npz")
    np.savez_compressed(str(tmp), **arrays)
    tmp.replace(p)


def _load_fold(fold_dir, split_key, fold, train_idx, oof_local, fold_seed, run_tag, last):
    if fold_dir is None or not _fold_path(fold_dir, split_key, fold).exists():
        return None
    z = np.load(str(_fold_path(fold_dir, split_key, fold)), allow_pickle=False)
    same = (
        np.array_equal(z["train_idx"], np.asarray(train_idx))
        and np.array_equal(z["oof_local"], np.asarray(oof_local))
        and int(z["fold_seed"]) == int(fold_seed)
        and str(z["lensb"]) == run_tag[0]
        and "pa_sig" in z.files
        and str(z["pa_sig"]) == run_tag[1]
    )
    if not same:
        log.warning(f"  saved fold {fold} in {fold_dir} is from a different run; training it again")
        return None

    def get(k):
        return z[k] if k in z.files else None

    if last and get("a_A") is None and get("a_B") is None:
        return None
    return dict(
        oof=z["oof"],
        val=z["val"],
        test=z["test"],
        gate=get("gate"),
        history=json.loads(str(z["history"])),
        a_A=get("a_A"),
        a_B=get("a_B"),
        a_C=get("a_C"),
    )


def run_kfold_oof(
    cfg: DictConfig,
    data: Dict,
    X_blocks: np.ndarray,
    M_blocks: np.ndarray,
    X_gene_agg: np.ndarray,
    split_key: str,
    seed: int,
    device: torch.device,
    fold_dir: Optional[Path] = None,
) -> Dict:

    train_idx, val_idx, test_idx = _get_split_indices(data, split_key)
    n_train = len(train_idx)
    n_folds = int(cfg.n_folds)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(cfg.kfold_seed))

    pw = compute_pos_weights(data["Y"][train_idx], data["label_mask"][train_idx], device)

    oof_preds = np.zeros((n_train, len(AB_NAMES)), dtype=np.float32)
    val_preds_f = []
    test_preds_f = []
    gate_test_f = []
    histories = []
    representations = None

    val_ds = _build_dataset(val_idx, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
    test_ds = _build_dataset(test_idx, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])

    run_tag = (os.environ.get("TRILENS_ATTENTION", "conditioned"), _pa_signature(data.get("X_pa")))
    aA = aB = aC = None
    for fold, (fit_es_local, oof_local) in enumerate(kf.split(np.arange(n_train))):
        fold_seed = seed + fold * 100
        last = fold == n_folds - 1
        saved = _load_fold(fold_dir, split_key, fold, train_idx, oof_local, fold_seed, run_tag, last)
        rep_path = _fold_path(fold_dir, split_key, fold).with_suffix(".representations.pkl") if fold_dir else None
        if saved is not None and last:
            if rep_path is not None and rep_path.exists():
                with open(rep_path, "rb") as f:
                    representations = pickle.load(f)
            else:
                saved = None
        if saved is not None:
            log.info(f"  [Fold {fold + 1}/{n_folds}] reused from {fold_dir}")
            oof_preds[oof_local] = saved["oof"]
            val_preds_f.append(saved["val"])
            test_preds_f.append(saved["test"])
            if saved["gate"] is not None:
                gate_test_f.append(saved["gate"])
            histories.append(saved["history"])
            if last:
                aA, aB, aC = saved["a_A"], saved["a_B"], saved["a_C"]
            continue
        set_seed(fold_seed)
        log.info(f"\n{'─' * 60}\n  [Fold {fold + 1}/{n_folds}]  seed={fold_seed}")

        fit_es_local = np.random.RandomState(fold_seed).permutation(fit_es_local)
        n_fe = len(fit_es_local)
        n_es = max(int(n_fe * 0.1), 50)
        fit_l = fit_es_local[: n_fe - n_es]
        es_l = fit_es_local[n_fe - n_es :]

        fit_s = train_idx[fit_l]
        es_s = train_idx[es_l]
        oof_s = train_idx[oof_local]
        log.info(f"  fit={len(fit_s)}  es={len(es_s)}  oof={len(oof_s)}")

        fit_ds = _build_dataset(fit_s, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
        es_ds = _build_dataset(es_s, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
        oof_ds = _build_dataset(oof_s, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])

        model = _create_model(cfg, M_blocks, use_a=True, use_b=True, use_c=True, fusion="moe", device=device)
        bs = int(cfg.training.batch_size) * 2
        last_fold = fold == n_folds - 1

        rep_parts = (("fit", fit_ds), ("es", es_ds), ("oof", oof_ds), ("val", val_ds), ("test", test_ds))
        if last_fold:
            with torch.random.fork_rng(devices=[]):
                rep_init = {part: _representations(model, ds, device, bs) for part, ds in rep_parts}
        model, hist = _train_until_convergence(model, fit_ds, es_ds, pw, cfg, device)
        histories.append(hist)
        if last_fold:
            representations = {
                "fold": fold,
                "fold_seed": fold_seed,
                "fit_idx": fit_s,
                "es_idx": es_s,
                "oof_idx": oof_s,
                "init": rep_init,
                "trained": {part: _representations(model, ds, device, bs) for part, ds in rep_parts},
            }

        oof_p, _, _, _, _ = _predict(model, oof_ds, device, bs)
        val_p, _, _, _, _ = _predict(model, val_ds, device, bs)
        tst_p, tst_g, aA, aB, aC = _predict(model, test_ds, device, bs, collect_gate=True, collect_attn=True)

        oof_preds[oof_local] = oof_p
        val_preds_f.append(val_p)
        test_preds_f.append(tst_p)
        if tst_g is not None:
            gate_test_f.append(tst_g)
        if last and rep_path is not None and representations is not None:
            rep_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = rep_path.with_name(rep_path.name + ".part")
            with open(tmp, "wb") as f:
                pickle.dump(representations, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(rep_path)
        _save_fold(
            fold_dir,
            split_key,
            fold,
            train_idx,
            oof_local,
            fold_seed,
            run_tag,
            last,
            oof_p,
            val_p,
            tst_p,
            tst_g,
            hist,
            aA,
            aB,
            aC,
        )

        fold_metrics = compute_metrics(tst_p, data["Y"][test_idx], data["label_mask"][test_idx], cfg)
        log.info(
            f"  Fold {fold + 1} test avg = {fold_metrics['_avg_prauc']:.4f}  "
            + "  ".join(f"{ab}={fold_metrics.get(ab, {}).get('prauc', 0):.3f}" for ab in AB_NAMES)
        )

        del model, fit_ds, es_ds, oof_ds
        torch.cuda.empty_cache()
        gc.collect()

    val_preds = np.mean(val_preds_f, axis=0)
    test_preds = np.mean(test_preds_f, axis=0)
    gate_test = np.mean(gate_test_f, axis=0) if gate_test_f else None

    return {
        "oof_preds": oof_preds,
        "val_preds": val_preds,
        "test_preds": test_preds,
        "gate_test": gate_test,
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "histories": histories,
        "a_A": aA,
        "a_B": aB,
        "a_C": aC,
        "representations": representations,
    }


def run_meta_stack(
    oof_trilens: np.ndarray,
    test_trilens: np.ndarray,
    oof_classical: Optional[np.ndarray],
    test_classical: Optional[np.ndarray],
    Y_tr: np.ndarray,
    mask_tr: np.ndarray,
    cfg: DictConfig,
    val_trilens: Optional[np.ndarray] = None,
    val_classical: Optional[np.ndarray] = None,
    return_val: bool = False,
):

    n_test = len(test_trilens)
    n_abs = len(AB_NAMES)
    test_meta = np.full((n_test, n_abs), np.nan, dtype=np.float32)
    has_val = val_trilens is not None and (oof_classical is None or val_classical is not None)
    val_meta = np.full((len(val_trilens), n_abs), np.nan, dtype=np.float32) if has_val else None
    meta_weights: Dict[str, Dict] = {}
    C = float(cfg.meta_C)

    classical_names = ["XGB_gene", "LGBM", "RF", "LogReg", "XGB_raw"]

    for ab_idx, ab in enumerate(AB_NAMES):
        valid_tr = mask_tr[:, ab_idx]
        if valid_tr.sum() < 2:
            test_meta[:, ab_idx] = test_trilens[:, ab_idx]
            if has_val:
                val_meta[:, ab_idx] = val_trilens[:, ab_idx]
            continue

        if oof_classical is not None:
            X_meta_tr = np.column_stack(
                [
                    oof_classical[valid_tr, :, ab_idx],
                    oof_trilens[valid_tr, ab_idx : ab_idx + 1],
                ]
            )
        else:
            X_meta_tr = oof_trilens[valid_tr, ab_idx : ab_idx + 1]

        y_meta_tr = Y_tr[valid_tr, ab_idx]

        if y_meta_tr.sum() == 0 or y_meta_tr.sum() == valid_tr.sum():
            log.warning(f"  Meta {ab}: single-class train split - using TriLens probs")
            test_meta[:, ab_idx] = test_trilens[:, ab_idx]
            if has_val:
                val_meta[:, ab_idx] = val_trilens[:, ab_idx]
            continue

        lr = LogisticRegression(C=C, max_iter=1000, solver="lbfgs", random_state=42)
        lr.fit(X_meta_tr, y_meta_tr)

        if test_classical is not None:
            X_meta_te = np.column_stack(
                [
                    test_classical[:, :, ab_idx],
                    test_trilens[:, ab_idx : ab_idx + 1],
                ]
            )
        else:
            X_meta_te = test_trilens[:, ab_idx : ab_idx + 1]

        test_meta[:, ab_idx] = lr.predict_proba(X_meta_te)[:, 1]
        if has_val:
            X_meta_va = (
                np.column_stack([val_classical[:, :, ab_idx], val_trilens[:, ab_idx : ab_idx + 1]])
                if oof_classical is not None
                else val_trilens[:, ab_idx : ab_idx + 1]
            )
            val_meta[:, ab_idx] = lr.predict_proba(X_meta_va)[:, 1]
        meta_weights[ab] = {
            "bases": (classical_names + ["TriLens"]) if oof_classical is not None else ["TriLens"],
            "coef": lr.coef_[0].tolist(),
            "intercept": float(lr.intercept_[0]),
        }
        log.info(f"  Meta {ab}: bases={len(lr.coef_[0])}  coef={np.array(lr.coef_[0]).round(3).tolist()}")

    if return_val:
        return test_meta, meta_weights, val_meta
    return test_meta, meta_weights


def _load_classical_arm(classical_path: Path, train_idx, val_idx, test_idx):
    classical_npz = np.load(str(classical_path), allow_pickle=False)

    oof_classical = classical_npz["oof_base_probs"][:, :5, :]
    test_classical = classical_npz["test_base_probs"][:, :5, :]
    val_classical = None
    if "val_base_probs" in classical_npz.files:
        val_classical = classical_npz["val_base_probs"][:, :5, :]

    keys = [("train_indices", train_idx), ("test_indices", test_idx)]
    if val_classical is not None:
        keys.append(("val_indices", val_idx))
    for key, idx in keys:
        if key not in classical_npz.files:
            raise ValueError(f"{classical_path}: no {key}, so the row order of the classical arm cannot be checked")
        if not np.array_equal(np.asarray(classical_npz[key]).ravel(), np.asarray(idx).ravel()):
            raise ValueError(
                f"{classical_path}: {key} differ from this run's split - "
                "the classical arm was built on another split or row order"
            )
    log.info(f"Loaded classical stack classical OOF from {classical_path} (split indices match)")
    return oof_classical, test_classical, val_classical


def run_kfold_and_stack(
    abl_id: str,
    cfg: DictConfig,
    data: Dict,
    X_blocks: np.ndarray,
    M_blocks: np.ndarray,
    X_gene_agg: np.ndarray,
    split_key: str,
    seed: int,
    device: torch.device,
    fold_dir: Optional[Path] = None,
) -> Dict:

    log.info(f"K-fold OOF ({split_key}) …")
    kfold = run_kfold_oof(cfg, data, X_blocks, M_blocks, X_gene_agg, split_key, seed, device, fold_dir=fold_dir)

    train_idx = kfold["train_idx"]
    val_idx = kfold["val_idx"]
    test_idx = kfold["test_idx"]

    classical_path = Path(str(cfg.classical_predictions))
    oof_classical = None
    test_classical = None
    val_classical = None
    if classical_path.exists():
        oof_classical, test_classical, val_classical = _load_classical_arm(classical_path, train_idx, val_idx, test_idx)
    elif classical_path.name.startswith("_intentionally_absent"):
        log.info("Neural arm only (the classical arm is stacked later by the diagnose_* script).")
    else:
        raise FileNotFoundError(
            f"classical-arm predictions not found at {classical_path}. Run experiment=classical_stack_random first."
        )

    log.info("Running meta-stacking …")
    test_meta, meta_weights, val_meta = run_meta_stack(
        oof_trilens=kfold["oof_preds"],
        test_trilens=kfold["test_preds"],
        oof_classical=oof_classical,
        test_classical=test_classical,
        Y_tr=data["Y"][train_idx],
        mask_tr=data["label_mask"][train_idx],
        cfg=cfg,
        val_trilens=kfold["val_preds"],
        val_classical=val_classical,
        return_val=True,
    )
    val_is_stack = val_meta is not None and (oof_classical is None or val_classical is not None)
    if val_meta is None:
        log.warning(
            "No classical validation probabilities: val_probs stays TriLens-only; do not tune thresholds on it."
        )
        val_meta = kfold["val_preds"]

    metrics = compute_metrics(test_meta, data["Y"][test_idx], data["label_mask"][test_idx], cfg)
    log.info(
        f"[{abl_id}] Stack avg test PR-AUC = {metrics['_avg_prauc']:.4f}  "
        + "  ".join(f"{ab}={metrics.get(ab, {}).get('prauc', 0):.3f}" for ab in AB_NAMES)
    )

    n_base = 6 if oof_classical is not None else 1
    oof_base_probs = np.zeros((len(train_idx), n_base, len(AB_NAMES)), np.float32)
    test_base_probs = np.zeros((len(test_idx), n_base, len(AB_NAMES)), np.float32)
    if oof_classical is not None:
        oof_base_probs[:, :5, :] = oof_classical
        test_base_probs[:, :5, :] = test_classical
    oof_base_probs[:, -1, :] = kfold["oof_preds"]
    test_base_probs[:, -1, :] = kfold["test_preds"]

    base_names = (
        ["XGB_gene", "LGBM", "RF", "LogReg", "XGB_raw", "TriLens"] if oof_classical is not None else ["TriLens"]
    )

    return {
        "abl_id": abl_id,
        "desc": ABLATION_DEFS[abl_id]["desc"],
        "mode": "kfold_stack",
        "split_key": split_key,
        "metrics": metrics,
        "test_probs": test_meta,
        "val_probs": val_meta,
        "val_trilens": kfold["val_preds"],
        "val_is_stack": bool(val_is_stack),
        "gate_test": kfold["gate_test"],
        "a_A": kfold.get("a_A"),
        "a_B": kfold.get("a_B"),
        "a_C": kfold.get("a_C"),
        "representations": kfold.get("representations"),
        "history": kfold["histories"],
        "meta_weights": meta_weights,
        "base_names": base_names,
        "oof_base_probs": oof_base_probs,
        "test_base_probs": test_base_probs,
        "oof_trilens": kfold["oof_preds"],
        "test_trilens": kfold["test_preds"],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


def compute_topgenes(
    attn: np.ndarray,
    Y_te: np.ndarray,
    mask_te: np.ndarray,
    gene_ids: List[str],
    n_top: int = 20,
) -> pd.DataFrame:
    rows = []
    for ab_idx, ab in enumerate(AB_NAMES):
        valid = mask_te[:, ab_idx]
        if valid.sum() < 2 or attn is None:
            continue
        y_ab = Y_te[valid, ab_idx]
        a_ab = attn[valid]
        res = y_ab == 1
        avg = a_ab[res].mean(axis=0) if res.sum() > 0 else a_ab.mean(axis=0)
        top = np.argsort(avg)[::-1][:n_top]
        for rank, gi in enumerate(top):
            if gi < len(gene_ids):
                rows.append(
                    {
                        "ab": ab,
                        "rank": rank + 1,
                        "gene_id": str(gene_ids[gi]),
                        "attn_weight": float(avg[gi]),
                        "n_resistant": int(res.sum()),
                    }
                )
    return pd.DataFrame(rows)


def save_all_outputs(
    all_results: Dict[str, Dict],
    data: Dict,
    output_dir: Path,
    cfg: DictConfig,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for abl_id, res in all_results.items():
        if "error" in res:
            continue
        for ab, ab_res in res.get("metrics", {}).items():
            if ab.startswith("_"):
                continue
            rows.append(
                {
                    "ablation": abl_id,
                    "desc": res.get("desc", "")[:50],
                    "split_key": res.get("split_key", ""),
                    "ab": ab,
                    "prauc": ab_res.get("prauc", np.nan),
                    "mcc": ab_res.get("mcc", np.nan),
                    "ci_lo": ab_res.get("ci_lo", np.nan),
                    "ci_hi": ab_res.get("ci_hi", np.nan),
                    "classical_reference_prauc": CLASSICAL_REFERENCE_PER_AB.get(ab, np.nan),
                    "delta_classical_reference": ab_res.get("prauc", np.nan)
                    - CLASSICAL_REFERENCE_PER_AB.get(ab, np.nan),
                }
            )
    if rows:
        pd.DataFrame(rows).to_csv(output_dir / "ablation_results.csv", index=False)
        log.info("Saved ablation_results.csv")

    best_abl = next(
        (
            k
            for k in (
                "full_stack",
                "branches_gated",
                "branches_averaged",
                "cross_antibiotic_branch_only",
                "presence_branch_only",
                "motif_branch_only",
            )
            if k in all_results and "error" not in all_results[k]
        ),
        None,
    )
    if best_abl:
        res = all_results[best_abl]
        te_idx = res.get("test_idx", np.array([]))
        Y_te = data["Y"][te_idx] if len(te_idx) > 0 else data["Y"][:1]
        mask_te = data["label_mask"][te_idx] if len(te_idx) > 0 else data["label_mask"][:1]

        for expert_key, gene_ids, out_name in [
            ("a_A", data["selected_genes"], "attention_A_topgenes.csv"),
            ("a_B", data["pa_gene_ids"], "attention_B_topgenes.csv"),
            ("a_C", data["selected_genes"], "attention_C_topgenes.csv"),
        ]:
            attn = res.get(expert_key)
            if attn is not None:
                df = compute_topgenes(attn, Y_te, mask_te, gene_ids)
                df.to_csv(output_dir / out_name, index=False)
                log.info(f"Saved {out_name}  ({len(df)} rows)")

    gate_res = next(
        (
            all_results[k]
            for k in ("full_stack", "branches_gated")
            if k in all_results and "error" not in all_results[k] and all_results[k].get("gate_test") is not None
        ),
        None,
    )
    if gate_res is not None and gate_res.get("gate_test") is not None:
        np.savez(str(output_dir / "gate_weights_test.npz"), gate=gate_res["gate_test"])
        log.info(f"Saved gate_weights_test.npz  shape={gate_res['gate_test'].shape}")

    meta_rows = []
    for abl_id in ("full_stack", "full_stack_phylogroup"):
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

    split_rows = []
    for abl_id in ("full_stack", "full_stack_phylogroup"):
        res = all_results.get(abl_id)
        if res and "error" not in res:
            for ab, ab_res in res.get("metrics", {}).items():
                if ab.startswith("_"):
                    continue
                split_rows.append(
                    {
                        "ablation": abl_id,
                        "split_key": res.get("split_key", ""),
                        "ab": ab,
                        "prauc": ab_res.get("prauc", np.nan),
                        "ci_lo": ab_res.get("ci_lo", np.nan),
                        "ci_hi": ab_res.get("ci_hi", np.nan),
                    }
                )
    if split_rows:
        pd.DataFrame(split_rows).to_csv(output_dir / "split_comparison.csv", index=False)
        log.info("Saved split_comparison.csv")

    full_stack_result = all_results.get("full_stack") or all_results.get("full_stack_phylogroup")
    if full_stack_result and "error" not in full_stack_result and "oof_trilens" in full_stack_result:
        np.savez(
            str(output_dir / "trilens_kfold_oof.npz"),
            oof_preds=full_stack_result["oof_trilens"],
            val_preds=full_stack_result.get("val_trilens", full_stack_result["val_probs"]),
            test_preds=full_stack_result["test_trilens"],
            train_idx=full_stack_result["train_idx"],
            val_idx=full_stack_result["val_idx"],
            test_idx=full_stack_result["test_idx"],
        )
        log.info("Saved trilens_kfold_oof.npz")

    full_stack_res = all_results.get("full_stack")
    if full_stack_res and "error" not in full_stack_res and "oof_base_probs" in full_stack_res:
        np.savez(
            str(output_dir / "predictions_saved.npz"),
            test_ens_probs=full_stack_res["test_probs"],
            test_base_probs=full_stack_res["test_base_probs"],
            oof_base_probs=full_stack_res["oof_base_probs"],
            val_ens_probs=full_stack_res.get("val_probs", np.zeros((0, len(AB_NAMES)))),
            test_indices=full_stack_res["test_idx"],
            val_indices=full_stack_res["val_idx"],
            train_indices=full_stack_res["train_idx"],
            base_names=np.array(full_stack_res.get("base_names", ["TriLens"])),
            model_name=np.array("TriLens-full-stack"),
            val_is_stack=np.array(bool(full_stack_res.get("val_is_stack", False))),
        )
        log.info("Saved predictions_saved.npz  (classical-stack-compatible format)")

    if full_stack_res and "error" not in full_stack_res:
        save_representations(full_stack_res, output_dir / "representations.npz")

    attn_res = next(
        (
            all_results[k]
            for k in ("full_stack", "branches_gated")
            if k in all_results and "error" not in all_results[k] and all_results[k].get("a_B") is not None
        ),
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
    cfg: DictConfig,
    output_dir: Path,
) -> None:
    lines = [
        "# TriLens stack - Experiment Summary",
        "",
        "## Reference: the classical stack",
        f"| classical stack avg test PR-AUC | {CLASSICAL_REFERENCE_AVG:.4f} |",
        "",
        "---",
        "",
        "## Ablation Results (test set)",
        "",
        "| Ablation | Description | Avg PR-AUC | Δ vs classical stack | Notes |",
        "|----------|-------------|-----------|-----------|-------|",
    ]

    valid = {k: v for k, v in all_results.items() if "error" not in v and "metrics" in v}
    for abl_id in [
        "motif_branch_only",
        "presence_branch_only",
        "cross_antibiotic_branch_only",
        "branches_averaged",
        "branches_gated",
        "full_stack",
        "full_stack_phylogroup",
    ]:
        res = valid.get(abl_id)
        if res is None:
            continue
        avg = res["metrics"].get("_avg_prauc", np.nan)
        delta = avg - CLASSICAL_REFERENCE_AVG if not np.isnan(avg) else np.nan
        sig = "***" if abs(delta) > 0.02 else ("**" if abs(delta) > 0.01 else ("*" if abs(delta) > 0.005 else ""))
        lines.append(f"| {abl_id} | {res['desc'][:40]} | {avg:.4f} | {delta:+.4f} {sig} | {res.get('split_key', '')} |")

    full_stack_result = valid.get("full_stack")
    if full_stack_result:
        lines += [
            "",
            "## Per-AB Detail: full_stack (full stack) vs classical stack",
            "",
            "| AB | stack PR-AUC | classical stack | Δ | CI |",
            "|----|-----------|------|---|---|",
        ]
        for ab in AB_NAMES:
            ab_res = full_stack_result["metrics"].get(ab, {})
            prauc = ab_res.get("prauc", np.nan)
            reference = CLASSICAL_REFERENCE_PER_AB.get(ab, np.nan)
            delta = prauc - reference if not np.isnan(prauc) else np.nan
            ci_lo = ab_res.get("ci_lo", np.nan)
            ci_hi = ab_res.get("ci_hi", np.nan)
            lines.append(f"| {ab} | {prauc:.4f} | {reference:.4f} | {delta:+.4f} | [{ci_lo:.3f}, {ci_hi:.3f}] |")

    gate_res = valid.get("full_stack") or valid.get("branches_gated")
    if gate_res and gate_res.get("gate_test") is not None:
        gate_avg = gate_res["gate_test"].mean(axis=0)
        lines += [
            "",
            "## Expert Routing (gate weights, full_stack test set)",
            "",
            "| AB | Expert A (motif CNN) | Expert B (P/A) | Expert C (RMRNet) |",
            "|----|---------------------|----------------|------------------|",
        ]
        for ab_idx, ab in enumerate(AB_NAMES):
            gw = gate_avg[ab_idx]
            lines.append(f"| {ab} | {gw[0]:.3f} | {gw[1]:.3f} | {gw[2]:.3f} |")

    t2 = valid.get("presence_branch_only")
    if t2:
        gen_prauc = t2["metrics"].get("GEN", {}).get("prauc", np.nan)
        lines += [
            "",
            "## P/A Lever (presence_branch_only: Expert B alone) - GEN Critical Check",
            f"- GEN PR-AUC (presence_branch_only Expert B): **{gen_prauc:.4f}**",
            f"- GEN classical stack: {CLASSICAL_REFERENCE_PER_AB['GEN']:.4f}",
            f"- P/A lever {'CONFIRMED ✓' if gen_prauc > CLASSICAL_REFERENCE_PER_AB['GEN'] else 'not decisive'} for GEN",
        ]

    lines += [
        "",
        "## Top Expert B Genes (attention_B_topgenes.csv)",
        "These are candidate acquired-resistance markers to be cross-referenced with CARD/RGI.",
        "",
        "*See attention_B_topgenes.csv for full per-AB tables.*",
    ]

    full_stack_avg = valid.get("full_stack", {}).get("metrics", {}).get("_avg_prauc", np.nan)
    lines += ["", "## Verdict", ""]
    if np.isnan(full_stack_avg):
        lines.append("full_stack did not complete - no final verdict available.")
    elif full_stack_avg > CLASSICAL_REFERENCE_AVG + 0.01:
        lines.append(
            f"**STRONG SUCCESS:** full stack avg PR-AUC {full_stack_avg:.4f} beats "
            f"classical stack {CLASSICAL_REFERENCE_AVG:.4f} "
            f"by {full_stack_avg - CLASSICAL_REFERENCE_AVG:.4f} (>1%)."
        )
    elif full_stack_avg > CLASSICAL_REFERENCE_AVG + 0.001:
        lines.append(
            f"**PARTIAL SUCCESS:** full stack avg PR-AUC {full_stack_avg:.4f} beats classical stack by "
            f"{full_stack_avg - CLASSICAL_REFERENCE_AVG:.4f}. Marginal but consistent with 35-45% prior."
        )
    elif full_stack_avg >= CLASSICAL_REFERENCE_AVG - 0.001:
        lines.append(
            f"**TIE:** full stack avg PR-AUC {full_stack_avg:.4f} ties classical stack {CLASSICAL_REFERENCE_AVG:.4f}. "
            "Gate analysis and per-AB lens contribution remain publishable."
        )
    else:
        lines.append(
            f"**NULL RESULT:** full stack avg PR-AUC {full_stack_avg:.4f} does not beat classical stack. "
            "The neural additions did not improve the ensemble."
        )

    lines += ["", f"*Split: {cfg.split_key} | Seed: {cfg.training.seed} | N_folds: {cfg.n_folds}*"]

    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved SUMMARY.md")


def run(cfg: DictConfig) -> None:
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
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(exist_ok=True)
    log.info(f"TriLens output dir: {output_dir.resolve()}")

    try:
        (output_dir / "config_used.yaml").write_text(OmegaConf.to_yaml(cfg), encoding="utf-8")
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        f"Device: {device}"
        + (
            f" ({torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB)"
            if device.type == "cuda"
            else ""
        )
    )

    log.info("─── Loading data ────────────────────────────────────────")
    data = load_all_data(cfg)

    split_key = str(cfg.split_key)
    train_idx, _, _ = _get_split_indices(data, split_key)

    log.info("─── Expert A: per-gene SNP blocks ───────────────────────")
    X_blocks, M_blocks, col_indices = build_expert_a_blocks(data, cfg, cache_dir, train_idx)

    cols_out = output_dir / "pergene_snp_columns.json"
    if not cols_out.exists():
        with open(cols_out, "w") as f:
            json.dump(col_indices, f)
        log.info("Saved pergene_snp_columns.json")

    log.info("─── Expert C: gene-aggregated features ─────────────────")
    X_gene_agg, feat_stats = build_expert_c_features(data, train_idx, cache_dir)

    ablations = list(cfg.ablations)
    all_results: Dict[str, Dict] = {}

    log.info(f"─── Ablations: {ablations} ─────────────────────────────────")

    for abl_id in ablations:
        if abl_id not in ABLATION_DEFS:
            log.warning(f"Unknown ablation {abl_id}, skipping")
            continue

        log.info(f"\n{'=' * 66}\n  Ablation {abl_id}: {ABLATION_DEFS[abl_id]['desc']}\n{'=' * 66}")
        t0 = time.time()

        try:
            if abl_id in ("full_stack", "full_stack_phylogroup"):
                sk = "phylogeny_split" if abl_id == "full_stack_phylogroup" else split_key
                result = run_kfold_and_stack(
                    abl_id,
                    cfg,
                    data,
                    X_blocks,
                    M_blocks,
                    X_gene_agg,
                    sk,
                    seed,
                    device,
                    fold_dir=output_dir / "folds",
                )
            else:
                result = run_standalone(abl_id, cfg, data, X_blocks, M_blocks, X_gene_agg, split_key, seed, device)
            result["runtime_min"] = round((time.time() - t0) / 60, 2)
            all_results[abl_id] = result
            avg = result["metrics"].get("_avg_prauc", np.nan)
            log.info(f"  [{abl_id}] done  avg_prauc={avg:.4f}  runtime={result['runtime_min']:.1f}min")

        except Exception as exc:
            log.exception(f"  [{abl_id}] FAILED: {exc}")
            all_results[abl_id] = {"abl_id": abl_id, "error": str(exc), "desc": ABLATION_DEFS[abl_id]["desc"]}
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    log.info("─── Saving outputs ──────────────────────────────────────")
    save_all_outputs(all_results, data, output_dir, cfg)
    write_summary(all_results, cfg, output_dir)

    log.info("\n── FINAL SUMMARY ──────────────────────────────────────────────")
    for abl_id in ablations:
        res = all_results.get(abl_id, {})
        if "error" in res:
            log.info(f"  {abl_id:4s}  ERROR: {str(res['error'])[:60]}")
        else:
            avg = res.get("metrics", {}).get("_avg_prauc", np.nan)
            log.info(
                f"  {abl_id:4s}  avg={avg:.4f}  "
                f"(classical stack={CLASSICAL_REFERENCE_AVG:.4f}  Δ={avg - CLASSICAL_REFERENCE_AVG:+.4f})  "
                f"{res.get('split_key', '')}  {res.get('runtime_min', 0):.1f}min"
            )
    log.info("─────────────────────────────────────────────────────────────────")
