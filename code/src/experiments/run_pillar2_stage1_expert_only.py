"""
src/experiments/run_pillar2_stage1_expert_only.py
==================================================
Pillar 2 layer ablation — Stage 1: TriLens's 3 experts, unweighted average,
gate bypassed.

Trains a fresh TriLens model with the identical architecture, hyperparameters,
random seed, 5-fold OOF protocol, and data split as the T6 headline run in
run_trilens_v10.py (reuses that file's functions directly — nothing there is
modified). The only difference is at prediction time: instead of
torch.sigmoid(gate-weighted fusion), this script computes
torch.sigmoid((l_A + l_B + l_C) / 3), i.e. the three experts' raw logits are
averaged BEFORE any gate is applied. This isolates how much the learned MoE
gate adds over naive averaging of the same trained experts.

A fresh training pass is required (not just re-inference) because the
original T6 pipeline never persists trained model weights to disk (verified:
no torch.save() call anywhere in run_trilens_v10.py) — only its prediction
outputs are saved. The 3-expert logits needed for this ablation are computed
inside the forward pass but discarded by run_trilens_v10.py's own _predict(),
so there is no way to recover them without retraining.

Output: results/pillar2_layer_ablation/stage1_expert_only_preds.npz
        (same key layout as results/trilens_v10/trilens_kfold_oof.npz:
        oof_preds, val_preds, test_preds, train_idx, val_idx, test_idx)
        results/pillar2_layer_ablation/stage1_run.log

Estimated runtime: ~35-45 minutes on an RTX 3060/4070-class GPU — the
original T6 K-fold OOF training (same 5-fold loop) took 38.6 min on record
(experiments/2026-06-05_16-11-31_trilens_v10/run_experiment.log).

Usage (PowerShell, from project root, after activating the amr conda env):
    conda activate amr
    python -m src.experiments.run_pillar2_stage1_expert_only
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from src.experiments.run_trilens_v10 import (
    AB_NAMES,
    _build_dataset,
    _create_model,
    _get_split_indices,
    _train_until_convergence,
    build_expert_a_blocks,
    build_expert_c_features,
    compute_pos_weights,
    load_all_data,
    set_seed,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]


def _predict_expert_avg(model, dataset, device, batch_size) -> np.ndarray:
    """Mirrors run_trilens_v10._predict()'s structure exactly; the only
    difference is aggregating l_A/l_B/l_C by unweighted mean instead of the
    learned MoE gate, before the single sigmoid at the very end."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_l = []
    with torch.no_grad():
        for batch in loader:
            X_b, X_pa, X_gc, _, _ = [t.to(device, non_blocking=True) for t in batch]
            out = model(X_b, X_pa, X_gc)
            l_avg = (out["l_A"] + out["l_B"] + out["l_C"]) / 3.0
            probs_l.append(torch.sigmoid(l_avg).cpu().float().numpy())
    return np.concatenate(probs_l, axis=0) if probs_l else np.zeros((0, len(AB_NAMES)))


def _load_config():
    base_cfg = OmegaConf.load(ROOT / "configs" / "config.yaml")
    # hydra's own block uses the Hydra-only "now" resolver, unavailable outside
    # a real Hydra run; drop it (and unused wandb block) before resolving.
    if "hydra" in base_cfg:
        del base_cfg["hydra"]
    if "wandb" in base_cfg:
        del base_cfg["wandb"]
    trilens_cfg = OmegaConf.load(ROOT / "configs" / "experiment" / "trilens_v10.yaml")
    stage1_cfg = OmegaConf.load(ROOT / "configs" / "experiment" / "pillar2_layer_ablation.yaml")
    cfg = OmegaConf.merge(base_cfg, trilens_cfg, stage1_cfg)
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))


def run(cfg) -> None:
    seed = int(cfg.training.seed)
    set_seed(seed)

    output_dir = Path(str(cfg.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    fh = logging.FileHandler(output_dir / "stage1_run.log", mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s  %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(
        f"Device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    log.info("─── Loading data (identical to run_trilens_v10.py) ───")
    data = load_all_data(cfg)
    split_key = str(cfg.split_key)
    train_idx, val_idx, test_idx = _get_split_indices(data, split_key)

    cache_dir = Path(str(cfg.existing_cache_dir))
    log.info(f"─── Expert A blocks (reusing existing cache: {cache_dir}) ───")
    X_blocks, M_blocks, _ = build_expert_a_blocks(data, cfg, cache_dir, train_idx)

    log.info(f"─── Expert C features (reusing existing cache: {cache_dir}) ───")
    X_gene_agg, _ = build_expert_c_features(data, train_idx, cache_dir)

    n_train = len(train_idx)
    n_folds = int(cfg.n_folds)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=int(cfg.kfold_seed))
    pw = compute_pos_weights(data["Y"][train_idx], data["label_mask"][train_idx], device)

    oof_preds = np.zeros((n_train, len(AB_NAMES)), dtype=np.float32)
    val_preds_f, test_preds_f = [], []

    val_ds = _build_dataset(val_idx, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])
    test_ds = _build_dataset(test_idx, X_blocks, data["X_pa"], X_gene_agg, data["Y"], data["label_mask"])

    t_start = time.time()
    for fold, (fit_es_local, oof_local) in enumerate(kf.split(np.arange(n_train))):
        fold_seed = seed + fold * 100
        set_seed(fold_seed)
        log.info(f"\n{'─'*60}\n  [Fold {fold+1}/{n_folds}]  seed={fold_seed}")

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
        model, _hist = _train_until_convergence(model, fit_ds, es_ds, pw, cfg, device)

        bs = int(cfg.training.batch_size) * 2
        oof_p = _predict_expert_avg(model, oof_ds, device, bs)
        val_p = _predict_expert_avg(model, val_ds, device, bs)
        tst_p = _predict_expert_avg(model, test_ds, device, bs)

        oof_preds[oof_local] = oof_p
        val_preds_f.append(val_p)
        test_preds_f.append(tst_p)

        del model, fit_ds, es_ds, oof_ds
        torch.cuda.empty_cache()
        gc.collect()

    val_preds = np.mean(val_preds_f, axis=0)
    test_preds = np.mean(test_preds_f, axis=0)

    np.savez(
        str(output_dir / "stage1_expert_only_preds.npz"),
        oof_preds=oof_preds,
        val_preds=val_preds,
        test_preds=test_preds,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )
    total_min = (time.time() - t_start) / 60
    log.info(f"Saved stage1_expert_only_preds.npz  total_runtime={total_min:.1f}min")


if __name__ == "__main__":
    run(_load_config())
