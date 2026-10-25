# src/experiments/run_stage2_gene_attention.py
"""Train GeneAttentionNet on Arcadia gene-aggregated features.

Pipeline:
    1. Build gene-aggregated features (4 stats × 5000 genes = 20,000-d).
    2. Reshape to (n_samples, 5000, 4) and per-stat standardise.
    3. Train GeneAttentionNet with multi-task masked BCE.
    4. Early stop on average val PR-AUC across antibiotics.
    5. Test: per-AB PR-AUC + MCC + bootstrap CI + attention extraction.

Output:
    {output_dir}/stage2_gene_attention_results.csv
    {output_dir}/stage2_gene_attention_predictions.npz
    {output_dir}/stage2_gene_attention_attention_weights.npz
    {output_dir}/SUMMARY.md
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

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
# Helpers
# ───────────────────────────────────────────────────────────────────────────────
def _pos_weights_per_ab(
    Y: np.ndarray, label_mask: np.ndarray, train_idx: np.ndarray,
) -> list[float]:
    """Compute scale_pos_weight per antibiotic using train-set label distribution."""
    pws: list[float] = []
    for ab_idx in range(Y.shape[1]):
        m = label_mask[train_idx, ab_idx]
        if m.sum() == 0:
            pws.append(1.0)
            continue
        y_ab = Y[train_idx, ab_idx][m]
        n_pos = float((y_ab == 1).sum())
        n_neg = float((y_ab == 0).sum())
        pws.append(max(1.0, n_neg / max(n_pos, 1.0)))
    return pws


def _make_loader(ds, batch_size, num_workers, shuffle):
    import platform
    if platform.system() == "Windows" and num_workers > 0:
        num_workers = 0
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        drop_last=False,
    )


# ───────────────────────────────────────────────────────────────────────────────
# Entry-point
# ───────────────────────────────────────────────────────────────────────────────
def run(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    exp_cfg = cfg.experiment
    out_dir = Path(exp_cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ckpts").mkdir(exist_ok=True)
    OmegaConf.save(cfg, out_dir / "config_used.yaml")

    print("=" * 80)
    print("  STAGE 2 — GeneAttentionNet (multi-task, gene-aware attention)")
    print("=" * 80)

    # ─── 1. Build features ────────────────────────────────────────────────────
    print("\n[1/5] Loading gene-aggregated features...")
    t0 = time.time()
    pkl_path = (str(cfg.paths.processed_dir).rstrip("/\\") +
                "/arcadia_v2_gene_filtered.pkl")
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=pkl_path,
        stats=tuple(exp_cfg.gene_stats),
    )
    print(f"  built in {time.time() - t0:.1f}s")

    X_flat = bundle["X_gene_features"]                    # (n, n_genes * n_stats)
    n_genes = len(bundle["gene_ids"])
    n_stats = len(bundle["stats_used"])
    print(f"  n_genes={n_genes}, n_stats={n_stats}")

    Y = bundle["Y"].astype(np.float32)                    # (n, 7)
    label_mask = bundle["label_mask"].astype(bool)        # (n, 7)
    label_names = [str(x) for x in bundle["label_names"]]
    splits = bundle["random_split"]
    train_idx = np.asarray(splits["train_idx"])
    val_idx = np.asarray(splits["val_idx"])
    test_idx = np.asarray(splits["test_idx"])

    # Reshape and standardise per-stat
    print("\n[2/5] Reshaping + standardising per-stat...")
    X_3d = reshape_gene_features(X_flat, n_genes, n_stats)
    X_tr, X_va, X_te, norm_stats = normalize_gene_features_per_stat(
        X_3d[train_idx], X_3d[val_idx], X_3d[test_idx]
    )
    print(f"  per-stat mean: {[round(m, 3) for m in norm_stats['mean']]}")
    print(f"  per-stat std : {[round(s, 3) for s in norm_stats['std']]}")

    # Place all splits back into a single tensor for easy indexing inside Dataset
    X_norm = np.zeros_like(X_3d, dtype=np.float32)
    X_norm[train_idx] = X_tr
    X_norm[val_idx] = X_va
    X_norm[test_idx] = X_te
    X_t = torch.from_numpy(X_norm)
    Y_t = torch.from_numpy(Y)
    mask_t = torch.from_numpy(label_mask)

    # ─── 3. Build datasets and loaders ────────────────────────────────────────
    print("\n[3/5] Building DataLoaders...")
    train_ds = GeneAttentionDataset(X_t, Y_t, mask_t, train_idx)
    val_ds = GeneAttentionDataset(X_t, Y_t, mask_t, val_idx)
    test_ds = GeneAttentionDataset(X_t, Y_t, mask_t, test_idx)

    bs = int(exp_cfg.training.batch_size)
    nw = int(exp_cfg.training.num_workers)
    train_loader = _make_loader(train_ds, bs, nw, shuffle=True)
    val_loader = _make_loader(val_ds, bs, nw, shuffle=False)
    test_loader = _make_loader(test_ds, bs, nw, shuffle=False)

    print(f"  Train n={len(train_ds)}, Val n={len(val_ds)}, Test n={len(test_ds)}")
    print(f"  batch_size={bs}, num_workers={nw}")

    # ─── 4. Build model ───────────────────────────────────────────────────────
    print("\n[4/5] Building model...")
    antibiotics = list(exp_cfg.antibiotics)
    # Reorder labels so we predict in cfg.antibiotics order
    ab_indices = [label_names.index(a) for a in antibiotics]
    # Sub-select Y / mask columns to match
    Y_sub = Y[:, ab_indices]
    mask_sub = label_mask[:, ab_indices]
    pos_weights = _pos_weights_per_ab(Y_sub, mask_sub, train_idx)
    print(f"  pos_weights: " +
          ", ".join(f"{a}={w:.2f}" for a, w in zip(antibiotics, pos_weights)))

    # Re-wrap datasets with subset labels matching antibiotic order
    Y_sub_t = torch.from_numpy(Y_sub.astype(np.float32))
    mask_sub_t = torch.from_numpy(mask_sub)
    train_ds = GeneAttentionDataset(X_t, Y_sub_t, mask_sub_t, train_idx)
    val_ds = GeneAttentionDataset(X_t, Y_sub_t, mask_sub_t, val_idx)
    test_ds = GeneAttentionDataset(X_t, Y_sub_t, mask_sub_t, test_idx)
    train_loader = _make_loader(train_ds, bs, nw, shuffle=True)
    val_loader = _make_loader(val_ds, bs, nw, shuffle=False)
    test_loader = _make_loader(test_ds, bs, nw, shuffle=False)

    model = GeneAttentionNet(
        n_genes=n_genes,
        n_stats=n_stats,
        n_antibiotics=len(antibiotics),
        antibiotic_names=antibiotics,
        hidden_dim=int(exp_cfg.model.hidden_dim),
        n_heads=int(exp_cfg.model.n_heads),
        n_layers=int(exp_cfg.model.n_layers),
        ffn_mult=int(exp_cfg.model.ffn_mult),
        dropout=float(exp_cfg.model.dropout),
        pos_weights=pos_weights,
        learning_rate=float(exp_cfg.training.learning_rate),
        weight_decay=float(exp_cfg.training.weight_decay),
        scheduler_t_max=int(exp_cfg.training.scheduler_t_max),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  total params: {n_params/1e6:.2f} M")

    # ─── 5. Train and evaluate ────────────────────────────────────────────────
    print("\n[5/5] Training...")
    run_name = "stage2_gene_attention"
    wandb_logger = None
    if cfg.get("use_wandb", True):
        try:
            wandb_logger = WandbLogger(
                project=exp_cfg.wandb.project,
                name=run_name,
                tags=list(exp_cfg.wandb.tags),
                save_dir=str(out_dir),
            )
        except Exception as ex:
            print(f"  [WARN] WandBLogger failed ({ex!r}); continuing without it.")

    es = EarlyStopping(
        monitor="val/pr_auc",
        mode="max",
        patience=int(exp_cfg.training.early_stop_patience),
        verbose=False,
    )
    mc = ModelCheckpoint(
        dirpath=str(out_dir / "ckpts"),
        filename=f"{run_name}_best",
        monitor="val/pr_auc",
        mode="max",
        save_top_k=1,
        save_last=False,
    )
    trainer = L.Trainer(
        max_epochs=int(exp_cfg.training.max_epochs),
        min_epochs=int(exp_cfg.training.min_epochs),
        accelerator="gpu",
        devices=1,
        precision=str(exp_cfg.training.precision),
        gradient_clip_val=float(exp_cfg.training.gradient_clip),
        callbacks=[es, mc],
        logger=wandb_logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        deterministic=False,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    fit_time = time.time() - t0
    print(f"  fit complete in {fit_time/60:.1f} min")

    # Load best checkpoint
    if mc.best_model_path:
        ckpt = torch.load(mc.best_model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])

    # ─── Test ─────────────────────────────────────────────────────────────────
    model.eval()
    device = next(model.parameters()).device
    all_logits, all_y, all_mask, all_attn = [], [], [], []
    with torch.no_grad():
        for xb, yb, mb in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)
            all_logits.append(logits.detach().cpu())
            all_y.append(yb.detach().cpu())
            all_mask.append(mb.detach().cpu())
            if model.last_attention is not None:
                all_attn.append(model.last_attention.cpu())

    logits_np = torch.cat(all_logits).numpy()
    y_np = torch.cat(all_y).numpy().astype(np.int32)
    mask_np = torch.cat(all_mask).numpy().astype(bool)
    probs_np = 1.0 / (1.0 + np.exp(-logits_np.astype(np.float64)))
    attn_np = (torch.cat(all_attn).numpy() if all_attn else
               np.zeros((logits_np.shape[0], 1, n_genes), dtype=np.float32))

    # Per-AB metrics + CIs
    rows = []
    boot_n = int(exp_cfg.evaluation.bootstrap_n)
    boot_seed = int(exp_cfg.evaluation.bootstrap_seed)
    for i, ab in enumerate(antibiotics):
        m = mask_np[:, i]
        if m.sum() == 0:
            continue
        yy = y_np[m, i]
        pp = probs_np[m, i].astype(np.float32)
        metrics = compute_metrics(yy, pp, threshold=0.5)
        pr_lo, pr_hi = bootstrap_pr_auc(yy, pp, n_boot=boot_n, seed=boot_seed)
        mcc_lo, mcc_hi = bootstrap_mcc(yy, pp, n_boot=boot_n, seed=boot_seed)
        rows.append({
            "antibiotic": ab,
            "test_pr_auc": metrics["pr_auc"],
            "test_pr_auc_lo": pr_lo, "test_pr_auc_hi": pr_hi,
            "test_mcc": metrics["mcc"],
            "test_mcc_lo": mcc_lo, "test_mcc_hi": mcc_hi,
            "test_roc_auc": metrics["roc_auc"],
            "test_f1": metrics["f1"],
            "test_f2": metrics["f2"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_sens_at_90_spec": metrics["sens_at_90_spec"],
            "test_brier": metrics["brier"],
            "n_test_valid": int(m.sum()),
            "n_pos_test": int(yy.sum()),
        })
        print(f"    {ab:<5}  PR-AUC: {metrics['pr_auc']:.4f} "
              f"[{pr_lo:.4f}, {pr_hi:.4f}]   MCC: {metrics['mcc']:.4f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stage2_gene_attention_results.csv", index=False)

    avg_pr = float(df["test_pr_auc"].mean()) if len(df) else 0.0
    avg_mcc = float(df["test_mcc"].mean()) if len(df) else 0.0

    # Persist artefacts
    np.savez_compressed(
        out_dir / "stage2_gene_attention_predictions.npz",
        logits=logits_np, y_true=y_np, mask=mask_np,
        antibiotics=np.array(antibiotics),
    )
    np.savez_compressed(
        out_dir / "stage2_gene_attention_attention_weights.npz",
        attention=attn_np.squeeze(1),                # (n_test, n_genes)
        gene_ids=np.array(bundle["gene_ids"]),
    )

    # Markdown summary
    lines = [
        "# Stage 2 — GeneAttentionNet results\n",
        f"- n_params: {n_params/1e6:.2f} M",
        f"- fit_time: {fit_time/60:.1f} min",
        f"- best_ckpt: {mc.best_model_path}",
        f"- avg test PR-AUC: **{avg_pr:.4f}**",
        f"- avg test MCC:    **{avg_mcc:.4f}**\n",
        "## Per-antibiotic\n",
        "| AB | PR-AUC | [lo, hi] | MCC | F2 |",
        "|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['test_pr_auc']:.4f} "
            f"| [{r['test_pr_auc_lo']:.4f}, {r['test_pr_auc_hi']:.4f}] "
            f"| {r['test_mcc']:.4f} | {r['test_f2']:.4f} |"
        )
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("  STAGE 2 GeneAttentionNet COMPLETE")
    print("=" * 80)
    print(f"  avg PR-AUC = {avg_pr:.4f}")
    print(f"  avg MCC    = {avg_mcc:.4f}")
    print(f"  Outputs    → {out_dir}")
