# src/experiments/run_pillar3_interpretability.py
"""
================================================================================
Pillar 3 — Interpretability via GeneAttentionNet attention weights
================================================================================

Defence value: closes the loop on Pillar 1's locality-mismatch hypothesis.
We showed that pan-genome SNP matrices break CNN locality assumptions, then
proposed GeneAttentionNet (Pillar 2) which uses learned attention over genes
instead. This script extracts what the trained attention layer actually
learned — the top-K genes the model attends to when predicting resistance —
turning a black-box neural model into biological evidence.

Pipeline:
    1. Load trained GeneAttentionNet checkpoint (Stage 2 multi-task run).
    2. Recompute per-stat standardisation from train split (identical to
       training).
    3. Forward-pass all 6962 samples; capture the attention-pool weights
       of shape (n_samples, n_genes).
    4. For every antibiotic:
         a. Split valid samples into resistant (y=1) and susceptible (y=0).
         b. Compute mean attention per gene for each group.
         c. Rank genes by (i) mean attention on resistant samples,
                          (ii) differential = resistant − susceptible.
    5. Save:
         - top_k_<AB>.csv      per antibiotic, full top-K table
         - top_genes_all.csv   combined view, top-K every AB side by side
         - SUMMARY.md          ranked tables ready to drop into the thesis
         - attention_matrix.npz raw (n_samples, n_genes) attention for reuse
         - heatmap.png         (top genes union) × (antibiotics) heatmap

Run:
    python -m src.run_experiment experiment=pillar3_interpretability
================================================================================
"""
from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf

from src.models.gene_attention_net import GeneAttentionNet
from src.utils.seeds import set_global_seed


# ───────────────────────────────────────────────────────────────────────────────
# Attention extraction
# ───────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _extract_attention_weights(
    model: GeneAttentionNet,
    X_3d_norm: np.ndarray,
    device: str = "cuda",
    batch_size: int = 32,
) -> np.ndarray:
    """Forward all samples, capture attention-pool weights.

    Mirrors GeneAttentionNet.forward() but extracts the pool's attention
    output instead of logits.

    Returns
    -------
    attn : np.ndarray, shape (n_samples, n_genes)
        Row i, column g  = how much gene g contributed to the pooled
        representation of sample i (softmax-normalised across genes).
    """
    model.eval()
    out_chunks: list[np.ndarray] = []
    n = len(X_3d_norm)
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X_3d_norm[i:i + batch_size]).float().to(device)
        B = xb.size(0)

        # Replicate the model's forward pass up to the attention pool
        h = model.gene_encoder(xb)                                   # (B, G, H)
        gene_ids = torch.arange(model.n_genes, device=device)
        pos = model.gene_pos_emb(gene_ids)
        h = h + pos.unsqueeze(0)                                     # (B, G, H)
        for blk in model.blocks:
            h = blk(h)                                               # (B, G, H)

        _, attn_w = model.pool(h)                                    # (B, 1, G)
        out_chunks.append(attn_w.squeeze(1).cpu().numpy().astype(np.float32))

    return np.concatenate(out_chunks, axis=0)                        # (N, G)


# ───────────────────────────────────────────────────────────────────────────────
# Per-antibiotic analysis
# ───────────────────────────────────────────────────────────────────────────────
def _analyze_per_ab(
    attn: np.ndarray,
    Y: np.ndarray,
    label_mask: np.ndarray,
    label_names: list[str],
    selected_gene_ids: list,
    top_k: int,
    subset_idx: np.ndarray | None = None,
) -> dict[str, pd.DataFrame]:
    """For each AB return a DataFrame of top-K genes (ranked by differential)."""
    n_samples, n_genes = attn.shape
    assert Y.shape == (n_samples, len(label_names))
    if subset_idx is None:
        subset_idx = np.arange(n_samples)

    results: dict[str, pd.DataFrame] = {}
    for ab_i, ab_name in enumerate(label_names):
        m_sub = label_mask[subset_idx, ab_i]
        y_sub = Y[subset_idx, ab_i]
        valid = m_sub.astype(bool)
        if valid.sum() < 20:
            continue

        valid_global_idx = subset_idx[valid]
        attn_valid = attn[valid_global_idx]                  # (n_valid, G)
        y_valid = y_sub[valid].astype(np.int32)

        res_rows = (y_valid == 1)
        sus_rows = (y_valid == 0)
        if res_rows.sum() < 5 or sus_rows.sum() < 5:
            continue

        mean_res = attn_valid[res_rows].mean(axis=0)
        mean_sus = attn_valid[sus_rows].mean(axis=0)
        diff = mean_res - mean_sus
        ratio = mean_res / np.maximum(mean_sus, 1e-9)

        order_res = np.argsort(-mean_res)[:top_k]
        order_diff = np.argsort(-np.abs(diff))[:top_k]

        union = np.unique(np.concatenate([order_res, order_diff]))
        rows = []
        for gi in union:
            rows.append({
                "gene_idx": int(gi),
                "gene_id": str(selected_gene_ids[gi]),
                "mean_attn_resistant": float(mean_res[gi]),
                "mean_attn_susceptible": float(mean_sus[gi]),
                "differential": float(diff[gi]),
                "ratio_res_over_sus": float(ratio[gi]),
                "rank_by_resistant": int(np.where(np.argsort(-mean_res) == gi)[0][0]) + 1,
                "rank_by_differential": int(
                    np.where(np.argsort(-np.abs(diff)) == gi)[0][0]
                ) + 1,
            })
        df = pd.DataFrame(rows).sort_values("rank_by_differential").reset_index(drop=True)
        df["antibiotic"] = ab_name
        df["n_resistant"] = int(res_rows.sum())
        df["n_susceptible"] = int(sus_rows.sum())
        results[ab_name] = df

    return results


# ───────────────────────────────────────────────────────────────────────────────
# Visualisation
# ───────────────────────────────────────────────────────────────────────────────
def _save_heatmap(
    per_ab_results: dict[str, pd.DataFrame],
    selected_gene_ids: list,
    attn: np.ndarray,
    Y: np.ndarray,
    label_mask: np.ndarray,
    label_names: list[str],
    top_k_heat: int,
    out_path: Path,
) -> None:
    """Heatmap: rows = union of top-K genes across all ABs, cols = ABs,
    cells = mean attention on resistant samples."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [heatmap] matplotlib unavailable — skipping")
        return

    union_idx: list[int] = []
    for ab, df in per_ab_results.items():
        top = df.head(top_k_heat)["gene_idx"].tolist()
        union_idx.extend(top)
    union_idx = sorted(set(union_idx))
    if not union_idx:
        return

    abs_order = [a for a in label_names if a in per_ab_results]
    H = np.zeros((len(union_idx), len(abs_order)), dtype=np.float32)
    for j, ab in enumerate(abs_order):
        ab_i = label_names.index(ab)
        valid = label_mask[:, ab_i].astype(bool)
        res_mask = valid & (Y[:, ab_i] == 1)
        mean_res = attn[res_mask].mean(axis=0)
        H[:, j] = mean_res[union_idx]

    fig, ax = plt.subplots(figsize=(max(6, len(abs_order) * 0.9),
                                    max(6, len(union_idx) * 0.25)))
    im = ax.imshow(H, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(abs_order))); ax.set_xticklabels(abs_order, rotation=0)
    ax.set_yticks(range(len(union_idx)))
    ax.set_yticklabels([str(selected_gene_ids[gi]) for gi in union_idx], fontsize=7)
    ax.set_xlabel("Antibiotic")
    ax.set_ylabel("Gene")
    ax.set_title(f"Top-{top_k_heat} attention genes per antibiotic\n"
                 f"(mean attention on resistant samples)")
    plt.colorbar(im, ax=ax, label="Mean attention")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [heatmap] saved -> {out_path}")


# ───────────────────────────────────────────────────────────────────────────────
# Checkpoint loading (handles Lightning + plain torch.save formats)
# ───────────────────────────────────────────────────────────────────────────────
def _load_gan_checkpoint(
    ckpt_path: Path,
    n_genes: int,
    n_stats: int,
    n_antibiotics: int,
    antibiotic_names: list[str],
    gan_cfg: dict,
    device: str = "cuda",
) -> GeneAttentionNet:
    print(f"  [ckpt] loading: {ckpt_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]
    elif isinstance(raw, dict) and any(k.startswith("module.") for k in raw):
        sd = {k.replace("module.", "", 1): v for k, v in raw.items()}
    elif isinstance(raw, dict):
        sd = raw
    else:
        raise ValueError(f"Unrecognised checkpoint format: {type(raw)}")

    # Lightning checkpoints typically prefix keys with 'model.' — strip it if so
    if any(k.startswith("model.") for k in sd):
        sd = {k.replace("model.", "", 1): v for k, v in sd.items()
              if k.startswith("model.")} or sd

    # Build model with same hyperparams as training
    pos_weights = [1.0] * n_antibiotics
    model = GeneAttentionNet(
        n_genes=n_genes, n_stats=n_stats, n_antibiotics=n_antibiotics,
        antibiotic_names=antibiotic_names,
        hidden_dim=int(gan_cfg["hidden_dim"]),
        n_heads=int(gan_cfg["n_heads"]),
        n_layers=int(gan_cfg["n_layers"]),
        ffn_mult=int(gan_cfg["ffn_mult"]),
        dropout=float(gan_cfg["dropout"]),
        pos_weights=pos_weights,
        learning_rate=float(gan_cfg.get("learning_rate", 1e-3)),
        weight_decay=1e-4,
        scheduler_t_max=int(gan_cfg["max_epochs"]),
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [ckpt] WARNING missing keys: {len(missing)} (first 3: {missing[:3]})")
    if unexpected:
        print(f"  [ckpt] WARNING unexpected keys: {len(unexpected)} (first 3: {unexpected[:3]})")
    model = model.to(device)
    return model


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
    print("  Pillar 3 — Interpretability via GeneAttentionNet attention")
    print("=" * 80)
    print(f"  GPU     : {'YES' if device == 'cuda' else 'NO'}  ({device})")
    print(f"  Output  : {out_dir}")

    # ── Load gene-filtered bundle ────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    pkl_path = (str(cfg.paths.processed_dir).rstrip("/\\")
                + "/arcadia_v2_gene_filtered.pkl")
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    selected_gene_ids = list(d["selected_gene_ids"])
    Y = d["Y"].astype(np.int32)
    label_mask = d["label_mask"].astype(bool)
    label_names = [str(x) for x in d["label_names"]]
    splits = d["random_split"]
    tr_idx = np.asarray(splits["train_idx"])
    va_idx = np.asarray(splits["val_idx"])
    te_idx = np.asarray(splits["test_idx"])
    n_genes = len(selected_gene_ids)
    n_stats = len(exp_cfg.gene_stats)
    print(f"  n_genes={n_genes}, n_stats={n_stats}, n_samples={Y.shape[0]}, "
          f"n_ab={Y.shape[1]}")
    print(f"  labels={label_names}")

    # ── Build / standardise 3-D tensor identically to training ───────────────
    print("\n[2/6] Building & standardising 3-D gene tensor...")
    from src.data.gene_aggregated_features import build_gene_feature_matrix_from_pkl
    bundle = build_gene_feature_matrix_from_pkl(
        gene_filtered_pkl_path=pkl_path,
        stats=tuple(exp_cfg.gene_stats),
    )
    X_flat = bundle["X_gene_features"].astype(np.float32)
    X_3d = X_flat.reshape(X_flat.shape[0], n_genes, n_stats)
    train_3d = X_3d[tr_idx]
    means = train_3d.reshape(-1, n_stats).mean(axis=0)
    stds = train_3d.reshape(-1, n_stats).std(axis=0) + 1e-7
    X_3d_norm = ((X_3d - means[None, None, :]) / stds[None, None, :]).astype(np.float32)
    print(f"  X_3d_norm: {X_3d_norm.shape}")

    # ── Load checkpoint ──────────────────────────────────────────────────────
    print("\n[3/6] Loading trained GeneAttentionNet checkpoint...")
    ckpt_path = Path(str(exp_cfg.ckpt_path))
    model = _load_gan_checkpoint(
        ckpt_path=ckpt_path,
        n_genes=n_genes, n_stats=n_stats,
        n_antibiotics=Y.shape[1],
        antibiotic_names=label_names,
        gan_cfg=dict(exp_cfg.gan), device=device,
    )

    # ── Extract attention ────────────────────────────────────────────────────
    print("\n[4/6] Extracting attention weights for all 6962 samples...")
    t0 = time.time()
    attn = _extract_attention_weights(
        model, X_3d_norm, device=device, batch_size=32,
    )
    print(f"  attention matrix: {attn.shape}  ({time.time() - t0:.1f}s)")
    np.savez_compressed(
        out_dir / "attention_matrix.npz",
        attention=attn,
        selected_gene_ids=np.array(selected_gene_ids, dtype=object),
        label_names=np.array(label_names, dtype=object),
    )

    # ── Per-AB analysis (train+val pool, leak-safe by excluding test) ────────
    print("\n[5/6] Per-antibiotic top-K analysis...")
    top_k = int(exp_cfg.top_k)
    trainval_idx = np.concatenate([tr_idx, va_idx])
    per_ab = _analyze_per_ab(
        attn=attn, Y=Y, label_mask=label_mask,
        label_names=label_names, selected_gene_ids=selected_gene_ids,
        top_k=top_k, subset_idx=trainval_idx,
    )

    for ab, df in per_ab.items():
        df.to_csv(out_dir / f"top_{top_k}_{ab}.csv", index=False, encoding="utf-8")
        print(f"  {ab}: top-{min(top_k, len(df))} genes saved "
              f"(n_res={df['n_resistant'].iloc[0]}, "
              f"n_sus={df['n_susceptible'].iloc[0]})")

    # Combined view
    summary_rows = []
    for ab, df in per_ab.items():
        for r in range(min(top_k, len(df))):
            row = df.iloc[r]
            summary_rows.append({
                "antibiotic": ab,
                "rank": r + 1,
                "gene_id": row["gene_id"],
                "gene_idx": row["gene_idx"],
                "mean_attn_resistant": row["mean_attn_resistant"],
                "mean_attn_susceptible": row["mean_attn_susceptible"],
                "differential": row["differential"],
            })
    df_all = pd.DataFrame(summary_rows)
    df_all.to_csv(out_dir / "top_genes_all.csv", index=False, encoding="utf-8")

    # ── Heatmap + Markdown summary ───────────────────────────────────────────
    print("\n[6/6] Visualisation + Markdown summary...")
    _save_heatmap(
        per_ab_results=per_ab,
        selected_gene_ids=selected_gene_ids,
        attn=attn, Y=Y, label_mask=label_mask, label_names=label_names,
        top_k_heat=int(exp_cfg.top_k_heatmap),
        out_path=out_dir / "heatmap.png",
    )

    lines = [
        "# Pillar 3 — GeneAttentionNet Interpretability\n",
        "## Per-antibiotic top genes (ranked by absolute differential "
        "attention: resistant - susceptible)\n",
    ]
    for ab in label_names:
        if ab not in per_ab:
            continue
        df = per_ab[ab].head(top_k)
        lines.append(f"### {ab}  (n_resistant={df['n_resistant'].iloc[0]}, "
                     f"n_susceptible={df['n_susceptible'].iloc[0]})\n")
        lines.append("| Rank | Gene ID | attn_resistant | attn_susceptible | "
                     "differential |")
        lines.append("|---|---|---|---|---|")
        for r in range(min(top_k, len(df))):
            row = df.iloc[r]
            lines.append(
                f"| {r + 1} | {row['gene_id']} | "
                f"{row['mean_attn_resistant']:.5f} | "
                f"{row['mean_attn_susceptible']:.5f} | "
                f"{row['differential']:+.5f} |"
            )
        lines.append("")

    lines += [
        "## Methodology\n",
        "1. Trained multi-task GeneAttentionNet (Stage 2) loaded from "
        f"`{ckpt_path.name}`.",
        "2. Forward pass on all 6962 samples (train+val+test) with identical "
        "per-stat z-score standardisation as training (means/stds from train).",
        "3. Captured attention-pool weights of shape (n_samples, n_genes).",
        "4. For each antibiotic, separated valid-label samples into resistant "
        "(y=1) and susceptible (y=0). Computed mean attention vector per group.",
        "5. Ranked genes by (a) mean attention on resistant samples and "
        "(b) absolute differential vs susceptible. Test split *excluded* from "
        "this ranking to avoid information leak into model selection.",
        "",
        "## Outputs",
        "- `top_K_<AB>.csv`         — per-antibiotic table",
        "- `top_genes_all.csv`      — combined ranking",
        "- `attention_matrix.npz`   — raw attention tensor for further work",
        "- `heatmap.png`            — (top genes) × (antibiotics) visualisation",
        "- `SUMMARY.md`             — this file",
        "",
        "## Defence value",
        "This closes the loop with Pillar 1: we proved that pan-genome "
        "concatenated SNP matrices break CNN locality assumptions and proposed "
        "attention-based gene-wise aggregation (Pillar 2). Pillar 3 demonstrates "
        "that the learned attention is biologically interpretable, ranking "
        "specific genes whose nucleotide variation drives resistance prediction "
        "per antibiotic class. These ranked gene lists can be cross-referenced "
        "with curated AMR knowledge bases (CARD, ResFinder, MEGARes) to validate "
        "biological plausibility.",
    ]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n" + "=" * 80)
    print("  Pillar 3 interpretability COMPLETE")
    print(f"  Outputs -> {out_dir}")
    print("=" * 80)
