# src/data/gene_attention_dataset.py
"""Dataset for GeneAttentionNet.

Inputs are gene-aggregated features (5000 genes × 4 stats) reshaped to a
3-D tensor (n_samples, n_genes, n_stats). Labels are multi-task (7
antibiotics) with a mask indicating which labels are observed per sample.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class GeneAttentionDataset(Dataset):
    """Wraps pre-built tensors. Pre-built so all splits share memory."""

    def __init__(
        self,
        X_gene: torch.Tensor,    # (n_samples, n_genes, n_stats) float32
        Y: torch.Tensor,         # (n_samples, n_antibiotics) float32
        mask: torch.Tensor,      # (n_samples, n_antibiotics) bool
        indices: np.ndarray,     # which rows belong to this split
    ):
        self.X = X_gene
        self.Y = Y
        self.mask = mask
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        return self.X[idx], self.Y[idx], self.mask[idx]


def reshape_gene_features(
    X_gene_flat: np.ndarray, n_genes: int, n_stats: int
) -> np.ndarray:
    """Reshape (n_samples, n_genes*n_stats) → (n_samples, n_genes, n_stats).

    Feature ordering is gene-major (gene_0_stat_0, gene_0_stat_1, ..., gene_1_stat_0,
    ...). This matches the output of build_gene_feature_matrix_from_pkl.
    """
    n_samples = X_gene_flat.shape[0]
    expected_cols = n_genes * n_stats
    if X_gene_flat.shape[1] != expected_cols:
        raise ValueError(
            f"Expected {expected_cols} columns ({n_genes} × {n_stats}), "
            f"got {X_gene_flat.shape[1]}."
        )
    return X_gene_flat.reshape(n_samples, n_genes, n_stats).astype(np.float32)


def normalize_gene_features_per_stat(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Standardize each stat channel using train-set statistics.

    The 4 stats (mean, var, n_nonref, max) have very different scales —
    e.g. ``var`` is typically < 2 but ``n_nonref`` can reach hundreds.
    Without normalisation the per-gene MLP would have to learn the
    rescaling, and BatchNorm-free architectures behave poorly.

    Returns the normalised tensors plus the per-stat mean/std for later
    test-time use.
    """
    # Compute per-stat statistics from train set, broadcast across all genes
    # X shape: (n, n_genes, n_stats). Stats over (n_samples, n_genes) axes
    # gives one mean/std per stat channel.
    train_mean = X_train.mean(axis=(0, 1), keepdims=True)         # (1, 1, n_stats)
    train_std = X_train.std(axis=(0, 1), keepdims=True) + eps

    def _norm(X):
        return ((X - train_mean) / train_std).astype(np.float32)

    return _norm(X_train), _norm(X_val), _norm(X_test), {
        "mean": train_mean.squeeze().tolist(),
        "std": train_std.squeeze().tolist(),
    }
