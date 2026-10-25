# src/data/transforms.py
"""Data transformation utilities for Pillar 1 experiments.

All transforms here are deterministic given a seed so that train/val/test
receive the *same* perturbation — essential for fair shuffling tests.
"""
from __future__ import annotations

import numpy as np


def make_column_permutation(n_features: int, seed: int = 42) -> np.ndarray:
    """Generate a single column permutation of length n_features.

    The same permutation should be applied to train/val/test for Exp 1.3.
    """
    rng = np.random.RandomState(seed)
    return rng.permutation(n_features)


def shuffle_columns(
    X: np.ndarray, perm: np.ndarray | None = None, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a column permutation to ``X``.

    Args:
        X: (n_samples, n_features) array.
        perm: pre-computed permutation. If None, generated with ``seed``.
        seed: seed used only when ``perm`` is None.

    Returns:
        (X_shuffled, perm).
    """
    if perm is None:
        perm = make_column_permutation(X.shape[1], seed=seed)
    return X[:, perm], perm
