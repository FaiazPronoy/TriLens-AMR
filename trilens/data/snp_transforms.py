from __future__ import annotations

import numpy as np


def make_column_permutation(n_features: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.permutation(n_features)
