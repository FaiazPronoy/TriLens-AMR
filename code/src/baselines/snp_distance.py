"""
SNP Hamming distance between all strains (W2, kNN on SNPs), as pre-registered in
docs/experiments/w2_lineage_lookup.md: every non-zero genotype code counts as "differs from
reference", and the distance is the number of the 535,439 SNP columns where exactly one of the two
strains differs from reference. Missing calls are reference (KNOWN_ISSUES 16).

    python -m src.baselines.snp_distance            # -> data/cache/snp_hamming.npy (int32, ~194 MB)

Computed in column chunks (D = s_i + s_j - 2 x_i.x_j) with float32 products; exact while counts
stay below 2**24.
"""
from __future__ import annotations

import pickle
import time

import numpy as np

from src.data.metadata import PROCESSED, ROOT

CACHE = ROOT / "data" / "cache" / "snp_hamming.npy"


def snp_hamming(X, chunk=40_000):
    """All-pairs Hamming distance of the non-reference indicator (X != 0), in column chunks."""
    X = np.asarray(X)
    n, m = X.shape
    if m >= 2 ** 24:
        raise ValueError("too many columns for exact float32 counts")
    dot = np.zeros((n, n), dtype=np.float32)
    s = np.zeros(n, dtype=np.float64)
    for a in range(0, m, chunk):
        B = (X[:, a:a + chunk] != 0).astype(np.float32)
        dot += B @ B.T
        s += B.sum(axis=1)
    return np.rint(s[:, None] + s[None, :] - 2.0 * dot).astype(np.int32)


def main():
    t0 = time.time()
    with open(PROCESSED / "arcadia_v2_snp_matrix.pkl", "rb") as f:
        X = pickle.load(f)["X_snp"]
    print(f"SNP matrix {X.shape} loaded in {time.time() - t0:.0f}s")
    D = snp_hamming(X)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.save(CACHE, D)
    print(f"snp_hamming {D.shape} in {time.time() - t0:.0f}s -> {CACHE}")


if __name__ == "__main__":
    main()
