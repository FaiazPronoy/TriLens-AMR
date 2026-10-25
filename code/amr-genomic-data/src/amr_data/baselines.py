"""PCA + XGBoost baseline (B8: memory-safe sparse PCA).

Ported from `arcadia-v2-phase-b.ipynb` cells 4 and 5. The original notebook
did `X = ds["X"].astype(np.float32)` on the full (6962, 535439) int8 matrix
-- a 3.73 GB -> ~14.9 GB cast, materialized in full before `TruncatedSVD`
ever runs. Cell 5 got away with it by `del`-ing and `gc.collect()`-ing
aggressively around the cast; cell 4 (the actual 5-dim PCA search that
produced the manifest's E7 numbers) did not.

`TruncatedSVD` accepts `scipy.sparse` input natively and never internally
materializes a dense copy for its randomized solver -- and the SNP matrix is
~83.3% zeros (value 0 = reference allele, verified directly against
`arcadia_v2_snp_matrix.pkl`'s own value-count dict during the audit), so a
`scipy.sparse.csr_matrix` built straight from the int8 array is both correct
and the "obvious" fix the audit recommended. `TruncatedSVD` on this sparse
matrix and on the equivalent dense float32 matrix solve the same
optimization problem -- results agree up to solver-level numerical noise,
well inside the manifest's ±0.002 PR-AUC tolerance for E7 (verified directly
in this refactor for dim=1024, the argmax dimension: variance retained
0.9465 matches the locked master_features.pkl's `pca_var_retained` to 4
decimal places -- see tests/test_baselines.py).
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from .config import load_params
from .metrics import safe_pr_auc

_params = load_params()["pca_baseline"]
DEFAULT_SEARCH_DIMS: list[int] = list(_params["search_dims"])
CHOSEN_DIM: int = _params["chosen_dim"]


def to_sparse(X_int8: np.ndarray, *, row_chunk_size: int = 500) -> sp.csr_matrix:
    """B8 fix: never cast the full matrix to dense float32, and never hand
    scipy the *whole* dense array in one `sp.csr_matrix(...)` call either.

    A naive `sp.csr_matrix(X_int8.astype(np.float32))` on the real
    (6962, 535439) matrix (~83.3% zeros, verified from
    `arcadia_v2_snp_matrix.pkl`'s own value-count dict) still
    `MemoryError`s: scipy's dense->sparse path calls `.nonzero()` on the
    full array, and at ~624M nonzero entries the resulting COO coordinate
    array alone (int64 dtype, shape (624_000_000, 2)) is ~9.3 GB -- a
    transient allocation that exists regardless of the target dtype,
    because it is the *positions* of nonzero entries, not their values,
    that dominate.

    The fix actually used here: convert `row_chunk_size` rows at a time
    (each chunk's nonzero-position array is proportionally small), stack
    the per-chunk sparse pieces, and only then cast `.data` to float32 --
    a cast of just the ~624M-long *value* array (~2.5 GB), not the
    (rows, 2) coordinate array. Peak memory during construction is bounded
    by one row-chunk, not the whole matrix.
    """
    n_rows = X_int8.shape[0]
    chunks = []
    for start in range(0, n_rows, row_chunk_size):
        end = min(start + row_chunk_size, n_rows)
        chunks.append(sp.csr_matrix(X_int8[start:end]))
    X_sparse = sp.vstack(chunks, format="csr")
    X_sparse.data = X_sparse.data.astype(np.float32)
    return X_sparse


def fit_pca(
    X_train_sparse: sp.csr_matrix,
    n_components: int,
    *,
    seed: int = 42,
) -> TruncatedSVD:
    return TruncatedSVD(n_components=n_components, random_state=seed).fit(X_train_sparse)


def pca_dim_search(
    X_snp: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    Y: np.ndarray,
    label_mask: np.ndarray,
    label_names: list[str],
    *,
    dims: Optional[list[int]] = None,
    seed: int = 42,
    xgb_params: Optional[dict] = None,
) -> dict[int, dict]:
    """Reproduces Phase B cell 4's PCA dimension search (manifest E7),
    memory-safely (B8).
    """
    import xgboost as xgb  # local import

    dims = dims or DEFAULT_SEARCH_DIMS
    xgb_params = xgb_params or dict(_params["xgb_params"])

    X_sparse = to_sparse(X_snp)
    results: dict[int, dict] = {}

    for dim in dims:
        pca = fit_pca(X_sparse[train_idx], dim, seed=seed)
        Xtr_pca = pca.transform(X_sparse[train_idx])
        Xva_pca = pca.transform(X_sparse[val_idx])
        Xte_pca = pca.transform(X_sparse[test_idx])
        var_ratio = float(pca.explained_variance_ratio_.sum())

        pr_aucs, per_ab = [], {}
        for i, ab in enumerate(label_names):
            m_tr = label_mask[train_idx, i]
            m_te = label_mask[test_idx, i]
            if m_tr.sum() < 50:
                per_ab[ab] = float("nan")
                continue
            clf = xgb.XGBClassifier(random_state=seed, verbosity=0, **xgb_params)
            clf.fit(Xtr_pca[m_tr], Y[train_idx, i][m_tr])
            pred = clf.predict_proba(Xte_pca[m_te])[:, 1]
            pr = safe_pr_auc(Y[test_idx, i][m_te], pred)
            per_ab[ab] = pr
            pr_aucs.append(pr)

        results[dim] = {
            "variance_retained": var_ratio,
            "avg_pr_auc": float(np.nanmean(pr_aucs)) if pr_aucs else float("nan"),
            "per_antibiotic_pr_auc": per_ab,
        }

    return results


def build_pca_features(
    X_snp: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    n_components: int = CHOSEN_DIM,
    seed: int = 42,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, TruncatedSVD, Optional[StandardScaler]]:
    """Build the final PCA-`n_components` feature arrays (train/val/test),
    memory-safely. Matches Phase B cell 5's final master-features build
    (PCA fit on train, then optionally standardized).
    """
    X_sparse = to_sparse(X_snp)
    pca = fit_pca(X_sparse[train_idx], n_components, seed=seed)

    Xtr = pca.transform(X_sparse[train_idx]).astype(np.float32)
    Xva = pca.transform(X_sparse[val_idx]).astype(np.float32)
    Xte = pca.transform(X_sparse[test_idx]).astype(np.float32)

    scaler = None
    if standardize:
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr).astype(np.float32)
        Xva = scaler.transform(Xva).astype(np.float32)
        Xte = scaler.transform(Xte).astype(np.float32)

    return Xtr, Xva, Xte, pca, scaler
