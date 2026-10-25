import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

from src.amr_data.baselines import build_pca_features, pca_dim_search, to_sparse


def test_to_sparse_preserves_values_and_shape():
    rng = np.random.RandomState(0)
    X = rng.choice([0, 0, 0, 1, 2, 3], size=(50, 200)).astype(np.int8)
    X_sparse = to_sparse(X, row_chunk_size=7)  # deliberately not a divisor of 50
    assert X_sparse.shape == X.shape
    assert X_sparse.dtype == np.float32
    assert np.array_equal(X_sparse.toarray(), X.astype(np.float32))


def test_to_sparse_matches_direct_conversion_small_case():
    rng = np.random.RandomState(1)
    X = rng.choice([0, 0, 0, 1, 2], size=(30, 100)).astype(np.int8)
    chunked = to_sparse(X, row_chunk_size=10)
    direct = sp.csr_matrix(X.astype(np.float32))
    assert np.array_equal(chunked.toarray(), direct.toarray())


def test_sparse_pca_agrees_with_dense_pca_small_synthetic():
    """The actual claim B8's fix rests on: TruncatedSVD on a sparse matrix
    and on the equivalent dense matrix solve the same problem. Verified
    here on a small synthetic matrix (real-scale verification -- dim=1024
    on the full 535,439-SNP matrix -- was run manually against the locked
    `arcadia_v2_master_features.pkl` reference; see docs/CHANGELOG_NUMBERS.md
    for that result, since it takes several minutes and is not suitable as
    a routine pytest case)."""
    rng = np.random.RandomState(0)
    X = rng.choice([0, 0, 0, 1, 2, 3], size=(300, 500)).astype(np.int8)
    X_sparse = to_sparse(X, row_chunk_size=50)
    X_dense = X.astype(np.float32)

    svd_sparse = TruncatedSVD(n_components=10, random_state=42).fit(X_sparse)
    svd_dense = TruncatedSVD(n_components=10, random_state=42).fit(X_dense)

    assert abs(svd_sparse.explained_variance_ratio_.sum()
               - svd_dense.explained_variance_ratio_.sum()) < 1e-6


def test_pca_dim_search_runs_on_real_data_subset(snp_matrix, master_features):
    """Structural smoke test on a SUBSET of the real SNP matrix (first 3,000
    columns, small search dims) -- keeps runtime to seconds while still
    exercising the real pipeline (real labels/splits/masks), rather than a
    fully synthetic dataset."""
    X_snp = snp_matrix["X_snp"][:, :3000]
    m = master_features
    rs = m["random_split"]

    results = pca_dim_search(
        X_snp, rs["train_idx"], rs["val_idx"], rs["test_idx"],
        m["Y"].astype(np.float64), m["label_mask"], list(m["label_names"]),
        dims=[8, 16],
        xgb_params={"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1, "tree_method": "hist"},
    )
    assert set(results.keys()) == {8, 16}
    for dim, r in results.items():
        assert 0.0 <= r["variance_retained"] <= 1.0
        assert not np.isnan(r["avg_pr_auc"])


@pytest.mark.slow
def test_build_pca_features_reproduces_locked_variance_full_scale(snp_matrix, master_features):
    """FULL-SCALE reproduction of manifest E7's dim=1024 variance-retained
    figure against the real, locked arcadia_v2_master_features.pkl. Marked
    slow: builds the full (6962, 535439) sparse matrix (~5 GB, ~3-4 min) and
    fits TruncatedSVD(1024) on the real ~4872-row train partition (several
    more minutes on CPU). Run explicitly with `pytest -m slow`.
    """
    X_snp = snp_matrix["X_snp"]
    rs = master_features["random_split"]
    _, _, _, pca, _ = build_pca_features(
        X_snp, rs["train_idx"], rs["val_idx"], rs["test_idx"],
        n_components=1024, seed=42, standardize=False,
    )
    variance_retained = float(pca.explained_variance_ratio_.sum())
    locked = float(master_features["pca_var_retained"])
    assert abs(variance_retained - locked) < 0.002
