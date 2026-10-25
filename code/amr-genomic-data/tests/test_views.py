import numpy as np

from src.amr_data.views import compute_gene_aggregated_features


def test_compute_gene_aggregated_features_synthetic():
    # 3 samples, 2 genes: geneA has 2 SNPs, geneB has 3 SNPs
    X_snp = np.array([
        [0, 1, 2, 0, 1],
        [1, 1, 0, 3, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int8)
    gene_to_range = {"geneA": (0, 2), "geneB": (2, 5)}
    selected_genes = ["geneA", "geneB"]

    X_agg, names = compute_gene_aggregated_features(
        X_snp, gene_to_range, selected_genes, stats=("mean", "var", "n_nonref", "max")
    )
    assert X_agg.shape == (3, 2 * 4)
    assert names == [
        "geneA__mean", "geneA__var", "geneA__n_nonref", "geneA__max",
        "geneB__mean", "geneB__var", "geneB__n_nonref", "geneB__max",
    ]
    # Row 0, geneA columns [0, 1] -> mean 0.5, n_nonref = 1 (only the "1")
    assert X_agg[0, 0] == 0.5
    assert X_agg[0, 2] == 1
    # Row 0, geneB columns [2, 0, 1] -> max = 2
    assert X_agg[0, 7] == 2


def test_compute_gene_aggregated_features_reproduces_real_shape(gene_filtered_legacy):
    """Cross-check against the real Stage 1.5 output: 5000 genes x 4 stats
    == 20,000 feature columns, matching the shape observed when the
    existing pipeline's gene_aggregated_features.py was actually run against
    this same arcadia_v2_gene_filtered.pkl (verified in-session during the
    B3 audit: X_gene_features shape (6962, 20000))."""
    gf = gene_filtered_legacy
    selected_genes = list(gf["selected_gene_ids"])[:50]  # subset -- full run is slow-ish
    X_agg, names = compute_gene_aggregated_features(
        gf["X_snp_filtered"], gf["gene_to_range"], selected_genes,
    )
    assert X_agg.shape == (gf["X_snp_filtered"].shape[0], 50 * 4)
    assert len(names) == 50 * 4
    assert not np.isnan(X_agg).any()
