import numpy as np
import pytest

from src.amr_data.gene_filter import (
    build_filtered_snp_matrix,
    build_gene_to_snp_ranges,
    compute_gene_importance_for_antibiotic,
    filter_genes,
    get_gene_size_stats,
    parse_gene_id,
    select_genes_for_split,
)


def test_parse_gene_id():
    assert parse_gene_id("LMHECDEF_00004:12:A>T") == ("LMHECDEF_00004", 12)
    assert parse_gene_id("Cluster_1284_+_+_x:99:G>C") == ("Cluster_1284_+_+_x", 99)


def test_parse_gene_id_raises_on_unparseable():
    with pytest.raises(ValueError):
        parse_gene_id("not-a-valid-position-string")


def test_build_gene_to_snp_ranges_groups_correctly():
    positions = np.array([
        "geneA:1:A>T", "geneA:5:C>G", "geneB:1:A>T", "geneA:9:G>A", "geneC:1:T>A",
    ])
    g2s = build_gene_to_snp_ranges(positions)
    assert g2s["geneA"] == [0, 1, 3]
    assert g2s["geneB"] == [2]
    assert g2s["geneC"] == [4]


def test_gene_size_stats():
    g2s = {"a": [0, 1, 2, 3, 4], "b": [5], "c": [6, 7]}
    stats = get_gene_size_stats(g2s)
    assert stats["n_genes"] == 3
    assert stats["total_snps"] == 8
    assert stats["min_size"] == 1
    assert stats["max_size"] == 5
    assert stats["n_with_at_least_5"] == 1  # only gene "a"


def test_build_gene_to_snp_ranges_reproduces_real_stage1_5_input_stats(gene_filtered_legacy_snp_matrix_stats):
    """Cross-check against the real Stage 1.5 manifest (docs Part E5):
    18,975 genes / 535,439 SNPs / median 12 / n>=5 SNPs == 13,799, computed
    from the real arcadia_v2_snp_matrix.pkl, not a copy of the number."""
    stats = gene_filtered_legacy_snp_matrix_stats
    assert stats["n_genes"] == 18975
    assert stats["total_snps"] == 535439
    assert stats["min_size"] == 1
    assert stats["max_size"] == 717
    assert stats["median_size"] == 12.0
    assert stats["n_with_at_least_5"] == 13799


def test_filter_genes_formula_is_intersection_not_union():
    g2s = {"a": [0, 1, 2, 3, 4], "b": [5, 6, 7, 8, 9], "c": [10]}  # c has 1 SNP, fails variability
    importance = {"a": 100.0, "b": 1.0, "c": 999.0}  # c would win on importance alone
    selected = filter_genes(g2s, importance, min_snps_per_gene=5, top_k_by_importance=1)
    # top-1 by importance among variability-passing genes is "a" (100 > 1);
    # "c" must NOT appear despite its high score, because it fails variability.
    assert selected == ["a"]


def test_filter_genes_keep_genes_union_bypasses_importance_cap():
    g2s = {"a": [0, 1, 2, 3, 4], "b": [5, 6, 7, 8, 9], "rescue": [10, 11, 12, 13, 14]}
    importance = {"a": 100.0, "b": 50.0, "rescue": 0.0}  # "rescue" would never make top-1
    selected = filter_genes(
        g2s, importance, min_snps_per_gene=5, top_k_by_importance=1, keep_genes={"rescue"}
    )
    assert "rescue" in selected
    assert "a" in selected  # the top-1 by importance is still included


def test_build_filtered_snp_matrix_shapes_and_ranges():
    X = np.arange(20).reshape(4, 5)  # 4 samples, 5 "SNPs"
    positions = np.array([f"g{i}:1:A>T" for i in range(5)])
    g2s = {f"g{i}": [i] for i in range(5)}
    X_f, pos_f, gene_to_range = build_filtered_snp_matrix(X, positions, g2s, ["g1", "g3"])
    assert X_f.shape == (4, 2)
    assert gene_to_range == {"g1": (0, 1), "g3": (1, 2)}
    assert np.array_equal(X_f[:, 0], X[:, 1])
    assert np.array_equal(X_f[:, 1], X[:, 3])


def test_compute_gene_importance_requires_train_idx():
    with pytest.raises(TypeError, match="requires train_idx"):
        compute_gene_importance_for_antibiotic(np.zeros((10, 10)), np.zeros(10), None, {})


def test_select_genes_for_split_requires_train_idx():
    with pytest.raises(TypeError, match="requires train_idx"):
        select_genes_for_split(
            np.zeros((10, 10)), np.array([f"g{i}:1:A>T" for i in range(10)]),
            np.zeros((10, 1)), np.ones((10, 1), dtype=bool), ["AB"], None, "some_split",
        )
