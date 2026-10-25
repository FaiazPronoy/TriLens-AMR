import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.amr_data.config import load_paths  # noqa: E402
from src.amr_data.io import load_pickle  # noqa: E402


@pytest.fixture(scope="session")
def paths():
    return load_paths()


@pytest.fixture(scope="session")
def master_features(paths):
    p = paths.processed("master_features_pkl")
    if not p.exists():
        pytest.skip(f"arcadia_v2_master_features.pkl not found at {p}")
    return load_pickle(p)


@pytest.fixture(scope="session")
def gene_filtered_legacy(paths):
    p = paths.processed("gene_filtered_legacy_pkl")
    if not p.exists():
        pytest.skip(f"arcadia_v2_gene_filtered.pkl not found at {p}")
    return load_pickle(p)


@pytest.fixture(scope="session")
def gene_pa(paths):
    p = paths.processed("gene_pa_pkl")
    if not p.exists():
        pytest.skip(f"arcadia_v2_gene_presence_absence.pkl not found at {p}")
    return load_pickle(p)


@pytest.fixture(scope="session")
def snp_matrix(paths):
    p = paths.processed("snp_matrix_pkl")
    if not p.exists():
        pytest.skip(f"arcadia_v2_snp_matrix.pkl not found at {p}")
    return load_pickle(p)


@pytest.fixture(scope="session")
def raw_phenotype_df(paths):
    p = paths.raw("phenotype_matrix")
    if not p.exists():
        pytest.skip(f"phenotype_matrix_08302024.csv not found at {p}")
    return pd.read_csv(p)


@pytest.fixture(scope="session")
def legacy_amr_metax_dataset(paths):
    p = paths.legacy("amr_metax_dataset_pkl")
    if not p.exists():
        pytest.skip(f"amr_metax_dataset.pkl not found at {p}")
    return load_pickle(p)


@pytest.fixture(scope="session")
def ported_cluster_labels(paths):
    p = paths.processed("ported_cluster_labels_pkl")
    if not p.exists():
        pytest.skip(
            f"ported_cluster_labels_v2.pkl not found at {p} -- "
            f"run `python scripts/port_cluster_labels.py` first."
        )
    return load_pickle(p)


@pytest.fixture(scope="session")
def gene_filtered_legacy_snp_matrix_stats(snp_matrix):
    """Real gene-grouping stats computed from the real, full 535,439-SNP
    position array -- used to cross-check gene_filter.py's grouping logic
    against docs manifest E5 without needing to run the (slow) full
    importance-ranking pipeline."""
    from src.amr_data.gene_filter import build_gene_to_snp_ranges, get_gene_size_stats

    g2s = build_gene_to_snp_ranges(snp_matrix["snp_positions"])
    return get_gene_size_stats(g2s)
