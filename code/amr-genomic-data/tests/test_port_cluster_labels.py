import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from port_cluster_labels import port_cluster_labels  # noqa: E402


def test_port_cluster_labels_full_match():
    legacy_names = np.array(["a", "b", "c"])
    legacy_clusters = np.array([0, 1, 2])
    v2_names = np.array(["c", "b", "a"])  # different order
    ported, matched, rate = port_cluster_labels(legacy_names, legacy_clusters, v2_names)
    assert rate == 1.0
    assert matched.all()
    assert ported.tolist() == [2, 1, 0]


def test_port_cluster_labels_partial_match():
    legacy_names = np.array(["a", "b"])
    legacy_clusters = np.array([0, 1])
    v2_names = np.array(["a", "b", "c", "d"])  # c, d unknown to legacy
    ported, matched, rate = port_cluster_labels(legacy_names, legacy_clusters, v2_names)
    assert rate == 0.5
    assert matched.tolist() == [True, True, False, False]
    assert ported[2] == -1 and ported[3] == -1


def test_port_cluster_labels_empty_v2():
    ported, matched, rate = port_cluster_labels(np.array([]), np.array([]), np.array([]))
    assert rate == 0.0
    assert len(ported) == 0


# ---------------------------------------------------------------------------
# Real data -- amr_metax_dataset.pkl was supplied after the initial audit.
# OPEN_QUESTIONS.md item 1 is resolved: 100% match rate.
# ---------------------------------------------------------------------------
def test_real_legacy_dataset_has_30_disjoint_clusters(legacy_amr_metax_dataset):
    legacy = legacy_amr_metax_dataset
    assert legacy["n_clusters"] == 30
    assert len(np.unique(legacy["cluster_labels"])) == 30

    ps = legacy["phylogeny_split"]
    tr, va, te = set(ps["train_clusters"].tolist()), set(ps["val_clusters"].tolist()), set(ps["test_clusters"].tolist())
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert tr | va | te == set(range(30))


def test_port_cluster_labels_real_match_rate_is_100_percent(legacy_amr_metax_dataset, master_features):
    legacy_names = np.asarray(legacy_amr_metax_dataset["genome_names"], dtype=str)
    legacy_clusters = np.asarray(legacy_amr_metax_dataset["cluster_labels"])
    v2_names = np.asarray(master_features["genome_names"], dtype=str)

    ported, matched, rate = port_cluster_labels(legacy_names, legacy_clusters, v2_names)
    assert rate == 1.0
    assert matched.all()
    assert (ported >= 0).all()
    assert len(np.unique(ported)) == 30


def test_ported_cluster_labels_pkl_exists_and_is_consistent(ported_cluster_labels, master_features):
    """Checks the actual saved output of `scripts/port_cluster_labels.py`
    (run for real against the real amr_metax_dataset.pkl)."""
    ported = ported_cluster_labels
    assert ported["match_rate"] == 1.0
    assert ported["n_clusters_in_source"] == 30
    assert len(ported["ported_cluster_labels"]) == master_features["n_samples"]
    assert ported["matched_mask"].all()
