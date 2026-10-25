import numpy as np
import pytest

from src.amr_data.splits import (
    _assert_group_disjoint,
    groups_from_phylogroup,
    lineage_holdout_split,
    phylogroup_stratified_split,
    random_split,
)


# ---------------------------------------------------------------------------
# B7 / random_split -- must reproduce the locked reference exactly
# ---------------------------------------------------------------------------
def test_random_split_matches_locked_reference(master_features):
    """The B7 nan-cast fix must not change the split. Verified against the
    real, currently-in-use arcadia_v2_master_features.pkl, not a synthetic
    stand-in -- this is the split every existing thesis result depends on.
    """
    m = master_features
    Y_bool = m["Y"]
    label_mask = m["label_mask"]
    label_names = list(m["label_names"])
    amp_idx = label_names.index("AMP")

    # Reconstruct the float Y (with NaN where missing) that the original
    # split logic operated on -- master_features.pkl stores the already
    # nan_to_num'd bool array, so mask tells us where the true NaNs were.
    Y_float = np.where(label_mask, Y_bool.astype(np.float64), np.nan)

    result = random_split(Y_float, amp_idx, seed=42)
    locked = m["random_split"]

    assert np.array_equal(result["train_idx"], locked["train_idx"])
    assert np.array_equal(result["val_idx"], locked["val_idx"])
    assert np.array_equal(result["test_idx"], locked["test_idx"])


def test_random_split_no_overlap_synthetic():
    rng = np.random.RandomState(0)
    n = 500
    Y = rng.randint(0, 2, size=(n, 1)).astype(np.float64)
    # inject some missingness
    Y[rng.choice(n, 40, replace=False), 0] = np.nan

    result = random_split(Y, 0)
    tr, va, te = set(result["train_idx"]), set(result["val_idx"]), set(result["test_idx"])
    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)
    assert len(tr | va | te) == n


# ---------------------------------------------------------------------------
# B1 -- phylogroup_stratified_split must NOT be a holdout (documents the bug
# under its honest name so a future refactor can't "fix" it by accident and
# silently change the numbers -- see Rule 1)
# ---------------------------------------------------------------------------
def test_phylogroup_stratified_split_puts_every_group_on_both_sides(master_features):
    cluster_labels = master_features["cluster_labels"]
    result = phylogroup_stratified_split(cluster_labels, seed=42)

    train_groups = set(cluster_labels[result["train_idx"]].tolist())
    test_groups = set(cluster_labels[result["test_idx"]].tolist())
    all_groups = set(cluster_labels.tolist())

    assert train_groups == all_groups
    assert test_groups == all_groups


def test_phylogroup_stratified_split_reproduces_locked_sizes(master_features):
    # Confirms the honest rename didn't change behaviour: identical sizes
    # to random_split (4872/1045/1045), which a genuine holdout could never
    # produce exactly -- this identical-size property IS the evidence for B1.
    cluster_labels = master_features["cluster_labels"]
    result = phylogroup_stratified_split(cluster_labels, seed=42)
    assert (len(result["train_idx"]), len(result["val_idx"]), len(result["test_idx"])) == (4872, 1045, 1045)


# ---------------------------------------------------------------------------
# NEW: lineage_holdout_split -- the highest-priority deliverable
# ---------------------------------------------------------------------------
def test_lineage_holdout_split_is_group_disjoint(master_features):
    cluster_labels = master_features["cluster_labels"]
    groups = groups_from_phylogroup(cluster_labels)
    result = lineage_holdout_split(groups, seed=42)

    g_train = set(groups[result["train_idx"]].tolist())
    g_val = set(groups[result["val_idx"]].tolist())
    g_test = set(groups[result["test_idx"]].tolist())

    assert not (g_train & g_test)
    assert not (g_train & g_val)
    assert not (g_val & g_test)


def test_lineage_holdout_split_sizes_are_uneven_not_locked_reference(master_features):
    cluster_labels = master_features["cluster_labels"]
    groups = groups_from_phylogroup(cluster_labels)
    result = lineage_holdout_split(groups, seed=42)
    sizes = (len(result["train_idx"]), len(result["val_idx"]), len(result["test_idx"]))
    assert sizes != (4872, 1045, 1045), (
        "A genuine group holdout landing on the exact same sizes as the "
        "stratified split would be an enormous coincidence and should be "
        "treated as a bug, not celebrated."
    )
    assert sum(sizes) == len(cluster_labels)


def test_lineage_holdout_split_on_real_30_cluster_grouping(ported_cluster_labels, master_features):
    """The best available grouping variable, now that amr_metax_dataset.pkl
    has been supplied and ported (OPEN_QUESTIONS.md item 1, resolved): 30
    genuinely disjoint clusters, matched 100% by genome name onto the
    current sample order. This supersedes `groups_from_phylogroup()` (only
    8 broad, coarser groups) as the recommended grouping variable."""
    groups = ported_cluster_labels["ported_cluster_labels"]
    result = lineage_holdout_split(groups, seed=42)

    g_train = set(groups[result["train_idx"]].tolist())
    g_val = set(groups[result["val_idx"]].tolist())
    g_test = set(groups[result["test_idx"]].tolist())
    assert not (g_train & g_test) and not (g_train & g_val) and not (g_val & g_test)

    sizes = (len(result["train_idx"]), len(result["val_idx"]), len(result["test_idx"]))
    assert sizes != (4872, 1045, 1045)
    assert sum(sizes) == master_features["n_samples"]

    # Real result (see docs/CHANGELOG_NUMBERS.md): sizes 4912/830/1220,
    # 20/5/5 clusters per split.
    assert len(g_train) + len(g_val) + len(g_test) == 30


def test_assert_group_disjoint_raises_on_synthetic_leak():
    groups = np.array([0, 0, 1, 1, 2, 2])
    leaking_split = {
        "train_idx": np.array([0, 1, 2]),
        "val_idx": np.array([4, 5]),
        "test_idx": np.array([2, 3]),  # group 1 leaks into both train and test
    }
    with pytest.raises(AssertionError, match="LEAK"):
        _assert_group_disjoint(groups, leaking_split)


def test_assert_group_disjoint_passes_on_clean_split():
    groups = np.array([0, 0, 1, 1, 2, 2])
    clean_split = {
        "train_idx": np.array([0, 1]),
        "val_idx": np.array([2, 3]),
        "test_idx": np.array([4, 5]),
    }
    _assert_group_disjoint(groups, clean_split)  # must not raise


def test_lineage_holdout_split_covers_all_rows():
    rng = np.random.RandomState(1)
    n = 1000
    groups = rng.randint(0, 40, size=n)  # 40 synthetic lineages
    result = lineage_holdout_split(groups, seed=7)
    all_idx = np.concatenate([result["train_idx"], result["val_idx"], result["test_idx"]])
    assert len(np.unique(all_idx)) == n
