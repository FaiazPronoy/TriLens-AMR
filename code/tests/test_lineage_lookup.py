import pickle

import numpy as np
import pytest

from src.baselines import lineage_lookup as ll
from src.data.metadata import PROCESSED, ROOT
from src.evaluation import shift

HAVE_DATA = (ROOT / "data" / "raw" / "strains_metadata.csv").exists() and \
    (PROCESSED / "mlst_lineage_holdout_split.pkl").exists()


def test_group_fraction_uses_training_labels_only():
    tr_g, tr_y = np.array(["a", "a", "b"], object), np.array([1, 0, 1])
    s, seen = ll.group_fraction(tr_g, tr_y, np.array(["a", "b", "c", None], object), prior=0.5, alpha=1)
    assert np.allclose(s[:2], [(1 + .5) / 3, (1 + .5) / 2]) and list(seen) == [True, True, False, False]


def test_st_lookup_falls_back_for_unseen_st():
    score, level = ll.st_lookup(np.array(["131", "131"], object), np.array(["B2", "B2"], object), np.array([1, 1]),
                                np.array(["131", "73", None, "9"], object), np.array(["B2", "B2", "D", "D"], object))
    assert list(level) == ["st", "phylogroup", "prior", "prior"]
    assert score[3] == 1.0


def test_hamming_matrix_matches_brute_force():
    rng = np.random.default_rng(0)
    X = rng.integers(0, 2, (30, 50)).astype(np.uint8)
    D = ll.hamming_matrix(X, chunk=7)
    brute = (X[:, None, :] != X[None, :, :]).sum(-1)
    assert np.array_equal(D, brute)


def test_knn_scores_and_tie_breaking():
    D = np.array([[1, 1, 0, 5]])           # train 2 nearest, then a tie between train 0 and 1
    y = np.array([1, 0, 0, 1])
    s = ll.knn_scores(D, y, ks=(1, 2, 3, 4, 50))
    assert s[1][0] == 0 and s[2][0] == 0.5 and s[3][0] == pytest.approx(1 / 3)
    assert s[4][0] == 0.5 and s[50][0] == 0.5     # k capped at the number of training strains


def test_choose_k_prefers_smallest_on_tie():
    y = np.array([0, 1, 0, 1])
    perfect = np.array([0., 1., 0., 1.])
    k, ap = ll.choose_k({5: perfect, 1: perfect, 3: 1 - perfect}, y)
    assert k == 1 and ap == 1.0


def test_shift_helpers():
    assert np.allclose(shift.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = y + rng.normal(0, 1, 300)
    d, lo, hi, p = shift.paired_bootstrap(y, s, s)
    assert d == 0 and lo == 0 and hi == 0 and p == 1.0
    m, lo, hi, which = shift.prevalence_matched_pr_auc(y, s, target_prev=0.7, n_rounds=50)
    assert which == "negatives" and lo <= m <= hi
    assert shift.prevalence_matched_pr_auc(y, s, target_prev=0.2, n_rounds=5)[3] == "positives"
    vme, me = shift.clinical(np.array([1, 1, 0, 0]), np.array([.9, .1, .9, .1]))
    assert vme == 0.5 and me == 0.5


@pytest.mark.skipif(not HAVE_DATA, reason="small_data.zip not unpacked into code/data/")
def test_st_lookup_on_the_st_holdout_never_sees_a_test_st():
    """W2 rule L2: on S3 every test strain falls back below the ST level."""
    from src.data.metadata import load_strain_table
    t = load_strain_table()
    with open(PROCESSED / "mlst_lineage_holdout_split.pkl", "rb") as f:
        sp = pickle.load(f)
    tr, te = np.asarray(sp["train_idx"]), np.asarray(sp["test_idx"])
    y = t["CIP"].values
    tr = tr[~np.isnan(y[tr])]
    st, pg = np.array(t.st.values, object), np.array(t.phylogroup.values, object)
    _, level = ll.st_lookup(st[tr], pg[tr], y[tr].astype(int), st[te], pg[te])
    assert (level != "st").all()
