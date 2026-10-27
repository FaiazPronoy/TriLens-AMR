import numpy as np
import pandas as pd

from src.evaluation.grouped import cluster_bootstrap, group_codes, nap, summary, within_st_concordance


def test_nap_is_zero_for_chance_and_one_for_perfect():
    y = np.array([1, 0, 0, 0] * 50, float)
    assert abs(nap(y, np.ones_like(y))) < 1e-9          # constant score: AP = prevalence
    assert nap(y, y) == 1.0


def test_nap_weights_equal_repetition():
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 40).astype(float), rng.random(40)
    w = rng.integers(0, 3, 40)
    rep = np.repeat(np.arange(40), w)
    assert np.isclose(nap(y, s, w), nap(y[rep], s[rep]))


def test_summary_counts():
    r = summary([1, 0, np.nan, 1], [0.9, 0.1, 0.5, 0.8])
    assert r["n"] == 3 and r["positives"] == 2 and r["auroc"] == 1.0


def test_constant_within_st_scores_half():
    st = ["a"] * 6 + ["b"] * 6
    y = [1, 1, 0, 0, 0, 0] + [1, 1, 1, 1, 0, 0]
    s = [0.2] * 6 + [0.9] * 6                            # ranks STs perfectly, nothing inside
    r = within_st_concordance(y, s, st)
    assert r["macro"] == 0.5 and r["n_st"] == 2 and r["n_pairs"] == 8 + 8


def test_within_st_perfect_and_missing_st_ignored():
    r = within_st_concordance([1, 0, 1, 0], [0.9, 0.1, 0.2, 0.3], ["a", "a", None, None])
    assert r["macro"] == 1.0 and r["n_st"] == 1


def test_group_codes_singletons_for_missing():
    g = group_codes(["5", "5", None, None, "7"])
    assert g[0] == g[1] and g[2] != g[3] and len(set(g)) == 4


def test_cluster_bootstrap_detects_a_real_gain_and_not_a_null():
    rng = np.random.default_rng(1)
    n = 600
    st = rng.integers(0, 60, n)
    y = rng.integers(0, 2, n).astype(float)
    Y = pd.DataFrame({"X": y})
    weak = pd.DataFrame({"X": rng.random(n)})
    strong = pd.DataFrame({"X": y * 0.6 + rng.random(n) * 0.4})
    pt, lo, hi, ng = cluster_bootstrap(Y, weak, strong, st, ["X"], n_boot=200)
    assert pt > 0.3 and lo > 0 and ng == 60
    pt0, lo0, hi0, _ = cluster_bootstrap(Y, weak, weak, st, ["X"], n_boot=50)
    assert pt0 == 0 and lo0 == 0 and hi0 == 0
