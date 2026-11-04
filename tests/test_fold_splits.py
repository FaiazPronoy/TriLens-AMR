import numpy as np

from trilens.splits.fold_splits import load_kfold, st_grouped, strain_st


def test_st_grouped_keeps_each_st_in_one_fold():
    st = np.array(["1"] * 10 + ["2"] * 3 + ["3"] * 3 + [None] * 2 + ["4"], dtype=object)
    folds, never = st_grouped(st, k=3)
    assert sorted(never.tolist()) == [16, 17]
    seen = {}
    for f, idx in enumerate(folds):
        for s in set(st[idx]):
            assert s not in seen
            seen[s] = f
    assert sum(len(f) for f in folds) == 17


def test_committed_splits_cover_all_strains():
    for name in ("st_grouped_5fold", "random_5fold"):
        d = load_kfold(name)
        assert d["n_strains"] == 6962 and len(d["folds"]) == 5
    st = strain_st()
    d = load_kfold("st_grouped_5fold")
    assert all(st[i] is None for i in d["never_tested"])
