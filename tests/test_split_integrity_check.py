import numpy as np

from trilens.analysis import split_integrity_check


def _npz(tmp_path, name, **arrays):
    p = tmp_path / f"{name}.npz"
    np.savez(p, **arrays)
    return np.load(p)


def test_missing_indices_fail(tmp_path):
    a = _npz(tmp_path, "a", test_ens_probs=np.zeros((3, 7)))
    b = _npz(tmp_path, "b", train_indices=np.arange(5), test_indices=np.arange(5, 8))
    ok, detail = split_integrity_check.same_indices(a, b)
    assert not ok and "missing" in detail


def test_same_and_different_order(tmp_path):
    a = _npz(tmp_path, "a", train_indices=np.arange(5), test_indices=np.arange(5, 8))
    b = _npz(tmp_path, "b", train_indices=np.arange(5), test_indices=np.arange(5, 8))
    c = _npz(tmp_path, "c", train_indices=np.arange(5)[::-1], test_indices=np.arange(5, 8))
    assert split_integrity_check.same_indices(a, b)[0]
    assert not split_integrity_check.same_indices(a, c)[0]
