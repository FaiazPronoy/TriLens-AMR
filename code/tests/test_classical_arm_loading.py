import numpy as np
import pytest

from src.experiments.run_trilens_v10 import _load_classical_arm

TR, VA, TE = np.arange(10), np.arange(10, 13), np.arange(13, 16)


def _save(tmp_path, **extra):
    arrays = dict(oof_base_probs=np.zeros((10, 6, 7)), test_base_probs=np.zeros((3, 6, 7)))
    arrays.update(extra)
    p = tmp_path / "predictions_saved.npz"
    np.savez(p, **arrays)
    return p


def test_matching_indices_load(tmp_path):
    p = _save(tmp_path, train_indices=TR, test_indices=TE)
    oof, test, val = _load_classical_arm(p, TR, VA, TE)
    assert oof.shape == (10, 5, 7) and test.shape == (3, 5, 7) and val is None


def test_missing_indices_are_an_error(tmp_path):
    p = _save(tmp_path)
    with pytest.raises(ValueError, match="train_indices"):
        _load_classical_arm(p, TR, VA, TE)


def test_reordered_rows_are_an_error(tmp_path):
    p = _save(tmp_path, train_indices=TR[::-1], test_indices=TE)
    with pytest.raises(ValueError, match="differ"):
        _load_classical_arm(p, TR, VA, TE)


def test_unreadable_file_is_an_error(tmp_path):
    p = _save(tmp_path, train_indices=TR, test_indices=TE)
    np.savez(p, something_else=np.zeros(1))
    with pytest.raises(KeyError):
        _load_classical_arm(p, TR, VA, TE)
