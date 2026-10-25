import numpy as np
import pytest

from src.amr_data import amr_ensemblenet as ae


def test_module_importable_without_tensorflow():
    # This module must be importable in an environment with no TF -- the
    # heavy imports are all local to the functions that need them.
    assert ae.VOCAB_SIZE == 5
    assert ae.EMBEDDING_DIM == 64


def test_soft_vote_is_simple_average():
    p_cnn = np.array([0.2, 0.8, 0.5])
    p_xgb = np.array([0.4, 0.6, 0.5])
    result = ae.soft_vote(p_cnn, p_xgb)
    assert np.allclose(result, [0.3, 0.7, 0.5])


def test_xgb_params_match_authors_hyperparameters():
    # Locks in the exact hyperparameters transcribed from the source
    # notebook -- if this ever drifts, it must be a deliberate, documented
    # change (Rule 1), not an accidental edit.
    assert ae.XGB_PARAMS["n_estimators"] == 1000
    assert ae.XGB_PARAMS["learning_rate"] == 0.05
    assert ae.XGB_PARAMS["max_depth"] == 6
    assert ae.XGB_PARAMS["subsample"] == 0.7
    assert ae.XGB_PARAMS["colsample_bytree"] == 0.7


def test_build_cnn_requires_tensorflow_with_clear_error():
    with pytest.raises(ModuleNotFoundError):
        ae.build_cnn1d_model_extended(100)
