import numpy as np
import pandas as pd
import pytest

from trilens.analysis import linear_model_and_clinical_errors as cl


def test_errors_use_the_right_denominators():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, np.nan], float)
    c = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1], float)
    e = cl.errors(y, c)
    assert e["n"] == 10 and e["n_R"] == 4 and e["n_S"] == 6
    assert e["VME"] == pytest.approx(0.25) and e["ME"] == pytest.approx(2 / 6) and e["CA"] == pytest.approx(0.7)
    assert not e["meets_CA"] and not e["meets_ME"]


def test_clopper_pearson_matches_known_values():
    lo, hi = cl.clopper_pearson(0, 100)
    assert lo == 0.0 and hi == pytest.approx(0.0362, abs=1e-3)
    lo, hi = cl.clopper_pearson(5, 100)
    assert lo == pytest.approx(0.0164, abs=1e-3) and hi == pytest.approx(0.1128, abs=1e-3)


@pytest.mark.parametrize(
    "d,lo,hi,expect",
    [
        (0.01, -0.03, 0.04, "a linear model suffices"),
        (0.08, 0.02, 0.15, "non-linearity matters"),
        (-0.08, -0.15, -0.02, "the linear model is better"),
        (0.03, -0.01, 0.07, "inconclusive"),
        (0.06, -0.01, 0.12, "inconclusive"),
    ],
)
def test_reading(d, lo, hi, expect):
    assert cl.reading(d, lo, hi) == expect


def test_pooled_error_diff_counts_only_the_reference_class():
    Y = pd.DataFrame({"AMP": [1, 1, 1, 1, 0, 0, 0, 0.0]})
    a = pd.DataFrame({"AMP": [0, 0, 1, 1, 0, 0, 0, 0.0]})
    b = pd.DataFrame({"AMP": [1, 1, 1, 1, 1, 1, 1, 1.0]})
    g = np.array([f"P{i % 4}" for i in range(8)], dtype=object)
    pt, lo, hi, n = cl.pooled_error_diff(Y, a, b, g, ["AMP"], "VME", n_boot=50)
    assert pt == pytest.approx(0.5) and n == 4
    pt, *_ = cl.pooled_error_diff(Y, a, b, g, ["AMP"], "ME", n_boot=50)
    assert pt == pytest.approx(-1.0)


def test_fixed_settings():
    assert cl.LR == dict(penalty="l1", C=1.0, solver="liblinear", max_iter=2000)
    assert (cl.CA_MIN, cl.ME_MAX, cl.VME_UPPER_MAX) == (0.899, 0.03, 0.075) and cl.MARGIN == 0.05
