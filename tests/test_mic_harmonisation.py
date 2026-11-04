import numpy as np
import pandas as pd
import pytest

from trilens.analysis import mic_harmonisation as mc


@pytest.mark.parametrize(
    "sign,v,expect",
    [("==", 8, (8, 8)), ("<=", 8, (0, 8)), ("<", 8, (0, 4)), (">=", 16, (16, np.inf)), (">", 16, (32, np.inf))],
)
def test_interval(sign, v, expect):
    assert mc.interval(sign, v) == expect


def test_sxt_component_converts_vitek_sulfamethoxazole_values():
    assert mc.sxt_component(19.0) == 1.0 and mc.sxt_component(304.0) == 16.0 and mc.sxt_component(9.5) == 0.5
    assert mc.sxt_component(4.0) == 4.0 and np.isnan(mc.sxt_component(26.95))


@pytest.mark.parametrize(
    "lo,hi,expect", [(16, 16, 1.0), (0, 8, 0.0), (8, 8, 0.0), (0, 16, np.nan), (32, np.inf, 1.0), (4, np.inf, np.nan)]
)
def test_harmonise_ampicillin(lo, hi, expect):
    got = mc.harmonise(lo, hi, 8, 8)
    assert (np.isnan(got) and np.isnan(expect)) or got == expect


def test_harmonise_counts_the_intermediate_zone_as_resistant():
    assert mc.harmonise(2, 2, 1, 4) == 1.0 and mc.harmonise(4, 4, 1, 4) == 1.0
    assert np.isnan(mc.harmonise(0, 2, 1, 4))


@pytest.mark.parametrize(
    "lo,hi,expect",
    [
        (4, 4, "near"),
        (16, 16, "near"),
        (2, 2, "far"),
        (32, 32, "far"),
        (0, 1, "far"),
        (32, np.inf, "far"),
        (0, 8, None),
        (16, np.inf, None),
    ],
)
def test_proximity_ampicillin(lo, hi, expect):
    assert mc.proximity(lo, hi, 8, 8) == expect


def test_m2_detects_errors_concentrated_near_the_breakpoint():
    rng = np.random.default_rng(0)
    n = 3000
    near = rng.random(n) < 0.3
    y = rng.integers(0, 2, n).astype(float)
    wrong = np.where(near, rng.random(n) < 0.4, rng.random(n) < 0.05)
    pred = np.where(wrong, 1 - y, y)
    Y = pd.DataFrame({"AMP": y})
    S = pd.DataFrame({"AP|AMP": pred * 0.9 + 0.05})
    R = pd.DataFrame({"AMP": pred})
    prox = {"AMP": np.where(near, "near", "far").astype(object)}
    groups = np.array([f"P{i % 40}" for i in range(n)], dtype=object)
    out, per = mc.m2(Y, S, R, prox, groups, ["AMP"], n_boot=200)
    assert out["model"]["concentrated"] and out["rule"]["concentrated"] and out["model"]["ratio"] > 4
