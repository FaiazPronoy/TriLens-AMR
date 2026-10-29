from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trilens.analysis import assembly_size_mechanism as sm
from trilens.analysis import coverage_and_rule as cr
from trilens.analysis import external_robustness as xr

ROOT = Path(__file__).resolve().parents[1]


def test_subset_folds_cover_the_subset_and_keep_sts_whole():
    st = np.array([str(i % 7) for i in range(200)], dtype=object)
    folds, never = cr.subset_folds(st)
    test = np.concatenate([te for _, te in folds])
    assert len(test) == 200 and len(np.unique(test)) == 200 and len(never) == 0
    for tr, te in folds:
        assert not (set(st[tr]) & set(st[te]))
        assert len(np.intersect1d(tr, te)) == 0


def test_subset_folds_put_unknown_st_in_every_training_set():
    st = np.array([None if i < 10 else str(i % 5) for i in range(100)], dtype=object)
    folds, never = cr.subset_folds(st)
    assert set(never) == set(range(10))
    for tr, te in folds:
        assert set(range(10)) <= set(tr) and not (set(range(10)) & set(te))


def test_strata_table_reports_both_sides():
    tab = pd.DataFrame(
        {
            "st": [str(i % 4) for i in range(20)],
            "study": ["s1"] * 10 + ["s2"] * 10,
            "country": ["United Kingdom"] * 10 + ["Norway"] * 10,
        }
    )
    Y = pd.DataFrame({"CTX": [1.0, 0.0] * 10})
    covered = np.array([True] * 10 + [False] * 10)
    s = cr.strata(tab, covered, Y).set_index("stratum")
    assert s.loc["covered", "strains"] == 10 and s.loc["uncovered", "strains"] == 10
    assert s.loc["covered", "pct_uk"] == 100.0 and s.loc["uncovered", "pct_norway"] == 100.0
    assert s.loc["covered", "n_CTX"] == 10


def test_rule_arm_scores_every_called_strain_and_nothing_else(tmp_path):
    pd.DataFrame({"blaCTX-M-15": [1, 0], "blaTEM-1": [0, 1]}, index=pd.Index([3, 7], name="strain_index")).to_csv(
        tmp_path / "acquired.csv"
    )
    pd.DataFrame({"gyrA_S83L": [0, 1]}, index=pd.Index([3, 7], name="strain_index")).to_csv(tmp_path / "point.csv")
    R = cr.rule_arm(tmp_path, np.array([3, 7]), 10)
    assert R.loc[3, "CTX"] == 1.0 and R.loc[7, "CTX"] == 0.0
    assert R.loc[7, "CIP"] == 1.0 and R.loc[3, "CIP"] == 0.0
    assert R.loc[7, "AMP"] == 1.0
    assert R.drop(index=[3, 7]).isna().all().all()


def test_cells_are_st_and_study_together():
    from trilens.evaluation.cluster_bootstrap import within_st_concordance

    n = 400
    rng = np.random.default_rng(0)
    st = np.array([str(i % 4) for i in range(n)], dtype=object)
    study = np.array(["a" if (i // 4) % 2 else "b" for i in range(n)], dtype=object)
    y = (study == "a").astype(float)
    feat = np.where(study == "a", 10.0, 0.0) + rng.normal(0, 0.01, n)
    cells = np.array([f"{s}|{t}" for s, t in zip(st, study)], dtype=object)
    assert within_st_concordance(y, feat, st)["macro"] > 0.95
    assert within_st_concordance(y, feat, cells)["n_st"] == 0


def test_w12_groups_are_disjoint_and_cover_the_statistics():
    assert set(sm.GROUPS["content"]) | set(sm.GROUPS["technical"]) == set(sm.ASM)
    assert not set(sm.GROUPS["content"]) & set(sm.GROUPS["technical"])
    assert sm.PASS_LEVEL == 0.55 and sm.ACQUIRED_DRUGS == ["AMC", "SXT", "GEN", "CTX", "CAZ"]


def test_endpoints_drop_the_excluded_rows():
    n = 300
    rng = np.random.default_rng(1)
    Y = pd.DataFrame({d: rng.integers(0, 2, n).astype(float) for d in xr.PD_NAME})
    S = pd.DataFrame(index=range(n))
    for d in xr.PD_NAME:
        S[f"AP|{d}"] = Y[d] * 0.8 + rng.random(n) * 0.2
        S[f"M|{d}"] = rng.random(n)
        S[f"R|{d}"] = rng.random(n)
    groups = np.array([f"P{i % 6}" for i in range(n)], dtype=object)
    keep = groups != "P0"
    full = xr.endpoints(Y, S, groups, np.ones(n, bool), 60)
    drop = xr.endpoints(Y, S, groups, keep, 60)
    assert full["AP_minus_M"][0] > 0.2 and drop["AP_minus_M"][0] > 0.2
    assert drop["AP_minus_M"][3] == 5 and full["AP_minus_M"][3] == 6


def test_w11_thresholds_are_the_preregistered_ones():
    assert xr.MIN_LABELLED == 100 and xr.OVERLAP_FRACTION == 0.01
    assert xr.PRIMARY == ["AMP", "CIP", "AMC", "SXT", "GEN", "CAZ"]


@pytest.mark.parametrize("delta,lo,expect", [(0.12, 0.07, True), (0.12, -0.01, False), (0.04, 0.01, False)])
def test_w10_part_b_rule_is_delta_and_ci(delta, lo, expect):
    assert bool(delta >= cr.B_DELTA and lo > 0) is expect
