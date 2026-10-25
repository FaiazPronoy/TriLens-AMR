import numpy as np

from src.amr_data.metrics import (
    bootstrap_ci,
    bootstrap_mcc,
    bootstrap_pr_auc,
    clinical_error_rates,
    compute_metrics,
    safe_pr_auc,
)


def test_safe_pr_auc_nan_when_single_class():
    y = np.array([1, 1, 1])
    p = np.array([0.9, 0.8, 0.7])
    assert np.isnan(safe_pr_auc(y, p))


def test_compute_metrics_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y, p, threshold=0.5)
    assert m["pr_auc"] == 1.0
    assert m["mcc"] == 1.0
    assert m["f1"] == 1.0


def test_clinical_error_rates_vme_and_me():
    # 2 resistant (both missed -> VME), 2 susceptible (1 falsely flagged -> ME)
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0, 0, 1, 0])
    rates = clinical_error_rates(y_true, y_pred)
    assert rates["VME"] == 1.0  # both resistant strains missed
    assert rates["ME"] == 0.5   # 1 of 2 susceptible strains falsely flagged
    assert rates["CA"] == 0.25  # only 1/4 correct


def test_bootstrap_ci_reasonable_bounds():
    rng = np.random.RandomState(0)
    n = 500
    y = rng.randint(0, 2, n)
    p = np.clip(y + rng.normal(0, 0.3, n), 0, 1)
    lo, hi = bootstrap_pr_auc(y, p, n_boot=200, seed=1)
    point = safe_pr_auc(y, p)
    assert lo <= point <= hi or abs(point - lo) < 0.05 or abs(point - hi) < 0.05


def test_bootstrap_mcc_returns_nan_for_degenerate_input():
    y = np.zeros(20)
    p = np.random.rand(20)
    lo, hi = bootstrap_mcc(y, p, n_boot=50, seed=1)
    assert np.isnan(lo) and np.isnan(hi)
