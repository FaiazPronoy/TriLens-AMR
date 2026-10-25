"""Statistical testing utilities for paired model comparisons."""
from __future__ import annotations

from typing import Callable
import numpy as np
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a metric.
    
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        metric_fn: Function taking (y_true, y_prob) returning float.
        n_bootstrap: Number of bootstrap iterations.
        alpha: Significance level (0.05 = 95% CI).
        seed: Random seed for reproducibility.
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    
    point = metric_fn(y_true, y_prob)
    
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        # Ensure at least one positive and one negative in resample
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            scores.append(metric_fn(y_true[idx], y_prob[idx]))
        except (ValueError, RuntimeWarning):
            continue
    
    scores = np.array(scores)
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    
    return float(point), float(lo), float(hi)


def delong_test(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
) -> tuple[float, float]:
    """DeLong's test for paired AUROC comparison.
    
    Tests whether AUROC of model A differs from model B on same samples.
    
    Args:
        y_true: Ground truth binary labels.
        y_prob_a: Predicted probabilities from model A.
        y_prob_b: Predicted probabilities from model B.
    
    Returns:
        (z_statistic, p_value)
    """
    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)
    
    n1 = np.sum(y_true == 1)
    n0 = np.sum(y_true == 0)
    
    pos_a = y_prob_a[y_true == 1]
    neg_a = y_prob_a[y_true == 0]
    pos_b = y_prob_b[y_true == 1]
    neg_b = y_prob_b[y_true == 0]
    
    # Structural components
    v10_a = np.array([np.mean((p > neg_a) + 0.5 * (p == neg_a)) for p in pos_a])
    v01_a = np.array([np.mean((pos_a > n) + 0.5 * (pos_a == n)) for n in neg_a])
    v10_b = np.array([np.mean((p > neg_b) + 0.5 * (p == neg_b)) for p in pos_b])
    v01_b = np.array([np.mean((pos_b > n) + 0.5 * (pos_b == n)) for n in neg_b])
    
    s10 = np.cov(v10_a, v10_b)
    s01 = np.cov(v01_a, v01_b)
    
    var_a = s10[0, 0] / n1 + s01[0, 0] / n0
    var_b = s10[1, 1] / n1 + s01[1, 1] / n0
    cov_ab = s10[0, 1] / n1 + s01[0, 1] / n0
    
    var_diff = var_a + var_b - 2 * cov_ab
    
    if var_diff <= 0:
        return 0.0, 1.0
    
    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    
    return float(z), float(p)


def cohens_d_paired(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> float:
    """Cohen's d effect size for paired samples.
    
    Args:
        scores_a: Scores from model A across antibiotics.
        scores_b: Scores from model B across antibiotics.
    
    Returns:
        Cohen's d. >0.2 small, >0.5 medium, >0.8 large.
    """
    diff = scores_a - scores_b
    if len(diff) < 2 or diff.std() == 0:
        return 0.0
    return float(diff.mean() / diff.std())


def paired_wilcoxon(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> tuple[float, float]:
    """Wilcoxon signed-rank test for paired model comparison.
    
    Args:
        scores_a: Scores from model A across antibiotics.
        scores_b: Scores from model B across antibiotics.
    
    Returns:
        (statistic, p_value)
    """
    if len(scores_a) < 3:
        return 0.0, 1.0
    try:
        stat, p = stats.wilcoxon(scores_a, scores_b, zero_method="wilcox")
        return float(stat), float(p)
    except ValueError:
        return 0.0, 1.0


def compute_paired_comparison(
    y_true_per_ab: list[np.ndarray],
    probs_a_per_ab: list[np.ndarray],
    probs_b_per_ab: list[np.ndarray],
    ab_names: list[str],
) -> dict[str, dict]:
    """Full statistical comparison between two models across antibiotics.
    
    Args:
        y_true_per_ab: Ground truth per antibiotic.
        probs_a_per_ab: Predictions from model A per antibiotic.
        probs_b_per_ab: Predictions from model B per antibiotic.
        ab_names: Antibiotic names.
    
    Returns:
        Dictionary with per-antibiotic and overall statistics.
    """
    results = {"per_antibiotic": {}, "overall": {}}
    
    pr_a, pr_b = [], []
    auc_a, auc_b = [], []
    
    for i, ab in enumerate(ab_names):
        y = y_true_per_ab[i]
        pa = probs_a_per_ab[i]
        pb = probs_b_per_ab[i]
        
        pr_a_i = average_precision_score(y, pa)
        pr_b_i = average_precision_score(y, pb)
        auc_a_i = roc_auc_score(y, pa)
        auc_b_i = roc_auc_score(y, pb)
        
        z_delong, p_delong = delong_test(y, pa, pb)
        
        results["per_antibiotic"][ab] = {
            "pr_auc_a": pr_a_i,
            "pr_auc_b": pr_b_i,
            "pr_auc_delta": pr_a_i - pr_b_i,
            "auroc_a": auc_a_i,
            "auroc_b": auc_b_i,
            "delong_z": z_delong,
            "delong_p": p_delong,
        }
        
        pr_a.append(pr_a_i)
        pr_b.append(pr_b_i)
        auc_a.append(auc_a_i)
        auc_b.append(auc_b_i)
    
    pr_a = np.array(pr_a)
    pr_b = np.array(pr_b)
    
    stat_pr, p_pr = paired_wilcoxon(pr_a, pr_b)
    d_pr = cohens_d_paired(pr_a, pr_b)
    
    results["overall"] = {
        "avg_pr_auc_a": float(pr_a.mean()),
        "avg_pr_auc_b": float(pr_b.mean()),
        "avg_pr_auc_delta": float((pr_a - pr_b).mean()),
        "wilcoxon_stat": stat_pr,
        "wilcoxon_p": p_pr,
        "cohens_d": d_pr,
        "n_antibiotics": len(ab_names),
    }
    
    return results