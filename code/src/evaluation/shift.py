"""
Metrics and tests for the generalization experiments (W1, W2, W5, W6).
Definitions: docs/experiments/w1_five_axis.md.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

N_BOOT = 2000


def _both_classes(y):
    return 0 < y.sum() < len(y)


def bootstrap_ci(y, score, metric=average_precision_score, n_boot=N_BOOT, seed=42):
    """Point estimate and percentile 95% CI over resamples of the evaluated strains."""
    y, score = np.asarray(y), np.asarray(score, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if _both_classes(y[i]):
            vals.append(metric(y[i], score[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(metric(y, score)), float(lo), float(hi)


def paired_bootstrap(y, score_a, score_b, metric=average_precision_score, n_boot=N_BOOT, seed=42):
    """metric(a) - metric(b) on the same strains: (delta, ci_lo, ci_hi, two-sided p)."""
    y = np.asarray(y)
    a, b = np.asarray(score_a, float), np.asarray(score_b, float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if _both_classes(y[i]):
            d.append(metric(y[i], a[i]) - metric(y[i], b[i]))
    d = np.asarray(d)
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(metric(y, a) - metric(y, b)), float(lo), float(hi), float(p)


def holm(pvalues):
    """Holm-adjusted p-values, in the input order."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def prevalence_matched_pr_auc(y, score, target_prev, n_rounds=N_BOOT, seed=42):
    """Mean PR-AUC after subsampling negatives (or positives) to target_prev, and the
    2.5-97.5% range of the resampling. Returns (mean, lo, hi, which_class_subsampled)."""
    y, score = np.asarray(y), np.asarray(score, float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)
    n_neg = int(round(len(pos) * (1 - target_prev) / target_prev))
    if n_neg <= len(neg):
        draw = lambda: np.r_[pos, rng.choice(neg, n_neg, replace=False)]
        which = "negatives"
    else:
        n_pos = int(round(len(neg) * target_prev / (1 - target_prev)))
        draw = lambda: np.r_[rng.choice(pos, n_pos, replace=False), neg]
        which = "positives"
    kept_pos, kept_neg = (len(pos), n_neg) if which == "negatives" else (n_pos, len(neg))
    if kept_pos < 1 or kept_neg < 1:
        return np.nan, np.nan, np.nan, which      # a side too small to match
    vals = []
    for _ in range(n_rounds):
        i = draw()
        vals.append(average_precision_score(y[i], score[i]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi), which


def clinical(y, score, threshold=0.5):
    """VME = resistant called susceptible / resistant; ME = susceptible called resistant / susceptible."""
    y = np.asarray(y)
    call = np.asarray(score, float) >= threshold
    r, s = y == 1, y == 0
    vme = float((~call & r).sum() / r.sum()) if r.any() else np.nan
    me = float((call & s).sum() / s.sum()) if s.any() else np.nan
    return vme, me


def safe_roc_auc(y, score):
    return float(roc_auc_score(y, score)) if _both_classes(np.asarray(y)) else np.nan
