"""Shared evaluation metrics: PR-AUC, MCC, F1/F2, VME/ME/CA, bootstrap CI.

Centralized so `baselines.py` and `giessen.py` don't each grow their own
slightly-different copy (the same duplication problem as B10, just
pre-empted here rather than audited after the fact).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    roc_auc_score,
)


def safe_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def safe_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def best_threshold_by_f2(y_true: np.ndarray, y_prob: np.ndarray,
                          grid: np.ndarray | None = None) -> float:
    """Threshold search on VALIDATION predictions only -- callers must not
    pass test-set probabilities here."""
    grid = grid if grid is not None else np.arange(0.05, 0.95, 0.02)
    best_thr, best_f2 = 0.5, -1.0
    for thr in grid:
        pred = (y_prob >= thr).astype(int)
        f2 = fbeta_score(y_true, pred, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2, best_thr = f2, thr
    return float(best_thr)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "threshold": float(threshold),
    }


def clinical_error_rates(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """VME (very major error, false susceptible), ME (major error, false
    resistant), CA (categorical agreement), CLSI-style clinical metrics."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    n_pos = tp + fn
    n_neg = tn + fp
    n = tp + tn + fp + fn
    return {
        "VME": fn / n_pos if n_pos else float("nan"),  # resistant called susceptible
        "ME": fp / n_neg if n_neg else float("nan"),   # susceptible called resistant
        "CA": (tp + tn) / n if n else float("nan"),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 1000,
    seed: int = 42,
    ci_lo: float = 2.5,
    ci_hi: float = 97.5,
) -> tuple[float, float]:
    """Percentile bootstrap CI for any (y_true, y_prob) -> scalar metric."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores: list[float] = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            s = metric_fn(y_true[idx], y_prob[idx])
            if not np.isnan(s):
                scores.append(s)
        except Exception:
            continue
    if len(scores) < max(50, n_boot // 20):
        return float("nan"), float("nan")
    arr = np.asarray(scores)
    return float(np.percentile(arr, ci_lo)), float(np.percentile(arr, ci_hi))


def bootstrap_pr_auc(y_true: np.ndarray, y_prob: np.ndarray, *, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    return bootstrap_ci(y_true, y_prob, safe_pr_auc, n_boot=n_boot, seed=seed)


def bootstrap_mcc(y_true: np.ndarray, y_prob: np.ndarray, *, n_boot: int = 1000, seed: int = 42,
                   threshold: float = 0.5) -> tuple[float, float]:
    def _mcc(y, p):
        pred = (p >= threshold).astype(int)
        if len(np.unique(pred)) < 2:
            return 0.0
        return float(matthews_corrcoef(y, pred))
    return bootstrap_ci(y_true, y_prob, _mcc, n_boot=n_boot, seed=seed)
