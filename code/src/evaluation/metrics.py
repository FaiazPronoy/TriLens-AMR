"""Standard AMR evaluation metrics."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    f1_score,
    fbeta_score,
    roc_curve,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
)


def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_specificity: float = 0.90,
) -> float:
    """Sensitivity (recall) at fixed specificity threshold.
    
    Standard clinical metric for AMR prediction.
    
    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities for positive class.
        target_specificity: Minimum specificity threshold.
    
    Returns:
        Sensitivity value at the specified specificity, or 0.0 if unreachable.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    valid = specificity >= target_specificity
    if not valid.any():
        return 0.0
    return float(tpr[valid].max())


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (ECE).
    
    Measures probability calibration quality.
    Lower is better. Range [0, 1].
    
    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of bins for calibration histogram.
    
    Returns:
        ECE value.
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (y_prob == bin_edges[i + 1])
        
        if mask.sum() == 0:
            continue
        
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        bin_weight = mask.sum() / n
        
        ece += bin_weight * abs(bin_acc - bin_conf)
    
    return float(ece)


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "mcc",
) -> float:
    """Find optimal threshold maximizing specified metric.
    
    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities.
        metric: One of 'mcc', 'f1', 'f2'.
    
    Returns:
        Threshold value in [0, 1].
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    best_score = -np.inf
    best_thr = 0.5
    
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if len(np.unique(y_pred)) < 2:
            continue
        
        if metric == "mcc":
            score = matthews_corrcoef(y_true, y_pred)
        elif metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "f2":
            score = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_thr = thr
    
    return float(best_thr)


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute all standard AMR evaluation metrics.
    
    Args:
        y_true: Binary ground truth labels.
        y_prob: Predicted probabilities.
        threshold: Classification threshold. If None, uses MCC-optimal threshold.
    
    Returns:
        Dictionary with metrics: pr_auc, auroc, mcc, f1, f2, sens_at_90spec,
        sens_at_95spec, brier, ece, precision, recall, threshold.
    """
    if threshold is None:
        threshold = find_best_threshold(y_true, y_prob, metric="mcc")
    
    y_pred = (y_prob >= threshold).astype(int)
    
    metrics = {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "sens_at_90spec": sensitivity_at_specificity(y_true, y_prob, 0.90),
        "sens_at_95spec": sensitivity_at_specificity(y_true, y_prob, 0.95),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }
    
    return metrics