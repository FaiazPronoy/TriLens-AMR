from __future__ import annotations

from typing import Dict

import numpy as np

CLSI_VME_MAX = 0.015
CLSI_ME_MAX = 0.03


def _clinical_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    P = TP + FN
    N = TN + FP
    VME = FN / P if P > 0 else np.nan
    ME = FP / N if N > 0 else np.nan
    CA = (TP + TN) / (P + N) if (P + N) > 0 else np.nan
    sens = TP / P if P > 0 else np.nan
    spec = TN / N if N > 0 else np.nan
    PPV = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    NPV = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    return dict(
        threshold=threshold,
        VME=VME,
        ME=ME,
        CA=CA,
        sensitivity=sens,
        specificity=spec,
        PPV=PPV,
        NPV=NPV,
        TP=TP,
        TN=TN,
        FP=FP,
        FN=FN,
    )
