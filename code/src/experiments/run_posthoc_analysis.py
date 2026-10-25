"""
src/experiments/run_posthoc_analysis.py
=========================================
Post-hoc analysis on saved CascadeStack v9.1 predictions.
No retraining. Operates entirely on previously saved probability arrays.

EXPs implemented
----------------
EXP-3  : Clinical metrics (VME, ME, CA, sens, spec, PPV, NPV at thr=0.5,
          and at thresholds fixed at 90% / 95% specificity).  CLSI flags.
EXP-4  : Statistical significance — paired bootstrap (1000 reps) + McNemar
          per AB.  Reports ΔPR-AUC with 95% CI + p-value with stars.
EXP-9  : Leave-one-out base-learner ablation: drop each base, refit LogReg
          meta on OOF, evaluate on test.  Quantifies GAN's honest contribution.
EXP-10 : Calibration — Brier score, ECE (10 bins), isotonic recalibration on
          val.  Reliability diagram bin data saved as CSV.
EXP-13 : Per-phylogroup breakdown — PR-AUC by phylogroup for groups ≥ 20
          test samples.

Expected prediction files (v9.1 must save these)
-------------------------------------------------
{v9_results_dir}/predictions_saved.npz  with keys:
    test_ens_probs    (n_test, n_abs)          ensemble sigmoid on test
    test_base_probs   (n_test, n_bases, n_abs) per-base sigmoid on test
    val_ens_probs     (n_val,  n_abs)          ensemble sigmoid on val
    val_base_probs    (n_val,  n_bases, n_abs) per-base sigmoid on val
    oof_base_probs    (n_train,n_bases, n_abs) OOF base probs (meta training)
    test_indices      (n_test,)                row indices into original pkl
    val_indices       (n_val,)
    train_indices     (n_train,)

Y_test and label_mask are ALWAYS loaded from the original data pkl (ground truth).

Dispatch line (add to src/run_experiment.py _DISPATCH):
    "posthoc_analysis": "src.experiments.run_posthoc_analysis",
"""
from __future__ import annotations

import gc
import logging
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from omegaconf import DictConfig
from scipy.stats import chi2 as chi2_dist
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve, brier_score_loss

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── Constants ───────────────────────────────────────────────────────────────

AB_NAMES = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]

# Default order matches v9.1 stacking column order (XGB_gene first, XGB_raw second-to-last).
# If predictions_saved.npz contains a 'base_names' key, PredictionStore overrides this
# automatically — so saving base_names in the npz is the robust long-term fix.
BASE_NAMES = ["XGB_gene", "LGBM", "RF", "LogReg", "XGB_raw", "GAN"]

# v9.1 locked reference PR-AUC (random_split test set, seed=42)
V9_PRAUC_REF = dict(zip(AB_NAMES,
    [0.9412, 0.9774, 0.7569, 0.8346, 0.7037, 0.8931, 0.8215]))

# CLSI thresholds: VME < 3%, ME < 3% for clinical acceptability
# (conservative; CLSI M100 Table 4B uses VME<1.5% for some drug classes)
CLSI_VME_MAX = 0.03
CLSI_ME_MAX  = 0.03


# ─── Prediction Store ─────────────────────────────────────────────────────────

class PredictionStore:
    """
    Loads saved v9.1 predictions from disk and provides convenient access.
    Falls back gracefully when files are missing, logging what is unavailable.
    Ground truth (Y, label_mask) is always loaded from the original pkl.
    """

    def __init__(self, v9_results_dir: str, data_pkl: str) -> None:
        self.v9_dir  = Path(v9_results_dir)
        self.pkl_path = Path(data_pkl)

        # Availability flags — set below
        self.has_test_ens   = False
        self.has_base_preds = False   # test_base_probs
        self.has_oof_probs  = False   # oof_base_probs for EXP-9
        self.has_val_ens    = False   # val predictions for EXP-10

        # Per-instance base names (may be overridden from npz 'base_names' key)
        self.base_names: List[str] = list(BASE_NAMES)

        # Loaded arrays (or None)
        self.test_ens_probs:   Optional[np.ndarray] = None  # (n_test, n_abs)
        self.test_base_probs:  Optional[np.ndarray] = None  # (n_test, n_bases, n_abs)
        self.val_ens_probs:    Optional[np.ndarray] = None  # (n_val, n_abs)
        self.val_base_probs:   Optional[np.ndarray] = None  # (n_val, n_bases, n_abs)
        self.oof_base_probs:   Optional[np.ndarray] = None  # (n_train, n_bases, n_abs)
        self.test_indices:     Optional[np.ndarray] = None  # (n_test,)
        self.val_indices:      Optional[np.ndarray] = None
        self.train_indices:    Optional[np.ndarray] = None

        # Ground truth (loaded from pkl, always available)
        self.Y:          np.ndarray = None  # (N, n_abs)
        self.label_mask: np.ndarray = None  # (N, n_abs) bool
        self.random_split:  dict    = None
        self.phylogroup_onehot: Optional[np.ndarray] = None  # (N, n_phylo)
        self.phylogroup_names:  Optional[List[str]]  = None

        self._load_ground_truth()
        self._load_predictions()

    # ─── Ground truth ───────────────────────────────────────────────────────

    def _load_ground_truth(self) -> None:
        log.info(f"Loading ground truth from {self.pkl_path}")
        with open(self.pkl_path, "rb") as f:
            raw = pickle.load(f)
        self.Y           = raw["Y"].astype(np.float32)
        self.label_mask  = raw["label_mask"].astype(bool)
        self.random_split = raw["random_split"]
        # Derive canonical test/val/train indices from random_split
        if self.test_indices is None:
            self.test_indices  = np.array(self.random_split.get("test") or self.random_split["test_idx"])
            self.val_indices   = np.array(self.random_split.get("val") or self.random_split["val_idx"])
            self.train_indices = np.array(self.random_split.get("train") or self.random_split["train_idx"])
        # Phylogroup
        if "phylogroup_onehot" in raw:
            self.phylogroup_onehot = raw["phylogroup_onehot"].astype(np.float32)
        elif "phylogroup" in raw:
            # Try to one-hot encode string labels
            pg = np.array(raw["phylogroup"])
            unique_pg = sorted(set(pg))
            onehot = np.zeros((len(pg), len(unique_pg)), dtype=np.float32)
            for i, g in enumerate(unique_pg):
                onehot[:, i] = (pg == g).astype(float)
            self.phylogroup_onehot = onehot
            self.phylogroup_names  = unique_pg
        else:
            log.warning("No phylogroup data in pkl — EXP-13 will be skipped")

    # ─── Predictions ────────────────────────────────────────────────────────

    def _load_predictions(self) -> None:
        pred_path = self.v9_dir / "predictions_saved.npz"
        if not pred_path.exists():
            log.warning(
                f"Prediction file not found: {pred_path}\n"
                f"  Add the following to src/experiments/run_cascadestack_v9.py "
                f"to enable post-hoc analysis:\n"
                f"    np.savez_compressed(\n"
                f"      results_dir / 'predictions_saved.npz',\n"
                f"      test_ens_probs=<(n_test,n_abs) float32>,\n"
                f"      test_base_probs=<(n_test,n_bases,n_abs) float32>,\n"
                f"      val_ens_probs=<(n_val,n_abs) float32>,\n"
                f"      val_base_probs=<(n_val,n_bases,n_abs) float32>,\n"
                f"      oof_base_probs=<(n_train,n_bases,n_abs) float32>,\n"
                f"      test_indices=test_idx, val_indices=val_idx,\n"
                f"      train_indices=train_idx,\n"
                f"    )"
            )
            return

        log.info(f"Loading predictions from {pred_path}")
        data = np.load(str(pred_path), allow_pickle=False)

        def _get(key: str, fallback_shape: Optional[tuple] = None) -> Optional[np.ndarray]:
            if key in data:
                arr = data[key].astype(np.float32)
                log.info(f"  Loaded {key}: shape={arr.shape}")
                return arr
            log.warning(f"  Key '{key}' not found in predictions_saved.npz")
            return None

        self.test_ens_probs  = _get("test_ens_probs")
        self.test_base_probs = _get("test_base_probs")
        self.val_ens_probs   = _get("val_ens_probs")
        self.val_base_probs  = _get("val_base_probs")
        self.oof_base_probs  = _get("oof_base_probs")

        if "test_indices" in data:
            self.test_indices  = data["test_indices"].astype(int)
        if "val_indices" in data:
            self.val_indices   = data["val_indices"].astype(int)
        if "train_indices" in data:
            self.train_indices = data["train_indices"].astype(int)

        # Override base_names from npz if saved — this is the robust fix for
        # column-order mismatches between the hardcoded default and the actual
        # v9.1 stacking order.  Save in v9.1: base_names=np.array([...names...])
        if "base_names" in data:
            self.base_names = [str(n) for n in data["base_names"]]
            log.info(f"  Base order from npz: {self.base_names}")
        else:
            log.info(
                f"  'base_names' not in npz — using default order: {self.base_names}\n"
                f"  ⚠️  Verify this matches your v9.1 stacking column order!"
            )

        self.has_test_ens   = self.test_ens_probs is not None
        self.has_base_preds = self.test_base_probs is not None
        self.has_oof_probs  = self.oof_base_probs is not None
        self.has_val_ens    = self.val_ens_probs is not None

    # ─── Convenience accessors ───────────────────────────────────────────────

    def get_test_ground_truth(self, ab_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_prob_ens) for valid test samples of ab_idx."""
        valid = self.label_mask[self.test_indices, ab_idx]
        y_true = self.Y[self.test_indices[valid], ab_idx]
        if self.has_test_ens:
            y_prob = self.test_ens_probs[valid, ab_idx]
        else:
            y_prob = np.full(valid.sum(), np.nan)
        return y_true, y_prob

    def get_val_ground_truth(self, ab_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        valid = self.label_mask[self.val_indices, ab_idx]
        y_true = self.Y[self.val_indices[valid], ab_idx]
        if self.has_val_ens:
            y_prob = self.val_ens_probs[valid, ab_idx]
        else:
            y_prob = np.full(valid.sum(), np.nan)
        return y_true, y_prob

    def get_base_test_probs(self, ab_idx: int, base_idx: int) -> Optional[np.ndarray]:
        """Return base-learner test probabilities for valid test samples."""
        if not self.has_base_preds:
            return None
        valid = self.label_mask[self.test_indices, ab_idx]
        return self.test_base_probs[valid, base_idx, ab_idx]

    @property
    def n_bases(self) -> int:
        if self.test_base_probs is not None:
            return self.test_base_probs.shape[1]
        return len(BASE_NAMES)


# ─── EXP-3: Clinical Metrics ──────────────────────────────────────────────────

def _clinical_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute full clinical metric set at a given decision threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    P  = TP + FN   # all positives (resistant)
    N  = TN + FP   # all negatives (susceptible)

    # CLSI definitions for susceptibility testing
    # VME (Very Major Error): resistant classified as susceptible — FN among true resistant
    # ME  (Major Error):      susceptible classified as resistant  — FP among true susceptible
    VME  = FN / P if P > 0 else np.nan
    ME   = FP / N if N > 0 else np.nan
    CA   = (TP + TN) / (P + N) if (P + N) > 0 else np.nan
    sens = TP / P if P > 0 else np.nan
    spec = TN / N if N > 0 else np.nan
    PPV  = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    NPV  = TN / (TN + FN) if (TN + FN) > 0 else np.nan

    return dict(threshold=threshold, VME=VME, ME=ME, CA=CA,
                sensitivity=sens, specificity=spec, PPV=PPV, NPV=NPV,
                TP=TP, TN=TN, FP=FP, FN=FN)


def _find_threshold_at_specificity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    target_spec: float,
) -> Optional[float]:
    """
    Find the smallest decision threshold that achieves >= target_spec specificity.
    (Smallest = most sensitive operating point still meeting the spec constraint.)
    Returns None if the target cannot be achieved.
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specs = 1.0 - fpr
    valid = specs >= target_spec
    if not valid.any():
        return None
    # Among valid (spec ≥ target), pick the one with highest TPR:
    # because tpr is non-decreasing as threshold decreases,
    # the last valid index maximises sensitivity.
    best_idx = np.where(valid)[0][-1]
    # thresholds[0] is a virtual +∞ point; handle edge case
    best_idx = min(best_idx, len(thresholds) - 1)
    return float(thresholds[best_idx])


def run_clinical_metrics(
    store: PredictionStore,
    specificity_targets: List[float],
) -> List[Dict]:
    """EXP-3: Clinical metrics per AB at multiple thresholds."""
    if not store.has_test_ens:
        log.warning("EXP-3 skipped: test_ens_probs not in predictions_saved.npz")
        return []

    rows = []
    for ab_idx, ab_name in enumerate(AB_NAMES):
        y_true, y_prob = store.get_test_ground_truth(ab_idx)
        if len(y_true) < 2 or y_true.sum() == 0:
            log.warning(f"  EXP-3/{ab_name}: degenerate labels, skipping")
            continue

        # Threshold 0.5 (default)
        m05 = _clinical_at_threshold(y_true, y_prob, 0.5)
        m05["ab"]         = ab_name
        m05["thr_source"] = "fixed_0.5"
        m05["CLSI_VME_ok"] = (m05["VME"] <= CLSI_VME_MAX) if not np.isnan(m05["VME"]) else None
        m05["CLSI_ME_ok"]  = (m05["ME"]  <= CLSI_ME_MAX)  if not np.isnan(m05["ME"])  else None
        rows.append(m05)

        # Specificity-fixed thresholds (90%, 95%)
        for target_spec in specificity_targets:
            t = _find_threshold_at_specificity(y_true, y_prob, target_spec)
            if t is None:
                log.warning(
                    f"  EXP-3/{ab_name}: cannot achieve {target_spec*100:.0f}% specificity"
                )
                continue
            m = _clinical_at_threshold(y_true, y_prob, t)
            m["ab"]         = ab_name
            m["thr_source"] = f"spec_{int(target_spec*100)}pct"
            m["CLSI_VME_ok"] = (m["VME"] <= CLSI_VME_MAX) if not np.isnan(m["VME"]) else None
            m["CLSI_ME_ok"]  = (m["ME"]  <= CLSI_ME_MAX)  if not np.isnan(m["ME"])  else None
            rows.append(m)

        log.info(
            f"  EXP-3/{ab_name}: VME={m05['VME']:.3f}  ME={m05['ME']:.3f}  "
            f"CA={m05['CA']:.3f}  sens={m05['sensitivity']:.3f}  "
            f"spec={m05['specificity']:.3f}  "
            f"CLSI_ok={'Y' if (m05['CLSI_VME_ok'] and m05['CLSI_ME_ok']) else 'N'}"
        )

    return rows


# ─── EXP-4: Statistical Significance ────────────────────────────────────────

def _sig_stars(p: float) -> str:
    if np.isnan(p):    return "n/a"
    if p < 0.001:      return "***"
    if p < 0.01:       return "**"
    if p < 0.05:       return "*"
    return "ns"


def _paired_bootstrap(
    y_true: np.ndarray,
    y_ens:  np.ndarray,
    y_base: np.ndarray,
    n_boot: int = 1000,
    seed:   int = 42,
) -> Tuple[float, float, float, float]:
    """
    Paired bootstrap test for difference in PR-AUC.
    Returns (observed_delta, ci_lo, ci_hi, p_two_sided).
    p-value: permutation-style using centered bootstrap distribution.
    """
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return np.nan, np.nan, np.nan, np.nan

    obs_ens  = average_precision_score(y_true, y_ens)
    obs_base = average_precision_score(y_true, y_base)
    obs_delta = obs_ens - obs_base

    rng  = np.random.default_rng(seed)
    n    = len(y_true)
    boot = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        try:
            d = (average_precision_score(yt, y_ens[idx])
                 - average_precision_score(yt, y_base[idx]))
            boot.append(d)
        except Exception:
            pass

    if len(boot) < 50:
        return obs_delta, np.nan, np.nan, np.nan

    boot  = np.array(boot)
    ci_lo = float(np.percentile(boot, 2.5))
    ci_hi = float(np.percentile(boot, 97.5))

    # Two-sided p: center boot distribution at 0, test |δ_shifted| ≥ |observed|
    shifted = boot - boot.mean()
    p = float(np.mean(np.abs(shifted) >= abs(obs_delta)))
    # Guard: never exactly 0 (avoid log(0) downstream)
    p = max(p, 1.0 / (n_boot + 1))

    return float(obs_delta), ci_lo, ci_hi, p


def _mcnemar_test(
    y_true: np.ndarray,
    y_pred_ens: np.ndarray,   # binarized
    y_pred_base: np.ndarray,  # binarized
) -> float:
    """McNemar test with continuity correction. Returns p-value."""
    b = int(((y_pred_ens == y_true) & (y_pred_base != y_true)).sum())  # ens ✓, base ✗
    c = int(((y_pred_ens != y_true) & (y_pred_base == y_true)).sum())  # ens ✗, base ✓
    denom = b + c
    if denom == 0:
        return np.nan  # no discordant pairs
    # McNemar with Yates' continuity correction
    statistic = (max(abs(b - c) - 1, 0) ** 2) / denom
    return float(chi2_dist.sf(statistic, df=1))


def run_significance_tests(
    store: PredictionStore,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> List[Dict]:
    """EXP-4: Paired bootstrap + McNemar per AB vs each base learner."""
    if not store.has_base_preds:
        log.warning("EXP-4 skipped: test_base_probs not in predictions_saved.npz")
        return []

    n_bases = store.n_bases
    base_names = store.base_names[:n_bases]   # use instance-level names (may come from npz)
    rows = []

    for ab_idx, ab_name in enumerate(AB_NAMES):
        y_true, y_ens = store.get_test_ground_truth(ab_idx)
        if len(y_true) < 2 or y_true.sum() == 0:
            continue

        y_pred_ens = (y_ens >= 0.5).astype(int)

        for b_idx, b_name in enumerate(base_names):
            y_base = store.get_base_test_probs(ab_idx, b_idx)
            if y_base is None:
                continue

            obs_delta, ci_lo, ci_hi, p_boot = _paired_bootstrap(
                y_true, y_ens, y_base, n_bootstrap, bootstrap_seed
            )
            y_pred_base = (y_base >= 0.5).astype(int)
            p_mcnemar   = _mcnemar_test(y_true, y_pred_ens, y_pred_base)

            prauc_base  = float(average_precision_score(y_true, y_base)) \
                          if y_true.sum() > 0 else np.nan
            prauc_ens   = float(average_precision_score(y_true, y_ens)) \
                          if y_true.sum() > 0 else np.nan

            rows.append({
                "ab":           ab_name,
                "base":         b_name,
                "prauc_ens":    round(prauc_ens,  5),
                "prauc_base":   round(prauc_base, 5),
                "delta_prauc":  round(obs_delta,  5) if not np.isnan(obs_delta) else np.nan,
                "ci_lo":        round(ci_lo, 5)       if not np.isnan(ci_lo)     else np.nan,
                "ci_hi":        round(ci_hi, 5)       if not np.isnan(ci_hi)     else np.nan,
                "p_bootstrap":  round(p_boot, 5)      if not np.isnan(p_boot)    else np.nan,
                "p_mcnemar":    round(p_mcnemar, 5)   if not np.isnan(p_mcnemar) else np.nan,
                "sig_bootstrap": _sig_stars(p_boot),
                "sig_mcnemar":   _sig_stars(p_mcnemar),
            })

        log.info(f"  EXP-4/{ab_name}: computed vs {len(base_names)} bases")

    return rows


# ─── EXP-9: Leave-One-Out Base-Learner Ablation ──────────────────────────────

def run_loo_ablation(
    store: PredictionStore,
    meta_C: float = 1.0,
    seed:   int   = 42,
) -> List[Dict]:
    """
    EXP-9: For each base learner, refit LogReg meta WITHOUT it, evaluate on test.
    Measures each base's marginal contribution including GAN's honest contribution.
    Requires oof_base_probs to be saved; logs instructions if missing.
    """
    if not store.has_oof_probs:
        log.warning(
            "EXP-9 skipped: oof_base_probs not in predictions_saved.npz.\n"
            "  Add to v9.1 runner: save oof_base_probs (n_train, n_bases, n_abs) "
            "before fitting the meta-learner."
        )
        return []
    if not store.has_test_ens or not store.has_base_preds:
        log.warning("EXP-9 skipped: also needs test_ens_probs and test_base_probs")
        return []

    n_bases    = store.n_bases
    base_names = store.base_names[:n_bases]   # use instance-level names (may come from npz)
    rows       = []

    for ab_idx, ab_name in enumerate(AB_NAMES):
        y_true, _ = store.get_test_ground_truth(ab_idx)
        if len(y_true) < 5:
            continue

        valid_tr = store.label_mask[store.train_indices, ab_idx]
        valid_te = store.label_mask[store.test_indices,  ab_idx]

        if valid_tr.sum() < 10 or valid_te.sum() < 2:
            continue

        # OOF base probs for this AB on valid train samples
        oof_X = store.oof_base_probs[valid_tr, :, ab_idx]   # (n_tr_valid, n_bases)
        y_oof = store.Y[store.train_indices[valid_tr], ab_idx]  # (n_tr_valid,)

        # Test base probs for this AB on valid test samples
        test_X = store.test_base_probs[valid_te, :, ab_idx]  # (n_te_valid, n_bases)
        y_te   = y_true                                       # already filtered

        # Full ensemble (all bases) PR-AUC — should match locked v9.1 results
        lr_full = LogisticRegression(max_iter=2000, C=meta_C, random_state=seed)
        lr_full.fit(oof_X, y_oof)
        prob_full = lr_full.predict_proba(test_X)[:, 1]
        prauc_full = float(average_precision_score(y_te, prob_full)) \
                     if y_te.sum() > 0 else np.nan

        for b_idx, b_name in enumerate(base_names):
            # Drop base b_idx
            keep_cols = [i for i in range(n_bases) if i != b_idx]
            oof_X_loo  = oof_X[:,  keep_cols]
            test_X_loo = test_X[:, keep_cols]

            lr_loo = LogisticRegression(max_iter=2000, C=meta_C, random_state=seed)
            lr_loo.fit(oof_X_loo, y_oof)
            prob_loo = lr_loo.predict_proba(test_X_loo)[:, 1]
            prauc_loo = float(average_precision_score(y_te, prob_loo)) \
                        if y_te.sum() > 0 else np.nan

            delta = prauc_full - prauc_loo   # positive = base was helpful

            rows.append({
                "ab":           ab_name,
                "base_removed": b_name,
                "prauc_full":   round(prauc_full, 5),
                "prauc_loo":    round(prauc_loo,  5),
                "delta":        round(delta, 5),    # + means base helped
                "pct_change":   round(delta / max(prauc_full, 1e-9) * 100, 3),
            })

        log.info(
            f"  EXP-9/{ab_name}: full={prauc_full:.4f}  "
            + "  ".join(
                f"{b_name}:-{prauc_full - r['prauc_loo']:.4f}"
                for b_name, r in zip(base_names, rows[-n_bases:])
            )
        )

    return rows


# ─── EXP-10: Calibration ─────────────────────────────────────────────────────

def _compute_ece(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected Calibration Error (uniform bins)."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n   = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        if mask.sum() == 0:
            continue
        frac_samples  = mask.sum() / n
        mean_conf     = float(y_prob[mask].mean())
        mean_acc      = float(y_true[mask].mean())
        ece           += frac_samples * abs(mean_conf - mean_acc)
    return float(ece)


def _reliability_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (bin_midpoints, mean_predicted, observed_fraction, counts).
    For plotting reliability (calibration) diagrams.
    """
    bin_edges  = np.linspace(0.0, 1.0, n_bins + 1)
    mids       = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mean_preds = np.full(n_bins, np.nan)
    obs_fracs  = np.full(n_bins, np.nan)
    counts     = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask   = (y_prob >= lo) & (y_prob <= hi if i == n_bins - 1 else y_prob < hi)
        counts[i] = mask.sum()
        if mask.sum() == 0:
            continue
        mean_preds[i] = float(y_prob[mask].mean())
        obs_fracs[i]  = float(y_true[mask].mean())

    return mids, mean_preds, obs_fracs, counts


def run_calibration(
    store:   PredictionStore,
    n_bins:  int = 10,
) -> Tuple[List[Dict], List[Dict]]:
    """
    EXP-10: Brier score, ECE, isotonic recalibration, reliability diagram data.
    Returns (calibration_metrics_rows, reliability_bin_rows).
    """
    if not store.has_test_ens:
        log.warning("EXP-10 skipped: test_ens_probs not available")
        return [], []

    metric_rows = []
    bin_rows    = []

    for ab_idx, ab_name in enumerate(AB_NAMES):
        y_true_te, y_prob_te = store.get_test_ground_truth(ab_idx)

        if len(y_true_te) < 2 or y_true_te.sum() == 0:
            continue

        brier_raw = float(brier_score_loss(y_true_te, y_prob_te))
        ece_raw   = _compute_ece(y_true_te, y_prob_te, n_bins)

        # Isotonic recalibration on val
        brier_cal = np.nan
        ece_cal   = np.nan
        y_prob_te_cal = y_prob_te

        if store.has_val_ens:
            y_true_va, y_prob_va = store.get_val_ground_truth(ab_idx)
            if len(y_true_va) >= 10 and y_true_va.sum() > 0:
                try:
                    ir = IsotonicRegression(out_of_bounds="clip")
                    ir.fit(y_prob_va, y_true_va)
                    y_prob_te_cal = ir.predict(y_prob_te)
                    brier_cal     = float(brier_score_loss(y_true_te, y_prob_te_cal))
                    ece_cal       = _compute_ece(y_true_te, y_prob_te_cal, n_bins)
                except Exception as exc:
                    log.warning(f"  EXP-10/{ab_name}: isotonic fit failed: {exc}")

        metric_rows.append({
            "ab":         ab_name,
            "brier_raw":  round(brier_raw, 6),
            "brier_cal":  round(brier_cal, 6) if not np.isnan(brier_cal) else np.nan,
            "ece_raw":    round(ece_raw, 6),
            "ece_cal":    round(ece_cal, 6)   if not np.isnan(ece_cal)   else np.nan,
            "brier_improvement": round(brier_raw - brier_cal, 6)
                                 if not np.isnan(brier_cal) else np.nan,
        })

        log.info(
            f"  EXP-10/{ab_name}: brier={brier_raw:.4f}→{brier_cal:.4f}  "
            f"ece={ece_raw:.4f}→{ece_cal:.4f}"
        )

        # Reliability diagram bins (before calibration)
        mids, mean_preds, obs_fracs, counts = _reliability_bins(
            y_true_te, y_prob_te, n_bins
        )
        for i, (mid, mp, of, cnt) in enumerate(
            zip(mids, mean_preds, obs_fracs, counts)
        ):
            bin_rows.append({
                "ab":           ab_name,
                "calibrated":   False,
                "bin_midpoint": float(mid),
                "mean_predicted": float(mp) if not np.isnan(mp) else np.nan,
                "observed_frac":  float(of) if not np.isnan(of) else np.nan,
                "count":          int(cnt),
            })

        # After calibration
        if not np.isnan(brier_cal):
            mids2, mp2, of2, cnt2 = _reliability_bins(y_true_te, y_prob_te_cal, n_bins)
            for mid, mp, of, cnt in zip(mids2, mp2, of2, cnt2):
                bin_rows.append({
                    "ab":           ab_name,
                    "calibrated":   True,
                    "bin_midpoint": float(mid),
                    "mean_predicted": float(mp) if not np.isnan(mp) else np.nan,
                    "observed_frac":  float(of) if not np.isnan(of) else np.nan,
                    "count":          int(cnt),
                })

    return metric_rows, bin_rows


# ─── EXP-13: Per-Phylogroup Breakdown ────────────────────────────────────────

def run_phylogroup_analysis(
    store:        PredictionStore,
    min_samples:  int = 20,
) -> List[Dict]:
    """EXP-13: Per-phylogroup PR-AUC for groups ≥ min_samples test samples."""
    if store.phylogroup_onehot is None:
        log.warning("EXP-13 skipped: phylogroup data not found in pkl")
        return []
    if not store.has_test_ens:
        log.warning("EXP-13 skipped: test_ens_probs not available")
        return []

    n_groups = store.phylogroup_onehot.shape[1]
    if store.phylogroup_names is None:
        pg_names = [f"phylogroup_{i}" for i in range(n_groups)]
    else:
        pg_names = store.phylogroup_names

    # Phylogroup membership for test samples
    pg_onehot_test = store.phylogroup_onehot[store.test_indices, :]

    rows = []
    for pg_idx in range(n_groups):
        pg_mask = pg_onehot_test[:, pg_idx].astype(bool)
        n_pg    = pg_mask.sum()
        if n_pg < min_samples:
            continue

        pg_name = pg_names[pg_idx] if pg_idx < len(pg_names) else f"pg_{pg_idx}"

        for ab_idx, ab_name in enumerate(AB_NAMES):
            # Intersection: phylogroup & valid label
            label_valid = store.label_mask[store.test_indices, ab_idx]
            combined    = pg_mask & label_valid
            n_combined  = combined.sum()
            if n_combined < 10:
                continue

            y_true = store.Y[store.test_indices[combined], ab_idx]
            y_prob = store.test_ens_probs[combined, ab_idx]

            if y_true.sum() == 0 or y_true.sum() == n_combined:
                rows.append({
                    "phylogroup": pg_name, "ab": ab_name,
                    "n_test": int(n_combined), "n_pos": int(y_true.sum()),
                    "prauc": np.nan, "delta_vs_overall": np.nan,
                    "note": "degenerate_labels",
                })
                continue

            prauc = float(average_precision_score(y_true, y_prob))
            overall_prauc = V9_PRAUC_REF.get(ab_name, np.nan)
            delta = prauc - overall_prauc if not np.isnan(overall_prauc) else np.nan

            rows.append({
                "phylogroup": pg_name,
                "ab":         ab_name,
                "n_test":     int(n_combined),
                "n_pos":      int(y_true.sum()),
                "prauc":      round(prauc, 5),
                "delta_vs_overall": round(delta, 5) if not np.isnan(delta) else np.nan,
                "note": "underperforms" if not np.isnan(delta) and delta < -0.05 else "",
            })

    if rows:
        log.info(
            f"  EXP-13: analysed {n_groups} phylogroups, "
            f"{len(rows)} rows with n_test >= {min_samples}"
        )
    return rows


# ─── Output Writing ──────────────────────────────────────────────────────────

def save_outputs(
    exp3_rows:     List[Dict],
    exp4_rows:     List[Dict],
    exp9_rows:     List[Dict],
    exp10_metrics: List[Dict],
    exp10_bins:    List[Dict],
    exp13_rows:    List[Dict],
    output_dir:    Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    def _save(rows: List[Dict], fname: str) -> None:
        if rows:
            pd.DataFrame(rows).to_csv(output_dir / fname, index=False)
            log.info(f"Saved {fname}  ({len(rows)} rows)")
        else:
            log.warning(f"Skipped {fname} (no data)")

    _save(exp3_rows,     "clinical_metrics.csv")
    _save(exp4_rows,     "significance_tests.csv")
    _save(exp9_rows,     "leave_one_out.csv")
    _save(exp10_metrics, "calibration.csv")
    _save(exp10_bins,    "reliability_bins.csv")
    _save(exp13_rows,    "per_phylogroup.csv")


def write_summary(
    exp3_rows:     List[Dict],
    exp4_rows:     List[Dict],
    exp9_rows:     List[Dict],
    exp10_metrics: List[Dict],
    exp13_rows:    List[Dict],
    output_dir:    Path,
) -> None:
    lines = [
        "# CascadeStack v9.1 — Post-hoc Analysis Summary",
        "",
        f"*Results computed on test split (random_split, seed=42)*",
        "",
        "---",
        "",
    ]

    # ── EXP-3 ────────────────────────────────────────────────────────────────
    lines += ["## EXP-3: Clinical Metrics (threshold = 0.5)", ""]
    if exp3_rows:
        df3  = pd.DataFrame(exp3_rows)
        df05 = df3[df3["thr_source"] == "fixed_0.5"]
        hdr  = "| AB | VME% | ME% | CA% | Sens | Spec | PPV | NPV | CLSI-VME | CLSI-ME |"
        sep  = "|----|----|----|----|------|------|-----|-----|----------|---------|"
        lines += [hdr, sep]
        for _, r in df05.iterrows():
            vme_ok  = "✅" if r["CLSI_VME_ok"] else ("❌" if r["CLSI_VME_ok"] is not None else "—")
            me_ok   = "✅" if r["CLSI_ME_ok"]  else ("❌" if r["CLSI_ME_ok"]  is not None else "—")
            lines.append(
                f"| {r['ab']} "
                f"| {r['VME']*100:.2f} "
                f"| {r['ME']*100:.2f} "
                f"| {r['CA']*100:.1f} "
                f"| {r['sensitivity']:.3f} "
                f"| {r['specificity']:.3f} "
                f"| {r['PPV']:.3f} "
                f"| {r['NPV']:.3f} "
                f"| {vme_ok} "
                f"| {me_ok} |"
            )
        n_pass = int(df05["CLSI_VME_ok"].sum() & df05["CLSI_ME_ok"].sum()
                     if "CLSI_VME_ok" in df05 else 0)
        lines += [
            "",
            f"**CLSI criteria met (VME < 3% AND ME < 3%):**  "
            f"{df05['CLSI_VME_ok'].values.tolist()}",
        ]
    else:
        lines += ["*EXP-3 not computed — predictions_saved.npz missing.*"]
    lines.append("")

    # ── EXP-4 ────────────────────────────────────────────────────────────────
    lines += ["## EXP-4: Statistical Significance vs Base Learners", ""]
    if exp4_rows:
        df4 = pd.DataFrame(exp4_rows)
        hdr = "| AB | Base | ΔPR-AUC | 95% CI | Bootstrap | McNemar |"
        sep = "|----|------|---------|--------|-----------|---------|"
        lines += [hdr, sep]
        for _, r in df4.iterrows():
            ci  = f"[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]" \
                  if not (np.isnan(r["ci_lo"]) or np.isnan(r["ci_hi"])) else "—"
            lines.append(
                f"| {r['ab']} | {r['base']} "
                f"| {r['delta_prauc']:+.4f} "
                f"| {ci} "
                f"| {r['sig_bootstrap']} "
                f"| {r['sig_mcnemar']} |"
            )
    else:
        lines += ["*EXP-4 not computed — base predictions missing.*"]
    lines.append("")

    # ── EXP-9 ────────────────────────────────────────────────────────────────
    lines += ["## EXP-9: Leave-One-Out Base-Learner Ablation", ""]
    if exp9_rows:
        df9 = pd.DataFrame(exp9_rows)
        avg_delta = df9.groupby("base_removed")["delta"].mean().sort_values(ascending=False)
        lines += [
            "Average PR-AUC drop when each base is removed (avg across ABs):",
            "",
            "| Base Removed | Avg ΔPR-AUC | Interpretation |",
            "|---|---|---|",
        ]
        for base, delta in avg_delta.items():
            interp = "contributing" if delta > 0.002 else \
                     ("hurting" if delta < -0.002 else "negligible")
            lines.append(
                f"| {base} | {delta:+.5f} | {interp} |"
            )
        gan_idx = avg_delta.index.tolist().index("GAN") \
                  if "GAN" in avg_delta.index else None
        if gan_idx is not None:
            gan_delta = avg_delta.iloc[gan_idx]
            lines.append(
                f"\n**GAN honest marginal contribution: {gan_delta:+.5f} avg PR-AUC.** "
                + ("Positive — GAN adds value. ✅" if gan_delta > 0.002
                   else ("Negative — GAN hurts meta-learner. ❌"
                         if gan_delta < -0.002
                         else "Near-zero — GAN adds negligible signal. ⚠️"))
            )
    else:
        lines += ["*EXP-9 not computed — oof_base_probs missing.*"]
    lines.append("")

    # ── EXP-10 ───────────────────────────────────────────────────────────────
    lines += ["## EXP-10: Calibration", ""]
    if exp10_metrics:
        df10 = pd.DataFrame(exp10_metrics)
        lines += [
            "| AB | Brier (raw) | Brier (cal) | ECE (raw) | ECE (cal) |",
            "|----|------------|------------|----------|----------|",
        ]
        for _, r in df10.iterrows():
            brc = f"{r['brier_cal']:.5f}" if not np.isnan(r.get("brier_cal", np.nan)) else "—"
            ecc = f"{r['ece_cal']:.5f}"   if not np.isnan(r.get("ece_cal",   np.nan)) else "—"
            lines.append(
                f"| {r['ab']} "
                f"| {r['brier_raw']:.5f} "
                f"| {brc} "
                f"| {r['ece_raw']:.5f} "
                f"| {ecc} |"
            )
    else:
        lines += ["*EXP-10 not computed.*"]
    lines.append("")

    # ── EXP-13 ───────────────────────────────────────────────────────────────
    lines += ["## EXP-13: Per-Phylogroup Breakdown", ""]
    if exp13_rows:
        df13 = pd.DataFrame(exp13_rows)
        under = df13[df13["note"] == "underperforms"][["phylogroup", "ab", "prauc", "delta_vs_overall"]]
        if len(under) > 0:
            lines += [
                f"Phylogroups with notable underperformance (ΔPR-AUC < −0.05):",
                "",
                under.to_markdown(index=False),
            ]
        else:
            lines += ["No phylogroup shows severe underperformance (ΔPR-AUC < −0.05). ✅"]
    else:
        lines += ["*EXP-13 not computed — phylogroup data unavailable.*"]

    lines += [
        "",
        "---",
        "*See clinical_metrics.csv, significance_tests.csv, leave_one_out.csv, "
        "calibration.csv, reliability_bins.csv, per_phylogroup.csv for full tables.*",
    ]

    (output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log.info("Saved SUMMARY.md")


# ─── Entry Point ─────────────────────────────────────────────────────────────

def run(cfg: DictConfig) -> None:
    from omegaconf import OmegaConf
    if OmegaConf.select(cfg, "output_dir") is None:
        try: cfg = cfg.experiment
        except: pass
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False))
    """Hydra entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Post-hoc analysis  →  {output_dir.resolve()}")

    store = PredictionStore(cfg.v9_results_dir, cfg.data_pkl)

    spec_targets = list(cfg.get("specificity_targets", [0.90, 0.95]))
    n_boot       = int(cfg.get("n_bootstrap",    1000))
    boot_seed    = int(cfg.get("bootstrap_seed",   42))
    ece_bins     = int(cfg.get("ece_n_bins",        10))
    phylo_min    = int(cfg.get("phylo_min_samples", 20))

    # ── EXP-3 ────────────────────────────────────────────────────────────────
    exp3_rows = []
    if cfg.get("run_exp3_clinical", True):
        log.info("─── EXP-3: Clinical metrics ─────────────────────────────────")
        exp3_rows = run_clinical_metrics(store, spec_targets)

    # ── EXP-4 ────────────────────────────────────────────────────────────────
    exp4_rows = []
    if cfg.get("run_exp4_significance", True):
        log.info("─── EXP-4: Significance tests ───────────────────────────────")
        exp4_rows = run_significance_tests(store, n_boot, boot_seed)

    # ── EXP-9 ────────────────────────────────────────────────────────────────
    exp9_rows = []
    if cfg.get("run_exp9_loo", True):
        log.info("─── EXP-9: Leave-one-out ablation ───────────────────────────")
        exp9_rows = run_loo_ablation(store, seed=boot_seed)

    # ── EXP-10 ───────────────────────────────────────────────────────────────
    exp10_metrics, exp10_bins = [], []
    if cfg.get("run_exp10_calibration", True):
        log.info("─── EXP-10: Calibration ─────────────────────────────────────")
        exp10_metrics, exp10_bins = run_calibration(store, ece_bins)

    # ── EXP-13 ───────────────────────────────────────────────────────────────
    exp13_rows = []
    if cfg.get("run_exp13_phylogroup", True):
        log.info("─── EXP-13: Per-phylogroup breakdown ────────────────────────")
        exp13_rows = run_phylogroup_analysis(store, phylo_min)

    save_outputs(
        exp3_rows, exp4_rows, exp9_rows,
        exp10_metrics, exp10_bins, exp13_rows,
        output_dir,
    )
    write_summary(
        exp3_rows, exp4_rows, exp9_rows,
        exp10_metrics, exp13_rows,
        output_dir,
    )
    log.info("Post-hoc analysis complete.")
