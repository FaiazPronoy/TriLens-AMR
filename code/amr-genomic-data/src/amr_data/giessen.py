"""Giessen external validation set (Part C).

Ported from `giessen-cross-validation.ipynb`. Real, local data (verified
directly during the audit, not just from the notebook's printed log):

    X.shape == (809, 60936); values in {0,1,2,3,4}; 100% label-complete
    CIP 366R/443S, CTX 358R/451S, CTZ(->CAZ) 276R/533S, GEN 188R/621S

Source: Ren et al. 2022, 10.1093/bioinformatics/btab681. See
docs/DATA_PROVENANCE.md.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from .config import Paths, load_paths, load_params

_params = load_params()["giessen"]
ANTIBIOTIC_NAME_MAP: dict[str, str] = dict(_params["antibiotic_name_map"])
SHARED_ANTIBIOTICS_WITH_ARCADIA: list[str] = list(_params["shared_antibiotics_with_arcadia"])


def load_giessen_raw(paths: Optional[Paths] = None) -> dict:
    """Load the two Giessen CSVs and normalize antibiotic names to the
    canonical set (CIP/CTX/CAZ/GEN) -- Giessen's own column is `CTZ` for
    ceftazidime; Arcadia/this thesis calls it `CAZ`. Normalizing here means
    every downstream consumer only ever sees canonical names.
    """
    paths = paths or load_paths()
    x_path = paths.giessen("x_csv")
    y_path = paths.giessen("y_csv")

    X_df = pd.read_csv(x_path)
    Y_df = pd.read_csv(y_path)

    id_col = X_df.columns[0]
    ids = X_df[id_col].values
    X = X_df.drop(columns=[id_col]).values.astype(np.float32)

    y_id_col = Y_df.columns[0]
    if not np.array_equal(ids.astype(str), Y_df[y_id_col].astype(str).values):
        raise ValueError(
            f"X and Y row order mismatch on id column '{id_col}'/'{y_id_col}' -- "
            f"do not proceed without fixing alignment first."
        )

    raw_label_cols = [c for c in Y_df.columns if c != y_id_col]
    canonical_names = [ANTIBIOTIC_NAME_MAP.get(c, c) for c in raw_label_cols]

    Y = Y_df[raw_label_cols].values.astype(np.float32)
    label_mask = ~np.isnan(Y)

    return {
        "X": X,
        "Y": Y,
        "label_mask": label_mask,
        "label_names": canonical_names,
        "raw_label_names": raw_label_cols,
        "ids": ids,
        "antibiotic_to_index": {ab: i for i, ab in enumerate(canonical_names)},
        "n_samples": X.shape[0],
        "n_features": X.shape[1],
        "source": "Ren et al. 2022, 10.1093/bioinformatics/btab681",
    }


def build_giessen_split(
    ds: dict,
    *,
    test_size: float = 0.30,
    stratify_col: Optional[str] = None,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Reproduce the notebook's split: 70/15/15 via two chained
    StratifiedShuffleSplit calls (test_size=0.30 then 0.50 of the
    remainder), stratified on whichever antibiotic is closest to a 50/50
    base rate (CIP, in practice) unless `stratify_col` is given explicitly.
    """
    Y, label_names = ds["Y"], ds["label_names"]
    n = Y.shape[0]

    if stratify_col is None:
        balance = {ab: abs(np.nanmean(Y[:, i]) - 0.5) for i, ab in enumerate(label_names)}
        stratify_col = min(balance, key=balance.get)
    strat_idx = label_names.index(stratify_col)
    strat_target = Y[:, strat_idx].copy()
    nan_mask = np.isnan(strat_target)
    if nan_mask.any():
        mode_val = int(np.nanmean(strat_target) > 0.5)
        strat_target[nan_mask] = mode_val

    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, temp_idx = next(sss1.split(np.zeros(n), strat_target))

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val_rel, test_rel = next(sss2.split(np.zeros(len(temp_idx)), strat_target[temp_idx]))
    val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]

    return {
        "train_idx": train_idx, "val_idx": val_idx, "test_idx": test_idx,
        "stratified_on": stratify_col,
    }


# ---------------------------------------------------------------------------
# Part C.4 -- fix the selection bias in the strategy-vs-baseline comparison
# ---------------------------------------------------------------------------
def full_strategy_grid(results_df: pd.DataFrame) -> pd.DataFrame:
    """The full 5 (strategy) x 4 (antibiotic) grid, unfiltered. This alone
    is not misleading -- it only becomes so if a reader is shown just the
    per-antibiotic best cell (see `honest_headline_strategy` below).
    `results_df` must have columns: strategy, antibiotic, split ("val"/
    "test"), prauc (and any other metrics to carry through).
    """
    return results_df.pivot_table(index="strategy", columns=["antibiotic", "split"], values="prauc")


def honest_headline_strategy(
    results_df: pd.DataFrame,
    *,
    metric_col: str = "prauc",
) -> tuple[str, pd.DataFrame]:
    """The un-biased headline number: pick ONE strategy, fixed across all
    antibiotics, chosen by its mean VALIDATION performance -- then report
    that (and only that) strategy's TEST performance per antibiotic.

    This is the fix for the bug in `giessen-cross-validation.ipynb` cell 8:
    that cell reported "our best MCC" per antibiotic against
    AMR-EnsembleNet, but the winning strategy differed per antibiotic
    (A_Early_SS for CIP, D_Hybrid_LS for CTX, B_Joint_LL for CTZ,
    E_ProximityAdaptive for GEN) -- i.e. it chose the best of five options
    separately for each of four comparisons, then compared each choice
    against one fixed external baseline. That is a biased comparison (the
    "garden of forking paths" / multiple-comparisons problem in miniature).

    `results_df` must have columns: strategy, antibiotic, split ("val"/
    "test"), and `metric_col`.

    Returns (chosen_strategy_name, test_results_df).
    """
    required_cols = {"strategy", "antibiotic", "split", metric_col}
    missing = required_cols - set(results_df.columns)
    if missing:
        raise ValueError(f"results_df missing columns: {missing}")

    val_df = results_df[results_df["split"] == "val"]
    mean_val_by_strategy = val_df.groupby("strategy")[metric_col].mean()
    if mean_val_by_strategy.empty:
        raise ValueError("No validation-split rows found -- cannot select a strategy honestly.")
    chosen = mean_val_by_strategy.idxmax()

    test_df = results_df[(results_df["split"] == "test") & (results_df["strategy"] == chosen)].copy()
    return chosen, test_df
