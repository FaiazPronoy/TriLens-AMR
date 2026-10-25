"""Three split regimes (B1, B7).

The original notebooks produced exactly two splits, called `random_split` and
`phylogeny_split`. The audit (docs/AUDIT_FINDINGS.md, B1) confirmed by direct
inspection of `arcadia_v2_master_features.pkl` that `phylogeny_split` is a
`train_test_split(..., stratify=cluster_labels)` call -- i.e. it PRESERVES
each phylogroup's proportion on both sides. It is the opposite of a lineage
holdout, and every claim in the thesis that "whole lineages were held out"
is not supported by that split.

This module provides three functions:

    random_split(...)                  -- unchanged behaviour, B7 nan-cast
                                           fix applied (verified byte-identical
                                           to the locked reference, see
                                           tests/test_splits.py)
    phylogroup_stratified_split(...)    -- the old "phylogeny_split", renamed
                                           honestly, identical logic
    lineage_holdout_split(...)          -- NEW: a genuine group holdout via
                                           GroupShuffleSplit, with mandatory
                                           disjointness assertions

Every split function returns a dict with keys "train_idx", "val_idx",
"test_idx" (np.ndarray of int64 row indices).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .config import load_params

_params = load_params()["splits"]
DEFAULT_TEST_SIZE: float = _params["test_size"]
DEFAULT_VAL_SIZE_OF_REMAINDER: float = _params["val_size_of_remainder"]
DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# B7 fix: NaN-safe stratification array
# ---------------------------------------------------------------------------
def _safe_strat_array(y_col: np.ndarray) -> np.ndarray:
    """Build a stratification array from a possibly-NaN label column.

    The original code did:
        np.where(np.isnan(Y[:, i]), -1, Y[:, i].astype(int))
    which casts the WHOLE array (NaNs included) to int before np.where
    selects -- undefined behaviour, and a RuntimeWarning on every run,
    even though (verified directly against the locked
    arcadia_v2_master_features.pkl random_split, see
    tests/test_splits.py::test_random_split_matches_locked_reference) it
    happens to produce the identical result in this specific case, because
    the discarded branch is never selected. This version avoids the
    warning by not casting the NaN branch at all.
    """
    return np.where(
        np.isnan(y_col), -1, np.nan_to_num(y_col, nan=0.0)
    ).astype(int)


# ---------------------------------------------------------------------------
# random_split -- unchanged behaviour
# ---------------------------------------------------------------------------
def random_split(
    Y: np.ndarray,
    stratify_col_idx: int,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size_of_remainder: float = DEFAULT_VAL_SIZE_OF_REMAINDER,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    """Stratified random split on a single label column (Phase B: AMP).

    `Y` may contain NaN for missing labels on the stratification column --
    those rows are stratified as their own group (-1), same as the original.
    """
    n = Y.shape[0]
    strat = _safe_strat_array(Y[:, stratify_col_idx])

    idx_all = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=seed, stratify=strat
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size_of_remainder,
        random_state=seed,
        stratify=strat[idx_trainval],
    )
    return {"train_idx": idx_train, "val_idx": idx_val, "test_idx": idx_test}


# ---------------------------------------------------------------------------
# phylogroup_stratified_split -- honest rename of the old "phylogeny_split"
# ---------------------------------------------------------------------------
def phylogroup_stratified_split(
    cluster_labels: np.ndarray,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size_of_remainder: float = DEFAULT_VAL_SIZE_OF_REMAINDER,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    """Stratified split on phylogroup labels.

    IMPORTANT: this is NOT a lineage holdout. `stratify=` makes
    `train_test_split` PRESERVE each phylogroup's proportion on both sides
    of the split -- every phylogroup present in the data will appear in
    train, val, AND test. If you need whole lineages excluded from train
    (a genuine generalisation test), use `lineage_holdout_split` instead.

    This function exists, under this honest name, to reproduce the
    existing pipeline's numbers exactly (identical logic, identical
    `random_state`) for anything already built on top of the old
    "phylogeny_split" -- not because it is the split you should use for a
    "held-out lineages" claim.
    """
    n = len(cluster_labels)
    idx_all = np.arange(n)
    idx_trainval, idx_test = train_test_split(
        idx_all, test_size=test_size, random_state=seed, stratify=cluster_labels
    )
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size_of_remainder,
        random_state=seed,
        stratify=cluster_labels[idx_trainval],
    )
    return {"train_idx": idx_train, "val_idx": idx_val, "test_idx": idx_test}


# ---------------------------------------------------------------------------
# lineage_holdout_split -- NEW (B1, the highest-priority deliverable)
# ---------------------------------------------------------------------------
def groups_from_phylogroup(cluster_labels: np.ndarray) -> np.ndarray:
    """Convenience grouping variable: the 8 broad phylogroup labels.

    This is the ONE grouping source that can be exercised end-to-end today
    against the data already in this repository (`cluster_labels` is saved
    in arcadia_v2_master_features.pkl). It gives a genuine holdout of whole
    phylogroups -- coarser than a real lineage/clade holdout, but strictly
    stronger than `phylogroup_stratified_split`, which holds nothing out at
    all.
    """
    return np.asarray(cluster_labels)


def groups_from_mlst(
    mlst_st: pd.Series,
    *,
    unassigned_policy: str = "singleton",
) -> np.ndarray:
    """Convenience grouping variable: Achtman sequence type (ST).

    `mlst_st` should already be the parsed ST string per strain (see
    `phylogroup.parse_mlst_st`), NOT the raw `MLST` column -- parse it first.

    Strains with no ST (`None`/NaN) are, by default (`unassigned_policy=
    "singleton"`), each assigned their own unique group so they cannot
    collectively form one giant pseudo-lineage that defeats the holdout by
    definition (an unassigned-ST group could otherwise contain hundreds of
    genuinely unrelated strains). Set `unassigned_policy="drop"` to instead
    exclude them from the grouping (caller must then filter the sample
    array to match).

    NOTE: this function is READY but has not been exercised end-to-end in
    this repository, because no local artifact currently retains raw
    per-strain MLST strings (only the already-collapsed 8-phylogroup
    `cluster_labels` survives into arcadia_v2_master_features.pkl -- see
    docs/AUDIT_FINDINGS.md, Part B). Once a metadata file with raw MLST
    strings is available, call `phylogroup.parse_mlst_st` per row first.
    """
    if unassigned_policy not in ("singleton", "drop"):
        raise ValueError(f"Unknown unassigned_policy: {unassigned_policy!r}")

    st = mlst_st.reset_index(drop=True)
    is_missing = st.isna()
    n_missing = int(is_missing.sum())

    groups = np.empty(len(st), dtype=object)
    known_mask = ~is_missing.values
    groups[known_mask] = "ST_" + st[known_mask].astype(str).values

    if unassigned_policy == "singleton":
        singleton_ids = [f"__singleton_{i}" for i in np.where(~known_mask)[0]]
        groups[~known_mask] = singleton_ids
    else:  # "drop" -- caller is expected to also drop these rows from X/Y
        groups[~known_mask] = None

    if n_missing:
        print(
            f"[groups_from_mlst] {n_missing}/{len(st)} strains have no ST "
            f"({unassigned_policy=}); see docstring for why they are not "
            f"lumped into one group."
        )
    return groups


def lineage_holdout_split(
    groups: np.ndarray,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size_of_remainder: float = DEFAULT_VAL_SIZE_OF_REMAINDER,
    seed: int = DEFAULT_SEED,
) -> dict[str, np.ndarray]:
    """A genuine group holdout: no group in `groups` appears on both sides
    of any partition. Sizes are NOT the fixed 4872/1045/1045 of the other
    two splits -- they depend on how group sizes happen to fall, which is
    exactly the point (a real holdout cannot target exact sample counts
    without breaking groups apart).

    Raises AssertionError (not a warning) if any leak is detected -- this
    must never silently pass.
    """
    n = len(groups)
    idx_all = np.arange(n)
    groups = np.asarray(groups)

    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trainval_pos, test_pos = next(gss_test.split(idx_all, groups=groups))
    idx_trainval, idx_test = idx_all[trainval_pos], idx_all[test_pos]

    gss_val = GroupShuffleSplit(
        n_splits=1, test_size=val_size_of_remainder, random_state=seed
    )
    train_pos, val_pos = next(
        gss_val.split(idx_trainval, groups=groups[idx_trainval])
    )
    idx_train, idx_val = idx_trainval[train_pos], idx_trainval[val_pos]

    result = {"train_idx": idx_train, "val_idx": idx_val, "test_idx": idx_test}
    _assert_group_disjoint(groups, result)
    return result


def _assert_group_disjoint(groups: np.ndarray, split: dict[str, np.ndarray]) -> None:
    """Mandatory leak check -- raises, does not warn (per spec)."""
    g_train = set(groups[split["train_idx"]].tolist())
    g_val = set(groups[split["val_idx"]].tolist())
    g_test = set(groups[split["test_idx"]].tolist())

    assert not (g_train & g_test), (
        f"LEAK: group appears in both train and test: {sorted(g_train & g_test)[:10]}"
    )
    assert not (g_train & g_val), (
        f"LEAK: group appears in both train and val: {sorted(g_train & g_val)[:10]}"
    )
    assert not (g_val & g_test), (
        f"LEAK: group appears in both val and test: {sorted(g_val & g_test)[:10]}"
    )


# ---------------------------------------------------------------------------
# Split diagnostics -- the table that was "currently missing entirely"
# ---------------------------------------------------------------------------
def split_diagnostics(
    splits: dict[str, dict[str, np.ndarray]],
    Y: np.ndarray,
    label_mask: np.ndarray,
    antibiotics: list[str],
) -> pd.DataFrame:
    """Per-split, per-antibiotic train/test label counts and prevalence.

    This is the table that distinguishes real generalisation loss (the
    model genuinely fails to transfer) from a prevalence artefact (train
    and test just happen to have different base rates for that antibiotic
    under that split). Every split regime should be run through this
    before its PR-AUC/MCC numbers are interpreted.

    `splits`: {split_name: {"train_idx":..., "val_idx":..., "test_idx":...}}
    """
    rows = []
    for split_name, idx_dict in splits.items():
        tr_idx, te_idx = idx_dict["train_idx"], idx_dict["test_idx"]
        for j, ab in enumerate(antibiotics):
            m = label_mask[:, j]
            tr_m = m[tr_idx]
            te_m = m[te_idx]
            n_tr = int(tr_m.sum())
            n_te = int(te_m.sum())
            prev_tr = float(Y[tr_idx[tr_m], j].mean()) if n_tr else float("nan")
            prev_te = float(Y[te_idx[te_m], j].mean()) if n_te else float("nan")
            rows.append({
                "split": split_name,
                "antibiotic": ab,
                "n_train_labelled": n_tr,
                "n_test_labelled": n_te,
                "prev_train": prev_tr,
                "prev_test": prev_te,
                "prev_delta": prev_te - prev_tr if n_tr and n_te else float("nan"),
            })
    return pd.DataFrame(rows)


def save_split_diagnostics(
    splits: dict[str, dict[str, np.ndarray]],
    Y: np.ndarray,
    label_mask: np.ndarray,
    antibiotics: list[str],
    out_path: Path,
) -> pd.DataFrame:
    df = split_diagnostics(splits, Y, label_mask, antibiotics)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
