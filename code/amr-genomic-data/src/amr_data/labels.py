"""ONE phenotype encoder (B2).

Phase B of the original notebooks contained two different encoders
(`pheno_to_binary` in cell 3, `to_bin` in cell 5) that accepted different
token sets and both silently returned NaN for anything unrecognized. This
module is the only encoder in the codebase; it uses the permissive superset
of the two, and it raises on an unrecognized non-missing token instead of
silently dropping the observation.

See docs/AUDIT_FINDINGS.md (B2) for the full diff of the two original
encoders and what could and could not be verified about which one actually
fed which downstream artifact.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .config import load_params

_params = load_params()["labels"]

RESISTANT_TOKENS: frozenset[str] = frozenset(_params["resistant_tokens"])
SUSCEPTIBLE_TOKENS: frozenset[str] = frozenset(_params["susceptible_tokens"])
ANTIBIOTICS: list[str] = list(_params["antibiotics"])
ANTIBIOTIC_COLUMN_MAP: dict[str, str] = dict(_params["antibiotic_column_map"])


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def encode_phenotype(value: Any) -> float:
    """Encode one raw phenotype token to {1.0, 0.0, NaN}.

    Intermediate ("I"/"int"/"intermediate") is encoded as resistant (1.0),
    following the source study's convention (this matches both of the
    original notebook's encoders on this specific point -- they only
    disagreed on which abbreviations they additionally accepted, not on
    how intermediate is treated).

    Missing (`NaN`, `None`, pandas-NA) is a legitimate "not tested" result
    and returns `NaN`. Anything else that isn't in the recognized token
    sets is NOT silently dropped -- it raises, because a phenotype value
    this pipeline doesn't understand is a data problem to surface, not to
    quietly discard (see docs/AUDIT_FINDINGS.md B2, point 3 of the spec).
    """
    if _is_missing(value):
        return float("nan")
    s = str(value).strip().lower()
    if s in RESISTANT_TOKENS:
        return 1.0
    if s in SUSCEPTIBLE_TOKENS:
        return 0.0
    raise ValueError(
        f"Unrecognized phenotype token: {value!r} (normalized: {s!r}). "
        f"Recognized resistant tokens: {sorted(RESISTANT_TOKENS)}. "
        f"Recognized susceptible tokens: {sorted(SUSCEPTIBLE_TOKENS)}. "
        f"If this is a genuine new token spelling, add it to "
        f"config/params.yaml under labels.resistant_tokens / "
        f"susceptible_tokens -- do not special-case it in code."
    )


def build_label_matrix(
    pheno_df: pd.DataFrame,
    antibiotics: list[str] | None = None,
    column_map: dict[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (Y, label_mask) from a phenotype dataframe (one row per sample,
    in the same row order as the SNP/feature matrix).

    Returns:
        Y: float32 array (n_samples, n_antibiotics), 1.0/0.0/NaN.
        label_mask: bool array, True where Y is non-NaN.
    """
    antibiotics = antibiotics or ANTIBIOTICS
    column_map = column_map or ANTIBIOTIC_COLUMN_MAP

    n = len(pheno_df)
    Y = np.full((n, len(antibiotics)), np.nan, dtype=np.float32)
    for i, ab in enumerate(antibiotics):
        col = column_map[ab]
        Y[:, i] = pheno_df[col].apply(encode_phenotype).values
    label_mask = ~np.isnan(Y)
    return Y, label_mask


def audit_raw_phenotype_values(pheno_df: pd.DataFrame,
                                antibiotics: list[str] | None = None,
                                column_map: dict[str, str] | None = None) -> pd.DataFrame:
    """B2 task 2: report every distinct raw token seen in the phenotype
    columns, with counts, so it is verifiable (not assumed) whether the two
    original encoders' disagreement ever actually fired.

    Requires the raw phenotype CSV to be available -- this was NOT present
    in this repository at refactor time (only Kaggle-side); see
    docs/OPEN_QUESTIONS.md item for phenotype token auditing. This function
    is provided so that whoever next has access to
    `phenotype_matrix_08302024.csv` can run it immediately.
    """
    antibiotics = antibiotics or ANTIBIOTICS
    column_map = column_map or ANTIBIOTIC_COLUMN_MAP
    rows = []
    for ab in antibiotics:
        col = column_map[ab]
        counts = pheno_df[col].astype(str).str.strip().str.lower().value_counts(dropna=False)
        for token, count in counts.items():
            rows.append({"antibiotic": ab, "column": col, "raw_token": token, "count": int(count)})
    return pd.DataFrame(rows)
