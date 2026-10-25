"""Verification checks.

`verify_dataset()` generalizes the best QA code found across the five
notebooks -- `autoencoder-pca-random-split.ipynb` cell 1, the "PREPROCESSED
DATASET VERIFICATION SCRIPT". That script hardcoded an expected shape of
`(6962, 80984)` for the (now superseded) dataset it was written for; this
version takes `expected_shape` as an optional parameter instead, so it works
for any of the pipeline's datasets without silently green-lighting the wrong
one (pass the shape you expect; if you don't pass one, that specific check is
skipped rather than faked).

`verify_gene_pa()` folds in the B4/B5/B6 hard assertions and uses
`np.array_equal` for the strain-order identity check (B9 -- the earlier
pre-flight cell crashed on `ValueError: truth value of an array... is
ambiguous" from an implicit boolean array comparison; the fix, ported from
the notebook's own working second attempt, is to use `np.array_equal`
explicitly, never a bare `==` on two arrays in an `if`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .config import load_params

_pa_params = load_params()["gene_pa"]


class QAFailure(AssertionError):
    """Raised when a QA check fails outright (not just a warning)."""


def verify_dataset(
    ds: dict[str, Any],
    *,
    expected_shape: Optional[tuple[int, int]] = None,
    max_strat_diff: float = 0.05,
) -> dict[str, Any]:
    """General dataset verification: keys, shapes, dtypes, split integrity,
    stratification balance, NaN/Inf sanity. Returns a report dict; raises
    QAFailure on any hard failure (overlap between splits, shape mismatch
    against `expected_shape` if given, non-binary Y, NaN/Inf in X).
    """
    report: dict[str, Any] = {"ok": True, "warnings": []}

    required_keys = ["X", "Y", "label_mask", "label_names", "random_split"]
    missing = [k for k in required_keys if k not in ds]
    if missing:
        raise QAFailure(f"Missing required keys: {missing}. Available: {list(ds.keys())}")

    X, Y, mask, label_names = ds["X"], ds["Y"], ds["label_mask"], list(ds["label_names"])
    report["X_shape"] = X.shape
    report["Y_shape"] = Y.shape
    report["label_names"] = label_names

    if X.shape[0] != Y.shape[0] or Y.shape[0] != mask.shape[0]:
        raise QAFailure(f"Sample count mismatch: X={X.shape[0]}, Y={Y.shape[0]}, mask={mask.shape[0]}")
    if Y.shape[1] != mask.shape[1] or Y.shape[1] != len(label_names):
        raise QAFailure(f"Label count mismatch: Y={Y.shape[1]}, mask={mask.shape[1]}, names={len(label_names)}")
    if expected_shape is not None and tuple(X.shape) != tuple(expected_shape):
        raise QAFailure(f"X shape {X.shape} != expected {expected_shape}")

    if np.isnan(X).any():
        raise QAFailure("X contains NaN")
    if np.isinf(X).any():
        raise QAFailure("X contains Inf")

    y_unique = np.unique(Y[~np.isnan(Y)]) if np.isnan(Y).any() else np.unique(Y)
    if not set(y_unique.tolist()).issubset({0.0, 1.0}):
        raise QAFailure(f"Y has non-binary values: {y_unique}")
    if (np.isnan(Y) != ~mask).any():
        raise QAFailure("label_mask does not align with non-NaN Y values")

    report["random_split"] = _verify_one_split(ds["random_split"], X.shape[0])
    if "phylogeny_split" in ds or "phylogroup_stratified_split" in ds:
        key = "phylogroup_stratified_split" if "phylogroup_stratified_split" in ds else "phylogeny_split"
        report[key] = _verify_one_split(ds[key], X.shape[0])

    # Stratification balance (informational -- not a hard failure)
    strat_rows = []
    for j, ab in enumerate(label_names):
        m = mask[:, j].astype(bool)
        tr = ds["random_split"]["train_idx"]
        te = ds["random_split"]["test_idx"]
        tr_p = float(Y[tr[m[tr]], j].mean()) if m[tr].any() else 0.0
        te_p = float(Y[te[m[te]], j].mean()) if m[te].any() else 0.0
        diff = abs(tr_p - te_p)
        strat_rows.append({"antibiotic": ab, "train_prev": tr_p, "test_prev": te_p, "diff": diff})
        if diff >= max_strat_diff:
            report["warnings"].append(f"{ab}: train/test prevalence diff {diff:.1%} >= {max_strat_diff:.0%}")
    report["stratification"] = pd.DataFrame(strat_rows)

    return report


def _verify_one_split(split: dict, n: int) -> dict:
    tr, va, te = np.asarray(split["train_idx"]), np.asarray(split["val_idx"]), np.asarray(split["test_idx"])
    tr_s, va_s, te_s = set(tr.tolist()), set(va.tolist()), set(te.tolist())
    if tr_s & va_s:
        raise QAFailure("Train-Val overlap in split")
    if tr_s & te_s:
        raise QAFailure("Train-Test overlap in split")
    if va_s & te_s:
        raise QAFailure("Val-Test overlap in split")
    coverage = len(tr_s | va_s | te_s) / n
    return {"n_train": len(tr), "n_val": len(va), "n_test": len(te), "coverage": coverage}


def verify_gene_pa(
    pa: dict[str, Any],
    gf: Optional[dict[str, Any]] = None,
    genome_names_reference: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """B4 + B5 + B6 + B9 checks on a gene presence/absence artifact.

    Raises QAFailure on:
      - implausible per-strain gene count (B4)
      - gene-id overlap with the filtered gene set below the configured
        minimum (B5), if `gf` is given
      - strain-order mismatch against a reference genome-name array (B9),
        if `genome_names_reference` is given
    """
    report: dict[str, Any] = {"ok": True}
    X_pa = pa["X_pa"]

    mean_genes = float(X_pa.sum(axis=1).mean())
    lo, hi = _pa_params["plausibility_range"]
    if not (lo < mean_genes < hi):
        raise QAFailure(
            f"Gene P/A orientation implausible: {mean_genes:.0f} genes/strain. "
            f"E. coli carries ~4000-5000. Check the presence encoding."
        )
    report["mean_genes_per_strain"] = mean_genes
    report["median_genes_per_strain"] = float(np.median(X_pa.sum(axis=1)))

    encoding_rule = pa.get("encoding_rule", "")
    if "== 0" not in encoding_rule and "value == 0" not in encoding_rule.replace("PLINK_", "PLINK "):
        report_note = f"encoding_rule field does not look like the S2 rule: {encoding_rule!r}"
        raise QAFailure(report_note)
    report["encoding_rule"] = encoding_rule

    matched = int(np.asarray(pa["matched_mask"]).sum())
    total = len(pa["matched_mask"])
    report["matched_strains"] = f"{matched}/{total}"

    if gf is not None:
        sel = set(str(g) for g in gf["selected_gene_ids"])
        pa_g = set(str(g) for g in pa["gene_ids"])
        overlap = sel & pa_g
        report["gene_id_overlap"] = f"{len(overlap)}/{len(sel)}"
        min_overlap = _pa_params["min_gene_id_overlap_with_filtered"]
        if len(overlap) < min_overlap:
            raise QAFailure(
                f"Gene ID overlap only {len(overlap)} (< {min_overlap}). "
                f"This is the exact failure mode of the `variant0/variant1/...` "
                f"auto-generated-ID bug (B5) -- check gene_ids_source == "
                f"'BIM chromosome column (chrom) — Prokka locus tags'."
            )

    if genome_names_reference is not None:
        pa_strains = np.asarray(pa.get("sample_genome_names", []), dtype=str)
        ref_strains = np.asarray(genome_names_reference, dtype=str)
        # B9 fix: np.array_equal explicitly, never a bare `==` in a boolean
        # context on two arrays.
        if not np.array_equal(pa_strains, ref_strains):
            mismatches = int((pa_strains != ref_strains).sum()) if len(pa_strains) == len(ref_strains) else -1
            raise QAFailure(f"Strain order mismatch (mismatched positions: {mismatches})")
        report["strain_order"] = "identical"

    return report


def export_unmatched_strains(
    pa: dict[str, Any],
    out_path: Path,
) -> pd.DataFrame:
    """B6: write the exact list of genome names with no P/A data to CSV."""
    matched_mask = np.asarray(pa["matched_mask"])
    genome_names = np.asarray(pa.get("sample_genome_names", []))
    unmatched_idx = np.where(~matched_mask)[0]
    df = pd.DataFrame({
        "row_index": unmatched_idx,
        "genome_name": genome_names[unmatched_idx] if len(genome_names) else ["<unknown>"] * len(unmatched_idx),
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def verify_giessen(ds: dict[str, Any]) -> dict[str, Any]:
    """Verification for the Giessen loader's output (Part C)."""
    report: dict[str, Any] = {}
    X, Y, mask, label_names = ds["X"], ds["Y"], ds["label_mask"], list(ds["label_names"])
    report["X_shape"] = X.shape
    if np.isnan(X).any():
        raise QAFailure("Giessen X contains NaN")
    vals = np.unique(X)
    report["X_unique_values"] = vals.tolist()
    if mask.mean() != 1.0:
        report.setdefault("warnings", []).append(
            f"Giessen label completeness is {mask.mean():.1%}, expected 100%"
        )
    else:
        report["label_completeness"] = 1.0
    for j, ab in enumerate(label_names):
        y = Y[:, j]
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        report[f"{ab}_counts"] = {"R": pos, "S": neg}
    return report
