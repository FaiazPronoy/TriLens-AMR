"""PLINK -> gene presence/absence matrix (B4, B5, B6, B9).

Ported from `arcadia-v2-phase-c.ipynb` cells 24-29, which were an iterative
patch sequence over one artifact:

    cell 25: first attempt, wrong orientation (`value > 0`), wrong gene-id
             source (PLINK's auto-generated `variant` column)
    cell 26: "# Re-run JUST the binarization + save steps with S2" -- fixes
             the orientation only
    cell 27: pre-flight check -- discovers gene-id overlap is 0/5000, and
             crashes on an array-in-boolean-context bug (B9) before it can
             even report that clearly
    cell 28: fixes the gene-id source (BIM `chrom` column, i.e. the real
             Prokka locus tags)
    cell 29: pre-flight check again, now clean -- 5000/5000 overlap, no crash

This module puts the FINAL, correct state (S2 encoding + chrom-column gene
IDs) directly in the one code path. There is no "S1 attempt" left to
accidentally run, and no separate patch cell to skip.

Re-verified directly against `arcadia_v2_gene_presence_absence.pkl` during
the audit (not just the notebook's printed log):

    X_pa.shape                   == (6962, 32441), dtype uint8
    encoding_rule                == "present = (PLINK_value == 0); A1=REF, --keep-allele-order"
    per-strain gene count         mean 3708.6, median 3331.0, min 0, max 8333
    matched strains               6949 / 6962
    overlap(selected_gene_ids [Stage 1.5], gene_ids) == 5000 / 5000 (100.0%)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import load_params

_params = load_params()["gene_pa"]
PLAUSIBILITY_RANGE: tuple[int, int] = tuple(_params["plausibility_range"])
MIN_GENE_ID_OVERLAP: int = _params["min_gene_id_overlap_with_filtered"]
UNMATCHED_STRAIN_POLICY: str = _params["unmatched_strain_policy"]


class GenePAOrientationError(AssertionError):
    """Raised when the resulting gene P/A matrix is biologically implausible
    (B4's hard assertion) -- almost always means the PLINK-value-to-presence
    rule got flipped back to the wrong orientation."""


class GenePAOverlapError(AssertionError):
    """Raised when the gene-ID catalogue does not sufficiently overlap the
    Stage 1.5 selected gene set (B5's hard assertion) -- the exact failure
    mode of the `variant0/variant1/...` auto-generated-ID bug."""


def load_plink_triple(bed_path: Path, bim_path: Path, fam_path: Path):
    """Load a PLINK .bed/.bim/.fam triple. Requires `pandas-plink` (local
    import so the rest of the package doesn't need it installed)."""
    from pandas_plink import read_plink1_bin

    G = read_plink1_bin(str(bed_path), str(bim_path), str(fam_path), verbose=False)
    return G


def extract_gene_ids(G) -> list[str]:
    """B5: gene IDs MUST come from the BIM `chrom` column (real Prokka locus
    tags), never from PLINK's auto-generated `variant` column
    (`variant0, variant1, ...`) -- that column carries no biological
    information and zeroes the overlap with every other gene catalogue in
    this pipeline.
    """
    gene_ids = [str(g) for g in G.chrom.values]
    n_unique = len(set(gene_ids))
    if n_unique != len(gene_ids):
        raise GenePAOverlapError(
            f"BIM chrom column is not unique ({n_unique} unique / {len(gene_ids)} "
            f"total) -- cannot use it as the gene-id source."
        )
    return gene_ids


def binarize_presence_absence(X_full: np.ndarray) -> np.ndarray:
    """B4: the one correct encoding rule, hardcoded.

    PLINK values are {0, 1, 2} (+ NaN for missing). This dataset's A1 allele
    is the reference allele with `--keep-allele-order`, so a PLINK value of
    0 (homozygous A1/A1 = homozygous reference) means the pan-genome gene
    call is PRESENT, and 2 (homozygous alt) means ABSENT. This was
    determined empirically in the source notebook by checking which
    orientation gives a biologically plausible per-strain gene count (S2:
    mean ~3,700; the naive `value > 0` orientation, S1, gives ~28,700,
    which is not a real bacterial genome's gene complement by nearly 6x).
    """
    X_filled = np.nan_to_num(X_full, nan=0).astype(np.uint8)
    return (X_filled == 0).astype(np.uint8)


def build_gene_presence_absence(
    G,
    existing_genome_names: list[str],
    sra_to_gname: dict[str, str],
    *,
    unmatched_policy: str = UNMATCHED_STRAIN_POLICY,
    selected_gene_ids: Optional[list[str]] = None,
) -> dict:
    """Build the final X_pa aligned to `existing_genome_names`' row order.

    Runs the B4 plausibility assertion unconditionally, and the B5 overlap
    assertion if `selected_gene_ids` (Stage 1.5's filtered set) is supplied.
    """
    if unmatched_policy not in ("zero", "drop", "nan_mask"):
        raise ValueError(f"Unknown unmatched_policy: {unmatched_policy!r}")

    gene_ids = extract_gene_ids(G)
    fam_sra_ids = [str(s) for s in G.sample.values]
    X_full = G.values

    X_bin = binarize_presence_absence(X_full)

    gname_to_sra = {v: k for k, v in sra_to_gname.items()}
    fam_sra_to_idx = {sra: i for i, sra in enumerate(fam_sra_ids)}

    n_existing = len(existing_genome_names)
    n_genes = len(gene_ids)
    X_pa_aligned = np.zeros((n_existing, n_genes), dtype=np.uint8)
    matched_mask = np.zeros(n_existing, dtype=bool)

    for i, gname in enumerate(existing_genome_names):
        sra = gname_to_sra.get(str(gname))
        if sra is None:
            continue
        fam_idx = fam_sra_to_idx.get(sra)
        if fam_idx is None:
            continue
        X_pa_aligned[i] = X_bin[fam_idx]
        matched_mask[i] = True

    # B6: explicit, configurable behaviour for unmatched strains.
    if unmatched_policy == "nan_mask":
        # Cannot store NaN in a uint8 array -- caller must consult
        # matched_mask together with X_pa; this policy only documents
        # intent (the array itself stays zero) so downstream code that
        # wants to exclude unmatched rows can do so via matched_mask.
        pass
    elif unmatched_policy == "drop":
        keep = matched_mask
        X_pa_aligned = X_pa_aligned[keep]
        # NOTE: caller is responsible for also filtering Y/label_mask/other
        # views to the same `keep` mask if this policy is used -- this
        # function only handles the P/A matrix itself.

    mean_genes = float(X_pa_aligned[matched_mask].sum(axis=1).mean()) if matched_mask.any() else float("nan")
    lo, hi = PLAUSIBILITY_RANGE
    if not (lo < mean_genes < hi):
        raise GenePAOrientationError(
            f"Gene P/A orientation implausible: {mean_genes:.0f} genes/strain "
            f"(expected {lo}-{hi} for E. coli). Check binarize_presence_absence()."
        )

    result = {
        "X_pa": X_pa_aligned,
        "gene_ids": gene_ids,
        "gene_ids_source": "BIM chromosome column (chrom) — Prokka locus tags",
        "sample_genome_names": list(existing_genome_names),
        "matched_mask": matched_mask,
        "unmatched_policy": unmatched_policy,
        "n_samples": n_existing,
        "n_genes": n_genes,
        "encoding_rule": "present = (PLINK_value == 0); A1=REF, --keep-allele-order",
        "per_strain_gene_count_stats": {
            "mean": mean_genes,
            "median": float(np.median(X_pa_aligned[matched_mask].sum(axis=1))) if matched_mask.any() else float("nan"),
        },
    }

    if selected_gene_ids is not None:
        sel = set(str(g) for g in selected_gene_ids)
        pa_g = set(str(g) for g in gene_ids)
        overlap = sel & pa_g
        result["overlap_with_selected"] = sorted(overlap)
        if len(overlap) < MIN_GENE_ID_OVERLAP:
            raise GenePAOverlapError(
                f"Gene ID overlap only {len(overlap)}/{len(sel)} "
                f"(< {MIN_GENE_ID_OVERLAP}). This is the exact failure mode "
                f"of the auto-generated-ID bug (B5) -- verify gene_ids_source "
                f"is the chrom column, not PLINK's `variant` column."
            )

    return result
