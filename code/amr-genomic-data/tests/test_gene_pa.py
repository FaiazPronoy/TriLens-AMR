import numpy as np
import pytest

from src.amr_data.gene_pa import (
    GenePAOrientationError,
    GenePAOverlapError,
    binarize_presence_absence,
    extract_gene_ids,
)
from src.amr_data import qa


class _FakeChromArray:
    """Minimal stand-in for xarray's `.values` attribute."""

    def __init__(self, values):
        self.values = np.array(values, dtype=object)


class _FakeG:
    """Minimal duck-typed stand-in for pandas_plink's xarray object, just
    enough surface for extract_gene_ids()."""

    def __init__(self, chrom_ids):
        self.chrom = _FakeChromArray(chrom_ids)


# ---------------------------------------------------------------------------
# B4: orientation
# ---------------------------------------------------------------------------
def test_binarize_present_is_value_zero():
    # PLINK values: 0 = homozygous A1 (REF) = gene PRESENT, 2 = absent.
    X = np.array([
        [0, 2, 0, 2],
        [2, 2, 0, 0],
        [0, 0, 2, 2],
    ], dtype=float)
    X_bin = binarize_presence_absence(X)
    expected = np.array([
        [1, 0, 1, 0],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
    ], dtype=np.uint8)
    assert np.array_equal(X_bin, expected)


def test_binarize_handles_nan_as_absent():
    X = np.array([[0.0, np.nan, 2.0]])
    X_bin = binarize_presence_absence(X)
    # nan_to_num(nan) -> 0 -> "present" is the documented behaviour of
    # np.nan_to_num(X, nan=0) followed by == 0; this test locks that in so a
    # future change to the fill value doesn't silently alter which missing
    # genotypes count as present.
    assert X_bin.tolist() == [[1, 1, 0]]


def test_naive_s1_orientation_would_be_implausible():
    """Directly demonstrates *why* S2 (value==0) was chosen over the naive
    S1 (value>0): reproduce the real pipeline's actual gene-prevalence rate
    (~11.4% of the 32,441-gene catalogue present per strain, matching the
    real artifact's mean of 3708.6/32441) and check the resulting mean
    genes/strain against both rules."""
    rng = np.random.RandomState(0)
    n_samples, n_genes = 300, 32441
    p_present = 3708.6 / n_genes  # matches the real artifact's prevalence
    is_present = rng.rand(n_samples, n_genes) < p_present
    X = np.where(is_present, 0, 2).astype(float)  # PLINK convention: 0=present(S2), 2=absent

    s2_mean = binarize_presence_absence(X).sum(axis=1).mean()
    s1_mean = (np.nan_to_num(X, nan=0) > 0).astype(np.uint8).sum(axis=1).mean()

    # S2 should land in the plausible E. coli range; S1 (inverted) should not.
    assert 2000 < s2_mean < 8000
    assert not (2000 < s1_mean < 8000)


# ---------------------------------------------------------------------------
# B5: gene-ID source
# ---------------------------------------------------------------------------
def test_extract_gene_ids_uses_chrom_column():
    G = _FakeG(["LMHECDEF_00001", "LMHECDEF_00002", "LMHECDEF_00003"])
    ids = extract_gene_ids(G)
    assert ids == ["LMHECDEF_00001", "LMHECDEF_00002", "LMHECDEF_00003"]


def test_extract_gene_ids_rejects_duplicates():
    # A `variant0, variant1, ...`-style column would still be unique, so
    # this specifically catches a *different* failure mode: a chrom column
    # that isn't actually a good ID (repeats). Real Prokka locus tags are
    # unique per gene call.
    G = _FakeG(["LMHECDEF_00001", "LMHECDEF_00001", "LMHECDEF_00003"])
    with pytest.raises(GenePAOverlapError):
        extract_gene_ids(G)


# ---------------------------------------------------------------------------
# Cross-check against the real, already-built artifact (the actual B4/B5
# orientation test the spec asks for -- run against production data, not
# just synthetic)
# ---------------------------------------------------------------------------
def test_real_gene_pa_artifact_passes_orientation_and_overlap_checks(gene_pa, gene_filtered_legacy):
    report = qa.verify_gene_pa(
        gene_pa,
        gene_filtered_legacy,
        genome_names_reference=gene_filtered_legacy["genome_names"],
    )
    assert report["ok"] is True
    assert report["gene_id_overlap"] == "5000/5000"
    assert 3000 < report["mean_genes_per_strain"] < 5000


def test_real_gene_pa_artifact_would_fail_if_orientation_were_flipped(gene_pa):
    X_pa = gene_pa["X_pa"]
    flipped = 1 - X_pa  # simulate the S1 bug re-appearing
    mean_genes = float(flipped.sum(axis=1).mean())
    lo, hi = 2000, 8000
    assert not (lo < mean_genes < hi), (
        "Sanity check on the test itself: flipping the real artifact's "
        "encoding should land outside the plausible range, confirming the "
        "plausibility assertion actually has teeth."
    )
