import math

import numpy as np
import pandas as pd
import pytest

from src.amr_data.labels import (
    ANTIBIOTIC_COLUMN_MAP,
    ANTIBIOTICS,
    RESISTANT_TOKENS,
    SUSCEPTIBLE_TOKENS,
    audit_raw_phenotype_values,
    build_label_matrix,
    encode_phenotype,
)


@pytest.mark.parametrize("token", ["r", "R", "res", "RES", "resistant", "Resistant",
                                    "i", "I", "int", "intermediate", "INTERMEDIATE"])
def test_resistant_tokens_encode_to_one(token):
    assert encode_phenotype(token) == 1.0


@pytest.mark.parametrize("token", ["s", "S", "sus", "SUS", "susceptible", "Susceptible"])
def test_susceptible_tokens_encode_to_zero(token):
    assert encode_phenotype(token) == 0.0


@pytest.mark.parametrize("value", [np.nan, None, float("nan"), pd.NA])
def test_missing_encodes_to_nan_not_error(value):
    result = encode_phenotype(value)
    assert math.isnan(result)


def test_unrecognized_token_raises():
    with pytest.raises(ValueError, match="Unrecognized phenotype token"):
        encode_phenotype("maybe")


def test_unrecognized_numeric_raises():
    # A stray numeric phenotype (e.g. an MIC value that slipped through)
    # must raise, not silently become NaN.
    with pytest.raises(ValueError):
        encode_phenotype(32)


def test_token_sets_are_the_permissive_superset():
    # B2: cell 3 accepted "res"/"sus"/"int" abbreviations, cell 5 did not.
    # This module must accept the union (cell 3's superset).
    assert {"r", "res", "resistant", "i", "int", "intermediate"} <= RESISTANT_TOKENS
    assert {"s", "sus", "susceptible"} <= SUSCEPTIBLE_TOKENS


def test_build_label_matrix_shapes():
    df = pd.DataFrame({
        ANTIBIOTIC_COLUMN_MAP["AMP"]: ["R", "S", None, "I"],
        ANTIBIOTIC_COLUMN_MAP["CIP"]: ["S", "R", "S", None],
        ANTIBIOTIC_COLUMN_MAP["AMC"]: ["S"] * 4,
        ANTIBIOTIC_COLUMN_MAP["SXT"]: ["S"] * 4,
        ANTIBIOTIC_COLUMN_MAP["GEN"]: ["S"] * 4,
        ANTIBIOTIC_COLUMN_MAP["CTX"]: ["S"] * 4,
        ANTIBIOTIC_COLUMN_MAP["CAZ"]: ["S"] * 4,
    })
    Y, mask = build_label_matrix(df)
    assert Y.shape == (4, len(ANTIBIOTICS))
    assert mask.shape == Y.shape
    amp_idx = ANTIBIOTICS.index("AMP")
    assert Y[0, amp_idx] == 1.0
    assert Y[1, amp_idx] == 0.0
    assert mask[2, amp_idx] == False  # noqa: E712 -- missing
    assert Y[3, amp_idx] == 1.0  # "I" -> resistant


def test_audit_raw_phenotype_values_reports_all_tokens():
    df = pd.DataFrame({
        ANTIBIOTIC_COLUMN_MAP["AMP"]: ["R", "R", "S", "res"],
        **{ANTIBIOTIC_COLUMN_MAP[ab]: ["S"] * 4 for ab in ANTIBIOTICS if ab != "AMP"},
    })
    report = audit_raw_phenotype_values(df, antibiotics=["AMP"])
    tokens = set(report["raw_token"])
    assert {"r", "s", "res"} <= tokens
    r_count = report.loc[report["raw_token"] == "r", "count"].iloc[0]
    assert r_count == 2


# ---------------------------------------------------------------------------
# B2 resolution -- real raw phenotype_matrix_08302024.csv now available
# (OPEN_QUESTIONS.md #3, resolved). See docs/AUDIT_FINDINGS.md and
# docs/CHANGELOG_NUMBERS.md for the full writeup: the two original notebook
# encoders (permissive "res"/"sus"/"int" vs. strict "resistant"/
# "susceptible"/"intermediate" only) NEVER actually diverged on real data,
# because the raw file only ever contains the full-word tokens.
# ---------------------------------------------------------------------------
def test_real_phenotype_matrix_contains_only_full_word_tokens(raw_phenotype_df):
    """The actual raw file, across all 7 antibiotic columns, contains
    exactly {resistant, susceptible, intermediate, nan} -- none of the
    abbreviated tokens ("res", "sus", "int", "r", "s", "i") that the two
    original notebook encoders disagreed about ever occur. This is the
    real-data proof that B2's divergence never fired in practice."""
    report = audit_raw_phenotype_values(raw_phenotype_df)
    observed_tokens = set(report["raw_token"]) - {"nan"}
    assert observed_tokens == {"resistant", "susceptible", "intermediate"}


def test_real_phenotype_matrix_encodes_without_error(raw_phenotype_df):
    # If any token outside the recognized sets existed, this would raise.
    Y, mask = build_label_matrix(raw_phenotype_df)
    assert Y.shape == (len(raw_phenotype_df), len(ANTIBIOTICS))


def test_real_phenotype_matrix_matches_locked_master_features_y(raw_phenotype_df, master_features):
    """The definitive B2 cross-check: build Y from the raw CSV with THIS
    module's encoder, align by Genome.Name to the locked
    arcadia_v2_master_features.pkl (built by the original notebook's
    stricter cell-5 encoder), and diff every non-missing cell.
    """
    pheno_indexed = raw_phenotype_df.set_index("Genome.Name")
    master_names = pd.Series(master_features["genome_names"]).astype(str)
    aligned = pheno_indexed.reindex(master_names.values)

    assert aligned.notna().any(axis=1).sum() == len(master_names), (
        "Every genome in the locked master features must be found in the "
        "raw phenotype file by name."
    )

    Y_new, mask_new = build_label_matrix(aligned)
    Y_locked = master_features["Y"].astype(np.float64)
    mask_locked = master_features["label_mask"]

    both_labeled = mask_new & mask_locked
    assert int(both_labeled.sum()) > 30000  # sanity: comparison isn't vacuous
    assert np.array_equal(mask_new, mask_locked)
    assert np.array_equal(Y_new[both_labeled], Y_locked[both_labeled])
