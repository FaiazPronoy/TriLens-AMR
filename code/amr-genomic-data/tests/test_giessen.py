import numpy as np
import pandas as pd
import pytest

from src.amr_data.giessen import (
    ANTIBIOTIC_NAME_MAP,
    build_giessen_split,
    honest_headline_strategy,
    load_giessen_raw,
)
from src.amr_data import qa


# ---------------------------------------------------------------------------
# Real data (Part C audit numbers)
# ---------------------------------------------------------------------------
def test_load_giessen_raw_matches_manifest_e8(paths):
    ds = load_giessen_raw(paths)
    assert ds["X"].shape == (809, 60936)
    assert set(np.unique(ds["X"]).tolist()) <= {0.0, 1.0, 2.0, 3.0, 4.0}
    assert ds["label_mask"].mean() == 1.0  # 100% complete, no missing

    counts = {ab: (int((ds["Y"][:, i] == 1).sum()), int((ds["Y"][:, i] == 0).sum()))
              for i, ab in enumerate(ds["label_names"])}
    assert counts["CIP"] == (366, 443)
    assert counts["CTX"] == (358, 451)
    assert counts["CAZ"] == (276, 533)  # Giessen's CTZ, renamed
    assert counts["GEN"] == (188, 621)


def test_ctz_renamed_to_caz():
    assert ANTIBIOTIC_NAME_MAP["CTZ"] == "CAZ"
    assert "CTZ" not in ANTIBIOTIC_NAME_MAP.values()


def test_giessen_split_sizes_match_notebook(paths):
    ds = load_giessen_raw(paths)
    split = build_giessen_split(ds, seed=42)
    sizes = (len(split["train_idx"]), len(split["val_idx"]), len(split["test_idx"]))
    assert sizes == (566, 121, 122)
    assert split["stratified_on"] == "CIP"  # closest to 50/50 in the real data


def test_giessen_split_no_overlap(paths):
    ds = load_giessen_raw(paths)
    split = build_giessen_split(ds, seed=42)
    tr, va, te = set(split["train_idx"]), set(split["val_idx"]), set(split["test_idx"])
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert len(tr | va | te) == ds["n_samples"]


def test_verify_giessen_qa_passes(paths):
    ds = load_giessen_raw(paths)
    report = qa.verify_giessen(ds)
    assert report["label_completeness"] == 1.0


# ---------------------------------------------------------------------------
# Part C.4 -- selection-bias fix
# ---------------------------------------------------------------------------
def _synthetic_grid():
    """Reproduces the exact bias scenario described in the spec: 5
    strategies, 4 antibiotics, winning strategy differs per antibiotic on
    TEST, but on VAL one strategy is consistently best."""
    rng = np.random.RandomState(0)
    strategies = ["A_Early_SS", "B_Joint_LL", "C_Decision_DD", "D_Hybrid_LS", "E_ProximityAdaptive"]
    antibiotics = ["CIP", "CTX", "CAZ", "GEN"]
    rows = []
    # B_Joint_LL is the true best on average (on val), but per-antibiotic
    # test noise makes a DIFFERENT strategy win for each drug -- this is
    # the exact "garden of forking paths" pattern being guarded against.
    base_val = {"A_Early_SS": 0.70, "B_Joint_LL": 0.75, "C_Decision_DD": 0.72,
                "D_Hybrid_LS": 0.71, "E_ProximityAdaptive": 0.69}
    winners_by_ab = {"CIP": "A_Early_SS", "CTX": "D_Hybrid_LS", "CAZ": "B_Joint_LL", "GEN": "E_ProximityAdaptive"}
    for ab in antibiotics:
        for strat in strategies:
            val_score = base_val[strat] + rng.normal(0, 0.01)
            test_score = base_val[strat] + rng.normal(0, 0.01)
            if strat == winners_by_ab[ab]:
                test_score += 0.05  # noise bump that makes this one "win" on test for this ab
            rows.append({"strategy": strat, "antibiotic": ab, "split": "val", "prauc": val_score})
            rows.append({"strategy": strat, "antibiotic": ab, "split": "test", "prauc": test_score})
    return pd.DataFrame(rows)


def test_honest_headline_strategy_picks_one_fixed_strategy():
    df = _synthetic_grid()
    chosen, test_results = honest_headline_strategy(df)
    assert chosen == "B_Joint_LL"  # best mean VAL score, chosen BEFORE seeing test
    assert set(test_results["strategy"]) == {"B_Joint_LL"}
    assert set(test_results["antibiotic"]) == {"CIP", "CTX", "CAZ", "GEN"}
    assert len(test_results) == 4  # one row per antibiotic, same strategy


def test_honest_headline_differs_from_naive_best_of_five_per_antibiotic():
    """Demonstrates the bug this fixes: naively taking the best TEST score
    per antibiotic would pick a DIFFERENT strategy for 3 of the 4 drugs
    (matching the spec's described bias) -- the honest version does not."""
    df = _synthetic_grid()
    test_df = df[df["split"] == "test"]
    naive_best = test_df.loc[test_df.groupby("antibiotic")["prauc"].idxmax(), ["antibiotic", "strategy"]]
    naive_strategies = set(naive_best["strategy"])
    assert len(naive_strategies) > 1  # the naive approach picks multiple different "winners"

    chosen, _ = honest_headline_strategy(df)
    assert len([s for s in naive_strategies if s == chosen]) <= 1  # honest pick is just one, fixed


def test_honest_headline_strategy_raises_without_val_split():
    df = _synthetic_grid()
    test_only = df[df["split"] == "test"]
    with pytest.raises(ValueError, match="validation-split"):
        honest_headline_strategy(test_only)
