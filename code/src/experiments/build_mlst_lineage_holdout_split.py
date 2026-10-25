"""
src/experiments/build_mlst_lineage_holdout_split.py
======================================================
STEP 0 + STEP 1 of the final experiment: build a GENUINE lineage holdout in
which no sequence type (ST) appearing in test/val appears anywhere in train.

This is the split the thesis's own Chapter 2 gap actually calls for. The
existing `phylogeny_split` is a *stratified re-partition* (every phylogroup
is proportionally present on BOTH sides -- nothing is held out); see
docs/AUDIT_FINDINGS.md B1. This script holds out whole Achtman STs.

STEP 0 -- MLST parsing and join
-------------------------------
Source: ecoli_7000_amr/.../dataset_generation/data/strains_metadata.csv
        (6,985 rows x 88 cols)

* ST is parsed with r'MLST\\.Escherichia_coli_1\\.(\\d+)' -- the Achtman
  scheme ONLY. Some genomes carry a `_2` (Pasteur) ST as well or instead;
  those are NOT mixed in, because the same integer means a completely
  different lineage under the two schemes. `_2`-only genomes (154 rows in
  the metadata) are treated as unknown-ST.
* The join is by genome/strain ID, NEVER by row position (this session
  already hit one row-position alignment bug). Two independent ID columns
  both match our 6,962 genomes 100%:
      metadata['Genome.Name']    <-> pkl['genome_names']
      metadata['SRA.Accession']  <-> pkl['sra_ids']
  Both are duplicate-free (6,985 unique values each). This script performs
  the join BOTH ways and asserts the two results agree on every genome --
  if they ever disagree, the join is wrong and the script fails loudly.

STEP 1 -- the split rules (fixed BEFORE seeing any result, not tuned)
--------------------------------------------------------------------
ST sizes are extremely uneven: the largest ST has 884 genomes (12.7% of the
dataset) while 214 STs have exactly one genome. A naive group split lets a
single clone dominate the test set.

  1. Unknown-ST genomes            -> TRAIN only (keeps the data, keeps test clean)
  2. Any ST with > 5% of all genomes -> TRAIN only
        (a test set filled by one clone measures performance on that clone,
         not generalisation across lineages. The rule is defined on SIZE,
         not on any observed score, so it cannot be outcome-tuned.)
  3. All remaining STs -> assigned WHOLE to train/val/test, seed 42,
        targeting ~15% test and ~15% val BY GENOME COUNT.
  4. No ST may ever appear in two partitions.

NOTE on GroupShuffleSplit: sklearn's `GroupShuffleSplit(test_size=f)`
allocates a fraction `f` of *GROUPS*, not of samples (verified empirically
against sklearn 1.6.1 on a synthetic uneven-group set: test_size=0.30 gave
30.4% of groups but 56.0% of samples). Because our ST sizes span 1..884,
a group-fraction target makes the resulting genome count unpredictable and
could silently violate the required 800-1400 test-size bound. This script
therefore uses the same underlying mechanism -- a seeded (42) random
permutation of the eligible STs -- but walks that permutation assigning
WHOLE STs until the genome-count target is met. It is deterministic, fixed
in advance, and group-exclusive by construction; it just controls the axis
(genomes) the experiment actually needs.

Hard assertions (all must pass or the script raises):
    train/val/test ST sets pairwise disjoint
    train/val/test genome-index sets pairwise disjoint, union == all 6,962
    every antibiotic has >= 20 resistant AND >= 20 susceptible in test
    800 <= len(test) <= 1400

Stop condition (NOT an assertion -- reported for a human decision):
    if any antibiotic's |test - train| resistance-prevalence shift exceeds
    20 percentage points, the script writes STOP_CONDITION_TRIGGERED into
    its summary and exits non-zero so the downstream GPU chain does not run.

Output: results/mlst_lineage_holdout/
    mlst_lineage_holdout_split.pkl   (train_idx/val_idx/test_idx + ST arrays)
    split_diagnostics.csv
    SUMMARY.md

CPU-only, seconds to run.

Usage:
    conda activate amr
    python -m src.experiments.build_mlst_lineage_holdout_split
"""
from __future__ import annotations

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MASTER_PKL = PROCESSED_DIR / "arcadia_v2_master_features.pkl"
METADATA_CSV = (ROOT / "ecoli_7000_amr" / "ecoli_7000_amr" /
                "dataset_generation" / "data" / "strains_metadata.csv")
OUT_DIR = ROOT / "results" / "mlst_lineage_holdout"
OUT_PKL = PROCESSED_DIR / "mlst_lineage_holdout_split.pkl"

ACHTMAN_PATTERN = re.compile(r"MLST\.Escherichia_coli_1\.(\d+)")
SEED = 42
BIG_ST_FRACTION = 0.05      # rule 2
TARGET_TEST_FRAC = 0.15     # rule 3, by genome count
TARGET_VAL_FRAC = 0.15
MIN_MATCH_RATE = 0.90
MIN_CLASS_COUNT_IN_TEST = 20
TEST_SIZE_BOUNDS = (800, 1400)
MAX_PREVALENCE_SHIFT_PP = 20.0


def parse_achtman_st(value) -> str | None:
    """Extract the Achtman (scheme _1) ST. Returns None if absent."""
    if not isinstance(value, str):
        return None
    m = ACHTMAN_PATTERN.search(value)
    return m.group(1) if m else None


def load_and_join() -> tuple[np.ndarray, dict]:
    """STEP 0. Returns (st_per_genome, info) aligned to the pkl's genome order."""
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"metadata not found: {METADATA_CSV}")
    meta = pd.read_csv(METADATA_CSV)
    with open(MASTER_PKL, "rb") as f:
        master = pickle.load(f)

    genome_names = np.asarray(master["genome_names"], dtype=str)
    sra_ids = np.asarray(master["sra_ids"], dtype=str)
    n = len(genome_names)

    meta_st = meta["MLST"].apply(parse_achtman_st)

    for col in ("Genome.Name", "SRA.Accession"):
        c = meta[col].astype(str)
        if c.duplicated().any():
            raise ValueError(
                f"metadata column {col} has duplicates -- cannot be used as a join key."
            )

    map_by_name = dict(zip(meta["Genome.Name"].astype(str), meta_st))
    map_by_sra = dict(zip(meta["SRA.Accession"].astype(str), meta_st))

    # Separate "genome absent from metadata" from "present but no Achtman ST".
    # Conflating them would silently turn a broken join into a pile of
    # unknown-ST genomes instead of an error.
    missing_by_name = [g for g in genome_names if g not in map_by_name]
    missing_by_sra = [s for s in sra_ids if s not in map_by_sra]
    if missing_by_name or missing_by_sra:
        raise ValueError(
            f"Join incomplete: {len(missing_by_name)} genomes absent from metadata "
            f"by Genome.Name, {len(missing_by_sra)} by SRA.Accession "
            f"(examples: {missing_by_name[:3]} / {missing_by_sra[:3]}). "
            f"Refusing to proceed."
        )

    st_by_name = np.array([map_by_name[g] for g in genome_names], dtype=object)
    st_by_sra = np.array([map_by_sra[s] for s in sra_ids], dtype=object)

    # Cross-validate the two independent joins. They must agree everywhere.
    disagree = [i for i in range(n) if st_by_name[i] != st_by_sra[i]]
    if disagree:
        raise ValueError(
            f"Join disagreement on {len(disagree)} genomes between "
            f"Genome.Name-join and SRA.Accession-join (first few: {disagree[:5]}). "
            f"Refusing to proceed -- the join key is not trustworthy."
        )

    st = st_by_name
    known = np.array([s is not None for s in st])
    match_rate = float(known.mean())

    info = {
        "n_genomes": n,
        "n_metadata_rows": len(meta),
        "n_with_st": int(known.sum()),
        "n_without_st": int((~known).sum()),
        "match_rate": match_rate,
        "join_keys_cross_validated": True,
        "n_distinct_st": len(set(st[known].tolist())),
    }
    return st, info


def build_split(st: np.ndarray, n: int) -> tuple[dict, dict]:
    """STEP 1. Returns (split_dict, diagnostics_dict)."""
    known_mask = np.array([s is not None for s in st])
    st_sizes = pd.Series(st[known_mask].tolist()).value_counts()

    big_threshold = BIG_ST_FRACTION * n
    big_sts = set(st_sizes[st_sizes > big_threshold].index.tolist())

    eligible_sts = [s for s in st_sizes.index.tolist() if s not in big_sts]
    eligible_mask = np.array(
        [(s is not None) and (s not in big_sts) for s in st], dtype=bool
    )

    target_test = int(round(TARGET_TEST_FRAC * n))
    target_val = int(round(TARGET_VAL_FRAC * n))

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(eligible_sts))
    shuffled = [eligible_sts[i] for i in perm]

    st_to_indices: dict[str, list[int]] = {}
    for i, s in enumerate(st):
        if eligible_mask[i]:
            st_to_indices.setdefault(s, []).append(i)

    test_sts: list[str] = []
    val_sts: list[str] = []
    n_test = n_val = 0
    cursor = 0
    while cursor < len(shuffled) and n_test < target_test:
        s = shuffled[cursor]
        test_sts.append(s)
        n_test += len(st_to_indices[s])
        cursor += 1
    while cursor < len(shuffled) and n_val < target_val:
        s = shuffled[cursor]
        val_sts.append(s)
        n_val += len(st_to_indices[s])
        cursor += 1
    train_eligible_sts = shuffled[cursor:]

    test_set, val_set = set(test_sts), set(val_sts)
    train_eligible_set = set(train_eligible_sts)

    test_idx, val_idx, train_idx = [], [], []
    for i, s in enumerate(st):
        if s is None:                    # rule 1
            train_idx.append(i)
        elif s in big_sts:               # rule 2
            train_idx.append(i)
        elif s in test_set:
            test_idx.append(i)
        elif s in val_set:
            val_idx.append(i)
        else:
            train_idx.append(i)

    split = {
        "train_idx": np.array(sorted(train_idx), dtype=np.int64),
        "val_idx": np.array(sorted(val_idx), dtype=np.int64),
        "test_idx": np.array(sorted(test_idx), dtype=np.int64),
    }
    diag = {
        "big_sts": sorted(big_sts, key=lambda s: -st_sizes[s]),
        "big_st_sizes": {s: int(st_sizes[s]) for s in
                         sorted(big_sts, key=lambda s: -st_sizes[s])},
        "big_st_threshold": big_threshold,
        "n_eligible_genomes": int(eligible_mask.sum()),
        "n_eligible_sts": len(eligible_sts),
        "train_sts": sorted(train_eligible_set | big_sts),
        "val_sts": sorted(val_set),
        "test_sts": sorted(test_set),
        "target_test": target_test,
        "target_val": target_val,
    }
    return split, diag


def assert_split_valid(split: dict, st: np.ndarray, Y: np.ndarray,
                       label_mask: np.ndarray, label_names: list[str], n: int) -> None:
    """All hard checks. Raises AssertionError on any violation."""
    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

    # index disjointness + completeness
    s_tr, s_va, s_te = set(tr.tolist()), set(va.tolist()), set(te.tolist())
    assert not (s_tr & s_te), f"train/test genome-index overlap: {len(s_tr & s_te)}"
    assert not (s_tr & s_va), f"train/val genome-index overlap: {len(s_tr & s_va)}"
    assert not (s_va & s_te), f"val/test genome-index overlap: {len(s_va & s_te)}"
    assert len(s_tr) + len(s_va) + len(s_te) == n, "partitions do not cover all genomes"

    # ST disjointness -- the whole point of the experiment
    def sts_of(idx):
        return {st[i] for i in idx if st[i] is not None}
    st_tr, st_va, st_te = sts_of(tr), sts_of(va), sts_of(te)
    assert not (st_tr & st_te), f"TRAIN/TEST ST LEAK: {sorted(st_tr & st_te)[:10]}"
    assert not (st_tr & st_va), f"TRAIN/VAL ST LEAK: {sorted(st_tr & st_va)[:10]}"
    assert not (st_va & st_te), f"VAL/TEST ST LEAK: {sorted(st_va & st_te)[:10]}"

    # test size bound
    lo, hi = TEST_SIZE_BOUNDS
    assert lo <= len(te) <= hi, f"test size {len(te)} outside [{lo},{hi}]"

    # per-antibiotic class counts in test
    for j, ab in enumerate(label_names):
        m = label_mask[te, j]
        y = Y[te, j][m]
        n_r = int((y == 1).sum())
        n_s = int((y == 0).sum())
        assert n_r >= MIN_CLASS_COUNT_IN_TEST, f"{ab}: only {n_r} resistant in test"
        assert n_s >= MIN_CLASS_COUNT_IN_TEST, f"{ab}: only {n_s} susceptible in test"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 0 -- MLST parsing and ID-based join")
    print("=" * 78)
    st, info = load_and_join()
    n = info["n_genomes"]
    print(f"  metadata rows        : {info['n_metadata_rows']}")
    print(f"  dataset genomes      : {n}")
    print(f"  join keys            : Genome.Name AND SRA.Accession, "
          f"cross-validated -> agree on all {n}")
    print(f"  genomes WITH Achtman ST : {info['n_with_st']} "
          f"({100*info['match_rate']:.1f}%)")
    print(f"  genomes WITHOUT ST      : {info['n_without_st']}")
    print(f"  distinct STs            : {info['n_distinct_st']}")

    if info["match_rate"] < MIN_MATCH_RATE:
        print(f"\n  STOP: match rate {100*info['match_rate']:.1f}% is below "
              f"{100*MIN_MATCH_RATE:.0f}% -- the join key may be wrong.")
        return 2

    with open(MASTER_PKL, "rb") as f:
        master = pickle.load(f)
    Y = master["Y"].astype(int)
    label_mask = master["label_mask"].astype(bool)
    label_names = [str(x) for x in master["label_names"]]

    print()
    print("=" * 78)
    print("STEP 1 -- lineage holdout split")
    print("=" * 78)
    split, diag = build_split(st, n)
    print(f"  rule 2 threshold: ST with > {diag['big_st_threshold']:.1f} genomes "
          f"(> {100*BIG_ST_FRACTION:.0f}%) -> train only")
    print(f"  STs forced to train: {diag['big_st_sizes']}")
    print(f"  eligible pool: {diag['n_eligible_genomes']} genomes "
          f"across {diag['n_eligible_sts']} STs")

    assert_split_valid(split, st, Y, label_mask, label_names, n)
    print("\n  ALL HARD ASSERTIONS PASSED "
          "(ST disjointness, index disjointness, test bounds, >=20 R/S per antibiotic)")

    tr, va, te = split["train_idx"], split["val_idx"], split["test_idx"]

    def n_sts(idx):
        return len({st[i] for i in idx if st[i] is not None})

    print()
    print("  partition sizes:")
    print(f"    train: {len(tr):5d} genomes, {n_sts(tr):4d} distinct STs "
          f"(incl. {info['n_without_st']} unknown-ST + "
          f"{sum(diag['big_st_sizes'].values())} big-ST genomes)")
    print(f"    val  : {len(va):5d} genomes, {n_sts(va):4d} distinct STs")
    print(f"    test : {len(te):5d} genomes, {n_sts(te):4d} distinct STs")

    rows = []
    max_shift = 0.0
    worst_ab = None
    print()
    print("  per-antibiotic resistance prevalence (train vs test):")
    print(f"    {'AB':<5} {'n_train':>8} {'n_test':>7} {'prev_train':>11} "
          f"{'prev_test':>10} {'shift_pp':>9}")
    for j, ab in enumerate(label_names):
        m_tr, m_te = label_mask[tr, j], label_mask[te, j]
        y_tr, y_te = Y[tr, j][m_tr], Y[te, j][m_te]
        p_tr, p_te = float(y_tr.mean()), float(y_te.mean())
        shift_pp = (p_te - p_tr) * 100
        if abs(shift_pp) > abs(max_shift):
            max_shift, worst_ab = shift_pp, ab
        print(f"    {ab:<5} {len(y_tr):8d} {len(y_te):7d} {100*p_tr:10.1f}% "
              f"{100*p_te:9.1f}% {shift_pp:+8.1f}")
        rows.append({
            "antibiotic": ab, "n_train_labelled": len(y_tr), "n_test_labelled": len(y_te),
            "prev_train": p_tr, "prev_test": p_te, "shift_pp": shift_pp,
            "n_test_resistant": int((y_te == 1).sum()),
            "n_test_susceptible": int((y_te == 0).sum()),
        })
    pd.DataFrame(rows).to_csv(OUT_DIR / "split_diagnostics.csv", index=False)
    print(f"\n  largest prevalence shift: {max_shift:+.1f} pp ({worst_ab})")

    payload = {
        "train_idx": tr, "val_idx": va, "test_idx": te,
        "st_per_genome": st,
        "info": info, "diag_meta": {k: v for k, v in diag.items()
                                     if k not in ("train_sts", "val_sts", "test_sts")},
        "train_sts": diag["train_sts"], "val_sts": diag["val_sts"],
        "test_sts": diag["test_sts"],
        "rules": {
            "scheme": "Achtman (MLST.Escherichia_coli_1) only; Pasteur _2 ignored",
            "unknown_st": "train only",
            "big_st_fraction": BIG_ST_FRACTION,
            "target_test_frac": TARGET_TEST_FRAC,
            "target_val_frac": TARGET_VAL_FRAC,
            "seed": SEED,
        },
    }
    with open(OUT_PKL, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"\n  saved split -> {OUT_PKL}")

    triggered = abs(max_shift) > MAX_PREVALENCE_SHIFT_PP
    lines = [
        "# MLST lineage holdout split -- Step 0 + Step 1 diagnostics\n",
        f"- metadata rows: {info['n_metadata_rows']}, dataset genomes: {n}",
        f"- join: `Genome.Name` AND `SRA.Accession`, cross-validated, agree on all {n}",
        f"- genomes with Achtman ST: **{info['n_with_st']} ({100*info['match_rate']:.1f}%)**; "
        f"without: {info['n_without_st']}",
        f"- distinct STs: {info['n_distinct_st']}",
        f"- STs forced to train (>{100*BIG_ST_FRACTION:.0f}% of data): "
        f"{diag['big_st_sizes']}\n",
        "## Partition sizes\n",
        "| Partition | Genomes | Distinct STs |",
        "|---|---|---|",
        f"| train | {len(tr)} | {n_sts(tr)} |",
        f"| val | {len(va)} | {n_sts(va)} |",
        f"| test | {len(te)} | {n_sts(te)} |\n",
        "All hard assertions passed: ST sets pairwise disjoint, genome-index sets "
        "pairwise disjoint and complete, test size within bounds, >=20 resistant and "
        ">=20 susceptible per antibiotic in test.\n",
        "## Prevalence shift (train -> test)\n",
        "| AB | n_train | n_test | prev_train | prev_test | shift (pp) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['antibiotic']} | {r['n_train_labelled']} | {r['n_test_labelled']} | "
            f"{100*r['prev_train']:.1f}% | {100*r['prev_test']:.1f}% | "
            f"{r['shift_pp']:+.1f} |"
        )
    lines.append(f"\n**Largest shift: {max_shift:+.1f} pp ({worst_ab})**\n")
    if triggered:
        lines.append(
            f"## STOP_CONDITION_TRIGGERED\n\n"
            f"A prevalence shift of {max_shift:+.1f} pp exceeds the "
            f"{MAX_PREVALENCE_SHIFT_PP:.0f} pp limit. No GPU stage was started. "
            f"Awaiting a human decision."
        )
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")

    marker = OUT_DIR / "SPLIT_OK.marker"
    if marker.exists():
        marker.unlink()

    if triggered:
        print(f"\n  *** STOP CONDITION TRIGGERED: {max_shift:+.1f} pp > "
              f"{MAX_PREVALENCE_SHIFT_PP:.0f} pp -- not starting any GPU stage. ***")
        return 3

    # Written ONLY when every check passed and the stop condition did not fire.
    # The downstream GPU chain gates on this file, so a triggered stop can
    # never silently let hours of GPU work start.
    marker.write_text(
        f"largest_prevalence_shift_pp={max_shift:+.2f}\n"
        f"test_genomes={len(te)}\ntest_sts={n_sts(te)}\n",
        encoding="utf-8",
    )
    print("\n  Stop condition NOT triggered -- safe to proceed to GPU stages.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
