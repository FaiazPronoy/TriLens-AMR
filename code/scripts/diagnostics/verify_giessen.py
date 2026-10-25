"""Verify Giessen dataset format before integrating into Stage 1 loaders.

Run from inside code/:
    python scripts/diagnostics/verify_giessen.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]   # the code/ folder
DATA_CSV = ROOT / "data" / "Giessen_dataset" / "cip_ctx_ctz_gen_multi_data.csv"
PHENO_CSV = ROOT / "data" / "Giessen_dataset" / "cip_ctx_ctz_gen_pheno.csv"


def hr(title: str = "") -> None:
    print("\n" + "=" * 78)
    if title:
        print(f"  {title}")
        print("=" * 78)


def main() -> int:
    hr("FILE EXISTENCE CHECK")
    for p in (DATA_CSV, PHENO_CSV):
        exists = p.exists()
        size_mb = p.stat().st_size / 1e6 if exists else 0
        print(f"  {'OK ' if exists else 'MISSING'}  {p}    ({size_mb:.1f} MB)")
    if not (DATA_CSV.exists() and PHENO_CSV.exists()):
        print("\nERROR: one or both files missing. Stopping.")
        return 1

    # -----------------------------------------------------------------
    hr("LOAD DATA CSV")
    print(f"  Loading {DATA_CSV.name} ...")
    data = pd.read_csv(DATA_CSV)
    print(f"  shape: {data.shape}")
    print(f"  dtypes (first 5): {dict(list(data.dtypes.items())[:5])}")
    print(f"  first 5 columns: {list(data.columns[:5])}")
    print(f"  last  3 columns: {list(data.columns[-3:])}")
    print(f"  memory:          {data.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    print("\n  First column head (first 5 rows):")
    print(data.iloc[:5, 0].to_string(index=False))

    snp_cols = [c for c in data.columns if c != "prename"]
    print(f"\n  n SNP-like columns (everything except 'prename'): {len(snp_cols)}")
    if snp_cols:
        print(f"  first 5 SNP col names: {snp_cols[:5]}")
        print(f"  last  3 SNP col names: {snp_cols[-3:]}")

    print("\n  Numeric sample (rows 0-2, SNP cols 1-5):")
    if len(snp_cols) >= 5:
        print(data[snp_cols[:5]].head(3).to_string(index=False))

    # Unique-value scan on a sample of SNP columns
    print("\n  Unique value scan (sample of 5 SNP cols):")
    sample_cols = snp_cols[: min(5, len(snp_cols))]
    for c in sample_cols:
        u = pd.unique(data[c].dropna())
        u_sorted = sorted(u.tolist())
        print(f"    {c:<15}  uniques: {u_sorted[:15]}  (n_uniques={len(u_sorted)})")

    # Aggregate value range across ALL SNP columns
    print("\n  Aggregate value range across ALL SNP columns:")
    try:
        sub = data[snp_cols].to_numpy()
        print(f"    min      = {np.nanmin(sub)}")
        print(f"    max      = {np.nanmax(sub)}")
        print(f"    nan_pct  = {100.0 * pd.isna(sub).sum() / sub.size:.3f} %")
        # Histogram-ish
        flat = sub[~pd.isna(sub)].astype(int) if sub.dtype != object else None
        if flat is not None:
            uniq, counts = np.unique(flat, return_counts=True)
            tot = counts.sum()
            print(f"    value distribution:")
            for u, c in zip(uniq, counts):
                print(f"      {u}: {c:>10,}  ({100.0 * c / tot:.2f} %)")
    except Exception as ex:  # noqa: BLE001
        print(f"    [could not aggregate: {ex!r}]")

    # -----------------------------------------------------------------
    hr("LOAD PHENO CSV")
    pheno = pd.read_csv(PHENO_CSV)
    print(f"  shape: {pheno.shape}")
    print(f"  columns: {list(pheno.columns)}")
    print(f"  dtypes : {dict(pheno.dtypes)}")
    print("\n  Head:")
    print(pheno.head().to_string(index=False))

    print("\n  Per-antibiotic value counts (incl. NaN):")
    for c in pheno.columns:
        if c == "prename":
            continue
        vc = pheno[c].value_counts(dropna=False).to_dict()
        print(f"    {c:<6}  {vc}")

    # -----------------------------------------------------------------
    hr("MERGE CHECK ON 'prename'")
    if "prename" not in data.columns:
        print("  ERROR: 'prename' missing in data CSV.")
        return 2
    if "prename" not in pheno.columns:
        print("  ERROR: 'prename' missing in pheno CSV.")
        return 2

    n_data = data["prename"].nunique()
    n_pheno = pheno["prename"].nunique()
    n_inter = len(set(data["prename"]) & set(pheno["prename"]))
    print(f"  unique prename in data : {n_data}")
    print(f"  unique prename in pheno: {n_pheno}")
    print(f"  intersection           : {n_inter}")
    print(f"  data-only              : {n_data - n_inter}")
    print(f"  pheno-only             : {n_pheno - n_inter}")

    merged = data.merge(pheno, on="prename", how="inner")
    print(f"  merged shape           : {merged.shape}")

    # -----------------------------------------------------------------
    hr("PER-ANTIBIOTIC USABLE SAMPLE COUNTS (after dropping NaN labels)")
    abs_ = [c for c in pheno.columns if c != "prename"]
    print(f"  Antibiotics present: {abs_}")
    for ab in abs_:
        sub = merged.dropna(subset=[ab])
        n = len(sub)
        n_pos = int((sub[ab] == 1).sum())
        n_neg = int((sub[ab] == 0).sum())
        n_other = n - n_pos - n_neg
        pct_pos = (100.0 * n_pos / n) if n else 0.0
        print(f"  {ab:<6}  n_usable={n:<4}  pos={n_pos:<4} ({pct_pos:.1f} %)  "
              f"neg={n_neg:<4}  other={n_other}")

    # -----------------------------------------------------------------
    hr("EXPECTATION vs ACTUAL")
    expected = {
        "samples (pheno rows)": 809,
        "SNPs (data SNP cols)": 60936,
        "antibiotics (pheno cols - prename)": 4,
    }
    actual = {
        "samples (pheno rows)": int(pheno.shape[0]),
        "SNPs (data SNP cols)": len(snp_cols),
        "antibiotics (pheno cols - prename)": len([c for c in pheno.columns
                                                    if c != "prename"]),
    }
    print(f"  {'metric':<40} {'expected':>10} {'actual':>10}   match?")
    for k in expected:
        e = expected[k]
        a = actual[k]
        ok = "OK" if e == a else "MISMATCH"
        print(f"  {k:<40} {e:>10} {a:>10}   {ok}")

    hr("DONE — paste output back to Claude")
    return 0


if __name__ == "__main__":
    sys.exit(main())
