"""
CORRECTED base-rate check for Lens B.
=====================================

prefix_baserate.py computed every lens's base rate over `selected_gene_ids`,
the 5,000 genes kept by the Stage 1.5 filter. That is the right catalogue for
Lens A (SNP blocks) and Lens C (per-gene statistics), but it is the WRONG one
for Lens B, which operates over the full 32,441-gene presence/absence matrix.

So the headline BIKPOGCH enrichment of 11-21x for Lens B was measured against a
denominator that lens never sees, and cannot be trusted as it stands. This script
recomputes it against the 32,441-gene catalogue.

If the enrichment survives, it is a real result: the lens the gate relies on most
concentrates on genes from one assembly across all seven antibiotics, which is
evidence that it encodes a lineage fingerprint rather than a mechanism.

If it collapses to about 1, the Lens B finding was an artefact of the wrong
denominator and must be dropped.

    conda activate amr
    cd code
    python scripts/diagnostics/prefix_baserate_lens_b.py
"""

import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]   # the code/ folder
DIAG = ROOT / "results" / "diagnostics"
DIAG.mkdir(parents=True, exist_ok=True)
PA_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_presence_absence.pkl"
LENSB = ROOT / "results" / "trilens_attnfix" / "attention_B_topgenes.csv"
OUT = DIAG / "prefix_baserate_lens_b.csv"


def prefix_of(x):
    m = re.findall(r"\b([A-Z]{8})_\d{4,5}\b", str(x))
    return m[0] if m else None


def find_gene_names(obj, depth=0):
    """Locate the 32,441-length array of gene identifiers inside the pickle."""
    if depth > 3:
        return None
    if isinstance(obj, dict):
        # prefer obviously-named keys
        for k in ("gene_names", "gene_ids", "genes", "pa_gene_names",
                  "columns", "loci", "gene_list"):
            if k in obj:
                v = np.asarray(obj[k]).ravel()
                if v.dtype.kind in "USO" and len(v) > 10000:
                    return v
        for k, v in obj.items():
            try:
                a = np.asarray(v).ravel()
                if a.dtype.kind in "USO" and len(a) > 10000:
                    print(f"   (using key '{k}', length {len(a)})")
                    return a
            except Exception:
                pass
            r = find_gene_names(v, depth + 1)
            if r is not None:
                return r
    return None


def main():
    if not PA_PKL.exists():
        sys.exit(f"MISSING: {PA_PKL}")
    if not LENSB.exists():
        sys.exit(f"MISSING: {LENSB}")

    with open(PA_PKL, "rb") as fh:
        pa = pickle.load(fh)

    print("=" * 78)
    print("Structure of arcadia_v2_gene_presence_absence.pkl")
    print("=" * 78)
    if isinstance(pa, dict):
        for k, v in pa.items():
            shape = getattr(v, "shape", None)
            print(f"  {k:34s} {type(v).__name__:12s} {shape if shape is not None else ''}")

    names = find_gene_names(pa)
    if names is None:
        sys.exit("\nCould not find the gene-name array. Look at the structure above "
                 "and set it manually in find_gene_names().")

    print(f"\nGene catalogue size: {len(names)}")

    base = Counter(p for p in (prefix_of(g) for g in names) if p)
    n_base = sum(base.values())
    print(f"Genes with a parseable prefix: {n_base}   distinct prefixes: {len(base)}\n")
    for p, c in base.most_common(12):
        print(f"  {p}  {c:6d}  ({c/n_base*100:5.2f}%)")

    d = pd.read_csv(LENSB)
    gcol = "gene_id" if "gene_id" in d.columns else d.columns[2]

    rows = []
    for ab, g in d.groupby("ab"):
        pref = [p for p in (prefix_of(x) for x in g[gcol]) if p]
        if not pref:
            continue
        c = Counter(pref)
        top_p, top_c = c.most_common(1)[0]
        p0 = base.get(top_p, 0) / n_base
        bt = binomtest(top_c, len(pref), p0, alternative="greater") if p0 > 0 else None
        rows.append(dict(
            ab=ab, top_prefix=top_p, observed=f"{top_c}/{len(pref)}",
            observed_pct=round(top_c / len(pref) * 100, 1),
            base_rate_pct=round(p0 * 100, 3),
            enrichment=round((top_c / len(pref)) / p0, 2) if p0 > 0 else None,
            p_value=f"{bt.pvalue:.2e}" if bt else "n/a"))

    res = pd.DataFrame(rows).sort_values("enrichment", ascending=False)
    print("\n" + "=" * 78)
    print("LENS B ENRICHMENT, against the 32,441-gene catalogue it actually reads")
    print("=" * 78)
    print(res.to_string(index=False))
    res.to_csv(OUT, index=False)

    med = res.enrichment.median()
    print(f"\nMedian enrichment across the seven antibiotics: {med:.2f}")
    print("""
VERDICT

  median enrichment < 2    ->  drop it. The earlier 11-21x was the wrong
                               denominator and there is no finding here.

  median enrichment > 5    ->  real. Lens B, which takes 0.675-0.880 of the
      with small p-values       gate weight, concentrates on genes from one
                               assembly across every antibiotic. That is direct
                               evidence for the lineage-fingerprint reading and
                               belongs in Section 5.6.
""")
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()

