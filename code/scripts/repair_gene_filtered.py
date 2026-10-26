"""
repair_gene_filtered.py
=======================
Rebuilds the SNP values of a gene-filtered dataset from the SNP matrix it was cut
from, keeping everything else (selected genes, column order, splits, labels).

Why: the October 2026 data check found that `arcadia_v2_gene_filtered.pkl` (random
split) has 42 of its 262,225 columns filled with values from -128 to 127, while
SNP genotypes are 0 to 3 and the SNP matrix itself is clean. The phylogroup and
lineage-holdout files, cut from the same matrix later, are clean. Each column of
the filtered file is a column of the SNP matrix (`snp_positions_filtered` names
it), so the clean values can be taken back from the matrix.

The repair is only accepted if
  * every filtered column is found in the SNP matrix by its position name,
  * every difference between the file and the rebuild lies inside the damaged
    block, i.e. between the first and the last column with an out-of-range value
    (inside that block some damaged cells happen to fall in 0..3; outside it the
    file must be identical to the matrix, so nothing else is touched),
and the values of the rebuilt file are all 0 to 3. The genes covering the damaged
block are written to the report.

    python scripts/repair_gene_filtered.py --src old.pkl --snp arcadia_v2_snp_matrix.pkl \
        --out new.pkl --report report.json

Exit codes: 0 repaired and written, 2 already identical to the matrix (nothing written),
1 refused.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

VALID_MAX = 3   # SNP genotype codes are 0..3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--snp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    rep: dict = {"src": a.src, "snp": a.snp}

    def done(code, status):
        rep["status"] = status
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, indent=1))
        print(json.dumps(rep, indent=1))
        return code

    with open(a.src, "rb") as f:
        gf = pickle.load(f)
    X_old = np.asarray(gf["X_snp_filtered"])
    bad = (X_old < 0) | (X_old > VALID_MAX)
    bad_cols = np.where(bad.any(axis=0))[0]
    rep.update(shape=list(X_old.shape), out_of_range_cells=int(bad.sum()),
               out_of_range_columns=int(len(bad_cols)), out_of_range_column_index=bad_cols.tolist()[:100])
    if "snp_positions_filtered" not in gf:
        return done(1, "refused: file has no snp_positions_filtered")
    pos_f = np.asarray(gf["snp_positions_filtered"]).astype(str)

    with open(a.snp, "rb") as f:
        snp = pickle.load(f)
    X_snp = snp["X_snp"]
    pos_all = np.asarray(snp["snp_positions"]).astype(str)
    del snp
    if X_snp.shape[0] != X_old.shape[0]:
        return done(1, f"refused: {X_snp.shape[0]} rows in the SNP matrix, {X_old.shape[0]} in the file")
    where, twice = {}, {}
    for i, p in enumerate(pos_all):
        if p in where:
            twice.setdefault(p, [where[p]]).append(i)
        where.setdefault(p, i)
    idx = np.array([where.get(p, -1) for p in pos_f], dtype=np.int64)
    rep["columns_not_found"] = int((idx < 0).sum())
    if (idx < 0).any():
        return done(1, "refused: some filtered columns are not in the SNP matrix")
    # A name used twice in the matrix is harmless if both columns hold the same values;
    # otherwise the mapping is ambiguous and the repair is refused.
    used = set(pos_f.tolist())
    dup_used = {p: cols for p, cols in twice.items() if p in used}
    ambiguous = sorted(p for p, cols in dup_used.items()
                       if any(not np.array_equal(X_snp[:, cols[0]], X_snp[:, c]) for c in cols[1:]))
    rep["duplicated_position_names_used"] = len(dup_used)
    rep["ambiguous_position_names"] = ambiguous[:20]
    if ambiguous:
        return done(1, "refused: some filtered columns name more than one, different, column of the SNP matrix")

    X_new = X_snp[:, idx]
    del X_snp
    diff = X_new != X_old
    diff_cols = np.where(diff.any(axis=0))[0]
    if len(bad_cols) == 0:
        rep.update(cells_changed=int(diff.sum()), columns_changed=int(len(diff_cols)))
        if len(diff_cols):
            rep["example_columns_changed"] = diff_cols.tolist()[:20]
            return done(1, "refused: values in 0..3 but different from the SNP matrix")
        return done(2, "clean: identical to the SNP matrix, nothing to repair")
    lo, hi = int(bad_cols.min()), int(bad_cols.max())
    in_range_changed = np.setdiff1d(diff_cols, bad_cols)
    outside = diff_cols[(diff_cols < lo) | (diff_cols > hi)]
    rep.update(cells_changed=int(diff.sum()), columns_changed=int(len(diff_cols)),
               damaged_block=[lo, hi], damaged_block_width=hi - lo + 1,
               columns_changed_without_out_of_range_values=in_range_changed.tolist()[:50],
               columns_changed_outside_damaged_block=int(len(outside)),
               new_min=int(X_new.min()), new_max=int(X_new.max()))
    g2r = gf.get("gene_to_range") or {}
    rep["genes_in_damaged_block"] = sorted(g for g, (a0, b0) in g2r.items() if a0 <= hi and b0 > lo)[:50]
    if len(outside):
        rep["example_columns_changed_outside_damaged_block"] = outside.tolist()[:20]
        return done(1, "refused: the file differs from the SNP matrix outside the damaged block")
    if X_new.min() < 0 or X_new.max() > VALID_MAX:
        return done(1, "refused: rebuilt values out of range")

    gf["X_snp_filtered"] = np.ascontiguousarray(X_new)
    gf["repair_note"] = (f"{len(bad_cols)} columns with out-of-range values re-read from the SNP matrix "
                         f"by scripts/repair_gene_filtered.py; all other columns identical")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(gf, f, protocol=pickle.HIGHEST_PROTOCOL)
    return done(0, "repaired")


if __name__ == "__main__":
    sys.exit(main())
