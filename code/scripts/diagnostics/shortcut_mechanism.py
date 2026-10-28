"""
shortcut_mechanism.py
=====================
W12: is W7's within-ST assembly signal batch or biology, and which statistics carry it?
Pre-registration: docs/experiments/w12_shortcut_mechanism.md.

W7 measured each assembly statistic's concordance with the phenotype inside a sequence type. This
repeats it inside (sequence type x submitting study) cells, so a pair is compared only when both
strains share a lineage AND a study. If the signal is a study artefact it falls to chance there; if
it is accessory-content biology it survives. The statistics are split into two groups fixed in the
pre-registration: content (Size, CDS, GC.Content) and technical (Contigs, Contig.N50).

Nothing is fitted; every number is a rank statistic over data already on disk.

    python scripts/diagnostics/shortcut_mechanism.py [--arg-dir results/arg_matrix]
        [--out results/w12_shortcut_mechanism]

Writes concordance.csv, correlations.csv and verdict.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.evaluation.grouped import within_st_concordance  # noqa: E402

AB = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]
# W7's own table, unchanged: the drugs whose determinants are acquired in the Pathogen Detection calls
ACQUIRED_DRUGS = ["AMC", "SXT", "GEN", "CTX", "CAZ"]
GROUPS = {"content": ["Size", "CDS", "GC.Content"], "technical": ["Contigs", "Contig.N50"]}
ASM = GROUPS["content"] + GROUPS["technical"]
NEAR_UNIVERSAL = 0.5     # as in W7: a gene in more than half the strains is treated as chromosomal
MIN_CELLS, MIN_PAIRS = 10, 100       # fixed in the pre-registration
PASS_LEVEL = 0.55


def inv() -> Path:
    r = os.environ.get("TRILENS_REPO_RESULTS")
    return (Path(r) if r else ROOT / "results") / "data_inventory"


def load(arg_dir):
    A = pd.read_csv(Path(arg_dir) / "acquired.csv", index_col=0)
    chromosomal = [c for c in A.columns if A[c].mean() > NEAR_UNIVERSAL]
    n_acq = A.drop(columns=chromosomal).sum(axis=1)
    cov = pd.read_csv(inv() / "covariates.csv").set_index("strain_index")
    tab = pd.read_csv(inv() / "strain_table.csv", dtype={"st": str, "study": str})
    idx = A.index.to_numpy()
    d = cov.loc[idx, ASM].astype(float).copy()
    d["n_acquired"] = n_acq.to_numpy()
    d["st"] = tab.loc[idx, "st"].to_numpy(object)
    d["study"] = tab.loc[idx, "study"].to_numpy(object)
    d["st_study"] = [f"{s}|{t}" for s, t in zip(d.st, d.study)]
    for ab in AB:
        d[ab] = pd.to_numeric(tab.loc[idx, ab], errors="coerce").to_numpy()
    return d


def concordance(d):
    rows = []
    for cells, name in [("st", "within_st"), ("st_study", "within_st_study")]:
        for grp, feats in GROUPS.items():
            for f in feats:
                for ab in AB:
                    r = within_st_concordance(d[ab], d[f], d[cells])
                    rows.append(dict(cells=name, group=grp, feature=f, drug=ab,
                                     concordance=r["macro"], n_cells=r["n_st"], n_pairs=r["n_pairs"]))
    return pd.DataFrame(rows)


def correlations(d):
    rows = []
    for cells, name in [("st", "within_st"), ("st_study", "within_st_study")]:
        sizes = d[cells].value_counts()
        keep = d[cells].isin(sizes[sizes >= MIN_CELLS].index)
        sub = d[keep]
        for grp, feats in GROUPS.items():
            for f in feats:
                x = sub.groupby(cells)[f].transform(lambda v: v - v.mean())
                y = sub.groupby(cells)["n_acquired"].transform(lambda v: v - v.mean())
                ok = x.notna() & y.notna()
                rho = stats.spearmanr(x[ok], y[ok]).statistic if ok.sum() > 10 else np.nan
                rows.append(dict(cells=name, group=grp, feature=f, rho_vs_acquired_genes=rho,
                                 n_strains=int(ok.sum()), n_cells=int(sub[cells].nunique())))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arg-dir", default="results/arg_matrix")
    ap.add_argument("--out", default="results/w12_shortcut_mechanism")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    d = load(a.arg_dir)
    C = concordance(d)
    C.to_csv(out / "concordance.csv", index=False)
    R = correlations(d)
    R.to_csv(out / "correlations.csv", index=False)

    def mean_size(cells):
        s = C[(C.cells == cells) & (C.feature == "Size") & C.drug.isin(ACQUIRED_DRUGS)
              & (C.n_cells >= MIN_CELLS) & (C.n_pairs >= MIN_PAIRS)]
        return (round(float(s.concordance.mean()), 4) if len(s) else None), len(s)

    within_st, n_st_drugs = mean_size("within_st")
    within_both, n_both_drugs = mean_size("within_st_study")
    grp = (C[(C.cells == "within_st_study") & (C.n_cells >= MIN_CELLS) & (C.n_pairs >= MIN_PAIRS)
             & C.drug.isin(ACQUIRED_DRUGS)].groupby("group").concordance.mean().round(4).to_dict())
    v = {
        "size_within_st": within_st, "drugs_reported_within_st": n_st_drugs,
        "size_within_st_study": within_both, "drugs_reported_within_st_study": n_both_drugs,
        "min_cells": MIN_CELLS, "min_pairs": MIN_PAIRS,
        "pass_rule": f"mean Size concordance over {ACQUIRED_DRUGS} within (ST x study) >= {PASS_LEVEL}",
        "group_means_within_st_study": grp,
        "size_rho_vs_acquired_within_st": R[(R.cells == "within_st")
                                            & (R.feature == "Size")].rho_vs_acquired_genes.round(4).tolist(),
        "size_rho_vs_acquired_within_st_study": R[(R.cells == "within_st_study")
                                                  & (R.feature == "Size")].rho_vs_acquired_genes.round(4).tolist(),
    }
    v["survives_study_conditioning"] = bool(within_both is not None and within_both >= PASS_LEVEL)
    v["reading"] = ("the within-lineage assembly signal survives conditioning on the submitting study: "
                    "assembly size behaves as a proxy for accessory content, not as a study artefact"
                    if v["survives_study_conditioning"] else
                    "conditioning on the submitting study removes most of the within-lineage assembly "
                    "signal: W7's shortcut is substantially explained by batch")
    if within_both is not None and within_st is not None:
        v["drop_from_conditioning"] = round(within_st - within_both, 4)
    (out / "verdict.json").write_text(json.dumps(v, indent=1, default=float))
    print(C[(C.n_cells >= MIN_CELLS) & (C.n_pairs >= MIN_PAIRS)].round(3).to_string(index=False))
    print(R.round(3).to_string(index=False))
    print(json.dumps(v, indent=1, default=float))


if __name__ == "__main__":
    main()
