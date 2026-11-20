from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trilens.determinants.amrfinder_runner import MIN_COVERAGE, MIN_IDENTITY, NOT_COUNTED_METHODS

ROOT = Path(__file__).resolve().parents[2]
KEY_GENES = [
    "blaTEM-1",
    "blaCTX-M-15",
    "blaCTX-M-14",
    "blaCTX-M-27",
    "blaOXA-1",
    "blaCMY-2",
    "aac(3)-IId",
    "aac(3)-IIe",
    "aac(6')-Ib-cr5",
    "ant(2'')-Ia",
    "sul1",
    "sul2",
    "dfrA17",
    "qnrS1",
]


def ours(hits):
    h = hits.dropna(subset=["symbol"])
    h = h[~h["method"].astype(str).str.upper().isin(NOT_COUNTED_METHODS) & h["type"].str.upper().eq("AMR")]
    acq = h[~h.is_point & (h.identity >= MIN_IDENTITY) & (h.coverage >= MIN_COVERAGE)]
    return acq.groupby("strain_index").symbol.apply(set).to_dict(), h[h.is_point].groupby("strain_index").symbol.apply(
        set
    ).to_dict()


def ncbi(joined, software):
    r = joined[joined.amrfinder_version.eq(software)]
    acq, pt, mode = {}, {}, {}
    for i, s, t in zip(r.strain_index.astype(int), r.AMR_genotypes, r.amrfinder_analysis_type):
        mode[i] = t
        for tok in str(s).split(",") if isinstance(s, str) else []:
            sym, _, suf = tok.strip().partition("=")
            if suf == "":
                acq.setdefault(i, set()).add(sym)
            elif suf == "POINT":
                pt.setdefault(i, set()).add(sym)
    return acq, pt, mode


def compare(a, b, strains, mode, drop=()):
    rows = []
    for i in strains:
        x, y = a.get(i, set()) - set(drop), b.get(i, set()) - set(drop)
        u = x | y
        rows.append(
            dict(
                mode=mode[i],
                exact=x == y,
                jaccard=len(x & y) / len(u) if u else 1.0,
                both=len(x & y),
                only_ours=len(x - y),
                only_ncbi=len(y - x),
            )
        )
    d = pd.DataFrame(rows)
    tot = d[["both", "only_ours", "only_ncbi"]].sum()
    return {
        "strains": len(d),
        "exact_share": round(float(d.exact.mean()), 4),
        "median_jaccard": round(float(d.jaccard.median()), 4),
        "calls_both": int(tot.both),
        "calls_only_ours": int(tot.only_ours),
        "calls_only_ncbi": int(tot.only_ncbi),
        "pair_agreement": round(float(tot.both / tot.sum()), 4),
        "exact_share_by_mode": d.groupby("mode").exact.mean().round(4).to_dict(),
    }


def kappa(x, y):
    pa = (x == y).mean()
    pe = x.mean() * y.mean() + (1 - x.mean()) * (1 - y.mean())
    return 1.0 if pe == 1 else float((pa - pe) / (1 - pe))


def run(a):
    hits = pd.read_csv(a.hits, low_memory=False)
    oa, op = ours(hits)
    joined = pd.read_csv(a.joined, sep="\t", dtype=str)
    na, npt, mode = ncbi(joined, a.software)
    strains = sorted(set(mode) & set(hits.strain_index.astype(int)))
    out = {
        "software": a.software,
        "ncbi_strains": len(mode),
        "overlap": len(strains),
        "acquired": compare(oa, na, strains, mode),
        "acquired_without_emrD": compare(oa, na, strains, mode, drop=["emrD"]),
        "point": compare(op, npt, strains, mode),
        "key_genes": [],
    }
    for g in KEY_GENES:
        x = np.array([g in oa.get(i, set()) for i in strains])
        y = np.array([g in na.get(i, set()) for i in strains])
        out["key_genes"].append(
            dict(gene=g, ncbi=int(y.sum()), ours=int(x.sum()), both=int((x & y).sum()), kappa=round(kappa(x, y), 4))
        )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hits", required=True)
    ap.add_argument("--joined", default=str(ROOT / "results" / "data_inventory" / "pd_amr_joined.tsv.gz"))
    ap.add_argument("--software", default="4.2.7")
    ap.add_argument("--out", default=str(ROOT / "results" / "determinant_matrix_bvbrc" / "concordance.json"))
    run(ap.parse_args())


if __name__ == "__main__":
    main()
