from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from scipy import stats

from trilens.evaluation.cluster_bootstrap import within_st_concordance

ROOT = Path(__file__).resolve().parents[2]
AB = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]

MECHANISM = {
    "AMP": "both",
    "CIP": "chromosomal",
    "AMC": "acquired",
    "SXT": "acquired",
    "GEN": "acquired",
    "CTX": "acquired",
    "CAZ": "acquired",
}
ASM = ["Size", "CDS", "Contigs", "Contig.N50", "GC.Content"]
NEAR_UNIVERSAL = 0.5
MIN_ST = 10


def inv() -> Path:
    r = os.environ.get("TRILENS_REPO_RESULTS")
    return (Path(r) if r else ROOT / "results") / "data_inventory"


def load(determinant_matrix):
    A = pd.read_csv(determinant_matrix, index_col=0)
    chromosomal = [c for c in A.columns if A[c].mean() > NEAR_UNIVERSAL]
    n_acq = A.drop(columns=chromosomal).sum(axis=1)
    cov = pd.read_csv(inv() / "covariates.csv").set_index("strain_index")
    tab = pd.read_csv(inv() / "strain_table.csv", dtype={"st": str, "study": str})
    idx = A.index.to_numpy()
    d = cov.loc[idx, ASM].copy()
    d["n_acquired"] = n_acq.to_numpy()
    d["st"] = tab.loc[idx, "st"].to_numpy(object)
    d["phylogroup"] = tab.loc[idx, "phylogroup"].to_numpy(object)
    for ab in AB:
        d[ab] = tab.loc[idx, ab].to_numpy()
    return d, chromosomal, A.shape[1] - len(chromosomal)


def centre_within(d, by, cols, min_n=MIN_ST):
    g = d.dropna(subset=[by]).copy()
    keep = g.groupby(by).size().loc[lambda s: s >= min_n].index
    g = g[g[by].isin(keep)]
    for c in cols:
        g[c] = g[c] - g.groupby(by)[c].transform("mean")
    return g, len(keep)


def correlations(d):
    rows = []
    for c in ASM:
        r, p = stats.spearmanr(d[c], d.n_acquired)
        rows.append(dict(scope="all strains", group="-", feature=c, n=len(d), rho=round(r, 3), p=p))
    for by, label in [("st", "within ST"), ("phylogroup", "within phylogroup")]:
        g, n_groups = centre_within(d, by, ASM + ["n_acquired"])
        for c in ASM:
            r, p = stats.spearmanr(g[c], g.n_acquired)
            rows.append(dict(scope=label, group=f"{n_groups} groups", feature=c, n=len(g), rho=round(r, 3), p=p))
    for pg, sub in d.dropna(subset=["phylogroup"]).groupby("phylogroup"):
        if len(sub) < 100:
            continue
        r, p = stats.spearmanr(sub.Size, sub.n_acquired)
        rows.append(dict(scope="one phylogroup", group=str(pg), feature="Size", n=len(sub), rho=round(r, 3), p=p))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--determinant-matrix", default="results/determinant_matrix/acquired.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "diagnostics")
    a = ap.parse_args()
    d, chromosomal, n_acq_cols = load(a.determinant_matrix)
    a.out.mkdir(parents=True, exist_ok=True)

    cor = correlations(d)
    cor.to_csv(a.out / "assembly_size_correlations.csv", index=False)
    print(
        f"strains {len(d)}; acquired-gene columns {n_acq_cols}; "
        f"treated as chromosomal (>{NEAR_UNIVERSAL:.0%} of strains): {chromosomal}"
    )
    print(
        f"acquired genes per strain: mean {d.n_acquired.mean():.2f}, median "
        f"{d.n_acquired.median():.0f}, max {d.n_acquired.max()}\n"
    )
    print(cor[cor.scope != "one phylogroup"].to_string(index=False))
    print("\nassembly size against acquired-gene count, inside each phylogroup:")
    print(cor[cor.scope == "one phylogroup"][["group", "n", "rho", "p"]].to_string(index=False))

    rows = []
    for ab in AB:
        for name, s in [
            ("assembly size", d.Size),
            ("CDS count", d.CDS),
            ("contig count", d.Contigs),
            ("acquired-gene count", d.n_acquired),
        ]:
            r = within_st_concordance(d[ab].to_numpy(), s.to_numpy(), d.st.to_numpy(object))
            rows.append(
                dict(
                    drug=ab,
                    mechanism=MECHANISM[ab],
                    feature=name,
                    within_st_auroc=round(r["macro"], 3),
                    n_st=r["n_st"],
                    n_pairs=r["n_pairs"],
                )
            )
    w = pd.DataFrame(rows)
    w.to_csv(a.out / "assembly_size_within_st.csv", index=False)
    print("\nwithin-ST concordance (0.5 = nothing inside a lineage):")
    print(w.pivot(index=["drug", "mechanism"], columns="feature", values="within_st_auroc").to_string())

    size = w[w.feature == "assembly size"].set_index("drug").within_st_auroc
    acq = w[w.feature == "acquired-gene count"].set_index("drug").within_st_auroc
    chrom = [k for k, v in MECHANISM.items() if v == "chromosomal"]
    acquired = [k for k, v in MECHANISM.items() if v == "acquired"]
    summary = {
        "strains": int(len(d)),
        "acquired_gene_columns": int(n_acq_cols),
        "chromosomal_genes_excluded": chromosomal,
        "size_vs_acquired_within_st": cor[(cor.scope == "within ST") & (cor.feature == "Size")]
        .iloc[0][["rho", "p", "n"]]
        .to_dict(),
        "mean_within_st_auroc": {
            "assembly size": round(float(size.mean()), 3),
            "acquired-gene count": round(float(acq.mean()), 3),
        },
        "mean_within_st_auroc_assembly_size": {
            "acquired-determinant drugs": round(float(size[acquired].mean()), 3),
            "chromosomal-determinant drugs": round(float(size[chrom].mean()), 3),
        },
        "min_within_st_auroc_assembly_size": round(float(size.min()), 3),
    }
    (a.out / "assembly_size_within_lineage.json").write_text(json.dumps(summary, indent=1, default=float))
    print("\n" + json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
