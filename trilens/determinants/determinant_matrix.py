from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOFTWARE = "4.2.7"
PRIMARY_DB = "2026-03-24.1"
NOT_COUNTED = {"PARTIAL", "PARTIAL_END_OF_CONTIG", "MISTRANSLATION", "HMM"}
MIN_STRAINS = 5
KEEP = [
    "Run",
    "biosample_acc",
    "target_acc",
    "asm_acc",
    "target_creation_date",
    "AMR_genotypes",
    "amrfinder_version",
    "refgene_db_version",
    "amrfinder_analysis_type",
    "AST_phenotypes",
]


def inv() -> Path:
    return Path(os.environ.get("TRILENS_REPO_RESULTS") or ROOT / "results") / "data_inventory"


def frozen_path() -> Path:
    return inv() / "pd_amr_joined.tsv.gz"


def freeze(joined):
    ids = pd.read_csv(inv() / "genome_ids.csv", dtype=str)
    pd_rows = pd.read_csv(joined, sep="\t", dtype=str)[KEEP]
    by_run = ids[["strain_index", "sra"]].merge(pd_rows, left_on="sra", right_on="Run").assign(match="run")
    by_bs = (
        ids.dropna(subset=["biosample"])[["strain_index", "biosample"]]
        .merge(pd_rows, left_on="biosample", right_on="biosample_acc")
        .assign(match="biosample")
    )
    rows = pd.concat([by_run.drop(columns="sra"), by_bs.drop(columns="biosample")], ignore_index=True)
    rows["strain_index"] = rows.strain_index.astype(int)
    rows["_run_first"] = rows.match.ne("run")
    rows = rows.sort_values(
        ["strain_index", "_run_first", "target_creation_date"], ascending=[True, True, False]
    ).drop_duplicates(["strain_index", "target_acc"])
    n_multi = int(rows.strain_index.duplicated().sum())
    rows = rows.drop_duplicates("strain_index").drop(columns="_run_first")
    out = frozen_path()
    rows.to_csv(out, sep="\t", index=False, compression={"method": "gzip", "mtime": 0})
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {
        "source_file": Path(joined).name,
        "strains": len(rows),
        "strains_with_extra_rows": n_multi,
        "match": rows.match.value_counts().to_dict(),
        "sha256": sha,
    }
    (inv() / "pd_amr_joined.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, indent=1))


def tokens(s):
    return [t.strip() for t in str(s).split(",") if t.strip()] if isinstance(s, str) else []


def build(db, out):
    rows = pd.read_csv(frozen_path(), sep="\t", dtype=str)
    rows["strain_index"] = rows.strain_index.astype(int)
    meta = json.loads((inv() / "pd_amr_joined.json").read_text())
    sha = hashlib.sha256(frozen_path().read_bytes()).hexdigest()
    if sha != meta["sha256"]:
        raise SystemExit(f"{frozen_path()} changed since it was frozen (SHA-256 {sha} != {meta['sha256']})")
    use = rows[rows.amrfinder_version.eq(SOFTWARE)]
    if db != "all":
        use = use[use.refgene_db_version.eq(db)]
    acq, pt = [], []
    for i, s in zip(use.strain_index, use.AMR_genotypes):
        for t in tokens(s):
            sym, _, suf = t.partition("=")
            if suf == "POINT":
                pt.append((i, sym))
            elif suf == "":
                acq.append((i, sym))
            elif suf not in NOT_COUNTED:
                raise SystemExit(f"unknown AMR_genotypes suffix {suf!r} in {t!r}: decide a rule first")
    strains = sorted(use.strain_index)

    def matrix(pairs):
        df = pd.DataFrame(pairs, columns=["strain_index", "symbol"]).drop_duplicates()
        x = pd.crosstab(df.strain_index, df.symbol).clip(upper=1).reindex(strains, fill_value=0)
        x.index.name = "strain_index"
        return x.loc[:, x.sum() >= MIN_STRAINS].astype("uint8")

    A, P = matrix(acq), matrix(pt)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    A.to_csv(out / "acquired.csv")
    P.to_csv(out / "point.csv")

    t = pd.read_csv(inv() / "strain_table.csv", dtype={"st": str, "study": str})
    t["included"] = t.strain_index.isin(strains)
    desc = []
    for col in ("country", "year", "study", "st", "phylogroup"):
        v = t[col].astype(str).where(t[col].notna(), "missing")
        top = v.value_counts().head(12).index
        tab = pd.crosstab(v.where(v.isin(top), "other"), t.included, normalize="columns").round(4)
        tab.columns = ["excluded" if not c else "included" for c in tab.columns]
        desc.append(tab.reset_index(names="value").assign(variable=col))
    pd.concat(desc).to_csv(out / "excluded_vs_included.csv", index=False)

    cov = {
        "source": meta["source_file"],
        "sha256": sha,
        "software": SOFTWARE,
        "database": db,
        "strains_with_table": len(strains),
        "strains_total": len(t),
        "coverage": round(len(strains) / len(t), 4),
        "acquired_columns": A.shape[1],
        "point_columns": P.shape[1],
        "analysis_type": use.amrfinder_analysis_type.value_counts().to_dict(),
        "top_acquired": A.sum().sort_values(ascending=False).head(40).astype(int).to_dict(),
    }
    (out / "coverage.json").write_text(json.dumps(cov, indent=1))
    print(json.dumps(cov, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("--joined", required=True)
    b = sub.add_parser("build")
    b.add_argument("--db", default=PRIMARY_DB)
    b.add_argument("--out", default="results/determinant_matrix")
    a = ap.parse_args()
    freeze(a.joined) if a.cmd == "freeze" else build(a.db, a.out)


if __name__ == "__main__":
    main()
