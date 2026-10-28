"""
Is there an external validation cohort in NCBI Pathogen Detection? (W8, feasibility only.)

Our dataset is 91% UK and Norway, which is the objection every external review raised. Pathogen
Detection's E. coli table carries, for each public isolate, AMRFinderPlus calls (`AMR_genotypes`)
and, for some, measured susceptibilities (`AST_phenotypes`). On our own strains only 4.3% have the
AST column, but ours are research collections; the table has 592,113 rows, mostly surveillance.

This counts, over the whole table, the isolates that
  - are NOT ours (SRA run and BioSample both unseen), and
  - have AST_phenotypes, and
  - were called with AMRFinderPlus 4.2.7,
then reports, per drug of ours, how many are resistant and susceptible, and the country, source and
year spread. Nothing is trained and no model is touched: this decides whether W8 is possible.

    python -m src.mechanism.ext_probe --out results/ext_probe

Writes ext_probe.json, per_drug.csv, and candidates.tsv.gz (the matching rows, ~1 column set).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PD = "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Escherichia_coli_Shigella/latest_snps/Metadata/"
# our drug -> the name Pathogen Detection uses in AST_phenotypes
DRUGS = {"AMP": "ampicillin", "CIP": "ciprofloxacin", "AMC": "amoxicillin-clavulanic acid",
         "SXT": "trimethoprim-sulfamethoxazole", "GEN": "gentamicin", "CTX": "cefotaxime",
         "CAZ": "ceftazidime", "CRO": "ceftriaxone"}
KEEP = ["Run", "biosample_acc", "bioproject_acc", "bioproject_center", "target_acc", "asm_acc", "geo_loc_name", "collection_date", "epi_type",
         "isolation_source", "host", "asm_stats_length_bp", "asm_stats_n_contig", "asm_stats_contig_n50",
         "AST_phenotypes", "AMR_genotypes", "amrfinder_version", "refgene_db_version",
         "amrfinder_analysis_type", "number_drugs_tested"]
SOFTWARE = "4.2.7"


def inv() -> Path:
    r = os.environ.get("TRILENS_REPO_RESULTS")
    return (Path(r) if r else ROOT / "results") / "data_inventory"


def newest_metadata_file():
    with urllib.request.urlopen(PD, timeout=120) as r:
        names = re.findall(r'href="([^"?/][^"]*)"', r.read().decode("utf-8", "replace"))
    files = sorted(n for n in names if n.endswith(".metadata.tsv"))
    if not files:
        raise SystemExit(f"no *.metadata.tsv in {PD}")
    return files[-1]


def parse_ast(s):
    """{'ampicillin': 'R', ...} from 'ampicillin=R,cefepime=S,...'; ND and I are kept as given."""
    out = {}
    if isinstance(s, str):
        for tok in s.split(","):
            if "=" in tok:
                d, call = tok.rsplit("=", 1)
                out[d.strip()] = call.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/ext_probe"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    ids = pd.read_csv(inv() / "genome_ids.csv", dtype=str)
    ours_run, ours_bs = set(ids.sra.dropna()), set(ids.biosample.dropna())

    name = newest_metadata_file()
    path = Path("/tmp") / name
    t0 = time.time()
    with urllib.request.urlopen(PD + name, timeout=900) as r, open(path, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    print(f"{name}: {path.stat().st_size / 1e6:.0f} MB in {time.time() - t0:.0f}s", flush=True)

    rows, n_total, n_ast = [], 0, 0
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=100_000, low_memory=False):
        n_total += len(chunk)
        c = chunk[chunk.AST_phenotypes.notna()]
        n_ast += len(c)
        c = c[~c.Run.isin(ours_run) & ~c.biosample_acc.isin(ours_bs)]
        c = c[c.amrfinder_version.eq(SOFTWARE)]
        if len(c):
            rows.append(c[[k for k in KEEP if k in c.columns]])
    path.unlink()
    cand = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEEP)
    cand.to_csv(a.out / "candidates.tsv.gz", sep="\t", index=False)

    per = []
    for ab, pdname in DRUGS.items():
        calls = cand.AST_phenotypes.map(lambda s, n=pdname: parse_ast(s).get(n))
        v = calls.value_counts()
        per.append(dict(drug=ab, pd_name=pdname, R=int(v.get("R", 0)), S=int(v.get("S", 0)),
                        I=int(v.get("I", 0)), ND=int(v.get("ND", 0)), tested=int(calls.notna().sum())))
    per = pd.DataFrame(per)
    per.to_csv(a.out / "per_drug.csv", index=False)

    country = cand.geo_loc_name.dropna().str.split(":").str[0]
    year = pd.to_numeric(cand.collection_date.dropna().str[:4], errors="coerce")
    summary = {
        "metadata_file": name, "rows_in_table": n_total, "rows_with_AST": n_ast,
        "candidates": len(cand),
        "note": "candidates = has AST, not one of our strains, AMRFinderPlus 4.2.7",
        "db_versions": cand.refgene_db_version.value_counts().head(8).to_dict(),
        "analysis_type": cand.amrfinder_analysis_type.value_counts().to_dict(),
        "countries": country.value_counts().head(15).to_dict(),
        "epi_type": cand.epi_type.value_counts(dropna=False).head(8).to_dict(),
        "isolation_source": cand.isolation_source.value_counts().head(12).to_dict(),
        "host": cand.host.value_counts().head(8).to_dict(),
        "year_quartiles": ({k: float(v) for k, v in year.describe().items()} if len(year) else {}),
        "per_drug": per.to_dict(orient="records"),
        "drugs_with_at_least_100_each": sorted(per[(per.R >= 100) & (per.S >= 100)].drug),
    }
    (a.out / "ext_probe.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
