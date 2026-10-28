"""
Is there a second species in NCBI Pathogen Detection to test the findings on? (W13, counts only.)
Pre-registration and gate: docs/experiments/w13_second_species.md.

For every organism folder under pathogen/Results/ (E. coli skipped: W8 counted it), reads the
newest latest_snps/Metadata/*.metadata.tsv in chunks and counts, among isolates with
AST_phenotypes: totals, AMRFinderPlus 4.2.7 share and database versions, R/S/I/ND per drug, SNP
cluster coverage, assembly-statistics coverage, country and isolation type. Nothing is trained and
no phenotype is related to any feature.

    python -m src.mechanism.species_probe --out results/species_probe [--base URL] [--only A,B]

Writes per_species.json (one record per organism), per_drug.csv, gate.csv and, for every species
that passes the gate, candidates_<organism>.tsv.gz (the AST rows, the columns W13 needs).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

BASE = "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/"
SKIP = {"Escherichia_coli_Shigella"}
SOFTWARE = "4.2.7"
KEEP = ["target_acc", "biosample_acc", "bioproject_acc", "asm_acc", "PDS_acc", "scientific_name",
        "geo_loc_name", "collection_date", "epi_type", "isolation_source", "host",
        "asm_stats_length_bp", "asm_stats_n_contig", "asm_stats_contig_n50",
        "AST_phenotypes", "AMR_genotypes", "amrfinder_version", "refgene_db_version",
        "amrfinder_analysis_type", "computed_types"]
ASM = ["asm_stats_length_bp", "asm_stats_n_contig", "asm_stats_contig_n50"]
# the pre-registered gate
MIN_ISOLATES, MIN_DRUGS, MIN_EACH = 2000, 3, 100
MIN_CLUSTER_SHARE, MIN_CLUSTERS, MIN_ASM_SHARE = 0.80, 50, 0.80
MAX_SPECIES = 2


def listing(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return re.findall(r'href="([^"?/][^"]*)"', r.read().decode("utf-8", "replace"))


def organisms(base):
    return sorted(n.rstrip("/") for n in listing(base) if n.endswith("/") and n.rstrip("/") not in SKIP)


def newest_metadata(base, org):
    url = f"{base}{org}/latest_snps/Metadata/"
    try:
        files = sorted(n for n in listing(url) if n.endswith(".metadata.tsv"))
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code} at {url}"
    return (url + files[-1], None) if files else (None, f"no *.metadata.tsv at {url}")


def parse_ast(s):
    out = {}
    if isinstance(s, str):
        for tok in s.split(","):
            if "=" in tok:
                d, call = tok.rsplit("=", 1)
                out[d.strip()] = call.strip()
    return out


def count(path, chunksize=100_000):
    """One pass over a metadata file: the AST rows (KEEP columns) and the column list."""
    rows, n_total, columns = [], 0, None
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize, low_memory=False):
        columns = columns or list(chunk.columns)
        n_total += len(chunk)
        if "AST_phenotypes" not in chunk:
            continue
        c = chunk[chunk.AST_phenotypes.notna()]
        if len(c):
            rows.append(c[[k for k in KEEP if k in c.columns]])
    ast = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=KEEP)
    return ast, n_total, columns or []


def summarise(org, ast, n_total, columns):
    sw = ast[ast.get("amrfinder_version", pd.Series(dtype=str)).eq(SOFTWARE)] if len(ast) else ast
    calls = [parse_ast(s) for s in sw.AST_phenotypes] if len(sw) else []
    drugs = sorted({d for c in calls for d in c})
    per = []
    for d in drugs:
        v = pd.Series([c.get(d) for c in calls]).value_counts()
        per.append(dict(organism=org, drug=d, R=int(v.get("R", 0)), S=int(v.get("S", 0)),
                        I=int(v.get("I", 0)), ND=int(v.get("ND", 0))))
    per = pd.DataFrame(per, columns=["organism", "drug", "R", "S", "I", "ND"])
    pds = sw["PDS_acc"] if "PDS_acc" in sw else pd.Series([None] * len(sw), dtype=object)
    asm_ok = sw[ASM].notna().all(axis=1) if set(ASM) <= set(sw.columns) else pd.Series([False] * len(sw))
    country = sw.geo_loc_name.dropna().str.split(":").str[0] if "geo_loc_name" in sw else pd.Series(dtype=str)
    n = len(sw)
    rec = {
        "organism": org, "rows_in_table": n_total, "rows_with_AST": int(len(ast)),
        "ast_with_amrfinder_4_2_7": n,
        "db_versions": sw.refgene_db_version.value_counts().head(6).to_dict() if n else {},
        "cluster_share": round(float(pds.notna().mean()), 4) if n else 0.0,
        "clusters": int(pds.nunique()) if n else 0,
        "assembly_stats_share": round(float(asm_ok.mean()), 4) if n else 0.0,
        "countries": country.value_counts().head(10).to_dict(),
        "epi_type": sw.epi_type.value_counts(dropna=False).head(5).to_dict() if "epi_type" in sw else {},
        "drugs_with_100_each": sorted(per[(per.R >= MIN_EACH) & (per.S >= MIN_EACH)].drug),
        "metadata_columns": columns,
    }
    rec["gate"] = {
        "isolates": n >= MIN_ISOLATES,
        "drugs": len(rec["drugs_with_100_each"]) >= MIN_DRUGS,
        "clusters": rec["cluster_share"] >= MIN_CLUSTER_SHARE and rec["clusters"] >= MIN_CLUSTERS,
        "assembly_stats": rec["assembly_stats_share"] >= MIN_ASM_SHARE,
    }
    rec["passes"] = all(rec["gate"].values())
    return rec, per, sw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/species_probe"))
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--only", default="", help="comma-separated organism folders (default: all)")
    ap.add_argument("--tmp", type=Path, default=Path("/tmp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    orgs = [o for o in a.only.split(",") if o] or organisms(a.base)
    print(f"{len(orgs)} organism folders: {', '.join(orgs)}", flush=True)

    recs, pers, kept = [], [], {}
    for org in orgs:
        url, err = newest_metadata(a.base, org)
        if err:
            recs.append({"organism": org, "error": err, "passes": False})
            print(f"{org}: {err}", flush=True)
            continue
        path = a.tmp / Path(url).name
        t0 = time.time()
        try:
            with urllib.request.urlopen(url, timeout=1800) as r, open(path, "wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            ast, n_total, cols = count(path)
        except Exception as e:                       # one bad organism must not stop the census
            recs.append({"organism": org, "error": f"{type(e).__name__}: {e}", "passes": False})
            print(f"{org}: failed ({type(e).__name__}: {e})", flush=True)
            continue
        finally:
            if path.exists():
                path.unlink()
        rec, per, sw = summarise(org, ast, n_total, cols)
        rec["metadata_file"] = Path(url).name
        recs.append(rec)
        pers.append(per)
        kept[org] = sw
        print(f"{org}: {n_total} rows, {rec['rows_with_AST']} with AST, {rec['ast_with_amrfinder_4_2_7']} "
              f"on 4.2.7, clusters {rec['cluster_share']:.0%} ({rec['clusters']}), asm {rec['assembly_stats_share']:.0%}, "
              f"drugs>=100 each: {len(rec['drugs_with_100_each'])}, gate {'PASS' if rec['passes'] else 'fail'} "
              f"({time.time() - t0:.0f}s)", flush=True)

    passing = sorted([r for r in recs if r.get("passes")], key=lambda r: -r["ast_with_amrfinder_4_2_7"])
    chosen = [r["organism"] for r in passing[:MAX_SPECIES]]
    for org in chosen:
        kept[org].to_csv(a.out / f"candidates_{org}.tsv.gz", sep="\t", index=False)
    (a.out / "per_species.json").write_text(json.dumps(recs, indent=1, default=str))
    (pd.concat(pers, ignore_index=True) if pers else pd.DataFrame()).to_csv(a.out / "per_drug.csv", index=False)
    gate = pd.DataFrame([{"organism": r["organism"], "passes": r.get("passes", False),
                          "ast_isolates": r.get("ast_with_amrfinder_4_2_7"),
                          "drugs_with_100_each": len(r.get("drugs_with_100_each", [])),
                          "cluster_share": r.get("cluster_share"), "clusters": r.get("clusters"),
                          "assembly_stats_share": r.get("assembly_stats_share"),
                          "error": r.get("error"), "chosen": r["organism"] in chosen} for r in recs])
    gate.to_csv(a.out / "gate.csv", index=False)
    print(gate.to_string(index=False))
    print(json.dumps({"chosen": chosen, "rule": f"the {MAX_SPECIES} passing species with the most AST "
                      "isolates, by that count alone"}, indent=1))


if __name__ == "__main__":
    main()
