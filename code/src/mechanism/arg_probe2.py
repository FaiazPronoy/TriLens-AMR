"""
Second ARG-route probe, after arg_probe found BV-BRC's ftp host unreachable from Kaggle (https timed
out, ftp refused) and NCBI Pathogen Detection reachable.

  1. NCBI Pathogen Detection, E. coli/Shigella: download latest_snps/Metadata/*.metadata.tsv and
     latest_snps/AMR/*.amr.metadata.tsv (names read from the listing, nothing assumed), report their
     columns and size, and how many of our 6,962 strains they cover by SRA run or BioSample.
     For the joined rows: which AMRFinderPlus software / database versions were used (any column whose
     name mentions 'version'), and the label-free prevalence of a few gene families.
  2. BV-BRC through its API host (www.bv-brc.org) instead of the ftp host, for 3 genomes, with the
     full error text kept.

No phenotype is read: this decides the route only (docs/experiments/w6_arg_rescue.md).

    python -m src.mechanism.arg_probe2 --out results/arg_probe2

Writes probe2.json, the column lists, the joined rows of both tables (our strains only) as gzipped
TSV, and bvbrc_api.csv.
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
_INV = Path(os.environ.get("TRILENS_REPO_RESULTS") or ROOT / "results") / "data_inventory"
PD = "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Escherichia_coli_Shigella/latest_snps/"
BVBRC_API = ("https://www.bv-brc.org/api/genome_sequence/?eq(genome_id,{g})&limit(1)"
             "&http_accept=application/dna+fasta")
FAMILIES = {"CTX-M": r"blaCTX-M", "aac(3)": r"aac\(3\)", "dfrA": r"dfrA", "blaTEM": r"blaTEM",
            "sul1": r"\bsul1\b", "gyrA point": r"gyrA_"}


def get(url, timeout=120):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read(), ""
    except Exception as e:  # noqa: BLE001
        return getattr(e, "code", None) or type(e).__name__, b"", f"{type(e).__name__}: {e}"[:300]


def download(url, dest):
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=600) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    return round(dest.stat().st_size / 1e6, 1), round(time.time() - t0)


def listing(sub, pattern):
    st, body, err = get(PD + sub)
    names = re.findall(r'href="([^"?/][^"]*)"', body.decode("utf-8", "replace")) if body else []
    return [n for n in names if re.search(pattern, n)], st, err


def join_table(path, sra, biosample, out_tsv):
    """Rows of a PD table whose run or BioSample column matches one of our strains."""
    head = pd.read_csv(path, sep="\t", nrows=5, dtype=str)
    cols = list(head.columns)
    run_cols = [c for c in cols if re.fullmatch(r"(?i)(run|sra_run|sra|Run)", c)]
    bs_cols = [c for c in cols if re.search(r"(?i)biosample", c)]
    hits, n_rows = [], 0
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=200_000, low_memory=False):
        n_rows += len(chunk)
        m = pd.Series(False, index=chunk.index)
        for c in run_cols:
            m |= chunk[c].isin(sra)
        for c in bs_cols:
            m |= chunk[c].isin(biosample)
        if m.any():
            hits.append(chunk[m])
    joined = pd.concat(hits) if hits else pd.DataFrame(columns=cols)
    joined.to_csv(out_tsv, sep="\t", index=False, compression="gzip")
    return cols, run_cols, bs_cols, n_rows, joined


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/arg_probe2"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/arg_probe2")
    work.mkdir(exist_ok=True)
    ids = pd.read_csv(_INV / "genome_ids.csv", dtype=str)
    sra, biosample = set(ids.sra.dropna()), set(ids.biosample.dropna())
    summary = {"our_strains": len(ids), "our_sra": len(sra), "our_biosample": len(biosample)}

    for key, sub, pat in [("metadata", "Metadata/", r"\.metadata\.tsv$"), ("amr", "AMR/", r"\.amr\.metadata\.tsv$")]:
        files, st, err = listing(sub, pat)
        rec = {"listing_status": st, "listing_error": err, "files": files}
        if files:
            path = work / files[0]
            try:
                rec["mb"], rec["download_s"] = download(PD + sub + files[0], path)
                cols, run_cols, bs_cols, n_rows, j = join_table(path, sra, biosample, a.out / f"{key}_joined.tsv.gz")
                (a.out / f"{key}_columns.txt").write_text("\n".join(cols))
                matched_sra = set().union(*[set(j[c]) for c in run_cols]) & sra if run_cols else set()
                matched_bs = set().union(*[set(j[c]) for c in bs_cols]) & biosample if bs_cols else set()
                covered = ids[ids.sra.isin(matched_sra) | ids.biosample.isin(matched_bs)]
                rec.update(columns=cols, run_columns=run_cols, biosample_columns=bs_cols, rows_in_file=n_rows,
                           joined_rows=len(j), strains_covered=int(covered.strain_index.nunique()),
                           strains_covered_by_sra=len(matched_sra))
                vcols = [c for c in cols if re.search(r"(?i)version", c)]
                rec["version_columns"] = {c: j[c].value_counts().head(15).to_dict() for c in vcols}
                gcols = [c for c in cols if re.search(r"(?i)genotype|element|gene_symbol|symbol", c)]
                rec["genotype_columns"] = gcols
                if gcols and len(j):
                    text = j[gcols].fillna("").astype(str).agg(" ".join, axis=1)
                    rec["family_prevalence_in_joined_rows"] = {
                        k: round(float(text.str.contains(p, regex=True).mean()), 4) for k, p in FAMILIES.items()}
                rec["example_row"] = j.head(1).to_dict(orient="records")
                path.unlink()
            except Exception as e:  # noqa: BLE001
                rec["error"] = f"{type(e).__name__}: {e}"[:500]
        summary[f"pathogen_detection_{key}"] = rec

    rows = []
    for g in ids.genome_id.head(3):
        t0 = time.time()
        st, body, err = get(BVBRC_API.format(g=g), timeout=60)
        rows.append(dict(genome_id=g, status=st, bytes=len(body), seconds=round(time.time() - t0, 1),
                         starts_with=body[:40].decode("utf-8", "replace"), error=err))
    pd.DataFrame(rows).to_csv(a.out / "bvbrc_api.csv", index=False)
    summary["bvbrc_api"] = rows
    (a.out / "probe2.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
