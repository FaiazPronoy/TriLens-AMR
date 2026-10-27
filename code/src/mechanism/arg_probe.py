"""
Probe for the per-strain ARG route (docs/novelty.md, "ARG rescue"). Answers, on 20 strains, the
questions every external review left open before anyone commits to 6,962 downloads:

  1. Does BV-BRC serve each strain's own assembly (<Genome.ID>.fna) and specialty-gene table?
     Tried over https and ftp; status, bytes and seconds are logged per file.
  2. Does NCBI Pathogen Detection publish an AMR-genotype table we can join on SRA accession?
     The directory listings are saved as they are; nothing about file names is assumed.
  3. How long does AMRFinderPlus take per assembly, and do its calls agree with the phenotypes
     (CTX-M or another ESBL vs CTX/CAZ resistance, aac(3)/ant(2'') vs GEN, dfrA vs SXT)?

    python -m src.mechanism.arg_probe --out results/arg_probe [--n 20] [--no-amrfinder]

The 20 strains are drawn with a fixed seed, half from CTX-resistant strains, so the ESBL check has
cases to look at. Nothing here is used for training. Writes probe.json, files.csv, calls.csv,
concordance.csv and the raw listings.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.metadata import AB_NAMES
from src.mechanism.amrfinder import parse_hits

ROOT = Path(__file__).resolve().parents[2]
# committed tables: in a Kaggle notebook they are in the cloned repository, not in WORK/results
_INV = Path(os.environ.get("TRILENS_REPO_RESULTS") or ROOT / "results") / "data_inventory"
MANIFEST = _INV / "genome_ids.csv"
LABELS = _INV / "strain_table.csv"
BVBRC = ["https://ftp.bvbrc.org/genomes/{g}/{g}{ext}", "ftp://ftp.bvbrc.org/genomes/{g}/{g}{ext}"]
EXTS = [".fna", ".PATRIC.spgene.tab"]
NCBI_PD = "https://ftp.ncbi.nlm.nih.gov/pathogen/Results/Escherichia_coli_Shigella/"
# Which determinant families should go with which phenotype (rule check only, not a model).
RULES = {
    "CTX": r"blaCTX-M|blaSHV-(?!1\b|11\b)|blaTEM-(?!1\b)|blaCMY|blaDHA|blaOXA-48|blaNDM|blaKPC|blaVIM",
    "CAZ": r"blaCTX-M|blaSHV-(?!1\b|11\b)|blaTEM-(?!1\b)|blaCMY|blaDHA|blaNDM|blaKPC|blaVIM",
    "GEN": r"aac\(3\)|ant\(2''\)|armA|rmt",
    "SXT": r"dfrA|dfrB",
}


def fetch(url, dest=None, timeout=120):
    """(status, bytes, seconds). Status is the HTTP code, 'ok' for ftp, or the error's class name."""
    t0 = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        if dest is not None:
            Path(dest).write_bytes(data)
        return getattr(r, "status", "ok"), len(data), round(time.time() - t0, 1), data
    except Exception as e:  # noqa: BLE001 - every failure is a result here
        code = getattr(e, "code", None)
        return (code or type(e).__name__), 0, round(time.time() - t0, 1), None


def pick(n, seed=0):
    ids = pd.read_csv(MANIFEST, dtype=str)
    lab = pd.read_csv(LABELS)
    df = ids.join(lab[AB_NAMES])
    rng = np.random.default_rng(seed)
    res = df.index[df.CTX == 1].to_numpy()
    rest = df.index[df.CTX != 1].to_numpy()
    take = np.r_[rng.choice(res, n // 2, replace=False), rng.choice(rest, n - n // 2, replace=False)]
    return df.loc[np.sort(take)]


def probe_ncbi(out):
    """Save the listings top-down; report any file whose name mentions amr or metadata."""
    found, seen = [], {}
    for sub in ["", "latest_snps/", "latest_snps/Metadata/", "latest_snps/AMR/"]:
        st, nb, sec, data = fetch(NCBI_PD + sub, timeout=60)
        seen[sub or "/"] = st
        if data:
            txt = data.decode("utf-8", "replace")
            (out / f"ncbi_listing_{(sub or 'root').strip('/').replace('/', '_') or 'root'}.html").write_text(txt)
            found += [sub + f for f in re.findall(r'href="([^"?/][^"]*)"', txt)
                      if re.search(r"amr|metadata", f, re.I)]
    return {"listing_status": seen, "candidate_files": sorted(set(found))[:50]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("results/arg_probe"))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--no-amrfinder", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    work = Path("/tmp/arg_probe_fna")
    work.mkdir(exist_ok=True)
    strains = pick(a.n)
    have_amr = shutil.which("amrfinder") is not None and not a.no_amrfinder

    rows, calls = [], []
    for _, s in strains.iterrows():
        g = s.genome_id
        for ext in EXTS:
            for tmpl in BVBRC:
                url = tmpl.format(g=g, ext=ext)
                dest = work / f"{g}{ext}"
                st, nb, sec, _ = fetch(url, dest)
                rows.append(dict(strain_index=int(s.strain_index), genome_id=g, file=ext,
                                 scheme=url.split(":")[0], status=st, bytes=nb, seconds=sec))
                if nb:
                    break
        fna = work / f"{g}.fna"
        if have_amr and fna.exists() and fna.stat().st_size > 0:
            tsv = work / f"{g}.amr.tsv"
            t0 = time.time()
            subprocess.run(["amrfinder", "-n", str(fna), "--organism", "Escherichia", "--plus",
                            "--threads", str(a.threads), "-o", str(tsv)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            secs = round(time.time() - t0, 1)
            h = parse_hits(tsv)
            amr = h[h["type"].str.upper() == "AMR"]
            calls.append(dict(strain_index=int(s.strain_index), genome_id=g, amrfinder_seconds=secs,
                              symbols=";".join(sorted(amr.symbol.astype(str).unique()))))
            spg = work / f"{g}.PATRIC.spgene.tab"
            if spg.exists() and spg.stat().st_size:
                shutil.copy(spg, a.out / spg.name)
        fna.unlink(missing_ok=True)

    files = pd.DataFrame(rows)
    files.to_csv(a.out / "files.csv", index=False)
    calls = pd.DataFrame(calls, columns=["strain_index", "genome_id", "amrfinder_seconds", "symbols"])
    calls.to_csv(a.out / "calls.csv", index=False)

    conc = []
    if len(calls):
        lab = pd.read_csv(LABELS).loc[calls.strain_index.astype(int)]
        for ab, pat in RULES.items():
            has = calls.symbols.fillna("").str.contains(pat, regex=True).to_numpy()
            y = lab[ab].to_numpy()
            ok = ~np.isnan(y)
            tab = pd.crosstab(pd.Series(has[ok], name="determinant"), pd.Series(y[ok], name="resistant"))
            conc.append(dict(drug=ab, n=int(ok.sum()), table=tab.to_dict()))
    pd.DataFrame(conc).to_json(a.out / "concordance.json", orient="records", indent=1)

    fna_ok = files[(files.file == ".fna") & (files.bytes > 0)]
    summary = {
        "strains": len(strains),
        "fna_found": int(fna_ok.genome_id.nunique()),
        "fna_scheme_used": fna_ok.scheme.value_counts().to_dict(),
        "fna_mb_mean": round(float(fna_ok.bytes.mean() / 1e6), 2) if len(fna_ok) else None,
        "fna_seconds_mean": round(float(fna_ok.seconds.mean()), 1) if len(fna_ok) else None,
        "spgene_found": int(files[(files.file == ".PATRIC.spgene.tab") & (files.bytes > 0)].genome_id.nunique()),
        "failures": {f"{k[0]} {k[1]}": v for k, v in files[files.bytes == 0].groupby(["file", "scheme"]).status
                     .agg(lambda s: s.astype(str).value_counts().to_dict()).items()},
        "amrfinder_ran": have_amr,
        "amrfinder_seconds_mean": round(float(calls.amrfinder_seconds.mean()), 1) if len(calls) else None,
        "ncbi_pathogen_detection": probe_ncbi(a.out),
    }
    if have_amr:
        summary["amrfinder_version"] = subprocess.run(["amrfinder", "--version"], capture_output=True,
                                                      text=True).stdout.strip()
        summary["amrfinder_database"] = subprocess.run(["amrfinder", "--database_version"],
                                                       capture_output=True, text=True).stdout.strip()
    (a.out / "probe.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
