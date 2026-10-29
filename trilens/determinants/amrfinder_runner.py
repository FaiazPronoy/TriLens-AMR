from __future__ import annotations

import argparse
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from trilens.data.strain_metadata import AB_NAMES, PROCESSED

MIN_IDENTITY, MIN_COVERAGE = 90.0, 80.0
INTRINSIC_PREVALENCE = 0.95


COLUMNS = {
    "Protein id": "protein_id",
    "Protein identifier": "protein_id",
    "Contig id": "contig_id",
    "Element symbol": "symbol",
    "Gene symbol": "symbol",
    "Element name": "name",
    "Sequence name": "name",
    "Type": "type",
    "Element type": "type",
    "Subtype": "subtype",
    "Element subtype": "subtype",
    "Class": "class",
    "Subclass": "subclass",
    "Method": "method",
    "% Coverage of reference": "coverage",
    "% Coverage of reference sequence": "coverage",
    "% Identity to reference": "identity",
    "% Identity to reference sequence": "identity",
}


def parse_hits(path):
    raw = pd.read_csv(path, sep="\t", dtype=str)
    d = raw.rename(columns={c: COLUMNS[c] for c in raw.columns if c in COLUMNS})
    missing = {"symbol", "class", "subclass", "type"} - set(d.columns)
    if missing:
        raise ValueError(f"{path}: not an AMRFinderPlus table (missing {sorted(missing)})")
    prot = d.get("protein_id", pd.Series(np.nan, index=d.index)).replace("NA", np.nan)
    d["locus"] = prot.fillna(d.get("contig_id", pd.Series(np.nan, index=d.index)))
    for c in ("coverage", "identity"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["is_point"] = d.get("subtype", "").astype(str).str.upper().eq("POINT")
    return d


def drugs_for(cls, subclass):
    cls, sub = str(cls).upper(), str(subclass).upper()
    out = []
    if cls == "BETA-LACTAM":
        out += ["AMP", "AMC"]
        if "CEPHALOSPORIN" in sub:
            out += ["CTX", "CAZ"]
    if cls == "QUINOLONE":
        out.append("CIP")
    if cls == "AMINOGLYCOSIDE" and "GENTAMICIN" in sub:
        out.append("GEN")
    if cls in ("TRIMETHOPRIM", "SULFONAMIDE"):
        out.append("SXT")
    return out


def determinants(hits, pa_prevalence=None):
    rows = []
    amr = hits[hits["type"].astype(str).str.upper() == "AMR"]
    for h in amr.to_dict("records"):
        for drug in drugs_for(h["class"], h["subclass"]):
            prev = None if pa_prevalence is None else pa_prevalence.get(h["locus"])
            weak = (h["identity"] < MIN_IDENTITY) or (h["coverage"] < MIN_COVERAGE)
            intrinsic = (not h["is_point"]) and prev is not None and prev >= INTRINSIC_PREVALENCE
            rows.append(
                dict(
                    locus=h["locus"],
                    drug=drug,
                    symbol=h["symbol"],
                    amr_class=h["class"],
                    subclass=h["subclass"],
                    point_mutation=bool(h["is_point"]),
                    identity=h["identity"],
                    coverage=h["coverage"],
                    pa_prevalence=prev,
                    below_cutoff=bool(weak),
                    intrinsic=bool(intrinsic),
                    counted=not weak and not intrinsic,
                    in_pa_matrix=prev is not None,
                )
            )
    cols = [
        "locus",
        "drug",
        "symbol",
        "amr_class",
        "subclass",
        "point_mutation",
        "identity",
        "coverage",
        "pa_prevalence",
        "below_cutoff",
        "intrinsic",
        "counted",
        "in_pa_matrix",
    ]
    return pd.DataFrame(rows, columns=cols)


def pa_prevalence():
    with open(PROCESSED / "arcadia_v2_gene_presence_absence.pkl", "rb") as f:
        pa = pickle.load(f)
    if not str(pa.get("encoding_rule", "")).startswith("present = (allele letter"):
        raise ValueError("use the rebuilt presence/absence matrix (outputs-gene_presence_rebuild)")
    keep = np.asarray(pa["matched_mask"]).astype(bool)
    prev = np.asarray(pa["X_pa"])[keep].mean(axis=0)
    return dict(zip(map(str, pa["gene_ids"]), prev.astype(float)))


def run_amrfinder(fasta, out_tsv, threads=4):
    cmd = [
        "amrfinder",
        "-n",
        str(fasta),
        "--organism",
        "Escherichia",
        "--plus",
        "--threads",
        str(threads),
        "-o",
        str(out_tsv),
    ]
    subprocess.run(cmd, check=True)
    ver = subprocess.run(["amrfinder", "--version"], capture_output=True, text=True).stdout.strip()
    db = subprocess.run(["amrfinder", "--database_version"], capture_output=True, text=True).stdout.strip()
    return {"command": " ".join(cmd), "software": ver, "database": db}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", type=Path, default=Path("data/raw/pangenome/pangenome_cds.fa"))
    ap.add_argument("--out", type=Path, default=Path("results/amrfinder_calls"))
    ap.add_argument("--hits", type=Path, help="existing AMRFinderPlus table (skips the run)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if not a.hits and not a.fasta.exists():
        raise FileNotFoundError(
            f"{a.fasta} missing: add raw/pangenome/pangenome_cds.fa (small_data.zip) "
            "to a dataset attached to this notebook"
        )
    a.out.mkdir(parents=True, exist_ok=True)
    tsv = a.out / "amrfinder_hits.tsv"
    if a.hits:
        info = {"hits_from": str(a.hits)}
        tsv.write_text(Path(a.hits).read_text())
    else:
        info = run_amrfinder(a.fasta, tsv, a.threads)
    hits = parse_hits(tsv)
    det = determinants(hits, pa_prevalence())
    det.to_csv(a.out / "determinants.csv", index=False)
    summary = {
        **info,
        "hits": len(hits),
        "amr_hits": int((hits["type"].str.upper() == "AMR").sum()),
        "ctx_m_hits": sorted(hits.loc[hits.symbol.str.contains("CTX-M", case=False, na=False), "locus"].unique()),
        "counted_per_drug": {d: int(det[(det.drug == d) & det.counted].locus.nunique()) for d in AB_NAMES},
        "loci_not_in_pa_matrix": int((~det.in_pa_matrix).sum()),
    }
    (a.out / "versions.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
