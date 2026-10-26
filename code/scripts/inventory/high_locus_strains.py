"""
high_locus_strains.py  (W0 addition)
====================================
Strains with unusually many loci in the rebuilt presence/absence matrix (KNOWN_ISSUES 12), and
what the metadata says about them. Label-free: no phenotype is read.

    python scripts/inventory/high_locus_strains.py

Outlier rule (pre-registered in docs/experiments/w0_high_locus_strains.md): loci per strain above
median + 4 * 1.4826 * MAD over the strains that have a presence/absence profile.

Writes results/data_inventory/high_locus_strains.csv (the flagged strains),
high_locus_by_study.csv and high_locus_summary.json.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.metadata import PROCESSED, RAW, load_strain_table  # noqa: E402

OUT = ROOT / "results" / "data_inventory"
Z_CUT = 4.0


def flag_high_locus(loci, matched, z_cut=Z_CUT):
    """Robust z-score of loci per strain; True where z > z_cut. Unmatched strains are never flagged."""
    loci = np.asarray(loci, float)
    ref = loci[matched]
    med = np.median(ref)
    scale = 1.4826 * np.median(np.abs(ref - med))
    z = np.where(matched, (loci - med) / scale, np.nan)
    return np.nan_to_num(z, nan=-np.inf) > z_cut, z, med, scale


def main():
    with open(PROCESSED / "arcadia_v2_gene_presence_absence.pkl", "rb") as f:
        pa = pickle.load(f)
    if not str(pa.get("encoding_rule", "")).startswith("present = (allele letter"):
        raise ValueError("use the rebuilt presence/absence matrix (outputs-pa_fix)")
    X = pa["X_pa"]
    matched = np.asarray(pa["matched_mask"]).astype(bool)
    ids = np.asarray(pa["gene_ids"]).astype(str)
    cds = ~np.char.startswith(ids, "Cluster_")
    loci = X.sum(axis=1)
    flag, z, med, scale = flag_high_locus(loci, matched)

    t = load_strain_table()
    md = pd.read_csv(RAW / "strains_metadata.csv", low_memory=False).set_index("Genome.Name").reindex(t.genome_name)
    d = pd.DataFrame({
        "strain_index": np.arange(len(t)), "sra": t.sra.values, "loci": loci,
        "cds_loci": X[:, cds].sum(axis=1), "intergenic_loci": X[:, ~cds].sum(axis=1), "robust_z": np.round(z, 2),
        "assembly_size": md["Size"].astype(float).values, "contigs": md["Contigs"].astype(float).values,
        "contig_n50": md["Contig.N50"].astype(float).values, "annotated_cds": md["CDS"].astype(float).values,
        "checkm_contamination": md["CheckM.Contamination"].astype(float).values,
        "study": t.study.values, "country": t.country.values, "year": t.year.values, "st": t.st.values,
        "phylogroup": t.phylogroup.values, "platform": md["Sequencing.Platform"].values,
    })
    d.loc[~matched, ["loci", "cds_loci", "intergenic_loci"]] = np.nan
    OUT.mkdir(parents=True, exist_ok=True)
    d[flag].sort_values("loci", ascending=False).to_csv(OUT / "high_locus_strains.csv", index=False)

    by = d[matched].assign(flagged=flag[matched]).groupby("study").agg(
        strains=("loci", "size"), mean_loci=("loci", "mean"), sd_loci=("loci", "std"),
        flagged=("flagged", "sum"), mean_assembly_mb=("assembly_size", lambda s: s.mean() / 1e6),
        median_contigs=("contigs", "median")).round(2).sort_values("strains", ascending=False)
    by.to_csv(OUT / "high_locus_by_study.csv")

    nor = d[(d.study == "unpublished:Norway") & matched]
    f_nor = d[flag & (d.study == "unpublished:Norway")]
    runs = sorted(f_nor.sra)
    ok = matched & d.annotated_cds.notna()
    rest, top = d[ok & ~flag], d[flag & ok]
    summary = {
        "strains_with_profile": int(matched.sum()), "strains_without_profile": int((~matched).sum()),
        "median_loci": float(med), "robust_sd_loci": round(float(scale), 1), "z_cut": Z_CUT,
        "loci_cut": round(float(med + Z_CUT * scale), 1), "n_flagged": int(flag.sum()),
        "flagged_by_study": d[flag].study.value_counts().to_dict(),
        "flagged_by_year": {str(int(k)): int(v) for k, v in d[flag].year.dropna().value_counts().sort_index().items()},
        "flagged_norway_sra_range": [runs[0], runs[-1]] if runs else None,
        "norway_flagged_in_ERR4034361_to_ERR4034705": int(f_nor.sra.between("ERR4034361", "ERR4034705").sum()),
        "norway_strains_in_that_sra_range": int(nor.sra.between("ERR4034361", "ERR4034705").sum()),
        "median_assembly_mb_flagged_vs_rest": [round(top.assembly_size.median() / 1e6, 3), round(rest.assembly_size.median() / 1e6, 3)],
        "median_annotated_cds_flagged_vs_rest": [float(top.annotated_cds.median()), float(rest.annotated_cds.median())],
        "median_contigs_flagged_vs_rest": [float(top.contigs.median()), float(rest.contigs.median())],
        "flagged_with_assembly_above_5p8mb": int((top.assembly_size > 5.8e6).sum()),
        "spearman_loci_vs_annotated_cds": round(float(d[ok][["loci", "annotated_cds"]].corr("spearman").iloc[0, 1]), 3),
    }
    (OUT / "high_locus_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(by.to_string())


if __name__ == "__main__":
    main()
