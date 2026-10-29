from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MIN_MAF = 0.01
PROGRESS_INTERVAL = 100000


def select_samples(vcf_samples, sra_map, pheno):
    sra_to_genome = dict(zip(sra_map["SRA.Accession"], sra_map["Genome.Name"]))
    pheno_genomes = set(pheno["Genome.Name"])
    indices, genomes, sra_ids = [], [], []
    for i, sra_id in enumerate(vcf_samples):
        genome = sra_to_genome.get(sra_id)
        if genome is not None and genome in pheno_genomes:
            indices.append(i)
            genomes.append(genome)
            sra_ids.append(sra_id)
    return np.array(indices), genomes, sra_ids


def genotype_column(genotypes, indices):
    col = np.zeros(len(indices), dtype=np.int8)
    for k, i in enumerate(indices):
        g = genotypes[i]
        if g is None:
            continue
        allele = g[0]
        if allele is None or allele < 0:
            continue
        col[k] = allele
    return col


def build(vcf_path, sra_map_path, pheno_path, out_path):
    from cyvcf2 import VCF

    sra_map = pd.read_csv(sra_map_path)
    pheno = pd.read_csv(pheno_path)
    vcf = VCF(str(vcf_path))
    vcf_samples = np.array(vcf.samples, dtype=str)
    indices, genomes, sra_ids = select_samples(vcf_samples, sra_map, pheno)
    n = len(indices)
    min_maf = max(1.0 / n, MIN_MAF)
    print(f"samples kept: {n}, minor allele frequency cut: {min_maf:.4f}")

    columns, positions = [], []
    n_streamed = n_indel = n_low_maf = 0
    start = datetime.now()
    for v in vcf:
        n_streamed += 1
        if n_streamed % PROGRESS_INTERVAL == 0:
            elapsed = (datetime.now() - start).total_seconds()
            print(f"  streamed {n_streamed:,}, kept {len(columns):,}, {elapsed:.0f} s", flush=True)
        ref = (v.REF or "").upper()
        alts = [(a or "").upper() for a in (v.ALT or [])]
        if len(ref) != 1 or not alts or any(len(a) != 1 for a in alts):
            n_indel += 1
            continue
        col = genotype_column(v.genotypes, indices)
        n_ref = int((col == 0).sum())
        n_alt = int((col > 0).sum())
        if min(n_ref, n_alt) / n < min_maf:
            n_low_maf += 1
            continue
        columns.append(col)
        positions.append(f"{v.CHROM}:{v.POS}:{ref}>{','.join(alts)}")
    vcf.close()

    X_snp = np.array(columns, dtype=np.int8).T
    print(
        f"streamed {n_streamed:,}; removed {n_indel:,} indels and {n_low_maf:,} rare variants; kept {X_snp.shape[1]:,}"
    )
    output = {
        "X_snp": X_snp,
        "snp_positions": np.array(positions),
        "selected_genome_names": np.array(genomes),
        "selected_sra_ids": np.array(sra_ids),
        "n_streamed": n_streamed,
        "n_kept": len(columns),
        "config": {
            "MIN_MAF": min_maf,
            "SNP_ONLY": True,
            "missing_handling": "missing genotype calls are set to the reference allele",
        },
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(output, f, protocol=4)
    print(f"saved {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")


def main():
    ap = argparse.ArgumentParser(description="Stream the Arcadia variant file into an integer SNP matrix.")
    ap.add_argument("--vcf", required=True, help="filtered_output.vcf.gz from the Arcadia release")
    ap.add_argument("--sra-map", required=True, help="phenotype_matrix/SRA_to_genome_name.csv")
    ap.add_argument("--phenotypes", required=True, help="phenotype_matrix/phenotype_matrix_08302024.csv")
    ap.add_argument("--out", default="data/processed/arcadia_v2_snp_matrix.pkl")
    a = ap.parse_args()
    build(a.vcf, a.sra_map, a.phenotypes, a.out)


if __name__ == "__main__":
    main()
