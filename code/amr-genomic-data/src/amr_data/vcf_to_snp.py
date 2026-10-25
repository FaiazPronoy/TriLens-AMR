"""Phase A: VCF -> integer SNP matrix (Arcadia convention).

Ported from `arcadia-v2-phase-a.ipynb` cell 3. Logic is unchanged from the
notebook (missing genotype -> reference allele 0, SNP-only, MAF filter on
the *selected-sample* population) -- only the hardcoded Kaggle paths (B11)
were removed in favour of `config/paths.yaml`, and the whole thing was
wrapped in importable functions with type hints instead of living in one
notebook cell.

Verification manifest (docs manifest E1, reproduced directly from
`arcadia_v2_snp_matrix.pkl` during the audit, not just the notebook's own
printed log):

    variants_streamed == 2_451_184
    skipped_indel     ==    43_799
    skipped_low_maf   == 1_871_946
    snps_retained     ==   535_439
    X_snp.shape       == (6962, 535439), dtype int8
    value counts      == {0: 3_103_894_858, 1: 608_408_091,
                          2: 14_874_231,    3: 549_138}

This module does not re-run the VCF streaming (no VCF file is present in
this repository -- it lives only in the Kaggle/Zenodo source). It is
provided so the step is reproducible from raw data by whoever has that file,
and so `tests/test_vcf_to_snp.py` can validate the parsing/MAF logic in
isolation on synthetic data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import Paths, load_params


@dataclass
class VcfToSnpConfig:
    min_maf: float = 0.01
    snp_only: bool = True
    progress_interval: int = 100_000


@dataclass
class VcfToSnpResult:
    X_snp: np.ndarray  # (n_samples, n_snps) int8
    snp_positions: np.ndarray  # str, "gene_id:pos:REF>ALT"
    selected_genome_names: np.ndarray
    selected_sra_ids: np.ndarray
    n_streamed: int
    n_indel: int
    n_low_maf: int
    config: dict


def select_samples(
    vcf_samples: np.ndarray,
    sra_to_genome: dict[str, str],
    pheno_genomes: set[str],
) -> tuple[np.ndarray, list[str], list[str]]:
    """Which VCF sample columns have both an SRA->genome mapping and a
    phenotype row. Returns (selected_indices, genome_names, sra_ids)."""
    selected_indices: list[int] = []
    selected_genomes: list[str] = []
    selected_sra: list[str] = []
    for i, sra_id in enumerate(vcf_samples):
        gname = sra_to_genome.get(sra_id)
        if gname is not None and gname in pheno_genomes:
            selected_indices.append(i)
            selected_genomes.append(gname)
            selected_sra.append(sra_id)
    return np.array(selected_indices), selected_genomes, selected_sra


def _is_snp(ref: str, alts: list[str]) -> bool:
    return len(ref) == 1 and bool(alts) and all(len(a) == 1 for a in alts)


def stream_vcf_to_snp_matrix(
    vcf_path: Path,
    sra_map_path: Path,
    pheno_path: Path,
    cfg: VcfToSnpConfig | None = None,
    *,
    progress_cb=None,
) -> VcfToSnpResult:
    """Stream a VCF and build the Arcadia-convention integer SNP matrix.

    Requires `cyvcf2` (import is local to this function so the rest of the
    package can be imported/tested without it installed).
    """
    from cyvcf2 import VCF  # local import -- see docstring

    cfg = cfg or VcfToSnpConfig()

    sra_map = pd.read_csv(sra_map_path)
    pheno = pd.read_csv(pheno_path)

    vcf = VCF(str(vcf_path))
    vcf_samples = np.array(vcf.samples, dtype=str)
    sra_to_genome = dict(zip(sra_map["SRA.Accession"], sra_map["Genome.Name"]))
    pheno_genomes = set(pheno["Genome.Name"])

    selected_indices, selected_genomes, selected_sra = select_samples(
        vcf_samples, sra_to_genome, pheno_genomes
    )
    n_sel = len(selected_indices)
    min_maf_abs = max(1.0 / n_sel, cfg.min_maf)

    snp_columns: list[np.ndarray] = []
    snp_positions: list[str] = []
    n_streamed = n_indel = n_low_maf = 0
    t_start = datetime.now()

    for v in vcf:
        n_streamed += 1
        if progress_cb and n_streamed % cfg.progress_interval == 0:
            progress_cb(n_streamed, len(snp_columns), (datetime.now() - t_start).total_seconds())

        ref = (v.REF or "").upper()
        alts = [(a or "").upper() for a in (v.ALT or [])]
        if cfg.snp_only and not _is_snp(ref, alts):
            n_indel += 1
            continue

        genos = v.genotypes
        g_sel = [genos[i] for i in selected_indices]
        col = np.zeros(n_sel, dtype=np.int8)
        for i, g in enumerate(g_sel):
            if g is None:
                continue
            a = g[0]
            if a is None or a < 0:
                continue
            col[i] = a

        n_ref = int((col == 0).sum())
        n_alt = int((col > 0).sum())
        maf = min(n_ref, n_alt) / n_sel
        if maf < min_maf_abs:
            n_low_maf += 1
            continue

        snp_columns.append(col)
        snp_positions.append(f"{v.CHROM}:{v.POS}:{ref}>{','.join(alts)}")

    vcf.close()

    X_snp = np.array(snp_columns, dtype=np.int8).T
    return VcfToSnpResult(
        X_snp=X_snp,
        snp_positions=np.array(snp_positions),
        selected_genome_names=np.array(selected_genomes),
        selected_sra_ids=np.array(selected_sra),
        n_streamed=n_streamed,
        n_indel=n_indel,
        n_low_maf=n_low_maf,
        config={
            "MIN_MAF": min_maf_abs,
            "SNP_ONLY": cfg.snp_only,
            "missing_handling": "Arcadia convention: missing -> ref (0)",
        },
    )
