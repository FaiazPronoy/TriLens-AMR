"""
repair_gene_pa.py
=================
Rebuilds the gene presence/absence matrix (`arcadia_v2_gene_presence_absence.pkl`)
from Arcadia's PLINK files with the correct orientation.

Why: the October 2026 data inventory found that the stored matrix marks the minority
state of each pan-genome locus, not its presence. gyrA, parC, rpoB and recA are in all 72
ECOR genomes but "present" in none of our 6,962 strains, and no column is "present" in
more than 50.3% of strains. The file was built with `present = (PLINK value == 0)`, but
PLINK chose the minor state as allele A1 for each locus, so the meaning of 0 flips from
locus to locus. The allele letters in the .bim file say which state is which.

What this script does
  1. reads the .bed/.bim/.fam directly (no pandas-plink, so the bit coding is explicit:
     00 = homozygous A1, 01 = missing, 10 = heterozygous, 11 = homozygous A2);
  2. for every locus, turns the call into the allele *letter* the strain carries;
  3. finds the letter that means "present": the one carried by the loci present in all 72
     ECOR genomes (`reference/ecor72_frequency.csv`);
  4. builds present = (letter == that letter), aligned to the old file's strain order;
  5. checks that, at every locus, the new column is either identical to the old one or its
     exact complement on all called cells (so nothing but the orientation changes), that
     loci in all ECOR genomes are present in most strains, and that prevalence agrees with
     ECOR frequency.

    python scripts/repair_gene_pa.py --source /tmp/pa_source/paths.json \
        --old data/processed/arcadia_v2_gene_presence_absence.pkl \
        --out results/pa_fix/arcadia_v2_gene_presence_absence.pkl --report results/pa_fix/pa_repair.json

Exit codes: 0 repaired and written, 1 refused (nothing written).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ECOR = HERE / "reference" / "ecor72_frequency.csv"

# thresholds of the checks (written before the first run)
CORE_MIN_MEDIAN = 0.90      # loci in all 72 ECOR genomes: median prevalence in our strains
OTHER_LETTER_MAX = 0.10     # ... and below this under the other letter
MIN_ECOR_CORR = 0.50        # Spearman correlation of prevalence with ECOR frequency


def read_bim(path):
    bim = pd.read_csv(path, sep=r"\s+", header=None, dtype=str,
                      names=["chrom", "id", "cm", "pos", "a1", "a2"])
    return bim


def read_fam(path):
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=str).iloc[:, 1].tolist()


def read_bed(path, n_samples, n_variants):
    """Return (n_variants, n_samples) uint8 codes: 0 hom A1, 1 missing, 2 het, 3 hom A2."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw[:3].tolist() != [0x6C, 0x1B, 0x01]:
        raise ValueError("not a SNP-major PLINK .bed file")
    per = (n_samples + 3) // 4
    body = raw[3:]
    if body.size != per * n_variants:
        raise ValueError(f".bed size {body.size} != {per} x {n_variants}")
    body = body.reshape(n_variants, per)
    codes = np.empty((n_variants, per * 4), dtype=np.uint8)
    for k in range(4):
        codes[:, k::4] = (body >> (2 * k)) & 0b11
    # the 2-bit value is already the code: 0b00 hom A1, 0b01 missing, 0b10 het, 0b11 hom A2
    return codes[:, :n_samples]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", help="presence_absence_all.bed (.bim/.fam next to it)")
    ap.add_argument("--source", help="paths.json written by fetch_gene_pa_source.py (sets --bed and --sra)")
    ap.add_argument("--old", required=True)
    ap.add_argument("--sra", help="SRA_to_genome_name.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--ecor", default=str(ECOR))
    a = ap.parse_args()
    if a.source:
        src = json.loads(Path(a.source).read_text())
        a.bed, a.sra = a.bed or src["bed"], a.sra or src["sra"]
    if not (a.bed and a.sra):
        ap.error("give --source, or --bed and --sra")
    bed = Path(a.bed)
    rep: dict = {"bed": str(bed), "source": json.loads(Path(a.source).read_text()) if a.source else None}

    def done(code, status):
        rep["status"] = status
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps(rep, indent=1, default=str))
        print(json.dumps(rep, indent=1, default=str))
        return code

    bim = read_bim(bed.with_suffix(".bim"))
    fam = read_fam(bed.with_suffix(".fam"))
    # PLINK writes allele "0" for the missing second allele of a locus where every strain is
    # in the same state (e.g. a core gene), so "0" is not a state of its own.
    letters = sorted((set(bim.a1) | set(bim.a2)) - {"0"})
    rep.update(n_variants=len(bim), n_fam=len(fam), allele_letters=letters,
               allele_order_counts={f"{k[0]}/{k[1]}": int(v) for k, v in bim.groupby(["a1", "a2"]).size().items()})
    rep["monomorphic_loci"] = int(((bim.a1 == "0") | (bim.a2 == "0")).sum())
    if len(letters) != 2:
        return done(1, f"refused: expected two allele letters besides 0, found {letters}")

    with open(a.old, "rb") as f:
        old = pickle.load(f)
    X_old = np.asarray(old["X_pa"], dtype=np.uint8)
    gene_ids = [str(g) for g in old["gene_ids"]]
    if bim.chrom.tolist() != gene_ids:
        return done(1, "refused: .bim loci are not the old file's gene_ids in the same order")

    # strain alignment exactly as the original build (genome name -> SRA -> .fam row)
    sra = pd.read_csv(a.sra)
    gname_to_sra = {str(g): str(s) for s, g in zip(sra["SRA.Accession"], sra["Genome.Name"])}
    fam_idx = {s: i for i, s in enumerate(fam)}
    rows = np.array([fam_idx.get(gname_to_sra.get(str(g)), -1) for g in old["sample_genome_names"]])
    matched = rows >= 0
    rep["matched"] = int(matched.sum())
    if not np.array_equal(matched, np.asarray(old["matched_mask"], dtype=bool)):
        return done(1, "refused: strain matching differs from the old file's matched_mask")

    codes = read_bed(bed, len(fam), len(bim))[:, rows[matched]].T       # (n_matched, n_loci)
    hom_a1, hom_a2 = codes == 0, codes == 3
    called = hom_a1 | hom_a2
    rep.update(missing_cells=int((codes == 1).sum()), het_cells=int((codes == 2).sum()),
               called_fraction=float(called.mean()))
    del codes
    zero_a1 = (bim.a1.to_numpy() == "0")[None, :]
    zero_a2 = (bim.a2.to_numpy() == "0")[None, :]
    rep["cells_carrying_allele_0"] = int(((hom_a1 & zero_a1) | (hom_a2 & zero_a2)).sum())
    if rep["cells_carrying_allele_0"]:
        return done(1, "refused: some strains carry allele 0, which PLINK uses only as 'no allele'")

    def carries(L):
        """cells whose strain carries allele letter L at that locus"""
        return (hom_a1 & (bim.a1.to_numpy() == L)[None, :]) | (hom_a2 & (bim.a2.to_numpy() == L)[None, :])

    ecor = pd.read_csv(a.ecor)
    ecor_n = pd.Series(ecor.ecor_genomes.values, index=ecor.gene_id).reindex(gene_ids).to_numpy()
    core = ecor_n == 72
    rep["ecor_core_loci"] = int(core.sum())

    # which letter means present?
    score = {}
    for L in letters:
        prev = carries(L).sum(0) / np.maximum(called.sum(0), 1)
        score[L] = float(np.median(prev[core]))
    rep["core_median_prevalence_by_letter"] = score
    present_letter = max(score, key=score.get)
    other = [L for L in letters if L != present_letter][0]
    rep["present_letter"] = present_letter
    if score[present_letter] < CORE_MIN_MEDIAN or score[other] > OTHER_LETTER_MAX:
        return done(1, "refused: no allele letter separates ECOR-core loci clearly")

    X_new_m = carries(present_letter).astype(np.uint8)                  # missing / het -> 0

    # nothing but the orientation may change: per locus, equal or complementary on called cells
    old_m = X_old[matched]
    same = ((old_m == X_new_m) | ~called).all(0)
    comp = ((old_m == 1 - X_new_m) | ~called).all(0)
    neither = ~(same | comp)
    rep.update(loci_unchanged=int((same & ~comp).sum()), loci_flipped=int((comp & ~same).sum()),
               loci_constant=int((same & comp).sum()), loci_inconsistent=int(neither.sum()),
               example_inconsistent=[gene_ids[j] for j in np.where(neither)[0][:10]])
    if neither.any():
        return done(1, "refused: some loci differ from the old file by more than a flip")

    X_new = np.zeros_like(X_old)
    X_new[matched] = X_new_m
    prev = X_new[matched].mean(0)
    from scipy.stats import spearmanr
    known = ~np.isnan(ecor_n)
    rho = float(spearmanr(prev[known], ecor_n[known]).correlation)
    per_strain = X_new[matched].sum(1)
    named = {}
    for gname in ("gyrA", "parC", "rpoB", "recA"):
        hit = ecor[(ecor.kind == "cds") & (ecor.ecor_cluster == gname)].gene_id
        named[gname] = [round(float(prev[gene_ids.index(g)]), 4) for g in hit if g in gene_ids]
    rep.update(spearman_prevalence_vs_ecor=rho, core_median_prevalence=float(np.median(prev[core])),
               core_fraction_below_50pct=float((prev[core] < 0.5).mean()),
               per_strain_loci={"mean": float(per_strain.mean()), "median": float(np.median(per_strain)),
                                "min": int(per_strain.min()), "max": int(per_strain.max())},
               max_prevalence_old=float(X_old[matched].mean(0).max()), max_prevalence_new=float(prev.max()),
               named_core_genes=named)
    if rho < MIN_ECOR_CORR:
        return done(1, f"refused: prevalence does not follow ECOR frequency (rho {rho:.2f})")

    out = dict(old)
    out["X_pa"] = X_new
    out["encoding_rule"] = (f"present = (allele letter == '{present_letter}') from the .bim alleles; "
                            "missing or heterozygous calls = 0; rebuilt by scripts/repair_gene_pa.py")
    out["repair_note"] = (f"orientation fixed: {rep['loci_flipped']} of {len(gene_ids)} loci flipped; "
                          "the old file marked the minority state of each locus")
    out["per_strain_gene_count_stats"] = rep["per_strain_loci"]
    out["gene_prevalence_stats"] = {"median": float(np.median(prev)), "core_gt95": int((prev > 0.95).sum()),
                                    "rare_lt05": int((prev < 0.05).sum())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return done(0, "repaired")


if __name__ == "__main__":
    sys.exit(main())
