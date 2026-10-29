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


CORE_MIN_MEDIAN = 0.90
OTHER_LETTER_MAX = 0.10
MIN_ECOR_CORR = 0.50


def read_bim(path):
    bim = pd.read_csv(path, sep=r"\s+", header=None, dtype=str, names=["chrom", "id", "cm", "pos", "a1", "a2"])
    return bim


def read_fam(path):
    return pd.read_csv(path, sep=r"\s+", header=None, dtype=str).iloc[:, 1].tolist()


def read_bed(path, n_samples, n_variants):
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

    return codes[:, :n_samples]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", help="presence_absence_all.bed (.bim/.fam next to it)")
    ap.add_argument("--source", help="paths.json written by gene_presence_source.py (sets --bed and --sra)")
    ap.add_argument("--old", help="an earlier matrix to check against: only the orientation may differ")
    ap.add_argument(
        "--master",
        default="data/processed/arcadia_v2_master_features.pkl",
        help="strain order to follow when --old is not given",
    )
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

    letters = sorted((set(bim.a1) | set(bim.a2)) - {"0"})
    rep.update(
        n_variants=len(bim),
        n_fam=len(fam),
        allele_letters=letters,
        allele_order_counts={f"{k[0]}/{k[1]}": int(v) for k, v in bim.groupby(["a1", "a2"]).size().items()},
    )
    rep["monomorphic_loci"] = int(((bim.a1 == "0") | (bim.a2 == "0")).sum())
    if len(letters) != 2:
        return done(1, f"refused: expected two allele letters besides 0, found {letters}")

    if a.old:
        with open(a.old, "rb") as f:
            old = pickle.load(f)
        X_old = np.asarray(old["X_pa"], dtype=np.uint8)
    else:
        with open(a.master, "rb") as f:
            master = pickle.load(f)
        old = {
            "gene_ids": bim.chrom.tolist(),
            "sample_genome_names": list(master["genome_names"]),
            "matched_mask": None,
        }
        X_old = None
    gene_ids = [str(g) for g in old["gene_ids"]]
    if bim.chrom.tolist() != gene_ids:
        return done(1, "refused: .bim loci are not the old file's gene_ids in the same order")

    sra = pd.read_csv(a.sra)
    gname_to_sra = {str(g): str(s) for s, g in zip(sra["SRA.Accession"], sra["Genome.Name"])}
    fam_idx = {s: i for i, s in enumerate(fam)}
    rows = np.array([fam_idx.get(gname_to_sra.get(str(g)), -1) for g in old["sample_genome_names"]])
    matched = rows >= 0
    rep["matched"] = int(matched.sum())
    if old["matched_mask"] is not None and not np.array_equal(matched, np.asarray(old["matched_mask"], dtype=bool)):
        return done(1, "refused: strain matching differs from the old file's matched_mask")

    codes = read_bed(bed, len(fam), len(bim))[:, rows[matched]].T
    hom_a1, hom_a2 = codes == 0, codes == 3
    called = hom_a1 | hom_a2
    rep.update(
        missing_cells=int((codes == 1).sum()), het_cells=int((codes == 2).sum()), called_fraction=float(called.mean())
    )
    del codes
    zero_a1 = (bim.a1.to_numpy() == "0")[None, :]
    zero_a2 = (bim.a2.to_numpy() == "0")[None, :]
    rep["cells_carrying_allele_0"] = int(((hom_a1 & zero_a1) | (hom_a2 & zero_a2)).sum())
    if rep["cells_carrying_allele_0"]:
        return done(1, "refused: some strains carry allele 0, which PLINK uses only as 'no allele'")

    def carries(L):
        return (hom_a1 & (bim.a1.to_numpy() == L)[None, :]) | (hom_a2 & (bim.a2.to_numpy() == L)[None, :])

    ecor = pd.read_csv(a.ecor)
    ecor_n = pd.Series(ecor.ecor_genomes.values, index=ecor.gene_id).reindex(gene_ids).to_numpy()
    core = ecor_n == 72
    rep["ecor_core_loci"] = int(core.sum())

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

    X_new_m = carries(present_letter).astype(np.uint8)

    if X_old is not None:
        old_m = X_old[matched]
        same = ((old_m == X_new_m) | ~called).all(0)
        comp = ((old_m == 1 - X_new_m) | ~called).all(0)
        neither = ~(same | comp)
        rep.update(
            loci_unchanged=int((same & ~comp).sum()),
            loci_flipped=int((comp & ~same).sum()),
            loci_constant=int((same & comp).sum()),
            loci_inconsistent=int(neither.sum()),
            example_inconsistent=[gene_ids[j] for j in np.where(neither)[0][:10]],
        )
        if neither.any():
            return done(1, "refused: some loci differ from the old file by more than a flip")

    X_new = np.zeros((len(rows), len(gene_ids)), dtype=np.uint8)
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
    rep.update(
        spearman_prevalence_vs_ecor=rho,
        core_median_prevalence=float(np.median(prev[core])),
        core_fraction_below_50pct=float((prev[core] < 0.5).mean()),
        per_strain_loci={
            "mean": float(per_strain.mean()),
            "median": float(np.median(per_strain)),
            "min": int(per_strain.min()),
            "max": int(per_strain.max()),
        },
        max_prevalence_old=None if X_old is None else float(X_old[matched].mean(0).max()),
        max_prevalence_new=float(prev.max()),
        named_core_genes=named,
    )
    if rho < MIN_ECOR_CORR:
        return done(1, f"refused: prevalence does not follow ECOR frequency (rho {rho:.2f})")

    out = dict(old)
    out["matched_mask"] = matched
    out["X_pa"] = X_new
    out["encoding_rule"] = (
        f"present = (allele letter == '{present_letter}') from the .bim alleles; "
        "missing or heterozygous calls = 0; rebuilt by trilens/data/gene_presence_rebuild.py"
    )
    if X_old is not None:
        out["repair_note"] = (
            f"orientation fixed: {rep['loci_flipped']} of {len(gene_ids)} loci flipped; "
            "the old file marked the minority state of each locus"
        )
    out["per_strain_gene_count_stats"] = rep["per_strain_loci"]
    out["gene_prevalence_stats"] = {
        "median": float(np.median(prev)),
        "core_gt95": int((prev > 0.95).sum()),
        "rare_lt05": int((prev < 0.05).sum()),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    return done(0, "repaired" if X_old is not None else "built")


if __name__ == "__main__":
    sys.exit(main())
