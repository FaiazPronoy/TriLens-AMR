"""
strain_ablation_table.py
========================
Builds the strain-conditioning ablation table (paper: tab:attnfix) from two
complete result folders:

    fixed     the delivered model (strain-conditioned Lens B)  -> results/
    standard  the same pipeline trained with TRILENS_LENSB=standard
              -> results/standard/  (the Kaggle notebook unpacks it there)

Every number is read from files the two runs wrote; nothing is typed in.

Usage (from code/):
    python scripts/diagnostics/strain_ablation_table.py
    python scripts/diagnostics/strain_ablation_table.py --standard results/standard

Output: results/diagnostics/strain_ablation.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _mean_t6(res: Path):
    p = res / "trilens_attnfix" / "ablation_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[(df.ablation == "T6") & (df.split_key == "random_split")]
    return float(df.prauc.mean()) if len(df) else None


def _three_way(res: Path, folder: str, col: str):
    p = res / folder / "three_way_comparison_FINAL.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if col not in df:
        return None
    hit = df[df.antibiotic.astype(str) == "AVERAGE"]
    return float(hit[col].iloc[0]) if len(hit) else float(df[df.antibiotic != "AVERAGE"][col].mean())


def _summary_value(res: Path, pattern: str):
    p = res / "pillar_mlst_final_comparison" / "SUMMARY.md"
    if not p.exists():
        return None
    m = re.search(pattern + r"[^0-9]*([0-9]+\.[0-9]+)", p.read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None


def _neural_holdout(res: Path):
    p = res / "pillar_trilens_mlst" / "trilens_mlst_results.csv"
    return float(pd.read_csv(p)["pr_auc"].mean()) if p.exists() else None


def _mean_vme(res: Path):
    for p, col in ((res / "diagnostics" / "holdout_clinical_metrics.csv", "random_VME"),
                   (res / "diagnostics" / "threshold_tuning.csv", "VME_at_0p5")):
        if p.exists():
            df = pd.read_csv(p)
            if col in df:
                return float(df[col].mean())
    return None


def _amr_hits(res: Path):
    p = res / "diagnostics" / "attention_amr_hits.csv"
    if not p.exists():
        return 0 if (res / "diagnostics" / "attention_diagnosis.csv").exists() else None
    df = pd.read_csv(p)
    return int(df[df.lens.astype(str) == "B"]["gene"].nunique()) if len(df) else 0


def _marker(res: Path, col: str):
    p = res / "diagnostics" / "lineage_marker_test.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    return float(df[col].mean()) if col in df else None


def _enrichment(res: Path):
    p = res / "diagnostics" / "prefix_baserate_lens_b.csv"
    if not p.exists():
        return None
    return float(np.median(pd.read_csv(p)["enrichment"]))


ROWS = [
    # measure, function, paper standard, paper fixed
    ("Random split, full stack",          lambda r: _mean_t6(r), 0.8462, 0.8486),
    ("Phylogroup-stratified split",       lambda r: _three_way(r, "pillar_t7_clean_final_comparison", "phylogeny_clean_gene_set"), 0.8519, 0.8531),
    ("Lineage holdout, full stack",       lambda r: _three_way(r, "pillar_mlst_final_comparison", "mlst_lineage_holdout"), 0.6300, 0.6337),
    ("Lineage holdout, classical alone",  lambda r: _summary_value(r, r"classical-only \(5 bases\), MLST holdout"), 0.628, 0.628),
    ("Holdout, neural lenses alone",      lambda r: _neural_holdout(r), 0.437, 0.472),
    ("Mean very major error rate",        lambda r: _mean_vme(r), 0.257, 0.249),
    ("Resistance genes by attention",     lambda r: _amr_hits(r), 0, 1),
    ("U_res of attended genes",           lambda r: _marker(r, "attention_U_res"), 0.0192, 0.0419),
    ("U_ST of attended genes",            lambda r: _marker(r, "attention_U_ST"), 0.3123, 0.3569),
    ("Enrichment on one assembly",        lambda r: _enrichment(r), 16.4, 9.0),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", default=str(ROOT / "results"))
    ap.add_argument("--standard", default=str(ROOT / "results" / "standard"))
    a = ap.parse_args()
    fixed, std = Path(a.fixed), Path(a.standard)

    def safe(fn, root):
        try:
            return fn(root)
        except Exception as exc:          # one unreadable file must not hide the rest
            print(f"  {root.name}: {exc!r}")
            return None

    out = []
    for name, fn, p_std, p_fix in ROWS:
        s, f = safe(fn, std), safe(fn, fixed)
        out.append(dict(measure=name,
                        standard=None if s is None else round(s, 4),
                        fixed=None if f is None else round(f, 4),
                        change=None if (s is None or f is None) else round(f - s, 4),
                        paper_standard=p_std, paper_fixed=p_fix))
    df = pd.DataFrame(out)
    dst = fixed / "diagnostics" / "strain_ablation.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {dst}")
    if df[["standard", "fixed"]].isna().any().any():
        print("Some cells are empty: the run that writes them has not finished yet.")


if __name__ == "__main__":
    main()
