from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def _mean_full_stack(res: Path):
    p = res / "trilens_stack_random" / "ablation_results.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[(df.ablation == "full_stack") & (df.split_key == "random_split")]
    return float(df.prauc.mean()) if len(df) else None


def _three_way(res: Path, folder: str, col: str):
    p = res / folder / "split_comparison.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if col not in df:
        return None
    hit = df[df.antibiotic.astype(str) == "AVERAGE"]
    return float(hit[col].iloc[0]) if len(hit) else float(df[df.antibiotic != "AVERAGE"][col].mean())


def _summary_value(res: Path, pattern: str):
    p = res / "trilens_stack_lineage" / "SUMMARY.md"
    if not p.exists():
        return None
    m = re.search(pattern + r"[^0-9]*([0-9]+\.[0-9]+)", p.read_text(encoding="utf-8"))
    return float(m.group(1)) if m else None


def _neural_holdout(res: Path):
    p = res / "trilens_neural_lineage" / "trilens_neural_lineage_results.csv"
    return float(pd.read_csv(p)["pr_auc"].mean()) if p.exists() else None


def _mean_vme(res: Path):
    for p, col in (
        (res / "diagnostics" / "holdout_clinical_errors.csv", "random_VME"),
        (res / "diagnostics" / "threshold_tuning.csv", "VME_at_0p5"),
    ):
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
    p = res / "diagnostics" / "assembly_provenance_enrichment.csv"
    if not p.exists():
        return None
    return float(np.median(pd.read_csv(p)["enrichment"]))


ROWS = [
    ("Random split, full stack", _mean_full_stack),
    ("Phylogroup-stratified split", lambda r: _three_way(r, "trilens_stack_phylogroup", "phylogeny_clean_gene_set")),
    ("Lineage holdout, full stack", lambda r: _three_way(r, "trilens_stack_lineage", "mlst_lineage_holdout")),
    ("Lineage holdout, classical alone", lambda r: _summary_value(r, r"classical-only \(5 bases\), MLST holdout")),
    ("Holdout, neural branches alone", _neural_holdout),
    ("Mean very major error rate", _mean_vme),
    ("Resistance genes by attention", _amr_hits),
    ("U_res of attended genes", lambda r: _marker(r, "attention_U_res")),
    ("U_ST of attended genes", lambda r: _marker(r, "attention_U_ST")),
    ("Enrichment on one assembly", _enrichment),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditioned", default=str(ROOT / "results"))
    ap.add_argument("--standard", default=str(ROOT / "results" / "standard_attention"))
    a = ap.parse_args()
    conditioned, std = Path(a.conditioned), Path(a.standard)

    def safe(fn, root):
        try:
            return fn(root)
        except Exception as exc:
            print(f"  {root.name}: {exc!r}")
            return None

    out = []
    for name, fn in ROWS:
        s, c = safe(fn, std), safe(fn, conditioned)
        out.append(
            dict(
                measure=name,
                standard=None if s is None else round(s, 4),
                conditioned=None if c is None else round(c, 4),
                change=None if (s is None or c is None) else round(c - s, 4),
            )
        )
    df = pd.DataFrame(out)
    dst = conditioned / "diagnostics" / "strain_conditioning.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved -> {dst}")
    if df[["standard", "conditioned"]].isna().any().any():
        print("Some cells are empty: the run that writes them has not finished yet.")


if __name__ == "__main__":
    main()
