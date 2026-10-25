"""
collect_paper_numbers.py
========================
Reads every headline number of the paper from the result files and puts it
next to the value currently printed in the paper (paper/journal/report.tex).
After a full re-run this is the one file to open when updating the paper.

Usage (from code/):
    python scripts/diagnostics/collect_paper_numbers.py

Output: results/diagnostics/paper_numbers.csv
    table, measure, paper, new, diff, status, source
status: SAME (within 0.0005), CLOSE (within 0.01), CHANGED, MISSING
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
DIAG = RES / "diagnostics"
AB = ["AMP", "CIP", "AMC", "SXT", "GEN", "CTX", "CAZ"]


def _csv(rel):
    p = RES / rel
    return pd.read_csv(p) if p.exists() else None


def main_t6(ab=None):
    df = _csv("trilens_attnfix/ablation_results.csv")
    if df is None:
        return None
    df = df[(df.ablation == "T6") & (df.split_key == "random_split")]
    if ab:
        df = df[df.ab == ab]
    return float(df.prauc.mean()) if len(df) else None


def layer_stage(n):
    df = _csv("pillar2_layer_ablation/layer_ablation_results.csv")
    if df is None:
        return None
    hit = df[df.stage.astype(str).str.startswith(f"stage{n}")]
    return float(hit.pr_auc.mean()) if len(hit) else None


def col_mean(rel, col, row_filter=None):
    df = _csv(rel)
    if df is None or col not in df:
        return None
    if row_filter is not None:
        df = df[row_filter(df)]
    df = df[df["antibiotic"].astype(str) != "AVERAGE"] if "antibiotic" in df else df
    return float(df[col].mean()) if len(df) else None


def three_way(rel, col):
    df = _csv(rel)
    if df is None or col not in df:
        return None
    hit = df[df.antibiotic.astype(str) == "AVERAGE"]
    return float(hit[col].iloc[0]) if len(hit) else None


def summary_number(rel, pattern, group=1):
    p = RES / rel
    if not p.exists():
        return None
    m = re.search(pattern, p.read_text(encoding="utf-8"))
    return float(m.group(group)) if m else None


def giessen_retention():
    g = _csv("pillar4_giessen_transfer/pillar4_per_ab_metrics.csv")
    m = _csv("trilens_attnfix/ablation_results.csv")
    if g is None or m is None:
        return None
    m = m[(m.ablation == "T6") & (m.split_key == "random_split")].set_index("ab").prauc
    r = [row.pr_auc / m[row.antibiotic] for row in g.itertuples() if row.antibiotic in m]
    return float(np.mean(r)) if r else None


def dedup_cost():
    df = _csv("pillar_dedup_eval/dedup_eval_results.csv")
    return None if df is None else float(df.pr_auc_before.mean() - df.pr_auc_after.mean())


def strain_row(measure, which):
    df = _csv("diagnostics/strain_ablation.csv")
    if df is None:
        return None
    hit = df[df.measure == measure]
    v = hit[which].iloc[0] if len(hit) else None
    return None if v is None or pd.isna(v) else float(v)


def _v9_solo(col):
    return col_mean("stage3_cascadestack_v9/v9_results.csv", f"solo_{col}_test_pr_auc")


def _mlst_solo(col):
    return col_mean("pillar_cascadestack_classical_mlst/classical5_results.csv", f"solo_{col}_test_pr_auc")


PREV = "diagnostics/matched_prevalence.csv"
CLIN = "diagnostics/holdout_clinical_metrics.csv"
MLST3 = "pillar_mlst_final_comparison/three_way_comparison_FINAL.csv"
T73 = "pillar_t7_clean_final_comparison/three_way_comparison_FINAL.csv"

ROWS = [
    # table, measure, paper value, getter, source
    ("tab:main", "Average PR-AUC, random split", 0.8486, lambda: main_t6(), "trilens_attnfix/ablation_results.csv"),
    *[("tab:main", f"{ab} PR-AUC, random split", v, (lambda a=ab: main_t6(a)), "trilens_attnfix/ablation_results.csv")
      for ab, v in zip(AB, [0.9440, 0.9773, 0.7482, 0.8352, 0.7037, 0.9019, 0.8299])],
    ("tab:ablation", "Stage 1, lenses only", 0.7453, lambda: layer_stage(1), "pillar2_layer_ablation/"),
    ("tab:ablation", "Stage 2, + MoE gate", 0.7807, lambda: layer_stage(2), "pillar2_layer_ablation/"),
    ("tab:ablation", "Stage 3, + stacking", 0.8486, lambda: layer_stage(3), "pillar2_layer_ablation/"),
    ("tab:baselines", "Phylogroup label only", 0.2914, lambda: col_mean("pillar_phylogroup_only/phylogroup_only_results.csv", "logreg_pr_auc"), "pillar_phylogroup_only/"),
    ("tab:baselines", "TabNet, random split", 0.6951, lambda: col_mean("tabnet_baseline/tabnet_results.csv", "pr_auc"), "tabnet_baseline/"),
    ("tab:baselines", "TabNet, lineage holdout", 0.3255, lambda: col_mean("pillar_tabnet_mlst/tabnet_mlst_results.csv", "pr_auc"), "pillar_tabnet_mlst/"),
    ("tab:baselines", "XGBoost on 1,024 PCA components", 0.8033, lambda: col_mean("baselines/xgb_phylogroup_results.csv", "test_pr_auc", lambda d: d.variant == "no_phylogroup"), "baselines/"),
    ("tab:baselines", "Random Forest, random", 0.7809, lambda: _v9_solo("RF-gene"), "stage3_cascadestack_v9/v9_results.csv"),
    ("tab:baselines", "Logistic Regression, random", 0.7923, lambda: _v9_solo("LogReg-gene"), "stage3_cascadestack_v9/v9_results.csv"),
    ("tab:baselines", "LightGBM, random", 0.8338, lambda: _v9_solo("LGBM-gene"), "stage3_cascadestack_v9/v9_results.csv"),
    ("tab:baselines", "XGBoost gene features, random", 0.8375, lambda: _v9_solo("XGB-gene"), "stage3_cascadestack_v9/v9_results.csv"),
    ("tab:baselines", "XGBoost chi2 SNPs, random (paper 0.8400, v9 file 0.8430)", 0.8400, lambda: _v9_solo("XGB-raw-chi2"), "stage3_cascadestack_v9/v9_results.csv"),
    ("tab:baselines", "Random Forest, holdout", 0.4931, lambda: _mlst_solo("RF-gene"), "pillar_cascadestack_classical_mlst/"),
    ("tab:baselines", "Logistic Regression, holdout", 0.4408, lambda: _mlst_solo("LogReg-gene"), "pillar_cascadestack_classical_mlst/"),
    ("tab:baselines", "LightGBM, holdout", 0.5516, lambda: _mlst_solo("LGBM-gene"), "pillar_cascadestack_classical_mlst/"),
    ("tab:baselines", "XGBoost gene features, holdout", 0.5507, lambda: _mlst_solo("XGB-gene"), "pillar_cascadestack_classical_mlst/"),
    ("tab:baselines", "XGBoost chi2 SNPs, holdout", 0.6299, lambda: _mlst_solo("XGB-raw-chi2"), "pillar_cascadestack_classical_mlst/"),
    ("tab:mlst_per_ab", "Phylogroup split, full stack", 0.8531, lambda: three_way(T73, "phylogeny_clean_gene_set"), T73),
    ("tab:mlst_per_ab", "Lineage holdout, full stack", 0.6337, lambda: three_way(MLST3, "mlst_lineage_holdout"), MLST3),
    ("tab:mlst_components", "Classical arm alone, holdout", 0.628, lambda: summary_number("pillar_mlst_final_comparison/SUMMARY.md", r"classical-only \(5 bases\), MLST holdout[^0-9]*([0-9.]+)"), "pillar_mlst_final_comparison/SUMMARY.md"),
    ("tab:mlst_components", "Neural lenses alone, holdout", 0.472, lambda: col_mean("pillar_trilens_mlst/trilens_mlst_results.csv", "pr_auc"), "pillar_trilens_mlst/"),
    ("tab:previnv", "Holdout as measured, average", 0.6337, lambda: col_mean(PREV, "pr_auc_raw"), PREV),
    ("tab:previnv", "Holdout prevalence-matched, average", 0.7226, lambda: col_mean(PREV, "pr_auc_matched"), PREV),
    ("tab:previnv", "Holdout ROC-AUC, average", 0.8439, lambda: col_mean(PREV, "roc_auc"), PREV),
    ("tab:previnv", "AMP holdout as measured", 0.9043, lambda: col_mean(PREV, "pr_auc_raw", lambda d: d.antibiotic == "AMP"), PREV),
    ("tab:clinical", "Mean VME at 0.5, random split", 0.249, lambda: col_mean("diagnostics/threshold_tuning.csv", "VME_at_0p5"), "diagnostics/threshold_tuning.csv"),
    ("tab:holdoutclin", "Mean VME, random split", 0.249, lambda: col_mean(CLIN, "random_VME"), CLIN),
    ("tab:holdoutclin", "Mean VME, lineage holdout", 0.559, lambda: col_mean(CLIN, "holdout_VME"), CLIN),
    ("tab:giessen_transfer", "Giessen retention (mean of per-drug ratios)", 0.900, giessen_retention, "pillar4_giessen_transfer/"),
    ("tab:umark", "U_ST of Lens B attended genes", 0.3569, lambda: col_mean("diagnostics/lineage_marker_test.csv", "attention_U_ST"), "diagnostics/lineage_marker_test.csv"),
    ("tab:umark", "U_res of Lens B attended genes", 0.0419, lambda: col_mean("diagnostics/lineage_marker_test.csv", "attention_U_res"), "diagnostics/lineage_marker_test.csv"),
    ("sec:statanalysis", "Near-duplicate test strains, random split", 47, lambda: summary_number("pillar_dedup_eval/SUMMARY.md", r"(\d+)/(\d+) test samples"), "pillar_dedup_eval/SUMMARY.md"),
    ("sec:statanalysis", "PR-AUC cost of removing them", 0.0041, dedup_cost, "pillar_dedup_eval/dedup_eval_results.csv"),
    *[(f"tab:attnfix", f"{m} ({w})", v, (lambda m=m, w=w: strain_row(m, w)), "diagnostics/strain_ablation.csv")
      for m, ps, pf in [("Random split, full stack", 0.8462, 0.8486),
                        ("Lineage holdout, full stack", 0.6300, 0.6337),
                        ("Holdout, neural lenses alone", 0.437, 0.472),
                        ("U_res of attended genes", 0.0192, 0.0419),
                        ("Enrichment on one assembly", 16.4, 9.0)]
      for w, v in (("standard", ps), ("fixed", pf))],
]


def main() -> None:
    out = []
    for table, measure, paper, fn, src in ROWS:
        try:
            new = fn()
        except Exception as exc:          # one unreadable file must not hide the rest
            print(f"  could not read {src}: {exc}")
            new = None
        if new is None:
            status, diff = "MISSING", None
        else:
            diff = new - paper
            tol = 0.0005 * (100 if abs(paper) > 2 else 1)
            status = "SAME" if abs(diff) <= tol else ("CLOSE" if abs(diff) <= 0.01 * (100 if abs(paper) > 2 else 1) else "CHANGED")
        out.append(dict(table=table, measure=measure, paper=paper,
                        new=None if new is None else round(new, 4),
                        diff=None if diff is None else round(diff, 4),
                        status=status, source=src))
    df = pd.DataFrame(out)
    DIAG.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIAG / "paper_numbers.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(df.drop(columns="source").to_string(index=False))
    print(f"\n{(df.status == 'SAME').sum()} same, {(df.status == 'CLOSE').sum()} close, "
          f"{(df.status == 'CHANGED').sum()} changed, {(df.status == 'MISSING').sum()} missing")
    print(f"Saved -> {DIAG / 'paper_numbers.csv'}")


if __name__ == "__main__":
    main()
