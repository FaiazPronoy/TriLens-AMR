"""
src/experiments/run_pillar_master_summary.py
==============================================
Master consolidated results table -- pulls every experiment's ALREADY-SAVED
numbers (no new computation beyond simple averaging/subtraction of existing
values) into one table. Designed to be re-run as pending GPU jobs
(CascadeStack-phylogeny, TabNet-phylogeny, T6-multiseed) complete -- any
result file not yet on disk is reported as "pending", never fabricated or
guessed.

Usage:
    conda activate amr
    python -m src.experiments.run_pillar_master_summary
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "pillar_master_summary"


def _safe_read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def _row(model, split, avg_pr_auc, avg_mcc, n_seeds, note):
    return {"model": model, "split": split, "avg_pr_auc": avg_pr_auc, "avg_mcc": avg_mcc,
            "n_seeds": n_seeds, "note": note}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    # ── T6 / T7 (TriLens-MoE-Stack v10) ──────────────────────────────────────
    abl_path = RESULTS_DIR / "trilens_v10" / "ablation_results.csv"
    abl_df = _safe_read_csv(abl_path)
    t6_per_ab, t7_per_ab = {}, {}
    if abl_df is not None:
        t6 = abl_df[(abl_df["ablation"] == "T6") & (abl_df["split_key"] == "random_split")]
        t7 = abl_df[(abl_df["ablation"] == "T7") & (abl_df["split_key"] == "phylogeny_split")]
        t6_per_ab = dict(zip(t6["ab"], t6["prauc"]))
        t7_per_ab = dict(zip(t7["ab"], t7["prauc"]))
        rows.append(_row("TriLens T6 (headline)", "random_split", t6["prauc"].mean(), t6["mcc"].mean(), 1, "seed=42 only"))
        rows.append(_row("TriLens T7", "phylogeny_split", t7["prauc"].mean(), t7["mcc"].mean(), 1, "seed=42 only, leakage-corrected"))
    else:
        rows.append(_row("TriLens T6 (headline)", "random_split", None, None, None, "MISSING FILE: " + str(abl_path)))

    # ── T6 multi-seed (Task 4, pending) ──────────────────────────────────────
    t6ms_path = RESULTS_DIR / "pillar_t6_multiseed" / "t6_multiseed_results.csv"
    t6ms_df = _safe_read_csv(t6ms_path)
    if t6ms_df is not None:
        seed_cols = [c for c in t6ms_df.columns if c.startswith("seed_")]
        rows.append(_row("TriLens T6 (multi-seed)", "random_split", t6ms_df["mean"].mean(), None, len(seed_cols), "mean across seeds"))
    else:
        rows.append(_row("TriLens T6 (multi-seed)", "random_split", None, None, None, "PENDING -- run run_pillar_t6_multiseed.py"))

    # ── CascadeStack v9.1 random_split (2-seed avg, existing) ────────────────
    v9_seed_avgs = []
    for seed in (123, 456):
        p = RESULTS_DIR / "stage3_cascadestack_v9" / f"SEED {seed}" / "v9_results.csv"
        df = _safe_read_csv(p)
        if df is not None:
            v9_seed_avgs.append(df["ensemble_pr_auc"].mean())
    if v9_seed_avgs:
        v9_mcc_avgs = []
        for seed in (123, 456):
            p = RESULTS_DIR / "stage3_cascadestack_v9" / f"SEED {seed}" / "v9_results.csv"
            df = _safe_read_csv(p)
            if df is not None:
                v9_mcc_avgs.append(df["ensemble_mcc"].mean())
        rows.append(_row("CascadeStack v9.1", "random_split", sum(v9_seed_avgs) / len(v9_seed_avgs),
                          sum(v9_mcc_avgs) / len(v9_mcc_avgs) if v9_mcc_avgs else None,
                          len(v9_seed_avgs), "seeds 123, 456"))
    else:
        rows.append(_row("CascadeStack v9.1", "random_split", None, None, None, "MISSING"))

    # ── CascadeStack v9.1 phylogeny_split (Task 1, pending) ──────────────────
    cs_phylo_avgs, cs_phylo_mccs = [], []
    for seed in (123, 456):
        p = RESULTS_DIR / "pillar_cascadestack_phylogeny" / f"SEED_{seed}" / "v9_results.csv"
        df = _safe_read_csv(p)
        if df is not None:
            cs_phylo_avgs.append(df["ensemble_pr_auc"].mean())
            cs_phylo_mccs.append(df["ensemble_mcc"].mean())
    if cs_phylo_avgs:
        rows.append(_row("CascadeStack v9.1", "phylogeny_split", sum(cs_phylo_avgs) / len(cs_phylo_avgs),
                          sum(cs_phylo_mccs) / len(cs_phylo_mccs), len(cs_phylo_avgs), "seeds 123, 456"))
    else:
        rows.append(_row("CascadeStack v9.1", "phylogeny_split", None, None, None,
                          "PENDING -- run run_pillar_cascadestack_phylogeny.py (Task 1)"))

    # ── TabNet ────────────────────────────────────────────────────────────
    tn_path = RESULTS_DIR / "tabnet_baseline" / "tabnet_results.csv"
    tn_df = _safe_read_csv(tn_path)
    if tn_df is not None:
        rows.append(_row("TabNet", "random_split", tn_df["pr_auc"].mean(), tn_df["mcc"].mean(), 1, "seed=42"))
    else:
        rows.append(_row("TabNet", "random_split", None, None, None, "PENDING"))

    tnp_path = RESULTS_DIR / "pillar_tabnet_phylogeny" / "tabnet_phylogeny_results.csv"
    tnp_df = _safe_read_csv(tnp_path)
    if tnp_df is not None:
        rows.append(_row("TabNet", "phylogeny_split", tnp_df["pr_auc"].mean(), tnp_df["mcc"].mean(), 1, "seed=42"))
    else:
        rows.append(_row("TabNet", "phylogeny_split", None, None, None, "PENDING -- run run_tabnet_baseline_phylogeny.py (Task 2)"))

    # ── T6 layer ablation (Stage 1/2/3) ──────────────────────────────────────
    la_path = RESULTS_DIR / "pillar2_layer_ablation" / "layer_ablation_results.csv"
    la_df = _safe_read_csv(la_path)
    if la_df is not None:
        for stage, label in [("stage1_expert_avg", "T6 Stage 1 (lens avg, no gate)"),
                              ("stage2_gate_fusion", "T6 Stage 2 (gate+fusion, no stacking)"),
                              ("stage3_full_t6", "T6 Stage 3 (full, == headline)")]:
            sub = la_df[la_df["stage"] == stage]
            if len(sub):
                rows.append(_row(label, "random_split", sub["pr_auc"].mean(), None, 1, "layer ablation"))

    # ── Dedup eval (before/after, random_split) ──────────────────────────────
    dedup_path = RESULTS_DIR / "pillar_dedup_eval" / "dedup_eval_results.csv"
    dedup_df = _safe_read_csv(dedup_path)
    if dedup_df is not None:
        rows.append(_row("TriLens T6 (before dedup)", "random_split", dedup_df["pr_auc_before"].mean(), None, 1, "same as headline"))
        rows.append(_row("TriLens T6 (after dedup)", "random_split", dedup_df["pr_auc_after"].mean(), None, 1, "near-duplicate test samples removed"))

    # ── Other existing baselines (exact column names verified against each
    # file's actual header before writing this, not guessed) ────────────────
    xgb_gene_path = RESULTS_DIR / "stage2_gene_xgb" / "stage2_gene_xgb_results.csv"
    xgb_gene_df = _safe_read_csv(xgb_gene_path)
    if xgb_gene_df is not None:
        sub = xgb_gene_df[xgb_gene_df["variant"] == "no_pg"]
        rows.append(_row("XGB-gene (stage2_gene_xgb, no_pg)", "random_split",
                          sub["test_pr_auc"].mean(), sub["test_mcc"].mean(), 1, "no phylogroup feature variant"))
    else:
        rows.append(_row("XGB-gene (stage2_gene_xgb, no_pg)", "random_split", None, None, None,
                          f"MISSING FILE: {xgb_gene_path.relative_to(ROOT)}"))

    gan_path = RESULTS_DIR / "gene_attention_v2" / "v2_vs_v1_vs_xgb.csv"
    gan_df = _safe_read_csv(gan_path)
    if gan_df is not None:
        rows.append(_row("GeneAttentionNet v2 (best variant E4)", "random_split",
                          gan_df["v2_prauc"].mean(), None, 1, "best-variant-per-AB, no MCC column in source file"))
    else:
        rows.append(_row("GeneAttentionNet v2 (best variant E4)", "random_split", None, None, None,
                          f"MISSING FILE: {gan_path.relative_to(ROOT)}"))

    se_path = RESULTS_DIR / "stacking_ensemble" / "ensemble_results.csv"
    se_df = _safe_read_csv(se_path)
    if se_df is not None:
        rows.append(_row("Stacking ensemble (3-base, early prototype)", "random_split",
                          se_df["ensemble_meta_pr_auc"].mean(), se_df["ensemble_meta_mcc"].mean(), 1, ""))
    else:
        rows.append(_row("Stacking ensemble (3-base, early prototype)", "random_split", None, None, None,
                          f"MISSING FILE: {se_path.relative_to(ROOT)}"))

    v8_path = RESULTS_DIR / "stage3_cascadestack_v8" / "v8_results.csv"
    v8_df = _safe_read_csv(v8_path)
    if v8_df is not None:
        rows.append(_row("CascadeStack v8 (single-seed)", "random_split",
                          v8_df["ensemble_pr_auc"].mean(), v8_df["ensemble_mcc"].mean(), 1, ""))
    else:
        rows.append(_row("CascadeStack v8 (single-seed)", "random_split", None, None, None,
                          f"MISSING FILE: {v8_path.relative_to(ROOT)}"))

    v11_path = RESULTS_DIR / "stage3_deepmeta_v11" / "v11_results.csv"
    v11_df = _safe_read_csv(v11_path)
    if v11_df is not None:
        rows.append(_row("DeepMetaStack v11", "random_split",
                          v11_df["deepmeta_pr_auc"].mean(), v11_df["deepmeta_mcc"].mean(), 1, "abandoned, underperforms v8/v9"))
    else:
        rows.append(_row("DeepMetaStack v11", "random_split", None, None, None,
                          f"MISSING FILE: {v11_path.relative_to(ROOT)}"))

    master_df = pd.DataFrame(rows, columns=["model", "split", "avg_pr_auc", "avg_mcc", "n_seeds", "note"])
    master_df.to_csv(OUTPUT_DIR / "master_results.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'master_results.csv'}")

    _write_master_md(master_df, t6_per_ab, t7_per_ab)


def _write_master_md(master_df: pd.DataFrame, t6_per_ab: dict, t7_per_ab: dict) -> None:
    lines = ["# MASTER RESULTS -- All Experiments Consolidated\n"]
    lines.append(
        "Every number below is read directly from an existing saved results "
        "file (path noted in `note` for baselines, or implied by folder name); "
        "rows marked PENDING have no file yet -- not guessed, not filled in.\n"
    )
    lines.append("## Master table\n")
    lines.append("| Model | Split | Avg PR-AUC | Avg MCC | N seeds | Note |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in master_df.iterrows():
        pr = f"{r['avg_pr_auc']:.4f}" if pd.notna(r["avg_pr_auc"]) else "—"
        mcc = f"{r['avg_mcc']:.4f}" if pd.notna(r["avg_mcc"]) else "—"
        ns = str(int(r["n_seeds"])) if pd.notna(r["n_seeds"]) else "—"
        lines.append(f"| {r['model']} | {r['split']} | {pr} | {mcc} | {ns} | {r['note']} |")
    lines.append("")

    if t6_per_ab and t7_per_ab:
        lines.append("## Per-antibiotic confounding score (T6 random_split PR-AUC minus T7 phylogeny_split PR-AUC)\n")
        lines.append("Existing numbers only, simple subtraction -- larger positive value means "
                     "more of that antibiotic's apparent performance depends on phylogeny-correlated "
                     "(potentially shortcut) signal.\n")
        lines.append("| Antibiotic | T6 (random_split) | T7 (phylogeny_split) | Confounding score (T6 - T7) |")
        lines.append("|---|---|---|---|")
        for ab in sorted(set(t6_per_ab) & set(t7_per_ab)):
            lines.append(f"| {ab} | {t6_per_ab[ab]:.4f} | {t7_per_ab[ab]:.4f} | {t6_per_ab[ab] - t7_per_ab[ab]:+.4f} |")
        lines.append("")

    n_pending = master_df["note"].astype(str).str.contains("PENDING").sum()
    n_missing = master_df["note"].astype(str).str.contains("MISSING").sum()
    lines.append(f"**Status: {n_pending} row(s) pending (GPU jobs not yet run), "
                 f"{n_missing} row(s) with an unexpectedly missing/mismatched file.** "
                 f"Re-run this script after each pending job completes.\n")

    (OUTPUT_DIR / "MASTER_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'MASTER_RESULTS.md'}")


if __name__ == "__main__":
    main()
