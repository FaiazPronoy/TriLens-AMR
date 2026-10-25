# src/experiments/aggregate_pillar1_results.py
"""Aggregate all Pillar 1 experiment results into a single report.

Reads:
    {paths.results_dir}/pillar1/exp1/exp1_results.csv
    {paths.results_dir}/pillar1/exp2/exp2_results.csv
    {paths.results_dir}/pillar1/exp3/exp3_results.csv
    {paths.results_dir}/baselines/xgb_phylogroup_results.csv

Writes:
    {paths.results_dir}/pillar1/SUMMARY.md
    {paths.results_dir}/pillar1/figures/pillar1_overview.png
        (4-panel: Exp1 bar chart with error bars, Exp2 model comparison,
         Exp3 asymmetric drop, baseline XGB phylogroup delta)
    {paths.results_dir}/pillar1/pillar1_combined.csv
    paper/outline.md   (Pillar 1 evidence section drafted)

Run as a script (no Hydra dispatch needed):
    python -m src.experiments.aggregate_pillar1_results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  [WARN] Missing: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as ex:  # noqa: BLE001
        print(f"  [WARN] Could not read {path}: {ex!r}")
        return None


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------
def _verdict_exp1(df: pd.DataFrame) -> tuple[str, dict]:
    ok = df[df["status"] == "ok"]
    if ok.empty or not {"arcadia", "giessen"}.issubset(set(ok["dataset"])):
        return "INCONCLUSIVE (data missing)", {}
    arc = ok[ok["dataset"] == "arcadia"]["test_pr_auc"].mean()
    gie = ok[ok["dataset"] == "giessen"]["test_pr_auc"].mean()
    gap = gie - arc
    verdict = "CONFIRMED" if gap >= 0.10 else "INCONCLUSIVE"
    return verdict, {
        "arcadia_avg_pr_auc": float(arc),
        "giessen_avg_pr_auc": float(gie),
        "gap": float(gap),
        "criterion": ">= 0.10",
    }


def _verdict_exp2(df: pd.DataFrame) -> tuple[str, dict]:
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return "INCONCLUSIVE (data missing)", {}
    naive = ok[ok["model"] == "cnn_1d_naive"]["test_pr_auc"].mean()
    ca_rows = ok[ok["model"].str.startswith("cnn_1d_contig_aware")]
    if ca_rows.empty:
        return "INCONCLUSIVE (no contig-aware results)", {}
    ca = ca_rows["test_pr_auc"].mean()
    delta = ca - naive
    verdict = "RECOVERED" if delta >= 0.05 else "INCONCLUSIVE"
    return verdict, {
        "naive_avg_pr_auc": float(naive),
        "contig_aware_avg_pr_auc": float(ca),
        "delta": float(delta),
        "criterion": ">= 0.05",
    }


def _verdict_exp3(df: pd.DataFrame) -> tuple[str, dict]:
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return "INCONCLUSIVE (data missing)", {}
    drops = {}
    for ds in ["arcadia", "giessen"]:
        for ab in ok["antibiotic"].unique():
            orig = ok[(ok["dataset"] == ds) & (ok["antibiotic"] == ab)
                      & (ok["variant"] == "orig")]
            shuf = ok[(ok["dataset"] == ds) & (ok["antibiotic"] == ab)
                      & (ok["variant"] == "shuf")]
            if len(orig) and len(shuf):
                drops[f"{ds}_{ab}"] = float(
                    orig["test_pr_auc"].iloc[0] - shuf["test_pr_auc"].iloc[0]
                )
    arc_drops = [v for k, v in drops.items() if k.startswith("arcadia")]
    gie_drops = [v for k, v in drops.items() if k.startswith("giessen")]
    if not arc_drops or not gie_drops:
        return "INCONCLUSIVE (missing dataset)", {"per_dataset_drops": drops}
    arc_avg = float(np.mean(arc_drops))
    gie_avg = float(np.mean(gie_drops))
    asym_ok = (gie_avg >= 0.05) and (arc_avg <= 0.02)
    verdict = "ASYMMETRIC" if asym_ok else "INCONCLUSIVE"
    return verdict, {
        "arcadia_avg_drop": arc_avg,
        "giessen_avg_drop": gie_avg,
        "per_dataset_drops": drops,
        "criterion": "Giessen >= 0.05 AND Arcadia <= 0.02",
    }


def _verdict_xgb(df: pd.DataFrame) -> tuple[str, dict]:
    ok = df[df["status"] == "ok"]
    if ok.empty or not {"no_phylogroup", "with_phylogroup"}.issubset(set(ok["variant"])):
        return "INCONCLUSIVE (data missing)", {}
    pivot = ok.pivot_table(index="antibiotic", columns="variant",
                            values="test_pr_auc")
    deltas = (pivot["with_phylogroup"] - pivot["no_phylogroup"]).to_dict()
    avg_delta = float(np.mean(list(deltas.values())))
    verdict = "PHYLOGROUP_SIGNAL" if avg_delta > 0 else "NO_SIGNAL"
    return verdict, {
        "per_antibiotic_delta": {k: float(v) for k, v in deltas.items()},
        "avg_delta": avg_delta,
    }


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def _draw_figure(
    df1: pd.DataFrame | None,
    df2: pd.DataFrame | None,
    df3: pd.DataFrame | None,
    df_xgb: pd.DataFrame | None,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Pillar 1 — Pan-Genome Locality Mismatch", fontsize=14,
                 fontweight="bold")

    # Panel 1: Exp 1 — bars per (dataset, antibiotic) with bootstrap CI
    ax = axes[0, 0]
    ax.set_title("Exp 1.1 — Naive 1D CNN on Giessen vs Arcadia")
    if df1 is not None and not df1.empty:
        ok = df1[df1["status"] == "ok"].copy()
        if not ok.empty:
            abs_ = list(ok["antibiotic"].unique())
            xs = np.arange(len(abs_))
            w = 0.38
            for i, ds in enumerate(["arcadia", "giessen"]):
                sub = ok[ok["dataset"] == ds].set_index("antibiotic")
                heights = [sub["test_pr_auc"].get(a, np.nan) for a in abs_]
                lows = [sub["test_pr_auc_lo"].get(a, np.nan) for a in abs_]
                highs = [sub["test_pr_auc_hi"].get(a, np.nan) for a in abs_]
                yerr = [
                    [h - lo if not np.isnan(lo) else 0 for h, lo in zip(heights, lows)],
                    [hi - h if not np.isnan(hi) else 0 for h, hi in zip(heights, highs)],
                ]
                ax.bar(xs + (i - 0.5) * w, heights, w,
                       yerr=yerr, label=ds.capitalize(),
                       capsize=3, alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels(abs_)
            ax.set_ylabel("Test PR-AUC")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Panel 2: Exp 2 — model comparison on Arcadia per antibiotic
    ax = axes[0, 1]
    ax.set_title("Exp 1.2 — Naive vs Contig-Aware on Arcadia")
    if df2 is not None and not df2.empty:
        ok = df2[df2["status"] == "ok"].copy()
        if not ok.empty:
            models = list(ok["model"].unique())
            abs_ = list(ok["antibiotic"].unique())
            xs = np.arange(len(abs_))
            w = 0.8 / max(1, len(models))
            for i, m in enumerate(models):
                sub = ok[ok["model"] == m].set_index("antibiotic")
                heights = [sub["test_pr_auc"].get(a, np.nan) for a in abs_]
                lows = [sub["test_pr_auc_lo"].get(a, np.nan) for a in abs_]
                highs = [sub["test_pr_auc_hi"].get(a, np.nan) for a in abs_]
                yerr = [
                    [h - lo if not np.isnan(lo) else 0 for h, lo in zip(heights, lows)],
                    [hi - h if not np.isnan(hi) else 0 for h, hi in zip(heights, highs)],
                ]
                ax.bar(xs + (i - (len(models) - 1) / 2) * w, heights, w,
                       yerr=yerr, label=m, capsize=3, alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels(abs_)
            ax.set_ylabel("Test PR-AUC")
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Panel 3: Exp 3 — asymmetric drop, orig vs shuf
    ax = axes[1, 0]
    ax.set_title("Exp 1.3 — Position Shuffling Drop")
    if df3 is not None and not df3.empty:
        ok = df3[df3["status"] == "ok"].copy()
        if not ok.empty:
            labels: list[str] = []
            orig_vals, shuf_vals = [], []
            for ds in ["arcadia", "giessen"]:
                for ab in ok["antibiotic"].unique():
                    o = ok[(ok["dataset"] == ds) & (ok["antibiotic"] == ab)
                           & (ok["variant"] == "orig")]
                    s = ok[(ok["dataset"] == ds) & (ok["antibiotic"] == ab)
                           & (ok["variant"] == "shuf")]
                    if len(o) and len(s):
                        labels.append(f"{ds[:3]}_{ab}")
                        orig_vals.append(float(o["test_pr_auc"].iloc[0]))
                        shuf_vals.append(float(s["test_pr_auc"].iloc[0]))
            xs = np.arange(len(labels))
            w = 0.4
            ax.bar(xs - w / 2, orig_vals, w, label="Original", alpha=0.85)
            ax.bar(xs + w / 2, shuf_vals, w, label="Shuffled", alpha=0.85)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels, rotation=30, ha="right")
            ax.set_ylabel("Test PR-AUC")
            ax.set_ylim(0, 1)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    # Panel 4: XGB phylogroup delta per antibiotic
    ax = axes[1, 1]
    ax.set_title("XGB Baseline — Phylogroup Δ PR-AUC")
    if df_xgb is not None and not df_xgb.empty:
        ok = df_xgb[df_xgb["status"] == "ok"].copy()
        if not ok.empty and {"no_phylogroup", "with_phylogroup"}.issubset(set(ok["variant"])):
            piv = ok.pivot_table(index="antibiotic", columns="variant",
                                  values="test_pr_auc")
            delta = piv["with_phylogroup"] - piv["no_phylogroup"]
            ax.bar(range(len(delta)), delta.values, color=[
                "tab:green" if v > 0 else "tab:red" for v in delta.values
            ])
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xticks(range(len(delta)))
            ax.set_xticklabels(delta.index, rotation=30, ha="right")
            ax.set_ylabel("Δ PR-AUC (with - without)")
            ax.grid(axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _md_table_exp1(df: pd.DataFrame) -> str:
    ok = df[df["status"] == "ok"].copy() if df is not None else None
    if ok is None or ok.empty:
        return "_(no data)_\n"
    cols = ["dataset", "antibiotic", "test_pr_auc", "test_pr_auc_lo",
            "test_pr_auc_hi", "test_mcc", "test_mcc_lo", "test_mcc_hi"]
    cols = [c for c in cols if c in ok.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in ok.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _md_table_xgb(df: pd.DataFrame) -> str:
    ok = df[df["status"] == "ok"].copy() if df is not None else None
    if ok is None or ok.empty:
        return "_(no data)_\n"
    cols = ["antibiotic", "variant", "test_pr_auc", "test_pr_auc_lo",
            "test_pr_auc_hi", "test_mcc", "test_f2", "test_sens_at_90_spec"]
    cols = [c for c in cols if c in ok.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in ok.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _write_summary_md(
    out_path: Path,
    df1: pd.DataFrame | None, df2: pd.DataFrame | None,
    df3: pd.DataFrame | None, df_xgb: pd.DataFrame | None,
    verdicts: dict, fig_rel: str,
) -> None:
    parts: list[str] = []
    parts.append("# Pillar 1 — Pan-Genome Locality Mismatch")
    parts.append("")
    parts.append("Generated by ``aggregate_pillar1_results.py``.\n")
    parts.append("## Headline Verdicts\n")
    parts.append("| Experiment | Verdict | Key Statistic | Criterion |")
    parts.append("|---|---|---|---|")

    v1 = verdicts.get("exp1", ("INCONCLUSIVE", {}))
    s1 = v1[1].get("gap", float("nan"))
    parts.append(
        f"| Exp 1.1 Naive CNN gap | **{v1[0]}** | "
        f"gap = {s1:+.4f} | {v1[1].get('criterion', '-')} |"
    )

    v2 = verdicts.get("exp2", ("INCONCLUSIVE", {}))
    s2 = v2[1].get("delta", float("nan"))
    parts.append(
        f"| Exp 1.2 Contig-aware recovery | **{v2[0]}** | "
        f"Δ = {s2:+.4f} | {v2[1].get('criterion', '-')} |"
    )

    v3 = verdicts.get("exp3", ("INCONCLUSIVE", {}))
    arc_d = v3[1].get("arcadia_avg_drop", float("nan"))
    gie_d = v3[1].get("giessen_avg_drop", float("nan"))
    parts.append(
        f"| Exp 1.3 Shuffle asymmetry | **{v3[0]}** | "
        f"Arc drop = {arc_d:+.4f}, Gie drop = {gie_d:+.4f} | "
        f"{v3[1].get('criterion', '-')} |"
    )

    v_xgb = verdicts.get("xgb", ("INCONCLUSIVE", {}))
    avg_d = v_xgb[1].get("avg_delta", float("nan"))
    parts.append(
        f"| XGB phylogroup baseline | **{v_xgb[0]}** | "
        f"avg Δ PR-AUC = {avg_d:+.4f} | Δ > 0 |"
    )
    parts.append("")
    parts.append(f"![overview]({fig_rel})\n")

    parts.append("## Exp 1.1 — Giessen vs Arcadia (Naive 1D CNN)\n")
    parts.append(_md_table_exp1(df1))
    parts.append("")

    parts.append("## Exp 1.2 — Contig-Aware vs Naive on Arcadia\n")
    parts.append(_md_table_exp1(df2))
    parts.append("")

    parts.append("## Exp 1.3 — Position Shuffling Drops\n")
    parts.append(_md_table_exp1(df3))
    parts.append("")

    parts.append("## XGB Baseline — With / Without Phylogroup\n")
    parts.append(_md_table_xgb(df_xgb))
    parts.append("")

    parts.append("## Methodological Implication\n")
    parts.append(
        "When all four verdicts above are positive (Exp 1.1 CONFIRMED, "
        "Exp 1.2 RECOVERED, Exp 1.3 ASYMMETRIC, XGB PHYLOGROUP_SIGNAL), "
        "the dataset-level evidence supports the *pan-genome locality "
        "mismatch* hypothesis: locality-assuming 1D CNNs are inappropriate "
        "for assembled pan-genome SNP matrices because adjacent columns "
        "no longer correspond to adjacent genomic positions. This justifies "
        "the gene-aware AMR-PhyloGene-MTL architecture proposed in Pillar 2."
    )
    parts.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")


# ---------------------------------------------------------------------------
# Paper outline draft
# ---------------------------------------------------------------------------
def _write_paper_outline(out_path: Path, verdicts: dict) -> None:
    v1 = verdicts.get("exp1", ("INCONCLUSIVE", {}))[1]
    v2 = verdicts.get("exp2", ("INCONCLUSIVE", {}))[1]
    v3 = verdicts.get("exp3", ("INCONCLUSIVE", {}))[1]
    v_xgb = verdicts.get("xgb", ("INCONCLUSIVE", {}))[1]

    text = f"""# AMR-PhyloGene-MTL — Paper Outline (auto-drafted)

## 1. Introduction
- Motivation: predicting AMR phenotypes from bacterial WGS.
- Existing CNN/Transformer baselines for SNP matrices implicitly assume
  spatial locality of input columns. This is valid for single-reference
  alignments but breaks for pan-genome representations where columns are
  ordered by contig assembly, not by chromosomal position.

## 2. Pan-Genome Locality Mismatch (Pillar 1 — Evidence)

### 2.1 Controlled CNN Comparison (Exp 1.1)
- Identical-architecture Naive 1D CNN trained with locked protocol on
  Giessen (single-reference, 60,936 SNPs) and Arcadia (pan-genome, 535,439
  SNPs across 1,271 contigs).
- Result: Giessen avg PR-AUC = {v1.get('giessen_avg_pr_auc', float('nan')):.4f},
  Arcadia avg PR-AUC = {v1.get('arcadia_avg_pr_auc', float('nan')):.4f}
  (gap = {v1.get('gap', float('nan')):+.4f}).
- Verdict: {verdicts.get('exp1', ('?', {}))[0]}.

### 2.2 Boundary-Aware Recovery (Exp 1.2)
- Per-contig CNN with attention aggregation tested on Arcadia.
- Δ PR-AUC vs naive = {v2.get('delta', float('nan')):+.4f}.
- Verdict: {verdicts.get('exp2', ('?', {}))[0]}.

### 2.3 Asymmetric Shuffling Drop (Exp 1.3)
- Single random column permutation applied to train/val/test of each dataset.
- Drop (orig - shuffled) on Giessen = {v3.get('giessen_avg_drop', float('nan')):+.4f};
  on Arcadia = {v3.get('arcadia_avg_drop', float('nan')):+.4f}.
- Verdict: {verdicts.get('exp3', ('?', {}))[0]}.

### 2.4 Phylogroup Signal Beyond PCA (XGB baseline)
- avg Δ PR-AUC (with phylogroup features) = {v_xgb.get('avg_delta', float('nan')):+.4f}.
- Verdict: {verdicts.get('xgb', ('?', {}))[0]}.

## 3. AMR-PhyloGene-MTL Architecture (Pillar 2)
- Within-gene 1D CNN, sparse top-K attention between genes, FiLM
  phylogroup conditioning, multi-task heads with soft co-resistance
  penalty (AMP→AMC, CTX→CAZ).
- Justified by Pillar 1 evidence: boundaries and phylogroup matter.

## 4. Biological Interpretation (Pillar 3)
- DeepSHAP attribution → gene-level aggregation → CARD database matching.

## 5. Generalisation (Pillar 4)
- Cross-dataset train/test (Arcadia ↔ Giessen) on common antibiotics
  (CIP, CTX, GEN, CAZ↔CTZ); per-phylogroup performance breakdown.

## 6. Statistical Rigor
- 1000× bootstrap CIs on all reported test metrics.
- Paired Wilcoxon between competing models across antibiotics.
- DeLong tests for paired ROC AUC differences where relevant.

## 7. Discussion
- Methodological contribution: locality mismatch as a general phenomenon
  in pan-genome ML.
- Biological contribution: gene-level attribution maps validated against
  CARD.
"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"),
        help="Path to the top-level results directory (default: ./results).",
    )
    parser.add_argument(
        "--paper-outline", type=Path, default=Path("paper/outline.md"),
        help="Path to write the paper outline draft.",
    )
    args = parser.parse_args()

    rd = args.results_dir
    df1 = _safe_read_csv(rd / "pillar1" / "exp1" / "exp1_results.csv")
    df2 = _safe_read_csv(rd / "pillar1" / "exp2" / "exp2_results.csv")
    df3 = _safe_read_csv(rd / "pillar1" / "exp3" / "exp3_results.csv")
    df_xgb = _safe_read_csv(rd / "baselines" / "xgb_phylogroup_results.csv")

    verdicts = {
        "exp1": _verdict_exp1(df1) if df1 is not None else ("INCONCLUSIVE", {}),
        "exp2": _verdict_exp2(df2) if df2 is not None else ("INCONCLUSIVE", {}),
        "exp3": _verdict_exp3(df3) if df3 is not None else ("INCONCLUSIVE", {}),
        "xgb": _verdict_xgb(df_xgb) if df_xgb is not None else ("INCONCLUSIVE", {}),
    }

    fig_path = rd / "pillar1" / "figures" / "pillar1_overview.png"
    _draw_figure(df1, df2, df3, df_xgb, fig_path)

    # Combined CSV
    combined_frames = []
    for name, d in [("exp1", df1), ("exp2", df2), ("exp3", df3),
                    ("baseline_xgb", df_xgb)]:
        if d is not None and not d.empty:
            d2 = d.copy()
            d2.insert(0, "source", name)
            combined_frames.append(d2)
    if combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True)
        combined.to_csv(rd / "pillar1" / "pillar1_combined.csv", index=False)

    _write_summary_md(
        rd / "pillar1" / "SUMMARY.md",
        df1, df2, df3, df_xgb,
        verdicts,
        fig_rel="figures/pillar1_overview.png",
    )

    _write_paper_outline(args.paper_outline, verdicts)

    # Console verdicts
    print("\n" + "=" * 80)
    print("  PILLAR 1 AGGREGATE VERDICTS")
    print("=" * 80)
    for k, (v, stats) in verdicts.items():
        print(f"  {k:<6}  →  {v}")
        for sk, sv in stats.items():
            if isinstance(sv, float):
                print(f"            {sk}: {sv:+.4f}")
            elif isinstance(sv, dict):
                continue
            else:
                print(f"            {sk}: {sv}")

    # JSON dump
    with open(rd / "pillar1" / "verdicts.json", "w") as f:
        json.dump({k: {"verdict": v[0], "stats": v[1]}
                   for k, v in verdicts.items()}, f, indent=2, default=str)

    print(f"\n  SUMMARY → {rd / 'pillar1' / 'SUMMARY.md'}")
    print(f"  FIGURE  → {fig_path}")
    print(f"  OUTLINE → {args.paper_outline}")


if __name__ == "__main__":
    main()
