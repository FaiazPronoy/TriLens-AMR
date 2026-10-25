"""
src/experiments/run_pillar2_layer_ablation.py
==============================================
Pillar 2 layer ablation — post-hoc assembly (Stages 2 & 3; Stage 1 optional).

Question: does each TriLens-MoE-Stack v10 (T6) component actually contribute
to the headline 0.8462 avg PR-AUC, or does it just add complexity?

Stage 1  Unweighted mean of the 3 expert logits (l_A, l_B, l_C), gate bypassed.
         Requires results/pillar2_layer_ablation/stage1_expert_only_preds.npz,
         produced by run_pillar2_stage1_expert_only.py (a separate, longer
         retraining job — see that file's docstring). If absent, Stage 1 is
         skipped and flagged as pending in SUMMARY.md.
Stage 2  MoE-gated fusion of the 3 experts — TriLens's own standalone output,
         before classical-base stacking. Read directly from
         results/trilens_attnfix/trilens_kfold_oof.npz (already computed by the
         original T6 run; no retraining needed).
Stage 3  Full T6 stack (Stage 2 + 5 classical bases + per-AB LogReg meta).
         Read directly from results/trilens_attnfix/ablation_results.csv
         (the existing headline number, T6 row, random_split) — not
         recomputed. results/trilens_attnfix/predictions_saved.npz is used only
         to add clinical (VME/ME) metrics at the final probability level.

All three stages are evaluated on the identical test set (test_idx from the
same random_split, verified below) so the comparison is apples-to-apples.

Usage:
    conda activate amr
    python -m src.experiments.run_pillar2_layer_ablation
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.experiments.run_posthoc_analysis import CLSI_VME_MAX, _clinical_at_threshold

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "pillar2_layer_ablation"
GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered.pkl"
# The main TriLens run. All three stages must come from the same run; the old
# default (trilens_v10, the pre-fix model) mixed pre-fix stages 1-2 with the
# delivered stage 3.
TRILENS_DIR = RESULTS_DIR / os.environ.get("TRILENS_MAIN_RUN", "trilens_attnfix")


def _load_labels():
    with open(GENE_FILTERED_PKL, "rb") as f:
        d = pickle.load(f)
    return d["Y"], d["label_mask"], list(d["label_names"])


def _per_ab_prauc(test_idx, test_preds, Y, mask, label_names):
    Yt, Mt = Y[test_idx], mask[test_idx]
    rows = {}
    for i, ab in enumerate(label_names):
        valid = Mt[:, i].astype(bool)
        yt, ys = Yt[valid, i], test_preds[valid, i]
        rows[ab] = float(average_precision_score(yt, ys))
    return rows


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Y, mask, label_names = _load_labels()

    rows = []
    reference_test_idx = None

    # ── Stage 2: TriLens standalone (gate + fusion, no classical stacking) ──
    s2_npz = np.load(TRILENS_DIR / "trilens_kfold_oof.npz")
    s2_test_idx = s2_npz["test_idx"]
    reference_test_idx = s2_test_idx
    s2_per_ab = _per_ab_prauc(s2_test_idx, s2_npz["test_preds"], Y, mask, label_names)
    for ab, pr in s2_per_ab.items():
        rows.append({"stage": "stage2_gate_fusion", "antibiotic": ab, "pr_auc": pr})
    print(f"Stage 2 (gate+fusion) avg PR-AUC: {np.mean(list(s2_per_ab.values())):.4f}")

    # ── Stage 3: full T6 headline (reused directly, not recomputed) ────────
    abl_df = pd.read_csv(TRILENS_DIR / "ablation_results.csv")
    t6 = abl_df[(abl_df["ablation"] == "T6") & (abl_df["split_key"] == "random_split")]
    s3_per_ab = dict(zip(t6["ab"], t6["prauc"]))
    for ab, pr in s3_per_ab.items():
        rows.append({"stage": "stage3_full_t6", "antibiotic": ab, "pr_auc": float(pr)})
    print(f"Stage 3 (full T6, from ablation_results.csv) avg PR-AUC: {np.mean(list(s3_per_ab.values())):.4f}")

    # Cross-check: recompute Stage 3 independently from predictions_saved.npz
    # to make sure the reused CSV number is internally consistent.
    s3_npz = np.load(TRILENS_DIR / "predictions_saved.npz", allow_pickle=True)
    s3_test_idx = s3_npz["test_indices"]
    assert np.array_equal(np.sort(s3_test_idx), np.sort(reference_test_idx)), (
        "Stage 3 test set does not match Stage 2 test set — not apples-to-apples!"
    )
    s3_recomputed = _per_ab_prauc(s3_test_idx, s3_npz["test_ens_probs"], Y, mask, label_names)
    for ab in label_names:
        if abs(s3_recomputed[ab] - s3_per_ab[ab]) > 1e-3:
            print(f"  WARNING: Stage 3 recomputed PR-AUC for {ab} "
                  f"({s3_recomputed[ab]:.4f}) differs from ablation_results.csv "
                  f"({s3_per_ab[ab]:.4f})")

    # ── Stage 1: unweighted expert average (optional — needs retrain) ──────
    stage1_npz_path = OUTPUT_DIR / "stage1_expert_only_preds.npz"
    s1_per_ab = None
    if stage1_npz_path.exists():
        s1_npz = np.load(stage1_npz_path)
        s1_test_idx = s1_npz["test_idx"]
        assert np.array_equal(np.sort(s1_test_idx), np.sort(reference_test_idx)), (
            "Stage 1 test set does not match Stage 2/3 test set — not apples-to-apples!"
        )
        s1_per_ab = _per_ab_prauc(s1_test_idx, s1_npz["test_preds"], Y, mask, label_names)
        for ab, pr in s1_per_ab.items():
            rows.append({"stage": "stage1_expert_avg", "antibiotic": ab, "pr_auc": pr})
        print(f"Stage 1 (expert avg, no gate) avg PR-AUC: {np.mean(list(s1_per_ab.values())):.4f}")
    else:
        print(f"Stage 1 pending — {stage1_npz_path} not found. "
              f"Run run_pillar2_stage1_expert_only.py first, then re-run this script.")

    # ── Save results CSV ─────────────────────────────────────────────────────
    out_df = pd.DataFrame(rows, columns=["stage", "antibiotic", "pr_auc"])
    out_df.to_csv(OUTPUT_DIR / "layer_ablation_results.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'layer_ablation_results.csv'}")

    # ── Stage 3 clinical metrics (VME/ME at threshold 0.5) ──────────────────
    clinical_rows = []
    Yt, Mt = Y[s3_test_idx], mask[s3_test_idx]
    for i, ab in enumerate(label_names):
        valid = Mt[:, i].astype(bool)
        yt, yp = Yt[valid, i], s3_npz["test_ens_probs"][valid, i]
        m = _clinical_at_threshold(yt, yp, 0.5)
        m["antibiotic"] = ab
        m["CLSI_ok"] = bool(m["VME"] <= CLSI_VME_MAX and m["ME"] <= CLSI_VME_MAX)
        clinical_rows.append(m)
    clinical_df = pd.DataFrame(clinical_rows)
    clinical_df.to_csv(OUTPUT_DIR / "stage3_clinical_metrics.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'stage3_clinical_metrics.csv'}")

    _write_summary(s1_per_ab, s2_per_ab, s3_per_ab, clinical_df, stage1_npz_path.exists())


def _write_summary(s1_per_ab, s2_per_ab, s3_per_ab, clinical_df, stage1_done: bool) -> None:
    label_names = list(s2_per_ab.keys())
    avg2 = float(np.mean(list(s2_per_ab.values())))
    avg3 = float(np.mean(list(s3_per_ab.values())))
    avg1 = float(np.mean(list(s1_per_ab.values()))) if s1_per_ab else None

    lines = []
    lines.append("# Pillar 2 — Layer Ablation: TriLens-MoE-Stack v10 (T6)\n")
    lines.append(
        "Isolates the contribution of each architectural layer: (1) the three "
        "experts averaged without gating, (2) the learned MoE gate fusing them "
        "(TriLens standalone), (3) the 5 classical base learners + meta-learner "
        "on top (full T6 headline). All stages evaluated on the identical "
        "test set (same `random_split`, seed=42, kfold_seed=42).\n"
    )
    lines.append("## Average PR-AUC by stage\n")
    lines.append("| Stage | Description | Avg PR-AUC | Δ vs previous stage |")
    lines.append("|---|---|---|---|")
    if stage1_done:
        lines.append(f"| Stage 1 | 3 experts, unweighted average (no gate) | {avg1:.4f} | — |")
        lines.append(f"| Stage 2 | + MoE gate (TriLens standalone) | {avg2:.4f} | {avg2 - avg1:+.4f} |")
        lines.append(f"| Stage 3 | + 5 classical bases + LogReg meta (T6 headline) | {avg3:.4f} | {avg3 - avg2:+.4f} |")
    else:
        lines.append("| Stage 1 | 3 experts, unweighted average (no gate) | *pending* | *pending* |")
        lines.append(f"| Stage 2 | + MoE gate (TriLens standalone) | {avg2:.4f} | *pending* |")
        lines.append(f"| Stage 3 | + 5 classical bases + LogReg meta (T6 headline) | {avg3:.4f} | {avg3 - avg2:+.4f} |")
    lines.append("")

    lines.append("## Per-antibiotic PR-AUC\n")
    header = "| Antibiotic | Stage 1 | Stage 2 | Stage 3 |"
    lines.append(header)
    lines.append("|---|---|---|---|")
    for ab in label_names:
        s1v = f"{s1_per_ab[ab]:.4f}" if s1_per_ab else "pending"
        lines.append(f"| {ab} | {s1v} | {s2_per_ab[ab]:.4f} | {s3_per_ab[ab]:.4f} |")
    lines.append("")

    lines.append("## Stage 3 clinical metrics (threshold = 0.5)\n")
    lines.append("| AB | VME% | ME% | CA% | CLSI (VME<3% and ME<3%) |")
    lines.append("|---|---|---|---|---|")
    n_pass = 0
    for _, r in clinical_df.iterrows():
        ok = "✅" if r["CLSI_ok"] else "❌"
        n_pass += int(r["CLSI_ok"])
        lines.append(f"| {r['antibiotic']} | {r['VME']*100:.2f} | {r['ME']*100:.2f} | {r['CA']*100:.2f} | {ok} |")
    lines.append("")
    lines.append(f"**CLSI criteria met (VME<3% and ME<3%): {n_pass}/{len(clinical_df)} antibiotics.**\n")

    lines.append("## Verdict\n")
    if stage1_done:
        gate_gain = avg2 - avg1
        stack_gain = avg3 - avg2
        lines.append(
            f"- The MoE gate adds **{gate_gain:+.4f}** avg PR-AUC over naive "
            f"unweighted expert averaging (Stage 1 → Stage 2).\n"
            f"- Classical-base stacking + meta-learner adds **{stack_gain:+.4f}** "
            f"avg PR-AUC over TriLens standalone (Stage 2 → Stage 3).\n"
        )
        if gate_gain > 0 and stack_gain > 0:
            lines.append("- **Both added components (gate, stacking) contribute positively — neither is pure complexity.**")
        elif gate_gain <= 0:
            lines.append("- **The learned gate does not clearly beat naive averaging on this test set — worth discussing in the defense.**")
        elif stack_gain <= 0:
            lines.append("- **Classical-base stacking does not clearly beat TriLens standalone on this test set — worth discussing in the defense.**")
    else:
        lines.append(
            f"- Stage 2 → Stage 3 (stacking gain): **{avg3 - avg2:+.4f}**.\n"
            f"- Stage 1 pending retrain (see run_pillar2_stage1_expert_only.py) — "
            f"gate contribution cannot be quantified yet."
        )

    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
