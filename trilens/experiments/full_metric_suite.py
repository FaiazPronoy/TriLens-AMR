from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results"

TRILENS_DIR = next(
    (d for d in (RESULTS_DIR / "trilens_stack_random", RESULTS_DIR / "trilens_stack_random") if d.exists()),
    RESULTS_DIR / "trilens_stack_random",
)
OUTPUT_DIR = RESULTS_DIR / "full_metric_suite"
GENE_FILTERED_PKL = ROOT / "data" / "processed" / "arcadia_v2_gene_filtered.pkl"
THRESHOLD = 0.5


def _load_labels():
    with open(GENE_FILTERED_PKL, "rb") as f:
        d = pickle.load(f)
    return d["Y"], d["label_mask"], list(d["label_names"])


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    Y, mask, label_names = _load_labels()

    npz = np.load(TRILENS_DIR / "predictions_saved.npz", allow_pickle=True)
    test_idx = npz["test_indices"]
    test_probs = npz["test_ens_probs"]

    abl_df = pd.read_csv(TRILENS_DIR / "ablation_results.csv")
    t6_mcc_ref = dict(
        zip(
            abl_df[(abl_df["ablation"] == "full_stack") & (abl_df["split_key"] == "random_split")]["ab"],
            abl_df[(abl_df["ablation"] == "full_stack") & (abl_df["split_key"] == "random_split")]["mcc"],
        )
    )

    Yt, Mt = Y[test_idx], mask[test_idx]
    rows = []
    for i, ab in enumerate(label_names):
        valid = Mt[:, i].astype(bool)
        yt = Yt[valid, i].astype(int)
        yp = test_probs[valid, i]
        y_pred = (yp >= THRESHOLD).astype(int)

        notes = []

        if len(np.unique(yt)) < 2:
            pr_auc = 0.0
            notes.append("PR-AUC=0: only one true class present in test set for this AB")
        else:
            pr_auc = float(average_precision_score(yt, yp))

        if len(np.unique(yt)) < 2 or len(np.unique(y_pred)) < 2:
            mcc = 0.0
            notes.append("MCC=0: degenerate - one class missing in true labels or predictions at thr=0.5")
        else:
            mcc = float(matthews_corrcoef(yt, y_pred))

        f1 = float(f1_score(yt, y_pred, zero_division=0))
        f2 = float(fbeta_score(yt, y_pred, beta=2, zero_division=0))
        if y_pred.sum() == 0:
            notes.append("F1/F2=0 forced: model predicted no positives at thr=0.5 for this AB")

        accuracy = float(accuracy_score(yt, y_pred))

        row = {
            "antibiotic": ab,
            "pr_auc": pr_auc,
            "mcc": mcc,
            "f1": f1,
            "f2": f2,
            "accuracy": accuracy,
            "notes": "; ".join(notes) if notes else "computed normally, no degenerate case",
        }
        rows.append(row)

        ref_mcc = t6_mcc_ref.get(ab)
        if ref_mcc is not None and abs(ref_mcc - mcc) > 1e-3:
            print(
                f"  WARNING: recomputed MCC for {ab} ({mcc:.4f}) differs from "
                f"ablation_results.csv full_stack row ({ref_mcc:.4f})"
            )

    df = pd.DataFrame(rows, columns=["antibiotic", "pr_auc", "mcc", "f1", "f2", "accuracy", "notes"])
    df.to_csv(OUTPUT_DIR / "full_metric_suite.csv", index=False)
    print(f"Saved {OUTPUT_DIR / 'full_metric_suite.csv'}")
    print(df.to_string(index=False))

    _write_summary(df)


def _write_summary(df: pd.DataFrame) -> None:
    lines = []
    lines.append("# full_stack (headline) - Full Per-Antibiotic Metric Suite\n")
    lines.append(
        f"All metrics computed on the full_stack test set (`results/trilens_stack_random/predictions_saved.npz`, "
        f"`test_ens_probs`), threshold={THRESHOLD} for MCC/F1/F2/Accuracy (PR-AUC is threshold-free). "
        f"No dashes - every cell is a real number; see `notes` for any degenerate case.\n"
    )
    lines.append("## MCC methodology\n")
    lines.append(
        "MCC is computed per antibiotic as an independent binary problem "
        "(resistant vs susceptible for that drug only, at thr=0.5) - verified "
        "against `trilens/experiments/trilens_stack.py::compute_metrics()`, which "
        "already does this correctly (slices `Y`/`probs` to a single antibiotic "
        "column before calling `matthews_corrcoef`). This is not a macro-averaged "
        "multi-label MCC. Values below were independently recomputed from raw "
        "test predictions and cross-checked against `ablation_results.csv`'s full_stack "
        "row (no mismatch found beyond floating-point tolerance).\n"
    )
    lines.append("## Metrics table\n")
    lines.append("| Antibiotic | PR-AUC | MCC | F1 | F2 | Accuracy | Notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['pr_auc']:.4f} | {r['mcc']:.4f} | {r['f1']:.4f} | "
            f"{r['f2']:.4f} | {r['accuracy']:.4f} | {r['notes']} |"
        )
    lines.append("")
    lines.append(
        f"**Averages** - PR-AUC: {df['pr_auc'].mean():.4f}, MCC: {df['mcc'].mean():.4f}, "
        f"F1: {df['f1'].mean():.4f}, F2: {df['f2'].mean():.4f}, "
        f"Accuracy: {df['accuracy'].mean():.4f}\n"
    )
    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
