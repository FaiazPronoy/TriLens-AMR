from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from trilens.experiments.near_duplicate_audit import (
    DUP_THRESHOLD,
    GENE_FILTERED_PKL,
    OUTPUT_DIR,
    RESULTS_DIR,
    _find_near_duplicates,
)

MLST_SPLIT_PKL = Path(__file__).resolve().parents[2] / "data" / "processed" / "mlst_lineage_holdout_split.pkl"
MLST_PREDICTIONS = RESULTS_DIR / "trilens_stack_lineage" / "predictions_saved.npz"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p, label in [(MLST_SPLIT_PKL, "MLST split"), (MLST_PREDICTIONS, "MLST final-stack predictions")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} ({label}) not found -- run the preceding MLST pipeline "
                f"steps first (lineage_holdout_split.py, then "
                f"stack_lineage.py)."
            )

    with open(GENE_FILTERED_PKL, "rb") as f:
        d = pickle.load(f)
    Y, mask, label_names = d["Y"], d["label_mask"], list(d["label_names"])
    X_snp = d["X_snp_filtered"]

    with open(MLST_SPLIT_PKL, "rb") as f:
        sp = pickle.load(f)
    train_idx = np.asarray(sp["train_idx"])
    st = np.asarray(sp["st_per_genome"], dtype=object)

    npz = np.load(MLST_PREDICTIONS, allow_pickle=True)
    test_idx = npz["test_indices"]
    test_probs = npz["test_ens_probs"]

    st_tr = {st[i] for i in train_idx if st[i] is not None}
    st_te = {st[i] for i in test_idx if st[i] is not None}
    assert not (st_tr & st_te), "TRAIN/TEST ST LEAK -- refusing to run dedup eval on a leaked split."
    print(f"ST disjointness re-verified. train={len(train_idx)} test={len(test_idx)}")

    dup_mask, exact_dist = _find_near_duplicates(X_snp, train_idx, test_idx)
    n_dup = int(dup_mask.sum())
    print(
        f"[mlst_lineage_holdout_split] {n_dup}/{len(test_idx)} test samples are "
        f"near-duplicates (<{DUP_THRESHOLD * 100:.1f}% SNP divergence from a "
        f"training genome) -- dropping them"
    )

    keep_mask = ~dup_mask
    dedup_test_idx = test_idx[keep_mask]

    Yt, Mt = Y[test_idx], mask[test_idx]
    rows = []
    for i, ab in enumerate(label_names):
        valid_full = Mt[:, i].astype(bool)
        valid_dedup = valid_full & keep_mask
        pr_before = float(average_precision_score(Yt[valid_full, i], test_probs[valid_full, i]))
        if valid_dedup.sum() >= 2 and len(np.unique(Yt[valid_dedup, i])) > 1:
            pr_after = float(average_precision_score(Yt[valid_dedup, i], test_probs[valid_dedup, i]))
        else:
            pr_after = float("nan")
        rows.append(
            {
                "split": "mlst_lineage_holdout_split",
                "antibiotic": ab,
                "pr_auc_before": pr_before,
                "pr_auc_after": pr_after,
                "delta": pr_after - pr_before if not np.isnan(pr_after) else float("nan"),
                "n_test_before": int(valid_full.sum()),
                "n_test_after": int(valid_dedup.sum()),
            }
        )

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "near_duplicate_results_lineage.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    np.save(OUTPUT_DIR / "mlst_dedup_test_idx.npy", dedup_test_idx)

    avg_before = df["pr_auc_before"].mean()
    avg_after = df["pr_auc_after"].mean()

    lines = [
        "# Deduplicated-test-set evaluation (MLST lineage holdout split)\n",
        f"{n_dup}/{len(test_idx)} test samples were flagged as near-duplicates of "
        f"a training genome (<{DUP_THRESHOLD * 100:.1f}% SNP divergence) and removed. "
        f"Training set unchanged.\n",
        f"Predictions source: {MLST_PREDICTIONS.relative_to(RESULTS_DIR.parent)} "
        f"-- the properly index-aligned final stack (5 classical + TriLens) from "
        f"stack_lineage.py, not an earlier intermediate.\n",
        f"**Avg PR-AUC before dedup: {avg_before:.4f}**  |  **after dedup: {avg_after:.4f}**  "
        f"|  **delta: {avg_after - avg_before:+.4f}**\n",
        "## Per-antibiotic\n",
        "| Antibiotic | PR-AUC before | PR-AUC after | Delta | N test before | N test after |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['pr_auc_before']:.4f} | {r['pr_auc_after']:.4f} | "
            f"{r['delta']:+.4f} | {r['n_test_before']} | {r['n_test_after']} |"
        )
    lines.append("")
    (OUTPUT_DIR / "SUMMARY_lineage.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY_lineage.md'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
