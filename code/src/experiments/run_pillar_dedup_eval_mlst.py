"""
src/experiments/run_pillar_dedup_eval_mlst.py
================================================
Extends the deduplicated-test-set evaluation to the MLST lineage holdout
split, completing the dedup-across-3-splits comparison (random_split,
phylogroup_split, mlst_lineage_holdout_split).

WHY THIS IS NOT A DIRECT CALL TO `run_pillar_dedup_eval.compute_dedup_eval()`
(unlike run_pillar_dedup_eval_phylogeny.py, which calls it unchanged):
that function does `d = _load_data()` (hardcoded to
`data/processed/arcadia_v2_gene_filtered.pkl`) then `split = d[split_name]`
-- it can only resolve a split that lives INSIDE that one pkl. The MLST
split lives in its own file (`data/processed/mlst_lineage_holdout_split.pkl`)
and was never written into arcadia_v2_gene_filtered.pkl (nor should it be --
that would violate the "never overwrite an existing file" rule). So this
script reuses `_find_near_duplicates()` -- the actual near-duplicate
detection algorithm, unchanged, same SNP feature space
(`arcadia_v2_gene_filtered.pkl`'s `X_snp_filtered`, matching what the other
two dedup evals used, for a like-for-like near-duplicate threshold across
all three) -- but supplies train_idx/test_idx from the MLST split file
directly instead of going through `compute_dedup_eval()`'s pkl-embedded-
split lookup. Everything else (DUP_THRESHOLD, SAMPLE_SCREEN_THRESHOLD,
N_SAMPLE_COLS, SEED) is imported unchanged from run_pillar_dedup_eval.py.

Uses `results/pillar_mlst_final_comparison/predictions_saved.npz` -- the
PROPERLY-ALIGNED final stack (5 classical + TriLens, index-verified) from
diagnose_mlst_final_comparison.py (Step 5), not an earlier intermediate
(e.g. TriLens-solo). This deliberately avoids the imprecision already
present in the existing phylogeny_split dedup eval, which had to use the
pre-alignment-fix T7 predictions because the properly-aligned version
didn't exist yet at the time it was run (see SESSION_HANDOFF.md).

Does not modify run_pillar_dedup_eval.py or any other existing file.
CPU-only, no retraining.

Usage:
    conda activate amr
    python -m src.experiments.run_pillar_dedup_eval_mlst
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.experiments.run_pillar_dedup_eval import (
    DUP_THRESHOLD,
    GENE_FILTERED_PKL,
    OUTPUT_DIR,
    RESULTS_DIR,
    _find_near_duplicates,
)

MLST_SPLIT_PKL = Path(__file__).resolve().parents[2] / "data" / "processed" / "mlst_lineage_holdout_split.pkl"
MLST_PREDICTIONS = RESULTS_DIR / "pillar_mlst_final_comparison" / "predictions_saved.npz"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for p, label in [(MLST_SPLIT_PKL, "MLST split"), (MLST_PREDICTIONS, "MLST final-stack predictions")]:
        if not p.exists():
            raise FileNotFoundError(
                f"{p} ({label}) not found -- run the preceding MLST pipeline "
                f"steps first (build_mlst_lineage_holdout_split.py, then "
                f"diagnose_mlst_final_comparison.py)."
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
    print(f"[mlst_lineage_holdout_split] {n_dup}/{len(test_idx)} test samples are "
          f"near-duplicates (<{DUP_THRESHOLD*100:.1f}% SNP divergence from a "
          f"training genome) -- dropping them")

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
        rows.append({
            "split": "mlst_lineage_holdout_split", "antibiotic": ab,
            "pr_auc_before": pr_before, "pr_auc_after": pr_after,
            "delta": pr_after - pr_before if not np.isnan(pr_after) else float("nan"),
            "n_test_before": int(valid_full.sum()), "n_test_after": int(valid_dedup.sum()),
        })

    df = pd.DataFrame(rows)
    out_csv = OUTPUT_DIR / "dedup_eval_results_mlst.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")
    np.save(OUTPUT_DIR / "mlst_dedup_test_idx.npy", dedup_test_idx)

    avg_before = df["pr_auc_before"].mean()
    avg_after = df["pr_auc_after"].mean()

    lines = [
        "# Deduplicated-test-set evaluation (MLST lineage holdout split)\n",
        f"{n_dup}/{len(test_idx)} test samples were flagged as near-duplicates of "
        f"a training genome (<{DUP_THRESHOLD*100:.1f}% SNP divergence) and removed. "
        f"Training set unchanged.\n",
        f"Predictions source: {MLST_PREDICTIONS.relative_to(RESULTS_DIR.parent)} "
        f"-- the properly index-aligned final stack (5 classical + TriLens) from "
        f"diagnose_mlst_final_comparison.py, not an earlier intermediate.\n",
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
    (OUTPUT_DIR / "SUMMARY_mlst.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY_mlst.md'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
