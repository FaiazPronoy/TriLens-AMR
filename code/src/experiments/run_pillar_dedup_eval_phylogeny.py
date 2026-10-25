"""
src/experiments/run_pillar_dedup_eval_phylogeny.py
====================================================
Extends `run_pillar_dedup_eval.py`'s deduplicated-test-set evaluation to
`phylogeny_split` (T7), now that
`results/pillar_trilens_t7_predictions/predictions_saved.npz` exists (from
`run_trilens_t7_with_predictions.py`). This was explicitly called out as the
natural next step in `run_pillar_dedup_eval.py`'s own docstring and
SUMMARY.md output ("phylogeny_split — NOT done yet ... unless T7 is rerun
with predictions saving added") -- it now has been, so this runs it.

Does NOT modify run_pillar_dedup_eval.py -- imports and reuses its
`compute_dedup_eval()` function directly (already split-name/predictions-path
parameterized, no changes needed there), matching the pattern used elsewhere
in this project for phylogeny_split variants (run_pillar_cascadestack_phylogeny.py
alongside run_cascadestack_v9.py, run_trilens_t7_with_predictions.py alongside
run_trilens_v10.py).

CPU-only, no retraining -- safe to run alongside a GPU job.

Usage:
    conda activate amr
    python -m src.experiments.run_pillar_dedup_eval_phylogeny
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.run_pillar_dedup_eval import OUTPUT_DIR, RESULTS_DIR, compute_dedup_eval

T7_PREDICTIONS = RESULTS_DIR / "pillar_trilens_t7_predictions" / "predictions_saved.npz"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not T7_PREDICTIONS.exists():
        raise FileNotFoundError(
            f"{T7_PREDICTIONS} not found -- run "
            f"src.experiments.run_trilens_t7_with_predictions first."
        )

    result = compute_dedup_eval(
        split_name="phylogeny_split",
        predictions_npz_path=T7_PREDICTIONS,
    )
    df = pd.DataFrame(result["rows"])
    out_csv = OUTPUT_DIR / "dedup_eval_results_phylogeny.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")

    import numpy as np
    np.save(OUTPUT_DIR / "phylogeny_split_dedup_test_idx.npy", result["dedup_test_idx"])

    avg_before = df["pr_auc_before"].mean()
    avg_after = df["pr_auc_after"].mean()

    lines = ["# Deduplicated-test-set evaluation (T7, phylogeny_split)\n"]
    lines.append(
        f"{result['n_dup']}/{result['n_test_total']} test samples "
        f"(phylogeny_split) were flagged as near-duplicates of a training "
        f"genome (<0.5% SNP divergence) and removed. Training set unchanged.\n"
    )
    lines.append(
        f"Predictions source: {T7_PREDICTIONS.relative_to(RESULTS_DIR.parent)} "
        f"-- the Expert-A/C-leak-fixed T7 rerun, NOT the original locked "
        f"trilens_v10/ablation_results.csv T7 row. See that script's docstring "
        f"for the caveat about classical-base predictions still being loaded "
        f"from a random_split-trained file regardless of split_key.\n"
    )
    lines.append(f"**Avg PR-AUC before dedup: {avg_before:.4f}**  |  **after dedup: {avg_after:.4f}**  "
                 f"|  **delta: {avg_after - avg_before:+.4f}**\n")
    lines.append("## Per-antibiotic\n")
    lines.append("| Antibiotic | PR-AUC before | PR-AUC after | Delta | N test before | N test after |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r['antibiotic']} | {r['pr_auc_before']:.4f} | {r['pr_auc_after']:.4f} | "
            f"{r['delta']:+.4f} | {r['n_test_before']} | {r['n_test_after']} |"
        )
    lines.append("")
    (OUTPUT_DIR / "SUMMARY_phylogeny.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {OUTPUT_DIR / 'SUMMARY_phylogeny.md'}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
