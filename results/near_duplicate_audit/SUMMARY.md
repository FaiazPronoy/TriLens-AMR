# Deduplicated-test-set evaluation (full_stack, random_split)

47/1045 test samples (random_split) were flagged as near-duplicates of a training genome (<0.5% SNP divergence) and removed. Training set unchanged (leakage direction is test-contaminated-by-train, not the reverse).

**Avg PR-AUC before dedup: 0.8502**  |  **after dedup: 0.8457**  |  **delta: -0.0045**

## Per-antibiotic

| Antibiotic | PR-AUC before | PR-AUC after | Delta | N test before | N test after |
|---|---|---|---|---|---|
| AMP | 0.9431 | 0.9429 | -0.0002 | 731 | 728 |
| CIP | 0.9750 | 0.9734 | -0.0016 | 910 | 863 |
| AMC | 0.7539 | 0.7615 | +0.0076 | 565 | 519 |
| SXT | 0.8423 | 0.8427 | +0.0004 | 526 | 524 |
| GEN | 0.7093 | 0.7066 | -0.0027 | 910 | 863 |
| CTX | 0.9043 | 0.8922 | -0.0121 | 808 | 762 |
| CAZ | 0.8231 | 0.8002 | -0.0229 | 900 | 853 |

## phylogeny_split - NOT done yet (scope limitation found during implementation)

`results/trilens_stack_random/predictions_saved.npz` only covers `random_split` (full_stack). full_stack_phylogroup's (phylogeny_split) raw per-sample predictions were never saved to disk by `trilens_stack.py` - only its aggregate PR-AUC/MCC/CI are saved in `ablation_results.csv`. The phylogroup-stratified split is therefore not audited here.