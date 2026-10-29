# Deduplicated-test-set evaluation (MLST lineage holdout split)

0/1045 test samples were flagged as near-duplicates of a training genome (<0.5% SNP divergence) and removed. Training set unchanged.

Predictions source: results/trilens_stack_lineage/predictions_saved.npz -- the properly index-aligned final stack (5 classical + TriLens) from stack_lineage.py, not an earlier intermediate.

**Avg PR-AUC before dedup: 0.6454**  |  **after dedup: 0.6454**  |  **delta: +0.0000**

## Per-antibiotic

| Antibiotic | PR-AUC before | PR-AUC after | Delta | N test before | N test after |
|---|---|---|---|---|---|
| AMP | 0.9074 | 0.9074 | +0.0000 | 701 | 701 |
| CIP | 0.9183 | 0.9183 | +0.0000 | 913 | 913 |
| AMC | 0.5025 | 0.5025 | +0.0000 | 540 | 540 |
| SXT | 0.8011 | 0.8011 | +0.0000 | 517 | 517 |
| GEN | 0.2557 | 0.2557 | +0.0000 | 914 | 914 |
| CTX | 0.6465 | 0.6465 | +0.0000 | 808 | 808 |
| CAZ | 0.4863 | 0.4863 | +0.0000 | 912 | 912 |
