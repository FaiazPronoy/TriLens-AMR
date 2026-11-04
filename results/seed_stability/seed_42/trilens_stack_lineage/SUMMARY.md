# FINAL: MLST lineage holdout vs the two earlier splits

| Antibiotic | Random split (full_stack) | Phylogroup-stratified (clean) | **MLST lineage holdout** |
|---|---|---|---|
| AMP | n/a | n/a | 0.9074 |
| CIP | n/a | n/a | 0.9183 |
| AMC | n/a | n/a | 0.5025 |
| SXT | n/a | n/a | 0.8011 |
| GEN | n/a | n/a | 0.2557 |
| CTX | n/a | n/a | 0.6465 |
| CAZ | n/a | n/a | 0.4863 |
| **AVERAGE** | n/a | n/a | **0.6454** |

## Component averages

- classical-only (5 bases), MLST holdout: **0.6221**
- classical-only (5 bases), phylogroup-stratified clean: 0.8522
- TriLens solo (no classical), MLST holdout: **0.5200**

All arms use the same gene-filtering method and the same meta-stacking protocol; only the split (and therefore the split-specific gene ranking) differs. ST disjointness between train and test was asserted at split construction, at classical training, at neural training, and again here.