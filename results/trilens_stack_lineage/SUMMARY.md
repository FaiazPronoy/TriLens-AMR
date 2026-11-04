# FINAL: MLST lineage holdout vs the two earlier splits

| Antibiotic | Random split (full_stack) | Phylogroup-stratified (clean) | **MLST lineage holdout** |
|---|---|---|---|
| AMP | 0.9431 | 0.9364 | 0.9074 |
| CIP | 0.9750 | 0.9613 | 0.9183 |
| AMC | 0.7539 | 0.8003 | 0.5025 |
| SXT | 0.8423 | 0.8385 | 0.8011 |
| GEN | 0.7093 | 0.7264 | 0.2557 |
| CTX | 0.9043 | 0.8816 | 0.6465 |
| CAZ | 0.8231 | 0.8484 | 0.4863 |
| **AVERAGE** | 0.8502 | 0.8561 | **0.6454** |

## Component averages

- classical-only (5 bases), MLST holdout: **0.6221**
- classical-only (5 bases), phylogroup-stratified clean: 0.8522
- TriLens solo (no classical), MLST holdout: **0.5200**

All arms use the same gene-filtering method and the same meta-stacking protocol; only the split (and therefore the split-specific gene ranking) differs. ST disjointness between train and test was asserted at split construction, at classical training, at neural training, and again here.