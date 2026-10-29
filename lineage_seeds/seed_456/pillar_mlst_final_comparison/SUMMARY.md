# FINAL: MLST lineage holdout vs the two earlier splits

| Antibiotic | Random split (T6) | Phylogroup-stratified (clean) | **MLST lineage holdout** |
|---|---|---|---|
| AMP | n/a | n/a | 0.9095 |
| CIP | n/a | n/a | 0.9164 |
| AMC | n/a | n/a | 0.4974 |
| SXT | n/a | n/a | 0.8129 |
| GEN | n/a | n/a | 0.2402 |
| CTX | n/a | n/a | 0.6095 |
| CAZ | n/a | n/a | 0.4624 |
| **AVERAGE** | n/a | n/a | **0.6355** |

## Component averages

- classical-only (5 bases), MLST holdout: **0.6221**
- classical-only (5 bases), phylogroup-stratified clean: 0.8522
- TriLens solo (no classical), MLST holdout: **0.4934**

All arms use the same gene-filtering method and the same meta-stacking protocol; only the split (and therefore the split-specific gene ranking) differs. ST disjointness between train and test was asserted at split construction, at classical training, at neural training, and again here.