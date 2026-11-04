# FINAL: MLST lineage holdout vs the two earlier splits

| Antibiotic | Random split (full_stack) | Phylogroup-stratified (clean) | **MLST lineage holdout** |
|---|---|---|---|
| AMP | n/a | n/a | 0.9072 |
| CIP | n/a | n/a | 0.9214 |
| AMC | n/a | n/a | 0.4960 |
| SXT | n/a | n/a | 0.7973 |
| GEN | n/a | n/a | 0.2468 |
| CTX | n/a | n/a | 0.6284 |
| CAZ | n/a | n/a | 0.4681 |
| **AVERAGE** | n/a | n/a | **0.6379** |

## Component averages

- classical-only (5 bases), MLST holdout: **0.6221**
- classical-only (5 bases), phylogroup-stratified clean: 0.8522
- TriLens solo (no classical), MLST holdout: **0.4971**

All arms use the same gene-filtering method and the same meta-stacking protocol; only the split (and therefore the split-specific gene ranking) differs. ST disjointness between train and test was asserted at split construction, at classical training, at neural training, and again here.