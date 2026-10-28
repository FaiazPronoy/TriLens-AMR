# FINAL: MLST lineage holdout vs the two earlier splits

| Antibiotic | Random split (T6) | Phylogroup-stratified (clean) | **MLST lineage holdout** |
|---|---|---|---|
| AMP | 0.9508 | 0.9402 | 0.9094 |
| CIP | 0.9751 | 0.9599 | 0.9163 |
| AMC | 0.7558 | 0.8030 | 0.5041 |
| SXT | 0.8505 | 0.8431 | 0.7908 |
| GEN | 0.7093 | 0.7230 | 0.2449 |
| CTX | 0.9020 | 0.8805 | 0.6200 |
| CAZ | 0.8248 | 0.8475 | 0.4936 |
| **AVERAGE** | 0.8526 | 0.8567 | **0.6399** |

## Component averages

- classical-only (5 bases), MLST holdout: **0.6221**
- classical-only (5 bases), phylogroup-stratified clean: 0.8522
- TriLens solo (no classical), MLST holdout: **0.5108**

All arms use the same gene-filtering method and the same meta-stacking protocol; only the split (and therefore the split-specific gene ranking) differs. ST disjointness between train and test was asserted at split construction, at classical training, at neural training, and again here.