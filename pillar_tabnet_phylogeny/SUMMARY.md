# TabNet Baseline — phylogeny_split

Same gene-aggregated features and TabNet hyperparameters as `results/tabnet_baseline/` (random_split version), evaluated on `phylogeny_split` instead, for a leakage-aware TabNet-vs-T6/T7 comparison. Total runtime: 68.3 min.

## Results

| Antibiotic | PR-AUC | MCC | Runtime (min) |
|---|---|---|---|
| AMP | 0.8509 | 0.6650 | 9.9 |
| CIP | 0.8925 | 0.8168 | 12.4 |
| AMC | 0.6362 | 0.3956 | 9.9 |
| SXT | 0.6814 | 0.4840 | 5.8 |
| GEN | 0.4173 | 0.4658 | 11.4 |
| CTX | 0.7971 | 0.6949 | 8.2 |
| CAZ | 0.6598 | 0.5729 | 10.8 |

**Average PR-AUC: 0.7050**

## Comparison to TabNet on random_split

| Antibiotic | random_split PR-AUC | phylogeny_split PR-AUC | Delta |
|---|---|---|---|
| AMP | 0.8912 | 0.8509 | -0.0403 |
| CIP | 0.8711 | 0.8925 | +0.0215 |
| AMC | 0.5685 | 0.6362 | +0.0677 |
| SXT | 0.6888 | 0.6814 | -0.0074 |
| GEN | 0.3744 | 0.4173 | +0.0429 |
| CTX | 0.7714 | 0.7971 | +0.0257 |
| CAZ | 0.5024 | 0.6598 | +0.1574 |