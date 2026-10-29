# TabNet Baseline -- MLST lineage holdout split

Same gene-aggregated features (random_split-ranked, matching the other 2 TabNet runs) and TabNet hyperparameters as `results/tabnet_random/` and `results/tabnet_phylogroup/` -- only the split differs. Total runtime: 57.5 min.

## Results

| Antibiotic | PR-AUC | MCC | Runtime (min) |
|---|---|---|---|
| AMP | 0.7269 | 0.3876 | 8.5 |
| CIP | 0.3524 | 0.2866 | 11.7 |
| AMC | 0.3463 | 0.1647 | 2.7 |
| SXT | 0.4676 | 0.2618 | 8.5 |
| GEN | 0.1232 | 0.1729 | 11.9 |
| CTX | 0.2755 | 0.0715 | 7.2 |
| CAZ | 0.0917 | 0.0097 | 7.0 |

**Average PR-AUC: 0.3405**

## Comparison across all 3 splits

| Antibiotic | random_split | phylogroup_split | mlst_lineage_holdout |
|---|---|---|---|
| AMP | 0.8912 | 0.8509 | 0.7269 |
| CIP | 0.8711 | 0.8925 | 0.3524 |
| AMC | 0.5685 | 0.6362 | 0.3463 |
| SXT | 0.6888 | 0.6814 | 0.4676 |
| GEN | 0.3744 | 0.4173 | 0.1232 |
| CTX | 0.7714 | 0.7971 | 0.2755 |
| CAZ | 0.5024 | 0.6598 | 0.0917 |