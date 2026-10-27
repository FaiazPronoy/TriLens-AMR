# TabNet Baseline (Iftu Sir)

Trained on identical gene-aggregated features (mean/var/n_nonref/max per gene, 5000 genes) and identical `random_split` as every other classical baseline in this project (stage2_gene_xgb, CascadeStack v9's XGB-gene/LGBM-gene/RF-gene/LogReg-gene) — directly comparable per-antibiotic PR-AUC. Total runtime: 54.6 min.

## Results

| Antibiotic | PR-AUC | MCC | Runtime (min) |
|---|---|---|---|
| AMP | 0.8912 | 0.6854 | 10.3 |
| CIP | 0.8711 | 0.8423 | 12.8 |
| AMC | 0.5685 | 0.3608 | 6.0 |
| SXT | 0.6888 | 0.4948 | 7.5 |
| GEN | 0.3744 | 0.3652 | 4.7 |
| CTX | 0.7714 | 0.6884 | 8.6 |
| CAZ | 0.5024 | 0.3809 | 4.7 |

**Average PR-AUC: 0.6668**
