# Phylogroup-Only Baseline - Experiment Summary

Features: `phylogroup_onehot` only (6962 x 8), random_split, seed=42.

| Antibiotic | n_test | n_pos_test | LogReg PR-AUC | XGBoost PR-AUC |
|---|---|---|---|---|
| AMP | 731 | 366 | 0.5562 | 0.5562 |
| CIP | 910 | 198 | 0.2452 | 0.2449 |
| AMC | 565 | 175 | 0.3606 | 0.3620 |
| SXT | 526 | 165 | 0.4144 | 0.4124 |
| GEN | 910 | 94 | 0.1066 | 0.1065 |
| CTX | 808 | 105 | 0.2015 | 0.2015 |
| CAZ | 900 | 110 | 0.1551 | 0.1551 |

**Average LogReg PR-AUC:** 0.2914
**Average XGBoost PR-AUC:** 0.2912

This isolates how much predictive signal comes from population structure (phylogroup membership) alone, with no genomic feature content whatsoever.
