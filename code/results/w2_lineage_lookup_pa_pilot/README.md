# W2 lineage-lookup baselines: P/A-only pilot

**P/A-only pilot.** Distance = Hamming on the 32,441 gene presence/absence columns, a stand-in for the SNP distance that runs on Kaggle. Pre-registration and reading: `docs/experiments/w2_lineage_lookup.md`.

Files: `lookup_results.csv` (every split × antibiotic × baseline), `classical_minus_lookup.csv` (S2, S3), `step1_cip_random/` (the first pilot step). `test_scores.npz` holds per-strain test scores for paired tests (not in git).

Test PR-AUC [95% CI] of the lookup chosen on validation (always kNN on P/A here), with the chosen k. '—' = antibiotic not usable on that split; † = secondary tier; ‡ = below the minimum counts.

| split | AMP | CIP | AMC | SXT | GEN | CTX | CAZ |
|---|---|---|---|---|---|---|---|
| S1_random | 0.820 [0.781, 0.856] k=50 | 0.889 [0.849, 0.924] k=20 | 0.607 [0.539, 0.673] k=5 | 0.632 [0.553, 0.707] k=50 | 0.458 [0.362, 0.566] k=3 | 0.723 [0.625, 0.809] k=10 | 0.644 [0.553, 0.726] k=10 |
| S2_phylogroup | 0.808 [0.766, 0.845] k=50 | 0.870 [0.822, 0.910] k=20 | 0.617 [0.552, 0.681] k=10 | 0.666 [0.584, 0.750] k=20 | 0.568 [0.470, 0.658] k=10 | 0.776 [0.709, 0.834] k=10 | 0.695 [0.619, 0.766] k=10 |
| S3_st_holdout | 0.582 [0.529, 0.639] k=50 | 0.242 [0.194, 0.295] k=5 | 0.313 [0.261, 0.381] k=50 | 0.429 [0.362, 0.506] k=20 | 0.083 [0.048, 0.166] k=20 | 0.212 [0.151, 0.297] k=10 | 0.198 [0.108, 0.327] k=20 |
| S4a_uk_to_norway | 0.670 [0.641, 0.696] k=10 | 0.576 [0.523, 0.631] k=3 | — | — | 0.172 [0.133, 0.218] k=5 | 0.231 [0.166, 0.304] k=10 | 0.148 [0.106, 0.202] k=5 |
| S4b_norway_to_uk | 0.817 [0.787, 0.847] k=20 | 0.816 [0.793, 0.838] k=20 | — | 0.932 [0.743, 0.966] k=5‡ | 0.369 [0.312, 0.430] k=10 | 0.524 [0.478, 0.569] k=50 | 0.423 [0.378, 0.474] k=50 |
| S5_time_norway | 0.741 [0.699, 0.780] k=20 | 0.608 [0.534, 0.679] k=5 | — | 0.529 [0.477, 0.586] k=50 | 0.314 [0.231, 0.413] k=50 | 0.337 [0.247, 0.440] k=5† | 0.354 [0.253, 0.473] k=20† |

Mean test PR-AUC over the usable (primary + secondary) antibiotics:

| split | ST lookup | phylogroup | kNN (P/A) | mean test prevalence |
|---|---:|---:|---:|---:|
| S1_random | 0.541 | 0.289 | 0.682 | 0.242 |
| S2_phylogroup | 0.564 | 0.287 | 0.715 | 0.247 |
| S3_st_holdout | 0.230 | 0.230 | 0.294 | 0.185 |
| S4a_uk_to_norway | 0.324 | 0.153 | 0.359 | 0.138 |
| S4b_norway_to_uk | 0.563 | 0.262 | 0.590 | 0.259 |
| S5_time_norway | 0.394 | 0.215 | 0.480 | 0.187 |

Classical arm (5 learners stacked) minus the best lookup, paired bootstrap, Holm across the 7 antibiotics of each split:

| split | antibiotic | classical | best lookup | Δ [95% CI] | Holm p |
|---|---|---:|---:|---:|---:|
| S2_phylogroup | AMP | 0.933 | 0.808 | +0.124 [+0.095, +0.157] | < 0.001 |
| S2_phylogroup | CIP | 0.961 | 0.870 | +0.091 [+0.061, +0.125] | < 0.001 |
| S2_phylogroup | AMC | 0.797 | 0.617 | +0.179 [+0.136, +0.221] | < 0.001 |
| S2_phylogroup | SXT | 0.828 | 0.666 | +0.162 [+0.086, +0.245] | < 0.001 |
| S2_phylogroup | GEN | 0.730 | 0.568 | +0.161 [+0.109, +0.216] | < 0.001 |
| S2_phylogroup | CTX | 0.878 | 0.776 | +0.101 [+0.069, +0.141] | < 0.001 |
| S2_phylogroup | CAZ | 0.840 | 0.695 | +0.145 [+0.105, +0.189] | < 0.001 |
| S3_st_holdout | AMP | 0.901 | 0.582 | +0.319 [+0.262, +0.368] | < 0.001 |
| S3_st_holdout | CIP | 0.916 | 0.242 | +0.674 [+0.611, +0.735] | < 0.001 |
| S3_st_holdout | AMC | 0.476 | 0.313 | +0.163 [+0.092, +0.239] | < 0.001 |
| S3_st_holdout | SXT | 0.774 | 0.429 | +0.345 [+0.274, +0.410] | < 0.001 |
| S3_st_holdout | GEN | 0.222 | 0.083 | +0.139 [+0.028, +0.268] | 0.009 |
| S3_st_holdout | CTX | 0.602 | 0.212 | +0.390 [+0.276, +0.492] | < 0.001 |
| S3_st_holdout | CAZ | 0.464 | 0.198 | +0.266 [+0.125, +0.387] | < 0.001 |
