# W2 lineage-lookup baselines: P/A-only pilot (rebuilt presence/absence matrix)

Pre-registration: `docs/experiments/w2_lineage_lookup.md`. Distance = Hamming on the 32,441 gene presence/absence columns of the rebuilt matrix (KNOWN_ISSUES 12), with the 13 strains that have no presence/absence profile left out of every partition; a stand-in for the SNP distance that runs on Kaggle. Produced by `python -m src.experiments.run_lineage_lookup`.

k grid: [1, 3, 5, 10, 20, 50]; smoothing alpha = 1; bootstrap 2,000 resamples.


The first pilot, on the old minority-state matrix, is kept in `superseded_minority_state_matrix/`; what changed and why is in `docs/experiments/w2_lineage_lookup.md` ("Pilot redone on the rebuilt presence/absence matrix").

| split | ST lookup | phylogroup | kNN (P/A) | kNN, superseded run |
|---|---:|---:|---:|---:|
| S1_random | 0.541 | 0.288 | 0.684 | 0.682 |
| S2_phylogroup | 0.564 | 0.286 | 0.716 | 0.715 |
| S3_st_holdout | 0.230 | 0.230 | 0.310 | 0.294 |
| S4a_uk_to_norway | 0.325 | 0.154 | 0.365 | 0.359 |
| S4b_norway_to_uk | 0.563 | 0.262 | 0.590 | 0.590 |
| S5_time_norway | 0.395 | 0.216 | 0.481 | 0.480 |

| split | antibiotic | classical | best lookup | Δ [95% CI] | Holm p |
|---|---|---:|---:|---:|---:|
| S2_phylogroup | AMP | 0.932 | 0.808 | +0.124 [+0.093, +0.158] | < 0.001 |
| S2_phylogroup | CIP | 0.961 | 0.870 | +0.091 [+0.062, +0.124] | < 0.001 |
| S2_phylogroup | AMC | 0.797 | 0.621 | +0.176 [+0.134, +0.221] | < 0.001 |
| S2_phylogroup | SXT | 0.828 | 0.665 | +0.163 [+0.088, +0.245] | < 0.001 |
| S2_phylogroup | GEN | 0.730 | 0.570 | +0.160 [+0.111, +0.219] | < 0.001 |
| S2_phylogroup | CTX | 0.877 | 0.778 | +0.098 [+0.065, +0.137] | < 0.001 |
| S2_phylogroup | CAZ | 0.842 | 0.699 | +0.143 [+0.103, +0.189] | < 0.001 |
| S3_st_holdout | AMP | 0.901 | 0.556 | +0.345 [+0.291, +0.394] | < 0.001 |
| S3_st_holdout | CIP | 0.916 | 0.239 | +0.677 [+0.617, +0.733] | < 0.001 |
| S3_st_holdout | AMC | 0.476 | 0.322 | +0.154 [+0.080, +0.226] | < 0.001 |
| S3_st_holdout | SXT | 0.774 | 0.430 | +0.344 [+0.273, +0.408] | < 0.001 |
| S3_st_holdout | GEN | 0.222 | 0.082 | +0.140 [+0.027, +0.268] | 0.009 |
| S3_st_holdout | CTX | 0.602 | 0.340 | +0.262 [+0.159, +0.354] | < 0.001 |
| S3_st_holdout | CAZ | 0.464 | 0.200 | +0.264 [+0.125, +0.385] | < 0.001 |
