# W2 lineage-lookup baselines: P/A-only pilot

Pre-registration: `docs/experiments/w2_lineage_lookup.md`. Distance = Hamming on the 32,441 gene presence/absence columns, a stand-in for the SNP distance that runs on Kaggle. Produced by `python -m src.experiments.run_lineage_lookup --pilot`.

k grid: [1, 3, 5, 10, 20, 50]; smoothing alpha = 1; bootstrap 2,000 resamples.
