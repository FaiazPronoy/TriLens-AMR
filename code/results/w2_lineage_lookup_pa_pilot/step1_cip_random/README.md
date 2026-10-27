# W2 lineage-lookup baselines: P/A-only pilot

Pre-registration: `docs/experiments/w2_lineage_lookup.md`. Distance = Hamming on the 32,441 gene presence/absence columns of the rebuilt matrix (KNOWN_ISSUES 12), with the 13 strains that have no presence/absence profile left out of every partition; a stand-in for the SNP distance that runs on Kaggle. Produced by `python -m src.experiments.run_lineage_lookup --pilot`.

k grid: [1, 3, 5, 10, 20, 50]; smoothing alpha = 1; bootstrap 2,000 resamples.
