# W2 lineage-lookup baselines (SNP distance)

Pre-registration: `docs/experiments/w2_lineage_lookup.md`. Distance = Hamming on the 535,439 SNP columns, every non-reference code counted as 'differs from reference' (`src/baselines/snp_distance.py`); all strains kept. Produced by `python -m src.experiments.run_lineage_lookup --distance snp --pilot`.

k grid: [1, 3, 5, 10, 20, 50]; smoothing alpha = 1; bootstrap 2,000 resamples.
