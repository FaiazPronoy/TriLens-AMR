# W9. Does the lineage-holdout gain survive a change of seed?

**Status:** pre-registered 2026-10-28, before any new seed has been trained. Seed 42 has already
been read (the analysis notebook); seeds 123 and 456 have not been run.

## Why

The one claim the paper still makes for the neural half of the model is that on the ST-disjoint
lineage holdout the full stack (neural + classical) beats the classical arm alone: mean nAP
+0.028, ST-clustered 95% CI [0.006, 0.052] (`holdout_validity_deltas.csv`, seed 42). The interval
is narrow and close to zero, and it comes from a single network seed. The obvious reviewer
question is whether a different initialisation would erase it.

## Design

- `src.experiments.run_lineage_seeds`: the neural stack on the lineage holdout, exactly
  `src.experiments.run_trilens_mlst`, with `SEED` set to 123 and 456 (the seeds used for the
  random-split multi-seed script, chosen before this test). The trilens_splits run is seed 42.
- Only the network seed changes (weight initialisation, dropout, batch order). The K-fold partition
  (`kfold_seed` 42), the gene set, the hyperparameters and the classical arm are identical across
  seeds, so the test isolates seed variance of the neural branch, not of the whole pipeline.
- Each seed's neural arm is stacked with the same classical arm by
  `diagnose_mlst_final_comparison`, and read by `holdout_validity.py`
  (2,000 ST-clustered bootstrap replicates, same seed as before).

## Endpoint and rule

Stack minus classical, mean nAP over the seven drugs, per seed, with its ST-clustered CI.

- **Pass (claim kept as "stable"):** all three seeds read, the mean of the three deltas > 0, and
  the CI lower bound > 0 in at least 2 of the 3 seeds.
- **Fail:** the gain is reported as seed-dependent and is not claimed as a finding; the paper keeps
  the classical arm as the lineage-holdout reference.

Also reported, not tested: per-seed neural-only nAP and within-ST concordance, and the across-seed
SD of the delta.
