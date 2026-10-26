# W1. Five-axis generalization test: pre-registration

**Status:** pre-registered, not run. Committed before any model is trained or scored on the new
splits (S4a, S4b, S5). The three existing splits (S1–S3) were already used in the thesis; their
numbers come from the current re-run.

## Question

How much of TriLens' performance survives when the test strains differ from the training strains
along each of five axes? The axes are random, phylogroup, sequence type, country and collection time.
Does the answer differ between the full model, its arms, its single learners and baselines that
cannot learn resistance mechanisms?

## Splits

The six splits of `docs/PLAN.md`:

- S1 random;
- S2 phylogroup-stratified;
- S3 ST holdout;
- S4a UK → Norway;
- S4b Norway → UK;
- S5 Norway ≤ 2013 → 2014–2017.

S4 and S5 are built by `code/src/splits/country_time.py` from `strains_metadata.csv`, with these
fixed rules:

- **S4a:** train side = all UK strains (England merged in). Test = all Norway strains. Strains of other
  or unknown countries are not used.
- **S4b:** the reverse.
- **S5:** Norway only. Train side = collection year ≤ 2013; test = 2014–2017.
- **Validation** = 15% of the train side, random with seed 42, stratified on the AMP label with
  "not tested" as its own stratum (the convention of the existing random split). Validation is used
  only for early stopping, choosing k and thresholds, and the gene ranking; never for a
  reported number.
- Asserted by tests (`code/tests/test_country_time_splits.py`):
  - train / val / test are disjoint;
  - every test strain has the test country (and, for S5, a year > 2013);
  - no train or val strain has the test country (S4) or a year > 2013 (S5);
  - the files are deterministic.

**Antibiotics per split:** those passing W0's pre-registered minimum (test ≥ 50 R and ≥ 50 S;
train ≥ 100 R and ≥ 100 S).

- S4a/S4b: AMP, CIP, GEN, CTX, CAZ.
- S5 primary: AMP, CIP, SXT, GEN. S5 secondary (train R between 50 and 99): CTX, CAZ. Secondary
  antibiotics are reported and marked, and kept out of the Holm families below.

**Gene filter:** refitted per split from that split's train + val (XGBoost gain on
χ²-prefiltered SNPs, as in `run_xgb_gene_importance_mlst.py`), 5,000 genes. Test never read.

## Models (identical hyperparameters on every split; none are tuned on any test set)

1. TriLens full stack (strain-conditioned Lens B, `TRILENS_LENSB=fixed`), seed 42;
2. classical arm alone (5 learners, LogReg meta-learner);
3. neural lenses alone (lenses + gate, 1-feature LogReg meta, as in `run_trilens_mlst.py`);
4. each classical learner solo: XGB-gene, LGBM-gene, RF-gene, LogReg-gene, XGB-χ²;
5. W2 baselines: ST lookup, kNN (SNP and P/A), phylogroup-only;
6. W5 rule baseline.

Hyperparameters are those of the delivered model and the current configs. The only per-split choices
are made on validation: early stopping, kNN's k and the tuned thresholds.

## Metrics (test side, per split × antibiotic)

| metric | detail |
|---|---|
| PR-AUC | average precision; 95% CI from 2,000 bootstrap resamples of test strains (percentile) |
| ROC-AUC | with DeLong 95% CI |
| PR-AUC, prevalence-matched | test negatives (or positives, if needed) subsampled to (a) the split's training prevalence and (b) the random split's test prevalence, 2,000 rounds; the mean and the 2.5–97.5% resampling range |
| VME, ME at 0.5 | VME = resistant called susceptible / resistant; ME = susceptible called resistant / susceptible |
| VME, ME at a tuned threshold | threshold maximising F2 on validation (primary); lowest threshold with validation ME ≤ 3% (secondary) |

Averages over antibiotics are reported without a CI (the per-drug resamples overlap). CLSI-style
limits for reference: VME ≤ 1.5%, ME ≤ 3%.

## Comparisons and statistics

- **Primary comparison:** TriLens full stack against (i) the best W2 lineage-lookup baseline of that
  split and antibiotic (best chosen on **validation** PR-AUC, not test), and (ii) the best single
  classical learner (chosen on validation).
- **Test:** paired bootstrap of ΔPR-AUC over the same test strains, 2,000 resamples, two-sided
  p = 2·min(share of resamples ≤ 0, share ≥ 0). ROC-AUC differences are also tested with paired DeLong.
- **Multiple comparisons:** Holm across the primary antibiotics of each split, separately for
  comparison (i) and (ii).

## Pass/fail rules (decided now)

| # | claim | passes if |
|---|---|---|
| P1 | TriLens learns more than lineage on a new lineage | on S3, TriLens > best lookup with Holm-adjusted p < 0.05 for **≥ 4 of 7** antibiotics |
| P2 | … and over time | on S5, the same for **≥ 2 of 4** primary antibiotics |
| P3 | … and across countries | on S4a and S4b, the same for **≥ 3 of 5** antibiotics in each direction |
| P4 | the hybrid is worth it under shift | on S3, S4 and S5, TriLens is not worse than the best single learner by more than 0.01 PR-AUC (upper end of the paired 95% CI of best − TriLens below 0.01) on a majority of the primary antibiotics |
| P5 | the five axes are not the same test | the per-antibiotic PR-AUC drop from S1 differs between axes (reported descriptively, with CIs; no test) |

If P1 fails, the paper's central claim ("beyond lineage") is not supported on the ST axis. Then the
paper is rewritten as a characterisation of where and why the model falls back on lineage, and the
owner decides. No re-tuning of the model is done to make P1 pass.

## Pilot

Before the GPU runs:

- the classical arm (XGB-χ² only) on S5, CIP, seed 42, CPU;
- sanity rule: test PR-AUC above the test prevalence and the run finishes without a leakage-check
  failure.

The pilot result is reported here, whatever it is.

## Compute

- `w1_gene_sets`: CPU, account B, about 12 h.
- `w1_classical`: GPU, account A, about 4.5 h.
- `w1_trilens`: GPU, account A, about 10.5 h.
- `w1_analysis`: CPU, about 1 h.

Total about 15 GPU h (estimates, see `docs/PLAN.md`).

## Deviations

Any change to this file after a test result exists is listed here with the date and the reason.

(none yet)
