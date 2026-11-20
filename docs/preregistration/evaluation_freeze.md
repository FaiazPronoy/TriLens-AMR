# P2. Evaluation freeze, and the rule for an architecture claim

**Status:** pre-registered 2026-11-20, binding from this commit.

**What had been run when this was committed:** only P1 (determinant calls for all strains) and
its concordance check against NCBI (`results/determinant_matrix_bvbrc/`).

**What had not been run:** no arm described below has been trained, and no fold file exists yet.

The three choices that were open in the draft were confirmed by the authors before this commit:
- the neural arm is secondary and runs on repeat 1 only;
- Yu et al. is not an untouched test;
- the pass rule in section 8.

## Why

Model versions v7 to v10 were iterated on the random-split test set, and the lineage holdout has
been read many times (full stack 0.645 mean AP,
`results/trilens_stack_lineage/split_comparison.csv`). Neither is an untouched test any more.
The next phase adds a model contribution, so it needs one place to develop, one place to report,
and a test that nobody has looked at.

The literature read on 20 November 2026 (`next-phase/literature_notes.md`) fixes what the new
model has to beat. Under phylogeny-aware or homology-aware splits, a catalogue rule is the
strongest method in most datasets (Hu et al. 2024). W14 showed that a sparse linear model on the
same calls beats the rule here. **L1 logistic regression on the determinant calls is therefore
the baseline that matters.**

## 1. Three tiers

| Tier | Data | Used for | May be read |
|---|---|---|---|
| Development | repeated ST-grouped 5-fold on Arcadia, 5 repeats (section 2) | every design choice, tuning, feature selection, ablation | freely |
| Internal report | the existing lineage holdout (`results/lineage_holdout_split/split.json`), the random split, and the W8 external cohort | one number per frozen configuration | once per frozen configuration; the paper states that these were read before |
| Untouched test | CABBAGE E. coli, de-duplicated (section 6) | the generalisation claim | once, after the configuration is frozen and tagged |

## 2. Folds

- Sequence type (Achtman scheme, as in `split.json`) never crosses folds.
- The 475 strains without a sequence type are put in every training part and are never tested.
  The development tier therefore tests 6,487 strains, not 6,962. This number goes beside every
  development-tier result.
- Repeat seeds: 42, 123, 456, 789, 1011. Within each repeat, sequence types are assigned to the
  5 folds by shuffling with that seed, then balancing strain counts greedily.
- Fold files are generated once by a pipeline step, written to `results/evaluation_folds/`, and
  committed in the same commit as this document or before any arm is trained.
- Asserts in code: no sequence type in two folds of one repeat; every typed strain tested exactly
  once per repeat; every arm reads the same file.

## 3. Population and calls

- Determinant calls come from P1: AMRFinderPlus 4.2.7, database 2026-03-24.1, run on the BV-BRC
  assembly of every strain (`results/determinant_calls/`).
- **No new arm is trained until P1 has finished.** Every arm, the baselines included, then uses
  the same pinned calls on the same strains.
- A strain whose call failed after the script's retries is excluded from every arm, and its count
  is reported. P1 finished with 6,961 of 6,962 strains called. Strain 562.65585 (AMC_615) is
  excluded: BV-BRC serves 2 contigs and 1,063,417 bp for it, against 146 contigs and 5,437,129 bp
  in its own metadata.
- The NCBI Pathogen Detection calls (5,072 strains) are not used for training in this phase. They
  serve only for the concordance check in P1 and for the reproduction check in section 5.

## 4. Arms

All arms are fitted inside the training part of each fold. That covers feature ranking, scaling,
tuning, calibration, thresholds and the meta-learner. Tuning, where an arm has any, is nested
inside an ST-grouped inner 4-fold on the training part.

| Arm | Definition |
|---|---|
| **L1-calls** (baseline) | logistic regression, L1, `C = 1.0`, liblinear, binary features unscaled, no tuning; features are every acquired-gene and point-mutation column of the pinned calls with at least 5 carriers in the training part. This is the W14 linear model. |
| **Classical stack** (reference) | the five classical learners and logistic meta-learner of `step09`, as in the paper |
| **Rule** (reference) | the W10 determinant rule on the pinned calls |
| **Residual model** (new) | starts from the L1-calls logit as a fixed offset, fitted on the training part. A second learner, fitted on the same training part, adds a correction from the inputs below. A per-drug gate scales the correction between 0 and 1 and is fitted in the inner folds. |

Inputs to the residual model's correction:

- the calls as missing-indicator pairs: called present, called absent, or no call;
- the gene family and class back-off of each call (AMRFinderPlus hierarchy, experiment 3);
- AMC mechanism features that can be read from an assembly:
  - blaTEM promoter type, by Lartigue 2002 and Sutcliffe numbering: P3, Pa/Pb (32 C>T),
    P4 (162 G>T) or Pc/Pd (30_38 deletion), plus 175 G>A as its own indicator; "not assembled"
    is a separate level, not P3. In Davies et al. 2020, every non-P3 type was resistant (41 of 41
    with 32 C>T).
  - ampC promoter and attenuator, positions −42 to +37 (Jaurin numbering): one indicator per
    observed pattern with at least 5 carriers in the training part. No pattern is chosen in
    advance.
  - blaOXA-1, inhibitor-resistant TEM, and more than one blaTEM copy in the assembly;
- the presence/absence lens, reduced inside the training part.

Copy number from read depth is **not** an input. BV-BRC assemblies carry no read depth. It is
added only if contig coverage proves recoverable for every strain, and that is decided before
this document is committed.

The learner for the correction is chosen on the development tier only. The candidates are named
here so the choice is not open-ended: L2 logistic regression, gradient boosting, and a small MLP.

**Neural lenses (confirmed).** The existing neural branches (SNP motif and presence/absence
attention) are too expensive for 5 repeats × 5 folds on the free GPU. The proposal:

- The primary claim uses the residual model with CPU-trainable lenses only.
- The full TriLens neural stack enters as a secondary arm on repeat 1 only, with 1 seed: 5
  training runs, inside the budget of 8. It is reported descriptively and does not count toward
  the pass rule.

## 5. Reference checks, before any new arm is compared

1. L1-calls on the W6 ST-grouped folds, using the NCBI calls, must reproduce W14's internal
   linear arm. That arm was fitted on the acquired-gene block only (W6 arm A), so this check uses
   the same block. Target: mean nAP over CTX, CAZ, GEN of 0.741, within ±0.002 (CTX 0.808,
   CAZ 0.677, GEN 0.738; `results/linear_model_and_clinical_errors/b_metrics.csv`, rows
   `linear_A`).
2. The classical stack on the random split must reproduce 0.846 mean AP
   (`results/classical_stack_random/classical_stack_results.csv`).

If either fails, work stops and the difference is reported before anything else is run.

## 6. Untouched test: CABBAGE

- Source: EMBL-EBI AMR portal (`ftp.ebi.ac.uk/pub/databases/amr_portal`), E. coli, labels from
  the "Updated phenotype EUCAST" field (2025 breakpoints). Resistant and intermediate both count
  as resistant, as in the rest of the paper.
- Remove every BioSample that is in Arcadia or in the W8 cohort, and any isolate whose SRA run
  matches one of them.
- Recall genes with the pinned AMRFinderPlus on CABBAGE's assemblies, so frozen models see
  features in their own space. CABBAGE's own annotation (AMRFinderPlus v4.0.23, database
  2025-07-16.1, per its supplement S1.8) is not used.
- CABBAGE takes Arcadia's own source studies through BV-BRC (its Table S5 lists PMID 32205351),
  so the BioSample de-duplication against Arcadia is required, not optional.
- Keep a drug only if it has at least 100 resistant isolates after de-duplication. The counts are
  produced by a script that reads labels and identifiers only, never a model output, and they are
  added here as an amendment before the test is run.

**Second untouched test (confirmed: none).** Yu et al. 2025 cannot serve as one. Their Mendeley split
files (opened 20 November 2026) list 5,704 E. coli genomes by BV-BRC ID, and 2,973 of them (52%)
are Arcadia strains. For AMC, 2,710 of 2,870 are. The proposal is to drop it as an untouched test.

Experiment 9 then becomes descriptive:
- Use Yu's clade labels as an alternative split on the strains shared with Arcadia, for the drugs
  both cover (AMP, CIP, AMC, CTX, CAZ).
- Report the results beside the development tier.
- No claim rests on them.

## 7. Metrics and intervals

- **nAP per drug** (as in W6 to W14) and its mean, with AP beside it. AP and nAP are never mixed
  in one comparison.
- **Intervals:** 2,000 bootstrap replicates, resampling sequence types (development tier) or
  BioProjects (external tiers). Differences are paired: both arms are scored on the same
  resample of the same strains.
- **The population is stated with every number.**

## 8. Pass rule for the model claim (confirmed)

The claim does not require higher ranking accuracy. It requires two things on the hard drugs:
- ranking no worse than the baseline;
- fewer very major errors (resistant isolates called susceptible) at the same major-error rate.

Both are computed per repeat on the development tier, residual model minus L1-calls, over
**AMC, GEN, CTX, CAZ**.

These four drugs are fixed before any new result. They are the paper's existing "hard" group: on
the lineage holdout the full stack loses 0.251 (AMC), 0.454 (GEN), 0.258 (CTX) and 0.337 (CAZ)
mean AP, against at most 0.057 on AMP, CIP and SXT
(`results/trilens_stack_lineage/split_comparison.csv`).

**Operating point.** Each arm gets one threshold per drug. It is chosen inside the training part
only: the highest threshold whose major error on the inner out-of-fold predictions is at most 3%,
which is the FDA major-error limit. That threshold is then applied unchanged to the test fold.

**Endpoints.**
- **E1 (ranking):** difference in mean nAP over the four drugs.
- **E2 (very major error):** difference in pooled very major error. That is resistant isolate-drug
  pairs called susceptible, divided by all resistant pairs, over the four drugs.
- **E3 (major error):** difference in pooled major error. That is susceptible pairs called
  resistant, divided by all susceptible pairs.

Intervals are paired and come from 2,000 bootstrap replicates resampling sequence types.

**The claim "the model makes fewer dangerous errors than its baseline without ranking worse" is
made only if all five of these hold:**

1. **Ranking not worse (E1).** The lower bound of the E1 interval is above **−0.010**, in at least
   4 of the 5 repeats.
2. **Fewer very major errors (E2).** The upper bound of the E2 interval is below **0**, in at
   least 4 of the 5 repeats.
3. **Major error not traded away (E3).** The upper bound of the E3 interval is at most
   **+0.010**, in the same repeats as in 2.
4. **No harm elsewhere.** On AMP, CIP and SXT, the mean over repeats of the nAP difference is
   above **−0.010** for each drug.
5. **Transfer.** On CABBAGE, take the hard drugs that pass the 100-resistant filter. The frozen
   residual model minus the frozen L1-calls must have a point estimate below 0 for E2, and above
   −0.010 for E1. Thresholds are carried over from training, not refitted.

**Additional claim, fixed now.** "The model also ranks better" is made only if both hold in at
least 4 of the 5 repeats:
- the E1 point estimate is at least **+0.010**;
- the lower bound of the E1 interval is above 0.

The classical stack is also compared and reported. The claim does not require beating it.

If the rule fails, the paper reports the per-drug comparison and makes no model claim. No other
metric, drug subset, threshold rule or baseline is substituted afterwards.

**Secondary endpoints**, reported but without a claim attached:
- the seven-drug mean;
- calibration error;
- the descriptive neural arm.

## 9. Known bounds, stated in advance

- **AMC labels are noisy at the breakpoint.** In the Oxford cohort, 40 of 261 isolates (15%)
  were called both resistant and susceptible across repeat EUCAST agar dilution (Davies et al.
  2020). That cohort contributes 921 of Arcadia's AMC-tested strains (PMID 32205351,
  `results/data_inventory/by_study.csv`).
- **Part of AMC resistance is a lineage property.** Lineage alone moved 19% of isolates across
  the EUCAST breakpoint (Matlock et al. 2025). An ST-grouped split withholds that information by
  design.

- **Labels are heterogeneous across source studies.** Lipworth et al. 2024 (appendix, Table S1)
  record that the Norwegian collection was tested by Etest, then disc diffusion, then EUCAST disc
  diffusion, each read with its year's breakpoints. Arcadia's "unpublished:Norway" study (2,963
  strains) is probably that collection; this is checked by BioProject before the folds are
  committed.

These bound the reachable AMC gain. A small or absent AMC gain is therefore not evidence that the
features are wrong.

## 10. Freezing

- A configuration is frozen by a git tag, `freeze-<name>`.
- After the tag, nothing changes (code, features, hyperparameters) until the internal report and
  the untouched test have been read.
- An untouched test is read once. If the configuration changes after that, the paper says so, and
  the later number is labelled exploratory.

## 11. Compute

- Every CPU arm runs all 5 repeats.
- GPU is spent only on the secondary neural arm (5 runs) and on experiment 8.
- The total neural budget is 8 runs. Any run beyond it needs the authors' approval first.

## Amendments

None yet.
