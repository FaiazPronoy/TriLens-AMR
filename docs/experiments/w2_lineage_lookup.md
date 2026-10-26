# W2. Lineage-lookup baselines: pre-registration

**Status:** pre-registered. Implementation (`code/src/baselines/lineage_lookup.py`) and the P/A-only CPU
pilot follow this commit. The SNP-distance kNN runs on Kaggle (`w2_snp_knn`).

## Question

How much of each model's score could be obtained by knowing only **who a strain's relatives are**? On
every split of W1, what does TriLens (and each of its parts) add above the best of these lookups?

## Baselines

All three use only training-side strains labelled for the antibiotic. None sees a test label.

### (a) ST lookup

Score of a test strain = resistant fraction of the training strains with the same Achtman ST,
smoothed towards the training prevalence p:

  score = (r_ST + α·p) / (n_ST + α), with α = 1.

- If the ST is unknown, or absent from training, the strain falls back to the same formula over its
  phylogroup, and then to p.
- The binary call ("majority label of the ST") is score ≥ 0.5.
- α = 1 is fixed here, not tuned.
- On S3 every test ST is absent from training by construction, so (a) reduces to the phylogroup
  fallback there. This is expected and reported as such.

### (b) kNN on genetic distance

Score = resistant fraction among the k nearest training strains (Hamming distance, unweighted, ties
broken by the lower strain index).

- **k is chosen per split and antibiotic on validation PR-AUC** from {1, 3, 5, 10, 20, 50}; the
  smallest k wins a tie. For the reported test score, the neighbours are searched among the training
  strains only (validation strains are not added), so test is scored exactly like validation.
- Two distance variants:
  - **P/A:** Hamming distance on the 32,441 gene presence/absence columns
    (`arcadia_v2_gene_presence_absence.pkl`). Hamming is unchanged by the column-orientation problem
    of W0 §4, so this variant is valid before that is fixed. CPU, here. **This is a stand-in.**
  - **SNP:** Hamming distance on the 535,439 SNP columns with every non-zero code counted as
    "differs from reference". Kaggle (`w2_snp_knn`). This is the variant used in the paper's
    headline table; the P/A variant is reported as a robustness check.

### (c) Phylogroup only

The existing `run_phylogroup_only_baseline.py` (logistic regression on the phylogroup one-hot) is the
paper's number on S1. For the other splits, the same idea is computed as the smoothed resistant
fraction of the phylogroup (formula of (a), α = 1). An unregularised one-hot logistic regression
converges to the unsmoothed fraction, so the two differ only for very small phylogroups. On S1
both are reported.

## Splits and antibiotics

S1–S5 with the antibiotics of `w1_five_axis.md`. All three baselines are cheap, so they also run on
S4/S5 antibiotics below the minimum. Those cells are marked and kept out of every test.

## Metrics and statistics

- As W1: PR-AUC with 95% bootstrap CI, ROC-AUC, prevalence-matched PR-AUC, VME/ME at 0.5.
- **Key table:** ΔPR-AUC = model − best lookup, per split × antibiotic. "Best lookup" is the baseline
  with the highest **validation** PR-AUC for that split and antibiotic.
- Paired bootstrap CI and Holm-adjusted p, as in W1.

## Pass/fail and sanity rules

These are for the baselines themselves. The model-level claims (P1–P3) are in `w1_five_axis.md`.

| # | rule | expected | if it fails |
|---|---|---|---|
| L1 | on S1, kNN (P/A) PR-AUC > phylogroup-only PR-AUC for ≥ 6 of 7 antibiotics | yes: relatives are informative within one population | implementation suspected; stop and debug before using any number |
| L2 | on S3, ST lookup = phylogroup fallback for 100% of test strains | yes, by construction | the split or the lookup is wrong (test in `code/tests/`) |
| L3 | every baseline's test PR-AUC is computed only from training labels | asserted in code and tests | blocker |

**Reported whatever the outcome:** how close the lookups come to TriLens on S1, i.e. how much of the
random-split score lineage alone explains.

## Pilot

- First: kNN (P/A) on S1, CIP only, seed-free (the method is deterministic). Pass = L1 holds for CIP.
- Then: all splits and antibiotics, P/A only.
- Comparison with models: where saved test predictions of the classical arm exist (S2 and S3 from
  the `outputs-classical` release, made on the clean phylogroup / holdout gene sets), the pilot
  also reports Δ = classical arm − best lookup. **These classical predictions are from a run made before
  the data repair** (they did not read the repaired file) and on a GPU XGBoost whose results drift
  slightly between runs. They are a pilot, replaced by the re-run.

Output: `code/results/w2_lineage_lookup_pa_pilot/`, every file marked "P/A-only pilot".

## Compute

- P/A variant: CPU, minutes, here.
- `w2_snp_knn`: CPU, Kaggle account B, about 1 h. It caches the 6,962 × 6,962 distance matrix once.

## Deviations

(none yet)
