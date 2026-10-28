# W14. Clinical error rates, and does a linear model do what XGBoost does?

**Status:** pre-registered 2026-10-29, before either part has been computed. Both parts re-use
outputs already on disk (W6 out-of-fold scores, W8 frozen external scores, the W13 EUCAST
labels); no model output of this experiment has been read.

## Why

Two reviews asked for what an AMR-genomics reader expects and the paper does not yet have:

1. **Clinical error rates.** Ranking metrics (nAP) say how well resistant isolates are ranked,
   not how often a call would be wrong. Diagnostic AMR work is judged by categorical agreement,
   major errors (a susceptible isolate called resistant) and very major errors (a resistant
   isolate called susceptible).
2. **Model complexity.** W10-B showed a model beats the determinant rule on unseen lineages
   (+0.122). A reviewer will ask whether that needs gradient boosting, or whether an ordinary
   sparse linear model on the same calls does it. If a linear model does, the paper's
   recommendation becomes simpler and stronger.

## Part A, clinical error rates on the external cohort

- **Cohort and labels:** the W8 primary cohort with **EUCAST v16.1 labels** from the MIC (W13);
  isolates whose EUCAST label is undetermined are excluded. CTX is excluded (under 1,000 MICs).
  Drugs: AMP, CIP, AMC, SXT, GEN, CAZ.
- **Arms and calls:** the frozen W8 XGBoost on all calls (AP), called resistant at score ≥ 0.5
  (as in the paper's clinical tables); the W8 determinant rule (already binary); and the Part B
  linear model on the same features at ≥ 0.5.
- **Definitions** (FDA, *Class II Special Controls Guidance Document: AST Systems*, 2009,
  glossary): major error = reference S, call R, as a share of reference-susceptible isolates;
  very major error = reference R, call S, as a share of reference-resistant isolates; categorical
  agreement = share of isolates where the call equals the reference. Intermediate is counted as
  resistant, as everywhere in the paper, so there are no minor errors.
- **Benchmarks** (same guidance, section X.A): categorical agreement > 89.9%; major error < 3%;
  very major error with an upper 95% confidence limit < 7.5% (Clopper–Pearson interval, as the
  guidance's binomial criterion). Each arm × drug is reported against these, **as context only**:
  these criteria were written for phenotypic devices against a reference method, not for genomic
  prediction, and the paper does not claim regulatory performance.
- **Primary comparison:** pooled over the six drugs, very major error rate of the model minus that
  of the rule, with a BioProject-clustered bootstrap CI (2,000 replicates). Reported in both
  directions; no pass rule, because either sign is informative. Major error difference likewise.

## Part B, a linear model

- **Model:** logistic regression, L1 penalty, `C = 1.0`, liblinear solver, binary features
  unscaled, **no tuning** — fixed here.
- **Internal (primary):** on the W6 ST-grouped 5-fold, the covered strains, the acquired-gene
  block (W6 arm A's features). Compared with W6's XGBoost arm A, from its saved out-of-fold scores,
  and with the rule. Endpoint: mean nAP over CTX, CAZ, GEN; difference XGBoost − linear with the
  ST-clustered CI.
- **External (secondary):** trained once on all called strains on all calls (W8 arm AP's
  features), frozen, applied to the W8 primary cohort with submitter labels; compared with W8's
  XGBoost AP and the rule, mean nAP over the six primary drugs, BioProject-clustered CI.
- **Rule for reading, fixed now:**
  - **a linear model suffices** if the XGBoost − linear CI lies wholly inside (−0.05, +0.05);
  - **non-linearity matters** if XGBoost − linear ≥ 0.05 with its CI above zero;
  - **the linear model is better** if linear − XGBoost ≥ 0.05 with its CI above zero;
  - **inconclusive** otherwise.
  Applied to the internal endpoint as primary and to the external one as secondary.

## Stated limits

- A fixed 0.5 threshold on an uncalibrated score is not how a deployed classifier would be set;
  it is a single, pre-stated operating point, used because the paper's clinical tables use it.
- The FDA criteria apply to phenotypic devices with a reference broth-microdilution method; here
  the "reference" is a submitter-measured MIC from mixed platforms, harmonised to EUCAST.
- L1 logistic regression with one fixed `C` is one linear model, not the best possible one.
