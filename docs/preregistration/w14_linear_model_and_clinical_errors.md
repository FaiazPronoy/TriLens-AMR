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
  very major error with an upper 95% confidence limit < 7.5% (Clopper-Pearson interval, as the
  guidance's binomial criterion). Each arm × drug is reported against these, **as context only**:
  these criteria were written for phenotypic devices against a reference method, not for genomic
  prediction, and the paper does not claim regulatory performance.
- **Primary comparison:** pooled over the six drugs, very major error rate of the model minus that
  of the rule, with a BioProject-clustered bootstrap CI (2,000 replicates). Reported in both
  directions; no pass rule, because either sign is informative. Major error difference likewise.

## Part B, a linear model

- **Model:** logistic regression, L1 penalty, `C = 1.0`, liblinear solver, binary features
  unscaled, **no tuning** - fixed here.
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

## Result (read 2026-10-29, after this pre-registration was committed at b118ef6)

### Part B, the linear model

**Internal (primary): a linear model suffices.** XGBoost − linear, mean nAP over CTX, CAZ, GEN on
the ST-grouped 5-fold: **−0.011 [−0.026, 0.007]**, wholly inside (−0.05, 0.05). Per drug: CTX
−0.026, CAZ −0.008, GEN 0.000, CIP −0.003. Linear − rule: **+0.137 [0.097, 0.203]** (XGBoost − rule
was +0.122 in W10). The margin over the rule on unseen lineages does not need gradient boosting.

**External (secondary): inconclusive by the rule, with the linear model ahead.** XGBoost − linear
over the six primary drugs: **−0.038 [−0.063, −0.019]**. The interval excludes zero but the
difference is under the pre-set 0.05 margin, so the pre-registered reading is "inconclusive"; the
direction is stated plainly: the linear model transfers better. Per drug, XGBoost is behind the
linear model on six of seven (CIP −0.027, AMC −0.019, SXT −0.020, GEN −0.092, CTX −0.222,
CAZ −0.073; AMP +0.001).

**Linear − rule, external: +0.104 [0.060, 0.166]**, against W8's XGBoost − rule +0.065
[0.005, 0.140]. Per drug the linear model is at or above the rule everywhere: AMP +0.018,
CIP +0.201, AMC +0.233, SXT +0.084, GEN +0.015, CTX +0.037, CAZ +0.069; the CI excludes zero for
CIP, AMC, SXT, GEN. External mean nAP: linear 0.849, XGBoost 0.785, rule 0.755.

So the two drugs where W8 found the rule ahead of the model (GEN −0.077, CTX −0.185) are
**XGBoost failures, not a failure of a learned model**: the sparse linear model on the same calls
is ahead of the rule on both (GEN +0.015 [0.004, 0.036]; CTX +0.037 [−0.033, 0.117]). This changes
the paper's recommendation from "a determinant rule is hard to beat" to "a sparse linear model on
the determinant calls beats the rule, and a more flexible model transfers worse".

### Part A, clinical error rates (EUCAST labels, threshold 0.5)

Pooled over the six drugs (11,601 resistant and 31,982 susceptible isolate-drug pairs):

| arm | CA | ME | VME | VME upper 95% |
|---|---|---|---|---|
| XGBoost | 0.914 | 2.7% | 25.1% | 25.9% |
| linear | 0.938 | 1.7% | 18.7% | 19.4% |
| rule | 0.942 | 2.7% | 14.4% | 15.1% |

**No arm meets the FDA very-major-error benchmark** (upper limit < 7.5%) pooled, and per drug
only the rule for ciprofloxacin does (2.6%, upper 3.5%). Categorical agreement and major error
are within the benchmarks pooled for all three arms; AMC and CAZ fail them for every arm.

Paired, BioProject-clustered: VME XGBoost − rule **+10.7 points [3.2, 18.1]**; linear − rule
**+4.3 [0.6, 8.4]**; XGBoost − linear +6.5 [2.1, 10.9]. ME linear − rule −1.0 [−4.1, 0.2];
XGBoost − rule 0.0 [−3.5, 1.2].

Read together with Part B: at the paper's fixed 0.5 threshold both learned models miss more
resistant isolates than the rule, while ranking them better (higher nAP). That is an operating-
point problem the threshold was not chosen to solve (stated limit above); the paper reports it
as it stands and does not move the threshold after seeing it. The clinically relevant
statement is the one the literature also makes: **very major errors are the unsolved problem for
genomic prediction**, for rules and models alike (AMC VME 35-50% in every arm here; compare
Lipworth et al. 2024).
