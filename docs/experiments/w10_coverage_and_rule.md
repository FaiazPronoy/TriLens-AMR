# W10. Is the ARG rescue a coverage artefact, and does a fixed rule do the same job?

**Status:** pre-registered 2026-10-29, before any arm of this experiment has been fitted. The
feasibility counts in the amendment below are strain counts and label prevalences only; no W10
score has been computed.

## Why

Nine independent reviews of the W6 and W8 package named the same two objections, and they are the
only ones that can still change the paper's conclusion.

1. **Coverage.** The Pathogen Detection calls cover 5,072 of 6,962 strains (72.9%), and the
   uncovered 1,890 are not missing at random. W6's rescue is therefore demonstrated on a subset
   chosen by annotation availability. If the catalogue-only collapse on CTX/CAZ/GEN is peculiar to
   that subset, the rescue is partly selection.
2. **The rule.** Externally, the model beats the pre-registered determinant rule by only
   +0.065 nAP [0.005, 0.140], and +0.021 [−0.049, 0.116] with intermediates excluded. Nobody has
   yet asked the same question *internally*: on the ST-grouped folds, does the rule already reach
   the rescued level (0.726)? If it does, "the catalogue omits determinants" is better stated as
   "a determinant rule recovers what the catalogue misses", and the model has no internal role
   either.

## Design

Both parts reuse the W6 machinery (`src.experiments.run_arg_rescue`), the same ST-grouped and
random 5-fold splits, the same XGBoost settings and the same nAP and ST-clustered bootstrap. No
hyperparameter is tuned here.

### Part A, the uncovered strains

Arms F0 (catalogue presence/absence) and M (metadata) are fitted on the **uncovered** strains
alone — they need no ARG call — with ST-grouped 5-fold over that subset, and compared with the
same two arms on the covered strains. The question is whether the pre-rescue collapse that W6
rescued is present in both strata.

Also reported, descriptive: country, study, ST count, phylogroup and per-drug label count and
prevalence in each stratum.

### Part B, the rule as an internal arm

Arm **R** is the W8 determinant rule (`src.experiments.run_external.RULES`, fixed on 2026-10-28
and unchanged here), scored on the covered strains with no training at all, next to F0 and F0A on
the identical folds and strains.

### Part C, stratified W6

The W6 primary endpoint (F0A − F0, mean nAP over CTX, CAZ, GEN, ST-grouped) recomputed within the
United Kingdom only and within Norway only, from the already-saved out-of-fold scores — a
re-scoring, not a refit. This asks whether the rescue holds inside a single country rather than
across the UK/Norway contrast.

## Endpoints and rules

**A, primary.** Mean nAP over CTX, CAZ, GEN of arm F0, uncovered stratum, against the same figure
for the covered stratum (0.450 as logged).

- **Pass (the rescue is not a coverage artefact):** the uncovered stratum's F0 figure is also low,
  within 0.10 of the covered stratum's.
- **Fail:** the uncovered strains are predicted substantially better from the catalogue alone
  (more than 0.10 higher). Then the collapse W6 rescued is specific to the annotated subset, and
  the paper must say the rescue is demonstrated on that subset and may not extend to the rest.

**B, primary.** Mean nAP over CTX, CAZ, GEN of arm R against arm F0A, ST-grouped, with the
ST-clustered paired CI.

- **The model adds something over the rule internally** if F0A − R ≥ 0.05 with the CI above zero.
- **It does not** otherwise. Then the paper reports the rescue as "a determinant rule recovers it",
  the modelling claim is dropped internally as well as externally, and the recommendation becomes
  to run the rule as a baseline rather than to fit a model. This outcome is reported either way; it
  is written here so it cannot be reframed after the fact.

Secondary, no pass/fail: R per drug against F0A per drug, and R on the random 5-fold.

**C, no pass/fail.** Reported as a sensitivity. A rescue that holds in only one country is stated
as such.

## Stated limits, whatever the result

- Part A compares two different strain sets, so a difference can be population difference rather
  than a coverage effect. It cannot separate the two, and the strata differ in country and study
  composition by construction (see the amendment).
- The uncovered stratum has few labels for AMP and SXT, so A is reported for CTX, CAZ, GEN and CIP
  only.
- Part B's rule was written against the external cohort's drugs and gene symbols. It is applied
  here unchanged, which is the honest comparison but may favour the model if our strains carry
  determinants the rule's patterns do not name.
- "Covered" means Pathogen Detection has a call for that strain; a strain with no call is not a
  strain with no resistance gene. Negatives inside a covered strain's row are real negatives, but
  an uncovered strain is unknown, not negative, and is never treated as negative.

## Amendment 2026-10-29, feasibility counts, before any arm is fitted

Coverage by stratum: covered 5,072, uncovered 1,890 (344 and 252 distinct STs).

- **Study 28720578 is 0% covered: all 1,439 of its strains lack a call.** They are therefore
  already absent from every W6 arm, so excluding that study — which several reviews asked for —
  changes no W6 number. The relevant test is Part A instead, and the paper states this.
- Study 34485958: 749 of 1,015 covered (74%). 30127495: 14 of 164 (9%).
  Norway (unpublished): 2,959 of 2,963 (100%). 32205351: 917 of 921 (100%).
- United Kingdom: 1,678 of 3,387 covered (50%).
- Labelled strains and prevalence, covered vs uncovered: CTX 3,752 (0.113) vs 1,657 (0.241);
  CAZ 4,088 (0.107) vs 1,885 (0.181); GEN 4,151 (0.122) vs 1,886 (0.107); CIP 4,147 (0.212) vs
  1,886 (0.264). AMP 4,308 vs 560 and SXT 3,257 vs 161, hence their exclusion from Part A.

So Part A is adequately powered for CTX, CAZ, GEN and CIP, and the two strata differ most in
country and study composition, which is stated as the limit above.
