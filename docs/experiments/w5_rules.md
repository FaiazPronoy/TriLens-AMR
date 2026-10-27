# W5. Rule-based baseline (ResFinder-like): pre-registration

**Status:** pre-registered, not run. It needs W4 step 1 (determinants) and step 4 (point
mutations). The rule table is fixed from those outputs **before** any strain is scored. No label is
used to build it.

## Question

How well does "resistant if a known determinant is present" do on every split? Since a rule learns
nothing, it cannot learn lineage. It is the honest reference under shift: a learned model that does not
beat it on S3–S6 has not shown more than catalogue knowledge.

## Rule

Per strain and drug:

- **Score** = number of determinants for the drug (W4 drug mapping) that the strain carries, plus
  the number of drug-relevant point mutations.
  - Carriage comes from the rebuilt presence/absence matrix (`1` = present).
  - Point mutations come from the SNP matrix.
- **Point mutations counted:**
  - CIP: gyrA S83L/A, gyrA D87N/G/H/Y, parC S80I/R, parC E84V/K/G (codons checked as in W4 step 4).
  - Other drugs: only those listed in AMRFinderPlus' *E. coli* point-mutation database for that drug class.
- **Primary call ("any"):** resistant if score ≥ 1.
- **Secondary call ("CIP strict"):** for CIP, resistant if ≥ 2 QRDR mutations or a plasmid quinolone
  gene. A single gyrA mutation often gives only reduced susceptibility.

## Evaluation

- On the test side of every split (S1–S6). The rule has no training, but only test strains are
  reported, so the numbers are paired with the models'.
- Metrics:
  - PR-AUC and ROC-AUC of the score (ties are frequent; average precision handles them);
  - sensitivity, specificity, VME and ME of the binary call;
  - 95% bootstrap CIs as in W1.
- **Comparison:** TriLens full stack, the classical arm and the best lookup against the rule, by paired
  bootstrap on the same strains, Holm across the antibiotics of each split.
- **Strains without a profile:** the 13 are left out. Strains whose determinant gene is absent from the
  pan-genome are not "fixed": the rule simply cannot see them, which is part of the result.

## Expected caveats, stated before seeing results

- The pan-genome is ECOR72-based, so acquired genes outside it are invisible (W4).
- AMC and the cephalosporins depend on enzyme variant and expression, which presence cannot capture.
- Missing SNP calls count as reference (KNOWN_ISSUES 16), so point mutations in absent genes are not
  called.

## Pass/fail

There is no pass/fail for the rule itself; it is a baseline. The paper's claim "the model adds
mechanism-level information beyond the catalogue under shift" requires: TriLens > rule (Holm p < 0.05)
on S3 for ≥ 4 of 7 drugs. A failure is reported as such.

## Compute

CPU, Kaggle (needs the SNP matrix), about 0.5 h after `w4_amrfinder`.

## Deviations

- **26 October 2026, before any data were scored (correction).** NCBI's own test output shows
  AMRFinderPlus reporting `blaEC`, the chromosomal ampC every *E. coli* carries, with Class
  BETA-LACTAM. Under the mapping above it would count as an AMP/AMC determinant in every strain, so
  the W5 "any determinant" call would be resistant for all strains.
  - Rule added: a **gene** determinant carried by ≥ 95% of strains (rebuilt P/A matrix) is excluded
    from the drug counts, as intrinsic. It is still listed in `determinants.csv`, flagged `intrinsic`.
  - Point mutations (for example ampC promoter variants) are not affected by this rule.
  - Decided from the tool's documented behaviour, not from any of our results.
