# W7. What the assembly-statistics shortcut is

**Status:** pre-registered 2026-10-28. The three measurements below were run once on the frozen
Pathogen Detection calls while this file was written; the numbers are in §Result and nothing in the
rules was changed after seeing them. Anything added later is a dated amendment.

## Why

`confounder_floor.py` (shortcut_floor, 2026-10-28) separated the metadata baseline's fields:

| split | assembly statistics only (nAP) | epidemiological fields only (nAP) |
|---|---|---|
| random | 0.471 | 0.356 |
| ST holdout | 0.059 | 0.150 |
| UK → Norway | 0.097 | 0.012 |
| Norway → UK | 0.232 | 0.056 |
| Norway, ≤2013 → 2014+ | 0.180 | 0.000 |

Across countries and across time the epidemiological fields (country, year, publication, platform)
are at chance, while assembly statistics keep a clear margin. So the part of the "metadata floor"
that transfers is not a study batch effect. The question is what assembly size stands in for.

## Hypothesis

Assembly size is a proxy for how many acquired resistance genes a strain carries: each acquired
determinant arrives on a mobile element that adds sequence.

## Measurements (label-free except 3)

1. Spearman correlation of each assembly statistic with the acquired-gene count (Pathogen Detection
   calls, `src.mechanism.arg_pd`; a gene present in more than 50% of strains is counted as
   chromosomal and excluded), computed (a) over all strains, (b) **within sequence type**, every
   column centred inside its ST (STs with ≥ 10 strains), (c) within phylogroup, and (d) inside each
   phylogroup separately.
2. The same within-phylogroup, so no single lineage can carry the result.
3. Within-ST concordance of each feature against each drug's phenotype: among resistant–susceptible
   pairs from the same ST, the share the feature ranks correctly. A feature that is constant inside
   an ST scores exactly 0.5. No model is fitted, no split is used.

## Pass/fail, fixed before the numbers were read

- **Claim 1 (the link is real, not between-lineage):** within-ST Spearman of assembly size with the
  acquired-gene count ≥ 0.20, p < 0.001, and the same sign in at least 4 of the phylogroups with
  ≥ 100 strains. Below that, the proxy explanation is dropped and the shortcut stays unexplained.
- **Claim 2 (the shortcut survives inside a lineage):** within-ST concordance of assembly size
  ≥ 0.55 for at least 5 of the 7 drugs. Below that, assembly size is a between-lineage effect only.
- **Claim 3, exploratory, no pass mark:** whether the shortcut is larger for the drugs whose
  determinants are acquired (AMC, SXT, GEN, CTX, CAZ) than for the chromosomal one (CIP). Reported
  either way; this is not evidence for or against claims 1 and 2.

## Result (2026-10-28, 5,072 strains, 85 acquired-gene columns)

- Claim 1 **passes**: within ST ρ = 0.325 (p = 8 × 10⁻¹⁰⁰, n = 4,033, 62 STs); within phylogroup
  ρ = 0.337; significant and positive in 6 of 7 phylogroups (A, n = 185, p = 0.15 is the exception).
  Slope between strains ≈ 13 kb per acquired gene, on a 5.09 Mb median genome.
- Claim 2 **passes**: assembly size scores 0.552–0.676 within ST, above 0.55 for 6 of 7 drugs
  (GEN 0.552, CAZ 0.568, CIP 0.590, SXT 0.612, AMP 0.624, CTX 0.658, AMC 0.676).
- Claim 3: the difference is small — 0.613 for acquired-determinant drugs against 0.590 for CIP. So
  **the shortcut is broad, not specific to acquired determinants**, and must not be written up as
  specific. The acquired-gene count itself scores far higher (mean 0.758).

## What this does and does not show

It shows that a single number from the assembly ranks resistant above susceptible strains inside
the same lineage, and that the number tracks acquired-gene count. It does not show causation: larger
genomes also differ in other ways, and short-read assembly size is affected by repeat collapse and
contamination. Contig N50 correlates negatively (ρ = −0.185), which is consistent with mobile
elements fragmenting an assembly, and is equally consistent with lower-quality assemblies being
both larger and more fragmented. The paper states both readings.

## Prior work we checked

A targeted search found work on *E. coli* genome expansion and pathogenicity, and on plasmid
prediction from assembly features (RFPlasmid), but nothing testing assembly size as a
within-lineage AMR shortcut in a machine-learning benchmark. That is not a novelty claim: the
search was targeted, not systematic, and the related work must be read and cited before any claim
of priority.
