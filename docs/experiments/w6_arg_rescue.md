# W6. Per-strain resistance genes: does adding them back fix lineage-held-out prediction?

**Status:** pre-registered on 2026-10-28, before any per-strain AMRFinderPlus call exists. Only
`arg_probe` (20 strains, used for route and timing, not for any rule below) may run before this
file is committed. Nothing in this file is changed after the first `arg_fetch` result is read;
any later change is added as a dated amendment below the original text.

## Question

The 72-reference pan-genome catalogue has no CTX-M or other ESBL gene, no gentamicin determinant
and no dfrA (`w4_amrfinder`). Is that why CTX, CAZ and GEN are predicted poorly on lineages the
model has not seen? If it is, adding each strain's own acquired genes should raise exactly those
drugs, and should not move CIP, whose mechanism (gyrA/parC point mutations) is not in the added
block.

## Inputs

- **Catalogue features (F0):** the rebuilt presence/absence matrix (`outputs-pa_fix`), loci with
  presence between 1% and 99% over the profiled strains. The 13 strains without a profile are left
  out of every arm.
- **ARG block (A):** AMRFinderPlus (`--organism Escherichia --plus`) on each strain's own BV-BRC
  assembly (`<Genome.ID>.fna`). Rows with Type = AMR, Subtype ≠ POINT, identity ≥ 90% and coverage
  ≥ 80%. One binary column per gene symbol; symbols present in fewer than 5 strains are dropped
  (a label-free rule). Point mutations are a separate block (P), used only in secondary analyses.
  Software and database versions are written next to every output; one version is used for all
  strains.
- **Labels:** the 7 paper drugs; intermediate = resistant (as in the paper). 16-drug results are
  secondary.

## Arms (identical model, folds and test strains)

| arm | features |
|---|---|
| F0 | catalogue presence/absence |
| F0+A | catalogue + per-strain acquired genes |
| A | per-strain acquired genes only |
| M | metadata only (as `confounder_floor.py`, all metadata) |

Model: XGBoost, `tree_method=hist`, 400 trees, depth 6, learning rate 0.05, subsample 0.8,
colsample_bytree 0.5, seed 0. No tuning, so no inner validation is needed and no arm gets more
tuning than another.

## Splits

- **Primary: ST-grouped 5-fold** (`results/splits/st_grouped_5fold.json`). Every Achtman ST, large
  ones included, lies whole in one fold; STs are dealt largest first to the fold with the fewest
  strains. Strains with unknown ST are in training in every fold and are never scored.
- Secondary: random 5-fold; the country (S4a, S4b) and time (S5) splits.

## Primary endpoint and rule

Per drug, from out-of-fold predictions pooled over the 5 folds:
`nAP = (AP − prevalence) / (1 − prevalence)`, AP = average precision.

- **Δ = mean nAP over {CTX, CAZ, GEN} of (F0+A) minus the same of (F0).**
- 95% CI: 2,000 bootstrap resamples **of STs** (strains of one ST are drawn together), paired, so
  both arms are scored on the same resample.
- **Pass:** Δ ≥ 0.10 **and** the CI's lower bound > 0.
- **Control:** CIP |nAP(F0+A) − nAP(F0)| ≤ 0.03. If the control fails, the result is reported as
  "the added block helps every drug", not as evidence about missing determinants.
- **Fail:** Δ < 0.10 or the CI includes 0. Then contribution 2 ("the catalogue lacks the causes")
  is not supported, and the paper is the benchmark alone.

AMC is reported but not in the primary set: clavulanate resistance has several mechanisms that the
database does not separate (W4 drug mapping).

## Coverage gate

The analysis runs on the strains with an AMRFinderPlus result. If that is fewer than 90% of the
profiled strains, all arms are restricted to the same strains, and the excluded strains' country,
year, study and ST distribution is reported against the included ones.

## Secondary analyses (reported, no pass/fail)

1. Δ for every drug (7 and 16), with CIs; the same on random 5-fold, so the random-to-ST gap can be
   compared with and without A.
2. Within-ST pairwise concordance (`src.eval.metrics.within_st_concordance`): among
   resistant–susceptible pairs from the same ST, the share ranked correctly; macro-averaged over STs
   with ≥ 1 such pair; reported for STs with ≥ 5 pairs too. An ST-only score is 0.5 by construction.
3. Intermediate excluded instead of resistant.
4. Arm A versus a rule: "resistant if any mapped determinant" (W4 drug mapping), to see how much the
   model adds over presence of the gene.
5. Point mutations (P) added to F0+A, for CIP.

## What this cannot show

A gene being present does not show it is expressed or that it causes the phenotype. A pass shows
that the catalogue omits information that predicts resistance in unseen lineages; it does not show
that omission is the only cause of the gap. Assemblies are short-read and BV-BRC-built; genes split
across contigs can be missed.
