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

- **Primary: ST-grouped 5-fold** (`results/splits_kfold/st_grouped_5fold.json`). Every Achtman ST, large
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
2. Within-ST pairwise concordance (`src.evaluation.grouped.within_st_concordance`): among
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

## Amendments

- 2026-10-28 (before any ARG data): the k-fold files moved from `results/splits/` to
  `results/splits_kfold/`, because `src.splits.from_json` reads every file in `results/splits/` as a
  train/val/test split. Content unchanged; no rule changed.
- 2026-10-28 (before any ARG–phenotype analysis; only route probes have run): **source of the ARG
  block changed.** `arg_probe` found BV-BRC's ftp host unreachable from Kaggle (https timed out, ftp
  refused) and `arg_probe2` found its API host refusing (HTTP 403). NCBI Pathogen Detection publishes
  AMRFinderPlus results for the same public SRA runs (`PDG000000004.6394.metadata.tsv`,
  `AMR_genotypes`), covering 5,225 of the 6,962 strains, 5,224 of them run with AMRFinderPlus 4.2.7.
  Rules for this source, fixed now:
  1. Join by SRA run; if none, by BioSample. A strain matching more than one row keeps the run match,
     then the latest `target_creation_date`. The joined rows are frozen in
     `code/results/data_inventory/pd_amr_joined.tsv.gz` (file name and SHA-256 recorded).
  2. **Primary set:** AMRFinderPlus 4.2.7 with database 2026-03-24.1 (the most common pair), so one
     version is used for every strain, as the original rule required. Sensitivity: every 4.2.7 row.
  3. A token with no suffix is a detected gene (acquired block A). `=POINT` goes to the point block P.
     `=PARTIAL`, `=PARTIAL_END_OF_CONTIG`, `=MISTRANSLATION` and `=HMM` are not counted; they stand in
     for the identity ≥ 90% / coverage ≥ 80% rule, which this table does not report. Symbols in fewer
     than 5 strains are dropped, as before.
  4. NCBI ran either a nucleotide-only or a combined search; the split is reported.
  5. The coverage gate applies (coverage is below 90%): every arm uses the same strains, and the
     excluded strains' country, year, study, ST and phylogroup are reported against the included.
  6. The table's `AST_phenotypes` column is not used as a feature or a label in W6.

## Result (read 2026-10-28, against the rules above; Kaggle logs of w6_rescue and w6_rescue_16)

- **Primary: pass.** ST-grouped 5-fold, mean nAP over CTX, CAZ, GEN: F0 0.450, F0A 0.726,
  Δ **+0.276 [0.178, 0.328]** (ST-clustered). CIP control Δ −0.004 (|Δ| ≤ 0.03): pass.
- Sensitivities: intermediate excluded +0.293 [0.217, 0.345]; all database versions (5,218 strains)
  +0.286 [0.202, 0.333]; 16-drug run, same primary +0.276. CIP control passes in all.
- Same three drugs, other splits (F0 → F0A): random 0.750 → 0.834; UK→Norway 0.249 → 0.429;
  Norway→UK 0.543 → 0.793; Norway 2013 time split 0.449 → 0.670.
- The ARG block alone (A) is as good as F0A on the ST-grouped split (0.730): on unseen lineages the
  catalogue adds nothing to the acquired-gene calls for these drugs.
- Metadata arm M: 0.634 on random 5-fold but 0.182 ST-grouped (see W7).
