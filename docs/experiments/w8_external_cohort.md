# W8. An external cohort from NCBI Pathogen Detection

**Status:** pre-registered 2026-10-28, before the feasibility probe has run. The probe
(`src.mechanism.ext_probe`) only counts; no model is fitted until the counts are read and the gate
below is met.

## Why

Every external review named the same fatal objection: our 6,962 strains are 91% UK and Norway, from
research collections, 2001–2017. Any claim about "genomic AMR prediction" needs a cohort we did not
train on.

Pathogen Detection's *E. coli*/*Shigella* table holds 592,113 isolates with AMRFinderPlus calls
(`AMR_genotypes`) and, for some, measured susceptibilities (`AST_phenotypes`). On our own strains
only 4.3% carry the AST column, because ours are research collections; the rest of the table is
largely surveillance, where susceptibilities are submitted. This is the same source and the same
software version (4.2.7) as the W6 feature block, so an external test costs no new pipeline.

## Feasibility gate (decided by the probe, not by us)

W8 runs only if, after removing every isolate that matches one of our SRA runs or BioSamples:

- at least **3 of our 7 drugs** have ≥ 100 resistant **and** ≥ 100 susceptible external isolates, and
- at least **2,000** external isolates remain in total.

Below that, W8 is reported as "attempted, not possible with this source" with the counts, and the
paper keeps its two-country limitation stated plainly.

## Design, if the gate is met

- **Frozen model.** The W6 arms are trained on our strains only and then applied unchanged. No
  threshold, feature set or hyperparameter is chosen on the external cohort.
- **Features.** Only the ARG block (arm A) and the metadata block transfer: the catalogue
  presence/absence features are defined by our pan-genome and do not exist for an outside isolate.
  So the external test compares **arm A (acquired genes) against the metadata-only arm**, and
  against a "resistant if any mapped determinant" rule (W4 drug mapping).
- **Labels.** `AST_phenotypes`; intermediate counted as resistant, as in the paper, with an
  intermediate-excluded sensitivity run. `ND` is treated as untested.
- **Grouping.** Confidence intervals bootstrap over **BioProject**, not isolate, because
  surveillance submissions arrive in large related batches.

## Primary endpoint and rule

Per drug, prevalence-normalised average precision `nAP = (AP − prevalence) / (1 − prevalence)`.

- **Pass:** arm A beats the metadata-only arm by ≥ 0.10 mean nAP over the eligible drugs, with the
  BioProject-clustered 95% CI above zero.
- **Fail:** below that. Then the honest reading is that a model trained on our two-country research
  collections does not transfer to surveillance isolates, which is itself worth reporting.

## Stated limits, whatever the result

- The external cohort is a different kind of population: largely US surveillance, with food and
  animal sources, later years, and different susceptibility-testing standards. A drop can be
  population difference rather than model failure, and the paper cannot separate the two.
- Pathogen Detection AST entries come from many laboratories with different breakpoints and are not
  harmonised here.
- These isolates were called by NCBI with database versions that may differ from 2026-03-24.1; the
  version mix is reported and the primary analysis is restricted to 4.2.7.
- The genotype–phenotype relationship itself is imperfect: Lipworth et al. (Lancet Microbe 2024,
  9,001 *E. coli*) report AMRFinder concordance 93.9% for ampicillin, 95.2% ceftriaxone, 99.0%
  gentamicin, 83.8% ciprofloxacin, 94.0% trimethoprim, but only 75.5% for amoxicillin–clavulanic
  acid, with a 75.4% very-major-error rate. So an ARG-based arm has a ceiling, and AMC is expected
  to be the weakest drug in any arm. (Read: full text, Table 1.)
