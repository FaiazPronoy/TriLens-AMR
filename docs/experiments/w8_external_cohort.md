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

## Amendment 2026-10-28, after the feasibility count and before any model or label is read

The probe (`PDG000000004.6394`) found **10,876** external isolates (AST present, not ours,
AMRFinderPlus 4.2.7); all seven of our drugs have ≥ 100 resistant and ≥ 100 susceptible, so the gate
is met. 7,388 are from the USA, 1,334 Canada, 718 UK; 6,158 environmental/food, 4,704 clinical;
median year 2019. Counts per drug (R/S): AMP 3,835/4,204, CIP 1,482/6,473, AMC 888/5,640,
SXT 1,440/5,928, GEN 1,226/6,399, CTX 733/291, CAZ 972/2,617. CTX is tested on only 1,054
isolates and is 70% resistant, so it is a selected panel and is reported but not in the primary mean.

Fixed now:

1. **Same database as training.** Primary analysis uses external isolates called with database
   2026-03-24.1 (10,232), the same version as the training block. All 4.2.7 rows are a sensitivity.
2. **Calls.** `AMR_genotypes` parsed with the W6 rules (`src.mechanism.arg_pd.tokens`): a token
   without suffix is a gene, `=POINT` goes to the point block, `PARTIAL`, `PARTIAL_END_OF_CONTIG`,
   `MISTRANSLATION`, `HMM` are not counted. External genes not among the training columns are
   dropped (a frozen model cannot use a column it never saw).
3. **The metadata arm** uses only the fields both sources have: assembly length, contig count, contig
   N50. Country, year and publication are left out: the training set has no USA strain, and W7
   showed these fields carry nothing across countries anyway. So this arm is the W7 assembly
   shortcut, frozen and transferred.
4. **Arms.** AP = acquired genes plus point mutations, i.e. every AMRFinderPlus call (**primary**,
   because CIP's determinants are point mutations and an acquired-only arm would be handicapped on
   it by construction), A = acquired genes only (secondary), M = the three assembly fields,
   R = the determinant rule below. Model: the W6 XGBoost
   settings, trained once on all our strains with a call (5,072), no tuning.
5. **Determinant rule** (symbol patterns, fixed here, not tuned):
   - AMP: any acquired β-lactamase `bla*` other than the chromosomal `blaEC*`
   - AMC: `blaOXA-1`, plasmid AmpC (`blaCMY`, `blaDHA`, `blaFOX`, `blaACC`), or a carbapenemase
     (`blaKPC`, `blaNDM`, `blaVIM`, `blaIMP`, `blaOXA-48`-like `blaOXA-48`/`-181`/`-232`/`-244`)
   - CTX, CAZ: `blaCTX-M`, `blaSHV` other than `blaSHV-1`/`-11`, `blaTEM` other than `blaTEM-1`,
     plasmid AmpC, or a carbapenemase
   - GEN: `aac(3)`, `ant(2'')`, `armA`, `rmt`
   - SXT: `dfrA` or `dfrB`
   - CIP: any `gyrA` point mutation, or `qnr`, or `aac(6')-Ib-cr`
6. **Clustering.** CIs bootstrap over `bioproject_acc`; isolates with none are their own cluster.
   `ext_probe` is extended to keep `bioproject_acc` and must be re-run once to supply it.
7. **Endpoint:** arm AP minus arm M, mean nAP over AMP, CIP, AMC, SXT, GEN, CAZ (CTX reported
   separately), pass if ≥ 0.10 with the clustered CI above zero. (The original text named arm A; it
   is AP for the CIP reason in point 4, changed before any model was fitted.) Secondary: arm AP
   against the rule R, per drug, so the paper can say how much a model adds over gene presence, which
   Lipworth et al. (2024) already show is strong for most of these drugs.

## Result (read 2026-10-28; Kaggle log of w8_external, candidates sha256 0e2ccc69…)

- **Primary: pass.** 10,232 external isolates, 62 BioProjects. Mean nAP over AMP, CIP, AMC, SXT,
  GEN, CAZ: AP 0.807, M 0.045; AP − M **+0.762 [0.676, 0.804]**. Also passes on all database
  versions (10,876 isolates, +0.773 [0.690, 0.822]) and with intermediates excluded
  (+0.753 [0.668, 0.808]).
- The assembly-size arm, which predicts within our data (W7), carries almost nothing externally
  (mean nAP 0.045).
- **Secondary, against the determinant rule R:** AP − R +0.065 [0.005, 0.140]; all versions
  +0.070 [0.011, 0.133]; intermediates excluded +0.021 [−0.049, 0.116]. Per drug the model is
  clearly ahead only for AMC (+0.214) and SXT (+0.065); the rule is ahead for GEN (−0.077) and CTX
  (−0.185, selected panel). So the honest claim is transfer beyond the shortcut, not a large gain
  over gene presence, consistent with Lipworth et al. (2024).
- Acquired-only arm A − M: +0.703 [0.623, 0.746]; CIP is where A falls behind AP (0.572 vs 0.885),
  as expected from point mutations.
