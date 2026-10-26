# Plan for the journal extension (W1–W6)

Working title: *Beyond lineage: a lineage-aware genomic model and a five-axis generalization test
for antimicrobial resistance in E. coli.*

This file says, for each workstream: the splits, the models, the metrics, the pass/fail rule, the
Kaggle notebook (CPU or GPU, account, estimated hours) and the order. Each experiment gets its own
pre-registration in `docs/experiments/<name>.md`, committed **before** it runs. A cheap pilot (one
antibiotic, one seed) comes first.

**Dependencies outside this plan:**

1. The full re-run now going on from `main` (`classical`, `trilens_random`, `trilens_splits`,
   `standard_random`, `standard_splits`, `shuffle_test`, `analysis`). W1–W6 compare against its
   numbers and reuse its outputs for the three existing splits.
2. **The presence/absence orientation question** (`docs/experiments/data_inventory.md` §4). It
   changes Lens B, so it must be settled **before** any new TriLens GPU run (W1 neural arm, W3, W6).
   The check is a CPU notebook of minutes (`w0_encoding`, below).

**GPU-hour estimates** are guesses (no run has been timed on a T4 yet; the thesis runs were on an
RTX 4080). Every notebook prints the minutes of each step. After the re-run, replace the numbers
below with those timings. Assumed per split, one seed: classical stack ≈ 1.5 GPU h; TriLens neural
arm ≈ 3.5 GPU h; gene ranking ≈ 4 CPU h (or ≈ 1 GPU h).

---

## The six splits (used by every workstream)

| id | split | file | sides | status |
|---|---|---|---|---|
| S1 | random | `arcadia_v2_master_features.pkl` (`random_split`) | 4,872 / 1,045 / 1,045 | exists |
| S2 | phylogroup-stratified | same file (`phylogeny_split`) | 4,872 / 1,045 / 1,045 | exists |
| S3 | sequence-type holdout | `mlst_lineage_holdout_split.pkl` | 4,836 / 1,081 / 1,045 | exists |
| S4a | country: train UK → test Norway | `country_split.pkl` (new) | UK minus val / 15% of UK / Norway | built here (CPU) |
| S4b | country: train Norway → test UK | same file | Norway minus val / 15% of Norway / UK | built here (CPU) |
| S5 | time: Norway, train 2002–2013 → test 2014–2017 | `time_split.pkl` (new) | ≤ 2013 minus val / 15% of ≤ 2013 / 2014–2017 | built here (CPU) |

Antibiotics per split (from W0's minimum counts):

- S1–S3: all 7.
- S4a/S4b: AMP, CIP, GEN, CTX, CAZ.
- S5: AMP, CIP, SXT, GEN as primary; CTX and CAZ as secondary.

Strains outside a split's two sides (other countries; Norway strains in S4 are all test; strains
without a year in S5) are not used by that split.

**Gene filter per split.** S4a, S4b and S5 each get their own 5,000-gene set, ranked on that split's
train + val exactly as `run_xgb_gene_importance_mlst.py` does for S3; test is never read.

- Only the ranked gene list and the importance CSV are saved (kilobytes). Each downstream
  notebook rebuilds the filtered SNP matrix from `arcadia_v2_snp_matrix.pkl` in minutes, so no
  1.9 GB pickle has to travel between notebooks.

---

## W0 follow-up: the encoding check

| notebook | what | needs | accelerator | account | est. hours |
|---|---|---|---|---|---|
| `w0_encoding` | P/A orientation from the raw `.bed`/`.bim`; missing calls per SNP column from the VCF; share of SNP columns in accessory genes | raw PLINK files and `filtered_output.vcf.gz` (Kaggle dataset `arcadia-presenceabsence-v2-ecoli-rawdownload`), the SNP and P/A pickles | CPU | B | ≈ 3 CPU h (VCF streaming); the orientation part alone is minutes |

---

## W1. Five-axis generalization test

Pre-registration: `docs/experiments/w1_five_axis.md`.

- **Splits:** S1–S5 (S4 in both directions).
- **Models (same on every split):**
  - TriLens full stack (strain-conditioned Lens B);
  - classical arm alone (the 5 learners stacked);
  - neural lenses alone (lenses + gate);
  - each classical learner solo (XGB-gene, LGBM-gene, RF-gene, LogReg-gene, XGB-χ²);
  - the W2 lineage-lookup baselines and the W5 rule baseline.
- **Metrics per split × antibiotic:**
  - PR-AUC with 95% bootstrap CI (2,000 resamples of test strains);
  - ROC-AUC (DeLong CI);
  - PR-AUC prevalence-matched to (a) the split's own training prevalence and (b) the random-split
    test prevalence;
  - VME and ME at 0.5, and at a threshold tuned on validation (F2-optimal; secondary: lowest
    threshold with validation ME ≤ 3%).
- **Statistics:** paired bootstrap (same test strains) for every model against TriLens full stack,
  with Holm correction across the antibiotics of each split.
- **Pass/fail:** see the pre-registration. In short, the paper's lineage claim needs TriLens full
  stack to beat the best lineage lookup on S3 and S5 for at least half of the usable antibiotics
  after Holm.
- **Notebooks:**

| notebook | what | needs | accelerator | account | est. hours |
|---|---|---|---|---|---|
| `w1_gene_sets` | gene ranking + 5,000-gene set for S4a, S4b, S5 | split files in the dataset | CPU | B | ≈ 12 CPU h (3 × 4; resumes across two sessions) |
| `w1_classical` | classical stack + solo learners on S4a, S4b, S5 | `w1_gene_sets` | GPU (XGBoost) | A | ≈ 4.5 GPU h |
| `w1_trilens` | neural arm on S4a, S4b, S5, seed 42 | `w1_gene_sets`, `w1_classical` | GPU | A | ≈ 10.5 GPU h (one notebook per split if the 12 h session is too short) |
| `w1_analysis` | stacking per split, all metrics, paired tests, the five-axis table | all W1, W2, W5 notebooks, plus `analysis` of the re-run | CPU | A or B | ≈ 1 CPU h |

---

## W2. Lineage-lookup baseline

Pre-registration: `docs/experiments/w2_lineage_lookup.md`.

- **Models:**
  - (a) ST lookup: resistant fraction of the training strains of the same ST;
  - (b) kNN: k nearest training strains by Hamming distance, k chosen on validation;
  - (c) phylogroup-only: resistant fraction of the phylogroup (exists as `run_phylogroup_only_baseline`).
  - kNN runs in two variants: gene presence/absence (CPU, here; a stand-in) and SNP calls (Kaggle).
- **Splits:** S1–S5.
- **Key table:** PR-AUC of each model minus the best lookup baseline, per split and antibiotic, with
  paired-bootstrap CIs and Holm-adjusted p.
- **Notebooks:**

| notebook | what | needs | accelerator | account | est. hours |
|---|---|---|---|---|---|
| (none) | ST lookup, phylogroup, kNN on P/A, all splits | small files only | CPU, here | — | done in minutes (pilot in `code/results/w2_lineage_lookup_pa_pilot/`) |
| `w2_snp_knn` | one 6,962 × 6,962 SNP Hamming matrix (cached, ≈ 200 MB), then kNN on every split | `arcadia_v2_snp_matrix.pkl` | CPU | B | ≈ 1 CPU h |

---

## W3. Is the model reading lineage or resistance?

Pre-registration: `docs/experiments/w3_lineage_vs_resistance.md` (to write before W3 runs).

- **Comparison:** strain-conditioned Lens B (`TRILENS_LENSB=fixed`) against standard
  (`standard`), both with the init-parity fix already on `main`.
- **Splits:** S1, S3, S5 at least; S2 and S4 if quota allows. Seeds: 3 per variant (5 if the
  spread is close to the difference).
- **Measures:**
  1. U_ST and U_res of the attended genes (`lineage_marker_test.py`, `strain_ablation_table.py`).
  2. **Lineage probe:** a multinomial logistic regression that predicts ST (STs with ≥ 20 training
     strains, the rest "other") from the model's internal representation (`z` of each lens and the
     gate output). Fitted on training strains and scored on test strains (balanced accuracy,
     macro-F1). Three references:
     - the same probe on raw P/A (upper reference);
     - on shuffled ST labels (chance);
     - on a randomly initialised, untrained model (architecture-only reference).
  3. Predictive performance on each split, to tie "less lineage" to "better under shift".
- **Pass rule (draft, finalised in the pre-registration):** strain conditioning reads less lineage if
  - the probe's balanced accuracy is lower than standard by more than 2 SD of the seed spread on at
    least 2 of the 3 splits; and
  - U_res is higher on at least 2 of 3.
- **Code needed:** the TriLens runs must save `z` per lens for train and test (`trilens_v10.py`
  currently saves attention and gate weights, not `z`). This is a small change, made on a branch
  before W3 runs.
- **Blocked by:** the P/A orientation decision (it changes what Lens B "attends to").
- **Notebooks:**

| notebook | what | accelerator | account | est. hours |
|---|---|---|---|---|
| `w3_seeds_fixed` | fixed Lens B, seeds 123 and 456, on S1, S3, S5 (seed 42 comes from the re-run and W1) | GPU | A | ≈ 21 GPU h (6 runs × 3.5) |
| `w3_seeds_standard` | standard Lens B, seeds 42, 123, 456, on S1, S3, S5 (S1/S3 seed 42 from the re-run) | GPU | B | ≈ 24.5 GPU h (7 runs) |
| `w3_probe` | probes, U_ST / U_res per run, the table | CPU | A or B | ≈ 1 CPU h |

---

## W4. Mechanism validation (CPU)

Pre-registration: `docs/experiments/w4_mechanism.md`.

1. **AMRFinderPlus** (bioconda or the GitHub release) on `pangenome_cds.fa` (18,494 CDS), in
   nucleotide mode, with `--organism Escherichia`. Map hits to P/A columns by locus tag, **after**
   the P/A orientation is fixed.
   - Also answers KNOWN_ISSUES §7 (is CTX-M in the pan-genome?).
2. **Known determinants among the model's genes:** per antibiotic, the fraction of the top-k attended
   (Lens B) and top-k SHAP genes that are AMRFinderPlus/CARD determinants for that drug class,
   against 10,000 random gene sets of the same size and prevalence bin (empirical p).
3. **SXT:** co-occurrence of aadA with dfrA, sul1 and intI1 across strains (phi coefficient and
   Fisher test, Holm), within ST as well as overall. Do the attended genes cover these genes?
4. **CIP epistasis:**
   - Genotype gyrA S83L, D87N/H/Y and parC S80I from the SNP matrix. The SNP name
     `LMHECDEF_04343:POS:REF>ALT` gives the position in the CDS; codon = ⌈POS/3⌉, checked against
     the translated CDS from `pangenome_cds.fa`.
   - Logistic regression of CIP-R on gyrA, parC and gyrA × parC, with ST as a random effect, or with
     the 10 top phylogenetic PCs.
   - The model's mean predicted probability per genotype combination.
5. **Candidate genes:** genes the model uses that are not known determinants, with their eggNOG
   annotation (`cds_eggNog.tsv`), listed as hypotheses only.

| notebook | what | needs | accelerator | account | est. hours |
|---|---|---|---|---|---|
| `w4_amrfinder` | AMRFinderPlus, mapping, steps 2, 3, 5 | `raw/pangenome/*`, `cds_eggNog.tsv`, attention/SHAP outputs of the re-run | CPU | B | ≈ 1 CPU h |
| `w4_snp_genotypes` | step 4 (needs the SNP matrix) | `arcadia_v2_snp_matrix.pkl` | CPU | B | ≈ 0.5 CPU h |

---

## W5. Rule-based baseline (ResFinder-like)

Pre-registration: `docs/experiments/w5_rules.md`.

- A strain is called resistant to a drug if it carries a known determinant of that drug class:
  - AMRFinderPlus gene hits (W4) through the P/A matrix;
  - known point mutations from the SNP matrix (CIP: gyrA S83/D87, parC S80/E84; others only where
    AMRFinderPlus' point-mutation database covers *E. coli*).
- The rule table is fixed from the AMRFinderPlus class/subclass column before any test data are scored.
- Score: number of determinants, as a ranking for PR-AUC; binary call for VME/ME.
- Run on S1–S5. Nothing is learned, so the rule is the honest baseline under shift.

| notebook | what | accelerator | account | est. hours |
|---|---|---|---|---|
| `w5_rules` | rule table, scoring on all splits | CPU | B | ≈ 0.5 CPU h (after `w4_amrfinder`, `w4_snp_genotypes`) |

---

## W6. Multi-seed and statistics

- Seeds 42, 123, 456 for TriLens full stack on S1–S5, and for the classical arm on S1–S5 (XGBoost on
  the GPU is not bit-for-bit reproducible; review item 7).
- Report mean ± SD over seeds and the bootstrap CI of the seed-42 run.
- Paired tests per seed against every baseline, Holm across antibiotics.
- Headline claims must hold for all 3 seeds.

| notebook | what | accelerator | account | est. hours |
|---|---|---|---|---|
| `w6_seeds` | TriLens full stack, seeds 123/456, S2 and S4a/S4b (S1, S3, S5 are covered by `w3_seeds_fixed`) | GPU | A | ≈ 21 GPU h |
| `w6_classical_seeds` | classical arm, seeds 123/456, S1–S5 | GPU | B | ≈ 18 GPU h |
| `w6_stats` | the final tables | CPU | A or B | ≈ 0.5 CPU h |

---

## Order and accounts

Assumes about 30 GPU h per account per week, with one GPU notebook per account at a time. CPU
notebooks cost no GPU quota and can run while a GPU notebook runs on the same account.

| week | account A (GPU) | account B (GPU) | CPU (either account) |
|---|---|---|---|
| now | re-run (`classical` → `trilens_random` → `standard_random` → `analysis`) | re-run (`shuffle_test` → `trilens_splits` → `standard_splits`) | `w0_encoding` (B) right away |
| after the re-run | `w1_classical` (4.5 h) → `w1_trilens` (10.5 h) | — | `w1_gene_sets` (B, first), `w2_snp_knn`, `w4_amrfinder`, `w4_snp_genotypes`, `w5_rules` |
| +1 | `w3_seeds_fixed` (21 h) | `w3_seeds_standard` (24.5 h) | `w1_analysis`, `w3_probe` |
| +2 | `w6_seeds` (21 h) | `w6_classical_seeds` (18 h) | `w6_stats` |

**Total new GPU:** ≈ 100 GPU h, about two weeks of both accounts after the re-run. If quota runs
short, cut in this order:

1. S2 and S4b from W6;
2. the fifth seed (never planned);
3. S5's secondary antibiotics.

Never cut S3 or S5 from W3: they carry the claim.

Notebooks are added as STAGES to `kaggle/make_nbs.py` when their pre-registration is committed, not
before.

---

## W7 (optional)

Selective prediction under shift, as a short section citing the NeurIPS 2026 negative result on
clade shift. More antibiotics from the 50 in the phenotype matrix, where both sides of S3 pass W0's
minimum counts.
