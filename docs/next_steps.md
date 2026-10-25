# Next steps toward the journal paper

Target: Microbial Genomics (Research4Life Group A waiver for Bangladesh; check at
submission), with Bioinformatics Advances or Computers in Biology and Medicine as backup.

**Rule for every experiment:** a cheap pilot first (CPU, existing files), with its pass/fail
rule written here *before* it runs. Only a pilot that passes goes to a Kaggle GPU run. A
pilot that fails becomes one sentence in the paper, not weeks of runs.

---

## What we already know (October 2026 checks, no training)

1. **Lineage concentration does not explain which drugs collapse.** Per-drug Cramér's V
   between ST and phenotype, top-ST share and ST entropy of resistant isolates were tested
   against the holdout drop. None is significant (Cramér's V ρ = +0.07; top-3 ST share
   ρ = +0.68, p = 0.09). CIP is the counter-example: 44% of its resistant isolates are
   ST131, yet it is one of the robust drugs, because gyrA/parC mutations are directly visible
   in the SNP view. **Dropped as a hypothesis.**
2. **The drop tracks prevalence** (ρ = −0.89, p = 0.007). Test prevalence is lower than
   train prevalence for every drug under the holdout (GEN 12.7% → 4.2%, CAZ 12.9% → 4.7%,
   CTX 14.1% → 8.8%). The prevalence-matched holdout still keeps the 3-versus-4 split, so
   prevalence is not the whole story.
3. **The feature space may lack the main ESBL gene.** The pan-genome (built from the 72 ECOR
   strains) was searched for resistance-gene motifs. Controls were found (AmpC
   `LMHECDEF_03843`, gyrA `LMHECDEF_04343`). TEM was found (`MFCAOJAD_04226`, present in
   49% of isolates; P(AMP-R | present) = 0.84 vs 0.20 absent). **CTX-M was not found** with
   four independent motifs across all 18,494 CDS. If confirmed, CTX/CAZ models cannot see
   their main cause and fall back on lineage, which fits their collapse.

---

## Order of work

### Step 0. Full re-run from the processed data (no uncertainty)

The seven notebooks in `kaggle/notebooks/` (order in `kaggle/README.md`). This is
reproduction, not an experiment. Done = `results/diagnostics/paper_numbers.csv` has no
MISSING row. Then the paper is updated from that file, and only after that do the
experiments below start, so every new number is compared against re-run numbers.

### Step 0b. Repeated random splits of the frozen pipeline (the overfitting answer)

The random-split test set was also used while the model was being developed (v7 to v10),
so a reviewer can say the design is tuned to that one split. The answer is to freeze the
pipeline as it is and run it on new random splits it has never seen.

- **What:** 3 to 5 new stratified random splits (new seeds), each with its own gene filter
  refitted on its own training strains, then the classical arm and TriLens exactly as now.
- **Pilot first (cheaper):** classical arm only on 3 new splits. Pass = mean PR-AUC within
  0.01 of the current classical arm (0.8469) and the spread small (sd below 0.01).
- **Then:** the full stack on the same splits. Report mean and sd next to 0.8486.
- **Cost:** the gene-filter refit is the long part (it took 5 to 8 hours per split on an
  RTX 4080); this needs its own notebook, not written yet.

### Step 1. Confirm the CTX-M finding (CPU, minutes)

Run AMRFinderPlus (nucleotide mode) on `code/data/raw/pangenome/pangenome_cds.fa` on Kaggle.

- **Pass:** no blaCTX-M hit. Then finding 3 goes into the paper as a mechanism.
- **Fail:** CTX-M is present. Then find which column it is and why the model did not use it;
  finding 3 is dropped.

### Step 2. Add acquired AMR genes as features (the most likely real improvement)

NCBI Pathogen Detection publishes AMRFinderPlus calls for public isolates. Join them to the
6,962 isolates by SRA / BioSample accession.

- **Gate:** at least 80% of isolates join. Below that, stop.
- **Pilot (CPU):** classical arm only (XGBoost) on current features + AMR-gene features,
  lineage holdout.
- **Pass:** holdout PR-AUC rises by at least 0.02 on at least 2 of CTX, CAZ, GEN, AMC, with no
  more than 0.005 loss on the random split. Only then the full TriLens run.

### Step 3. Leave-lineage-out cross-validation (CPU pilot)

Split STs into 5 groups; each is the test set once; refit the gene filter in every fold.

- **Pilot:** gene presence/absence + XGBoost.
- **Pass:** the same 3 drugs are robust in at least 4 of 5 folds. Then the neural arm on Kaggle.
- **If it fails,** the paper's main claim needs rewording before anything else.

### Step 4. A lineage-aware fix (CPU pilot)

ST-balanced reweighting, dropping features with high `U_ST`, or population-structure PCs as
covariates.

- **Pass:** holdout PR-AUC up by at least 0.01 on at least 2 fragile drugs, random-split loss
  at most 0.005, in at least 3 folds of step 3.

### Step 5. Distance-based abstention (CPU, uses saved predictions)

Distance of each test isolate to the nearest training isolate (gene P/A Jaccard).

- **Pass:** Spearman ρ ≥ 0.3 between distance and per-isolate error, and abstaining on the
  most distant isolates lowers VME. Otherwise dropped.

### Step 6. Zero-shot external validation (largest effort, last)

Train on Arcadia, test without retraining on another collection (Moradigaravand 2018, or
BV-BRC) using harmonised features (AMRFinderPlus genes + known point mutations).

- **Gate:** assemblies or gene calls are obtainable for that collection.

---

## Before submission

- Fill every `PASTE_LINK_HERE` (`code/docs/DATA.md`, `code/CITATION.cff`).
- Push `code/` to GitHub, archive a release on Zenodo for a DOI.
- Post a bioRxiv preprint.
- Decide authorship order, venue and APC support with Prof. Alam (corresponding author).
- Report the multi-seed runs (`results/pillar_t6_multiseed/`), say plainly that Giessen was
  retrained, and keep the CLSI limitation.
