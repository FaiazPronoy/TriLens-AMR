# TriLens-AMR

Predicting antibiotic resistance from bacterial genomes, and measuring what such a prediction is
really worth when the test strains are not relatives of the training strains.

This repository holds the full code, the committed results and the manuscript for a study of 6,962
*Escherichia coli* genomes and 7 antibiotics. It also holds the undergraduate thesis the work grew
out of.

## What the study found

| Question | Answer |
|---|---|
| How well does the model do on a random split? | average precision 0.850, ahead of every outside baseline |
| How much of that needs no gene at all? | 0.705 from strain metadata and assembly statistics alone |
| What happens when whole lineages are held out? | 0.645; correcting for the prevalence shift gives 0.732, a true gap of 0.118 |
| Is the drop the same for every drug? | no: 3 drugs lose at most 0.026, 4 lose 0.152 to 0.239, nothing in between |
| Do per-strain determinant calls recover the loss? | yes, +0.276 normalised average precision on unseen lineages |
| Do frozen models transfer to an outside cohort? | yes, on 10,232 isolates from NCBI Pathogen Detection |
| Is a simple model enough? | yes: a sparse linear model on determinant calls matches gradient boosting and beats a fixed rule by 0.104 |
| Is it safe for clinical use? | no: very major error rates are above the regulatory benchmark for every approach |

Three pre-registered tests failed their own pass rule. They are reported as failures, which is why
the rules were fixed in writing beforehand.

## The model

TriLens reads each genome three ways and learns how much to trust each one.

| Branch | Input | Looks for |
|---|---|---|
| Motif | SNPs inside each selected gene | changes within a gene |
| Presence | the pan-genome presence/absence matrix | which genes a strain carries |
| Cross-antibiotic | labels of the other drugs | resistance that travels together |

A mixture-of-experts gate weighs the three branches per strain. Their output is then stacked with
five classical learners (XGBoost on gene features, LightGBM, random forest, logistic regression,
XGBoost on chi-squared filtered SNPs) under one logistic meta-learner.

| Stage | Mean average precision, random split |
|---|---|
| three branches, averaged | 0.780 |
| plus the gate | 0.796 |
| plus the five classical learners | 0.850 |

## The three splits

The same model, the same features, three ways of deciding which strains go in the test set.

| Split | How the test set is chosen | Mean average precision |
|---|---|---|
| Random | at random, 70/15/15 | 0.850 |
| Phylogroup-stratified | balanced across phylogroups | 0.856 |
| Lineage holdout | no sequence type is in both train and test | 0.645 |

A lineage holdout also lowers the share of resistant strains in the test set, which lowers average
precision on its own. Removing that effect leaves a true gap of 0.118.

## Repository map

| Folder | What is in it |
|---|---|
| `trilens/` | the library: data loading, models, splits, experiments, analysis |
| `pipeline/` | 30 numbered steps, in the order the paper reports them |
| `configs/` | Hydra configuration, one file per experiment |
| `results/` | every committed result table, one folder per run |
| `kaggle/` | notebooks that run the whole pipeline on free Kaggle GPUs |
| `docs/preregistration/` | the documents that fixed each test before its data were read |
| `paper/` | the manuscript |
| `thesis/` | the undergraduate thesis |
| `tests/` | the test suite |
| `data/` | not in git; see `data/README.md` for what goes here |

Inside `trilens/`:

| Package | What it does |
|---|---|
| `data/` | builds the SNP matrix, the labels, the presence/absence matrix and the gene features |
| `splits/` | random, phylogroup, lineage, country, time and cross-validation splits |
| `models/` | the three branches, the gate and the plain CNN baseline |
| `evaluation/` | cluster bootstrap intervals, clinical error rates, training helpers |
| `experiments/` | one module per trained model or ablation |
| `determinants/` | AMRFinderPlus calls and the strain by determinant matrix |
| `analysis/` | everything measured on top of saved predictions |
| `utils/` | seeds, logging and release upload |

## The pipeline

Every step is a module, run the same way. The numbers follow the order of the paper, not the order
a person has to run them in: steps 1 to 8 prepare the data, 9 to 21 are the internal study, and 22
to 30 are the pre-registered experiments.

```bash
python -m pipeline.step09_train_classical_stack --help
python -m pipeline.step09_train_classical_stack --split random
```

A step with several parts asks for one with `--part`, `--split` or `--model`, and `all` runs them in
order. `--help` on any step lists its choices and the module behind each one.

| Step | What it does | Writes to |
|---|---|---|
| 01 | Build the SNP matrix from the release VCF | `data/processed` |
| 02 | Build labels, the random and phylogroup splits and the PCA features | `data/processed` |
| 03 | Build the lineage holdout split | `results/lineage_holdout_split` |
| 04 | Download and rebuild the gene presence/absence matrix | `results/gene_presence_rebuild` |
| 05 | Select the informative genes, once inside each split | `data/processed` |
| 06 | Repair the damaged SNP block | `results/diagnostics` |
| 07 | Build the strain table and the covariates | `results/data_inventory` |
| 08 | Build the country, time and cross-validation fold splits | `results/splits, results/fold_splits` |
| 09 | Train the classical stack on each split | `results/classical_stack_*` |
| 10 | Train the outside baselines: PCA, phylogroup label, TabNet | `results/pca_baseline and others` |
| 11 | Train the TriLens stack on each split | `results/trilens_stack_*` |
| 12 | Ablate the components of the stack | `results/component_ablation` |
| 13 | Stack the neural and classical arms on the other two splits | `results/trilens_stack_*` |
| 14 | Correct the holdout for its prevalence shift | `results/diagnostics, results/prevalence_invariant_metrics` |
| 15 | Measure clinical errors and tune thresholds | `results/diagnostics` |
| 16 | Audit the splits and the near-duplicate strains | `results/diagnostics, results/near_duplicate_audit` |
| 17 | Run the position shuffle control | `results/position_shuffle_test` |
| 18 | Retrain the method on the Giessen collection | `results/giessen_transfer` |
| 19 | Explain the models: gain, SHAP, attention, marker genes | `results/gene_importance_*, results/diagnostics` |
| 20 | Test seed stability | `results/seed_stability` |
| 21 | Measure the metadata floor | `results/diagnostics` |
| 22 | Build the strain by determinant matrix | `results/determinant_matrix` |
| 23 | Rescue unseen lineages with determinant calls | `results/determinant_rescue` |
| 24 | Test assembly size as a shortcut | `results/diagnostics, results/assembly_size_mechanism` |
| 25 | Screen the external cohort | `results/external_cohort_screen` |
| 26 | Validate frozen models on the external cohort | `results/external_cohort, results/external_robustness` |
| 27 | Compare the model with the determinant rule | `results/coverage_and_rule` |
| 28 | Harmonise the labels with MIC values | `results/mic_harmonisation` |
| 29 | Test a second species | `results/species_screen, results/campylobacter` |
| 30 | Compare a linear model, and the clinical errors | `results/linear_model_and_clinical_errors` |
## Running it on Kaggle

The full pipeline needs more GPU hours than one machine gives comfortably, so it is split into 17
notebooks that run on free Kaggle GPUs. Each notebook clones this repository, runs its steps, zips
its results and uploads the zip to a GitHub release. A notebook that needs an earlier notebook's
output downloads that release, so they can be run one at a time.

Generate them with:

```bash
python kaggle/make_notebooks.py
```

Each notebook states at the top what to switch on: the accelerator, Internet on, the `trilens-data`
dataset added as input, and a `GITHUB_TOKEN` secret so it can upload its results. Then Save & Run
All. If a session stops, run it again: finished steps are skipped.

| Notebook | What it runs | Accelerator | Needs first |
|---|---|---|---|
| `stage01_gene_presence_matrix` | Gene presence/absence matrix with the right orientation | CPU | nothing |
| `stage02_classical_models` | Classical stack, baselines, Giessen transfer and gene importance | GPU | nothing |
| `stage03_trilens_random` | TriLens stack on the random split, and its component ablation | GPU | stage01, stage02 |
| `stage04_trilens_other_splits` | TriLens neural branches on the phylogroup split and the lineage holdout | GPU | stage01, stage02 |
| `stage05_standard_attention_random` | Ablation model: standard gated attention, random split | GPU | stage01, stage02 |
| `stage06_standard_attention_other_splits` | Ablation model: standard gated attention, phylogroup split and lineage holdout | GPU | stage01, stage02 |
| `stage07_position_shuffle` | Position shuffle test | GPU | nothing |
| `stage08_analysis` | Analysis of the three splits | CPU | stage01, stage02, stage03, stage04, stage05, stage06, stage07 |
| `stage09_seed_stability` | Seed stability of the holdout margin | GPU | stage01, stage02, stage04 |
| `stage10_metadata_floor` | Metadata and assembly floor | CPU | nothing |
| `stage11_determinant_rescue` | Determinant calls on unseen lineages | CPU | stage01 |
| `stage12_assembly_size` | Assembly size within lineages | CPU | stage01 |
| `stage13_external_cohort_screen` | External cohort screen | CPU | nothing |
| `stage14_external_cohort` | Frozen models on the external cohort | CPU | stage13 |
| `stage15_coverage_and_rule` | Coverage check and model against rule | CPU | stage01, stage11 |
| `stage16_external_robustness` | External robustness and the assembly mechanism | CPU | stage13, stage14 |
| `stage17_species_screen` | Second species screen | CPU | nothing |
## Running it locally

Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Conda works too: `conda env create -f environment.yml`.

Steps 1 and 2 read the release VCF and need `cyvcf2`, which is in `requirements-data.txt`.

Then put the data in place as `data/README.md` describes, and run a step:

```bash
python -m pipeline.step07_build_strain_table
python -m pipeline.step21_measure_metadata_floor --splits random,mlst
```

Many analysis steps read only `results/`, which is in git, so they run with no data at all.

Two environment variables matter:

| Variable | What it does |
|---|---|
| `TRILENS_ROOT` | the project root, if it is not the current folder |
| `TRILENS_REPO_RESULTS` | a folder of committed results to read reference files from |

## Reproducing the numbers

Everything is driven by one seed (42). Splits, gene selection and model fitting are all fixed, so a
re-run on the same library versions gives the same numbers. The versions are pinned in
`requirements.txt`, and every result folder keeps the `config_used.yaml` of the run that wrote it.

```bash
pytest
ruff check .
```

The test suite is 208 tests and needs no data: it checks the split logic, the metrics, the matrix
rebuild, the determinant parsing and the Kaggle notebooks.

## The paper and the thesis

| Folder | What it is |
|---|---|
| `paper/` | the manuscript, written for NAR Genomics and Bioinformatics |
| `thesis/` | the undergraduate thesis, BRAC University, August 2026 |
| `docs/preregistration/` | the pre-registration documents for W6 to W14 |

Where the thesis and the paper disagree, the paper is the corrected version. The presence/absence
matrix was rebuilt, the lineage holdout was added and the pre-registered experiments came later.

## Data

The genomes and phenotypes are public, and no data file is redistributed here. The training data are
the Arcadia Science *E. coli* release, the external cohort and its determinant calls come from NCBI
Pathogen Detection, and the second collection is that of Ren et al. (2022). `data/README.md` lists
each source and every file the pipeline builds from it.

## Citing this work

`CITATION.cff` holds the machine-readable citation. The paper is the reference to cite once it is
published.

## Licence

MIT, for the code and the result tables. See `LICENSE`. The data belong to their original releases
and keep their own terms.
