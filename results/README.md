# results/

One folder per run. Every small text file (`.csv`, `.json`, `.md`, `.yaml`) is in git, so the
numbers in the paper can be checked without running anything. Model arrays and checkpoints
(`.npz`, `.pkl`, `.pt`) are not in git; `kaggle/` rebuilds them.

Each folder that a training run writes also holds `config_used.yaml`, the exact configuration of
that run.

## Main results

| Folder | What it holds |
|---|---|
| `trilens_stack_random/` | the full stack on the random split, with the component ablation, attention tables and meta weights |
| `trilens_stack_phylogroup/` | the full stack on the phylogroup-stratified split |
| `trilens_stack_lineage/` | the full stack on the lineage holdout, and the comparison of all three splits |
| `trilens_neural_lineage/`, `trilens_neural_phylogroup/` | the neural branches alone, without the classical learners |
| `classical_stack_random/`, `classical_stack_phylogroup/`, `classical_stack_lineage/` | the five classical learners, stacked, on each split |
| `component_ablation/` | what each added part of the stack contributes |
| `standard_attention/` | the same runs with standard attention instead of strain-conditioned attention |

## Baselines

| Folder | Model |
|---|---|
| `pca_baseline/` | XGBoost on 1,024 PCA components |
| `phylogroup_only_baseline/` | the phylogroup label and nothing else |
| `tabnet_random/`, `tabnet_phylogroup/`, `tabnet_lineage/` | TabNet on gene-level features |
| `giessen_transfer/` | the same method retrained on the Giessen collection |

## Pre-registered experiments

W6 to W14. `docs/preregistration/README.md` maps each folder to the document that fixed its pass
rule in advance.

| Folder | Experiment |
|---|---|
| `determinant_matrix/`, `determinant_matrix_all_databases/` | the strain by determinant matrix from AMRFinderPlus calls |
| `determinant_rescue/` | W6, determinant calls on unseen lineages |
| `external_cohort_screen/`, `external_cohort/` | W8, the external cohort and its screen |
| `seed_stability/` | W9, three seeds |
| `coverage_and_rule/` | W10, coverage and the fixed rule |
| `external_robustness/` | W11, leave one BioProject out |
| `assembly_size_mechanism/` | W12, batch or biology |
| `species_screen/`, `campylobacter/` | W13, the second species |
| `mic_harmonisation/` | W13, MIC values against the labels |
| `linear_model_and_clinical_errors/` | W14, a linear model and the clinical error rates |

## Checks and data tables

| Folder | What it holds |
|---|---|
| `data_inventory/` | the strain table, covariates, label counts and the orientation evidence for the presence/absence matrix |
| `gene_presence_rebuild/` | the report of the matrix rebuild, including how many loci were flipped |
| `lineage_holdout_split/` | the reference lineage holdout split, so it can be reproduced exactly |
| `splits/`, `fold_splits/` | the country, time and cross-validation fold splits |
| `near_duplicate_audit/` | what happens when near-duplicate test strains are removed |
| `prevalence_invariant_metrics/`, `full_metric_suite/` | metrics that do not move with prevalence, and the full metric table |
| `position_shuffle_test/` | the position shuffle control |
| `gene_importance_gain/`, `gene_importance_shap/`, `base_learner_shap/`, `marker_gene_annotation/` | which genes the models use, and whether they are known resistance genes |
| `diagnostics/` | clinical errors, thresholds, prevalence matching, split integrity, the metadata floor, attention diagnosis and the assembly-size shortcut |

## diagnostics/

| File | Written by |
|---|---|
| `holdout_clinical_errors.csv` | `step15 --part holdout` |
| `threshold_tuning.csv` | `step15 --part thresholds` |
| `matched_prevalence.csv` | `step14 --part matching` |
| `leakage_checks.csv` | `step16 --part integrity` |
| `metadata_floor.csv` | `step21` |
| `attention_diagnosis.csv`, `attention_amr_hits.csv` | `step19 --part attention` |
| `lineage_marker_test.csv` | `step19 --part lineage_markers` |
| `assembly_provenance_enrichment.csv` | `step19 --part provenance` |
| `strain_conditioning.csv` | `step19 --part strain_conditioning` |
| `assembly_size_correlations.csv`, `assembly_size_within_st.csv`, `assembly_size_within_lineage.json` | `step24 --part within_lineage` |
| `holdout_margin.csv`, `holdout_margin_deltas.csv` | `step20 --part margin` |
| `data_repair.json` | `step06` |
