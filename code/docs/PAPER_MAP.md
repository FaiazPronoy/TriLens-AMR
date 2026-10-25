# Paper map

Every table and figure in the thesis, against the script that produced it and the result file
the numbers were read from.

Tables are identified by their LaTeX label rather than their printed number, so this stays
correct if the thesis is recompiled and numbering shifts.

**After the full re-run** (`../../kaggle/`), `results/diagnostics/paper_numbers.csv` holds
the new value of every headline number next to the paper's. The status column below
describes the files from the original runs.

**Status column:**
`verified` the file was opened and the numbers matched during the final number audit.
`inferred` the mapping follows from the script and folder names but was not re-checked cell by cell.
`pending` the file on disk is from the pre-fix model; re-run first (see `KNOWN_ISSUES.md`, issue 1).

All paths are relative to `code/`.

---

## The runs everything else refers to

| Run folder | What it is | Script |
|---|---|---|
| `results/trilens_attnfix/` | **The delivered model.** Random split, strain-conditioned gated attention. Every headline number comes from here. | `src/experiments/run_trilens_v10.py` |
| `results/trilens_v10/` | The same model with standard gated attention (pre-fix), kept for the before/after comparison. Cannot be regenerated from current code. | (older version of the same script) |
| `results/pillar_trilens_mlst/` | The delivered model under the sequence-type lineage holdout | `src/experiments/run_trilens_mlst.py` |
| `results/pillar_trilens_t7_clean_gene_set/` | The delivered model under the phylogroup-stratified split | `src/experiments/run_trilens_t7_clean_gene_set.py` |

If you only download one thing, download `results/trilens_attnfix/`.

---

## Chapter 5: results tables

| Label | Caption (short) | Result file | Script | Status |
|---|---|---|---|---|
| `tab:main` | Per-antibiotic PR-AUC, MCC, F1, F2, accuracy, random split | `results/trilens_attnfix/ablation_results.csv` (PR-AUC, MCC); F1/F2/accuracy recomputed from `trilens_attnfix/predictions_saved.npz` | `run_trilens_v10.py`, `scripts/diagnostics/verify_thesis_numbers.py` | verified |
| `tab:clinical` | Clinical error rates at threshold 0.5 | `results/diagnostics/threshold_tuning.csv` (`VME_at_0p5` etc.) | `scripts/diagnostics/threshold_tuning.py` | pending (CPU re-run; the file on disk predates the delivered run) |
| `tab:threshold` | Thresholds that bring VME under the CLSI limit | `results/diagnostics/threshold_tuning.csv` | `scripts/diagnostics/threshold_tuning.py` | pending (CPU re-run) |
| `tab:ablation` | Three-stage layer ablation (lenses / +gate / +stacking) | `results/pillar2_layer_ablation/layer_ablation_results.csv` | `run_pillar2_stage1_expert_only.py`, `run_pillar2_layer_ablation.py` | pending: the file on disk mixes pre-fix stages 1-2 (0.7324, 0.7547) with the delivered stage 3; the paper values 0.7453 / 0.7807 are not in any surviving file |
| `tab:gate` | Normalised gate weight per lens, per antibiotic | `results/trilens_attnfix/gate_weights_test.npz` | `run_trilens_v10.py` | verified |
| `tab:ci` | Bootstrap 95% confidence intervals, 1,000 resamples | `results/trilens_attnfix/ablation_results.csv` (`ci_lo`, `ci_hi`) | `run_trilens_v10.py` | verified |
| `tab:baselines` | External baselines and internal components | `results/baselines/xgb_phylogroup_results.csv` (PCA 0.8033, variant `no_phylogroup`); components from `results/stage3_cascadestack_v9/v9_results.csv` (`solo_*`; XGBoost chi-squared reads 0.8430 there, the paper says 0.8400), `results/tabnet_baseline/tabnet_results.csv`, `results/pillar_phylogroup_only/phylogroup_only_results.csv`, `results/trilens_attnfix/ablation_results.csv`; holdout column from `results/pillar_cascadestack_classical_mlst/` and `results/pillar_tabnet_mlst/` | `run_baseline_xgb_phylogroup.py`, `run_cascadestack_v9.py`, `run_tabnet_baseline*.py`, `run_phylogroup_only_baseline.py`, `run_cascadestack_classical_mlst.py` | verified (random), pending (holdout full-model cell) |
| `tab:vsraw` | Proposed model against its strongest single component, both splits | `results/trilens_attnfix/ablation_results.csv`, `results/pillar_mlst_final_comparison/` | `run_trilens_v10.py`, `diagnose_mlst_final_comparison.py` | pending (holdout half) |
| `tab:progression` | Design progression across model versions | `results/stage2_gene_xgb/`, `results/gene_attention_v2/`, `results/stage3_deepmeta_v11/`, `results/stage3_cascadestack_v9/`, `results/trilens_attnfix/` | the matching `run_*.py` for each | verified |
| `tab:mlst_per_ab` | Per-antibiotic drop under the lineage holdout, three splits | `results/pillar_mlst_final_comparison/three_way_comparison_FINAL.csv` | `diagnose_t7_clean_final_comparison.py`, then `diagnose_mlst_final_comparison.py` | pending |
| `tab:previnv` | Prevalence-matched holdout: the 0.634 → 0.723 correction | `results/diagnostics/matched_prevalence.csv` | `scripts/diagnostics/matched_prevalence_check.py` | pending |
| `tab:holdoutclin` | Clinical error rates under the lineage holdout | `results/diagnostics/holdout_clinical_metrics.csv` | `scripts/diagnostics/holdout_clinical_metrics.py` | pending |
| `tab:mlst_components` | Neural arm vs classical arm vs full model, under holdout | `results/pillar_trilens_mlst/` (0.472), `results/pillar_cascadestack_classical_mlst/` (0.628), `results/pillar_mlst_final_comparison/` (0.6337) | `run_trilens_mlst.py`, `run_cascadestack_classical_mlst.py`, `diagnose_mlst_final_comparison.py` | pending (0.472, 0.6337); verified (0.628) |
| `tab:giessen_transfer` | Cross-dataset retention on the Giessen collection | `results/pillar4_giessen_transfer/pillar4_per_ab_metrics.csv`, `pillar4_generalization_gap.csv`; Arcadia column from `trilens_attnfix` | `run_pillar4_giessen_transfer.py` | verified |
| `tab:arcadia` | Comparison with the dataset authors' own analysis | no result file (literature comparison) | n/a | n/a |
| `tab:attnfix` | Standard vs strain-conditioned gated attention | `results/diagnostics/strain_ablation.csv`, built from `results/` (fixed) and `results/standard/` (the same pipeline with `TRILENS_LENSB=standard`) | `scripts/diagnostics/strain_ablation_table.py`; Kaggle notebooks `standard_random`, `standard_splits` | pending (re-run); the original standard-model files are `results/trilens_v10/` and `results/pre_fix/` |
| `tab:umark` | Lineage-marker test, `U_ST` and `U_res` | `results/pillar_gene_annotation/annotated_markers.csv`, `results/diagnostics/lineage_marker_test.csv` | `run_pillar_gene_annotation.py`, `scripts/diagnostics/lineage_marker_test.py` | inferred |
| `tab:phylogroup` | Phylogroup-only null baseline | `results/pillar_phylogroup_only/phylogroup_only_results.csv` | `run_phylogroup_only_baseline.py` | verified |

---

## Chapter 4: data and method tables

| Label | Caption (short) | Source | Script | Status |
|---|---|---|---|---|
| `tab:designchoices` | Summary of design decisions | narrative | n/a | n/a |
| `tab:labels` | Label coverage and prevalence per antibiotic | `data/processed/arcadia_v2_master_features.pkl` | `notebooks/master_features_and_splits.ipynb` | verified |
| `tab:variants` | Variant counts through the filtering stages | `data/processed/arcadia_v2_snp_matrix.pkl` | `notebooks/snp_matrix_from_vcf.ipynb`, `run_stage1_5.py` | verified |
| `tab:phylogroups` | Phylogroup composition of the 6,962 strains | `data/processed/arcadia_v2_master_features.pkl` | `notebooks/master_features_and_splits.ipynb` | verified |
| `tab:splits` | The three evaluation splits and their partition sizes | `data/processed/mlst_lineage_holdout_split.pkl` | `build_mlst_lineage_holdout_split.py` | verified |
| `tab:views` | The three genomic views and their shapes | `src/data/pergene_snp_blocks.py`, `gene_aggregated_features.py` | n/a | inferred |
| `tab:giessen` | Composition of the Giessen collection | `data/Giessen_dataset/*.csv` | `src/data/giessen_loader.py` | verified |
| `tab:hyperparameters` | Hyperparameters for all learners | `configs/experiment/*.yaml` | n/a | verified |

---

## Figures

Every figure, the script that draws it and its inputs are listed in
[`FIGURES.md`](FIGURES.md).

---

## The numbers most likely to be checked

| Number | Where it comes from | Status |
|---|---|---|
| **0.8486** avg PR-AUC, random split | `results/trilens_attnfix/ablation_results.csv`, `T6` + `random_split` | verified |
| **0.8033** PCA gradient-boosting baseline | `results/pillar1/pillar1_combined.csv` | verified |
| **0.6951** TabNet, random split | `results/tabnet_baseline/tabnet_results.csv` | verified |
| **0.2914** phylogroup-only null | `results/pillar_phylogroup_only/phylogroup_only_results.csv` | verified |
| **0.8400** strongest single component (XGB, chi²-filtered SNPs) | `results/trilens_attnfix/` base-learner columns | verified |
| **0.6337** full model, lineage holdout | `results/pillar_mlst_final_comparison/` | pending |
| **0.6299** strongest component, lineage holdout | `results/pillar_cascadestack_classical_mlst/` | verified |
| **0.7226** prevalence-matched holdout | `results/diagnostics/matched_prevalence.csv` | pending |
| **0.3255** TabNet, lineage holdout | `results/pillar_tabnet_mlst/tabnet_mlst_results.csv` | verified |
| **+0.035** gain from the strain-conditioning fix (holdout neural arm) | `results/pre_fix/pillar_trilens_mlst/` (0.437) vs the re-run (0.472) | pending |
| **90.0%** Giessen retention | `results/pillar4_giessen_transfer/pillar4_generalization_gap.csv` | verified |
| **47** near-duplicate test strains, cost 0.0041 | `results/pillar_dedup_eval/` after re-running `run_pillar_dedup_eval.py` | pending (file on disk is pre-fix) |
| **647/81/81** Giessen split | `results/pillar4_giessen_transfer/SUMMARY.md` | see `KNOWN_ISSUES.md`, issue 5 |
