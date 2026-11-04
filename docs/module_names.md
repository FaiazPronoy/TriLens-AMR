# Old names and new names

The pre-registration documents in `docs/preregistration/` and the result files written before the
repository was reorganised use the earlier module and folder names. They are kept word for word,
so this table maps each one to its current name. Nothing here needs to be used in new code.

## Modules

| Old | Now |
|---|---|
| `scripts.diagnostics.assembly_shortcut` | `trilens.analysis.assembly_size_within_lineage` |
| `scripts.diagnostics.attention_diagnosis` | `trilens.analysis.attention_diagnosis` |
| `scripts.diagnostics.campylobacter` | `trilens.analysis.campylobacter` |
| `scripts.diagnostics.check_leakage` | `trilens.analysis.split_integrity_check` |
| `scripts.diagnostics.clinical_and_linear` | `trilens.analysis.linear_model_and_clinical_errors` |
| `scripts.diagnostics.confounder_floor` | `trilens.analysis.metadata_floor` |
| `scripts.diagnostics.coverage_and_rule` | `trilens.analysis.coverage_and_rule` |
| `scripts.diagnostics.external_robustness` | `trilens.analysis.external_robustness` |
| `scripts.diagnostics.holdout_clinical_metrics` | `trilens.analysis.holdout_clinical_errors` |
| `scripts.diagnostics.holdout_validity` | `trilens.analysis.holdout_margin` |
| `scripts.diagnostics.lineage_marker_test` | `trilens.analysis.lineage_marker_test` |
| `scripts.diagnostics.matched_prevalence_check` | `trilens.analysis.prevalence_matching` |
| `scripts.diagnostics.mic_check` | `trilens.analysis.mic_harmonisation` |
| `scripts.diagnostics.prefix_baserate_lens_b` | `trilens.analysis.assembly_provenance_enrichment` |
| `scripts.diagnostics.shortcut_mechanism` | `trilens.analysis.assembly_size_mechanism` |
| `scripts.diagnostics.strain_ablation_table` | `trilens.analysis.strain_conditioning_table` |
| `scripts.diagnostics.threshold_tuning` | `trilens.analysis.threshold_tuning` |
| `scripts.fetch_gene_pa_source` | `trilens.data.gene_presence_source` |
| `scripts.github_sync` | `trilens.utils.github_sync` |
| `scripts.inventory.data_inventory` | `trilens.analysis.data_inventory` |
| `scripts.repair_gene_filtered` | `trilens.data.snp_block_repair` |
| `scripts.repair_gene_pa` | `trilens.data.gene_presence_rebuild` |
| `src.data.arcadia_snp_loader` | `trilens.data.arcadia_snp_loader` |
| `src.data.gene_aggregated_features` | `trilens.data.gene_statistics` |
| `src.data.gene_filtering` | `trilens.data.gene_selection_rules` |
| `src.data.giessen_loader` | `trilens.data.giessen_loader` |
| `src.data.metadata` | `trilens.data.strain_metadata` |
| `src.data.pergene_snp_blocks` | `trilens.data.gene_snp_blocks` |
| `src.data.transforms` | `trilens.data.snp_transforms` |
| `src.evaluation.grouped` | `trilens.evaluation.cluster_bootstrap` |
| `src.experiments._shared` | `trilens.evaluation.training_helpers` |
| `src.experiments.aggregate_pillar1_results` | `trilens.experiments.position_shuffle_summary` |
| `src.experiments.build_gene_filtered_dataset` | `trilens.experiments.gene_filtered_dataset` |
| `src.experiments.build_gene_filtered_dataset_mlst` | `trilens.experiments.gene_filtered_dataset_lineage` |
| `src.experiments.build_gene_filtered_dataset_phylogeny` | `trilens.experiments.gene_filtered_dataset_phylogroup` |
| `src.experiments.build_mlst_lineage_holdout_split` | `trilens.experiments.lineage_holdout_split` |
| `src.experiments.diagnose_mlst_final_comparison` | `trilens.experiments.stack_lineage` |
| `src.experiments.diagnose_prevalence_invariant_metrics` | `trilens.experiments.prevalence_invariant_metrics` |
| `src.experiments.diagnose_t7_clean_final_comparison` | `trilens.experiments.stack_phylogroup` |
| `src.experiments.run_arg_rescue` | `trilens.experiments.determinant_rescue` |
| `src.experiments.run_baseline_xgb_phylogroup` | `trilens.experiments.pca_baseline` |
| `src.experiments.run_cascadestack_classical_mlst` | `trilens.experiments.classical_stack_lineage` |
| `src.experiments.run_cascadestack_classical_phylogeny_clean` | `trilens.experiments.classical_stack_phylogroup` |
| `src.experiments.run_cascadestack_v9` | `trilens.experiments.classical_stack` |
| `src.experiments.run_external` | `trilens.experiments.external_cohort` |
| `src.experiments.run_lineage_seeds` | `trilens.experiments.seed_stability` |
| `src.experiments.run_phylogroup_only_baseline` | `trilens.experiments.phylogroup_only_baseline` |
| `src.experiments.run_pillar1_exp3` | `trilens.experiments.position_shuffle_test` |
| `src.experiments.run_pillar2_layer_ablation` | `trilens.experiments.component_ablation` |
| `src.experiments.run_pillar2_stage1_expert_only` | `trilens.experiments.branch_average_ablation` |
| `src.experiments.run_pillar3_shap` | `trilens.experiments.gene_importance_shap` |
| `src.experiments.run_pillar3_xgb_importance` | `trilens.experiments.gene_importance_gain` |
| `src.experiments.run_pillar4_giessen_transfer` | `trilens.experiments.giessen_transfer` |
| `src.experiments.run_pillar_dedup_eval` | `trilens.experiments.near_duplicate_audit` |
| `src.experiments.run_pillar_dedup_eval_mlst` | `trilens.experiments.near_duplicate_audit_lineage` |
| `src.experiments.run_pillar_full_metric_suite` | `trilens.experiments.full_metric_suite` |
| `src.experiments.run_pillar_gene_annotation` | `trilens.experiments.marker_gene_annotation` |
| `src.experiments.run_pillar_shap_t6` | `trilens.experiments.base_learner_shap` |
| `src.experiments.run_stage1_5` | `trilens.experiments.gene_selection` |
| `src.experiments.run_tabnet_baseline` | `trilens.experiments.tabnet_baseline` |
| `src.experiments.run_tabnet_baseline_mlst` | `trilens.experiments.tabnet_baseline_lineage` |
| `src.experiments.run_tabnet_baseline_phylogeny` | `trilens.experiments.tabnet_baseline_phylogroup` |
| `src.experiments.run_trilens_mlst` | `trilens.experiments.trilens_lineage` |
| `src.experiments.run_trilens_t7_clean_gene_set` | `trilens.experiments.trilens_phylogroup` |
| `src.experiments.run_trilens_v10` | `trilens.experiments.trilens_stack` |
| `src.experiments.run_xgb_gene_importance` | `trilens.experiments.gene_importance_ranking` |
| `src.experiments.run_xgb_gene_importance_mlst` | `trilens.experiments.gene_importance_ranking_lineage` |
| `src.experiments.run_xgb_gene_importance_phylogeny` | `trilens.experiments.gene_importance_ranking_phylogroup` |
| `src.mechanism.amrfinder` | `trilens.determinants.amrfinder_runner` |
| `src.mechanism.arg_fetch` | `trilens.determinants.genome_download_calls` |
| `src.mechanism.arg_pd` | `trilens.determinants.determinant_matrix` |
| `src.mechanism.ext_probe` | `trilens.determinants.external_cohort_screen` |
| `src.mechanism.species_probe` | `trilens.determinants.species_screen` |
| `src.models.cnn_variants` | `trilens.models.plain_cnn` |
| `src.models.gene_attention_net` | `trilens.models.gene_attention_network` |
| `src.models.trilens` | `trilens.models.trilens_network` |
| `src.splits.country_time` | `trilens.splits.country_and_time_splits` |
| `src.splits.from_json` | `trilens.splits.split_files` |
| `src.splits.kfold` | `trilens.splits.fold_splits` |
| `src.utils.seeds` | `trilens.utils.seeds` |

## Result folders

| Old | Now |
|---|---|
| `results/arg_calls/` | `results/determinant_calls/` |
| `results/arg_matrix/` | `results/determinant_matrix/` |
| `results/arg_matrix_alldb/` | `results/determinant_matrix_all_databases/` |
| `results/baselines/` | `results/pca_baseline/` |
| `results/ext_probe/` | `results/external_cohort_screen/` |
| `results/lineage_seeds/` | `results/seed_stability/` |
| `results/pa_fix/` | `results/gene_presence_rebuild/` |
| `results/pillar1/` | `results/position_shuffle_test/` |
| `results/pillar2_layer_ablation/` | `results/component_ablation/` |
| `results/pillar3_shap/` | `results/gene_importance_shap/` |
| `results/pillar3_xgb_importance/` | `results/gene_importance_gain/` |
| `results/pillar4_giessen_transfer/` | `results/giessen_transfer/` |
| `results/pillar_cascadestack_classical_mlst/` | `results/classical_stack_lineage/` |
| `results/pillar_cascadestack_classical_phylogeny_clean/` | `results/classical_stack_phylogroup/` |
| `results/pillar_dedup_eval/` | `results/near_duplicate_audit/` |
| `results/pillar_gene_annotation/` | `results/marker_gene_annotation/` |
| `results/pillar_metrics_full/` | `results/full_metric_suite/` |
| `results/pillar_mlst_final_comparison/` | `results/trilens_stack_lineage/` |
| `results/pillar_phylogroup_only/` | `results/phylogroup_only_baseline/` |
| `results/pillar_prevalence_invariant_metrics/` | `results/prevalence_invariant_metrics/` |
| `results/pillar_shap_t6/` | `results/base_learner_shap/` |
| `results/pillar_t7_clean_final_comparison/` | `results/trilens_stack_phylogroup/` |
| `results/pillar_tabnet_mlst/` | `results/tabnet_lineage/` |
| `results/pillar_tabnet_phylogeny/` | `results/tabnet_phylogroup/` |
| `results/pillar_trilens_mlst/` | `results/trilens_neural_lineage/` |
| `results/pillar_trilens_t7_clean_gene_set/` | `results/trilens_neural_phylogroup/` |
| `results/species_probe/` | `results/species_screen/` |
| `results/splits_kfold/` | `results/fold_splits/` |
| `results/stage1_5/` | `results/gene_selection/` |
| `results/stage3_cascadestack_v9/` | `results/classical_stack_random/` |
| `results/tabnet_baseline/` | `results/tabnet_random/` |
| `results/trilens_attnfix/` | `results/trilens_stack_random/` |
| `results/trilens_attnfix_rerun/` | `results/trilens_stack_random/` |
| `results/trilens_v10/` | `results/trilens_stack_random/` |
| `results/w10_coverage_rule/` | `results/coverage_and_rule/` |
| `results/w11_external_robustness/` | `results/external_robustness/` |
| `results/w12_shortcut_mechanism/` | `results/assembly_size_mechanism/` |
| `results/w13_campylobacter/` | `results/campylobacter/` |
| `results/w13_mic/` | `results/mic_harmonisation/` |
| `results/w14_clinical_linear/` | `results/linear_model_and_clinical_errors/` |
| `results/w6_rescue/` | `results/determinant_rescue/` |
| `results/w8_external/` | `results/external_cohort/` |

## Config names

| Old | Now |
|---|---|
| `baseline_xgb_phylogroup` | `pca_baseline` |
| `cnn_1d_contig_aware` | `contig_aware_cnn` |
| `cnn_1d_naive` | `plain_cnn` |
| `pillar1_exp3` | `position_shuffle_test` |
| `stage1_5_gene_filtering` | `gene_selection` |

## Other names that changed

| Old | Now |
|---|---|
| `T1` to `T7` (ablation ids) | `motif_branch_only`, `presence_branch_only`, `cross_antibiotic_branch_only`, `branches_averaged`, `branches_gated`, `full_stack`, `full_stack_phylogroup` |
| `planB_` / `planC_` prefixes | `gene_level_` / `snp_level_` (gain) and `shap_gene_level_` / `shap_snp_level_` (SHAP) |
| `Pillar 1` experiment 3 | the position shuffle test |
| `Pillar 2` | the three branches and the gate |
| `Pillar 3` | gene importance, gain and SHAP |
| `Pillar 4` | the Giessen transfer |
| `CascadeStack v9` / `v9.1` | the classical stack |
| `TriLens v10` / `MoE-Stack v10` | the TriLens stack |
| `attnfix` | strain-conditioned attention (the delivered model) |
| `standard` results folder | `standard_attention/` |
| `EXP3_ONLY` | `SHUFFLE_ONLY` |
| `TRILENS_LENSB=fixed|standard` | `TRILENS_ATTENTION=conditioned|standard` |
| `W4` AMRFinderPlus output | `results/amrfinder_calls/` |
