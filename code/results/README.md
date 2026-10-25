# results/

One folder per run. Small text files (`.csv`, `.md`, `.json`, `.yaml`) are in git; the
`.npz` arrays and checkpoints are not (see `../docs/DATA.md`).

**These are the original runs.** The full re-run on Kaggle (`../../kaggle/`) rebuilds this
whole folder from the processed data. When it is done, rename this folder to
`results_original` and unzip `analysis_outputs.zip` as the new `results/`. In the re-run,
`standard/` holds the strain-conditioning ablation model (standard Lens B).

## The runs the thesis numbers come from

| Folder | What | Status |
|---|---|---|
| `trilens_attnfix/` | **delivered model**, random split, avg PR-AUC 0.8486 | current |
| `trilens_v10/` | same model before the attention fix (avg 0.8462), kept for the comparison | pre-fix on purpose |
| `pillar_cascadestack_classical_mlst/` | classical arm under the lineage holdout, 0.6299 | current |
| `pillar_cascadestack_classical_phylogeny_clean/` | classical arm under the phylogroup split | current |
| `pillar_trilens_mlst/` | delivered model under the lineage holdout | **re-run needed** |
| `pillar_trilens_t7_clean_gene_set/` | delivered model under the phylogroup split | **re-run needed** |
| `pillar_mlst_final_comparison/` | three-way comparison, full stack under the holdout (0.6337) | **re-run needed** |
| `pillar_t7_clean_final_comparison/` | phylogroup full stack (0.8531) | **re-run needed** |
| `diagnostics/` | outputs of `scripts/diagnostics/` | see below |

Folders marked "re-run needed" are empty until the re-run (`../docs/REPRODUCE.md`,
section 5). Their pre-fix versions are in `pre_fix/`, so a script can never read an old file
by mistake.

## pre_fix/

Outputs computed from the model **before** the strain-conditioning fix. The thesis reports
some of them on purpose (the "before" column of `tab:attnfix`). Do not use them for any
other table. AMP on the random split reads 0.9408 in these files and 0.9440 in the delivered
model.

## diagnostics/

| File | Script | Thesis table |
|---|---|---|
| `threshold_tuning.csv` | `threshold_tuning.py` | `tab:clinical`, `tab:threshold`; written after the re-run |
| `lineage_marker_test.csv` | `lineage_marker_test.py` | `tab:attnfix`, `tab:umark` |
| `prefix_baserate_lens_b.csv` | `prefix_baserate_lens_b.py` | `tab:attnfix` |
| `attention_diagnosis.csv`, `attention_amr_hits.csv` | `attention_diagnosis.py` | `tab:attnfix` |
| `matched_prevalence.csv` | `matched_prevalence_check.py` | `tab:previnv`; written after the re-run |
| `holdout_clinical_metrics.csv` | `holdout_clinical_metrics.py` | `tab:holdoutclin`; written after the re-run |
| `strain_ablation.csv` | `strain_ablation_table.py` | `tab:attnfix` (re-run only) |
| `paper_numbers.csv` | `collect_paper_numbers.py` | every headline number, paper against new |

`diagnostics/pre_fix/` holds the earlier versions of these files.
