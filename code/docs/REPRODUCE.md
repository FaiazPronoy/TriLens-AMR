# Reproducing the results

Run order matters: later stages read files written by earlier ones. Run every command from
inside `code/`.

**The full re-run is on Kaggle:** seven notebooks in `../../kaggle/notebooks/` run every
step below, in this order, from an empty `results/` folder. See `../../kaggle/README.md`.
This file lists the same steps as commands, for a local machine.

Three environment variables:

| Variable | Default | Meaning |
|---|---|---|
| `TRILENS_ROOT` | `.` | the `code/` folder |
| `TRILENS_MAIN_RUN` | `trilens_attnfix_rerun` for the training run, `trilens_attnfix` for readers | folder of the main TriLens run. Set it to `trilens_attnfix` when `results/` starts empty |
| `TRILENS_LENSB` | `fixed` | `standard` trains the ablation model without strain conditioning |

---

## 0. Environment

```bash
conda env create -f environment.yml
conda activate amr
python scripts/diagnostics/verify_env.py      # imports, GPU, data files, loaders
```

Hydra experiments:

```bash
python -m src.run_experiment experiment=<name>
```

`<name>` is a file in `configs/experiment/` without the `.yaml`. Standalone scripts:

```bash
python -m src.experiments.<script_name>
```

**Seed is fixed at 42 everywhere.** Point estimates regenerate exactly. Bootstrap interval
bounds move by a few thousandths between runs (Monte Carlo resampling).

**Hardware.** One consumer GPU or a Kaggle T4. The longest single step is the PCA baseline,
about 50 minutes on CPU.

---

## 1. Build the datasets

Skip this if `data/processed/` is already filled (see `DATA.md`).

The base files were built on Kaggle by the notebooks in `notebooks/`:

| Notebook | Writes |
|---|---|
| `snp_matrix_from_vcf.ipynb` | `arcadia_v2_snp_matrix.pkl` (6,962 × 535,439) |
| `master_features_and_splits.ipynb` | `arcadia_v2_master_features.pkl` (labels, mask, phylogroups, PCA-1024, random and phylogroup splits) |
| `tabular_baselines_and_gene_pa.ipynb` | `arcadia_v2_gene_presence_absence.pkl` (6,962 × 32,441), plus the tabular neural baselines |

Then the split-specific gene filters:

```bash
python -m src.run_experiment experiment=stage1_5_gene_filtering   # random split -> arcadia_v2_gene_filtered.pkl
python -m src.experiments.build_mlst_lineage_holdout_split          # the sequence-type partition
python -m src.experiments.build_gene_filtered_dataset_mlst          # refit the filter on that split
python -m src.experiments.build_gene_filtered_dataset_phylogeny     # refit on the phylogroup split
```

The gene filter is refitted inside each split's own training partition on purpose. One
shared filter would let test genomes choose the features.

---

## 2. The proposed model

The classical half of the stack comes first; the TriLens run reads its out-of-fold
predictions.

```bash
python -m src.run_experiment experiment=stage3_cascadestack_v9      # classical arm, random split
python -m src.run_experiment experiment=trilens_v10                 # the delivered model
```

Writes `results/$TRILENS_MAIN_RUN/` (default `trilens_attnfix_rerun`, so the delivered run
in `results/trilens_attnfix/`, average PR-AUC **0.8486**, is never overwritten).

The strain-conditioning ablation (`tab:attnfix`) is the same commands with
`TRILENS_LENSB=standard`, in a separate results folder. See `KNOWN_ISSUES.md`, issue 3.

---

## 3. Baselines

```bash
python -m src.run_experiment experiment=baseline_xgb_phylogroup   # XGBoost on PCA-1024 -> 0.8033 (variant no_phylogroup)
python -m src.experiments.run_phylogroup_only_baseline            # -> 0.2914
python -m src.experiments.run_tabnet_baseline                     # -> 0.6951
python -m src.experiments.run_tabnet_baseline_phylogeny
python -m src.experiments.run_tabnet_baseline_mlst                # -> 0.3255
```

The locality check behind the Motif Pattern Encoder (a 1D CNN on the whole pan-genome,
ordered against shuffled variants):

```bash
python -m src.run_experiment experiment=pillar1_exp3
python -m src.experiments.aggregate_pillar1_results --paper-outline /tmp/outline.md
```

`pillar1_exp1` and `pillar1_exp2` (CNN on Giessen against Arcadia, contig-aware CNN) are not
reported in the paper and take 10+ GPU hours; the Kaggle re-run skips them.

---

## 4. Ablations

```bash
python -m src.experiments.run_pillar2_stage1_expert_only   # stage 1, GPU, reuses the main run's caches
python -m src.experiments.run_pillar2_layer_ablation       # the table; reads the main run
```

Three-stage table: lenses 0.7453 → +gate 0.7807 → +stacking 0.8486.

---

## 5. The lineage holdout and the phylogroup split (core experiment)

These are the runs that must be repeated before release (`KNOWN_ISSUES.md`, issue 1).
Both need a GPU. The classical arms come first:

```bash
python -m src.experiments.run_cascadestack_classical_phylogeny_clean
python -m src.experiments.run_cascadestack_classical_mlst
```

```bash
# phylogroup split first: the holdout comparison reads its output
python -m src.experiments.run_trilens_t7_clean_gene_set        # GPU, writes pillar_trilens_t7_clean_gene_set/
python -m src.experiments.diagnose_t7_clean_final_comparison   # average must be 0.8531

# lineage holdout
python -m src.experiments.run_trilens_mlst                     # GPU, a few hours; average about 0.472 (0.437 = fix missing)
python -m src.experiments.diagnose_mlst_final_comparison       # full stack 0.6337; AMP random column must be 0.9440
```

The classical arm does not depend on the Lens B fix; the standard-model ablation reuses it.

Then the diagnostics that read the holdout predictions:

```bash
python scripts/diagnostics/matched_prevalence_check.py   # 0.634 -> 0.723; AMP pr_auc_raw must be 0.9043
python scripts/diagnostics/holdout_clinical_metrics.py   # random_VME for AMP must be 0.109
python scripts/diagnostics/threshold_tuning.py           # VME_at_0p5 for AMP must be 0.109
python -m src.experiments.run_pillar_dedup_eval_mlst
python -m src.experiments.run_pillar_dedup_eval          # random split; reads trilens_attnfix now
python -m src.experiments.run_pillar_full_metric_suite   # reads trilens_attnfix now
```

The correction resamples the holdout test set to the training prevalence. Nothing is
retrained and no resistant strain is discarded.

---

## 6. Interpretability

```bash
python -m src.run_experiment experiment=pillar3_shap
python -m src.run_experiment experiment=pillar3_xgb_importance
python -m src.run_experiment experiment=pillar3_interpretability   # thesis only: needs the stage-2 GeneAttentionNet checkpoint, not in the Kaggle re-run
python -m src.experiments.run_pillar_shap_t6
python -m src.experiments.run_pillar_gene_annotation

# strain-conditioned attention checks (read results/trilens_attnfix/, CPU only)
python scripts/diagnostics/lineage_marker_test.py        # U_res 0.0419, U_ST 0.3569
python scripts/diagnostics/prefix_baserate_lens_b.py
python scripts/diagnostics/attention_diagnosis.py        # enrichment 9.0x
python scripts/diagnostics/prove_lens_b_invariance.py    # structural check, no data needed
```

---

## 7. External validation

```bash
python -m src.run_experiment experiment=pillar4_giessen_transfer
```

This **retrains** the classical stack on Giessen. It is not zero-shot transfer: the two
collections share no feature vector (Arcadia is a pan-genome with 535K variant positions,
Giessen a single reference with about 60K). See `KNOWN_ISSUES.md`, issue 5, for the split.

---

## 8. Everything else

Thesis-only analyses, not reported in the paper and not part of the Kaggle re-run:

```bash
python -m src.run_experiment experiment=posthoc_analysis   # calibration, per-phylogroup, significance
python -m src.run_experiment experiment=phylogeny_split_eval
python -m src.experiments.run_pillar_master_summary        # collects everything into one CSV
```

---

## 9. Figures and the final check

```bash
python scripts/figures/make_all_figures.py        # see FIGURES.md
python scripts/diagnostics/strain_ablation_table.py   # tab:attnfix, needs results/standard/
python scripts/diagnostics/collect_paper_numbers.py   # every paper number, paper vs new
python scripts/diagnostics/verify_thesis_numbers.py
```

`verify_thesis_numbers.py` recomputes the random-split tables from the saved predictions and
prints PASS/FAIL per cell against the values transcribed from `thesis/chapters/chapter_5.tex`.

---

## What you should get

| | Expected |
|---|---|
| Random split, average PR-AUC | 0.8486 |
| Phylogroup split, full stack | 0.8531 |
| Lineage holdout, full stack | 0.6337 |
| Lineage holdout, neural arm alone | 0.472 |
| Prevalence-matched holdout | 0.7226 |
| PCA gradient-boosting baseline | 0.8033 |
| TabNet, random split | 0.6951 |

If a point estimate differs, check first that the gene filter was refitted on the right
split, and that no script picked up a `trilens_v10` (pre-fix) file.
