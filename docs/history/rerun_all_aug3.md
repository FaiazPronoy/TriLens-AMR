# Rebuilding every number in the thesis from this folder

Written 2026-08-03, after the working copy on the second machine was lost. The
code here is complete: nothing has to be rewritten. What is missing is the
*output* of two model runs, and the diagnostics that read them.

Read the "Stale-file hazard" section before running anything. It is the reason
four wrong numbers reached the thesis.

---

## 0. What you need on disk

Already present in this folder, do not regenerate:

| File | Used by |
|---|---|
| `data/processed/arcadia_v2_master_features.pkl` | everything |
| `data/processed/arcadia_v2_gene_filtered.pkl` | random split |
| `data/processed/arcadia_v2_gene_filtered__mlst_holdout.pkl` | lineage holdout |
| `data/processed/arcadia_v2_gene_filtered__phylogeny_clean.pkl` | phylogroup split |
| `data/processed/arcadia_v2_gene_presence_absence.pkl` | Lens B, prefix test |
| `data/processed/arcadia_v2_snp_matrix.pkl` | figures |
| `data/processed/ported_cluster_labels_v2.pkl`, `mlst_lineage_holdout_split.pkl` | splits |
| `data/Giessen_dataset/*.csv` | external validation |

**Not** present, and **not** needed unless you want to rebuild the `.pkl` files
from scratch: the raw Zenodo downloads (`vcf_files/`, `presence_absence_data/`,
`strains_metadata.csv`, `SRA_to_genome_name.csv`).

Environment (Python 3.11+, 64-bit):

    pip install "numpy" pandas matplotlib scikit-learn pyyaml scipy xgboost lightgbm torch pytorch-lightning hydra-core omegaconf

`cyvcf2` and `pandas-plink` are only needed for raw VCF/PLINK parsing. Skip them.

---

## 1. Already correct — do NOT rerun

These outputs are in `results/` and every thesis table built from them has been
checked cell by cell against the file:

| Folder | Feeds |
|---|---|
| `results/trilens_attnfix/` | `tab:main`, `tab:ci`, `tab:gate`, `tab:ablation` Stage 3, `tab:baselines` random column |
| `results/pillar_cascadestack_classical_mlst/` | `tab:baselines` holdout column, `tab:mlst_components` 0.628 |
| `results/tabnet_baseline/`, `results/pillar_tabnet_mlst/` | `tab:baselines` TabNet rows |
| `results/pillar_phylogroup_only/` | `tab:phylogroup`, null baseline 0.2914 |
| `results/pillar4_giessen_transfer/` | `tab:giessen_transfer` Giessen column |
| `results/stage2_gene_xgb/`, `results/gene_attention_v2/`, `results/stage3_deepmeta_v11/` | `tab:progression` |
| `results/pillar3_shap/` | SHAP columns of `tab:umark` |

`results/trilens_attnfix/` is the one that matters most. It is the delivered
strain-conditioned model. If you ever delete it, everything below has to be
redone as well.

---

## 2. The two runs that are missing

Both need a GPU. Both write into folders that already contain **pre-fix**
output, which they will overwrite — that is intended.

### 2a. TriLens on the MLST lineage holdout

    python src/experiments/run_trilens_mlst.py

Writes `results/pillar_trilens_mlst/`. Takes a few hours.

Sanity check when it finishes: `trilens_mlst_results.csv` should average about
**0.472**, not 0.437. 0.437 means the strain-conditioning fix did not take
effect — stop and check `src/models/trilens.py` still has `attn_Vc` / `attn_Uc`.

Feeds: `tab:mlst_components` (the 0.472 row).

### 2b. TriLens on the phylogroup-stratified split

    python src/experiments/run_trilens_t7_clean_gene_set.py

Writes `results/pillar_trilens_t7_clean_gene_set/`.

Then:

    python src/experiments/diagnose_t7_clean_final_comparison.py

Writes `results/pillar_t7_clean_final_comparison/three_way_comparison_FINAL.csv`.

Feeds: the phylogroup column of `tab:mlst_per_ab` (average should be **0.8531**,
not 0.8519).

---

## 3. Diagnostics — run in this order

Each one reads the outputs above. All are fast (minutes).

    python src/experiments/diagnose_mlst_final_comparison.py

Writes `results/pillar_mlst_final_comparison/`. Check
`three_way_comparison_FINAL.csv`: the random column for AMP must read **0.9440**.
If it reads 0.9408 the script picked up `trilens_v10` instead of
`trilens_attnfix` — see the stale-file section.

Feeds: `tab:mlst_per_ab` (all three columns).

    python matched_prevalence_check.py

Writes `matched_prevalence_results.csv`. AMP `pr_auc_raw` must be **0.9043**.

Feeds: `tab:previnv` in full, and the abstract's 0.723 / 0.126 / 0.017–0.031 /
0.176–0.244.

    python holdout_clinical_metrics.py

Writes `holdout_clinical_metrics.csv`. `random_VME` for AMP must be **0.109**.

Feeds: `tab:holdoutclin`.

    python threshold_tuning.py

Writes `threshold_tuning_results.csv`. `VME_at_0p5` for AMP must be **0.109**.

Feeds: `tab:threshold`.

    python lineage_marker_test_attnfix.py
    python prefix_baserate_lensB_attnfix.py
    python attention_diagnosis_attnfix.py

These three read `results/trilens_attnfix/attention_B_topgenes.csv`, which is
already on disk, so they can be run **right now** without any GPU work.

Feed: the strain-conditioned rows of `tab:attnfix` — `U_res` 0.0419, `U_ST`
0.3569, enrichment 9.0x.

---

## 4. Figures

    cd amr-genomic-data
    python scripts/make_figures.py

Writes ten figures into `amr-genomic-data/results/figures/`. Copy
`fig_coresistance.pdf`, `fig_clinical_heatmap.pdf`, `fig_ci_forest.pdf` and
`fig_lineage_marker.pdf` into the thesis `images/` folder.

If `fig_coresistance` raises an AssertionError, the co-resistance structure of
the data has changed and Chapter 4's text needs revisiting — the message says
exactly what to check.

---

## 5. Verifying the thesis against all of it

    python verify_thesis_numbers.py

Recomputes the random-split tables from saved predictions and prints PASS/FAIL
per cell against the values typeset in `chapters/chapter_5_final.tex`.

---

## Stale-file hazard — read this

Every run writes `predictions_saved.npz` and most write a results CSV with a
fixed name. The pre-fix model and the strain-conditioned model therefore produce
**files with identical names**. A script that picks the wrong one produces a
table that looks completely normal and is wrong by 0.003 to 0.009 per cell.

This actually happened. Four numbers reached the thesis this way:

- `tab:giessen_transfer`'s Arcadia column carried the pre-fix values
  (0.9774 / 0.8215 / 0.8931, average 0.8489 instead of 0.8532)
- `tab:progression` said 0.8300 where `stage2_gene_xgb/SUMMARY.md` says 0.8297
- `tab:attnfix`'s mean VME said 0.258 where the recorded per-drug values give 0.257
- Section 5.5.2's 0.8512 / 0.8519 pair sat next to a 0.8531 elsewhere with no
  explanation of why they differed

Three defences are now in place:

1. `threshold_tuning.py`, `holdout_clinical_metrics.py`,
   `matched_prevalence_check.py` and `diagnose_mlst_final_comparison.py` all
   prefer the strain-conditioned run and fall back to the pre-fix one only if it
   is absent. Each prints which file it used.
2. `threshold_tuning.py` and `holdout_clinical_metrics.py` write a `source_run`
   column into their CSV, so a file that has left this machine still says where
   it came from.
3. `holdout_clinical_metrics.py` no longer hardcodes the random-split comparison
   columns. It recomputes them from the same predictions the rest of Chapter 5
   uses, so the two halves of that table cannot drift apart.

The check that catches all of it in one line: **AMP on the random split is
0.9440 for the delivered model and 0.9408 for the pre-fix one.** If any file
anywhere says 0.9408, it describes the old model.
